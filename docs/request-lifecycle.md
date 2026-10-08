# Request lifecycle in a paged, continuously batched server

How a request travels through an engine like vLLM or SGLang, in the order it happens:
what startup allocates, how a cold request is scheduled and computed, how later
requests share its work, how the system stays correct under concurrency, and how
requests leave.

Written as a reference for the next steps after `gpt2/serving`: paging, chunked
prefill, prefix caching. Part V maps every concept back to the code in this repo.

---

## At a glance

```
STARTUP (once)
  profile memory → size the KV pool → free list → block table (host + device)
  → persistent input buffers → capture CUDA graphs

PER REQUEST
  arrive → validate → tokenize → WAITING queue

PER ITERATION (the engine-core loop, one thread)
  1 schedule   spend the token budget across running + waiting
  2 allocate   prefix-cache lookup, block delta, free-list pop, fork / CoW decisions
  3 pack       build the five arrays
  4 sync       patch dirty block-table rows, copy arrays into persistent buffers
  5 forward    per layer: project → write KV → attend
  6 sample     logits at sampled rows only → token ids → host
  7 update     append tokens, check stop, stream, retire / abort / preempt
```

| Part | Covers |
|---|---|
| I — Startup | bytes per token, pool sizing, free list, block table, address translation, buffers |
| II — The first request, cold | arrive → schedule → find blocks → pack → sync → forward → sample → decode |
| III — Later requests | prefix cache, forks and copy-on-write, refcounts, the general dirty rule |
| IV — Correctness and completeness | the concurrency model; finish, abort, evict, preempt |
| V — Reference | kernel internals, scaling, scope, vLLM/SGLang, observability, repo mapping |

---

## Vocabulary

| Term | Meaning |
|---|---|
| **request** | one user-submitted generation job |
| **sequence** | one KV history. Usually 1:1 with a request; `n > 1` sampling gives several per request |
| **logical block** | a sequence's own block index, `pos // block_size`, always contiguous from 0 |
| **physical block** | an index into the shared KV pool. Arbitrary, scattered |
| **offset** | `pos % block_size`, position *within* a block. Never stored anywhere |
| **slot** | a flat pool index, `physical * block_size + offset` |
| **block table** | logical → physical map. A page table |
| **free list** | physical block ids not owned by any sequence. Flat, no sequence axis |
| **refcount** | how many sequences point at a physical block |
| **cached block** | refcount 0 but still registered in the prefix hash table; evictable |
| **token budget** | `max_num_batched_tokens`, how much work one forward may contain |
| **`q_len`** | tokens a sequence contributes *this step* |
| **`kv_len`** | tokens a sequence may *attend to* (history + new) |
| **`num_computed_tokens`** | how many of a request's tokens already have K/V in the pool |
| **activations** | temporary tensors of one forward (hidden states, q/k/v, MLP, logits); headroom, not pool |
| **preemption** | taking blocks back from a *running* request so another can grow |
| **thrashing** | repeated preempt → readmit → preempt, redoing work instead of progressing |

---

## Two examples

**The running example** threads through Parts I–IV. It is small enough to trace every
slot:

```
block_size = 4,  num_blocks = 10,  max_num_seqs = 4,  max_model_len = 16
system prompt S = s0 s1 s2 s3 s4 s5          (6 tokens)

R1 = S + a0              arrives at iteration 1      — cold
R2 = S + b0 b1 b2        arrives at iteration 4      — shares S with R1
R3 = S + c0,   n = 2     arrives at iteration 5      — shares S, then forks
```

**At scale** is used only where magnitudes matter — budgets, chunking, kernel
occupancy. `block_size = 16`, budget `max_num_batched_tokens = 512`:

```
A  decoding,  133 tokens of history
B  new,       1200-token prompt, nothing computed yet
C  decoding,  6 tokens of history
```

Byte sizes throughout use this repo's 124M config: 12 layers, 12 heads, head_dim 64,
`n_embd` 768, fp16.

---

# Part I — Startup

Everything expensive is allocated once, at load, and never moved. After startup the
engine never calls `cudaMalloc`; "allocating" KV means handing out an integer.

## 1. What the pool stores, and what it costs

### Bytes per token

The number every other figure divides by:

```
bytes/token = 2 (K,V) × n_layer × n_kv_head × head_dim × dtype_bytes
            = 2 × 12 × 12 × 64 × 2
            = 36,864  =  36 KiB
```

Equivalently `2 × n_layer × n_embd × dtype_bytes`, since `n_kv_head × head_dim ==
n_embd` under MHA.

Note **`n_kv_head`, not query heads**. That asymmetry is why GQA exists — query heads
stay high for model quality while KV heads drop for memory. Three of the five factors
are architectural; two are the real levers:

| lever | effect on 36 KiB |
|---|---|
| fp16 → fp8 KV | 18 KiB |
| GQA, 12 → 2 KV heads | 6 KiB |
| both | 3 KiB — **12× deeper** context for the same pool |

For scale, Llama-3-70B (80 layers, 8 KV heads, head_dim 128, fp16) is 320 KiB/token —
only ~9× this model despite 560× the parameters, because of GQA. Without it, 2.6
MB/token.

KV has **no dependence on sequence length or batch size** — it is strictly linear in
total live tokens. That linearity, not parameter count, is what makes KV the serving
constraint.

### What is *not* cached

Nothing in the embedding path enters the pool, for two different reasons.

**`wte` and `wpe` are weights, not state.** 50257×768 and 1024×768 — loaded once,
shared by every request forever, in the weights allocation. They do not scale with
concurrency.

**The embedding activation dies inside the forward.** `x = wte + wpe` produces a
`(768,)` hidden state that layer 0 consumes and nobody ever wants again:

```
token id → wte + wpe → x ──┐
                           │  layer 0: x → q,k,v
                           │      k,v → POOL   (persisted)
                           │      q   → used once, dropped
                           │      attn → MLP → x'
                           │  layer 1: x' → q,k,v …
                           └─ x, x', q, MLP activations: all transient
final x → lm_head → logits → sample
```

The cache rule is strict: **store exactly what future tokens must read.** Future
tokens read past tokens' K and V. They never read a past token's embedding, hidden
state, Q, or MLP activation. K and V *are* the embedding, in the only form downstream
tokens consume it.

**Q is never cached.** All three projections are computed, only K and V stored:

```
token 5's K and V  →  read by tokens 6, 7, 8, … forever
token 5's Q        →  used once, for token 5's own output, then never again
```

Caching Q would cost 50% more memory for zero reuse. Each layer keeps its own K/V, so
one token produces 12 K entries and 12 V entries — which is why a paged block spans
all layers.

**Consequence: K/V are position-bound.** With learned absolute `wpe`, position is baked
into `x` *before* layer 0, so by the time K and V are projected they already contain
positional information. A cached block cannot be relocated to a different offset. Same
conclusion with RoPE, just applied inside attention. This is why prefix caching works
only on a *prefix* (§16).

## 2. Sizing the pool

### Everything that lives on the GPU at runtime

| category | what is in it | lifetime | sized by |
|---|---|---|---|
| **weights** | `wte`, `wpe`, 12 blocks, `lm_head` | process | the model — ~250 MB fp16 at 124M |
| **KV pool** | K/V for every live token | per block, scheduler-managed | **the remainder** |
| **activations** | x, q/k/v, attention output, MLP hidden, logits | one forward, reused layer by layer | tokens per forward (the budget) |
| **workspaces** | cuBLAS scratch, split-K partials `(m, l, o)`, sampling buffers | one kernel / one step | kernels, batch size |
| **fixed bookkeeping** | block table, persistent input buffers (§6), CUDA-graph memory pool | process | `max_num_seqs`, budget |
| **outside PyTorch** | CUDA context, allocator fragmentation | process | driver / runtime |

The KV pool is the only row that grows with the number of live tokens. Everything else
is either fixed or bounded by the size of one forward. So the pool is the **remainder,
not a setting**:

```python
# profile a dummy forward at the maximum budget to measure peak activation + workspace, then:
num_blocks = (gpu_memory_utilization × total − weights − peak_activations) / bytes_per_block
```

`gpu_memory_utilization` is below 1.0 (typically ~0.9) because the last row cannot be
measured precisely: the CUDA context takes a few hundred MB that PyTorch does not
track, and fragmentation wastes more. The fraction is a safety margin for memory you
cannot measure.

At scale, `block_size = 16`:

```
one block  = 16 tokens × 36 KiB = 576 KiB
10 GiB     = 18,204 blocks = 291,264 token slots
```

### Why activations need headroom

Activations are not allocated up front the way the pool is. They are **headroom left
free**, because the forward pass needs scratch memory and the pool must not take it.

A forward creates temporary tensors that exist only while it runs. Layer 0's `x` is
gone once layer 1 has used it — but at some instant during the forward they all need
real memory:

```
x            (N, 768)          hidden state, layer to layer
q, k, v      (N, 2304)         before k,v are scattered into the pool
attn out     (N, 768)
MLP hidden   (N, 3072)         the 4× expansion — the largest per-layer tensor
logits       (rows, 50257)     at the end
```

If the activation term were left out of the formula, the pool would fill the GPU, and
the first large forward would hit out-of-memory halfway through a layer — *after* the
scheduler had committed to the batch, breaking the guarantee in §10 that the GPU is only
handed satisfiable work. The scheduler controls KV memory block by block; it controls
activation memory only indirectly, through the token budget. So that memory is set
aside in advance.

It is **profiled rather than computed** because:

- **Only the peak matters, not the sum.** Inference has no backward pass, so nothing is
  kept for gradients. Buffers are freed and reused layer by layer: you need one layer's
  high-water mark plus the logits, not 12 layers' worth.
- **It scales with tokens per forward, not with history.** The budget sets `N`.
  Context length does not change it — history lives in the KV pool.
- **The exact figure is hard to predict.** Allocator caching, kernel workspaces and
  CUDA-graph pools all add to it, so one dummy forward at the maximum budget measures
  the real high-water mark.

How big it is, 124M config, `N = 512`, fp16:

| tensor | size |
|---|---|
| MLP hidden, 512 × 3072 | 3 MiB |
| q, k, v, 512 × 2304 | 2.3 MiB |
| **logits, 512 × 50257, fp32** | **~103 MB** |

At this model size the logits dominate everything else. That is why "logits only where
you will sample" (§13) is a memory optimisation as well as a compute one: gathering 3
rows instead of 512 shrinks 103 MB to ~0.6 MB, and every byte saved goes back to the
pool. For a large model the per-layer tensors take over — Llama-70B at a budget of 8192
tokens has an MLP intermediate of 8192 × 28672 × 2 B ≈ 470 MB for that one tensor, and
a few GB of headroom overall.

That is the real tension in the sizing: **a bigger token budget buys throughput per
iteration and costs KV blocks.** The embedding table, by contrast, costs 77 MB of pool
once and then stops mattering.

### The capacity check

```
max_model_len  ≤  num_blocks × block_size
```

vLLM refuses to start otherwise. This one check guarantees that **any single admitted
request can run to completion alone** — the property that makes preemption terminate
(§21). Running example: 16 ≤ 10 × 4 = 40 ✓.

## 3. The pool tensor

```
kv_pool  =  list over layers, each
            (2, num_blocks, block_size, n_kv_head, head_dim)
             ▲      ▲            ▲
            K/V   addressing   payload
```

The pool does not know sequences exist. Identity lives entirely in the block table.

**The layer axis stays outside the tensor** — a list of 12 tensors, not a 6-D tensor:

- **Layers are never indexed together.** Layer 3's attention runs at a different moment
  than layer 4's. The kernel wants a plain base pointer, not a stride across layers.
- **Layers can differ.** Sliding-window on some layers only, hybrid attention/Mamba
  stacks, per-layer fp8 KV, cross-attention. One dense tensor forces uniformity.
- **Allocation is friendlier** in 12 pieces than one contiguous multi-gigabyte block.

The *inner* axis order is kernel-dictated, not canonical. vLLM's older paged kernel
used **different shapes for K and V** — the K cache split head_dim into
`(…, head_size/x, block_size, x)` for vectorised coalesced loads while V stayed
`(…, head_size, block_size)`. Treat `(block_size, n_kv_head, head_dim)` as the logical
content and expect the real layout to be whatever the kernel wants.

## 4. The free list and the block table

Two separate structures, easily conflated:

```python
free_list   = deque([9, 2, 7, 5, 1, 4, 8, 3, 6, 0])             # flat, no sequence axis
block_table = (max_num_seqs, max_blocks_per_seq) int32           # the mapping
```

Allocation moves an id from one to the other; release moves it back.

**Both block-table dimensions are capacities, not counts.** `max_num_seqs` is how many
sequences can run at once; a row is a *slot* that is reused as requests come and go.
The second axis is derived, and should be clamped — a sequence cannot own more blocks
than exist:

```python
max_blocks_per_seq = min(ceil(max_model_len / block_size), num_blocks)
```

At scale, 256 × 512 × 4 bytes = **512 KiB of int32 manages ten gigabytes of KV.** The
expensive thing is allocated once and never moved; all flexibility lives in a tiny
index tensor.

### The block table exists twice

One logical mapping, two physical residences:

```
HOST (authority, mutable)                        DEVICE (consumer, read-only)
┌──────────────────────────────────┐             ┌─────────────────────────────┐
│ free_list       [9, 2, 7, …]     │             │ block_table                 │
│ refcounts       {9: 1, 2: 1, …}  │   H2D       │   (max_num_seqs,            │
│ prefix hashes   {h0: 9, …}       │  ───────▶   │    max_blocks_per_seq) int32│
│ ─────────────────────────────────│   dirty     │                             │
│ pinned staging mirror (dense,    │   rows      │  read inside the attention  │
│ same layout as the device tensor)│   only      │  kernel, per logical block  │
└──────────────────────────────────┘             └─────────────────────────────┘
   written by the scheduler                         never written by the GPU
```

Neither side can be dropped:

- **CPU only** fails on the read path. The kernel does one lookup per logical block,
  per head, per sequence, every step — thousands of lookups, which must come from
  device memory at device latency.
- **GPU only** fails on the write path. Allocation is branchy control flow — pop the
  free list, check the watermark, pick a preemption victim, bump refcounts — that needs
  the current table to decide. Keeping it only on device means a read-back and a sync
  every iteration.

The flow is **one-directional**: host writes, device reads, the GPU never modifies the
block table. No coherence problem, no read-back. The staging mirror exists so the H2D
transfer is one contiguous memcpy rather than a gather from Python lists.

## 5. Address translation — the four index spaces

The thing most worth keeping straight before anything is scheduled:

```
block_table[seq][logical]  =  physical      and     offset = pos % block_size
            ▲      ▲           ▲                            ▲
   the address you ask with    what comes back      bypasses the table entirely
```

| quantity | range (at scale) | source |
|---|---|---|
| `seq` | 0..255 | which request slot — the scheduler assigns it |
| `logical` | 0..511 | `pos // block_size` |
| `physical` | 0..18,203 | **the block table's stored value** |
| `offset` | 0..15 | `pos % block_size` |

**The block table stores only physical block ids** — never offsets, never sequence
indices. A row is a plain list: positions are logical, contents are physical.

```python
block_table[R1] = [9, 2, 7]   # R1's blocks, in order
#   position in the list  →  logical block  (0, 1, 2  always contiguous)
#   contents of the list  →  physical block (9, 2, 7  arbitrary, scattered)
```

So `[9, 2]` is **two scattered blocks, not a coordinate pair.** A `(block, offset)`
pair exists only one step later, *built* from the table's output plus the modulo, and
flattened into a slot:

```
slot = block_table[seq][pos // block_size] * block_size + (pos % block_size)

R1, pos 9  →  logical 2, offset 1  →  physical block_table[R1][2] = 7
           →  kv_pool[layer][:, 7, 1]   →  slot 7*4 + 1 = 29
```

Axis 0 of the pool needs a table lookup; axis 1 is pure modular arithmetic.

Every sequence believes it owns a contiguous `0 … n`. None of them do. This is virtual
memory, one for one:

| OS | engine |
|---|---|
| virtual page | logical block |
| page table | block table |
| physical frame | physical block |
| frame allocator | free list |
| shared page (COW) | shared prefix block, refcounted |
| multi-level page table | hierarchical block table (Part V-B) |

**A row is only partially valid.** A sequence holding 3 blocks uses columns 0–2; the
rest are stale values from whoever had the slot before. Nothing in the table says where
a sequence ends, which is why a length travels alongside it:

```
block_table[R1] = [9, 2, 7, ⟨stale⟩]
seqused_k[R1]   = 9   →  kernel walks ceil(9/4) = 3 entries, ignores the rest
```

`block_table` says *where*, `seqused_k` says *how many*. Neither is sufficient alone.

**Writes and reads resolve addresses in different places:**

| | resolved where | how many addresses |
|---|---|---|
| **write** — `slot_mapping` | host, at pack time | `N` (tokens this step) |
| **read** — `block_table` walk | device, in the kernel | `Σ kv_len / block_size`, far more |

The host precomputes write slots because there are only as many as there are new
tokens. It cannot precompute read addresses — a long-context sequence needs thousands
of block ids every step — so reads pay a device-side indirection instead.

## 6. Persistent input buffers and CUDA graphs

Startup also allocates the per-iteration inputs **once**, at maximum size, at fixed
device addresses:

```
input_ids       (max_num_batched_tokens,)
positions       (max_num_batched_tokens,)
slot_mapping    (max_num_batched_tokens,)
query_start     (max_num_seqs + 1,)        cu_seqlens_q
seq_lens        (max_num_seqs,)            seqused_k
block_table     (max_num_seqs, max_blocks_per_seq)
```

Then it captures CUDA graphs for decode at a set of batch-size buckets (1, 2, 4, 8, 16,
…). A graph replays its kernels with the **exact pointers** it recorded, which shapes
everything later:

- **"Ship the arrays" means copy *into* these buffers.** Each iteration overwrites the
  first `N` entries; nothing new is allocated.
- **Decode batches are padded up to the nearest bucket.** A batch of 5 replays the
  8-graph. The 3 padding rows get `slot_mapping = −1`, and the write kernel skips
  negative slots — vLLM's `reshape_and_cache` does exactly this — so padding never
  touches the pool.

Mixed and prefill-heavy batches have varying token counts and run eagerly, or with
piecewise graphs that capture everything except attention. This is the paged version
of a static-shape decode loop: shapes are fixed by the bucket, and all variation lives
in the *contents* of fixed buffers.

## 7. Starting state

```
free_list      [9, 2, 7, 5, 1, 4, 8, 3, 6, 0]      pops from the front
block_table    (4, 4), every entry stale           max_blocks_per_seq = min(4, 10) = 4
refcounts      all 0
prefix hashes  empty
waiting = running = []
```

---

# Part II — The first request, cold

R1 = `s0 s1 s2 s3 s4 s5 a0` (7 tokens), `max_tokens = 8`. Nothing is cached.

## 8. Arrival

### The request object

Every later step mutates fields of one per-request record:

```python
class Request:
    request_id
    prompt_ids:          list[int]
    output_ids:          list[int]       # grows by one per decode step
    num_computed_tokens: int             # how many tokens have K/V in the pool
    block_ids:           list[int]       # this sequence's block-table row
    sampling_params                      # temperature, top_p, top_k, seed, max_tokens, stop, n …
    status                               # WAITING | RUNNING | PREEMPTED | FINISHED_* | ABORTED
    arrival_time, priority
```

"Prefilling" and "decoding" are **not statuses**. They are derived: a request is still
prefilling while `num_computed_tokens < len(prompt_ids)`.

```
  WAITING ──admit──▶ RUNNING ──eos / stop / length──▶ FINISHED
     ▲                 │   │
     └──── preempt ────┘   └──── client abort ───────▶ ABORTED
```

### Validate before queueing

Reject up front anything the engine could never finish:

```python
if len(prompt_ids) >= max_model_len:  reject               # can never fit
max_tokens = min(max_tokens, max_model_len - len(prompt_ids))
```

Together with the startup capacity check (§2), this guarantees every queued request can
complete. Without it, a prompt needing more blocks than the pool holds sits in
`WAITING` forever. R1: 7 + 8 = 15 ≤ 16 ✓.

### Backpressure

The waiting queue is bounded. When it is full, or a request has waited past a timeout,
the frontend rejects with a retryable error (HTTP 429/503) instead of queueing without
limit. Latency under overload is decided here, not in the scheduler.

Then tokenize — in the frontend, not the engine core — and push onto `WAITING`. That
push is the request's **only** interaction with engine state until it finishes; §20
explains why.

## 9. Schedule — spend the token budget

```python
budget = max_num_batched_tokens
for req in running:                      # decodes are cheap, take them first
    take(req, 1); budget -= 1
for req in waiting_or_partial:           # spend the rest on prompt tokens
    n = min(budget, req.prompt_len - req.num_computed_tokens)
    take(req, n); budget -= n
```

There is no prefill branch and no decode branch. The scheduler only decides *how many
tokens* each sequence contributes. "Prefill" means `q_len > 1`; "decode" means
`q_len == 1`.

Iteration 1: nothing is running, so R1 takes `min(budget, 7) = 7` tokens.

At scale:

```
budget 512
  A  running → 1 token                        budget 511
  C  running → 1 token                        budget 510
  B  waiting → min(510, 1200) = 510 tokens    budget 0

scheduler output:  {A: 1, B: 510, C: 1}
```

B's prompt does not block anyone. It takes 510 tokens now and the rest over later
iterations — **chunked prefill**.

### Policy

The loop above is FCFS with decodes first. Real schedulers expose:

- **Decode-first vs prefill-first.** Decode-first protects inter-token latency of
  running requests (TPOT); prefill-first improves time-to-first-token (TTFT) for new
  arrivals. Chunked prefill is the compromise.
- **`max_num_seqs`** caps concurrency independently of the token budget.
- **A long-prefill cap** — the most tokens one request may take per step — so one huge
  prompt cannot eat the whole budget and stall everyone's decode.
- **Priority** — ordering the waiting queue, and choosing preemption victims (§21), by
  priority instead of arrival.
- **Starvation** — with priorities or prefix-aware ordering, an old low-priority
  request can wait forever without an aging rule.

A leftover-budget corner: if 3 tokens of budget remain and the next request has a
1200-token prompt, it *can* take a 3-token chunk — correct, but a poor forward.
Schedulers carry a minimum-chunk policy rather than spending budget to zero.

### It is CPU scheduling — with three differences

The policy layer is classic OS process scheduling, applied to requests instead of
processes. Most of the vocabulary maps one-to-one:

| OS CPU scheduling | LLM engine |
|---|---|
| ready queue | `WAITING` |
| running set | `RUNNING` |
| FCFS | admit in arrival order |
| priority scheduling | `priority` on the request |
| aging (anti-starvation) | aging rule for long-waiting requests |
| time quantum | long-prefill cap / chunk size |
| preemption | preempting a running request (§21) |
| swap to disk | swap preemption (KV → CPU memory) |
| kill and restart | recompute preemption |
| interactive vs. batch jobs | decode (latency-sensitive) vs. prefill (throughput) |
| admission control / thrashing | the watermark (§10, §21) |
| unknown CPU burst length | unknown output length |

The central tension is the same too. Interactive responsiveness versus batch throughput
is TPOT versus TTFT, and chunked prefill plays the role time-slicing plays in an OS: no
long job monopolises the processor.

Where it differs:

- **It schedules a batch, not one job.** An OS core runs one process per time slice, so
  its question is "who runs next?". Here the question is "what mix of requests goes
  into this forward, and how many tokens does each get?" — closer to packing a bin each
  tick, constrained by the budget, than to picking one job from a queue.
- **Two resources are scheduled together.** An OS CPU scheduler and its memory manager
  are mostly separate. Here every decision must pass both gates at once — compute
  (tokens) and memory (blocks) — so the scheduler is the CPU scheduler *and* the page
  allocator. That is why preemption is triggered by memory pressure, not by a timer.
- **Preemption costs are lopsided.** An OS context switch saves registers and loses
  nothing. Recompute preemption throws away KV that must be rebuilt, so it is a last
  resort, not a routine time-slice expiry. (The prefix cache gets some of it back.)

Two smaller ones: there is **no fixed quantum** — one iteration is the scheduling tick,
its length varies with the batch, and a decode gives up its turn after every token anyway.
And run time is **as unpredictable as an OS burst**: output length is unknown in
advance, so shortest-job-first is only ever approximated, by predicting output length;
most engines stick with FCFS plus priorities.

So "an OS scheduler and page allocator fused into one loop, where each tick is one
forward pass" is a fair description. Together with the virtual-memory table in §5, the
engine is essentially a small operating system whose single "CPU" is a batched forward.

### Budget and block size are deliberately unaligned

Two independent quantisations measuring two different resources:

| | measures | unit | chosen for |
|---|---|---|---|
| `max_num_batched_tokens` | **compute** in one forward | 1 token | iteration latency / GEMM efficiency |
| `block_size` | **memory** granularity | 16 tokens | fragmentation vs. kernel tiling |

A 510-token chunk is not a multiple of 16, and that costs nothing — because positions
are absolute within a sequence and the slot formula uses `pos % block_size`. A chunk
boundary landing mid-block means the next chunk **resumes writing into the same
partially filled block** at the right offset:

```
chunk 1:  pos    0 … 509    L0 … L31      L31 covers pos 496..511, holds 14
chunk 2:  pos  510 … 1019   L31 … L63     ← starts at offset 14 of L31, already owned
chunk 3:  pos 1020 … 1199   L63 … L74     ← starts at offset 12 of L63, already owned
```

No padding to block boundaries, no realignment. A chunk can end anywhere.

## 10. Find free blocks

### Step by step

```python
# 1. prefix-cache lookup (Part III). On a cold cache every lookup misses.
hit_blocks, num_computed = lookup(req)                       # R1: [], 0

# 2. how many NEW blocks this step's tokens need
need = ceil((num_computed + num_new) / block_size) - len(req.block_ids)
                                                             # R1: ceil(7/4) - 0 = 2
# 3. can memory back it?
if len(free_list) - need < watermark:
    leave in WAITING           # for a running request: preempt someone (§21)

# 4. take them
req.block_ids += [free_list.popleft() for _ in range(need)]  # R1: pop 9, pop 2
for b in new_blocks: refcount[b] = 1
```

R1 after admission:

```
block_table[R1] = [9, 2]          refcount 9 → 1, 2 → 1
free_list       = [7, 5, 1, 4, 8, 3, 6, 0]
row 0 is dirty                    (a block id entered it — §12)
```

### Allocation is incremental

Block count is driven by **cumulative** `kv_len`, but each step allocates only the
*delta*:

```python
to_allocate = ceil(new_kv_len / block_size) - ceil(old_kv_len / block_size)
```

At scale:

```
CHUNKED PREFILL — 1200-token prompt, 510-token chunks, block_size 16
  kv    0 ->  510   blocks   0 ->  32   allocate 32
  kv  510 -> 1020   blocks  32 ->  64   allocate 32
  kv 1020 -> 1200   blocks  64 ->  75   allocate 11

DECODE — one token per step
  kv 1200 -> 1201   blocks  75 ->  76   allocate  1   <- boundary crossed
  kv 1201 -> 1202   blocks  76 ->  76   allocate  0
```

The `ceil` on the old length already counted the partially filled tail block, so it is
never recounted. For one-token decode steps the condition reduces to
`num_computed_tokens % block_size == 0`.

- **Decode barely allocates.** 15 of every 16 steps do nothing; the 16th appends one
  block. The "allocator on the hot path" worry mostly evaporates.
- **Chunked prefill allocates per chunk, not per prompt.** A 1200-token prompt needs
  75 blocks eventually but only 32 at admission.

### The last chance to say no

```
1. spend token budget        →  q_len per sequence          (compute gate)
2. block delta               →  how many new blocks
3. free list + watermark     →  can memory back it?         (memory gate)
      no → shrink the chunk, leave it waiting, or preempt
4. pop free list             →  patch block-table rows, mark dirty
5. resolve slot_mapping      →  reads the now-updated table
   ──────────────────────────── commit ────────────────────────────
6. forward: compute k,v per layer → write to those addresses
```

**The destination is allocated before the values exist.** That is why `slot_mapping` is
host-computable at all: a slot is a function of `(pos, block_table)` and never of the
values.

Step 3 is the last point where the system can say no. Once the forward launches, the
kernel holds concrete addresses and will write to them — there is no failed allocation
to catch mid-forward. So admission control, the watermark, preemption and copy-on-write
all live in the scheduler and nowhere else: **the GPU is handed only decisions already
guaranteed satisfiable.**

Two gates, not one. The budget answers *how much compute fits in one forward*; the free
list answers *whether memory can hold the result*. A sequence can pass the first and
fail the second — that is the preemption path.

### Nothing reserves the generated length

```
RowKVCache (this repo):   reserve W = 1024 per row, up front
                          prompt 47 + max_new 32 → 79 used, 1024 reserved, 92% idle
                          never fails, never preempts

paged:                    allocate ceil(47/16) = 3 blocks now
                          add 1 more every 16 generated tokens
                          can run out → preemption
```

A sequence keeps asking for blocks until the free list is empty. The watermark is the
knob in between: refuse new admissions that would leave too little headroom for
sequences already running.

## 11. Pack into one flat row

R1, iteration 1:

| pos | logical | offset | physical | slot |
|---|---|---|---|---|
| 0 | 0 | 0 | 9 | 36 |
| 1 | 0 | 1 | 9 | 37 |
| 2 | 0 | 2 | 9 | 38 |
| 3 | 0 | 3 | 9 | 39 |
| 4 | **1** | 0 | **2** | **8** ← jumps backwards |
| 5 | 1 | 1 | 2 | 9 |
| 6 | 1 | 2 | 2 | 10 |

```
flat_ids     [s0 s1 s2 s3 s4 s5 a0]
flat_pos     [ 0  1  2  3  4  5  6]
cu_seqlens_q [0, 7]
seqused_k    [7]
slot_mapping [36 37 38 39  8  9 10]
```

Slot 39 → 8 is the block boundary: blocks are not adjacent, so the address jumps. The
write does not care — it is just an index.

### The five arrays

| Array | Shape | Granularity | Purpose |
|---|---|---|---|
| `flat_ids` | `(N,)` | per token | tokens to embed — concatenated, no padding |
| `flat_pos` | `(N,)` | per token | each token's **absolute position in its own sequence** |
| `cu_seqlens_q` | `(n_seq+1,)` | per boundary | who owns which rows |
| `seqused_k` | `(n_seq,)` | per sequence | how far back each may attend — `computed + q` |
| `slot_mapping` | `(N,)` | per token | where each token's K/V is **written** in the pool |

### At scale — why flat

```
                    512 rows, no padding, no batch dimension
      ┌───┬────────────────────────────────────┬───┐
      │ a │ b₀ b₁ b₂ … b₅₀₉                    │ c │
      └───┴────────────────────────────────────┴───┘
       0   1                                   511  512

cu_seqlens_q = [0, 1, 511, 512]      slice boundaries, n+1 entries for n sequences
seqused_k    = [134, 510, 7]         what each may attend to
```

Sequence *i* owns `flat[cu_seqlens_q[i] : cu_seqlens_q[i+1]]`.

`q_len ≠ kv_len` is the whole point. A contributes 1 token but sees 134; B contributes
510 and sees 510. Three attention problems of different shapes in one batch —
`1×134`, `510×510` causal, `1×7` — and no rectangle contains them without waste.
Padded, you would need 3×510 query slots, 1018 of 1530 doing nothing, plus a
`(3, 510, 644)` mask. The ragged form is 512 rows and two small integer arrays.

**On `cu_seqlens_k`:** cumulative *only* when K is also packed contiguously. With a
paged cache there is nothing to accumulate — the kernel receives raw lengths
(`seqused_k`) plus the block table and walks it itself.

### `slot_mapping` — one scatter for the whole batch

Because every token's destination is precomputed, the entire batch is written with
**one scatter per layer**, with no per-request loop and no prefill/decode branch:

```python
pool[slot_mapping] = k_new        # vLLM calls this reshape_and_cache
```

The kernel never performs allocator lookups on the write path — the host resolved them.
And **`slot_mapping` has no layer term**: the host computes it once per iteration and
all 12 layers reuse it, each against its own layer's pool.

The scatter is only correct if `slot_mapping` has no duplicates; §20 covers what
guarantees that.

### Where packing runs

On the **CPU**, every iteration, before the forward: it needs Python-side block tables
and per-request state. At high throughput this becomes a real bottleneck — hundreds of
requests through a Python loop each step — which is what async scheduling (§20) hides.

## 12. Sync to the device

Iteration 1 ships two kinds of data, both host → device, both into the buffers
allocated at startup (§6):

```
block_table   row 0 ← [9, 2]        PATCHED — only the row that changed
input_ids, positions, slot_mapping,
query_start, seq_lens               OVERWRITTEN — first N entries
```

```python
# scheduling wrote into pinned host mirrors; now one async copy per buffer
block_table_gpu[r0:r1, :max_blocks].copy_(bt_mirror[r0:r1, :max_blocks], non_blocking=True)
slot_mapping_gpu[:N].copy_(slot_mapping_pinned[:N], non_blocking=True)
# … same for input_ids, positions, query_start, seq_lens
```

Pinned source plus `non_blocking=True` means the host does not stall. The copies are
enqueued on the same stream as the forward, so they are ordered before the kernels that
read them. On a separate copy stream you need an event.

The two kinds are shipped differently for a reason:

| | lifetime | per-iteration transfer |
|---|---|---|
| the five arrays | one iteration | rebuilt from scratch, shipped whole |
| `block_table` | process | resident on device, **patched incrementally** |

The five arrays change completely every step. The block table is ~99.9% identical step
to step, so patching a resident tensor beats re-shipping it.

Which rows count as "changed" depends on every way a block id can enter a row. Part III
introduces the rest; §19 states the general rule.

## 13. Forward — where KV is written

### Most of the model does not care

Embeddings, LayerNorm, QKV projection, MLP and output projection are per-token. They
see `(N, C)` and run one large GEMM — bigger and more efficient than several padded
ones. Flattening is free for them. Positions enter once, at the embedding:
`x = wte[flat_ids] + wpe[flat_pos]`.

### The write sits inside every layer

Between the QKV projection and the attention math — 12 separate times per forward:

```
embed(flat_ids, flat_pos)
  ↓
layer 0:  ln → c_attn → q,k,v → [WRITE pool[0][slot_mapping]] → attend(reads pool[0]) → mlp
layer 1:  ln → c_attn → q,k,v → [WRITE pool[1][slot_mapping]] → attend(reads pool[1]) → mlp
  ⋮
layer 11: ln → c_attn → q,k,v → [WRITE pool[11][slot_mapping]] → attend(reads pool[11]) → mlp
  ↓
ln_f → gather sampled rows → lm_head → logits
```

**K and V do not exist until you are inside the layer.** They are not inputs to the
forward; they *are* the output of that layer's `c_attn`.

The write cannot move. **Not earlier:** layer 5's K depends on layers 0–4 having run.
**Not later:** the new token must attend to itself — defer the write past the attention
and the token is invisible to its own query. Within a layer it is strictly
**project → write → attend**.

Three real implementations, all in the same place:

| | writes | reads |
|---|---|---|
| classic paged (vLLM) | `reshape_and_cache` kernel | `paged_attention` kernel |
| fused (`flash_attn_with_kvcache`) | **same kernel** | same kernel |
| separate-new-KV | pool write deferred | kernel takes new k,v *and* the cache |

**Why "reshape":** the pool's physical layout is whatever the attention kernel wants,
historically not the projection's layout at all. The kernel permutes *while*
scattering rather than materialising a reshaped tensor first.

Scale per iteration at 512 tokens: 1.5 MB written per layer, **18 MB added to the
pool in total**. That is the entire mutation of the pool for the iteration; the 12
attention reads are read-only.

### Classic vs fused, concretely

**The maths and the result are identical. The difference is how many GPU programs run,
and therefore how the new K/V travels through memory.** One decode step, one layer, one
new token, 100 tokens of history:

```
CLASSIC — two kernels, one after the other
  kernel 1 (write)
    read  k_new, v_new                    from HBM
    write k_new, v_new → cache[slot 100]  to HBM
    ── kernel ends; the next cannot start until it finishes ──
  kernel 2 (attend)
    read  q
    read  cache[0 … 100]                  ← includes slot 100, just written
    softmax(q·Kᵀ)·V, write output

FUSED — one kernel does both
    read  q, k_new, v_new
    write k_new, v_new → cache[slot 100]
    read  cache[0 … 99]                   ← history only
    use   k_new, v_new already in hand    ← never read back
    softmax(q·Kᵀ)·V, write output
```

| | classic | fused |
|---|---|---|
| kernel launches per layer | 2 | 1 |
| new K/V re-read from memory | yes — written by kernel 1, re-read by kernel 2 | no — used from registers |
| wait between write and attend | a kernel boundary | none |
| result | same | same |

How much it matters is less than it sounds. Fused avoids re-reading **one** token's
K/V, but both still read the whole history — at decode that is ~99% of the traffic
either way. The real saving is **one launch per layer**, a few µs each, which is a large
share of a tiny decode kernel. CUDA graphs (§6) already remove most launch overhead, so
the gap shrinks further. Fused is a performance optimisation of the same computation,
not a different algorithm.

### Why fusing is not a rearrangement of Python

"Just reuse what is already in memory" is the right idea, but *which* memory matters:

```
registers / shared memory (SRAM)   per kernel, tiny, very fast    ← gone when the kernel ends
global memory (HBM)                shared by all kernels, large   ← the only thing that survives
```

When kernel 1 ends, its registers are released; kernel 2 is a separate program and
cannot see them. **The only way data passes between kernels is through HBM** — so
classic already "reuses" `k_new`, by reading it from HBM where kernel 1 left it. That
round trip is exactly what fused removes, and it can only be removed by putting the
write and the attention in the *same* kernel.

Each PyTorch op launches its own pre-built kernel: the indexed cache write is one, SDPA
is another (an opaque FlashAttention kernel). No ordering of Python calls merges them,
and `torch.compile` fuses elementwise ops but treats flash SDPA as a black box. So the
routes are: call a library kernel that already fuses them, or write one — both in
Part V-A.

### Separate new K/V — the merge, worked

The third design attends over the cached history and the new tokens **as two inputs**
and combines them:

```
o_hist, (m,l)_hist = attend(q, cache)          ← history only, from the pool
o_new,  (m,l)_new  = attend(q, k_new, v_new)   ← new tokens, causal, passed in directly
o = merge(o_hist, o_new)                       ← the (m, l, o) merge from Part V-A
```

Splitting softmax this way is **exact**. One query, two history keys and one new key:

```
history:  k1, k2   scores 1, 2        new:  k3   score 3

WANTED — attention over all three:
  e^(1-3), e^(2-3), e^(3-3) = 0.135, 0.368, 1        sum 1.503
  out = (0.135·v1 + 0.368·v2 + v3) / 1.503

PART 1 — history only            PART 2 — new key only
  m = 2                            m = 3
  l = e^-1 + e^0 = 1.368           l = 1
  o = 0.368·v1 + v2                o = v3

MERGE — rescale both to the common max, then add
  m   = max(2, 3) = 3
  c_h = e^(2-3) = 0.368            c_n = e^(3-3) = 1
  l   = 1.368·0.368 + 1            = 1.503
  o   = 0.368·(0.368·v1 + v2) + v3 = 0.135·v1 + 0.368·v2 + v3
  out = o / l                                                        ✓ identical
```

Softmax weights are `e^score / Σ e^score`; splitting the keys only splits that sum, and
`e^(m_old − m)` puts each piece on the same scale before adding. It is the same algebra
as online softmax and split-K.

The payoff: **attention no longer needs the new tokens to be in the pool**, so the
write can happen later or never.

- **Speculative decoding** — attend over 4 draft tokens, accept 2, write only those 2.
  With classic, all 4 would already be in the pool and 2 writes would need undoing.
- **Prefix-cache hits** — the history half is a paged read over shared blocks, the new
  half a dense causal block; each uses the kernel that suits its shape.

A fused kernel turns out to be this same merge with the write added inside it — see the
Triton version in Part V-A, whose step 3 is exactly the "new key only" half.

### Attention: the only op that needs sequence structure

Run plain attention over the flat array and R1 would attend to another request's keys.
Segmentation comes from `cu_seqlens_q` (which queries belong to whom) plus
`block_table` + `seqused_k` (which keys they may see). Each CTA finds its sequence,
loads its query tile, then walks that sequence's block-table row, translating logical
→ physical **on the device** and loading one block at a time. KV is never gathered into
contiguous memory and the `q × kv` matrix is never materialised. Part V-A has the
kernel in full: the block walk, online softmax, split-K, and FlashAttention vs
FlashDecoding.

### Logits only where you will sample

After the last layer there is one hidden vector per token in the flat batch. Only a
few of those tokens need a next-token prediction: **the last token of each sequence,
and only if that sequence samples this step.** So pick those rows *before* the
expensive `lm_head` projection, not after.

A plain GPT forward does the opposite — it projects every position and the caller
keeps one:

```python
x = ln_f(x)                     # (B, T, 768)
logits = lm_head(x)             # (B, T, 50257)   ← every position
next_logits = logits[:, -1, :]  # keep only the last
```

A 47-token prefill computes 47 × 50257 logits and throws away 46 rows of them.

#### The gather

```python
hidden = ln_f(x)                                   # (N, C): one row per flat token

# 1. where each sequence's last token sits in the flat batch
last_rows = cu_seqlens_q[1:] - 1                   # (n_seq,)

# 2. only sequences whose known tokens are all processed after this step sample;
#    a mid-prefill chunk's last token is not the end of its prompt
will_sample = num_computed + q_len >= num_tokens   # (n_seq,) bool
rows = last_rows[will_sample]                      # (n_sample,)

# 3. project only those rows
logits = lm_head(hidden[rows])                     # (n_sample, vocab), not (N, vocab)
# logits[i] belongs to the i-th SAMPLING sequence, not the i-th sequence
```

`num_tokens` is every token the request already knows — prompt plus outputs so far. It
equals the prompt length for a fresh request, but not for one readmitted after
preemption (§21), whose generated tokens are recomputed as if they were prompt.

**Line 1.** `cu_seqlens_q[i+1]` is where sequence *i* ends in the flat batch, so minus
one is its last row.

**Line 2.** A sequence still mid-prompt gets no prediction: the "next token" after its
chunk is not a sample, it is the prompt's own next token, arriving in the next chunk.

**Line 3.** `hidden[rows]` gathers a handful of rows, and only those are projected.

#### At scale — a mid-prefill chunk is dropped

A decoding, B prefilling 510 of its 1200 tokens, C decoding. The flat batch is 512
tokens:

```
cu_seqlens_q = [0, 1, 511, 512]          A: row 0   B: rows 1–510   C: row 511
last_rows    = [1, 511, 512] - 1  =  [0, 510, 511]

will_sample:   A  decoding                          → True
               B  0 + 510 < 1200, still mid-prompt  → False
               C  decoding                          → True

rows   = [0, 510, 511][T, F, T]  =  [0, 511]
hidden[rows]           (512, 768) → (2, 768)
logits = lm_head(...)  (2, 50257)            logits[0] → A,  logits[1] → C,  B has none
```

#### The running example, four cases

**Iteration 1 — a cold prefill.** R1 alone, 7 tokens.

```
cu_seqlens_q [0, 7]   last_rows [6]
R1: 0 + 7 >= 7 → True        rows [6]        1 × 50257 instead of 7 × 50257
```

**Iteration 4 — a prefix hit samples on its first step.** R1 decodes, R2 computes
positions 4–8 after its hit.

```
cu_seqlens_q [0, 1, 6]   last_rows [0, 5]
R1: decoding                     → True
R2: 4 + 5 >= 9 → True            ← prompt finishes this step, even though it never ran before
rows [0, 5]        hidden (6, 768) → (2, 768)
```

**Iteration 5 — one row, two samples.** R3 has `n = 2`.

```
cu_seqlens_q [0, 1, 2, 5]   last_rows [0, 1, 4]
all three sample → rows [0, 1, 4] → logits (3, 50257)
R3's row (logits[2]) is sampled TWICE → x0, y0 → the fork (§17)
```

So logit rows map neither 1:1 to sequences nor 1:1 to samples. The engine keeps both
mappings to route each sampled token back to its request.

**Iteration 6 — pure decode, nothing to drop.**

```
cu_seqlens_q [0, 1, 2, 3, 4]   last_rows [0, 1, 2, 3]   all sample
rows [0, 1, 2, 3]  — the gather selects every row
```

In pure decode every sequence contributes exactly one row and that row samples, so
`n_sample == N` and the gather is a no-op. **The saving comes from prefill-heavy
batches** — which is also when it is biggest.

#### Where it runs, and why it matters

The gather has to happen *between* `ln_f` and `lm_head`, inside the model. So the
model must accept the rows (or return hidden states) instead of projecting
unconditionally:

```python
def forward(self, idx, pos, …, sample_rows=None):
    x = …all blocks…
    x = self.transformer.ln_f(x)
    if sample_rows is not None:
        x = x[sample_rows]           # gather BEFORE the projection
    return self.lm_head(x)
```

`lm_head` is `(rows, 768) × (768, 50257)` — at 124M the largest single matmul and the
largest activation in the step:

| | logits computed | memory at fp32 |
|---|---|---|
| all 512 rows | 512 × 50257 | ~103 MB |
| gathered, 2 rows | 2 × 50257 | ~0.4 MB |

The saving is both compute and memory, and the memory goes straight back to the KV pool
(§2).

## 14. Sample, stop, stream

### Sampling a mixed batch

Every row carries its own sampling parameters, stored as per-row tensors so one batched
sampler serves all of them:

```
temperature (n_sample,)   top_p (n_sample,)   top_k (n_sample,)   one RNG generator per seeded request
```

Temperature 0 is greedy (argmax). Repetition, frequency and presence penalties need
each request's token history on the device — an output-token count per request — kept
up to date incrementally rather than rebuilt each step.

### Back to the host

The sampled ids, `(n_sample,)`, are copied device → host. This is the **one GPU→CPU
sync per iteration** — everything else flows host → device — and async scheduling
(§20) exists largely to hide it.

R1 samples `t0`, destined for position 7. Host update: `num_computed_tokens = 7`,
`output_ids = [t0]`.

### Stop conditions

Checked on the host after every sample:

- **EOS** (unless `ignore_eos`), and any `stop_token_ids`.
- **`max_tokens`** reached → `FINISHED_LENGTH`.
- **Stop strings.** These match *decoded text*, not ids, and can span several tokens:
  `"\n\nUser:"` might arrive as three tokens. Keep a tail of the last
  `len(longest_stop) − 1` characters and search new text together with it.

A request that stops is retired in the same update step (§21). The stopping token —
EOS in particular — is never fed back, so its K/V is never computed.

### The length stop — a truncated answer, reported honestly

EOS means the model decided it was done. A length stop means the engine cut it off —
from the client's side, often a truncated answer mid-sentence. The engine still needs
it, and the job is to make it visible rather than silent.

**Two different limits, only one chosen by the user:**

| limit | set by | why it exists |
|---|---|---|
| `max_tokens` | the client, per request | cost and latency budget — "don't spend more than this on me" |
| `max_model_len − prompt_len` | the server, unavoidably | the context window: past it there is no position to embed and no guaranteed KV capacity (§2, §8) |

Without some bound the engine has no defence against **runaway generation**: a model
that never emits EOS — stuck repeating a phrase, say — would hold its blocks forever,
crowd out other requests and burn GPU time. The length stop guarantees every request
leaves, and the preemption-termination argument in §21 depends on every request being
finite.

**Tell the client why it stopped.** The engine records which condition fired and the
API returns it — the OpenAI-style convention vLLM and SGLang follow:

```
finish_reason = "stop"     → EOS, a stop token, or a stop string — the model finished
finish_reason = "length"   → max_tokens or the context window — cut off
```

A well-behaved client checks it. On `"length"` the answer is incomplete, and it can:

- **retry with a larger `max_tokens`**, if it set one too low;
- **continue** — send prompt plus partial output back as a new request. Cheap, because
  the whole previous context is a prefix-cache hit (§16) and only new tokens compute;
- **ask for brevity**, if the context window itself was the limit.

**Where it genuinely goes wrong:**

- **Bad defaults.** If the server's default `max_tokens` is small — some older APIs
  defaulted to 16 — and the client never sets it, users see mysterious truncation. The
  sensible default is "up to the context window", `max_model_len − prompt_len`, which
  is what validation (§8) clamps to anyway.
- **Base models rarely stop on their own.** An instruction-tuned chat model is trained to
  emit EOS when its answer is complete, so `"stop"` is the normal outcome. A *base*
  model — this repo's GPT-2 — was trained on continuous web text and has little reason
  to emit `<|endoftext|>` at a natural answer boundary. For it, `"length"` is the
  *usual* outcome, not an edge case. That is a property of the model, not a serving bug.

So the length stop is a safety bound the engine cannot do without. What makes it
acceptable is honest reporting, sensible defaults, and cheap continuation.

### Incremental detokenization

Byte-level BPE tokens do not align with characters: one token can end halfway through a
multi-byte UTF-8 character. So detokenize incrementally — decode the accumulated bytes,
**emit only up to the last complete character**, and hold the rest for the next step.
Emitting eagerly produces `�` in the stream.

The same holdback applies to stop strings: text that could be the beginning of a stop
string is not streamed until the next token resolves it, so a stop string never leaks
half-printed.

**Flush on finish.** Holdback assumes another token is coming. When a request stops —
on length especially, which can cut mid-character — the held-back bytes and text need
an explicit decision: drop an incomplete UTF-8 sequence or emit a replacement
character, and release held-back text that turned out not to be the start of a stop
string after all. Otherwise the stream ends one partial character or a few characters
short.

### Streaming

Each request has its own output queue. The engine pushes `(request_id, new_token_ids,
finished)` after every step; the frontend detokenizes and streams the text delta to the
client. Detokenization lives in the frontend, off the engine core's critical path.

## 15. Decode — iterations 2 and 3

### Why the sampled token is fed back

Sampling produces only an **id**. Its K/V have never been computed, so nothing can
attend to it until they are:

```
iteration 1
  pool holds nothing for R1
  feed s0…a0 (pos 0..6), write their K/V, attend
  logits at pos 6  →  sample t0       ← "the token at position 7", but no K/V yet

iteration 2
  feed t0                              ← this is why it appears in flat_ids
  compute its K and V, write at position 7
  attend over 0..7  →  sample t1 for position 8
```

The fed-back token does two jobs: its K/V enter the pool so future tokens can attend to
it, and its hidden state produces the logits for the next position. That is why
`q_len` is exactly 1 in decode — everything earlier is already cached.

### Iteration 2 — no allocation

```
R1 feeds t0 at pos 7:  delta = ceil(8/4) - ceil(7/4) = 0  → no new block
slot = block_table[R1][1]*4 + 3 = 2*4 + 3 = 11

flat_ids [t0]   flat_pos [7]   cu_seqlens_q [0, 1]   seqused_k [8]   slot_mapping [11]
```

No block id entered any row, so **no block-table copy at all** — only the five arrays
move. Block 2 is now full: `[s4 s5 a0 t0]`. Part III shows why that matters. Sample
`t1` for position 8.

### Iteration 3 — growth

```
R1 feeds t1 at pos 8:  8 % 4 == 0 → delta = ceil(9/4) - ceil(8/4) = 1 → pop 7
block_table[R1] = [9, 2, 7]                 row 0 dirty again
slot = 7*4 + 0 = 28                         seqused_k [9]
```

Sample `t2` for position 9.

```
block 9  [s0 s1 s2 s3]   full
block 2  [s4 s5 a0 t0]   full
block 7  [t1  ·  ·  · ]   the open tail
free     [5, 1, 4, 8, 3, 6, 0]
```

That is the decode steady state: 3 of every 4 steps here (15 of 16 at `block_size = 16`)
touch only the five arrays; the next appends one id.

### Prefill and decode are the same write

| | prefill (iteration 1) | decode (iterations 2, 3) |
|---|---|---|
| rows contributed | `q_len` (7) | 1 |
| slot pattern | consecutive run, jumping at block edges | one isolated offset |
| allocation before | `ceil(q_len / bs)` | 1 every `bs` steps, else 0 |
| kernel | `reshape_and_cache` | `reshape_and_cache` |

`slot = f(pos)` has no phase term — `pos` belongs to a token, not to a phase. Prefill's
run is coalesced and decode's singleton is scattered, and it does not matter: per layer
decode writes 3 KiB and reads `kv_len × 3 KiB`, so at `kv_len = 134` it is **134:1**.
Decode is bandwidth-bound on reading history; its write is noise.

---

# Part III — Later requests

## 16. Prefix cache — R2 reuses R1's work

### Why sharing is sound

A token's K/V is a deterministic function of the token ids up to and including it, at
their positions. If two sequences have **identical tokens at identical positions**,
their K/V are bit-identical, and both block tables can point at the same physical
blocks.

### Registration — which blocks become shareable, and when

A block is registered under its hash **when it becomes full**, whether prefill or
decode filled it. Hashes are **chained**, Merkle-style:

```python
h[0] = H(None,  tokens[0:4])
h[1] = H(h[0],  tokens[4:8])
h[2] = H(h[1],  tokens[8:12])
```

The chain is mandatory: identical tokens at the same offset but after *different*
history produce different K/V, so each key covers all preceding context — plus anything
else that changes the values (LoRA adapter id, image hashes for multimodal).

R1's registrations by the end of iteration 3:

```
H(None, [s0 s1 s2 s3])  → 9      filled by prefill, iteration 1
H(h0,   [s4 s5 a0 t0])  → 2      filled by DECODE,  iteration 2
block 7  [t1 · · ·]               partial — not registered
```

Block 2 contains a generated token. Decode-produced blocks are as shareable as prompt
blocks — which is what makes multi-turn chat cheap: turn 2's prompt is turn 1's prompt
plus turn 1's output, and all of it hits.

**Partial blocks are never registered.** More tokens may still be appended, so they
have no stable hash. Whether registration happens at schedule time or after the forward
completes is an engine detail; either is safe, because any request that hits a block
reads it after the write in stream order.

### The block hash, concretely

There is no single standard formula, but every block-hashing engine has the same shape:
**serialise (parent hash, the block's token ids, extra keys) to bytes, then apply a
cryptographic hash.**

```
h_0 = SHA256( ROOT    ‖ ids[0:4]      ‖ extra )
h_i = SHA256( h_{i-1} ‖ ids[4i:4i+4]  ‖ extra )          ‖ = byte concatenation
```

```python
import hashlib, struct

ROOT = b"\x00" * 32          # parent of block 0 (a fixed seed)

def block_hash(parent: bytes, token_ids, extra: bytes = b"") -> bytes:
    h = hashlib.sha256()
    h.update(parent)                                          # 32 bytes: whole prefix before this block
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))  # 4 bytes per token id, fixed width
    h.update(struct.pack("<I", len(extra)) + extra)           # length-prefixed extra keys
    return h.digest()                                         # 32 bytes, becomes the next parent

def hashes_for(token_ids, block_size=4, extra=b""):
    out, parent = [], ROOT
    for i in range(len(token_ids) // block_size):             # full blocks only
        parent = block_hash(parent, token_ids[i*block_size:(i+1)*block_size], extra)
        out.append(parent)
    return out
```

| input | why it is there |
|---|---|
| `parent` | K/V depend on *all* preceding tokens; chaining covers the whole prefix without rehashing it |
| `token_ids` | exactly `block_size` ids — only full blocks get a hash |
| `extra` | anything else that changes the K/V values: LoRA adapter id, image/audio hashes, a per-tenant salt |

Run on the running example (token ids stand in for `s0…`, `a0`, `b0…`):

```
R1  ['96acd3c53eff', 'ada33e72b2b2']     block 0 = s0..s3,  block 1 = s4 s5 a0 t0
R2  ['96acd3c53eff', '8332441aa735']     block 0 identical → hit;  block 1 differs → miss
                                         (R2's third block is partial → no hash)

same tokens (7,7,7,7) at two positions:  ['1d8f1434083f', '182b99f0ff59']   equal? False
tenant salt on block 0:                  '96acd3c53eff' vs 'e80a991da986'
```

#### Every hash depends on its parent

Change anything earlier, and every hash from that point on changes:

```
R1: [s0 s1 s2 s3] [s4 s5 a0 t0] [t1 t2 t3 t4]
     h0             h1             h2
R2: [s0 s1 s2 s3] [s4 s5 b0 b1] [ … ]
     h0  ✓ same     h1' ✗          h2' ✗ differs too, even if its own tokens matched R1's
```

That one property produces three behaviours:

| consequence | why |
|---|---|
| **the hash identifies the whole prefix**, not just the block | the parent carries everything before it |
| **position is encoded** — no position field needed | the same tokens at a different depth have a different parent |
| **lookup stops at the first miss** | after a miss, every later hash is built on a parent no one registered |

Position matters because K/V are position-bound (§1), and the chain encodes it: `h2`
means "these tokens, preceded by exactly `b0` and `b1`", which can only sit at positions
8–11. Hashing blocks *independently* with an explicit position would still be wrong,
because from layer 1 onward every K mixes in the whole prefix through attention. The
chain covers tokens, position and prefix at once. Position would need to go into `extra`
only if positions stopped counting from 0 along the sequence — per-request offsets, or
per-request position-encoding settings such as RoPE scaling.

#### Three details that make it correct

- **Unambiguous serialisation.** Token ids are packed fixed-width. Joined as text,
  `[1, 23]` and `[12, 3]` would both become `"123"` — two blocks, one hash.
- **The full digest is the parent.** Truncating it, say to 8 bytes, brings back the
  collision risk the cryptographic hash exists to remove.
- **The root seed decides stability across restarts.** A fixed `ROOT` gives every
  process the same hashes — required when hashes are shared between replicas or with an
  external KV store (Part V-C). A random per-process root makes hashes meaningless
  outside the process: fine for a local cache, wrong for a shared one.

#### Choosing the hash function

A collision means a request silently reuses **another request's K/V** — wrong output
with no error, and in a multi-tenant server, a data leak. A 64-bit non-cryptographic
hash such as Python's built-in `hash` is fast but collisions are possible and can be
engineered; SHA-256 makes them practically impossible. vLLM moved from Python's built-in
hash to SHA-256 after a reported collision issue.

A **per-tenant salt** in `extra` goes further: tenants who share a system prompt still
get different hashes, so they never share blocks — which also closes the timing side
channel of "this prefix was cached, so someone else sent it".

Cost: one SHA-256 over ~50 bytes per block, under a microsecond, computed once per block
on the CPU when it fills. Each request keeps its list of block hashes and extends it
incrementally; nothing is rehashed.

### How the table is stored

Two directions, because two operations need them:

```python
# 1. lookup: "has anyone computed this prefix block?"
hash_to_block: dict[BlockHash, BlockId]
    h0  → 9
    h1  → 2
    h1' → 5

# 2. eviction: "this block is being reused — which hash entry do I remove?"
class Block:
    block_id:   int
    ref_cnt:    int
    block_hash: BlockHash | None      # None = never registered (partial, or not yet full)
```

**Block id, not slot.** Sharing happens at block granularity — a hit hands a whole block
to a new block table. A slot (`block * block_size + offset`, §11) is an address *inside*
a block, derived when packing a write; storing it would duplicate what the block id
already says.

**The reverse pointer** exists for eviction. When the allocator pops a cached
(refcount-0) block for reuse, its table entry must go, or a later lookup would "hit" a
block now holding someone else's tokens. `block.block_hash` names the entry to delete in
O(1).

vLLM's real map is `hash → {block_id: block}` — a hash to a *set* of blocks. Two
requests that both missed in the same step can compute identical content into two
different blocks; either serves a later hit, and the duplicate is wasted memory until it
is evicted.

### The lookup path

At admission, hash the prompt's full blocks, then walk them in order — one dict lookup
per block — and **stop at the first miss**:

```python
def admit(request):
    hashes = hashes_for(request.prompt_ids)          # one per FULL block

    hit_blocks = []
    for h in hashes:
        block = hash_to_block.get(h)
        if block is None:
            break                                    # first miss → stop
        if block.ref_cnt == 0:
            remove_from_evictable(block)             # resurrect a cached block
        block.ref_cnt += 1
        hit_blocks.append(block)

    num_computed = min(len(hit_blocks) * block_size,
                       len(request.prompt_ids) - 1)  # keep ≥1 token to compute
    need = ceil(len(request.prompt_ids) / block_size) - len(hit_blocks)
    request.block_ids = [b.id for b in hit_blocks] + [free_list.popleft() for _ in range(need)]
```

**Stopping at the first miss is required, not just an optimisation.** Usually a later
block cannot hit anyway: `h2 = H(h1, …)` is built on the request's own `h1`, which no
one registered. The exception is eviction — a block under `h1` evicted while its child
under `h2` is still cached (the orphan case, §21). Then `h2` *would* hit, but the request
has no K/V for block 1, and attention needs the entire history. A cached block after a
gap is useless: the usable hit is always a **contiguous run from block 0**.

The lookup runs only at **admission** — a new request, or one readmitted after
preemption. Running requests never look up; when one of their blocks fills it is
*registered*, the write side of the same table.

### Lookup — iteration 4

R2 = `s0 s1 s2 s3 s4 s5 b0 b1 b2` (9 tokens). Walk the prompt's full blocks and **stop
at the first miss**:

```
L0 [s0 s1 s2 s3]   H(None,[s0..s3])     → HIT  block 9     refcount 1 → 2
L1 [s4 s5 b0 b1]   H(h0,[s4,s5,b0,b1])  → MISS (block 2 holds s4 s5 a0 t0)
                                           stop
L2 [b2]            partial — never looked up

num_computed_tokens = 4
need = ceil(9/4) - 1 = 2  → pop 5, pop 1
block_table[R2] = [9, 5, 1]                       row 1 dirty
```

R2 computes only positions 4–8, in the same forward as R1's decode:

| seq | pos | logical | offset | physical | slot |
|---|---|---|---|---|---|
| R1 | 9 | 2 | 1 | 7 | 29 |
| R2 | 4 | 1 | 0 | 5 | 20 |
| R2 | 5 | 1 | 1 | 5 | 21 |
| R2 | 6 | 1 | 2 | 5 | 22 |
| R2 | 7 | 1 | 3 | 5 | 23 |
| R2 | 8 | **2** | 0 | **1** | **4** |

```
flat_ids     [t2 | s4 s5 b0 b1 b2]
flat_pos     [ 9 |  4  5  6  7  8]     ← two position scales in one array; R2 starts at 4
cu_seqlens_q [0, 1, 6]
seqused_k    [10, 9]
slot_mapping [29 | 20 21 22 23  4]
```

**A prefix hit makes a new request look like a later chunk.** R2 has never run, yet
`q = 5 < kv = 9`. Nothing in the batch says "prefill"; only `q` and `kv` exist, and the
kernel treats every combination the same way:

```
R1:  q=1, kv=10    decoding
R2:  q=5, kv=9     a new request with 4 tokens of history it never computed
```

R1 samples `t3` (pos 10); R2 samples `u0` (pos 9). Block 5 `[s4 s5 b0 b1]` is now full
and gets registered.

### Three constraints

- **Full blocks only** — a partial block has no stable hash.
- **Prefix only.** Because of the chain, once block *i* misses, blocks *i+1*… are
  unusable even if their tokens match. Combined with position-bound K/V (§1), this is
  why it is *longest prefix match* and not substring matching.
- **Keep at least one token to compute.** A forward must produce logits. If the whole
  prompt hits, drop the last block and recompute it.

### Granularity loss

The shared prompt S is 6 tokens; R2 reused **4**. `s4 s5` share a block with
request-specific tokens, so their K/V is computed and stored again. By the end of
iteration 6 it exists in **four** separate blocks (2, 4, 5, 8).

Reusable tokens are `floor(shared_prefix_len / block_size) × block_size`, so up to
`block_size − 1` are always lost. At `block_size = 16`, a 6-token shared prompt shares
nothing.

### What it is worth

A 2000-token shared system prompt across 100 requests, at scale:

```
                prefill tokens      KV blocks
naive            100 × 2000 = 200k  12,500  = 7.4 GB
prefix-cached              2,000       125  =  74 MB
```

The single largest throughput lever in multi-tenant serving.

### The alternative: SGLang's radix tree

SGLang replaces "hash → block" with a **radix tree over token ids**. Each path from the
root spells a token prefix; each node stores where those tokens' K/V live in the pool.
Lookup walks the tree **comparing actual tokens** — no hashing at all. (This describes
SGLang's `RadixCache` at a high level; check the current `radix_cache.py` for
specifics.)

Two differences from block hashing, before any code:

- **An edge holds a variable-length run of tokens**, not a fixed block. Chains of
  single-child nodes are compressed into one edge — that is what makes it a *radix*
  tree.
- **The value is one KV slot per token.** SGLang's pool is traditionally token-granular
  (`page_size = 1`, Part V-D), so a node's value lists a slot for every token in its key.

#### The structure

```python
class Node:
    def __init__(self, key=(), value=(), parent=None):
        self.key = list(key)          # token ids on the edge INTO this node
        self.value = list(value)      # KV slot index for each of those tokens
        self.children = {}            # first token of child's key -> child
        self.parent = parent
        self.lock_ref = 0             # running requests using this node

class RadixCache:
    def __init__(self):
        self.root = Node()

    @staticmethod
    def _common(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    def _split(self, child, n):
        """Cut child's edge after n tokens. Returns the new middle node."""
        mid = Node(child.key[:n], child.value[:n], child.parent)
        mid.lock_ref = child.lock_ref
        mid.parent.children[mid.key[0]] = mid     # grandparent now points at mid
        child.key, child.value = child.key[n:], child.value[n:]
        child.parent = mid
        mid.children[child.key[0]] = child        # mid points at the shortened child
        return mid

    def match_prefix(self, tokens):
        """Longest cached prefix: returns (slots, last matched node)."""
        node, i, slots = self.root, 0, []
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            n = self._common(child.key, tokens[i:])
            if n < len(child.key):                # diverged INSIDE this edge
                child = self._split(child, n)
            slots += child.value
            i += n
            node = child
        return slots, node

    def insert(self, tokens, slots):
        """Add tokens -> slots; returns how many were already cached."""
        node, i = self.root, 0
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            n = self._common(child.key, tokens[i:])
            if n < len(child.key):
                child = self._split(child, n)
            i += n
            node = child
        if i < len(tokens):                       # attach the unmatched remainder as a leaf
            leaf = Node(tokens[i:], slots[i:], node)
            node.children[tokens[i]] = leaf
        return i
```

#### Worked through, step by step

The same requests as the running example, plus an R4 = `s0 s1 s2 s3 s4 d1 d2` that
diverges one token earlier. Slot numbers here are illustrative token-granular slots, not
the block example's.

**1. R1 finishes prefill.** Its 8 tokens are in slots 10–17. Nothing has diverged, so
`insert` attaches everything as **one edge**:

```
root
└─ [s0 s1 s2 s3 s4 s5 a0 t0]   [10 11 12 13 14 15 16 17]
```

**2. R2 arrives: `match_prefix`.** It diverges at `b0` vs `a0` — position 6 of an
8-token edge, so *inside* it:

```
_common([s0 … s5 a0 t0], [s0 … s5 b0 b1 b2]) = 6      6 < 8 → _split(node, 6)

before:  root ── [s0 s1 s2 s3 s4 s5 a0 t0] / [10 … 17]
after:   root ── [s0 s1 s2 s3 s4 s5] / [10 11 12 13 14 15]   ← new middle node
                     └── [a0 t0] / [16 17]                    ← the original node, shortened
```

R1's 8 slots are not shared as a unit — they are **divided**: 10–15 go to the new shared
node, 16–17 stay with R1's tail. `match_prefix` returns `[10 … 15]`.

**Nothing moves in GPU memory.** The K/V in slots 10–17 stays where it is; `_split` only
slices two Python lists. The tree is bookkeeping on top of the pool.

**3. R2 computes `b0 b1 b2`** into slots 20–22 and inserts its full sequence. The walk
matches `[s0 … s5]` fully, finds no child keyed `b0`, and attaches a leaf:

```
root
└─ [s0 s1 s2 s3 s4 s5]  [10 11 12 13 14 15]
    ├─ [a0 t0]          [16 17]            R1's tail
    └─ [b0 b1 b2]       [20 21 22]         R2's tail
```

R2 reuses **all 6** shared tokens. Block hashing at `block_size = 4` reused 4 — the
granularity loss above disappears, because the tree splits exactly where sequences
diverge.

**4. R3 arrives** (`s0 … s5 c0`). Its match ends *exactly* at the end of `[s0 … s5]`, so
no split is needed — just a new leaf:

```
root
└─ [s0 s1 s2 s3 s4 s5]  [10 11 12 13 14 15]
    ├─ [a0 t0]          [16 17]
    ├─ [b0 b1 b2]       [20 21 22]
    └─ [c0]             [30]
```

**5. R4 arrives** (`s0 s1 s2 s3 s4 d1 d2`). It diverges at `d1` vs `s5` — inside the
*already shared* edge, at position 5:

```
_common([s0 s1 s2 s3 s4 s5], [s0 s1 s2 s3 s4 d1 d2]) = 5      5 < 6 → _split(node, 5)

root
└─ [s0 s1 s2 s3 s4]     [10 11 12 13 14]      R1 R2 R3 R4
    ├─ [s5]             [15]                  R1 R2 R3
    │   ├─ [a0 t0]      [16 17]               R1
    │   ├─ [b0 b1 b2]   [20 21 22]            R2
    │   └─ [c0]         [30]                  R3
    └─ [d1 d2]          [40 41]               R4 — computed, then inserted
```

The whole subtree `[a0 t0]`, `[b0 b1 b2]`, `[c0]` moved **for free**: `_split` creates the
middle node and re-parents the *original* node under it, and that node keeps its
`children`. A one-token node like `[s5]` is fine — node lengths are set entirely by where
sequences diverge.

#### The rules

- **A node is created by a split exactly where two token sequences diverge.**
- **`value` always lines up with `key`** — one slot per token; a split cuts both at the
  same index.
- **The slots for any prefix are the concatenated `value`s along its path.** R2's full
  KV is `[10 … 15] + [20 21 22]`.
- **A lookup ending mid-edge splits it; one ending at a node boundary does not.**
- **The tree shows sharing level by level** — read down a path and the sharer count
  falls: 4, then 3, then 1.

#### Locking, eviction, and why there is no merge

- **Lock / unlock** replaces vLLM's per-block refcount. A request that uses a matched
  prefix increments `lock_ref` on every node from its last matched node up to the root;
  it decrements on finish. A locked node cannot be evicted.
- **Evict** removes the least-recently-used **leaf** with `lock_ref == 0`, freeing its
  slots; if its parent becomes a leaf, the parent becomes a candidate. **Only leaves are
  evicted, so a cached prefix is always contiguous from the root** — the orphan problem
  of §21 cannot happen. vLLM gets tail-first eviction by convention (release in reverse);
  the tree gets it from its structure.
- **No merge after eviction.** Evicting `[d1 d2]` leaves `[s0 … s4]` with one child,
  `[s5]`. A strictly compressed tree would merge them back into `[s0 … s5]`; engines
  generally do not. Unmerged is still correct — it costs one extra node hop during a
  lookup that already compares every prompt token. Merging is riskier than it looks:
  running requests hold pointers to their last matched node, a request ending exactly at
  the parent locks it but not the child, and a divergence point that appeared once tends
  to reappear — merge-then-resplit churn.

#### Scheduling on top of the tree

Because the tree knows how much of each waiting prompt is cached, SGLang can order the
waiting queue by **longest prefix match** — prefer requests sharing the longest cached
prefix. That raises hit rates when many requests branch from a few shared prompts
(few-shot templates, agent loops, tree search), with the usual starvation risk of any
priority policy (§9).

#### Block hashing vs radix tree

| | vLLM: chained block hashes | SGLang: radix tree |
|---|---|---|
| key | `H(parent, block tokens)` | the token ids themselves |
| granularity | whole blocks | individual tokens (or pages, with `page_size > 1`) |
| collisions | possible in principle; SHA-256 makes them negligible | impossible — tokens are compared, not hashed |
| shared prefix of 6 tokens, block size 4 | reuses 4 | reuses 6 |
| lookup cost | one dict lookup per block | walk the tree comparing tokens |
| sharing count | `ref_cnt` per block | `lock_ref` per node |
| eviction | LRU over refcount-0 blocks; tail-first by convention | LRU over unlocked **leaves**; tail-first by structure |
| complexity | a dict and a pointer per block | node splitting, tree upkeep |

Neither is better outright. Block hashing is simpler and fits fixed-size paged blocks;
the radix tree matches exactly and enables prefix-aware scheduling, at the cost of a more
involved structure. With `page_size > 1`, SGLang compares keys page by page, which
brings back some of the same granularity trade-off in exchange for paged-kernel
efficiency.

## 17. Forks and copy-on-write — R3 samples twice

### Prefix caching never needs copy-on-write

Shared prefix blocks are full, and **a full block has no writable slot**: R2 writes
position 4 into its own block 5, never into block 9. Immutability is not a policy added
on top of sharing — it is what "full" means. Copy-on-write (CoW) arises only from
**forks**, where sequences share a *partial, still-growing* block. That is `n > 1`
sampling and beam search.

### Iteration 5 — R3 admitted with a prefix hit

R3 = `s0 … s5 c0` (7 tokens), `n = 2`:

```
L0 [s0..s3]   → HIT block 9       refcount 2 → 3
L1 [s4 s5 c0] partial — not looked up
num_computed_tokens = 4
need = ceil(7/4) - 1 = 1  → pop 4
block_table[R3] = [9, 4]                     row 2 dirty
```

R1 and R2 decode alongside; neither crosses a block boundary:

```
flat_ids     [t3 | u0 | s4 s5 c0]
flat_pos     [10 |  9 |  4  5  6]
cu_seqlens_q [0, 1, 2, 5]
seqused_k    [11, 10, 7]
slot_mapping [30 |  5 | 16 17 18]
```

### The fork

R3's last row is sampled **twice**, giving `x0` and `y0`, both for position 7. R3
becomes two sequences:

```
R3a (row 2) = [9, 4]
R3b (row 3) = [9, 4]          row copied verbatim → row 3 dirty
refcount  9: 3 → 4    4: 1 → 2
                          ▲
          block 4 = [s4 s5 c0 ·]   partial AND shared — the state prefix caching never creates
```

### Iteration 6 — copy before write

Both want position 7 → logical 1, offset 3 → **the same slot of block 4**. The decision
is made serially on the host, before anything is written:

```
HOST, scheduling
  R3a: refcount[4] == 2 > 1  → CoW
         8 = free_list.popleft()
         blocks_to_copy += (4 → 8)
         block_table[R3a] = [9, 8]          row 2 dirty
         refcount[4] 2 → 1
  R3b: refcount[4] == 1      → sole owner, writes in place

GPU, one stream, in order
  1. block-table patch (rows 2–3) + the five arrays
  2. copy block 4 → block 8              all 12 layers, K and V; carries s4 s5 c0
  3. forward
```

```
flat_ids     [t4 | u1 | x0 | y0]
flat_pos     [11 | 10 |  7 |  7]
cu_seqlens_q [0, 1, 2, 3, 4]
seqused_k    [12, 11, 8, 8]
slot_mapping [31 |  6 | 35 | 19]
                         ▲    ▲
                     8*4+3  4*4+3    distinct — the scatter cannot collide
```

Things worth keeping straight:

- **"On write" means just before it.** A write that landed in a shared block would
  already have corrupted the other sequence, and nothing downstream could detect it.
- **One copy, not two.** The last holder inherits the block outright; *k* forks cost
  *k − 1* copies. Who gets the copy is just the order the scheduler reached them —
  reverse it and the outcome is identical.
- **Once per fork per shared partial block.** Position 8 opens a fresh logical block
  for each; after the first divergent write they share nothing mutable.
- **The copy preserves the shared past.** Block 4 holds `s4 s5 c0`, which R3a must
  still attend over, so block 8 starts as a byte-identical duplicate.

This is the **only operation in the design that moves KV bytes** — everywhere else,
allocation hands out an integer. It is also why beam search is expensive to serve:
beams fork and die continuously, so CoW fires on every fork event.

### What engines actually do

The above is the classic design — vLLM's V0 block manager, which surfaced the copies as
`blocks_to_copy`. vLLM V1 dropped CoW entirely: `n > 1` fans out into `n` independent
requests that share the prompt's **full** blocks through the prefix cache, and each
recomputes the partial tail. A few tokens of redundant prefill buy an allocator with no
copy path. Check current source before relying on either.

## 18. Refcounts — one lifecycle

| event | refcount effect |
|---|---|
| block popped from the free list | → 1 |
| prefix hit on an in-use block | +1 |
| prefix hit on a **cached** block | 0 → 1, and it leaves the evictable set |
| fork | +1 on every block in the copied row |
| CoW | −1 on the shared block; the new block → 1 |
| finish / abort / preempt | −1 on every block in the row |
| reaches 0, block registered | **cached** — hash kept, evictable |
| reaches 0, block never registered | **free** |
| cached block popped by the allocator | hash evicted → reused as free |

The running example through iteration 6:

| | 9 | 2 | 7 | 5 | 1 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| it 1 — R1 admitted | 1 | 1 | | | | | |
| it 3 — R1 growth | 1 | 1 | 1 | | | | |
| it 4 — R2 admitted, hit 9 | **2** | 1 | 1 | 1 | 1 | | |
| it 5 — R3 admitted, hit 9 | **3** | 1 | 1 | 1 | 1 | 1 | |
| end of 5 — fork | **4** | 1 | 1 | 1 | 1 | **2** | |
| it 6 — CoW for R3a | 4 | 1 | 1 | 1 | 1 | **1** | 1 |

### Three block states, not two

```
in use      refcount > 0     some running sequence points at it
cached      refcount == 0    content valid, hash still registered, EVICTABLE
free        no hash entry    pure free space
```

A block whose refcount hits 0 is **not necessarily freed**. If it was registered it
stays findable and can serve a future hit, which resurrects it to refcount 1. Only when
the allocator needs space does it evict a cached block — drop its hash entry, reuse
it. A prefix lingers as long as memory pressure allows.

vLLM keeps cached and free blocks in **one LRU queue**: the allocator pops from the
front, and popping a block that still has a hash evicts that hash on the spot. Eviction
order is covered in §21.

Because blocks are shared, release is always `refcount -= 1; if 0: …`, never an
unconditional return to the pool.

## 19. The dirty rule, in general

**A block-table row syncs to the device exactly when a physical block id enters it.**

| cause | entries written |
|---|---|
| admission | whole row, `0:need` |
| prefix hit at admission | the hit entries (shared ids, refcount bumped) |
| fork | the whole row, copied into a new slot |
| decode growth, `num_computed_tokens % block_size == 0` | one |
| copy-on-write | one (replaced) |

**Never dirty:** decode inside a block, finish, abort, preempt, evict. Acquiring a block
requires a sync; releasing never does, because `seqused_k` bounds how far the kernel
reads, so stale tails are harmless. The row is overwritten when the slot is reused — a
write you were doing anyway.

The running example:

| iteration | dirty rows | cause |
|---|---|---|
| 1 | 0 | R1 admitted |
| 2 | — | none: **no block-table copy** |
| 3 | 0 | R1 growth at pos 8 |
| 4 | 1 | R2 admitted, with a prefix hit |
| 5 | 2 | R3 admitted, with a prefix hit |
| 6 | 2, 3 | row 3 written by the fork at the end of 5; row 2 by CoW |
| end of 6 | — | R2 finishes, R3b aborted (§21): releases are never dirty |

### One batched copy per iteration

```python
# scheduling phase — host memory only
for req in scheduled:
    if new ids entered req's row:
        bt_mirror[req.slot, lo:hi] = ids
        dirty_rows.add(req.slot)

# single sync point, after all scheduling, before the forward
if dirty_rows:
    r0, r1 = min(dirty_rows), max(dirty_rows) + 1
    block_table_gpu[r0:r1, :max_blocks].copy_(bt_mirror[r0:r1, :max_blocks], non_blocking=True)
```

Not one copy per `pop()`. **Launch overhead (~5–10 µs) dominates small transfers**, so
the question is "how many copies", not "how many bytes" — hence one copy spanning
`min..max` dirty row even if it drags clean rows along.

At moderate table sizes engines often skip dirty tracking and copy the live rectangle
`[:num_reqs, :max_blocks_in_use]` unconditionally — ~20 µs at 512 KiB, lost in a
multi-millisecond iteration. Bounding the second axis to the blocks actually in use is
the optimisation that is always present. Dirty tracking earns its keep at long context,
where the table is tens of megabytes (Part V-B).

### State after iteration 6

```
block_table                      seqused_k
  row 0  R1   [9, 2, 7, ·]          12
  row 1  R2   [9, 5, 1, ·]          11
  row 2  R3a  [9, 8, ·, ·]           8
  row 3  R3b  [9, 4, ·, ·]           8

pool
  blk 0  free
  blk 1  [b2 u0 u1  · ]   R2                 partial
  blk 2  [s4 s5 a0 t0]    R1                 full, registered
  blk 3  free
  blk 4  [s4 s5 c0 y0]    R3b                full, registered
  blk 5  [s4 s5 b0 b1]    R2                 full, registered
  blk 6  free
  blk 7  [t1 t2 t3 t4]    R1                 full, registered
  blk 8  [s4 s5 c0 x0]    R3a                full, registered
  blk 9  [s0 s1 s2 s3]    R1 R2 R3a R3b      full, registered, refcount 4

free_list  [3, 6, 0]
```

Four sequences, none contiguous, all believing they are. One block serves four of them.
`s4 s5` is stored four times — the granularity loss from §16, made visible. And the pool
has no idea any of this exists; all of it lives in four short integer lists and a
refcount map.

---

# Part IV — Correctness and completeness

## 20. Concurrency — one writer

### Where concurrency lives

Requests never write to the block table. **The scheduler does — and there is exactly
one of it.**

```
many connections            ──┐
  HTTP / gRPC handlers        │  concurrent (async, threads, processes)
  tokenization                │
                            ──┘
                              ▼
              [ input queue: new requests, aborts ]   ← the ONLY shared mutable handoff
                              ▼
              ┌──────────────────────────────────┐
              │   ENGINE CORE LOOP               │  single-threaded
              │   owns: request states,          │
              │         free list, refcounts,    │
              │         prefix hashes,           │
              │         block table (host)       │
              │   schedule → allocate → pack →   │
              │   sync → forward → sample →      │
              │   update                         │
              └──────────────────────────────────┘
                              ▼
              [ output queue: (request_id, new ids, finished) ]
                              ▼
  detokenization, stop-string checks, streaming   ──┐  concurrent again
```

Arrival is concurrent; **scheduling is serial.** A request handler's entire interaction
with engine state is pushing onto the input queue. vLLM V1 runs the engine core in its
own process, talking to the API-server process over IPC, to keep this loop off the
frontend's event loop; GPU workers (one per rank) receive its decisions.

### Ordered, not concurrent

Within one iteration several requests *cause* block-table writes — iteration 6 has a
fork row and a CoW — but all are performed by one thread, in order, in one pass. No lock
is needed because there is no second writer to exclude. Each request occupies its own
slot, so its own row; two requests touch the same *entry* only through sharing, and
that is refcounts plus CoW, resolved in that same serial pass.

Single-writer is not a concession to lock overhead. **Scheduling decisions are
inherently global:** "can I admit this?" depends on total free blocks, the watermark and
every running sequence; "who do I preempt?" requires ranking all of them. Neither is
answerable from inside one request. Serialising is the shape of the problem — the same
reason a token budget spent *across* requests needs someone who sees them all at once.

### The invariant the GPU relies on

```
after the scheduling pass, slot_mapping contains no duplicate entries
```

This is what makes `pool[slot_mapping] = k_new` correct, and what CoW exists to
guarantee. A scatter with duplicate indices is **nondeterministic** — `index_put_`
without `accumulate` makes no promise about which write wins. So a missing CoW is not a
crash or a detected conflict: one sequence silently inherits another's K/V and produces
a plausible-looking wrong token. That is why the check lives in the scheduler, where it
is cheap and exhaustive, rather than anywhere near the kernel.

Generally: **all contention is resolved in serial host code, and the device receives
only conflict-free precomputed addresses.** The GPU does translation, never arbitration.

### Stream ordering

The few orderings that matter are given by enqueueing on one stream:

```
[ block-table patch + arrays ] → [ CoW block copies ] → [ forward ] → [ sampled ids → host ]
```

The copy must land before the scatter writes into the new block, and the old block must
not be overwritten before the copy reads it. Same stream, in that order, gives both.

### Async scheduling

The device → host copy of sampled ids is a sync point: naively, the host idles while
the GPU computes, then the GPU idles while the host schedules and packs. vLLM V1's async
scheduling and SGLang's overlap scheduler prepare iteration *N+1* on the CPU while the
GPU runs *N*.

- **Placeholder tokens.** *N+1* needs tokens that *N* has not produced yet. They are
  packed as placeholders and patched in on the device once sampling lands.
- **Still one host thread.** The concurrency is host vs. GPU, not request vs. request.
  The GPU reads only the buffers shipped for *N*; *N+1*'s patches go out with *N+1*'s
  launch; stream order keeps each copy ahead of the kernels that read it.
- **Aborts and stops lag by a step.** A request that finishes in *N* may already be in
  *N+1*. Its extra sampled token is discarded when it arrives.

### Tensor parallel

The scheduler's decisions are broadcast, so every rank applies the same allocation, the
same CoW and the same slot mapping to its own pool shard. No cross-rank coordination is
needed, because the decision was already made once, on the host (Part V-C).

## 21. Leaving — finish, abort, evict, preempt

### Finish — R2 samples EOS

In iteration 6, R2's sample for position 11 is EOS. In the update step, R2 is retired
and its blocks are released **in reverse order**:

```
block 1  [b2 u0 u1 ·]    refcount 1 → 0    never registered (partial)  → free
block 5  [s4 s5 b0 b1]   refcount 1 → 0    registered                  → cached, evictable
block 9                  refcount 4 → 3
device: nothing — row 1 stays stale until slot 1 is reused
```

The EOS token is never fed back, so its K/V is never computed.

### Abort — R3b's client disconnects

The frontend pushes an abort onto the input queue. The engine core handles it before the
next scheduling pass, through exactly the same release path:

```
block 4  [s4 s5 c0 y0]   refcount 1 → 0    registered → cached
block 9                  refcount 3 → 2
```

With async scheduling a step including R3b may already be in flight; its output is
discarded on arrival. Abort handling matters more than it looks: without it, every
disconnected client keeps generating to `max_tokens` and holding blocks.

State after both:

```
in use    9 (R1, R3a)   2, 7 (R1)   8 (R3a)
cached    5, 4          ← LRU order, oldest first
free      3, 6, 0, 1
```

### Evict — tails first

When the allocator needs a block it pops the front of the LRU queue; popping a cached
block evicts its hash first. **Order matters.** Suppose R1 later finishes and block 9's
refcount also reaches 0. Released in reverse, `7, 2, 9` enter the queue in that order, so
7 — the tail of the chain — is evicted first.

Evicting 9 first would **orphan** 2 and 7: still registered, still occupying memory, but
unreachable, because every lookup chains through `h0` and stops at the first miss.
Releasing in reverse order (vLLM does exactly this) makes LRU evict chains from the tail
inward, so cached prefixes shrink rather than break.

### Preempt — when memory runs out mid-generation

**Preemption is taking memory back from a request that is already running**, so a
different running request can keep going.

It is needed because admission only checks that a request's *prompt* fits. Nothing
reserves room for the tokens it will generate (§10). Every running request grows by one
block every `block_size` tokens, so the pool can fill up while requests are
mid-generation. Admission can refuse a *new* request; it cannot refuse a running one,
which must grow to take its next step. So someone gives up memory.

Take the state after iteration 6, but suppose the pool had **no free or cached
blocks**. R1 is at position 12 — `12 % 4 == 0` — and needs a fourth block.

**Pick a victim** — the most recently admitted running request (or the lowest
priority): R3a.

**Recompute** — the common default, and the only mode in vLLM V1:

```
release R3a in reverse:  block 8  refcount 1 → 0 → cached → popped at once for R1 (hash evicted)
                         block 9  refcount 4 → 3
R3a.status = PREEMPTED;  num_computed_tokens = 0;  block_ids = []
keep prompt_ids and output_ids [x0];  push R3a to the FRONT of WAITING
```

Nothing visible is lost — R3a's emitted tokens are kept. What is lost is the GPU work
that built its KV.

When readmitted, R3a's effective prompt is `s0 … s5 c0 x0` (8 tokens). Lookup hits block
9 — still in use — and misses on L1, whose hash was just evicted, so it recomputes 4
tokens, not 8. **The prefix cache softens recompute:** a preempted request often gets
much of its history back for free.

**Swap** — the alternative: copy the victim's blocks to a CPU swap pool and free them
on the GPU; on resume, allocate blocks and copy back. No recompute, but PCIe traffic both
ways, plus a second pool to size and manage.

This is the same idea as an OS evicting a process's pages under memory pressure, except
the "pages" are usually thrown away and rebuilt rather than swapped.

### Thrashing — preempt, readmit, preempt again

Thrashing is the system spending its time **undoing and redoing work** instead of making
progress. Here it is a loop:

```
iteration 10: pool full, R1 needs a block   → preempt R3, release its 50 blocks
iteration 11: plenty of free blocks         → scheduler readmits R3
              R3 re-prefills 200 tokens, takes 50 blocks again
iteration 12: pool full again, R1 needs one → preempt R3 again
iteration 13: readmit R3, re-prefill 200 tokens again
…
```

Each round burns a full prefill of R3's history and produces almost nothing new.
Throughput collapses while the GPU looks fully busy.

The root cause: **the combined memory demand of the running set exceeds the pool**, and
the scheduler readmits work as soon as a little space frees up. It is OS thrashing
exactly — the combined working set of running processes exceeds RAM, so pages keep
getting evicted and reloaded.

Three defences:

- **The watermark.** Admit only when the free list stays above a margin *after*
  admission: `len(free) − need ≥ watermark`. That reserves growth headroom for running
  requests, so freshly preempted work is not readmitted into a pool about to refill.
- **Victim choice.** Preempting the *newest* request evicts the one with the least
  invested work, so each recompute is cheap. Preempting the oldest throws away the most.
- **Limiting concurrency.** A lower `max_num_seqs` means fewer requests growing at once,
  so their combined growth is less likely to overrun the pool.

### Thrashing is possible; livelock is not

Termination is guaranteed by checks made long before. Validation (§8) ensures every
request fits in `max_model_len`; the capacity check (§2) ensures `max_model_len` fits in
the pool. So once everyone else has been preempted, the oldest running request can
always run to completion alone. Under thrashing the system is slow, but it always makes
progress. The watermark exists to keep it from getting that slow.

---

# The two indirections

```
requests ──cu_seqlens_q──▶ flat token row ──▶ GPU tiles
                                                 │
sequence ──block_table──▶ physical blocks ───────┘
```

**`cu_seqlens_q`** dissolves the prefill/decode distinction. A batch stops being a
rectangle and becomes a token stream with boundaries, so "510 tokens", "1 token" and "5
tokens after a prefix hit" are the same kind of thing.

**`block_table`** dissolves the coupling between concurrency and context length. KV is
allocated by the block instead of by worst-case sequence width.

And the control split every part of this document reduces to: **the host owns
allocation and write-address resolution; the device owns read translation.**

---

# Part V — Reference

## A. Attention kernel internals

### The block walk

```
grid = (query_tiles, heads[, kv_splits])

each CTA:
  1. which sequence owns my tile?             ← from cu_seqlens_q
  2. load query tile → SRAM
  3. read seqused_k and the block_table row
  4. for logical = 0 … kv_len / block_size:
         physical = block_table[seq, logical]  ← translation, on device
         load kv_pool[physical] → SRAM         ← one block
         QKᵀ, mask by absolute position,
         online-softmax accumulate, then ·V
  5. write output rows at their flat positions
```

KV is **never gathered** into contiguous memory and the `q_len × kv_len` attention
matrix is **never materialised**.

### Online softmax

Each block yields a *partial* softmax, corrected retroactively. Running state per
query row:

```
m  running max      (scalar)
l  running sum      (scalar)
o  running output   (head_dim)
```

On each block:

```
m_new      = max(m, s.max())
correction = exp(m - m_new)          # shrink everything accumulated so far
l          = l * correction + Σ exp(s - m_new)
o          = o * correction + exp(s - m_new) @ V_block
```

then divide once at the end: `out = o / l`.

Exact, not approximate — `softmax(x)ᵢ = exp(xᵢ − m) / Σ exp(xⱼ − m)` holds for any
`m`, so changing `m` rescales numerator and denominator identically. The max is
there for numerical stability.

State is `O(head_dim)` per query row, **independent of context length**. That is
what lets a 130k-token context be attended to from a few dozen KB of SRAM.

### Split-K (FlashDecoding)

The merge is associative and commutative, so KV ranges can be processed in
parallel and combined:

```
merge((m₁,l₁,o₁), (m₂,l₂,o₂)):
    m  = max(m₁, m₂)
    c₁ = exp(m₁ - m);  c₂ = exp(m₂ - m)
    l  = l₁c₁ + l₂c₂
    o  = o₁c₁ + o₂c₂
```

Needed because decode has `q_len = 1`:

```
8 seqs × 12 heads             =  96 CTAs  on a 108-SM GPU → one thin wave
8 seqs × 12 heads × 8 splits  = 768 CTAs  → full occupancy
```

Prefill already has plenty of query tiles and does not need it. This is vLLM's
`paged_attention_v1` (single pass) versus `v2` (split-K with reduction).

#### What "KV split" means

Partitioning the **key/value sequence axis** — the history — into contiguous ranges,
one per CTA. Every CTA uses the *same* query vector and the *same* head; only the key
range differs.

```
one (sequence, head), kv_len = 1024, q_len = 1, num_splits = 4

   ┌───────────┬───────────┬───────────┬───────────┐
   │ k₀…k₂₅₅   │ k₂₅₆…k₅₁₁ │ k₅₁₂…k₇₆₇ │ k₇₆₈…k₁₀₂₃│
   └───────────┴───────────┴───────────┴───────────┘
      CTA 0        CTA 1       CTA 2       CTA 3
     (m,l,o)₀    (m,l,o)₁    (m,l,o)₂    (m,l,o)₃
           └──────────┴─── merge ───┴──────────┘
```

It is the odd one out among the four parallel axes:

```
batch split   →  different sequences  →  disjoint output rows
head split    →  different heads      →  disjoint output columns
query split   →  different queries    →  disjoint output rows
KV split      →  SAME output row      →  requires a reduction
```

The first three partition the *output*, so workers never collide. KV split partitions
the *summation producing one output row*, which is the entire reason a second pass and
the `(m, l, o)` merge exist.

It is also easy in decode specifically: a single query at position *p* sees keys
`0…p`, so every split is fully visible. In prefill, splitting KV would leave some
splits entirely masked for some query tiles — wasted work — another reason it is a
decode-phase technique.

### FlashAttention vs FlashDecoding

Same algorithm, different bottleneck. FlashDecoding is not a replacement; it is
FlashAttention with one more parallelisation axis, added because decode breaks the
original's assumptions.

**FlashAttention** attacks memory traffic. Standard attention materialises the `T × T`
score matrix in HBM — at `T = 4096` that is 16M floats per head per layer, and
attention is memory-bound. Tiling plus online softmax drops HBM traffic from `O(T²)`
to roughly `O(T²·d / SRAM)`. It parallelises over batch, heads and **query tiles**,
which works because training and prefill have many query rows.

**FlashDecoding** attacks occupancy. At `q_len = 1` the query-tile axis vanishes:

```
prefill:  4 seqs × 12 heads × 32 query tiles = 1536 CTAs   saturated
decode:   4 seqs × 12 heads ×  1             =   48 CTAs   on 108 SMs — idle
```

and each of those 48 then serially walks a huge history. Splitting KV restores
parallelism.

| | FlashAttention | FlashDecoding |
|---|---|---|
| Target phase | training, prefill | decode |
| `q_len` | large | 1 (a few with speculation) |
| Bottleneck | HBM traffic for `T×T` | GPU occupancy |
| Parallel over | batch, heads, **query tiles** | batch, heads, **KV splits** |
| Passes | 1 | 2 (partials, then reduce) |
| Extra cost | — | partial `(m,l,o)` in HBM |
| Exact? | yes | yes |

In a mixed batch both appear in one launch, chosen per request by shape: a prefill
chunk with `q_len = 510` needs no split, while `q_len = 1` decodes get split hard.

Two things worth keeping straight:

- **Neither approximates anything.** Both match naive attention up to float rounding.
  The "flash" is about memory movement and scheduling, not dropped terms.
- **Paged KV is orthogonal.** Either kernel can run over contiguous KV or over a block
  table; the block walk simply replaces contiguous striding in the inner loop.
  (*FlashDecoding++* is a separate follow-up that fixes a global max ahead of time to
  avoid the rescaling synchronisation — easily confused with the above.)

Two consequences of order-independence:

- Scattered physical blocks cost nothing — visit order is free.
- Results are **not bitwise reproducible**. `num_splits` varies with batch size, so
  the same prompt can take a different reduction path and flip a sampled token.
  That is the origin of batch-size-dependent nondeterminism.

What *is* fixed: partials may only merge within the same `(sequence, head)`, and
causal masking depends on each key's absolute position
(`logical_block × block_size + offset`) — a property of the key, applied before it
enters the accumulator, independent of visit order.

### Implementing a fused decode kernel

Two routes from classic (§13) to fused. Both need an NVIDIA GPU — FlashAttention and
Triton do not run on macOS.

#### Route 1 — a library kernel: `flash_attn_with_kvcache`

FlashAttention ships the fused kernel. Pass the new `k, v` and it writes them into the
cache in place **and** attends, in one launch:

```python
from flash_attn import flash_attn_with_kvcache

out = flash_attn_with_kvcache(
    q,                          # (B, T_new, H, D)
    k_cache, v_cache,           # (B, W, H, D)   ← T before H
    k=k_new, v=v_new,           # (B, T_new, H, D) — appended in place
    cache_seqlens=pos.int(),    # (B,) per-row write position
    causal=True,
)                               # → (B, T_new, H, D)
```

`cache_seqlens` is one value **per row**, so each row appends at its own position —
the same per-row-pointer model as `RowKVCache.pos`. The kernel writes the cache but
does not advance the pointers; the caller still does. It needs fp16/bf16 on Ampere or
newer. It also accepts `block_table` for a paged cache, but in FlashAttention-2 the
page size must be a multiple of 256 — far larger than the 16 used here.

The catch is **layout**: it wants `(B, W, H, D)`, while this repo's cache is
`(B, H, W, D)`. Adopting it means storing T before H and dropping the transposes:

```python
# row_cache.py — store T before H
shape = (batch_size, W, config.n_head, config.n_embd // config.n_head)

# gpt.py decode path — q/k/v stay (B, T, H, D)
q = q.view(B, T, n_head, hs)
k = k.view(B, T, n_head, hs)
v = v.view(B, T, n_head, hs)
y = flash_attn_with_kvcache(q, cache.key[i], cache.value[i], k=k, v=v,
                            cache_seqlens=cache.pos.int(), causal=True)
y = y.reshape(B, T, C)        # already (B, T, H, D) — no transpose back
```

#### Route 2 — write it in Triton

The route that makes fusion concrete. A decode-only kernel — one new token per row —
that keeps the `(B, H, W, D)` layout. One program instance per (row, head):

```python
import triton
import triton.language as tl

@triton.jit
def fused_decode_kernel(
    Q, Knew, Vnew, Kc, Vc, Pos, Out,
    s_qb, s_qh,                  # q / k_new / v_new strides: (B, H, 1, D)
    s_cb, s_ch, s_ct,            # cache strides: (B, H, W, D)
    sm_scale,
    D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    pos = tl.load(Pos + b)

    # load this token's q, k, v into registers
    qkv = b * s_qb + h * s_qh + d
    q  = tl.load(Q + qkv).to(tl.float32)
    kn = tl.load(Knew + qkv)
    vn = tl.load(Vnew + qkv)

    # 1. WRITE — append to the cache at column pos
    base = b * s_cb + h * s_ch
    tl.store(Kc + base + pos * s_ct + d, kn)
    tl.store(Vc + base + pos * s_ct + d, vn)

    # 2. ATTEND over history 0 … pos-1 with online softmax
    m   = tl.full([], float('-inf'), tl.float32)
    l   = tl.full([], 0.0, tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, pos, BLOCK_T):
        t = start + tl.arange(0, BLOCK_T)
        valid = t < pos
        k = tl.load(Kc + base + t[:, None] * s_ct + d[None, :],
                    mask=valid[:, None], other=0.).to(tl.float32)
        v = tl.load(Vc + base + t[:, None] * s_ct + d[None, :],
                    mask=valid[:, None], other=0.).to(tl.float32)
        s = tl.where(valid, tl.sum(k * q[None, :], axis=1) * sm_scale, float('-inf'))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        corr, p = tl.exp(m - m_new), tl.exp(s - m_new)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        l   = l * corr + tl.sum(p, axis=0)
        m   = m_new

    # 3. the new token itself — from REGISTERS, never re-read from the cache
    s_new = tl.sum(kn.to(tl.float32) * q) * sm_scale
    m_new = tl.maximum(m, s_new)
    corr, p_new = tl.exp(m - m_new), tl.exp(s_new - m_new)
    acc = acc * corr + p_new * vn.to(tl.float32)
    l   = l * corr + p_new

    tl.store(Out + qkv, (acc / l).to(Out.dtype.element_ty))


def fused_decode(q, k_new, v_new, cache_k, cache_v, pos):
    B, H, _, D = q.shape
    out = torch.empty_like(q)
    fused_decode_kernel[(B, H)](
        q, k_new, v_new, cache_k, cache_v, pos, out,
        q.stride(0), q.stride(1),
        cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
        D ** -0.5, D=D, BLOCK_T=64)
    return out
```

Three things to notice:

- **Step 3 is where fusion pays off.** The loop reads only `t < pos` — history. The new
  token's contribution comes from `kn, vn`, still in registers from the load at the top.
  Two separate kernels cannot do this.
- **Step 3 is the `(m, l, o)` merge** from §13: history in one piece, the new token in
  the other, combined by rescaling. A fused kernel is the separate-new-K/V maths with the
  write added in the same program.
- **No read-after-write hazard.** The program stores column `pos` but never loads it,
  so when the store lands does not matter.

It replaces both steps of the decode path:

```python
# classic: write, then attend — two kernels
k, v = kv_cache.update(layer_idx, k, v)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

# fused: one kernel
y = fused_decode(q, k, v, kv_cache.key[layer_idx], kv_cache.value[layer_idx], kv_cache.pos)
```

`q, k, v` are slices of one `c_attn` output, so they share strides — the kernel relies
on that. Two gaps against a full decode path:

- **Inactive rows.** A free row would still have column `pos` written. Harmless — the
  next prefill overwrites from column 0 and the output is discarded — but an `active`
  flag with an early return is cleaner.
- **Prefill stays classic.** This kernel handles one token per row. For prefill, launch
  overhead is negligible next to the compute, so there is nothing to gain.

#### Verify against classic

```python
y_ref = classic_decode(q, k, v, cache_ref, pos)        # update() + SDPA
y_fus = fused_decode(q, k, v, cache_k, cache_v, pos)
torch.testing.assert_close(y_fus, y_ref, atol=2e-3, rtol=2e-3)   # fp16 tolerance
torch.testing.assert_close(cache_k, cache_ref.key[layer_idx])     # the write landed too
```

This is a teaching kernel, not a tuned one: one program per (row, head) and no split-K,
so at small batch it under-occupies the GPU for exactly the reason FlashDecoding exists
(above). Benchmark against classic before assuming a win — with CUDA graphs already
removing most launch overhead, expect a modest gain, not a dramatic one.

## B. Scaling

### The dense block table reserves an unreachable rectangle

Both dimensions of `(max_num_seqs, max_blocks_per_seq)` are worst cases that cannot be
worst cases at the same time. With `max_num_seqs = 256`, `max_model_len = 1.2M`,
`block_size = 16`:

```
table entries = 256 × 75,000 = 19,200,000 int32   (76.8 MB)

80 GB pool / 36 KiB per token = 2.17M token slots = 135,600 physical blocks
```

Of 19.2M entries, **at most 135,600 can simultaneously hold a valid id — 0.7%.** The
reachable region is a hyperbola, `num_seqs × blocks_per_seq ≤ 135,600`, and the two
configured maxima sit at opposite ends of it:

```
256 sequences busy  →  each averages 530 blocks  = 8,480 tokens
one seq at 1.2M     →  75,000 blocks = 55% of the entire pool → at most ~1.8 of them
```

This is the reserve-versus-utilise criticism from §10, one level up:

```
data:   RowKVCache  W=1024/row, 79 used    →  paging fixed this
index:  block_table 75,000/row, ~30 used   →  still reserved per row
```

Paging fixed reserve-per-row for the data and left the pattern in place for the index.
Entries are 1000× cheaper than blocks, so it took a million-token context to notice.

**The dense rectangle buys O(1) in-place append.** Crossing a boundary writes one int32
at `[seq, logical]` — no reallocation, no compaction. A packed ragged layout (flat id
array plus per-sequence offsets) reserves nothing, but appending to sequence 3 would
shift sequences 4…n. The waste is the price of appendability.

**Where it bites, and where it does not.** 76.8 MB of resident table costs ~130 blocks
of pool, 0.1% — not a problem. What is not fine is **bandwidth**: a naive whole-table
copy per iteration is ~3 ms over PCIe, comparable to the forward itself. Kernel-side
there is no cost, since the walk is bounded by `ceil(seqused_k / block_size)`.

### Three fixes, cheapest first

| | index saving | cost | when |
|---|---|---|---|
| clamp to `num_blocks` | 4.1× | none | always |
| raise `block_size` | 8× | `≤ bs−1` slots/seq + coarser prefix cache | long sequences only |
| hierarchical table | removes the scaling | kernel indirection + complexity | when 1+2 still do not fit |

**Clamp** (§4). With a 10 GiB pool, `min(75,000, 18,204)` → 18.6 MB instead of 76.8 MB.
Zero cost; it helps only when the declared context outruns the pool. Do it always.

**Raise `block_size`.** Divides the axis, and composes with the clamp:

```
block_size  16  →  75,000 entries/seq  →  76.8 MB     num_blocks 18,204
block_size 128  →   9,375 entries/seq  →   9.6 MB     num_blocks  2,275
            128 + clamp → 256 × 2,275 × 4 = 2.3 MB    33× smaller than naive
```

But it is **not an index-memory optimisation.** The fragmentation cost:

```
block_size  16:  256 × ≤15  slots × 36 KiB =  142 MB   (1.3% of a 10 GiB pool)
block_size 128:  256 × ≤127 slots × 36 KiB = 1.20 GB   (11%)
```

Saving 16 MB of index to pay up to 1.06 GB of fragmentation is a ~65× losing trade *if
sequences are short*. Raise it for other reasons, in the matching workload:

- **long sequences** — 127 wasted slots out of 1.2M is 0.01%; fragmentation vanishes
  and fewer kernel lookups, longer contiguous DRAM reads and smaller H2D copies come
  free.
- **short sequences** — fragmentation and prefix-cache granularity (§16) both dominate;
  stay at 16.

16 is the common default; 32–128 appear in long-context configs, and kernels typically
cap it (128 is a usual ceiling).

**Hierarchical table.** A small outer array per sequence pointing at *index blocks*
drawn from their own pool, allocated on demand — the index itself paged, so its memory
scales with occupancy instead of `max_num_seqs × max_model_len`. Costs one more
dependent load in the kernel inner loop and real custom-kernel work. This is the fix
operating systems reached for, for the same reason: flat page tables scale with the
address space while occupancy scales with physical memory. Neither vLLM nor SGLang ships
one (Part V-D).

### Long context — what actually binds

At 1.2M tokens in this config:

```
index:  256 × 75,000 × 4  =   77 MB     ← block_size fixes this
data:   1.2M × 36 KiB     = 44.2 GB     ← for ONE sequence; block_size does nothing
```

A 10 GiB pool holds 291,264 token slots — about **24% of a single** 1.2M-token
sequence. Declaring `max_model_len = 1.2M` costs 77 MB of table and **reserves no KV
at all**: it is an admissibility ceiling, not an allocation — and the capacity check
(§2) would reject it against this pool anyway.

What makes million-token contexts servable is bytes per token (§1), not the index: GQA
plus fp8 KV takes 44.2 GB to ~3.7 GB. Index tuning and data tuning are separate knobs.

GPT-2's `wpe` is a learned absolute table of 1024 rows, so this architecture has a hard
ceiling at 1024 positions — there is no position 1,200,000 to embed. Long context needs
RoPE or ALiBi first, where position is a rotation applied at attention time rather than
a lookup.

## C. Scope — one replica

The unit is the **model replica** (one TP×PP group), not the host. Within a replica
there is exactly one logical block table, replicated to every rank.

```
            ONE scheduler, ONE block table, ONE free list  (host)
                              │  broadcast
        ┌─────────────────────┼─────────────────────┐
     rank 0                rank 1                rank 2
  block_table (mirror)  block_table (mirror)  block_table (mirror)   ← identical
  pool shard            pool shard            pool shard             ← different payload
```

What is sharded is the **payload**, never the addressing:

- **Tensor parallel** splits the KV-head axis: each rank's pool is
  `(2, num_blocks, block_size, n_kv_head/tp, head_dim)`. Block 42 exists on every rank;
  each holds a different slice of its heads.
- **Pipeline parallel** splits layers: rank 0 holds block 42's layers 0–5, rank 1 its
  layers 6–11.

The ids **must** be identical. Every rank attends over the same logical history for
the same sequence; if rank 0 believed logical block 6 was physical 88 while rank 1
believed 91, the shards would attend to different pasts and their partial outputs would
not combine. So `num_blocks` is agreed once (the minimum over ranks) and decisions are
made in one place and broadcast. A replica can span hosts, so "per host" is not the
right boundary.

**Across replicas, nothing is shared** — separate pools, free lists and namespaces.
The consequence that bites: **the prefix cache is per replica.** A request routed to a
different replica than the one that cached its prefix loses the hit. Hence
**prefix-aware routing** — hash the prompt prefix and steer matching requests to the
replica holding it. Load balance and cache locality pull against each other, and that
tradeoff is the router's whole job.

Sharing KV across replicas needs a separate layer, with its own addressing:

- **External KV store / offload** — LMCache, Mooncake, vLLM's KV connector. A shared
  tier keyed by prefix hash; a replica pulls blocks in and allocates its *own* local
  blocks to receive them.
- **Disaggregated prefill/decode** — prefill on machine P, decode on D, KV shipped over
  RDMA. Independent pools and tables on each side; the transfer includes an explicit id
  translation. DistServe, Splitwise and Mooncake are the references; this is a
  distributed-systems project rather than a scheduler change.

## D. What vLLM and SGLang actually do

Neither ships a hierarchical block table — both use a **flat, dense** one.

**vLLM** is dense, as described here: a `(max_num_seqs, max_blocks_per_seq)` int32
tensor on device, mirrored by a host numpy array the scheduler patches and copies
forward each iteration. `block_size` defaults to 16. V1 has no CoW (§17) and preempts
by recompute only (§21).

**SGLang** goes the opposite direction. Its historical design is **token-granular**: a
`req_to_token` pool of shape `(max_num_reqs, max_context_len)` mapping
(request, position) → a flat KV slot. Effectively `page_size = 1`, so no blocks at all,
and an index table `block_size` times larger:

```
256 reqs, 1.2M context, int32
  vLLM, block_size 16:  256 ×    75,000 × 4 =   76.8 MB
  SGLang, page_size 1:  256 × 1,200,000 × 4 =    1.23 GB
```

What that buys: **zero internal fragmentation** and token-granular prefix sharing, which
lets RadixAttention's tree match at token resolution instead of needing a full block of
identical tokens — §16 walks through the tree step by step. The cost is the index blowup — which is why larger page sizes were
added later. SGLang converged *toward* blocks from the other side, not toward hierarchy.

**Why flat survives in both:** resident index memory was never the binding constraint.
What they engineer around is the transfer and the Python overhead — bounding the copied
rectangle to the live batch, async/overlap scheduling, and larger block/page sizes for
long context.

**One naming trap.** "Hierarchical" in these projects means something else:

- **SGLang HiCache / HiRadixCache** — multi-tier KV *storage*, GPU → CPU → disk.
  Tiering the data, not the page table.
- **Radix tree / prefix-cache hash** — a *sharing* index over block or token content.
  Tree-shaped, but it answers "has anyone computed this prefix?", not "where does
  logical block 6 live?"

## E. Observability

The knobs in this document — budget, `block_size`, watermark, `max_num_seqs`,
`gpu_memory_utilization`, routing — are tuned against a handful of metrics:

| metric | what it tells you | knob it tunes |
|---|---|---|
| TTFT — time to first token | queueing + prefill latency | prefill priority, long-prefill cap, budget |
| TPOT / ITL — time per output token | decode latency under load | decode-first policy, `max_num_seqs` |
| KV utilisation | how full the pool runs | `gpu_memory_utilization`, `block_size` |
| prefix hit rate | how much prefill is skipped | `block_size`, prefix-aware routing |
| preemptions per second | memory thrash | watermark, `max_num_seqs` |
| waiting-queue depth | overload | backpressure limits, scale-out |
| tokens per iteration | how full each forward is | budget, CUDA-graph buckets |

## F. Out of scope

Each of these extends the design above rather than replacing it:

- **Speculative decoding** — decode steps with `q_len > 1` (draft tokens verified in one
  forward). Slots written for rejected tokens must be rolled back: `num_computed_tokens`
  goes backwards, and a trailing block allocated only for rejected tokens is released.
- **Multi-LoRA** — per-row adapter ids in the batch; the adapter id joins the prefix
  hash.
- **Structured output** — grammar-derived logit masks applied per row before sampling.
- **Sliding-window and hybrid models** — blocks that fall out of the window are released
  mid-sequence; different layer groups may need different block tables.
- **fp8 KV** — quantisation scales stored alongside the pool.
- **KV offload tiers and multimodal encoder caches** — more pools, more addressing.

## G. Mapping to `src/gpt2/serving`

| This repo | Engine equivalent |
|---|---|
| `queue` | `WAITING` queue |
| `active` bool `(B,)` | `RUNNING` set |
| `_admit` — atomic prefill, one `(1, L)` forward per request | prefill scheduling, but chunked and mixed into the same forward as decode |
| `_decode` — `(B, 1)` rectangle | one slice of a ragged mixed batch |
| `slot_req[row]` | sequence → block-table row |
| `RowKVCache.pos` `(B,)` | `seqused_k`, plus `block_table` for addressing |
| `_mask()` | `cu_seqlens_q` / `seqused_k` — geometry, not a tensor |
| fixed `B` slots of width `W` | `max_num_seqs` **and** `max_num_batched_tokens`, two independent limits |
| `_retire` — eot, `max_new_tokens`, cache full | stop checks + release, plus preemption |
| `sampler` lambda over the whole batch | per-row sampling parameters |
| — | free list, refcounts, prefix cache, CoW, abort, streaming: no equivalent |

A row in `RowKVCache` **is** a block — the degenerate case where `block_size == W`,
one block per sequence, and the block table is the identity map. Paging makes that
mapping explicit and lets it be many-to-one:

```python
# this repo — identity, sequence IS the row
self.key[layer][row, :, pos]

# paged — one indirection, sequence appears nowhere in the tensor
kv_pool[layer][block_table[seq][pos // 16], pos % 16]
```

That identity is also why `seq` and `physical` are so easy to conflate: here they are
genuinely the same number, both "which request" and "where in memory". Paging is
precisely the change that pulls them apart.

The design already gets the hard parts right: per-row pointers, slot reuse, and a
mask *derived* from state rather than maintained alongside it.

### How this repo's cache is updated

Per layer, inside `CausalSelfAttention.forward` ([gpt.py:32](../src/gpt2/gpt.py#L32)):

```
x                      (B, T, C)
  │ c_attn + split, view + transpose
q, k, v                (B, n_head, T, head_size)
  │
  ├── k, v = kv_cache.update(layer_idx, k, v, row=cache_row)   ← gpt.py:43
  │        the returned k, v REPLACE the local ones
  │
  └── F.scaled_dot_product_attention(q, k, v, attn_mask=...)
```

This is the **classic** pattern from §13: `update()` is kernel 1 (the write),
SDPA is kernel 2 (the read of the whole cache). The "flash attention" comment at
[gpt.py:52](../src/gpt2/gpt.py#L52) refers to SDPA fusing mask, softmax and dropout
*within attention* — it does not fuse the cache write into attention. Those are two
different senses of "fused". Part V-A shows how to fuse them.

`update()` is **write, then return everything**:

| | `k` in | written to | `k` out |
|---|---|---|---|
| prefill | `(1, 12, 47, 64)` | `[row, :, 0:47]` | `(1, 12, 1024, 64)` — full-width view |
| decode | `(4, 12, 1, 64)` | `[rows, :, pos]`, one column each | `(4, 12, 1024, 64)` |

The layer computes k/v for the *new* tokens only and gets back the *whole history* to
attend over. `q` is never cached, so decode attention is lopsided — `(4,12,1,64)`
queries against `(4,12,1024,64)` keys, with the mask hiding unwritten columns. That
mask is load-bearing, not an optimisation: the buffers are zero-filled, so without it
attention would average over a thousand columns of zeros. Hence the guard at
[gpt.py:132](../src/gpt2/gpt.py#L132).

`advance()` runs **once per forward**, after all 12 layers — twelve `update()` calls
all write at the same `pos`, and one `advance()` moves it afterwards. The paged
equivalent is that `slot_mapping` has no layer term: the host does the position
arithmetic once up front instead of advancing a pointer after.

Two properties of the dense design:

- **Writes are scattered per row, reads are whole-buffer.** Each row writes one column
  at its own `pos`, but attention reads all `W` columns for every row and lets the mask
  sort it out. You touch 1024 columns to use 134. Paged attention exists precisely to
  read only the blocks holding real tokens.
- **`update()` is the only mutation during a forward.** `advance()` moves pointers,
  `release()` resets them, `reset()` clears — but no k/v is written anywhere else,
  which is why `assert pos[row] == 0` in `_prefill` is sufficient to rule out
  double-writes.

The two write paths collapse into one under paging:

```python
# this repo — prefill: contiguous block in one row
self.key[layer][sl, :, :T] = key                     # row_cache.py:65
# this repo — decode: one column per row
self.key[layer][rows, :, self.pos] = key[:,:,0,:]    # row_cache.py:74

# paged — one flat index per token, both cases
pool[slot_mapping] = k_new
```

Two gaps worth naming before starting the work:

**Chunked prefill is structurally unrepresentable today.** A second chunk is `q=3` at
`pos=510` — `_prefill` rejects it (`assert pos[row] == 0`) and `_decode` rejects it
(`assert key.size(2) == 1`). No path accepts "many tokens into a row that already has
content", which is also exactly the shape of a prefix-cache hit (§16). What is needed is
per-row write *spans*, `pos[r] : pos[r]+T`, plus a mask causal *within* the chunk.
[gpt.py:132](../src/gpt2/gpt.py#L132) already adds a `tril` when `T > 1`, which is the
mask half of it.

**`release()` zeroes 24 buffers per retire**, proportional to `W`, on the hot path at
high turnover. Only `pos[row] = 0` is needed for correctness — `_mask()` already blanks
everything at or past `pos`. The zeroing is defence-in-depth. Under paging the
equivalent is a refcount decrement, because `seqused_k` makes clearing unnecessary.

### Suggested order of work

1. **Paging** — block pool, free list, block table, refcounts. The allocator is ~80
   lines. Making attention *read* that layout without a gather is the part that needs a
   custom CUDA kernel; a PyTorch gather-then-attend version will be correct and slow.
2. **Chunked prefill** — requires leaving the rectangle: per-row write spans and a mask
   causal *within* the chunk. It also unlocks step 3, since a prefix hit is a chunk that
   starts mid-sequence.
3. **Prefix caching** — chained block hashes, registration on full, the cached state,
   tail-first LRU eviction. SGLang's RadixAttention is this plus a radix tree and a
   scheduling policy that orders requests by longest prefix match to maximise hits.
4. **Leaving properly** — abort, stop strings with incremental detokenization, and
   recompute preemption with the capacity check that guarantees it terminates.

Copy-on-write only becomes necessary for `n > 1` sampling or beam search, and vLLM V1
shows it can be avoided even then.

---

## Caveat

vLLM and SGLang move quickly. The architectural shapes here are stable, but exact
tensor layouts, class names, defaults and policies drift between releases — the V0 →
V1 removal of CoW and swap is one example. Read vLLM's V1 scheduler, KV-cache manager
and block-pool code, and SGLang's memory-pool and `scheduler.py`, directly before
copying any specific design. Notes written September 2026, restructured October 2026.
