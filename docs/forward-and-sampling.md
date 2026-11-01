# Forward and sampling — one iteration through the GPU

[request-lifecycle.md](request-lifecycle.md) follows a request across many iterations:
scheduling, block allocation, prefix caching, preemption. This document zooms into
**one iteration** — from the moment the CPU launches the forward to the moment the
sampled tokens are back on the CPU — and explains what the hardware, PyTorch and the
kernels each do along the way.

It uses one batch throughout: **iteration 4 of the running example** in
request-lifecycle.md (§16). R1 decodes position 9; R2 computes positions 4–8 after a
prefix hit on R1's first block.

```
flat batch:   index  0  |  1   2   3   4   5
              token  t2 |  s4  s5  b0  b1  b2
              req    R1 |  R2  R2  R2  R2  R2
              pos    9  |  4   5   6   7   8

block_table:  R1 → [9, 2, 7]     R2 → [9, 5, 1]       block_size = 4
KV pool, per layer: (num_blocks, block_size, n_head, head_dim) = (…, 4, 12, 64)
```

| Step | Covers |
|---|---|
| 0 — Foundations | the hardware, CTAs and warps, who owns which memory, tensors vs addresses |
| 1 — Launching work | enqueueing, PyTorch / Triton / CUDA C++, CUDA graphs, why `int()` syncs |
| 2 — Inputs reach the GPU | copying the step's data into fixed buffers |
| 3 — The layer: project and write | the transformer block, self-attention, q/k/v, the KV write |
| 4 — Read and attend | rows and tiles, the grid, what a CTA loads, online softmax, FA / FD |
| 5 — Sample | logits for the right rows, sampling on the GPU |
| 6 — Back to the CPU | the one sync, why it feels slow, async scheduling |
| Appendix A | the model's weights: per layer or shared, how they are stored |
| Appendix B | writing your own Triton kernels |

Sizes use this repo's 124M GPT-2: 12 layers, 12 heads, head_dim 64, `n_embd` 768.

---

# Step 0 — Foundations

## 0.1 The hardware

```
CPU (host)                              GPU (device)
  runs Python, the scheduler,             HBM (global memory) ~40–80 GB   ← large, relatively slow
  allocates memory, launches kernels      ├── weights, KV pool, activations, input buffers
                                          SMs (~108 on an A100, 132 on an H100)
  CPU memory ⇄ PCIe ⇄                     └── each SM: SRAM ~100–230 KB (fast) + registers (fastest)
```

| term | meaning |
|---|---|
| **kernel** | a function that runs on the GPU |
| **SM** (streaming multiprocessor) | one of the GPU's ~100 processors; runs CTAs |
| **thread** | one tiny worker executing the kernel |
| **warp** | 32 threads that execute in lockstep — the GPU's real unit of execution |
| **CTA** (thread block) | a group of threads (e.g. 256 = 8 warps) on **one SM**, sharing its SRAM |
| **grid** | all CTAs of one launch; the launcher chooses its shape |
| **stream** | the GPU's work queue; kernels run in the order they were enqueued |
| **tile** | a chunk of a matrix that one CTA loads into SRAM and works on |

Two rules drive everything below:

1. **HBM is slow, SRAM is fast.** Load a tile once, do as much maths on it as possible,
   write back as little as possible.
2. **The CPU plans, the GPU executes.** Python, memory allocation, shapes and launches
   happen on the CPU; the numbers live, and are computed, on the GPU.

## 0.2 CTAs and warps

A CTA is a group of threads, and the hardware runs it as **warps of 32**:

```
CTA (256 threads)
├── warp 0:  threads   0 –  31    ┐
├── warp 1:  threads  32 –  63    │ all on the SAME SM,
├── …                             │ sharing that SM's SRAM,
└── warp 7:  threads 224 – 255    ┘ able to wait for each other (a barrier)
```

| | threads | warps |
|---|---|---|
| allowed range | 32 – 1024 | 1 – 32 |
| **typical** | **128 – 256** | **4 – 8** |
| Triton default (`num_warps=4`) | 128 | 4 |

**More warps hide memory latency.** An HBM load takes hundreds of cycles; while one warp
waits, the SM switches to another that is ready, at no cost. With 1–2 warps the SM
mostly idles.

**But an SM's resources are shared by every CTA on it** — roughly 2048 threads, 32 CTAs,
65,536 registers and ~164 KB (A100) / ~228 KB (H100) of SRAM. Whichever runs out first
sets how many CTAs fit at once (the *occupancy*):

```
simple add kernel: 256 threads, 32 registers each, no SRAM
  by threads:    2048 / 256          = 8 CTAs per SM
  by registers:  65536 / (256 × 32)  = 8 CTAs per SM          → 64 warps active: full

attention kernel: 128 threads, ~80 KB of SRAM for the q_tile + a K/V block
  by SRAM:       164 KB / 80 KB      = 2 CTAs per SM          → only 8 warps active
```

Attention kernels are limited by SRAM, not threads — they need large tiles in SRAM for
reuse. On newer GPUs, warps within one CTA can take **different roles**: FlashAttention-3
on H100 has some warps only loading the next K/V block while others compute on the
current one.

## 0.3 Who owns which memory

| memory | who creates it | when | lifetime |
|---|---|---|---|
| **HBM**: weights, KV pool, buffers, outputs | **PyTorch** (`torch.empty`, `.to("cuda")`) | startup, or before a launch | until freed |
| **SRAM and registers**: tiles inside a kernel | **the compiler** sizes them; the hardware reserves them when a CTA starts | each CTA | until the CTA finishes |

- **Kernels never allocate HBM.** They receive pointers to memory PyTorch already created
  — which is why outputs are allocated *before* a launch (`out = torch.empty_like(x)`).
  `tl.load` and `tl.store` **read and write** existing memory; they create nothing.
- **On-chip memory is never allocated in code.** Triton works out each CTA's SRAM and
  register needs at compile time from the tile sizes — hence tile sizes must be
  compile-time constants (`tl.constexpr`). Too large, and compile or launch fails with
  "out of resources: shared memory". Nothing in SRAM survives a launch; results must be
  stored back to HBM.
- PyTorch allocates through a **caching allocator**: it takes large chunks from CUDA once
  and reuses freed pieces, rather than calling CUDA on every allocation.

**Where tensors live.** A tensor has two parts: the **object** (shape, dtype, strides, a
pointer) is a Python object in CPU memory; the **data** lives on its device.

```python
x = torch.randn(4, 4, device="cuda")   # data generated directly on the GPU — never on the CPU
x = torch.randn(4, 4).cuda()           # generated on the CPU, then copied over PCIe — slower
y = x @ x                              # result is on the GPU too: ops stay on their inputs' device
y.device                               # cuda:0
```

Data moves to the CPU **only when asked** — `.cpu()`, `.to("cpu")`, `.item()`,
`.tolist()`, `print(y)` — and each of those waits for the GPU (§1). Mixing devices is an
error, not a silent copy: `cuda_tensor @ cpu_tensor` raises "Expected all tensors to be on
the same device" (a 0-dim CPU scalar is the one exception). That is why a whole forward
stays on the GPU: weights and inputs are there, so every intermediate is created there.

"Allocating a block" to a request (request-lifecycle.md §10) creates no memory either: the
pool is one tensor PyTorch created at startup, and the scheduler hands out a block
**number** that kernels turn into addresses inside it.

## 0.4 Tensors and addresses

**PyTorch indexing is offset arithmetic.** Every tensor stores **strides** — how far to
jump in memory per step along each axis — and indexing computes
`offset = Σ index × stride`:

```python
k_pool = torch.arange(10 * 4 * 12 * 64).reshape(10, 4, 12, 64)   # (num_blocks, block_size, H, D)
k_pool.stride()          # (3072, 768, 64, 1)

t = k_pool[9, :, 3, :]   # block 9, every token slot, head 3, every dim  → shape (4, 64)
t.storage_offset()       # 27840 = 9·3072 + 3·64
t.stride()               # (768, 1)
# a VIEW: same memory, nothing copied

t[2, 5] == k_pool.flatten()[9·3072 + 2·768 + 3·64 + 5]   # = flat[29381] → True
#    │  └ d = 5: the 6th number of the head-3 vector
#    └─── token slot 2 of block 9
```

`t[2, 5]` is only a check that the formula is right. Real kernels never pick one
dimension: they load all 64 numbers of a vector and compute on them together (a dot
product sums over all 64; tensor cores multiply whole tiles in one instruction).

**Kernels have no tensors.** Inside a GPU kernel there are no shapes, strides or indexing
syntax — only raw **pointers** and numbers. So the host passes strides as plain numbers
(`k_pool.stride(0)`, …) and the kernel spells the arithmetic out:

```
PyTorch (on the CPU)            kernel (on the GPU)
k_pool[9, :, 3, :]       ≡      K_pool + 9·stride_block + t·stride_tok + 3·stride_head + d
```

**Views versus copies.** Basic slicing like `k_pool[9, :, 3, :]` is a view — a regular
stride pattern over the same memory. Indexing with a tensor of scattered indices, like
`k_pool[block_table]` with `[9, 5, 1]`, cannot be described by strides, so PyTorch
**gathers a copy**. That difference decides whether paged attention can stay in PyTorch
(Step 4, "Why not just index in PyTorch").

Newer Triton offers **block pointers / tensor descriptors** (`tl.make_block_ptr`,
`tl.make_tensor_descriptor`): give a base pointer, shape and strides once, then load tiles
by block coordinates — closer to tensor indexing, and on H100 mapped onto the TMA copy
hardware. The arithmetic underneath is the same.

---

# Step 1 — Launching work

## Launching only enqueues

```python
y = x @ x        # puts a kernel in the stream and returns in ~0.02 ms; the GPU runs it later
```

**Without `torch.cuda.synchronize()` the work still runs.** `synchronize()` starts
nothing; it only makes the CPU wait until the queue is empty. It is for **timing and
debugging**:

- GPU follow-on work (`z = y + 1`) is ordered by the stream — no waiting needed.
- Anything that brings a value to the CPU — `print(y)`, `.cpu()`, `.item()`, `.tolist()`,
  `int(t)`, `if t > 0:` — **waits automatically**. You cannot read an unfinished result.
- The queue is finite: enqueue roughly a thousand launches ahead and the next one blocks.
- Kernel errors surface at the next sync, often on an unrelated line.
  `CUDA_LAUNCH_BLOCKING=1` makes every launch synchronous for debugging.

```python
t0 = time.perf_counter()
y = x @ x
t1 = time.perf_counter()         # ~0.02 ms: only the launch — misleading as "compute time"
torch.cuda.synchronize()
t2 = time.perf_counter()         # ~5–20 ms: the real compute
```

## PyTorch, Triton and CUDA C++

Three ways GPU code gets written; all end in the same CUDA launch on the same stream.

| way | the GPU code | who launches |
|---|---|---|
| **PyTorch op** (eager) | prebuilt kernels shipped in PyTorch, or libraries like cuBLAS | PyTorch's dispatcher |
| **`torch.compile`** | Triton kernels *generated* by PyTorch's compiler, plus prebuilt ones | PyTorch |
| **hand-written Triton** | your `@triton.jit` function, compiled on first call, cached | Triton's launcher, on PyTorch's current stream |
| **CUDA C++** | `.cu` code compiled ahead of time by `nvcc` | `<<<grid, threads>>>`, via a C++ extension |

**Triton is a language plus a compiler for writing GPU kernels** — a separate project from
PyTorch, designed to work with it. A `@triton.jit` function *looks* like Python but the
Python interpreter never runs it: Triton compiles its source (Triton IR → MLIR → LLVM →
PTX → GPU binary). The syntax is Python; the meaning is different:

| inside `@triton.jit` | normal Python / PyTorch |
|---|---|
| runs on the GPU, once per CTA | runs on the CPU |
| works on **blocks** of values (`tl.arange(0, 1024)`) | works on objects and whole tensors |
| only `tl.*` operations: `tl.load`, `tl.store`, `tl.dot`, `tl.exp`, … | anything |
| no `torch.*`, no tensor indexing, no Python lists | all available |
| tile sizes are compile-time constants | sizes can be anything |

**PyTorch code runs on the GPU without Triton.** Eager PyTorch calls prebuilt kernels —
that is how this repo's GPT-2 runs. Triton enters in two cases:

```
plain PyTorch            → prebuilt CUDA kernels                         (no Triton)
PyTorch + torch.compile  → Inductor translates + fuses → Triton → GPU    (Triton, generated)
@triton.jit kernel       → Triton compiler → GPU                         (Triton, hand-written)
```

`torch.compile`'s gain is **fusion**: `torch.relu(x * 2 + 1)` is three kernels and three
round trips through HBM in eager mode, one Triton kernel with one read and one write when
compiled (`TORCH_LOGS="output_code"` prints the generated kernels). Large matmuls still go
to cuBLAS and attention to FlashAttention. It cannot invent new algorithms — a Python loop
over blocks will not become a paged-attention kernel — and it has the **same restrictions
as CUDA graphs**: `.item()`, `int(tensor)` or an `if` on a GPU value cause a *graph
break*, and that part runs as plain Python.

## Triton, line by line

```python
import torch, triton
import triton.language as tl                         # tl: the ops usable inside kernels

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid    = tl.program_id(0)                        # which CTA am I — not which thread
    offs   = pid * BLOCK + tl.arange(0, BLOCK)       # the BLOCK indices this CTA handles
    mask   = offs < n                                # stop at the end of the array
    x_vals = tl.load(x_ptr + offs, mask=mask)        # this chunk's values, copied into the CTA
    y_vals = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x_vals + y_vals, mask=mask)   # write into memory PyTorch created

n   = 2500                                           # length of the arrays
x   = torch.randn(n, device="cuda")
y   = torch.randn(n, device="cuda")
out = torch.empty(n, device="cuda")                  # the output: allocated by PyTorch, first

grid = (triton.cdiv(n, 1024),)                       # (3,): how many CTAs
add_kernel[grid](x, y, out, n, BLOCK=1024)           # launch
torch.testing.assert_close(out, x + y)
```

| piece | meaning |
|---|---|
| `n` | the length of the arrays, passed so CTAs know where the data ends |
| `BLOCK = 1024` | **your choice**: how many elements each CTA handles; must be a power of 2 (`tl.arange` requires it) |
| `triton.cdiv(n, 1024)` | **ceiling division**: 2500 / 1024 = 2.44 → 3 CTAs, so every element is covered |
| `grid = (3,)` | a tuple giving how many CTAs along each axis; here 1-D, so `program_id(0)` ∈ {0, 1, 2} |
| `add_kernel[grid](…)` | Triton's launch syntax: `[ ]` says how many CTAs, `( )` passes the data — CUDA's `<<<3, 256>>>(…)` in Python form |
| `x_ptr`, `y_ptr`, `out_ptr` | matched to `x, y, out` by position; inside the kernel each is a **pointer** to the tensor's first element in GPU memory |
| `n` inside | a plain number |
| `BLOCK` inside | a **compile-time constant**, baked into the compiled kernel — changing it recompiles |

**Pointers and offsets.** `x_ptr` refers to the *whole* array; `x_ptr + offs` is a vector
of addresses, one per index, scaled by the element size automatically (`+ 1024` means
1024 elements, not bytes). `tl.load` reads those addresses into the CTA; the original
arrays stay where they are.

```
x_ptr ──▶ x[0] … x[1023] | x[1024] … x[2047] | x[2048] … x[2499] ┆ (past the end)
           └─── CTA 0 ───┘ └──── CTA 1 ──────┘ └──── CTA 2 ──────┘┆ masked off

CTA 0:  offs = 0    … 1023   mask: all True
CTA 1:  offs = 1024 … 2047   mask: all True
CTA 2:  offs = 2048 … 3071   mask: True up to 2499, False after
```

Without the mask, CTA 2 would read past the end of the array — garbage or a crash. In
Triton you write code for **one CTA working on a vector of BLOCK elements**; Triton spreads
it across the CTA's threads.

## CUDA C++ — code per thread

```cpp
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's global index
    if (i < n) out[i] = x[i] + y[i];                 // guard the tail, like Triton's mask
}
add_kernel<<<(n + 255) / 256, 256>>>(x, y, out, n);  // CTAs, 256 threads each
```

`blockIdx.x` is the CTA's index (Triton's `pid`), `blockDim.x` is 256, `threadIdx.x` is
0–255. **256 is threads per CTA**, the programmer's choice: a multiple of 32 (warps), at
most 1024, typically 128–256 (§0.2). CUDA code is written per **thread**; Triton code per
**CTA**.

## CUDA graphs — record once, replay with one call

A forward pass is ~300 launches (≈10 kernels per layer × 12 layers, plus embedding,
norm, `lm_head`, sampling), each costing the CPU a few µs. A CUDA graph records them
once:

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_buf = model(input_ids_buf, positions_buf)   # record, once

input_ids_buf.copy_(new_ids)                        # every step: new data into the SAME buffers
positions_buf.copy_(new_pos)
g.replay()                                          # all ~300 kernels, one call
result = out_buf                                    # same output tensor, overwritten each replay
```

**On replay, no Python runs — only the recorded kernels.** Every requirement follows:

| requirement | why | what breaks it |
|---|---|---|
| no CPU-GPU syncs during capture | the GPU is not running during capture | `.item()`, `.cpu()`, `print(t)`, **`int(t)`** |
| fixed memory addresses | kernels replay with the pointers they recorded | new input tensors each step — use `.copy_()` into fixed buffers |
| fixed shapes | sizes are baked into kernel arguments | a different batch size needs its own graph (buckets + padding) |
| no Python branches on GPU values | Python ran once; the branch it took is baked in | `if logits.max() > 0:` |
| CPU-side values are frozen | Python does not rerun | `past = int(pos)` keeps its capture-time value |
| memory allocated before capture | replays rely on fixed addresses | allocating new buffers between replays |
| warm up before capture | first calls do one-time setup (handles, autotuning, compilation) | capturing on the very first call |
| everything on the GPU | only GPU work is recorded | a tensor created on the CPU then `.cuda()` inside `forward` |

To make sure everything is on the GPU: the model on the GPU, every input a preallocated
GPU buffer, every tensor created inside `forward` created with `device=x.device`, and no
sync on the captured path. Then try — capture **fails loudly** ("operation not permitted
when stream is capturing") if anything syncs.

## Why `int()` forces a sync

`int()` is **Python's** built-in, and Python runs on the CPU. A Python `int` is an object
only the interpreter understands; the GPU cannot make one. So `int(gpu_tensor)` must wait
until every queued kernel that might change the tensor has finished, copy 8 bytes, and
wrap them — a sync.

The GPU *does* have "int": the value stays a tensor and kernels use it directly.

| CPU / Python way | GPU / tensor way |
|---|---|
| `past = int(self.pos)` | keep `self.pos` as a 1-element GPU tensor |
| `past + 1` | `self.pos + 1` |
| `range(past, past + T)` | `self.pos + torch.arange(T, device=…)` |
| `x[:, :, past] = k` | `x.index_copy_(2, self.pos, k)` |

The CPU genuinely needs numbers in only three places, because it allocates memory and
launches work: **shapes**, **grid sizes**, and **Python control flow**. Any of those
depending on a GPU-computed value forces a sync. The GPU has control flow of its own —
branches inside kernels, conditional nodes in recent CUDA graphs — but none of it hands
Python a number.

## This repo's CUDA graph

[decode_step.py](../src/gpt2/inference/static_batch/decode_step.py) applies every rule above:

```python
self.token = torch.zeros(B, 1, …, device=device)      # fixed input buffers, allocated once
self.pos   = torch.zeros(B, 1, …, device=device)
self.mask  = torch.zeros(B, W, …, device=device)

s = torch.cuda.Stream()                               # warm up on a side stream
with torch.cuda.stream(s):
    for _ in range(3):
        model(self.token, kv_cache=self.cache, pos_ids=self.pos, attn_mask=self.mask)

with torch.cuda.graph(self.graph):                    # capture once
    self.logits, _ = model(self.token, …, pos_ids=self.pos, …)   # self.logits = fixed output
```

and [inference.py](../src/gpt2/inference/static_batch/inference.py) feeds it every step:

```python
step.token.copy_(next_token)
step.pos.copy_(row_pos.unsqueeze(1))
step.graph.replay()
```

**Why `pos_ids=self.pos` is passed.** Without it, [gpt.py](../src/gpt2/gpt.py)'s default
path computes `past = kv_cache.seq_len()` — `int(self.pos)`
([kv_cache.py](../src/gpt2/inference/static_batch/kv_cache.py)). During capture that is a sync, so
capture fails; even if it were allowed, `torch.arange(past, past + T)` would be recorded as
"fill with the capture-time value", embedding the same position on every replay. Passing
`self.pos` makes the embedding kernel read positions **from GPU memory** on every replay.
(With left-padding each row's real position differs anyway, so per-row `pos_ids` is
needed for correctness too.)

The cache follows the same rule: its write column is read from the GPU tensor `self.pos`
inside `index_copy_`, and `self.pos += T` is a recorded GPU op that advances on every
replay. `T` itself *is* frozen at 1 — fine, because every decode step has `T = 1`. A
tensor-only default path would avoid the trap entirely:

```python
pos_ids = kv_cache.pos + torch.arange(T, device=idx.device)   # T is a shape: a Python number is fine
```

---

# Step 2 — Inputs reach the GPU

The CPU has packed the five arrays for this step (request-lifecycle.md §11). They go
**into the fixed buffers allocated at startup**:

```python
input_ids_buf[:6].copy_(torch.tensor([t2, s4, s5, b0, b1, b2]), non_blocking=True)
positions_buf[:6].copy_(torch.tensor([9, 4, 5, 6, 7, 8]),      non_blocking=True)
slot_map_buf[:6].copy_(torch.tensor([29, 20, 21, 22, 23, 4]),  non_blocking=True)
cu_seqlens_buf[:3].copy_(torch.tensor([0, 1, 6]),              non_blocking=True)
seq_lens_buf[:2].copy_(torch.tensor([10, 9]),                  non_blocking=True)
block_table_buf[1].copy_(torch.tensor([9, 5, 1, 0]))           # only R2's row changed — admitted
```

- `copy_` writes into the existing buffer, keeping the address a graph recorded.
- `non_blocking=True` lets the CPU continue without waiting. It is only truly
  asynchronous from **pinned** host memory, which the GPU can read directly; ordinary
  memory forces a staged, effectively blocking copy. Do not overwrite the pinned source
  until the copy has run — double-buffer, or wait on an event.
- The copy is queued on the same stream, so the kernels after it see the new values.
- The five arrays ship **every** step; the block table is only **patched** when a block
  id entered a row (request-lifecycle.md §19).

Then the forward's first kernels run:

```python
x = wte[input_ids] + wpe[positions]      # (6, 768) — no batch axis; two requests, one flat stream
```

The embeddings are used **once, here**. No layer reads `wte` or `wpe` again; layers read
`x`, which starts as the embedding and is updated layer by layer (Step 3). `wte` is used
once more at the very end, as `lm_head`, through weight tying (Appendix A).

---

# Step 3 — The layer: project and write

## The transformer block

Each of the 12 layers is a `Block` with two halves — [gpt.py](../src/gpt2/gpt.py):

```python
def forward(self, x, …):
    x = x + self.attn(self.ln_1(x), …)     # attention: tokens exchange information
    x = x + self.mlp(self.ln_2(x))         # MLP: each token processes what it has
    return x                               # → the next block's x
```

`x` is **one running vector per token** — the *residual stream* — flowing through all 12
layers. Each half reads it, computes an update, and **adds** it in:

```
embed: x = wte[token] + wpe[pos]                       (6 tokens × 768)
  ├─ layer 0 ─┬─ ln_1 → attention → + ─┬─ ln_2 → MLP → + ─┐
  │           └──────── x ─────────────┘└────── x ─────────┘
  ├─ layer 1 ─┬─ ln_1 → attention → + …   (reads layer 0's output)
  ⋮
  └─ layer 11 → x  →  ln_f → lm_head → logits
```

| | attention | MLP |
|---|---|---|
| mixes **across tokens**? | **yes** — each token pulls information from earlier tokens | **no** — each token on its own |
| informally | gather what is relevant from the context | think about what was gathered |
| in a serving engine | needs the KV cache, block tables, special kernels | a plain matmul over the flat batch |

**The MLP**, applied to each token separately:

```python
h = ln_2(x)          # (6, 768)   normalise
h = c_fc(h)          # (6, 3072)  expand 4×        W: 768 × 3072
h = gelu(h)          # (6, 3072)  non-linearity
h = c_proj(h)        # (6, 768)   project back     W: 3072 × 768
x = x + h            # add the update to the residual stream
```

The non-linearity is what lets the network compute more than linear combinations; the MLP
holds about two-thirds of each layer's weights. Since it never looks at other tokens, it
does not care that the batch mixes requests.

**Why `x = x + h`.** Three jobs:

1. **Gradients flow through depth.** `d(x + h)/dx = I + dh/dx`; the identity is a direct
   path, so gradients neither vanish nor explode across many layers.
2. **Nothing is overwritten.** The token's identity, its position and what earlier layers
   worked out stay in the stream for later layers to read.
3. **Doing nothing is easy.** A layer that is not useful for some input outputs `h ≈ 0`;
   a fresh network starts close to the identity, so training is stable early.

Because 24 sublayers add into the same stream, its variance grows with each addition; the
model starts the projections that write into it smaller, by `1/√(2 × n_layer)`
([gpt.py](../src/gpt2/gpt.py), `c_proj.weight`). **Layer norm** handles the other kind of
stability — activation scale — and GPT-2 places it *before* each sublayer (*pre-norm*),
leaving the residual path a pure identity.

**From the MLP into the next attention.** The next layer's q, k and v are computed from
the MLP-updated `x`, with that layer's own weights. So **each layer's K/V differ**, even
for the same token — the reason the pool has a slice per layer — and layers cannot be
skipped or reordered.

## Self-attention in four moves

Each token **looks at earlier tokens and pulls in information from the relevant ones**:

1. **q, k, v.** Each token's vector times three learned matrices: `q = x @ W_q` ("what am
   I looking for?"), `k = x @ W_k` ("what do I contain?"), `v = x @ W_v` ("what do I pass
   on?").
2. **Compare.** Dot product of the token's **q** with every visible token's **k**, divided
   by `√head_dim`. The **causal mask** allows only itself and earlier tokens.
3. **Softmax.** Scores become positive weights summing to 1.
4. **Mix.** The output is the weighted sum of the visible tokens' **v**.

A tiny example — three tokens, vectors of size 2, one head; `W_q` and `W_v` leave vectors
unchanged and `W_k` swaps the two numbers:

```
token   x        q        k        v
the     [1, 0]   [1, 0]   [0, 1]   [1, 0]
cat     [0, 1]   [0, 1]   [1, 0]   [0, 1]
sat     [1, 1]   [1, 1]   [1, 1]   [1, 1]

"sat":  scores q·k   the 1   cat 1   sat 2
        ÷ √2         0.71    0.71    1.41
        softmax      0.25    0.25    0.50
        output       0.25·[1,0] + 0.25·[0,1] + 0.50·[1,1] = [0.75, 0.75]
```

GPT-2 does this **12 times in parallel** — 12 heads of 64 numbers, each with its own slice
of the weights and its own pattern — then joins the 12 outputs, applies the output
projection, and adds the result to the residual stream.

**Why this produces the KV cache.** A new token needs its **own q** — computed now, used
once — and **every earlier token's k and v**, which never change. So the cache stores K
and V, never Q, and a decode step computes just the new token's q, k, v and reads the
rest.

## Project — q, k, v for the new tokens

```python
qkv = ln_1(x) @ W_qkv + b_qkv     # (6, 768) @ (768, 2304) → (6, 2304)
q, k, v = qkv.split(768)          # each (6, 768), viewed as (6 tokens, 12 heads, 64)
```

- **`ln_1`** is LayerNorm: per token, rescale the 768 numbers to mean 0 and variance 1,
  then a learned scale and shift. Not the linear transform — that is the `@ W_qkv`.
- **`W_qkv`** is three `768 × 768` matrices side by side, `[W_q | W_k | W_v]`, so one
  matmul produces all three — fused QKV, because one large matmul beats three small ones.
  In this repo it is `c_attn` ([gpt.py](../src/gpt2/gpt.py)); `nn.Linear` stores its
  weight as `(2304, 768)` and computes `x @ weight.T + bias`.
- **Each layer has its own `W_qkv`** — 12 different matrices.
- **Heads are slices**, not separate matrices: columns 0–63 of `W_q` produce head 0's
  query, 64–127 head 1's, and so on.

This one matmul covers all 6 tokens, whichever request they belong to — and only the
**6 new tokens**. Older positions are not in the batch; their K/V are already in the pool.

| | goes to | lives |
|---|---|---|
| `k`, `v` | **written into the pool** (below) | as long as the request, read by future steps |
| `q` | **used by attention now** (Step 4) | a temporary activation, gone after this layer |

## Write — store the new K/V in the pool

**Why write before reading.** Attention needs K/V for *every* position a query can see:

| query | keys it attends | where they come from |
|---|---|---|
| R1 pos 9 | 0–9 | 0–8: R1's own earlier iterations · 9: written **just now** |
| R2 pos 4 | 0–4 | 0–3: **block 9, written by R1 in iteration 1** (the prefix hit) · 4: just now |
| R2 pos 8 | 0–8 | 0–3: R1's block 9 · 4–8: just now |

After the write, old and new K/V are all in the pool, so Step 4 reads them uniformly. R2
reads K/V it never computed (its logical block 0 is physical block 9), and R2's new
tokens attend to **each other and to themselves** — keys that exist only because this
write came first. The write also serves the future: iteration 5 reads these as history.
It cannot move: **not earlier**, because layer *L*'s K depends on layers 0…L−1; **not
later**, because the new token must see itself.

The write is one small kernel — vLLM calls it `reshape_and_cache`. In Triton:

```python
@triton.jit
def reshape_and_cache(K_new, K_pool, SlotMapping, H: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)                             # which new token
    h = tl.program_id(1)                             # which head
    slot = tl.load(SlotMapping + i)                  # where it goes, e.g. 29
    if slot >= 0:                                    # -1 = padding row: skip
        d = tl.arange(0, D)
        k = tl.load(K_new + i * H * D + h * D + d)           # READ this token's key, head h
        tl.store(K_pool + slot * H * D + h * D + d, k)       # WRITE it into the pool's slot

# host side
k_pool = torch.empty(num_blocks, block_size, 12, 64,           # created ONCE at startup, by PyTorch
                     dtype=torch.float16, device="cuda")
reshape_and_cache[(6, 12)](k_new, k_pool, slot_mapping, H=12, D=64)   # every layer, every step
```

For iteration 4 that writes:

```
k[0]  R1 pos 9 → slot 29  (block 7, offset 1)     k[1]  R2 pos 4 → slot 20  (block 5, offset 0)
…                                                 k[5]  R2 pos 8 → slot 4   (block 1, offset 0)
```

---

# Step 4 — Read and attend

## Rows and tiles

After the projection, for one head, `q` is a matrix with **one row per token** in the flat
batch — a row is that token's 64-number query vector:

```
q for head h:
row 0  │ q of t2 (R1, pos 9) │      row 3  │ q of b0 (R2, pos 6) │
row 1  │ q of s4 (R2, pos 4) │      row 4  │ q of b1 (R2, pos 7) │
row 2  │ q of s5 (R2, pos 5) │      row 5  │ q of b2 (R2, pos 8) │
```

Rows are grouped into **tiles**. The kernel has a maximum tile size — 4 rows here to keep
the example small; real kernels use 64 or 128 — and tiles are cut **per request**, never
mixing requests. The host builds the tile table from `cu_seqlens_q = [0, 1, 6]`:

| tile | request | rows | queries |
|---|---|---|---|
| 0 | R1 | 0 | 1 |
| 1 | R2 | 1–4 | 4 |
| 2 | R2 | 5 | 1 |

Every CTA is **built for 4 rows** — same code, same SRAM — and a tile with fewer rows
**masks off** the unused ones, like `mask = offs < n` in the add kernel.

- **A tile never mixes requests** because all its rows share **one walk through one block
  table**; R1 and R2 have different histories.
- **Tiles exist for reuse**: each K/V block is loaded from HBM into SRAM **once** and used
  by every row in the tile. One row per CTA would reload R2's blocks 5 times.
- **Decode gets almost no reuse**: one row in a 64- or 128-row tile — part of why decode
  gets its own treatment (FlashDecoding, below).

## The grid: one CTA per (query tile, head)

```
grid = (3 tiles, 12 heads) → 36 CTAs

tile = tl.program_id(0)    # → which request and which rows, from the tile table
h    = tl.program_id(1)    # → which head's 64-number slice of q and of the K/V blocks
```

| CTA | `program_id(0)` | `program_id(1)` | work |
|---|---|---|---|
| 1 | 0 | 0 | R1's query, head 0, walking R1's block table |
| 2 | 1 | 0 | R2's queries 4–7, head 0, walking R2's block table |
| … | … | … | … |
| 36 | 2 | 11 | R2's query 8, head 11 |

**Two axes because the work is independent along both**: query tiles (no query needs
another's output within a layer) and heads (each has its own q/k/v slice and softmax;
outputs are combined only afterwards by the output projection).

**Tiles on axis 0** is mostly convention — a 1-D grid with `tile = pid // 12`,
`h = pid % 12` works too. Two practical reasons for this order: CUDA allows ~2 billion
CTAs on axis 0 but only 65,535 on the others, and the tile count is the one that can be
huge; and CTAs start with axis 0 changing fastest, so consecutive tiles of the same head —
which read the same K/V blocks in a long prefill — run close together and find those
blocks still in the L2 cache.

## What a CTA loads

Each CTA needs two things: **q** from this step's projection, and **K, V** from the pool.

```
out = softmax(q · Kᵀ / √64) · V
          ▲     ▲              ▲
          │     └──────────────┴── K, V: from the KV pool — history plus what Step 3 wrote
          └─────────────────────── q: from this step's projection — never cached
```

One physical block's K part is `(4 slots, 12 heads, 64)`; a CTA handles one head, so it
takes **only that head's slice** of each block:

```
k_pool[layer][9]  — block 9, keys for positions 0–3 (written by R1 in iteration 1)

            head 0     head 1    …    head 3     …    head 11
slot 0   │ 64 nums │ 64 nums │ … │ 64 nums │ … │ 64 nums │
slot 1   │ 64 nums │ 64 nums │ … │ 64 nums │ … │ 64 nums │
slot 2   │ 64 nums │ 64 nums │ … │ 64 nums │ … │ 64 nums │
slot 3   │ 64 nums │ 64 nums │ … │ 64 nums │ … │ 64 nums │
                                  └───────┘
                    the CTA for head 3 takes this column: (4, 64)

K = k_pool[layer][physical, :, 3, :]
#                 │         │  │  └─ all 64 numbers of the vector
#                 │         │  └──── head index 3 — the fourth head (heads are 0–11)
#                 │         └─────── all token slots in the block
#                 └───────────────── the physical block id
```

| CTA | loads |
|---|---|
| tile 1, head 3 | head 3's slice of R2's blocks 9 and 5 (its queries reach position 7) |
| tile 1, head 7 | head 7's slice of the **same** blocks |
| tile 2, head 3 | head 3's slice of R2's blocks 9, 5 **and 1** (its query is position 8) |
| tile 0, head 3 | head 3's slice of **R1's** blocks 9, 2, 7 |

Different heads read different columns of the same blocks, and **block 9 is read by both
R1's and R2's CTAs** — prefix sharing, invisible to the kernel, which just follows the
block table.

**Streaming, not grabbing everything.** SRAM holds the q_tile, the *current* block's K and
V, and the softmax state; each block is loaded, used and overwritten by the next. That is
why online softmax exists (below).

**The last block may be partly filled.** R2's history is 9 tokens, so block 1 holds a real
token only in slot 0 (position 8); slots 1–3 are stale. The kernel knows the length from
`seqused_k = 9` and masks those scores to −∞, giving them zero weight.

**Layout.** In the logical layout `(block_size, n_head, head_dim)`, a head's four slot
vectors are *not* adjacent in memory — 11 other heads sit between them. GPUs read fastest
when neighbouring threads read neighbouring addresses (*coalescing*), so real kernels
often store the pool as `(num_blocks, n_head, block_size, head_dim)`, making each head's
slice of a block one contiguous chunk. Same content, kernel-chosen order
(request-lifecycle.md §3).

## Inside one CTA — tile 1, head h

```python
q_tile = load(q[1:5, h, :])                         # (4, 64): R2's queries for positions 4–7
m   = full(4, -inf)        # running max score, per query row          ┐ online-softmax state,
l   = zeros(4)             # running sum of exp(score), per query row  │ in registers / SRAM
acc = zeros(4, 64)         # running weighted sum of V, per query row  ┘

for logical in range(7 // 4 + 1):                   # keys up to position 7 → logical blocks 0, 1
    physical = block_table[R2][logical]             # [9, 5, 1] → 9, then 5 — on the device
    K = load(k_pool[layer][physical, :, h, :])      # (4, 64): one block, HBM → SRAM
    V = load(v_pool[layer][physical, :, h, :])
    s = q_tile @ K.T / 8                            # (4, 4) scores — all 64 dims at once; 8 = √64
    s = mask_future(s, q_pos=[4..7], k_pos=[4*logical .. 4*logical+3])
    m, l, acc = online_softmax_step(m, l, acc, s, V)

store(out[1:5, h, :], acc / l[:, None])             # (4, 64): SRAM → HBM
```

- **Logical vs physical.** A request numbers its own blocks from 0
  (`logical = pos // block_size`); the pool's real block is
  `block_table[request][logical]`. The translation happens **on the GPU, inside the
  kernel**.
- **A tile loops only over blocks it can see.** Tile 1's latest query is position 7, so
  block 1 (position 8) is skipped; tile 2 visits all three.
- **Data movement.** The q_tile enters SRAM **once**; K/V blocks stream through; the
  score matrix is never written to HBM, and K/V are never gathered into a copy.

## How `load` actually happens

`load(...)` above is shorthand for: compute the addresses, read them (coalesced, masked),
place the result on-chip, and make sure it is complete before anyone uses it. Element
`[physical, t, h, d]` of the pool sits at
`physical·3072 + t·768 + h·64 + d` (strides from §0.4).

**In Triton** — `tl.load` on a 2-D grid of addresses; the compiler decides registers vs
SRAM and adds the synchronisation:

```python
physical = tl.load(BlockTable + seq * stride_bt + logical)       # one number, e.g. 9
offs_t = tl.arange(0, BLOCK_SIZE)                                # [0, 1, 2, 3]  token slots
offs_d = tl.arange(0, HEAD_DIM)                                  # [0 … 63]      dimensions
k_ptrs = (K_pool + physical * stride_block
                 + offs_t[:, None] * stride_tok                  # one row per slot
                 + h * stride_head
                 + offs_d[None, :])                              # → a (4, 64) grid of addresses
valid = (logical * BLOCK_SIZE + offs_t) < seq_len                # stale slots in the last block
K = tl.load(k_ptrs, mask=valid[:, None], other=0.0)              # → a (4, 64) tile
```

**In CUDA C++** — an explicit shared-memory array, filled cooperatively:

```cpp
__shared__ float K_s[BLOCK_SIZE][HEAD_DIM];          // in SRAM, shared by the whole CTA
for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
    int t = i / HEAD_DIM, d = i % HEAD_DIM;          // 256 elements, 128 threads → 2 each
    K_s[t][d] = k_pool[physical * stride_block + t * stride_tok + h * stride_head + d];
}
__syncthreads();   // nobody uses K_s until EVERY thread has finished copying
```

`__syncthreads()` is the "can wait for each other" property that makes a CTA
*cooperative*. Neighbouring threads read neighbouring `d`, so a warp's 32 reads combine
into a few wide transactions — coalescing.

**On newer GPUs**, copies can skip the threads: `cp.async` (A100) starts an HBM → SRAM copy
in the background, and **TMA** (H100) copies a whole tile with one instruction. That
enables **double buffering** — block *i+1* is copied in while block *i* is computed —
which FlashAttention-3 is built around.

## Online softmax — attention one block at a time

Softmax needs the max and the sum over **all** keys, but the CTA sees one block at a time
and cannot keep every score. Keep a running max `m`, sum `l` and weighted sum `acc`, and
**rescale earlier work** when a larger score appears:

```python
def online_softmax_step(m, l, acc, s, V):
    m_new = max(m, s.max())
    corr  = exp(m - m_new)          # shrink earlier work to the new scale
    p     = exp(s - m_new)
    return m_new, l * corr + p.sum(), acc * corr + p @ V
# after the last block: out = acc / l
```

```
scores [1, 2], then [3]:
  block 1:  m = 2,  p = [0.368, 1],  l = 1.368,  acc = 0.368·v1 + v2
  block 2:  m = 3,  corr = 0.368,  p = [1]
            l = 1.368·0.368 + 1 = 1.503,   acc = 0.135·v1 + 0.368·v2 + v3
  out = (0.135·v1 + 0.368·v2 + v3) / 1.503        ← identical to ordinary softmax
```

**Exact, not approximate** — `exp(xᵢ − m) / Σ exp(xⱼ − m)` holds for any `m`. Memory is
`O(head_dim)` per query row **regardless of history length**.

## FlashAttention vs FlashDecoding — which axis is split across CTAs

**FlashAttention splits queries.** "Each query must attend to its previous K/V — so why is
this parallel?" Because within one layer no query needs another query's *output*: each
needs only K/V, and all of those exist before attention starts (Step 3). Generation is
sequential only **across iterations** — token 10 cannot be chosen until token 9 is
sampled. Prefill has many query tiles, so plenty of CTAs:

```python
grid = (num_query_tiles, num_heads)      # 512-token chunk, 64-row tiles → 8 × 12 = 96 CTAs
# each CTA: one q_tile, walk the block table, online softmax → final outputs for its rows
```

**FlashDecoding splits keys.** A decoding request has one query, so only
`requests × heads` CTAs — 8 × 12 = 96 on a 108-SM GPU, some idle — each walking a long
history alone. So the history is split into ranges; each CTA produces a *partial*
`(m, l, acc)`, and a second small kernel merges them:

```python
# kernel 1 — grid = (num_requests, num_heads, num_splits), e.g. 8 × 12 × 8 = 768 CTAs
def split_cta(seq, h, split):
    q = load(q[row_of(seq), h])                       # the single query
    m, l, acc = -inf, 0.0, zeros(64)
    for logical in my_range_of_blocks(seq, split):
        physical = block_table[seq][logical]
        K, V = load(k_pool[physical, :, h]), load(v_pool[physical, :, h])
        m, l, acc = online_softmax_step(m, l, acc, K @ q * scale, V)   # no causal mask needed
    store(partial[seq, h, split], (m, l, acc))

# kernel 2 — grid = (num_requests, num_heads)
def merge_cta(seq, h):
    parts = load(partial[seq, h, :])
    m   = max(p.m for p in parts)
    l   = sum(p.l   * exp(p.m - m) for p in parts)
    acc = sum(p.acc * exp(p.m - m) for p in parts)
    store(out[row_of(seq), h], acc / l)
```

| | FlashAttention | FlashDecoding |
|---|---|---|
| split across CTAs | query rows (tiles) | key ranges (splits) |
| each CTA produces | final outputs for its rows | a partial result for one query |
| extra step | none | a merge kernel |
| suits | prefill — many queries | decode — one query, long history |

A mixed batch like this one uses either one variable-length kernel that picks per
sequence, or a prefill kernel plus a decode kernel. Both read through the same block
table. (request-lifecycle.md Part V-A has the merge algebra and a fused write + attend
kernel.)

## Why not just index in PyTorch?

PyTorch can express the same thing without a custom kernel:

```python
K = k_pool[block_table]                                          # (B, max_blocks, bs, H, D)
K = K.reshape(B, max_blocks * bs, H, D).transpose(1, 2)          # (B, H, L, D)
V = …same for v_pool…
mask = torch.arange(L, device="cuda") < seq_lens[:, None]
out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask[:, None, None, :])
```

It is correct and has no Python loop. It is slow because `k_pool[block_table]` is
**advanced indexing over scattered blocks**, which **gathers a copy** (§0.4):

```
gather version, per layer:  read the pool → write a copy → SDPA reads the copy   ≈ 3 passes
paged kernel, per layer:    read the pool straight into SRAM                     = 1 pass
```

- **Decode attention is memory-bound**, so three passes cost roughly three times as much.
- **The copy can be enormous**: 256 sequences × 8k tokens × 3 KiB per token per layer
  ≈ **6 GiB of temporary memory, per layer, every step** — taken from the KV pool.
- **The rectangle comes back**: every sequence is padded to the longest, then masked.

The kernel performs **exactly the same indexing**, `block_table[seq][logical]`, but uses
each index **while loading** — read in place instead of gathered. The PyTorch version is
still worth writing first: it gets paging correct, and it stays as the reference that a
Triton kernel is tested against (`assert_close(triton_out, gather_out)`).

## Then the rest of the layer

Output projection, residual, layer norm, MLP — plain per-token matmuls on the flat
`(6, 768)`. Then the next layer repeats Steps 3–4 against its own pool.

---

# Step 5 — Sample, on the GPU

FlashAttention and FlashDecoding belong to the forward; sampling runs after the last layer
and is ordinary tensor ops.

## Logits only for rows that will sample

```python
last_rows   = cu_seqlens[1:] - 1                       # [0, 5] — last row of each request
will_sample = num_computed + q_len >= num_tokens       # R1: decoding ✓   R2: 4 + 5 ≥ 9 ✓
rows        = last_rows[will_sample]                   # [0, 5]
logits      = lm_head(ln_f(x)[rows])                   # (2, 50257), not (6, 50257)
```

`lm_head` is the largest matmul at this model size; a request still mid-prompt gets no row
at all (request-lifecycle.md §13).

## Sampling without a CPU round trip

```python
logits = apply_penalties(logits, token_counts)         # counts kept on the GPU
logits = logits / temperature[:, None]                 # per-row settings, stored as tensors
logits = top_k_top_p_mask(logits, top_k, top_p)
probs  = logits.softmax(-1)
q      = torch.empty_like(probs).exponential_(1.0)
ids    = (probs / q).argmax(-1)                        # exact sampling; avoids multinomial's sync
ids    = torch.where(temperature == 0, logits.argmax(-1), ids)    # greedy rows
```

`probs / exponential noise → argmax` samples exactly from the distribution (the Gumbel-max
trick in another form). `ids` is 2 integers, **still on the GPU** — R1's token for
position 10, R2's for position 9.

---

# Step 6 — Back to the CPU

## The one sync, and why it feels slow

```python
next_tokens = ids.tolist()        # blocks until EVERYTHING queued before it has finished
```

The data is tiny — 2 KB even for 256 requests. What costs is **waiting**: `.tolist()`
blocks until the whole forward has finished, so "the sync" measures the forward.
Synchronously, CPU and GPU take turns:

```
CPU:  [sched+pack 3ms]                [wait…][process 1ms][sched+pack 3ms]
GPU:                 [forward 10ms   ]                                  [forward 10ms]
step ≈ 3 + 10 + 1 = 14 ms            ↑ CPU blocked        ↑ GPU idle
```

## Keeping data on the GPU — async scheduling

The next step's input *is* the token just sampled, and it is already on the GPU:

```python
ids_N = sample(logits_N)                           # on the GPU
input_ids_buf[decode_rows_N1] = ids_N              # GPU → GPU: step N+1's input, no CPU involved
ids_host.copy_(ids_N, non_blocking=True)           # background copy to pinned host memory
done_N.record()                                    # CUDA event: "step N finished"

launch(step_N1)                                    # scheduled assuming one token per request

done_N.synchronize()                               # step N finished long ago — no real wait
for req, tok in zip(reqs_N, ids_host.tolist()):
    append token, check stop, detokenize, stream   # overlaps with step N+1 on the GPU
```

```
CPU:  [sched N+1][process N−1]  [sched N+2][process N]  …
GPU:  [forward N 10ms        ]  [forward N+1        ]  …
step ≈ max(4, 10) = 10 ms → ~30% more throughput here; nearly 2× for small models
```

The two tricks: **the sampled ids go straight into the next step's input buffer on the
GPU**, and **the CPU schedules optimistically**, assuming one token per request. The cost:
a request that emits EOS at step N is already in step N+1, so one extra token is computed
and discarded.

| work | why it is on the CPU | one step behind? |
|---|---|---|
| stop strings, detokenization, streaming | needs text and Python | yes |
| EOS / `max_tokens` | finishing a request is a scheduler decision | yes — one wasted token |
| block allocation | the free list lives on the CPU | done optimistically |
| admitting new requests | the scheduler | yes |

How much it helps depends on the ratio: with a long GPU step (big model, big batch) CPU
time is a small fraction; with a short one — a 124M GPT-2 on a fast GPU — CPU overhead can
rival the forward, and async can nearly double throughput. request-lifecycle.md §20 covers
how async scheduling fits the single-writer scheduler; Part V-G lists the syncs in this
repo's `scheduler.py`.

## Out to the request

```
row 0 → R1 → append token, num_computed 10 → check stop → detokenize (hold back partial UTF-8)
                                                        → R1's output queue → SSE chunk to the client
row 1 → R2 → same
```

…and the loop returns to scheduling (request-lifecycle.md §9).

---

# The whole thing in one picture

```
Step 0  memory   PyTorch allocates HBM (weights, KV pool, buffers); kernels get pointers + strides;
                 the compiler sizes each CTA's SRAM and registers
Step 1  CPU      launch = enqueue; a CUDA graph replays ~300 kernels in one call
Step 2  CPU→GPU  five arrays (every step) + dirty block-table rows → fixed buffers, non-blocking
                 embed: x = wte[ids] + wpe[pos]                       (6, 768), no batch axis
Step 3  GPU      per layer: ln_1 → project → q, k, v (new tokens only) → WRITE K/V at slot_mapping
Step 4  GPU      grid (tile, head) → q_tile in SRAM → walk block_table, head slice of each block
                 HBM → SRAM → online softmax → out;   prefill: split queries (FA); decode: split keys (FD)
                 → o_proj, + residual, ln_2 → MLP, + residual → next layer
Step 5  GPU      gather sampling rows → lm_head → penalties, temperature, top-k/p → sample
                 ids stay on the GPU
Step 6  GPU→GPU  ids → next step's input_ids (no wait)
        GPU→CPU  ids → pinned buffer, read one step later → stop checks, detokenize, stream
```

The rule underneath every step: **values that change between steps live in GPU buffers
that kernels read; the CPU asks for a number only when it genuinely has to decide
something.**

---

# Appendix A — The model's weights

## Per layer, or shared?

| component | learned | parameters |
|---|---|---|
| `wte`: token embeddings, 50257 × 768 | **once** | 38.6M |
| `wpe`: position embeddings, 1024 × 768 | **once** | 0.8M |
| `c_attn` (`W_qkv`), 768 × 2304 + bias | **each layer** | 1.77M |
| attention `c_proj`, 768 × 768 + bias | each layer | 0.59M |
| MLP `c_fc`, 768 × 3072 + bias | each layer | 2.36M |
| MLP `c_proj`, 3072 × 768 + bias | each layer | 2.36M |
| `ln_1`, `ln_2` | each layer | 0.003M |
| **one layer** | | **7.09M × 12 = 85.1M** |
| `ln_f` | once | ~0 |
| `lm_head` | **tied to `wte`** | 0 extra |
| **total** | | **124,439,808** |

- **Every layer has its own weights** — attention, MLP and layer norms. That is why the
  same token gets different q, k, v in every layer, and why the KV pool has a slice per
  layer.
- **The embeddings are used once**, at the input — not inside the layers.
- **The one shared weight is between embedding and output**: `wte.weight =
  lm_head.weight` ([gpt.py](../src/gpt2/gpt.py)). The same matrix turns tokens into
  vectors at the start and vectors into token scores at the end, saving 38.6M parameters.
- Within a layer, the 12 heads are column slices of one matrix, not separate matrices.
  Some models share more — ALBERT reuses one layer's weights everywhere, and GQA lets
  several query heads share one K/V head within a layer — but mainstream GPT-style models
  keep independent weights per layer.

## How they are stored

`GPT` builds the model as nested modules; `h` is a `ModuleList` of 12 separate `Block`
objects, each owning its own weights:

```
GPT
├── transformer                 (an nn.ModuleDict)
│   ├── wte    (50257, 768)
│   ├── wpe    (1024, 768)
│   ├── h      ModuleList
│   │   ├── 0  Block ── ln_1, attn(c_attn, c_proj), ln_2, mlp(c_fc, c_proj)
│   │   ⋮
│   │   └── 11 Block
│   └── ln_f
└── lm_head    (50257, 768)     ← the same tensor as wte
```

`model.state_dict()` flattens the tree into **name → tensor**, using the attribute path as
the name — the layer number is part of it:

```
transformer.wte.weight                    (50257, 768)
transformer.h.0.attn.c_attn.weight        (2304, 768)     ← layer 0's W_qkv
transformer.h.0.mlp.c_fc.weight           (3072, 768)
transformer.h.1.attn.c_attn.weight        (2304, 768)     ← layer 1's: a different tensor
…
transformer.ln_f.weight                   (768,)
lm_head.weight                            (50257, 768)    ← same storage as transformer.wte.weight
```

**Why the names start with `transformer.`** Only because the modules are stored under an
attribute called `transformer` (`self.transformer = nn.ModuleDict(...)`). The name was
chosen to **match Hugging Face's `GPT2LMHeadModel`**, whose body is `self.transformer` and
whose head is `self.lm_head`. Matching names lets `from_pretrained` copy weights by name
with no renaming table. Keeping the head outside the body is also Hugging Face's
convention: the same body can carry different heads (next-token prediction,
classification).

**On disk**, a checkpoint is that name → tensor map:

| format | how | notes |
|---|---|---|
| PyTorch `.pt` / `.bin` | `torch.save(model.state_dict(), path)` | Python pickle; loading untrusted files can run code |
| **safetensors** | a JSON header (name, dtype, shape, byte offset) + raw tensor bytes | no code execution; the standard for released models; can be memory-mapped |

Loading copies each named tensor into the parameter with the same name; mismatched names
or shapes fail. This repo's `from_pretrained` ([gpt.py](../src/gpt2/gpt.py)) walks both
state dicts by name and **transposes** `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and
`mlp.c_proj`, because the original GPT-2 code stored them as `(in, out)` instead of
`nn.Linear`'s `(out, in)` — a small version of what vLLM and SGLang do when loading a
checkpoint into their own model files.

**On the GPU**, `model.to("cuda")` copies every parameter into HBM — ~500 MB in fp32 or
~250 MB in fp16 for 124M parameters, the tied `wte`/`lm_head` stored once. The weights
stay there for the life of the server: the fixed "weights" slice of memory next to the KV
pool (request-lifecycle.md §2).

---

# Appendix B — Writing your own Triton kernels

## The pattern

```python
@triton.jit
def my_kernel(X, Out, n, BLOCK: tl.constexpr):       # 1. the kernel: Triton, one instance per CTA
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)                 # read
    tl.store(Out + offs, x * 2 + 1, mask=mask)       # compute and write

def my_op(x):                                        # 2. the wrapper: PyTorch allocates and launches
    out = torch.empty_like(x)
    my_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out

x = torch.randn(1_000_000, device="cuda")            # 3. check against PyTorch, then measure
torch.testing.assert_close(my_op(x), x * 2 + 1)
print(triton.testing.do_bench(lambda: my_op(x)), "ms")
```

- **Always test against a PyTorch reference.** Kernel bugs — a wrong stride, a missing
  mask — usually give wrong numbers, not errors.
- **Benchmark with `triton.testing.do_bench`**, which handles warm-up and synchronisation.
- **Tune with `@triton.autotune`** once it works: it tries several `BLOCK` and `num_warps`
  settings and keeps the fastest for your GPU.
- **It needs an NVIDIA or AMD GPU** — Triton does not run on macOS.

## Why write Triton

**Compared with PyTorch:**

1. **Kernels that do not exist** — paged attention walking a block table, writing into
   pool slots, a new quantization format, a custom sampling rule. Often the main reason:
   necessity, not speed.
2. **Fusion — keeping data on-chip.** Each PyTorch op reads from and writes to HBM; a
   chain of memory-bound ops can be several times faster as one kernel. FlashAttention is
   the extreme case.
3. **Control** over tile sizes, memory layout, masking and data-driven indexing.
4. **Fewer launches**, which matters most in small decode steps.

**Compared with CUDA C++:**

| | Triton | CUDA C++ |
|---|---|---|
| you write | code per **tile** | code per **thread** |
| threads, shared memory, sync, coalescing | handled by the compiler | by hand |
| tensor cores, pipelining | mostly automatic | by hand |
| tuning | `@triton.autotune` | manual |
| iteration | edit Python, re-run (JIT) | compile C++, build an extension |
| hardware | NVIDIA and AMD | NVIDIA |
| peak performance | often close to hand-written CUDA | the ceiling |

**The costs:** correct kernels are hard to write and debug; existing kernels (cuBLAS,
FlashAttention, FlashInfer) are expert-tuned and usually beat a first attempt; the newest
hardware features may need CUDA (FlashAttention-3 uses CUTLASS); and every custom kernel
is code to maintain. Write one when no library provides the operation, when a chain of
memory-bound ops can be fused, or when a research idea needs it — not when
`torch.compile` or an existing kernel already does the job.

## A learning path for this repo

1. Triton's official tutorials, in order: vector add → fused softmax → matmul → fused
   attention (a FlashAttention kernel — the design of Step 4).
2. `reshape_and_cache` (Step 3), tested against PyTorch slot writes.
3. Paged attention: first the **PyTorch gather version** (Step 4) as the reference, then
   the Triton kernel, checked with `assert_close` and compared as contexts grow.
4. The fused decode kernel in request-lifecycle.md Part V-A, tested against the current
   `RowKVCache` + SDPA path.
