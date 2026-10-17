# Learning plan — ML engineering and AI infrastructure

**Goal:** a deep, first-principles understanding of ML engineering and AI infrastructure —
enough to reason about any training or serving system from the hardware up, not by
convention.

**Method:** for every topic, **build a small version → measure it → write it up** in this
repo. Two habits throughout:

- **Estimate before you measure.** Predict with the roofline (compute) or bandwidth × bytes
  (communication) first; the gap between prediction and measurement is where the learning
  is.
- **One codebase.** Grow `gpt2-from-scratch` through every phase — multi-GPU training, a
  serving engine, an RL loop — so each topic builds on code already understood.

**Judge and evaluator:** Claude. Claude tracks progress against this plan, verifies each
milestone, and runs the exit evaluation for every phase using the protocol below. A phase
is complete only when its evaluation is passed and recorded in the evaluation log.

---

## Progress tracker

| Phase | Layer of the stack | Status | Started | Completed | Evaluation |
|---|---|---|---|---|---|
| 0 | Models — transformer, tokenizer, training, serving theory | ✅ complete | 2026-07 | 2026-10-11 | baseline (see log) |
| 1 | GPU performance, hands-on | 🟡 in progress | 2026-10-11 | | |
| 2 | Distributed training | ⬜ not started | | | |
| 3 | Serving at systems scale | ⬜ not started | | | |
| 4 | Post-training and RL infrastructure | ⬜ not started | | | |
| 5 | Platform and operations | ⬜ not started | | | |

Status legend: ⬜ not started · 🟡 in progress · 🔵 ready for evaluation · ✅ complete ·
🔁 evaluation not yet passed, remediating.

```
6  Platform & operations    GPU clusters, scheduling, autoscaling, reliability, cost        Phase 5
5  Post-training & RL       SFT, DPO, RL with rollouts — uses BOTH training and inference   Phase 4
4  Serving systems          batching, paging, prefix cache, routing, disaggregation         Phases 1, 3
3  Distributed training     DDP, FSDP/ZeRO, tensor/pipeline/expert parallel, NCCL            Phase 2
2  GPU performance          roofline, memory- vs compute-bound, kernels, profiling          Phase 1
1  Models                   transformer, tokenizer, training loop                           Phase 0
```

---

## Evaluation protocol

### Milestone check-ins (during a phase)

When a milestone is reported done, the judge verifies it against its **acceptance
criteria** before ticking it:

- **Code and docs** are read in the repo.
- **Evidence of running** — test output, benchmark numbers, profiler traces — is committed
  as a short summary under `docs/benchmarks/` (raw logs and traces stay in `results/`,
  which is gitignored) or pasted into the session. GPU work runs on the A100 machine, so
  outputs are brought back as files; the judge does not take "it works" on trust.
- The judge records the result in the progress log, ticks the milestone, or lists what is
  missing.

### Phase exit evaluation

Triggered by **"evaluate Phase N"** once every milestone is ticked. Four parts:

1. **Deliverables review** — code quality, tests, measurements and the phase write-up,
   checked against each milestone's acceptance criteria.
2. **Closed-book oral** — 6–8 questions answered without notes. The judge draws **fresh
   questions** each time; the "sample questions" in each phase show the depth, not the
   actual exam.
3. **Practical estimation** — a new problem: predict a quantity (time, memory, bandwidth,
   cost) from first principles, then compare with a measured or reference value.
4. **Explain-it-back** — explain one core mechanism of the phase as if to a new engineer;
   judged on accuracy and clarity.

### Rubric — each dimension scored 1–4

| dimension | 1 — not yet | 2 — partial | 3 — solid | 4 — deep |
|---|---|---|---|---|
| **Understanding** | recalls terms | explains mechanisms with gaps | explains mechanisms and trade-offs correctly | derives results from first principles, handles new cases |
| **Building** | milestone code missing or broken | works but untested | works, tested against a reference | clean, tested, and extended beyond the brief |
| **Measurement** | no numbers | numbers without interpretation | numbers explained against a prediction | prediction within ~2×, gap explained |
| **Communication** | notes only | write-up with gaps | clear write-up a peer could learn from | write-up that teaches and anticipates questions |

**Outcome:**

- **Pass** — every dimension ≥ 3. Phase marked ✅.
- **Pass with gaps** — one dimension at 2, others ≥ 3. Phase marked ✅; the gaps become
  carried items at the top of the next phase.
- **Not yet** — any dimension at 1, or two or more at 2. Phase marked 🔁 with a specific
  remediation list; re-evaluated when the list is done.

Scores, outcome and gaps are recorded in the evaluation log.

---

## Phase 0 — Foundations ✅

What exists, as evidence of the baseline:

- [x] Tokenizer from scratch — `learning/bpe_base.py`, review notes in `project.md`
- [x] Attention and GPT experiments — `learning/attention.py`, `learning/v2.py`, `learning/v3.py`
- [x] GPT-2 124M training package — `src/gpt2/`, fresh A100 run (`scripts/train_a100.py`)
- [x] KV-cached batched inference, nucleus sampling, static-shape decode with a CUDA graph —
      `src/gpt2/inference/`
- [x] Continuous-batching scheduler with per-row KV cache — `src/gpt2/serving/`
- [x] Serving theory, written up — `docs/request-lifecycle.md`, `docs/forward-and-sampling.md`

---

## Phase 1 — GPU performance, hands-on 🟡

**Why first:** everything else rests on it. The roofline model — every operation is
limited either by memory bandwidth or by compute, and arithmetic intensity (FLOPs per byte
moved) says which — explains FlashAttention, quantization, batching and most of serving.

**Hardware:** one NVIDIA GPU (the A100 machine). Triton and profiling do not run on macOS.

### Concepts

- [x] Memory hierarchy: HBM vs SRAM vs registers *(theory — forward-and-sampling.md Step 0)*
- [x] Threads, warps, CTAs, grids, occupancy *(theory — Step 0)*
- [x] Kernel launch, streams, synchronisation, CUDA graphs *(theory — Step 1)*
- [x] Triton vs PyTorch vs CUDA C++; `torch.compile` *(theory — Step 1)*
- [x] Tiling, online softmax, FlashAttention vs FlashDecoding *(theory — Step 4)*
- [ ] Roofline model and arithmetic intensity — applied to real shapes
- [ ] Coalescing and memory bandwidth — measured, not just described
- [ ] Tensor cores and mixed precision (fp16, bf16, fp8): what changes in the roofline
- [ ] Profiling: `torch.profiler` and Nsight Systems timelines, read fluently
- [ ] Kernel fusion: when it pays, measured

### Milestones

- [ ] **P1.1 Roofline analysis of GPT-2.** For decode attention, an MLP matmul and
  `lm_head`, at batch sizes 1, 8, 64 and prefill length 512: compute FLOPs, bytes moved and
  arithmetic intensity; predict memory- or compute-bound on an A100 and the expected time.
  *Accept when:* `docs/roofline.md` has the prediction table, the measured times, and an
  explanation of every gap larger than 2×.
- [ ] **P1.2 Profile the serving scheduler.** Trace `src/gpt2/serving/scheduler.py` with
  `torch.profiler` and Nsight Systems; identify launches, syncs and GPU idle time; apply the
  one-`tolist()`-per-step fix; measure again.
  *Accept when:* before/after tokens/s and trace findings are in `docs/benchmarks/`, with a short write-up
  of what the timeline showed.
- [ ] **P1.3 Triton tutorials.** Vector add, fused softmax, matmul, fused attention — run,
  modified, and benchmarked against PyTorch.
  *Accept when:* each has a short note (what was fused, measured speed-up, one surprise).
- [ ] **P1.4 Paging in PyTorch.** Block pool, free list, block table, `slot_mapping` writes,
  gather-then-attend attention. *(Started first — order changed 2026-10-17.)*
  - [x] **a** `BlockAllocator`: free list, allocate, free — unit-tested, including exhaustion
  - [x] **b** `PagedKVCache` write path: `slot_mapping`, scatter into the pool — tested
  - [x] **c** read path: gather by block table — tested against `RowKVCache` contents
  - [x] **d** scheduler integration: admit by free blocks + watermark, release on retire
        (reserve-at-admission policy)
  - [~] **e** end to end: identical greedy outputs ✓; memory and tokens/s measured on MPS ✓;
        **A100 run still to do**
  - [ ] **f — yours** on-demand growth: one block when `row_len % block_size == 0`, with the
        watermark keeping headroom; raise a clear error if growth fails (preemption comes
        later). Tested: greedy outputs still identical; blocks in use track actual lengths.
  - [ ] **g — yours** explain-back: walk the judge through the write path, the gather, the
        scratch block and the admission rules, without notes.

  *Judge's note (2026-10-17):* a–e were implemented by Claude at your request, so they earn
  no Building credit on their own. P1.4 Building is judged on **f** (your code) and
  Understanding on **g**. Code: `src/gpt2/serving/{block_allocator,paged_cache,paged_scheduler}.py`,
  tests: `tests/gpt2/test_{block_allocator,paged_cache,paged_scheduler}.py`.
  *Accept when:* tests show greedy outputs identical to the current `RowKVCache` path, and
  memory use, concurrency and tokens/s are measured against it (summary in `docs/benchmarks/paging.md`).
- [ ] **P1.5 Triton kernels.** `reshape_and_cache`, then a paged decode-attention kernel
  walking the block table with online softmax.
  *Accept when:* `assert_close` against the gather version passes, and a benchmark across
  context lengths compares both to the bandwidth roofline.

### Reading

- Horace He, *Making Deep Learning Go Brrrr From First Principles*
- NVIDIA, *GPU Performance Background User's Guide* (deep-learning performance docs)
- GPU MODE lectures; Triton's official tutorials
- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj) — reference, optional
- Papers: FlashAttention (1, 2, 3)

### Sample evaluation questions

- Derive the arithmetic intensity of an `(M×K)·(K×N)` fp16 matmul. At what `M` does an
  A100 become compute-bound for `K = N = 768`?
- Why does decode attention stay memory-bound as batch size grows, while the MLP becomes
  compute-bound?
- A kernel needs 96 KB of shared memory per CTA. How many CTAs fit on an A100 SM, and what
  does that do to latency hiding?
- What does FlashAttention change about HBM traffic, and why is the result exact?
- Practical: predict the time of a given op on an A100 within 2×, then measure it.

---

## Phase 2 — Distributed training ⬜

**Why:** the largest gap, and the core of large-model infrastructure.

**Hardware:** at least 2 GPUs for most milestones; an 8-GPU node (rented) for the parallelism
comparisons.

### Concepts

- [ ] Collectives — all-reduce, all-gather, reduce-scatter — and their cost models
- [ ] Data parallelism (DDP) and gradient bucketing / overlap
- [ ] ZeRO stages 1–3 and FSDP: sharding optimizer state, gradients, parameters
- [ ] Tensor parallelism (Megatron-style column/row splits), sequence parallelism
- [ ] Pipeline parallelism and the pipeline bubble
- [ ] Expert parallelism for MoE
- [ ] Interconnects — NVLink vs InfiniBand — and why they decide which parallelism goes where
- [ ] Mixed precision training; activation checkpointing
- [ ] Checkpointing, fault tolerance, restarts at scale

### Milestones

- [ ] **P2.1 Collective micro-benchmarks.** Measure all-reduce bandwidth against message
  size on 2+ GPUs. *Accept when:* measured curve vs the ring all-reduce cost model,
  `2(N−1)/N × bytes / bandwidth`, with the gap explained.
- [ ] **P2.2 DDP GPT-2.** *Accept when:* loss matches the single-GPU run; scaling
  efficiency measured at 1, 2 and (if available) 8 GPUs.
- [ ] **P2.3 FSDP / ZeRO.** *Accept when:* per-GPU memory measured for each sharding level
  and matched to a predicted memory budget.
- [ ] **P2.4 Hand-written tensor parallelism.** MLP and attention split across GPUs.
  *Accept when:* outputs match the unsplit model; communication vs compute time measured.
- [ ] **P2.5 Memory and cost budget for a 7B training run.** Paper exercise: parameters,
  gradients, optimizer state, activations; choose a parallelism layout for a given cluster.
  *Accept when:* a written plan with every number derived.

### Reading

- *How to Scale Your Model* (Google DeepMind scaling book)
- Hugging Face, *The Ultra-Scale Playbook: Training LLMs on GPU Clusters*
- Stas Bekman, *Machine Learning Engineering* (open book)
- Papers: Megatron-LM, ZeRO, PyTorch FSDP, GPipe

### Sample evaluation questions

- How much memory does Adam mixed-precision training need per parameter? What does each
  ZeRO stage remove?
- Why is tensor parallelism usually kept within a node and data parallelism across nodes?
- Derive the pipeline bubble fraction for `p` stages and `m` micro-batches.
- Practical: estimate the step time of a given model and cluster, split into compute and
  communication.

---

## Phase 3 — Serving at systems scale ⬜

**Why:** finish the serving layer — from a single-GPU engine to a fleet.

### Concepts

- [ ] Chunked prefill and mixed prefill/decode batches
- [ ] Prefix caching: block hashing vs radix tree, eviction, routing
- [ ] Async scheduling and CPU overhead
- [ ] Quantization: fp8/int8 weights and KV cache — effect on memory-bound decode
- [ ] Speculative decoding: when it helps, and the acceptance-rate maths
- [ ] Tensor-parallel inference
- [ ] Prefill/decode disaggregation and KV transfer
- [ ] SLOs (TTFT, TPOT), goodput, and the latency/throughput trade-off

### Milestones

- [ ] **P3.1 Flat token stream.** Chunked prefill with `cu_seqlens` and a token budget.
  *Accept when:* outputs match the reference; TTFT/TPOT measured under a mixed workload.
- [ ] **P3.2 Prefix caching.** Chained block hashes, refcounts, tail-first LRU eviction.
  *Accept when:* hit rate and prefill tokens saved measured on a shared-prompt workload.
- [ ] **P3.3 Async scheduling.** *Accept when:* throughput gain measured against the P1.2
  baseline, with traces.
- [ ] **P3.4 One of quantization or speculative decoding**, implemented and measured.
- [ ] **P3.5 Read vLLM V1.** Scheduler, KV-cache manager, model runner.
  *Accept when:* a write-up mapping vLLM's components onto `request-lifecycle.md`, noting
  where it differs.

### Reading

- Papers: Orca, vLLM/PagedAttention, SGLang, DistServe, Sarathi-Serve (chunked prefill),
  speculative decoding (Leviathan et al.)
- vLLM V1 source; `request-lifecycle.md` and `forward-and-sampling.md` as the map

### Sample evaluation questions

- Given a model, GPU and traffic mix, choose block size, token budget and `max_num_seqs`,
  and justify each.
- When does prefill/decode disaggregation pay off, and what does it cost?
- What acceptance rate does speculative decoding need to beat plain decode?

---

## Phase 4 — Post-training and RL infrastructure ⬜

**Why:** where every layer meets. RL for LLMs alternates generation (inference: rollouts)
and learning (distributed training) on the same model — weights must sync from trainer to
engine, rollouts are long and uneven, and GPUs are shared between phases.

### Concepts

- [ ] SFT data and loss masking
- [ ] Preference optimisation: RLHF (PPO), DPO
- [ ] RL with verifiable rewards; GRPO
- [ ] Rollout generation at scale — why it dominates RL step time
- [ ] Weight synchronisation trainer → inference engine
- [ ] Colocated vs disaggregated RL; on- vs off-policy staleness

### Milestones

- [ ] **P4.1 SFT** of the GPT-2 on a small instruction set. *Accept when:* before/after
  samples and loss curves, loss masking tested.
- [ ] **P4.2 DPO.** *Accept when:* preference accuracy measured before and after.
- [ ] **P4.3 RL loop** (GRPO-style) using this repo's serving engine for rollouts.
  *Accept when:* reward improves on a verifiable task; rollout vs training time measured
  and explained.
- [ ] **P4.4 Read an RL framework** (verl or OpenRLHF). *Accept when:* a write-up of how it
  places rollout and training on GPUs and syncs weights.

### Reading

- Papers: InstructGPT, DPO, DeepSeekMath (GRPO), HybridFlow (verl)
- OpenRLHF and verl source

### Sample evaluation questions

- Why does rollout generation dominate an RL step, and what are three ways to shrink it?
- Trace one RL iteration through memory: where do weights, optimizer state and KV cache live?

---

## Phase 5 — Platform and operations ⬜

**Why:** the layer of day-to-day production work. With Phases 1–4 done, capacity,
scheduling and cost decisions become first-principles answers instead of conventions.

### Concepts

- [ ] Capacity planning from SLOs and traffic
- [ ] Autoscaling signals and policies for GPU fleets
- [ ] Multi-tenancy, isolation, fairness, admission control
- [ ] Observability: metrics that matter for training and serving
- [ ] Failure modes: GPU/node failures, stragglers, cascading overload
- [ ] Cost: utilisation, batching efficiency, hardware choice

### Milestones

- [ ] **P5.1 Capacity plan.** Given a model, traffic profile and SLOs, derive GPU count,
  configuration and cost. *Accept when:* every number is derived and sanity-checked.
- [ ] **P5.2 Design doc** for a multi-tenant LLM serving platform: routing, autoscaling,
  isolation, observability, failure handling. *Accept when:* reviewed against a mock design
  review.
- [ ] **P5.3 Load test** of this repo's engine under realistic traffic, with an
  autoscaling policy evaluated in simulation.

### Reading

- Google, *Site Reliability Engineering* (selected chapters)
- Production serving stacks built on vLLM / SGLang (routing, autoscaling designs)

### Sample evaluation questions

- Design review: a 70B model at a given QPS with p99 TTFT and TPOT targets — size it,
  configure it, and say what breaks first.

---

## Progress log

| date | phase | entry |
|---|---|---|
| 2026-10-11 | 0 | Serving theory deep-dive complete: request lifecycle (scheduling, paging, prefix caching incl. block hashing and radix tree, CoW, refcounts, concurrency, preemption) and one-iteration GPU walkthrough (launch, CUDA graphs, tiles, online softmax, FA/FD, sampling, async). Written up in two docs. |
| 2026-10-11 | 1 | Plan created. Phase 1 started; five concepts already covered in theory (see Phase 1 checklist). Next: P1.1 roofline analysis. |
| 2026-10-17 | 1 | Order changed by choice: **P1.4 (paging in PyTorch) first**, then P1.1. Judge's note: recommended predict-then-measure first; agreed to build first, with P1.4 measured against the `RowKVCache` baseline so measurement is not skipped. P1.4 split into sub-steps a–e (see Phase 1). |
| 2026-10-17 | 1 | P1.4 a–e implemented by Claude at your request; full suite 136 passed (CPU + MPS). MPS benchmark at 124M: same 289 MB of KV holds **32 requests vs 4** — but throughput unchanged (199 tok/s). Open question recorded in `docs/benchmarks/paging.md` for P1.1/P1.2. Remaining for P1.4: A100 run, **f** (on-demand growth, yours), **g** (explain-back). |
| 2026-10-17 | 1 | Local spin-up on real weights (`runs/keep/gpt2_124M_step013147.pt`, MPS): row and paged schedulers give **identical** greedy text; 6 prompts in 1.29 s paged vs 2.22 s row. Chat UI added (`scripts/chat_server.py`); 4 simultaneous requests ran as one batch in the same time as one. Replies are poor because the model is a base model, not chat-tuned — the motivation for P4.1 (SFT). |

## Evaluation log

| date | phase | Understanding | Building | Measurement | Communication | outcome | carried gaps |
|---|---|---|---|---|---|---|---|
| 2026-10-11 | 0 (baseline) | 3 | 3 | 2 | 3 | baseline — not a gate | measurement: theory is ahead of practice; Phase 1 closes this |

**Baseline notes (Phase 0).** Understanding of serving mechanisms is solid and often deep
— questions went to the mechanism behind each answer (why `int()` syncs, why tiles never
mix requests, why gather indexing copies). Building is real: a trained model, KV-cached
inference with a CUDA graph, and a continuous-batching scheduler. The gap is
**measurement** — little has yet been predicted and then measured — which is exactly what
Phase 1 is designed to fix.
