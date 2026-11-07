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
      `src/gpt2/inference/static_batch/`
- [x] Continuous-batching scheduler with per-row KV cache — `src/gpt2/inference/continuous_batch/`
- [x] Serving theory, written up — `docs/request-lifecycle.md`, `docs/forward-and-sampling.md`

---

## Phase 1 — GPU performance, hands-on 🟡

**Why first:** everything else rests on it. The roofline model — every operation is
limited either by memory bandwidth or by compute, and arithmetic intensity (FLOPs per byte
moved) says which — explains FlashAttention, quantization, batching and most of serving.

**Hardware:** one NVIDIA GPU — an **A100-SXM4-40GB on Colab** (Ampere: fp16/bf16 tensor cores ~312 TFLOP/s dense, HBM2 ~1.55 TB/s, ridge ≈ 200 FLOPs/byte; FlashAttention kernels available). A T4 is the fallback (fp16 ~65 TFLOP/s, ~320 GB/s, no bf16). Every roofline prediction uses the specs of the GPU it is measured on. Triton and profiling do not run on macOS.

### Concepts

- [x] Memory hierarchy: HBM vs SRAM vs registers *(theory — forward-and-sampling.md Step 0)*
- [x] Threads, warps, CTAs, grids, occupancy *(theory — Step 0)*
- [x] Kernel launch, streams, synchronisation, CUDA graphs *(theory — Step 1)*
- [x] Triton vs PyTorch vs CUDA C++; `torch.compile` *(theory — Step 1)*
- [x] Tiling, online softmax, FlashAttention vs FlashDecoding *(theory — Step 4)*
- [x] Roofline model and arithmetic intensity — applied to real shapes *(derived for GPT-2 decode in session, 2026-10-18; measurement is P1.1)*
- [ ] Coalescing and memory bandwidth — measured, not just described
- [ ] Tensor cores and mixed precision (fp16, bf16, fp8): what changes in the roofline
- [ ] Profiling: `torch.profiler` and Nsight Systems timelines, read fluently
- [ ] Kernel fusion: when it pays, measured

### Milestones

- [ ] **P1.1 Roofline analysis of GPT-2.** For decode attention, an MLP matmul and
  `lm_head`, at batch sizes 1, 8, 64 and prefill length 512: compute FLOPs, bytes moved and
  arithmetic intensity; predict memory- or compute-bound on the GPU used (T4 or A100) and the expected time.
  *Accept when:* `docs/roofline.md` has the prediction table, the measured times, and an
  explanation of every gap larger than 2×.
- [ ] **P1.2 Profile the serving scheduler.** Trace `src/gpt2/inference/continuous_batch/scheduler.py` with
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
  - [x] **e** end to end: identical greedy outputs ✓; memory and tokens/s measured on MPS ✓
        and on the A100 ✓ (`docs/benchmarks/paging.md`)
  - [ ] **f — yours** on-demand growth: one block when `row_len % block_size == 0`, with the
        watermark keeping headroom; raise a clear error if growth fails (preemption comes
        later). Tested: greedy outputs still identical; blocks in use track actual lengths.
  - [ ] **g — yours** explain-back: walk the judge through the write path, the gather, the
        scratch block and the admission rules, without notes.

  *Judge's note (2026-10-17):* a–e were implemented by Claude at your request, so they earn
  no Building credit on their own. P1.4 Building is judged on **f** (your code) and
  Understanding on **g**. Code: `src/gpt2/inference/continuous_batch/{block_allocator,paged_cache,paged_scheduler}.py`,
  tests: `tests/gpt2/test_{block_allocator,paged_cache,paged_scheduler}.py`.
  *Accept when:* tests show greedy outputs identical to the current `RowKVCache` path, and
  memory use, concurrency and tokens/s are measured against it (summary in `docs/benchmarks/paging.md`).
- [ ] **P1.5 Triton kernels.** `reshape_and_cache`, then a paged decode-attention kernel
  walking the block table with online softmax.
  *Accept when:* `assert_close` against the gather version passes, and a benchmark across
  context lengths compares both to the bandwidth roofline.

  *Decided 2026-10-18: written by you, reviewed by Claude* — so it earns Building credit
  (unlike P1.4 a–e). Suggested order:
  - [x] **a — decode kernel, one program per (row, head).** Walk the block table up to the
        real length, online softmax, no gather and no mask. Test with `assert_close`
        against `PagedKVCache._gather` + SDPA, including partial last blocks and rows of
        different lengths.
  - [ ] **b — wire it in** for decode only (prefill keeps the gather path); greedy outputs
        identical to the row engine.
  - [ ] **c — measure on the A100:** kernel time vs context length against the bandwidth
        roofline, and the Q5 numbers (attention and gather share, peak memory, tokens/s)
        compared with your Q5 prediction.
  - [ ] **d — `reshape_and_cache`** (the write side) as a Triton kernel, replacing `index_copy_`.
  - [x] **e — decode CPU overhead** *(code by Claude at your request; measured by you)*:
        decisions from host mirrors (`*_cpu` / `*_gpu` naming), syncs ~66 → 1 per step,
        decode captured in a CUDA graph. A100, 32 rows, gather path: **10.32 → 9.38 → 3.85
        ms per decode step**; tok/s ×2.1 at 128 new tokens, ×1.5 at 32 (prefill unchanged).

- [ ] **P1.7 Prefill.** Measured 2026-11-07: ~0.31 s for 32 prompts (~10 ms per prompt)
  against ~0.2 ms of real work per prompt (weights 0.17 ms + 72 × 250M FLOPs 0.06 ms) —
  ~98% overhead. Prefill is ~half the run at 32 new tokens, so after P1.5 it is the
  bottleneck. Measure every step the same way: prefill = run time − decode steps × ms per step.
  - [ ] **a — prompt-only attention.** Today prefill gathers the row to the full
        W = 1024 and attends with an explicit mask (memory-efficient kernel over 1,024
        columns; `paged_cache.py` `update(row=…)` → `_gather`, `gpt.py` builds the tril mask).
        A fresh request has no history: write its K/V into the pool, then attend over the
        prompt's own K/V with `is_causal=True` (Flash). *Accept when:* identical greedy text
        to the row engine; prefill ms per prompt measured before/after.
  - [ ] **b — batched prefill.** Admit up to N waiting requests in one forward, under a
        token budget. First padded to the longest prompt (simple), then packed with
        `cu_seqlens` (varlen: no padding). *Accept when:* identical text; prefill time vs the
        roofline estimate (~2 ms compute-bound for 32 prompts ≈ 2,300 tokens); tok/s at 32
        and 128 new tokens against your prediction.
  - Prefill and decode stay separate phases within a step here. Mixing them in one batch
    (chunked prefill, P3.1) and running them on different GPUs (disaggregation, Phase 3)
    are the serving-scale follow-ups.

- [ ] **P1.6 Pretrain GPT-2 124M properly on the A100, and measure its quality by
  log-likelihood.** Fresh run with `scripts/train_a100.py` (GPT-2 reference recipe:
  524K-token batches, 2.5B tokens, ~6 h at ~116K tok/s). The current checkpoint saw
  ~215M tokens (val loss 4.17) — ~2% of GPT-2's training. Decided 2026-10-17: fresh run
  (not a continuation); HellaSwag evaluated inside the training loop every 500 steps.
  **Code (a–c, e–g) written by Claude at your request; your part is evaluation itself:**
  **d** (predictions), **h** (running and analysing the results) and an explain-back of
  what each metric measures and why. P1.6 is judged on those — Understanding,
  Measurement and Communication — not on the tooling.

  *Locally, on the Mac — no GPU time:*
  - [x] **a — likelihood eval.** `src/gpt2/evaluation/likelihood.py`: validation NLL on a
        **fixed** window of the val shard (same tokens for every model), reported as
        nats/token, perplexity `exp(nll)`, bits/token `nll / ln 2`.
        *Accept:* tests show perplexity = exp(nll) and identical results on repeat runs.
  - [x] **b — HellaSwag scorer.** `src/gpt2/evaluation/hellaswag.py`: download and cache
        `hellaswag_val.jsonl` (10,042 examples, from the `rowanz/hellaswag` GitHub repo);
        render each example as context + 4 endings with a mask over the ending tokens;
        score each ending's log-likelihood — **summed** (`acc`) and **per-token mean**
        (`acc_norm`); batch several examples with right-padding.
        *Accept:* a test shows the masked score equals a hand computation on a tiny model,
        and a test shows padding/batching does not change any score.
  - [x] **c — `scripts/eval_checkpoint.py`.** Any checkpoint, or `--hf gpt2` for OpenAI's
        weights → JSON with val loss, perplexity, bits/token, HellaSwag `acc` and
        `acc_norm`. Run the **baselines**: your step-13,147 checkpoint and OpenAI GPT-2.
        *Accept:* both in `docs/benchmarks/training.md`, with the device and val window stated.
  - [ ] **d — predictions, written before training starts.** Final val loss (F4),
        final HellaSwag `acc_norm`, MFU from tokens/s (F1). Dated, in `training.md`.

  *On Colab:*
  - [x] **e — HellaSwag in the loop.** Every 500 steps inside `train()`, logged as a
        `hella` record next to `val` in `log.txt` (`read_log` must accept it).
        *Accept:* the overhead per eval is measured and stated.
  - [x] **f — data for Colab.** `prepare_data.py` gains `--out-dir` and `--max-shards`
        (stop after 251 = 1 val + 250 train; streaming, so the full 10B tokens are never
        downloaded). Shards built on local disk for speed, copied to Drive for persistence.
  - [ ] **g — training notebook.** Mount Drive; clone/install; data from Drive (or build
        it); **a 20-step smoke run first**; then the full run with `run_dir` on Drive,
        `resume_from='latest'`, `keep_last_n_checkpoints=2`.
        *Accept:* you deliberately interrupt the smoke run and show it resumes from its
        checkpoint, not from step 0.
  - [ ] **h — results.** Val-loss and HellaSwag curves; a final table vs the baselines
        and your predictions; MFU explained with the roofline; chat samples before/after;
        the new checkpoint loaded in the chat UI. (Chat *behaviour* still needs SFT — P4.1.)

  *Pitfalls to check yourself against:* shift by one (logits at position *t* predict
  token *t+1*); the mask must line up with the shifted targets; HellaSwag endings are
  encoded with a leading space; `model.eval()` and no-grad for evaluation; bf16 autocast
  on CUDA; the model's vocab is padded to 50,304 while targets stay below 50,257;
  OpenAI's GPT-2 loads through the existing `GPT.from_pretrained`.

### Deep-dive questions

`docs/study/phase1-questions.md` — 26 questions across the roofline, arithmetic
intensity, the KV cache, this engine's measurements, profile reading and training.
Answer in writing with the arithmetic shown; each section is judged when marked ready.

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
- [ ] **P3.3 Async scheduling.** Overlap the per-step token copy and CPU bookkeeping with
  the next GPU step (SGLang / vLLM V1), or multi-step decode (k steps per sync). Also
  per-request sampling (temperature as a per-row tensor inside the graph).
  *Accept when:* throughput gain measured against the P1.5e baseline (3.85 ms/step), with traces.
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
| 2026-10-17 | 1 | Code pushed (4 commits) with `notebooks/colab_gpu_profile.ipynb`. First Colab session was a **Tesla T4**; switched to an **A100-SXM4-40GB** — the plan's target. Rooflines use the measured GPU's specs. |
| 2026-10-17 | 1 | **A100 results in.** Paged at the same 144 MB of KV: 2.3× (16 seqs) and **3.3×** (32 seqs) the row cache's throughput — unlike MPS (no gain). Profile (prefill-heavy, 8 new tokens): prefill 81% of wall; GPU 85% idle under the profiler; attention (memory-efficient kernel) 41% of GPU time, gather 12%. Five analysis questions recorded in `docs/benchmarks/paging.md` — yours, judged under P1.2. |
| 2026-10-17 | 1 | **Analysis attempt 1 reviewed — revise and resubmit.** Strong: #5 (shrinking GPU work raises the idle share — Amdahl), #4 (GPU waits on CPU). Issues: #1 compared the wrong pair (3.3× is paged vs row on the A100, not A100 vs Mac; no CPU/MPS swapping); #2 missed that prefill wall (648 ms) ≫ its GPU time (92 ms), i.e. engine overhead, not KV volume; #3 memory-efficient attention is a fallback forced by the explicit mask, not a choice; #4 transfers are 1.3%, and two doc changes were not named. To resubmit: revised answers in the doc citing Part A numbers; Part A on MPS for #1. |
| 2026-10-17 | 1 | Roofline concepts worked through in session (bandwidth vs capacity, FLOPs per byte, ridge ≈ 200, batch ≈ intensity for weight matmuls, KV bytes do not batch). Question bank added (`docs/study/phase1-questions.md`). Chat replies `email@example.com`: name-like prompts match FineWeb's anonymised contact text, and multi-turn history feeds the bad reply back. Root cause of weak replies: the checkpoint saw ~215M tokens (val 4.17). Added **P1.6**: proper 2.5B-token pretraining on the A100, then SFT (P4.1) for chat behaviour. |
| 2026-10-17 | 1 | P1.6 planned (later the same day: code by Claude, evaluation by you): quality measured by log-likelihood (val NLL / perplexity / bits per token on a fixed val window; HellaSwag `acc` and `acc_norm`), against your old checkpoint and OpenAI GPT-2. Decisions: **fresh run**; **HellaSwag in the loop every 500 steps**. **You build P1.6** (a–h) as a learning exercise; the judge reviews each step. |
| 2026-10-17 | 1 | P1.6 tooling built (a, b, c, e, f; notebook g ready): eval module, `eval_checkpoint.py`, HellaSwag + throughput logged in the loop, streamed data prep, Colab notebook with a resume smoke test. 151 tests pass. **Baselines (MPS, fixed 1M-token val window):** old checkpoint val nll 4.268 (ppl 71.4), HellaSwag acc_norm **25.2% — chance**; OpenAI GPT-2 nll 3.266 (ppl 26.2), acc_norm **29.55%**, matching the published figure — validates the scorer. Next (yours): **d** — predictions in `docs/benchmarks/training.md`, before the run. |
| 2026-10-18 | 1 | **Paging analysis, attempt 2 reviewed.** Accepted: **Q2** (prefill is engine-bound — batch it), **Q4** (syncs + launches without CUDA graphs). Revise: **Q1** — core insight right (ridge: A100 ≈ 200 rows, Mac ≈ 24 measured), but arithmetic: 1 byte/param used (fp16 = 2, Mac ran fp32 = 4), weights multiplied by rows (read once per step; KV is what grows), FLOPs = 2×params, Mac bandwidth 0.18 GB/ms measured not 0.4, 3.3× misread again, no measurement proposal; **Q3** — the blocker is the explicit attn_mask (causal is supported), the fix is reading only real blocks (not a smaller block size), sync is Q4 not Q3; **Q5** — incomplete, no numbers; restore attempt 1's Amdahl point. Measured M1 Pro: 4.35 TFLOP/s fp32, 4.73 fp16, 181 GB/s. |
| 2026-10-18 | 1 | **Q1 attempt 3:** fixed bytes/param (fp16 2, fp32 4), FLOPs = 2×params, measured Mac bandwidth. Still open: weights multiplied by rows (read once per step — with that fixed, the answer is in the numbers: at 32 rows the Mac's compute 3.5 ms exceeds its 2.7 ms weight read, the A100's 0.025 ms is negligible beside 0.16 ms); 3.3× still misread; no measurement proposal; summary balance point stale (Mac fp32 ≈ 48 rows, A100 ≈ 200). Q3, Q5 still to revise. Roofline concepts worked through: KV caps decode intensity at 250M/(L·36KB) ≈ 6.6 at L=1024 (~40K tok/s ceiling on the A100), ≈ 68 with real blocks at L≈100. |
| 2026-10-18 | 1 | **Roofline concept demonstrated** in your own words: weight bytes read once per step while FLOPs scale with rows; KV bytes scale with rows and become the bottleneck; ~200 rows to reach the A100 ridge with weights alone, but at 1024 context memory caps decode at ~40 tokens/ms using ~3% of compute; levers are bandwidth or fewer KV bytes (fp8, GQA, real blocks). Concept ticked; CPU overhead noted as the third bottleneck. |
| 2026-10-18 | 1 | **Q1 attempt 4 / Q5 attempt 2 reviewed.** Q1: direction right (per-row cost dominates on the Mac), but '40-row limit' treats a rate (tokens/ms) as a batch cap — reframe as step time = fixed + R × per-row (A100 ≈ 0.075 ms/row, Mac fp32 ≈ 1.25 ms/row at 1024 width, 3 passes); 3.3× still misread; measurement still missing. Q5: real length ≠ block size; compute allows 312G/250M ≈ 1,248 tokens/ms (200 is FLOPs/byte, not tokens) — still memory-bound at 405 tokens/ms; the table rows (attention 41%, gather 12%, +71 MB, tokens/s) not yet addressed. Q3 unchanged. |
| 2026-10-18 | 1 | **Q1 attempt 5:** A100 half improved — capacity chain (144 MB → 256 blocks → ~36 requests) and CPU overhead now present. Still recurring: '8 → 32' (it is 4 → 32, so 8× not 4×), capacity (144 MB) compared with bandwidth (1.5 GB/ms), '40 / 3.8 rows' as a batch cap (it is tokens/ms), the paged engine described as reading less (it gathers full width, 3 passes — reads more), measurement proposal missing for the third review. Gave a fill-in skeleton (capacity → step time = fixed + R × per-row → measurement). |
| 2026-10-18 | 1 | **Q1 accepted on substance** (re-read against the stated expectations): capacity 4 → ~32 requests in the same memory ✓; Mac memory-limited so more rows don't help, A100 has headroom and is held back by CPU launches ✓. To finish: one measurement sentence; fix the leftover '8 → 32, 4×' (4 → 32, 8×); drop 'reads a lot less data'. Judge's note: earlier reviews over-weighted wording details — expectations are now stated up front as three points. |
| 2026-10-18 | 1 | **Q3/Q5 expectations stated; P1.5 assigned to you.** Q3: the 1,024-wide gather and the explicit mask (not 'causal') explain attention's 41%; `int()` belongs to Q4. Q5: wording to be split into 5a (kernel only, CPU overhead fixed) and 5b (overhead also removed); L ≈ 100 is the sequence length, not block size; 200 is FLOPs/byte, compute allows ~1,248 tokens/ms. Gather peak explained: R × 3 MB per layer (16 → 32 rows measured +49 MB vs 48 predicted). |
| 2026-10-18 | 1 | **Q1 (a/b/c) and Q5 reviewed.** Q1a ✓. Q1b: byte-per-step arithmetic right; open — compute time divided bytes by FLOP/s (use FLOPs), today's paged engine still reads full width (not real tokens), the A100 conclusion (fixed CPU ~8 ms ≫ 0.3–1 ms GPU) not stated, Mac ran fp32; Q1c missing. Q5: diagnosis right but no predictions yet — 5a numbers and 5b ceiling (32 tokens ÷ 0.24 ms, already computed in Q1b) still to write. |
| 2026-10-18 | 1 | **Q5 accepted.** 5a: tokens/s rises only a little because the CPU side is unchanged (Amdahl) ✓; attention ÷ (9 × 3) ✓; gather removed ✓. 5b: 32 ÷ 0.24 ms ≈ 133 tokens/ms vs compute ~1,250 → memory-bound ✓. Polish: the copy is 32 × 1,024 × 3 KB ≈ 96 MB per layer (peak 638 → ~540 MB), index_copy stays at 4%. Remaining in paging.md: Q1 fixes and Q1c, Q2–Q4. |
| 2026-10-18 | 1 | **Q1 accepted (a, b, c).** b: A100 step ≈ 8 ms CPU + 0.26→0.94 ms GPU, so 8× rows barely lengthens the step; Mac per-row dominated ✓. c: time decode at R = 2…32 — flat on A100, linear on Mac ✓. Nuance noted: the model predicts ~7×; measured 3.3× because prefill (one request per forward) is the rest — links to Q2. Remaining in paging.md: Q2–Q4. |
| 2026-10-18 | 1 | **Q2 accepted.** a: prefill:decode 4.3:1 → ~50% at 32 new tokens, so mainly the engine ✓. b: 60–100 rows ≈ 60–100 FLOPs/byte < ridge ~200 ✓; fixed CPU cost per forward amortised by more rows ✓. c: batch prefill, prefill share drops ✓. Remaining in paging.md: Q3, Q4. |
| 2026-10-18 | 1 | **Q3 and Q4 accepted on substance — paging analysis Q1–Q5 complete.** Q3: full 1,024 width vs ~112 real ✓, `attn_mask` rules out Flash ✓, read only real blocks ✓; polish: state the ratio (~9×), and why block size alone doesn't help (the gather pads to W regardless), drop the old answer line. Q4: launch overhead (~8 ms vs ~0.3 ms GPU) ✓, CUDA graph + remove `int()` syncs ✓, profiler inflates CPU time → compare with/without ✓; polish: name the sync site (`scheduler.py:115`, `int(tok[row])` per row), Triton cuts GPU work, not idle time. |
| 2026-11-07 | 1 | **P1.5a accepted — your kernel.** `src/gpt2/inference/continuous_batch/paged_attention.py`, written by you from the reviewed example; on Colab (CUDA) all 10 tests in `tests/gpt2/test_paged_attention.py` pass: matches gather + SDPA in fp32 (1e-4) and fp16 (2e-3), every edge length (1, 15, 16, 17, 100, 1024), never reads past the real length, shape/dtype kept. Draft review caught: `tl.range` vs `tl.arange`, `tl.maximum` vs `tl.max` (reduce vs elementwise), store inside the loop, a misspelled `running_max` that would have silently zeroed `rescale`. Also done (by Claude, at your request): decode syncs ~66 → 1 per step via host mirrors, `_cpu`/`_gpu` naming, CUDA-graph decode — graph test not yet run on CUDA. Next: b (wire in), c (measure). |
| 2026-11-07 | 1 | **CUDA-graph decode verified on CUDA:** `test_graph_decode_matches_eager` passes (12 prompts through 2 rows — replays pick up rows admitted between them; every block returned) and `test_graph_refuses_a_different_sampler` passes. Correctness only — speed not yet measured (P1.5c). |
| 2026-11-07 | 1 | **P1.5c part 1 measured (A100, paged 32 rows, gather path).** ms per decode step (Δ between 32 and 128 new tokens): before 10.32 → fix 1 (1 sync) 9.38 → fix 1 + CUDA graph 3.85. tok/s at 128 new: 2,481 → 2,725 → 5,126 (2.1×); at 32 new: 1,550 → 2,384 (1.5×) — Amdahl: prefill unchanged. Syncs cost ~0.9 ms (~14 µs each); launches ~5.5 ms. 3.85 ms sits above the ~2.6 ms gather-path floor → step is now GPU-bound; next lever is wiring the kernel in (P1.5b). Caveats: row line's ms/step used the paged step count (row has 4× more steps → ~7.8 ms); peak MB not comparable across processes (the after-run also holds the graph engine's pool). |
| 2026-11-07 | 1 | **Prefill plan agreed → P1.7** (a: prompt-only causal attention instead of the 1,024-wide gather; b: batched prefill, padded then varlen). Clarified: the Triton kernel covers decode only — prefill still gathers to W. Chunked prefill, disaggregation, prefix caching stay in Phase 3 (P3.1, P3.2); overlap scheduling and per-row sampling added to P3.3. Order: P1.5b → P1.7a → P1.7b. |

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
