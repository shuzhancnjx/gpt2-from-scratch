# Phase 1 — deep-dive question bank

Questions in the style of the roofline discussion: each asks you to **derive a number,
predict, then check it** — not to recall a definition. Answer under each question in
your own words, show the arithmetic, and cite measurements where a question asks for
them. Mark a section **ready for review** when done; the judge reviews it against the
same standard as the paging analysis.

Reference numbers (A100-SXM4-40GB): fp16/bf16 tensor cores ~312 TFLOP/s (312 GFLOP/ms),
fp32 CUDA cores ~19.5 TFLOP/s, HBM ~1.55 TB/s (1.5 GB/ms), 40 GB HBM, L2 40 MB, 108 SMs,
PCIe ~25 GB/s. Your Mac: look up your chip's memory bandwidth and GPU throughput and
use those — every roofline uses the specs of the device it describes.

Start with `docs/benchmarks/paging.md` (questions 1–5 there); this bank goes deeper.

---

## A · Hardware and the roofline

**A1.** Capacity vs bandwidth: state both for the A100-40GB. Name one serving decision
limited by each, from this repo's experiments.

**A2.** Derive the ridge point for (a) fp16 tensor cores and (b) fp32 CUDA cores.
Which of GPT-2's operations actually run on tensor cores, and which on CUDA cores?

**A3.** Your Mac: find its bandwidth and compute peak, derive its ridge, and place a
batch-32 decode step's matmuls on its roofline. Does this explain the MPS benchmark?

**A4.** PCIe at ~25 GB/s: how long does it take to copy one decode step's inputs for
32 rows (token ids, positions, slot mapping)? The whole 144 MB KV pool? Use the answer
to explain why a *sync* costs far more than the *copy* it waits for.

**A5.** Why do kernels tile? For one 16-token K/V block, how many times is it used once
loaded into SRAM by (a) a 64-row prefill q_tile, (b) a 1-row decode query? What does
that ratio do to each one's FLOPs per byte?

## B · FLOPs and arithmetic intensity

**B1.** For each of a layer's four matmuls (QKV, attention output, MLP expand, MLP
project) and for `lm_head`: FLOPs, bytes and FLOPs per byte at M = 1, 32 and 512.
Which are memory-bound, which compute-bound, on the A100?

**B2.** For the MLP expand matmul, solve for the M at which it crosses the ridge.
Explain why the exact intensity falls below M as M grows.

**B3.** Derive decode attention's FLOPs and bytes for one row with history L, then for
B rows. Show what happens to its FLOPs per byte as B grows, and why it differs from the
weight matmuls.

**B4.** GELU: FLOPs per byte, then predicted time for a `(32, 3072)` and a `(512, 3072)`
input on the A100. Measure both with `triton.testing.do_bench` and explain any gap.

**B5.** One whole decode step at B = 32: total weight bytes, KV bytes and FLOPs at real
lengths (~100 tokens) and at the engine's width (1024). Predict the step time for each
and compare with Part A's measured ms/step.

## C · The KV cache

**C1.** Generating token 1,025 at batch 1: FLOPs and bytes **without** a cache (recompute
the context) and **with** one. What did the cache trade for what?

**C2.** KV bytes per token for this GPT-2 in fp16. Then for a 7B model with GQA
(32 layers, 8 KV heads, head dim 128). After loading each model's fp16 weights on a
40 GB A100, how many tokens of KV fit — as batch × context?

**C3.** Why does KV traffic not get cheaper per token with batching, when weight traffic
does? What does that imply for serving long contexts?

## D · This engine, measured

**D1.** The paged gather: bytes per decode step at `max_seqs = 32`, W = 1024, versus at
the real lengths. Predict the gather's time on the A100 and on your Mac; compare with
the profile's 14.6 ms of gather/indexing.

**D2.** The A100 peak memory rose 71 MB at 32 sequences. Derive that from the size of
the gathered K/V tensors per layer. Why does it not grow 12× with the layer count?

**D3.** Why does an explicit `attn_mask` rule out PyTorch's FlashAttention kernel? What
would let this engine use FlashAttention (consider `is_causal`, variable-length
attention, and a paged kernel)?

**D4.** Prefill took 648 ms of wall time for 92 ms of GPU time. Estimate the per-prefill
cost of launches (kernels per forward × µs per launch) and syncs. Does it account for
the gap?

**D5.** Why can decode use a CUDA graph easily but prefill cannot? What would it take to
graph prefill?

**D6.** List every CPU-GPU sync per step in `scheduler.py` and `paged_scheduler.py`.
Estimate what each costs, and which one removing would help most.

## E · Reading profiles

**E1.** Why does `torch.profiler` with `record_shapes=True` inflate CPU time? How would
you get undistorted step times (CUDA events, Nsight Systems, a lower-overhead profile)?

**E2.** In the A100 trace, find one decode step. Measure its GPU-busy time and the gaps
between kernels. What fills the gaps?

**E3.** Why can the *sum* of kernel times exceed the GPU's busy span, and why did the
analysis use the union of intervals instead?

## F · Training on the A100 (milestone P1.6)

**F1.** Derive training FLOPs per token ≈ 6 × parameters (forward 2N, backward 4N).
Using `train_a100.py`'s measured ~116,000 tokens/s, compute the run's MFU (model FLOPs
utilisation) against 312 TFLOP/s. What would you expect to limit it?

**F2.** The training script notes that the logits tensor dominates memory. Compute its
size at B = 16 and B = 32 (B × T × 50,304, fp32). Why fp32 here when the model runs in bf16?

**F3.** Gradient accumulation: 32 micro-steps of B = 16 make one 524,288-token step. Why
is the update mathematically the same as one B = 512 step, and what does it cost in
throughput?

**F4.** Predict the final validation loss of the 2.5B-token run from the GPT-2 / nanoGPT
reference results, then compare with what you get. Why does a 16,384-token batch for
215M tokens (the current checkpoint) end near 4.17?
