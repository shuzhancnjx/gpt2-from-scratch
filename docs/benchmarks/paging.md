# P1.4 — paged vs row KV cache

Script: [`scripts/bench_paging.py`](../../scripts/bench_paging.py). Both engines get the
**same KV memory**: `RowKVCache` with 4 full-width rows, and a block pool holding exactly
as many tokens (plus one scratch block) shared by up to `--paged-seqs` requests. Random
weights; 32 prompts of 16–128 tokens; 32 new tokens each; greedy.

Correctness is established separately: `tests/gpt2/test_paged_scheduler.py` checks that
the paged scheduler produces text identical to the uncached greedy reference and to the
row scheduler, across slot counts, block sizes, devices and a deliberately tight pool.

## Results — Apple GPU (MPS), 124M configuration, fp32, W = 1024, block size 16

| engine | max requests resident | KV memory reserved | wall time | throughput |
|---|---|---|---|---|
| row (4 slots) | 4 | 288.0 MB | 5.13 s | 199.4 tok/s |
| paged (`--paged-seqs 16`) | 16 | 289.1 MB | 5.52 s | 185.5 tok/s |
| paged (`--paged-seqs 32`) | **32** | 289.1 MB | 5.16 s | 198.6 tok/s |

Peak allocated memory is not reported on MPS; the A100 run will add it.

## What the numbers say

- **Capacity: the expected win.** In the same 289 MB, paging held all 32 requests at
  once versus 4 for the row cache — memory now scales with each request's actual length
  (here ~6–10 blocks of 16) instead of a full 1024-token row. With 32 requests the cap was
  the workload size, not memory: the pool's 256 usable blocks would fit roughly 35
  requests of this mix.
- **Throughput: no gain.** Eight times the concurrency produced the same tokens/s. That
  is the open question below.

## Open question — to explain in P1.1 / P1.2

> Why did 8× more concurrency give no throughput gain?

Hypotheses to test, not conclusions:

1. The gather copies `max_seqs × W` tokens per layer every decode step — proportional to
   the configured width, not to actual lengths — so per-step cost grows with concurrency.
2. Attention still runs over all `W` columns for every row and masks most of them: the
   rectangle is back (forward-and-sampling.md, "Why not just index in PyTorch?").
3. Prefill runs one request at a time in both engines and may dominate total time at this
   size, hiding any decode-side gain.

Each predicts something measurable: per-step decode time versus `max_seqs` (1, 2), the
prefill share of wall time (3). The roofline (P1.1) and a profiler trace (P1.2) should
separate them. If (1) and (2) dominate, a paged-attention kernel that reads only real
blocks (P1.5) is what turns capacity into throughput.

## Still to run

- [ ] **A100** (`--device cuda`, fp16): same configurations, plus peak allocated memory —
      which includes the per-step gather copy.

Smoke test (tiny model, CPU) — not a performance result, only that the script runs:
row 2128 tok/s vs paged 1731 tok/s, 4 vs 9 resident, 1.0 MB each.
