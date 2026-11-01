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

## Results — NVIDIA A100-SXM4-40GB (Colab), 124M configuration, fp16, W = 1024, block size 16

Same workload; fp16 halves the KV footprint, so both engines get 144 MB.

| engine | max requests resident | KV memory reserved | peak allocated | wall time | throughput | vs row |
|---|---|---|---|---|---|---|
| row (4 slots) | 4 | 144.0 MB | 566 MB | 2.14 s | 478.8 tok/s | — |
| paged (`--paged-seqs 16`) | 16 | 144.6 MB | 589 MB | 0.93 s | 1,102.6 tok/s | **2.3×** |
| paged (`--paged-seqs 32`) | 32 | 144.6 MB | 638 MB | 0.65 s | 1,581.5 tok/s | **3.3×** |

Raw files: `results/a100/bench_paging_cuda*.json`.

**On the A100 the throughput gain appears**: 8× the concurrency gives 3.3×, where the Mac
gave none. Peak memory grows with `max_seqs` (+23 MB at 16, +71 MB at 32) while the KV
reservation does not move — the transient cost of the per-step gather.

## Profile — A100, one paged run, 32 sequences

`scripts/profile_serving.py` Part B: 32 prompts, **8 new tokens each** (so deliberately
prefill-heavy), `torch.profiler` with `record_shapes`. Trace:
`results/a100/paged_cuda_32seqs.json` (open in https://ui.perfetto.dev).

**Where the wall time went** (scheduler phases, synchronised):

| phase | wall | GPU kernel time inside it |
|---|---|---|
| `sched.admit` (prefill, one request at a time) | 648 ms (81%) | 92 ms |
| `sched.decode` | 149 ms (19%) | 32 ms |
| `sched.retire` | 4 ms | 0.2 ms |

Across the profiled window the GPU was **busy 124 ms of 804 ms — 85% idle**.

**What the GPU did** (124 ms of kernel time):

| category | time | share |
|---|---|---|
| attention — `fmha_cutlassF…` (PyTorch memory-efficient attention) | 50.7 ms | 40.8% |
| matmul (GEMM) | 25.7 ms | 20.7% |
| elementwise / other | 20.6 ms | 16.6% |
| gather / indexing — the paged read path | 14.6 ms | 11.7% |
| layer norm | 5.9 ms | 4.7% |
| KV write — `index_copy` | 5.3 ms | 4.3% |
| copies | 1.6 ms | 1.3% |

Caveats: profiling adds large CPU-side overhead (78,558 CPU ops recorded, plus
`record_shapes`), which inflates wall time and the idle share — read ratios, not absolute
times. And Part A's table (printed in Colab, not saved) is the cleaner breakdown at
32 new tokens; it belongs here too.

## Analysis — yours (judged)

The facts above reshape the open question. To answer, in this file:

1. **Mac vs A100.** Both engines get the same 144 MB of KV memory. The same code gave no
   gain on MPS and 3.3× on the A100. Cover:
   - a. **Capacity:** how 144 MB becomes 4 resident requests (row) vs ~32 (paged).
   - b. **Step time = fixed cost + R × per-row cost:** which term dominates on each
     machine, with rough numbers — and so why more rows help on one and not the other.
   - c. **One measurement** that would confirm (b).


Answer:  

a, 1024 token * 36k per token is about 37M per row,  as a reuslt, 144M hold about 4 rows,  but for paged,  144M can hold about 256 blocks (144M/ (16 * 36k)). in the exp,  each test hold average about 7 blocks, that is about 36 requests (rows)

b. each step,  will move data is 250M (fp16) weights + row number * 36k (bypes per token). for baseline with 1024,  that is, 

A100: 
with 1024 full row cache 
250 M + 4 * 1024 * 36k = 394M, 1.5G/ms bandith,  that is 0.26ms,  greater than computate time 394 M /312G 
250 M + 32 * 1024 * 36k = 1402 M, 1.5G/ms,  that is, 0.94 ms,  greater compute time is 1402 / 312 G 

with reading real tokens: 

250 M + 4 * 100 * 36k = 265M, 1.5G/ms bandith,  that is 0.17ms  greater than compute time 
250 M + 32 * 100 * 36k = 365 M, 1.5G/ms,  that is, 0.24 ms, greater than compuate time 


On Mac: 

250 M + 4 * 1024 * 36k = 394M, 0.14G/ms bandith,  that is 2.81ms 
250 M + 32 * 1024 * 36k = 1402 M, 0.14G/ms,  that is, 10ms 


250 M + 4 * 100 * 36k = 265M, 0.14G/ms bandith,  that is 1.9 ms  greater than compute time 
250 M + 32 * 100 * 36k = 365 M, 0.14 G/ms,  that is, 2.6ms ms, greater than compuate time 

that is why on Mac, we see the time in almost linearly increaseing along with row size as it is when using full row 1024, but on A100,  the cpu took about 8ms,  so the total time increased from 8.26 to 8.94 ms,  while row increses from 4 to 32, so the 0.68 ms increase for gpu as compared to 8ms cpu time, it can be ingored, that is why A100 incresed 3.3x, 

as for c,  just need to time the decode step with r = 2, 4, 8, 32,  A100 should be remaain almost the same, but MAC linearly increases 


2. **Hypothesis 3 — prefill.** Prefill is 81% of wall time in this profile. The run
   generated only 8 new tokens per request, and the engine prefills one request per
   forward. Cover:
   - a. **Workload or engine?** Roughly what share would prefill take at 32 new tokens
     (Part A), and what does that tell you?

     it is both,  worklaod and engine,  the prefill time and decode is 4.3:1,  with 32 new tokens,  

    total time unit will 4.3 + 4 * 1 = 8.3, and prefill time will 4.3 /8.3 = 50% 


   - b. **Why one request per prefill under-uses the A100:** how many token rows does each
     weight matmul get, compared with the ridge (~200 FLOPs/byte)?

     1. one prompt with about 60 to 100 tokens,  that is 60 to 100 Flops/byte,  far below 200 (312/1.5) Flops/byte, 
      2. cpu launch overhead is about the same for more rows,, so the same amount of cpu time,  bigger rows would make the average cpu time lower. 

   - c. **The change that cuts it,** and which number would show it worked.

   increase the batch size of prefill,  and time percentage of prefill drops 

3. **Hypotheses 1 and 2 — attention.** Attention is the largest GPU cost (41%), more than
   every matmul combined, though real sequences are under 160 tokens. Cover:
   - a. **Width:** how many columns each row attends over vs its real length, and the
     rough ratio of wasted work.

      1024 is the what being attended to, that is about (1024 - 160) /160 waste

   - b. **Kernel choice:** what argument we pass to SDPA that rules out FlashAttention,
     so PyTorch falls back to the memory-efficient kernel.

     attn_mask rules out the flash attention 

   - c. **The fix,** and why a smaller block size alone would not change the width.
   
   in the current, each row is padded to 1024,  so if not read real blocks, smaller blocks alone does not help


4. **The idle GPU.** The GPU is 85% idle (with caveats). Cover:
   - a. **What it waits for** during a decode step: the two CPU-side causes, with where
     each happens in the code.

     it is waiting for:  the cpu launch,  about 8ms vs 0.26 ms for 4 rows 

   - b. **Two changes** from your docs that remove them.

      using cuda graph to remove cpu launchs, and remove int sync (int(tok[row])), 

   - c. **The caveat:** why the profiler overstates idle time, and how you would measure
     the idle share without it.

      profiler add time into the cpu time,  do a comparison by with and without profiler to show how much idle time added due to profiler

5. **Predict before P1.5.** Today, on the A100 with 32 requests of ~100 tokens each:
   1,581 tokens/s · attention 41% and gather 12% of GPU time · peak memory 638 MB ·
   GPU 85% idle.
   - a. **After P1.5 only** (attention reads just the real blocks; the CPU side is
     unchanged): predict tokens/s, the attention and gather shares, and peak memory.
     One sentence on why tokens/s moves as much or as little as you predict.
   - b. **If the CPU overhead were also gone:** what is the most tokens/s the A100
     could give for these 32 requests? Is that limit memory or compute?

Answer: 


without changing the cpu bottle nexst,  the gpu would still idle, right?  the 1581 token/s will only increase a little

when using paging and trition direcdtly reading blocks from gpu,  

attention time,  1024 / (7 * 16) = 9,  and tritioin read blocks directly from gpu pool,  the 2 reads + writes reduce to 1 read,  so attention time: 41% / *( 9 * 3)

no memory for holding copies (112 * 3.6K * 32 = 96), the peak 636M - 96M = 540M,    

as trition read data directly from gpu,  so indexing, no gathering,  12% reduced to 0, but index copying will remain

but the index copying for updating new kv values remains the same 


the bottle next is the cpu launch time and sync time causing gpu idling, and not enough data to maximumly utilize gpu's bandith and compute power, 

without cpu overhead,  

memory: 32 rows,  will be 365 M data,  0.24ms, that is, 32 / 0.24 = 133 tokens/ms 
compute: 250M * 32 / 312 = 0.0256ms,  32 / 0.0256 = 1250 token/ms 

that is memory bound,  


Smoke test (tiny model, CPU) — not a performance result, only that the script runs:
row 2128 tok/s vs paged 1731 tok/s, 4 vs 9 resident, 1.0 MB each.
