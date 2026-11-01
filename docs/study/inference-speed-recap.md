# Inference speed — recap

> **Read as little as possible from HBM, serve as many requests as possible per read,
> and keep the CPU out of the GPU's way.**

## The root fact

Decode is **memory-bound**, not compute-bound. Each step emits one token per request,
but to do so the GPU must read **all model weights** plus **every request's KV cache**
from HBM. The arithmetic is tiny relative to the bytes moved, so almost every
optimization either moves fewer bytes or does more useful work per byte.

## The five levers

**1. Batch, so one weight read serves many requests.**
Weights are read once per step whether the batch is 1 or 32, so batching amortizes
that fixed cost — throughput rises almost for free until you hit compute or KV-memory
limits. **Continuous batching** (`scheduler.py`) keeps the batch full by admitting new
requests the moment old ones retire.

**2. Cut CPU overhead — two distinct problems.**
- *Launch overhead*: each kernel launch costs a few µs of CPU time, and decode runs
  hundreds of small kernels per step. Fix: **CUDA graphs** (record once, replay) and
  fewer, fused kernels.
- *CPU–GPU sync*: `.item()`, `.cpu()`, `print(tensor)`, or a Python `if` on a GPU value
  make the CPU wait for the GPU to drain, leaving the GPU idle afterwards. Fix: keep
  decisions on GPU tensors, or keep a CPU-side mirror (`row_blocks` beside the GPU
  `block_table`).

**3. Paged KV cache.**
The main win is **memory**: without paging, each request reserves `max_tokens` of
contiguous cache up front, mostly wasted. Fixed-size blocks handed out on demand mean
no fragmentation and little waste, so **more requests fit** — bigger batches, back to
lever 1. The paged *kernel* adds a second win: it reads only the real blocks, in
place, through the block table, instead of gathering a padded copy.

**4. KV cache and FlashAttention — different problems.**
- *KV cache* avoids **recomputing** past tokens' K/V: per-token work drops from
  re-processing the whole sequence to processing one token.
- *FlashAttention / online softmax* avoids **writing the score matrix to HBM**: K/V are
  processed tile by tile in registers with a running max and sum. Exact, not
  approximate — and fewer HBM bytes means faster.

**5. Keep hot state on the GPU.**
Tensors the per-step loop touches (KV pool, block table, `seq_lens`, positions) live on
the GPU. The CPU only makes coarse decisions — admit, retire, allocate blocks — from its
own copies.

## Trade-off test for any new optimization

Ask which of the three it helps and what it costs in the others:

| optimization | helps | costs |
|---|---|---|
| quantization | fewer HBM bytes | dequantization compute |
| bigger batches | more requests per read | KV memory, per-token latency |
| CUDA graphs | CPU out of the way | static shapes, less flexibility |

## Not yet covered

- **Lower precision**: bf16/fp16, then int8/fp4 for weights and KV — fewer bytes per number.
- **Kernel fusion**: merge small ops so intermediates never touch HBM (e.g. writing K/V
  from the QKV matmul's epilogue).
- **Speculative decoding**: several tokens per weight read, via a small draft model
  verified by the large one.

## Paged decode kernel — what each piece does

1. **Pick the work**: `request = program_id(0)`, `head = program_id(1)`; grid `(B, H)`.
2. **Set up**: `dim_offsets` (0..63), `slot_offsets` (0..15); load this request's query
   for this head and its `seq_len`; init `running_max/denominator/numerator`.
3. **Loop over logical blocks** (`cdiv(seq_len, BLOCK_SIZE)`):
   - `tl.load(block_table[request, j])` → physical block **ID** (data, so runtime, not constexpr).
   - `is_real_token = position < seq_len` — False only in the last, partial block.
   - Address grid `phys*block_stride + slot*slot_stride + head*head_stride + dim` →
     `(16, 64)`: 16 tokens of this request, this head.
   - Masked load of K and V (padding rows → zeros).
4. **Online softmax**: `scores = sum(K * q) * scale`; `where(mask, scores, -inf)` so
   padding gets exactly zero weight; rescale old sums by `exp(old_max - new_max)`, add
   this block's weighted values.
5. **Normalize and store**: `numerator / denominator`, cast to output dtype,
   `tl.store` into `output[request, head, :]` — a separate `(B, H, D)` tensor PyTorch
   allocated before launch, not the KV pool.

Supporting facts:
- **Contiguous** = no gaps *and* row-major order (strides are products of the sizes to
  the right). The kernel adds `dim_offsets` directly, so D must have stride 1, and it
  reuses query's strides for `output` — hence `query.contiguous()` + `empty_like`.
- **constexpr** values set tile *shapes* (`BLOCK_SIZE`, `HEAD_DIM`; powers of 2) and are
  baked into the compiled binary. Changing one recompiles. Everything that is data
  (pointers, strides, `seq_lens`, block IDs) is a runtime argument.
- **Ordering per layer**: write the new token's K/V into the pool first (`cache.update`),
  then run attention — the token must attend to itself. The token sampled in step N is
  written to the cache during step N+1.
