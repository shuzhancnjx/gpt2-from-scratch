"""Prefill attention in Triton: FlashAttention over packed prompts (CUDA only).

The decode kernel (decode_attention.py) served prefill as one query per program: no tensor
cores, and every token of a prompt re-reads the whole prompt's K/V -- 51% of a 1024-token
prefill (profile_step.py, A100). But a prompt starts from position 0, so the keys it
attends over are exactly its own packed k/v: nothing has to come from the pool. This
kernel tiles that work the FlashAttention way.

    q, k, v     (N, H, D)  the packed prompts, prompt 0 | prompt 1 | ... (N = sum(lens))
    positions   (N,)       each token's position in its own prompt (PrefillBatch.positions)

Token i's prompt starts at packed index i - positions[i], so it attends over keys
[i - positions[i], i]: causal, never across prompts -- no cu_seqlens needed.

One program per (tile of BLOCK_M queries, head). It loads its queries once, then walks
the key tiles from the earliest prompt start in the tile up to its last query:

    scores = q_tile @ k_tile^T      (BLOCK_M, BLOCK_N) on tensor cores
    mask   = start_i <= j <= i      own prompt, causal (a tile may span two prompts)
    online softmax: running max m, running sum l, rescaled accumulator acc
    acc   += p @ v_tile             (BLOCK_M, D) on tensor cores

Each K/V tile is loaded once per query TILE instead of once per query token. Padding
tokens (a CUDA graph's bucket) have position 0: each attends only to itself, finite and
unused.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, positions_ptr, n_tokens,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride,
    v_token_stride, v_head_stride, o_token_stride, o_head_stride,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SIZE: tl.constexpr, IEEE: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    queries = tile * BLOCK_M + tl.arange(0, BLOCK_M)            # packed query indices
    dims = tl.arange(0, HEAD_SIZE)
    is_query = queries < n_tokens

    positions = tl.load(positions_ptr + queries, mask=is_query, other=0)
    starts = queries - positions                                # each query's prompt start

    q = tl.load(q_ptr + queries[:, None] * q_token_stride + head * q_head_stride + dims[None, :],
                mask=is_query[:, None], other=0.0)

    # keys this tile needs: from the earliest prompt start to its last query
    first_key = tl.min(tl.where(is_query, starts, n_tokens), axis=0)
    first_key = (first_key // BLOCK_N) * BLOCK_N
    end_key = tl.minimum((tile + 1) * BLOCK_M, n_tokens)

    running_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    for key_tile in range(first_key, end_key, BLOCK_N):
        keys = key_tile + tl.arange(0, BLOCK_N)
        is_key = keys < n_tokens
        k = tl.load(k_ptr + keys[:, None] * k_token_stride + head * k_head_stride + dims[None, :],
                    mask=is_key[:, None], other=0.0)
        if IEEE:                                                # fp32 inputs: exact, no TF32
            scores = tl.dot(q, tl.trans(k), input_precision='ieee')
        else:
            scores = tl.dot(q, tl.trans(k))
        scores = scores * softmax_scale
        allowed = ((keys[None, :] >= starts[:, None]) & (keys[None, :] <= queries[:, None])
                   & is_key[None, :])
        scores = tl.where(allowed, scores, -float('inf'))

        # online softmax. A row with nothing allowed in this tile (it belongs to a later
        # prompt than these keys) keeps max -inf: shift by 0 instead, so exp gives 0, not NaN
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        shift = tl.where(new_max == -float('inf'), 0.0, new_max)
        weights = tl.exp(scores - shift[:, None])
        rescale = tl.exp(running_max - shift)
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)

        v = tl.load(v_ptr + keys[:, None] * v_token_stride + head * v_head_stride + dims[None, :],
                    mask=is_key[:, None], other=0.0)
        if IEEE:
            acc = acc * rescale[:, None] + tl.dot(weights, v, input_precision='ieee')
        else:
            acc = acc * rescale[:, None] + tl.dot(weights.to(v.dtype), v)
        running_max = new_max

    out = acc / running_sum[:, None]                            # every real query sees itself
    tl.store(o_ptr + queries[:, None] * o_token_stride + head * o_head_stride + dims[None, :],
             out.to(o_ptr.dtype.element_ty), mask=is_query[:, None])


def prefill_attention(query, key, value, positions, block_m=64, block_n=64):
    """
    query, key, value  (N, H, D)  packed prompts; any strides with a unit last dim (the
                                  views of the model's fused qkv projection, no copy)
    positions          (N,), int  each token's position in its own prompt
    returns            (N, H, D)  contiguous: causal attention within each prompt
    """
    for t in (query, key, value):
        assert t.stride(-1) == 1, 'the head dim must be contiguous'
    n_tokens, num_heads, head_dim = query.shape
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    grid = (triton.cdiv(n_tokens, block_m), num_heads)
    _prefill_attention_kernel[grid](
        query, key, value, output, positions, n_tokens,
        query.stride(0), query.stride(1), key.stride(0), key.stride(1),
        value.stride(0), value.stride(1), output.stride(0), output.stride(1),
        head_dim ** -0.5,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_SIZE=head_dim,
        IEEE=query.dtype == torch.float32,
    )
    return output
