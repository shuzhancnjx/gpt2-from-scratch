"""Decode attention in Triton (P1.5a): one query per row, K/V read through the block table.

Reads K/V straight from the block pool -- no gather to the full width W, no mask.
One program per (row, head); row r has seq_lens[r] tokens, the last one new this step:

    1. store the new token's k/v at its pool slot (slots[r]) -- the cache's write,
       fused in: no separate index_copy_ per layer
    2. start the online softmax WITH the new token, straight from registers (reading it
       back from the pool would race the store above)
    3. walk the row's history, positions [0, seq_len - 1), TILE tokens per iteration:
       each token finds its own physical block (block_table[pos // block_size]), so a
       tile spans several blocks -- fewer iterations and more loads in flight than one
       16-token block at a time

It served prefill too, one row per prompt token (seq_len = t + 1), until the profile
showed that is the wrong shape for prefill -- one query per program, no tensor cores, a
prompt's K/V re-read by each of its tokens. Prefill now has its own FlashAttention-style
kernel over the packed prompts (prefill_attention.py).

Triton needs an NVIDIA GPU: this module imports on Colab, not on the Mac.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    query_ptr, key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, block_table_ptr,
    seq_lens_ptr, slots_ptr, output_ptr,
    query_row_stride, query_head_stride, new_row_stride, new_head_stride,
    output_row_stride, output_head_stride,
    pool_block_stride, pool_token_stride, pool_head_stride, table_row_stride,
    soft_max_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):

    request = tl.program_id(0) # which request
    head = tl.program_id(1) # which head

    dim_offsets = tl.arange(0, HEAD_SIZE) # number of dimensions within each head
    tile_offsets = tl.arange(0, TILE) # tokens per iteration (spanning TILE / BLOCK_SIZE blocks)

    query = tl.load(query_ptr + request * query_row_stride + head * query_head_stride + dim_offsets).to(tl.float32)
    new_key = tl.load(key_ptr + request * new_row_stride + head * new_head_stride + dim_offsets)
    new_value = tl.load(value_ptr + request * new_row_stride + head * new_head_stride + dim_offsets)

    # 1. the new token into the pool, at its slot (= physical block * block_size + offset)
    slot = tl.load(slots_ptr + request)
    new_offsets = slot * pool_token_stride + head * pool_head_stride + dim_offsets
    tl.store(key_pool_ptr + new_offsets, new_key)
    tl.store(value_pool_ptr + new_offsets, new_value)

    # 2. the softmax starts with the new token, from registers: max = its score, sum = 1
    running_max = tl.sum(new_key.to(tl.float32) * query, axis=0) * soft_max_scale
    running_denominator = 1.0
    running_numerator = new_value.to(tl.float32)

    # 3. the history: every earlier token of this row, TILE at a time
    history = tl.load(seq_lens_ptr + request) - 1
    for tile_start in range(0, history, TILE):

        token_positions = tile_start + tile_offsets
        is_real_token = token_positions < history

        physical_block = tl.load(block_table_ptr + request * table_row_stride + token_positions // BLOCK_SIZE,
                                 mask=is_real_token, other=0) # each token's block in the pool

        element_offsets = (physical_block[:, None] * pool_block_stride
                            + (token_positions % BLOCK_SIZE)[:, None] * pool_token_stride
                              + head * pool_head_stride
                              + dim_offsets[None, :])

        keys = tl.load(key_pool_ptr + element_offsets, mask=is_real_token[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * soft_max_scale # (TILE,)
        scores = tl.where(is_real_token, scores, -float('inf'))

        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)

        values = tl.load(value_pool_ptr + element_offsets, mask=is_real_token[:, None], other=0.0).to(tl.float32)

        running_numerator = running_numerator * rescale + tl.sum(weights[:, None] * values, axis=0)
        running_denominator = running_denominator * rescale + tl.sum(weights, axis=0)
        running_max = new_max

    attention_output = running_numerator / running_denominator
    tl.store(output_ptr + request * output_row_stride + head * output_head_stride + dim_offsets,
                attention_output.to(output_ptr.dtype.element_ty))


def decode_attention(query, key, value, key_pool, value_pool, block_table, seq_lens, slots,
                     tile=64):
    """
    query        (B, H, D)                       one query per row
    key, value   (B, H, D)                       this step's new token per row -- written
                                                 into the pool here (the cache's write)
    key_pool     (num_blocks, block_size, H, D)  PagedKVCache.key[layer], contiguous
    value_pool   (num_blocks, block_size, H, D)  PagedKVCache.value[layer]
    block_table  (B, max_blocks), int            logical -> physical block per row
    seq_lens     (B,), int                       tokens each row attends over, incl. the new one
    slots        (B,), int                       the new token's flat pool slot,
                                                 physical block * block_size + offset
    returns      (B, H, D)

    query / key / value: any strides with a unit last dim (views of the fused qkv, no copy).
    """
    for t in (query, key, value):
        assert t.stride(-1) == 1, 'the head dim must be contiguous'
    assert key.stride() == value.stride()

    num_requests, num_head, head_dim = query.shape

    block_size = key_pool.shape[1]

    assert key_pool.stride() == value_pool.stride() and key_pool.is_contiguous(), \
        'slots index a flat view of the pool'

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)   # contiguous

    _decode_attention_kernel[(num_requests, num_head)](
        query, key, value, key_pool, value_pool, block_table, seq_lens, slots, output,
        query_row_stride=query.stride(0),
        query_head_stride=query.stride(1),
        new_row_stride=key.stride(0),
        new_head_stride=key.stride(1),
        output_row_stride=output.stride(0),
        output_head_stride=output.stride(1),
        pool_block_stride=key_pool.stride(0),
        pool_token_stride=key_pool.stride(1),
        pool_head_stride=key_pool.stride(2),
        table_row_stride=block_table.stride(0),
        soft_max_scale=head_dim ** (-0.5),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_dim,
        TILE=max(tile, block_size),
    )

    return output
