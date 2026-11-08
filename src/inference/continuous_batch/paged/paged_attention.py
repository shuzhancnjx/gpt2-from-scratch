"""Paged attention in Triton (P1.5a): one query per row, K/V read through the block table.

Reads K/V straight from the block pool -- no gather to the full width W, no mask.
One program per (row, head); row r attends over the first seq_lens[r] tokens of the
sequence its block-table row describes.

    decode   one row per request: its new token, seq_len = pos + 1
    prefill  one row per PROMPT TOKEN: token t of a prompt uses its request's block
             table and seq_len = t + 1 -- causal attention, every prompt in one launch

Triton needs an NVIDIA GPU: this module imports on Colab, not on the Mac.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    query_ptr, key_pool_ptr, value_pool_ptr, block_table_ptr, seq_lens_ptr, output_ptr, 
    query_row_stride,  query_head_stride, pool_block_stride, pool_token_stride,  pool_head_stride, table_row_stride, 
    soft_max_scale, 
    BLOCK_SIZE: tl.constexpr, 
    HEAD_SIZE: tl.constexpr, 
):

    request = tl.program_id(0) # which request 
    head = tl.program_id(1) # which head 

    dim_offsets = tl.arange(0, HEAD_SIZE) # number of dimensions within each head 
    token_offsets = tl.arange(0, BLOCK_SIZE) # number of tokens within a block 

    query = tl.load(query_ptr + request * query_row_stride + head * query_head_stride + dim_offsets).to(tl.float32)
    seq_len = tl.load(seq_lens_ptr + request) # how many tokens for this request

    running_max = -float('inf')
    running_denominator = 0.0 
    running_numerator = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for logical_block in range(0, num_blocks): 

        physical_block = tl.load(block_table_ptr + request * table_row_stride + logical_block) # which block in kv cache holds the token

        token_positions = logical_block * BLOCK_SIZE + token_offsets  
        is_real_token = token_positions < seq_len 

        element_offsets = (physical_block * pool_block_stride
                            + token_offsets[:, None] * pool_token_stride
                              + head * pool_head_stride 
                              + dim_offsets[None, :])

        keys = tl.load(key_pool_ptr + element_offsets, mask=is_real_token[:, None], other=0.0).to(tl.float32)    

        scores = tl.sum(keys * query[None, :], axis=1) * soft_max_scale # blocks's token (block_size, 1)
        scores = tl.where(is_real_token, scores, -float('inf'))

        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)

        values = tl.load(value_pool_ptr + element_offsets, mask=is_real_token[:, None], other=0.0).to(tl.float32)

        running_numerator = running_numerator * rescale + tl.sum(weights[:, None] * values, axis=0)
        running_denominator = running_denominator * rescale + tl.sum(weights, axis=0)
        running_max = new_max

    attention_output = running_numerator / running_denominator
    tl.store(output_ptr + request * query_row_stride + head * query_head_stride + dim_offsets, 
                attention_output.to(output_ptr.dtype.element_ty))

def paged_attention(query, key_pool, value_pool, block_table, seq_lens):
    """
    query        (B, H, D)                       one query per row
    key_pool     (num_blocks, block_size, H, D)  PagedKVCache.key[layer]
    value_pool   (num_blocks, block_size, H, D)  PagedKVCache.value[layer]
    block_table  (B, max_blocks), int            logical -> physical block per row
    seq_lens     (B,), int                       tokens each row attends over, incl. its own
    returns      (B, H, D)
    """

    query = query.contiguous() 

    num_requests, num_head, head_dim = query.shape

    block_size = key_pool.shape[1]

    assert key_pool.stride() == value_pool.stride() and key_pool.stride(-1) == 1

    output = torch.empty_like(query)

    _paged_attention_kernel[(num_requests, num_head)](
        query,  key_pool, value_pool, block_table, seq_lens, output,
        query_row_stride=query.stride(0), 
        query_head_stride=query.stride(1), 
        pool_block_stride=key_pool.stride(0), 
        pool_token_stride=key_pool.stride(1), 
        pool_head_stride=key_pool.stride(2), 
        table_row_stride=block_table.stride(0), 
        soft_max_scale= head_dim ** (-0.5), 
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_dim
    )

    return output
