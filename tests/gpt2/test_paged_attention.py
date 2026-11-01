"""Paged decode attention (Triton) against the path it replaces: gather to full width,
then masked SDPA. Needs CUDA -- skipped on the Mac, run on Colab.

The failure modes worth catching are silent ones: a swapped stride, a running max
that never updates, a mask that lets stale slots in. Each gives plausible numbers,
so every check compares against the reference rather than against "looks right".
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton needs CUDA')

# Edge lengths: one token, a partial block, exactly one block, one past a block,
# a typical request, and the full width.
SEQ_LENS = [1, 15, 16, 17, 100, 1024]


def reference(query, key_pool, value_pool, block_table, seq_lens):
    """What PagedKVCache._gather + gpt.py do today."""
    num_requests, num_heads, head_dim = query.shape
    full_width = block_table.shape[1] * key_pool.shape[1]

    def gather(pool):
        copied = pool[block_table]                    # (B, max_blocks, block_size, H, D) -- a copy
        return copied.reshape(num_requests, full_width, num_heads, head_dim).transpose(1, 2)

    columns = torch.arange(full_width, device=query.device)
    is_real_token = columns[None, :] < seq_lens[:, None]
    out = F.scaled_dot_product_attention(query.unsqueeze(2), gather(key_pool), gather(value_pool),
                                         attn_mask=is_real_token[:, None, None, :])
    return out.squeeze(2)


def make(dtype, seq_lens=SEQ_LENS, H=12, D=64, block_size=16, max_blocks=64, num_blocks=512):
    torch.manual_seed(0)
    dev = 'cuda'
    B = len(seq_lens)
    ids = torch.randperm(num_blocks - 1, device=dev)[:B * max_blocks] + 1   # scattered; never scratch (0)
    return dict(
        query=torch.randn(B, H, D, device=dev, dtype=dtype),
        key_pool=torch.randn(num_blocks, block_size, H, D, device=dev, dtype=dtype),
        value_pool=torch.randn(num_blocks, block_size, H, D, device=dev, dtype=dtype),
        block_table=ids.view(B, max_blocks),
        seq_lens=torch.tensor(seq_lens, device=dev),
    )


@pytest.mark.parametrize('dtype, tol', [(torch.float32, 1e-4), (torch.float16, 2e-3)])
def test_matches_the_gather_path(dtype, tol):
    from gpt2.inference.continuous_batch.paged_attention import paged_decode_attention
    args = make(dtype)
    torch.testing.assert_close(paged_decode_attention(**args), reference(**args), atol=tol, rtol=tol)


@pytest.mark.parametrize('seq_len', SEQ_LENS)
def test_each_length_on_its_own(seq_len):
    """Pinpoints which edge fails, if the batched test does."""
    from gpt2.inference.continuous_batch.paged_attention import paged_decode_attention
    args = make(torch.float32, seq_lens=[seq_len])
    torch.testing.assert_close(paged_decode_attention(**args), reference(**args), atol=1e-4, rtol=1e-4)


def test_never_reads_past_the_real_length():
    """Overwrite every slot beyond each row's length: the output must not move."""
    from gpt2.inference.continuous_batch.paged_attention import paged_decode_attention
    args = make(torch.float32)
    before = paged_decode_attention(**args)
    block_size = args['key_pool'].shape[1]
    for r, L in enumerate(args['seq_lens'].tolist()):
        for j, phys in enumerate(args['block_table'][r].tolist()):
            first_stale = max(0, L - j * block_size)
            args['key_pool'][phys, first_stale:] = 1e4
            args['value_pool'][phys, first_stale:] = 1e4
    torch.testing.assert_close(paged_decode_attention(**args), before)


def test_output_shape_and_dtype():
    from gpt2.inference.continuous_batch.paged_attention import paged_decode_attention
    args = make(torch.float16)
    out = paged_decode_attention(**args)
    assert out.shape == args['query'].shape and out.dtype == torch.float16
