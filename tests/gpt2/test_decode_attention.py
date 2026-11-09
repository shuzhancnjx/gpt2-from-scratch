"""Decode attention (Triton) against the path it replaces: write the new token, gather to
full width, then masked SDPA. Needs CUDA -- skipped on the Mac, run on Colab.

The kernel also WRITES: each row's new k/v goes to its slot (position seq_len - 1), and
attention uses the new token from registers, the history from the pool.

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


def reference(query, key, value, key_pool, value_pool, block_table, seq_lens, slots):
    """What PagedKVCache.write + _gather + gpt.py do on the SDPA path, on copies of the pools."""
    num_requests, num_heads, head_dim = query.shape
    full_width = block_table.shape[1] * key_pool.shape[1]
    key_pool, value_pool = key_pool.clone(), value_pool.clone()
    key_pool.view(-1, num_heads, head_dim)[slots] = key
    value_pool.view(-1, num_heads, head_dim)[slots] = value

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
    block_table = ids.view(B, max_blocks)
    lens = torch.tensor(seq_lens, device=dev)
    new = lens - 1                                                          # the new token's position
    slots = block_table[torch.arange(B, device=dev), new // block_size] * block_size + new % block_size
    qkv = torch.randn(B, 3, H, D, device=dev, dtype=dtype)                  # strided, like the model's
    return dict(
        query=qkv[:, 0], key=qkv[:, 1], value=qkv[:, 2],
        key_pool=torch.randn(num_blocks, block_size, H, D, device=dev, dtype=dtype),
        value_pool=torch.randn(num_blocks, block_size, H, D, device=dev, dtype=dtype),
        block_table=block_table, seq_lens=lens, slots=slots,
    )


@pytest.mark.parametrize('dtype, tol', [(torch.float32, 1e-4), (torch.float16, 2e-3)])
def test_matches_the_gather_path(dtype, tol):
    from inference.continuous_batch.paged.decode_attention import decode_attention
    args = make(dtype)
    torch.testing.assert_close(decode_attention(**args), reference(**args), atol=tol, rtol=tol)


@pytest.mark.parametrize('seq_len', SEQ_LENS)
def test_each_length_on_its_own(seq_len):
    """Pinpoints which edge fails, if the batched test does."""
    from inference.continuous_batch.paged.decode_attention import decode_attention
    args = make(torch.float32, seq_lens=[seq_len])
    torch.testing.assert_close(decode_attention(**args), reference(**args), atol=1e-4, rtol=1e-4)


def test_never_reads_past_the_real_length():
    """Overwrite every slot beyond each row's length: the output must not move."""
    from inference.continuous_batch.paged.decode_attention import decode_attention
    args = make(torch.float32)
    before = decode_attention(**args)
    block_size = args['key_pool'].shape[1]
    for r, L in enumerate(args['seq_lens'].tolist()):
        for j, phys in enumerate(args['block_table'][r].tolist()):
            first_stale = max(0, L - j * block_size)
            args['key_pool'][phys, first_stale:] = 1e4
            args['value_pool'][phys, first_stale:] = 1e4
    torch.testing.assert_close(decode_attention(**args), before)


def test_writes_the_new_token_and_ignores_what_its_slot_held():
    """The new k/v land at their slots, and the attention uses them -- not the stale
    value the slot held before (the kernel takes the new token from registers)."""
    from inference.continuous_batch.paged.decode_attention import decode_attention
    args = make(torch.float32)
    H, D = args['query'].shape[1:]
    flat = lambda pool: pool.view(-1, H, D)
    flat(args['key_pool'])[args['slots']] = 1e4                 # stale garbage at the slots
    flat(args['value_pool'])[args['slots']] = 1e4
    expected = reference(**args)
    out = decode_attention(**args)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(flat(args['key_pool'])[args['slots']], args['key'])
    torch.testing.assert_close(flat(args['value_pool'])[args['slots']], args['value'])


def test_tile_size_does_not_change_the_result():
    """A tile spans several blocks; 16 (one block) and 128 must agree with 64."""
    from inference.continuous_batch.paged.decode_attention import decode_attention
    outs = [decode_attention(**make(torch.float32), tile=t) for t in (16, 64, 128)]
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=1e-5, rtol=1e-5)


def test_output_shape_and_dtype():
    from inference.continuous_batch.paged.decode_attention import decode_attention
    args = make(torch.float16)
    out = decode_attention(**args)
    assert out.shape == args['query'].shape and out.dtype == torch.float16
