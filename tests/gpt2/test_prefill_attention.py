"""The FlashAttention-style prefill kernel (paged/prefill_attention.py) against causal
SDPA run on each prompt alone. CUDA only: Triton needs an NVIDIA GPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton needs CUDA')

H, D = 4, 64


def packed(lens, dtype, fused=False):
    """q, k, v (N, H, D) and positions for prompts of these lengths packed end to end.
    fused=True: strided views of one (N, 3*H*D) tensor, as the model's qkv projection
    produces them."""
    torch.manual_seed(0)
    N = sum(lens)
    if fused:
        qkv = torch.randn(N, 3 * H * D, device='cuda', dtype=dtype)
        q, k, v = (t.view(N, H, D) for t in qkv.split(H * D, dim=1))
    else:
        q, k, v = (torch.randn(N, H, D, device='cuda', dtype=dtype) for _ in range(3))
    positions = torch.cat([torch.arange(n, device='cuda') for n in lens])
    return q, k, v, positions


def reference(q, k, v, lens):
    out, start = [], 0
    for n in lens:
        s = slice(start, start + n)
        out.append(torch.nn.functional.scaled_dot_product_attention(
            *(t[s].transpose(0, 1).float() for t in (q, k, v)), is_causal=True).transpose(0, 1))
        start += n
    return torch.cat(out).to(q.dtype)


# tile is 64 tokens: prompts shorter than, equal to, and spanning tiles; 1-token prompts
LENS = [[1], [5], [64], [65], [1, 1, 1], [70, 3, 130, 1], [16, 200, 33, 64, 7], [500, 524]]


@pytest.mark.parametrize('lens', LENS)
@pytest.mark.parametrize('dtype,tol', [(torch.float32, 1e-4), (torch.float16, 2e-3)])
def test_matches_per_prompt_sdpa(lens, dtype, tol):
    from inference.continuous_batch.paged.prefill_attention import prefill_attention
    q, k, v, positions = packed(lens, dtype)
    torch.testing.assert_close(prefill_attention(q, k, v, positions), reference(q, k, v, lens),
                               atol=tol, rtol=tol)


def test_reads_strided_qkv_views_in_place():
    from inference.continuous_batch.paged.prefill_attention import prefill_attention
    lens = [70, 3, 130]
    q, k, v, positions = packed(lens, torch.float16, fused=True)
    assert not q.is_contiguous()
    torch.testing.assert_close(prefill_attention(q, k, v, positions), reference(q, k, v, lens),
                               atol=2e-3, rtol=2e-3)


def test_padding_tokens_do_not_touch_real_ones():
    """A CUDA graph's bucket pads with position-0 tokens: each attends only to itself,
    and the real prompts' outputs are unchanged."""
    from inference.continuous_batch.paged.prefill_attention import prefill_attention
    lens = [70, 3, 130]
    q, k, v, positions = packed(lens, torch.float16)
    real = prefill_attention(q, k, v, positions)
    pad = 50
    q2, k2, v2 = (torch.cat([t, torch.randn(pad, H, D, device='cuda', dtype=t.dtype)]) for t in (q, k, v))
    padded = prefill_attention(q2, k2, v2, torch.cat([positions, torch.zeros(pad, dtype=torch.long, device='cuda')]))
    torch.testing.assert_close(padded[:sum(lens)], real)
    assert torch.isfinite(padded[sum(lens):]).all()
    torch.testing.assert_close(padded[sum(lens):], v2[sum(lens):])       # softmax over one key
