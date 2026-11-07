"""The shared sampling module and BaseKVCache, the contract gpt.py relies on."""

import pytest
import torch

from model.gpt2.gpt import GPTConfig
from inference.continuous_batch import PagedKVCache, RowKVCache
from inference.base_kv_cache import BaseKVCache
from inference.sampling import greedy, make_sampler, sample
from inference.static_batch.kv_cache import KVCache

CFG = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=16, block_size=32)


def test_every_cache_meets_the_contract():
    caches = [KVCache(2, 'cpu', CFG, max_tokens=16),
              RowKVCache(2, 'cpu', CFG, max_tokens=16),
              PagedKVCache(CFG, num_blocks=9, block_size=4, max_seqs=2, max_blocks_per_seq=4, device='cpu')]
    for cache in caches:
        assert isinstance(cache, BaseKVCache), type(cache).__name__
    assert [c.fused_decode for c in caches] == [False, False, False]


def test_base_kv_cache_is_abstract():
    with pytest.raises(TypeError):
        BaseKVCache()


def test_greedy_and_temperature_zero_agree():
    logits = torch.randn(3, 10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(sample(logits, temperature=0), greedy(logits))
    assert make_sampler(temperature=0) is greedy
    assert greedy(logits).shape == (3, 1)


def test_top_k_only_draws_from_the_k_largest():
    logits = torch.tensor([[5.0, 4.0, 3.0, -1.0, -2.0]])
    torch.manual_seed(0)
    seen = {sample(logits, temperature=1.0, top_k=2).item() for _ in range(200)}
    assert seen <= {0, 1} and len(seen) == 2


def test_top_p_keeps_the_smallest_set_reaching_p():
    # probabilities ~ [0.64, 0.24, 0.09, 0.03]: top_p=0.8 keeps tokens 0 and 1
    logits = torch.log(torch.tensor([[0.64, 0.24, 0.09, 0.03]]))
    torch.manual_seed(0)
    seen = {sample(logits, temperature=1.0, top_p=0.8).item() for _ in range(300)}
    assert seen == {0, 1}
    assert sample(logits, temperature=1.0, top_p=0).item() == 0


def test_make_sampler_fixes_the_settings():
    logits = torch.tensor([[5.0, 4.0, 3.0, -1.0]])
    s = make_sampler(temperature=1.0, top_k=1)
    assert all(s(logits).item() == 0 for _ in range(20))
