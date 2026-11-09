"""BaseKVCache: the contract gpt.py relies on -- every cache implements it."""

import pytest

from model.gpt2.gpt import GPTConfig
from inference.base_kv_cache import BaseKVCache
from inference.continuous_batch import PagedKVCache, RowKVCache
from inference.static_batch.kv_cache import KVCache

CFG = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=16, block_size=32)


def test_every_cache_meets_the_contract():
    caches = [KVCache(2, 'cpu', CFG, max_tokens=16),
              RowKVCache(2, 'cpu', CFG, max_tokens=16),
              PagedKVCache(CFG, num_blocks=9, block_size=4, max_seqs=2, max_blocks_per_seq=4, device='cpu')]
    for cache in caches:
        assert isinstance(cache, BaseKVCache), type(cache).__name__
    assert not caches[2].use_kernel              # Triton only on CUDA: SDPA on this CPU cache


def test_base_kv_cache_is_abstract():
    with pytest.raises(TypeError):
        BaseKVCache()
