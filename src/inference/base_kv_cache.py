"""BaseKVCache: the contract between gpt.py and a KV cache.

gpt.py never asks which cache it has; it calls these four methods. Every cache
subclasses this base and implements them:

    static_batch/kv_cache.py           KVCache        one write pointer for the whole batch
    continuous_batch/row/row_cache.py  RowKVCache     a pointer per row, a full-width row each
    continuous_batch/paged/paged_cache.py PagedKVCache  blocks through a block table

Optional fast path: a cache may set `fused_decode = True` and override
`decode_attention(layer_idx, q, k, v) -> (B, H, 1, D)`. gpt.py then hands the whole
decode attention to the cache (PagedKVCache with the Triton kernel) instead of
calling update() and SDPA.
"""

from abc import ABC, abstractmethod


class BaseKVCache(ABC):

    fused_decode = False                        # True: gpt.py calls decode_attention()

    @abstractmethod
    def update(self, layer_idx, key, value, req_pool_idx=None):
        """Write this layer's new k/v (B, H, T, D); return the k/v to attend over.

        req_pool_idx=None: every sequence slot writes T tokens at its own position (decode).
        req_pool_idx=int:  only that request's slot writes T tokens (prefill).
        """

    @abstractmethod
    def advance(self, T, req_pool_idx=None, active=None):
        """Move the write position by T, once per forward (not per layer)."""

    @abstractmethod
    def seq_len(self, req_pool_idx=None):
        """Tokens already written -- the next position."""

    @abstractmethod
    def reset(self):
        """Back to empty, in place."""

    def decode_attention(self, layer_idx, query, key, value):
        raise NotImplementedError(f'{type(self).__name__} has no fused decode path')
