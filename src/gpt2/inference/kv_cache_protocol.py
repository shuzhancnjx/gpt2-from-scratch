"""The contract between gpt.py and a KV cache.

gpt.py never asks which cache it has; it calls these methods. Three caches implement
them -- static_batch.KVCache (one shared pointer), continuous_batch RowKVCache (a
pointer per row) and PagedKVCache (blocks through a block table) -- without inheriting
anything: a Protocol only writes the contract down, so a type checker (and
isinstance, with runtime_checkable) can hold each cache to it.

Optional: a cache may set `fused_decode = True` and provide
`decode_attention(layer_idx, q, k, v) -> (B, H, 1, D)`. Then gpt.py hands the whole
decode attention to the cache (PagedKVCache with the Triton kernel) instead of
calling update() and SDPA.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class KVCacheProtocol(Protocol):

    def update(self, layer_idx, key, value, row=None):
        """Write this layer's new k/v (B, H, T, D); return the k/v to attend over.

        row=None: every row writes T tokens at its own position (decode).
        row=int:  that one row writes T tokens (prefill).
        """

    def advance(self, T, row=None, active=None):
        """Move the write position by T, once per forward (not per layer)."""

    def seq_len(self, row=None):
        """Tokens already written -- the next position."""

    def reset(self):
        """Back to empty, in place."""
