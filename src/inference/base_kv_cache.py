"""BaseKVCache: the contract between gpt.py and a KV cache, and PrefillBatch.

gpt.py never asks which cache it has. It makes one call per layer, by phase:

    prefill_attend(layer, q, k, v, prefill)   store the packed prompts' k/v and return
                                              causal attention within each prompt
    decode_attend(layer, q, k, v, attn_mask)  store the new k/v and return attention
                                              over the cache

plus read (the decode buffers) and advance / seq_len / reset to keep the books. All are
abstract: each cache decides how its memory is written and read, and which kernel
attends over it. sdpa_prefill / sdpa_decode are the shared SDPA implementations, for a
cache (or a device) without its own kernel.

Every cache subclasses this base:

    static_batch/kv_cache.py              KVCache       one write pointer for the batch
    continuous_batch/row/row_cache.py     RowKVCache    a pointer per row, a full-width row each
    continuous_batch/paged/paged_cache.py PagedKVCache  blocks through a block table
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PrefillBatch:
    """Several fresh requests' prompts, packed end to end into one forward.

        input ids      (1, sum(lens))   prompt 0 | prompt 1 | ... | prompt N-1
        slots          [N]              each request's req_pool_idx (its row)
        lens           [N]              each prompt's length
        rows           (sum(lens),)     the slot of every packed token
        positions      (sum(lens),)     every token's position in its own prompt (from 0)

    No padding anywhere. Attention stays inside each prompt (prefill_attend), the
    cache writes each token to (rows[i], positions[i]), and the output layer runs only on
    each prompt's last token (last). This object is metadata only: the packed data is
    the hidden state x itself.
    """
    slots: list
    lens: list
    slots_gpu: torch.Tensor
    lens_gpu: torch.Tensor
    rows: torch.Tensor
    positions: torch.Tensor
    last: torch.Tensor

    @classmethod
    def build(cls, slots, lens, device):
        lens_gpu = torch.tensor(lens, dtype=torch.long, device=device)
        slots_gpu = torch.tensor(slots, dtype=torch.long, device=device)
        starts = torch.cumsum(lens_gpu, 0) - lens_gpu                  # where each prompt begins
        total = int(sum(lens))
        rows = torch.repeat_interleave(slots_gpu, lens_gpu, output_size=total)
        positions = torch.arange(total, device=device) - torch.repeat_interleave(
            starts, lens_gpu, output_size=total)
        return cls(list(slots), list(lens), slots_gpu, lens_gpu, rows, positions,
                   last=starts + lens_gpu - 1)


class BaseKVCache(ABC):

    @abstractmethod
    def prefill_attend(self, layer_idx, query, key, value, prefill):
        """Prefill: store the packed prompts' k/v (1, H, sum(lens), D) -- token i at slot
        prefill.rows[i], position prefill.positions[i] -- and return causal attention
        within each prompt, never across prompts."""

    @abstractmethod
    def decode_attend(self, layer_idx, query, key, value, attn_mask):
        """Decode: store the new k/v (B, H, T, D) at each row's position and return
        attention over the cache. attn_mask hides slots a row may not see (unwritten,
        padding, idle rows)."""

    @abstractmethod
    def read(self, layer_idx, num_rows):
        """The k/v of rows [0, num_rows) to attend over in decode, W wide."""

    @abstractmethod
    def advance(self, T, prefill=None, active=None):
        """Move the write positions, once per forward (not per layer)."""

    @abstractmethod
    def seq_len(self):
        """Tokens already written -- the next position(s)."""

    @abstractmethod
    def reset(self):
        """Back to empty, in place."""

    # -- shared implementation ------------------------------------------------

    @staticmethod
    def sdpa_prefill(query, key, value, prefill):
        """Causal attention within each packed prompt, never across: (1, H, sum(lens), D).
        One SDPA per prompt (FlashAttention on CUDA) -- the prefill_attend of any cache
        without its own kernel."""
        out, start = [], 0
        for n in prefill.lens:
            s = slice(start, start + n)
            out.append(F.scaled_dot_product_attention(query[:, :, s], key[:, :, s],
                                                      value[:, :, s], is_causal=True))
            start += n
        return torch.cat(out, dim=2)

    def sdpa_decode(self, layer_idx, query, attn_mask):
        """Masked SDPA over rows [0, B) of the cache: (B, H, T, D). The mask hides the
        slots a row may not see (unwritten, padding, idle rows)."""
        k, v = self.read(layer_idx, query.size(0))
        return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask)
