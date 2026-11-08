"""A KV cache whose rows advance independently.

inference/static_batch/kv_cache.py keeps one scalar write pointer for the whole batch.
That works because left-padding aligns every row's last real token at the same
column, so all rows step together.

Continuous batching breaks that: rows are admitted and retired at different
times, so at any moment row 0 may be 300 tokens in while row 3 just arrived at
token 5. One pointer cannot express that, so `pos` becomes a (B,) tensor and
the decode write becomes a per-row scatter.

Losing the shared pointer also loses the *reason* for left-padding. Each row
now occupies slots [0, pos[r]) of its own row, so a slot index is a position
again -- no cumsum, no padding offset.
"""

import torch

from model.gpt2.gpt import GPTConfig
from inference.base_kv_cache import BaseKVCache


class RowKVCache(BaseKVCache):

    def __init__(self, batch_size, device, config: GPTConfig, max_tokens=None,
                 dtype=torch.float32):
        W = max_tokens if max_tokens is not None else config.block_size
        shape = (batch_size, config.n_head, W, config.n_embd // config.n_head)

        self.key = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.value = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

        self.batch_size = batch_size
        self.width = W
        self.device = device

        # One write pointer per row -- the whole point of this class.
        self.pos_gpu = torch.zeros(batch_size, dtype=torch.long, device=device)

    # -- writes ------------------------------------------------------------

    def write(self, layer_idx, key, value, prefill=None):
        """Store this layer's k/v.

        prefill=None   -- decode: every row writes one token at its own pos.
        prefill=batch  -- packed prompts: token i goes to row rows[i], slot positions[i].
        """
        if prefill is not None:
            rows, slots = prefill.rows, prefill.positions          # (sum(lens),) each
            new_k, new_v = key[0].transpose(0, 1), value[0].transpose(0, 1)   # (sum(lens), H, D)
        else:
            assert key.size(2) == 1, f'decode writes one token per row, got {key.size(2)}'
            rows, slots = torch.arange(key.size(0), device=key.device), self.pos_gpu
            new_k, new_v = key[:, :, 0, :], value[:, :, 0, :]                 # (B, H, D)
        # Advanced indexing on dims 0 and 2 with a slice between: index tensors on the
        # device, so the decode write stays capturable in a CUDA graph.
        self.key[layer_idx][rows, :, slots] = new_k
        self.value[layer_idx][rows, :, slots] = new_v

    def read(self, layer_idx, num_rows):
        """Rows [0, num_rows), full width: a view, no copy (the mask hides unwritten slots)."""
        return self.key[layer_idx][:num_rows], self.value[layer_idx][:num_rows]

    # -- attention ----------------------------------------------------------

    def prefill_attend(self, layer_idx, query, key, value, prefill):
        self.write(layer_idx, key, value, prefill)
        return self.sdpa_prefill(query, key, value, prefill)

    def decode_attend(self, layer_idx, query, key, value, attn_mask):
        self.write(layer_idx, key, value)
        return self.sdpa_decode(layer_idx, query, attn_mask)

    # -- bookkeeping -------------------------------------------------------

    def advance(self, T, prefill=None, active=None):
        """Move the write pointer. Called once per forward, never per layer.

        Advancing inside write() would shear the layers apart: write runs
        n_layer times per forward, so row pointers would jump by n_layer * T.
        """
        if prefill is not None:
            self.pos_gpu[prefill.slots_gpu] = prefill.lens_gpu   # each prompt written from 0
        elif active is not None:
            self.pos_gpu += active.long() * T          # only rows still generating
        else:
            self.pos_gpu += T

    def seq_len(self):
        return self.pos_gpu

    def release(self, req_pool_idx):
        """Retire a row so a queued request can take its slot."""
        self.pos_gpu[req_pool_idx] = 0
        for k, v in zip(self.key, self.value):
            k[req_pool_idx].zero_()
            v[req_pool_idx].zero_()

    def reset(self):
        self.pos_gpu.zero_()
        for k, v in zip(self.key, self.value):
            k.zero_()
            v.zero_()
