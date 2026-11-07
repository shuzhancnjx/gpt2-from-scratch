"""A KV cache whose rows advance independently.

gpt2/inference/static_batch/kv_cache.py keeps one scalar write pointer for the whole batch.
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

from gpt2.model.gpt import GPTConfig
from gpt2.inference.base_kv_cache import BaseKVCache


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

    def update(self, layer_idx, key, value, row=None):
        """Write this layer's k/v and return the full buffers to attend over.

        row=None  -- decode: every row writes one token at its own pos.
        row=int   -- prefill: one row writes a contiguous block from slot 0.
        """
        assert layer_idx is not None, 'layer_idx is required when a cache is passed'

        if row is not None:
            return self._prefill(layer_idx, key, value, row)
        return self._decode(layer_idx, key, value)

    def _prefill(self, layer_idx, key, value, row):
        T = key.size(2)
        assert key.size(0) == 1, f'prefill writes one row, got a batch of {key.size(0)}'
        assert int(self.pos_gpu[row]) == 0, f'row {row} is not free (pos={int(self.pos_gpu[row])})'
        assert T <= self.width, f'prompt of {T} exceeds cache width {self.width}'

        # slice(), not fancy indexing: basic slicing gives a VIEW, so attention
        # reads the buffer in place instead of copying (B=1, nh, W, hs) per layer.
        sl = slice(row, row + 1)
        # Both sides stay 4-D here: basic slices preserve every axis, unlike the
        # decode path where advanced indexing collapses the sequence dimension.
        self.key[layer_idx][sl, :, :T] = key
        self.value[layer_idx][sl, :, :T] = value
        return self.key[layer_idx][sl], self.value[layer_idx][sl]

    def _decode(self, layer_idx, key, value):
        assert key.size(2) == 1, f'decode writes one token per row, got {key.size(2)}'

        rows = torch.arange(key.size(0), device=key.device)
        # Advanced indexing on dims 0 and 2 with a slice between: both index
        # tensors live on the device, so this stays capturable in a CUDA graph.
        self.key[layer_idx][rows, :, self.pos_gpu] = key[:, :, 0, :]
        self.value[layer_idx][rows, :, self.pos_gpu] = value[:, :, 0, :]
        return self.key[layer_idx], self.value[layer_idx]

    # -- bookkeeping -------------------------------------------------------

    def advance(self, T, row=None, active=None):
        """Move the write pointer. Called once per forward, never per layer.

        Advancing inside update() would shear the layers apart: update runs
        n_layer times per forward, so row pointers would jump by n_layer * T.
        """
        if row is not None:
            self.pos_gpu[row] = T                      # prefill starts from 0
        elif active is not None:
            self.pos_gpu += active.long() * T          # only rows still generating
        else:
            self.pos_gpu += T

    def seq_len(self, row=None):
        return int(self.pos_gpu[row]) if row is not None else self.pos_gpu

    def release(self, row):
        """Retire a row so a queued request can take its slot."""
        self.pos_gpu[row] = 0
        for k, v in zip(self.key, self.value):
            k[row].zero_()
            v[row].zero_()

    def reset(self):
        self.pos_gpu.zero_()
        for k, v in zip(self.key, self.value):
            k.zero_()
            v.zero_()
