"""A paged KV cache with the same interface as RowKVCache.

RowKVCache reserves a full-width row (W slots) per request up front. Here the KV
memory is one pool of fixed-size blocks per layer,

    key[layer], value[layer]   (num_blocks, block_size, n_head, head_dim)

and each request owns only the blocks it was given, recorded in a block table:

    block_table   (max_seqs, max_blocks_per_seq)   row r, logical block j -> physical block

update() keeps RowKVCache's contract -- write the new tokens, return buffers to
attend over -- so gpt.py is unchanged:

  write: every new token goes to   slot = block_table[row, pos // bs] * bs + pos % bs,
         computed on the device from tensors (no int(), no sync), then one
         index_copy_ into a flat (num_blocks * bs, H, D) view of the pool.
  read:  gather the row's blocks in logical order into (B, H, W, D) with
         W = max_blocks_per_seq * block_size. Column i is position i, exactly as in
         RowKVCache, so the scheduler's mask (cols <= pos) works unchanged.

The gather COPIES -- scattered blocks have no single stride pattern, so advanced
indexing must materialise them. That is the cost a paged-attention kernel removes
(milestone P1.5); here it buys a drop-in replacement that is easy to verify.

Rows with no request point at the SCRATCH block (see block_allocator.py). The
decode step writes every row, so all idle rows write the same scratch slot in one
index_copy_ -- duplicate indices, undefined winner. That is harmless only because
scratch is never handed to a request and is only ever read in masked columns.
Real requests must never share a slot; tests check that.
"""

import torch

from gpt2.gpt import GPTConfig
from gpt2.inference.continuous_batch.block_allocator import SCRATCH


class PagedKVCache:

    def __init__(self, config: GPTConfig, num_blocks, block_size, max_seqs,
                 max_blocks_per_seq, device, dtype=torch.float32):
        head_size = config.n_embd // config.n_head
        shape = (num_blocks, block_size, config.n_head, head_size)

        # zeros, not empty: masked columns still flow through attention as
        # 0-weight * value, and an uninitialised NaN would poison that product.
        self.key = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.value = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

        self.block_table_gpu = torch.full((max_seqs, max_blocks_per_seq), SCRATCH,
                                      dtype=torch.long, device=device)
        self.pos_gpu = torch.zeros(max_seqs, dtype=torch.long, device=device)

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.width = max_blocks_per_seq * block_size
        self.batch_size = max_seqs
        self.device = device

    # -- block table (host decisions → device) -------------------------------

    def set_blocks(self, row, blocks):
        """Record which physical blocks `row` owns, in logical order.

        Unused entries point at scratch. One small host→device copy, made only when
        a row's blocks change (admit, grow, retire) -- never on an ordinary step.
        """
        if len(blocks) > self.max_blocks_per_seq:
            raise ValueError(f'{len(blocks)} blocks exceed max_blocks_per_seq={self.max_blocks_per_seq}')
        padded = list(blocks) + [SCRATCH] * (self.max_blocks_per_seq - len(blocks))
        self.block_table_gpu[row] = torch.tensor(padded, dtype=torch.long, device=self.device)

    def slots(self, rows, positions):
        """Flat pool index for each (row, position) pair -- tensors in, tensor out."""
        physical = self.block_table_gpu[rows, positions // self.block_size]
        return physical * self.block_size + positions % self.block_size

    # -- writes and reads ------------------------------------------------------

    def update(self, layer_idx, key, value, row=None):
        """Write this layer's new k/v, then return the gathered buffers to attend over.

        row=None  -- decode: every row writes one token at its own pos.
        row=int   -- prefill: one row writes T tokens starting at its pos.
        """
        assert layer_idx is not None, 'layer_idx is required when a cache is passed'
        B, H, T, D = key.shape

        if row is not None:
            assert B == 1, f'prefill writes one row, got a batch of {B}'
            positions = self.pos_gpu[row] + torch.arange(T, device=self.device)
            rows = torch.full((T,), row, dtype=torch.long, device=self.device)
            sel = slice(row, row + 1)
        else:
            assert T == 1, f'decode writes one token per row, got {T}'
            rows = torch.arange(B, device=self.device)
            positions = self.pos_gpu[:B]
            sel = slice(0, B)

        slots = self.slots(rows, positions)                       # (B*T,), in (row, t) order
        self._flat(self.key[layer_idx]).index_copy_(0, slots, self._tokens_first(key))
        self._flat(self.value[layer_idx]).index_copy_(0, slots, self._tokens_first(value))

        return self._gather(self.key[layer_idx], sel), self._gather(self.value[layer_idx], sel)

    def _flat(self, pool):
        return pool.view(self.num_blocks * self.block_size, *pool.shape[2:])

    @staticmethod
    def _tokens_first(x):
        """(B, H, T, D) → (B*T, H, D): one row per token, matching `slots`."""
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B * T, H, D)

    def _gather(self, pool, sel):
        """The rows' blocks in logical order → (B, H, W, D). A copy (see module doc)."""
        table = self.block_table_gpu[sel]                              # (B, max_blocks)
        B = table.shape[0]
        blocks = pool[table]                                       # (B, max_blocks, bs, H, D)
        return blocks.reshape(B, self.width, *pool.shape[2:]).transpose(1, 2)

    # -- bookkeeping (same contract as RowKVCache) ----------------------------

    def advance(self, T, row=None, active=None):
        """Move the write pointer once per forward, never per layer."""
        if row is not None:
            self.pos_gpu[row] = T                      # prefill starts from 0
        elif active is not None:
            self.pos_gpu += active.long() * T          # only rows still generating
        else:
            self.pos_gpu += T

    def seq_len(self, row=None):
        return int(self.pos_gpu[row]) if row is not None else self.pos_gpu

    def release(self, row):
        """Point the row back at scratch. The pool is NOT cleared: stale blocks are
        harmless because the mask hides everything at or past pos."""
        self.pos_gpu[row] = 0
        self.block_table_gpu[row] = SCRATCH

    def reset(self):
        self.pos_gpu.zero_()
        self.block_table_gpu.fill_(SCRATCH)
