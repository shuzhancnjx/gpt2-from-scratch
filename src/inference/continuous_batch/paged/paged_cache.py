"""A paged KV cache with the same interface as RowKVCache.

RowKVCache reserves a full-width row (W slots) per request up front. Here the KV
memory is one pool of fixed-size blocks per layer,

    key[layer], value[layer]   (num_blocks, block_size, n_head, head_dim)

and each request owns only the blocks it was given, recorded in a block table:

    block_table   (max_seqs, max_blocks_per_seq)   row r, logical block j -> physical block

It implements BaseKVCache like RowKVCache does, so gpt.py cannot tell them apart:

  write: every new token goes to   slot = block_table[row, pos // bs] * bs + pos % bs,
         computed on the device from tensors (no int(), no sync) once per forward
         (begin_forward), then one index_copy_ per layer into a flat
         (num_blocks * bs, H, D) view of the pool.
  read:  decode gathers the rows' blocks in logical order into (B, H, W, D) with
         W = max_blocks_per_seq * block_size. Column i is position i, exactly as in
         RowKVCache, so the scheduler's mask (cols <= pos) works unchanged. (Or, with
         CUDA, the Triton kernel reads the blocks in place.) Prefill reads
         nothing back: each packed prompt attends over itself.

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

from model.gpt2.gpt import GPTConfig
from inference.base_kv_cache import BaseKVCache
from inference.continuous_batch.paged.block.block_allocator import SCRATCH


class PagedKVCache(BaseKVCache):

    def __init__(self, config: GPTConfig, num_blocks, block_size, max_seqs,
                 max_blocks_per_seq, device, dtype=torch.float32, prefilled_rows=0):
        """
        num_blocks          blocks in the pool, per layer, incl. the scratch block (0)
        block_size          tokens per block
        max_seqs            decode rows: the batch every decode step runs
        max_blocks_per_seq  block-table width: the most blocks one request may own
                            (W = max_blocks_per_seq * block_size)
        prefilled_rows      extra block-table / pos rows for requests that are prefilled
                            but not decoding yet (the scheduler's prefilled buffer);
                            promote() moves one into a decode row
        """
        head_size = config.n_embd // config.n_head
        shape = (num_blocks, block_size, config.n_head, head_size)

        # zeros, not empty: masked columns still flow through attention as
        # 0-weight * value, and an uninitialised NaN would poison that product.
        self.key = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.value = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

        # Rows [0, max_seqs) are the decode batch. Rows [max_seqs, max_seqs + prefilled_rows)
        # hold PREFILLED requests waiting for a decode row: their blocks are written, they
        # just are not decoding yet. promote() moves one into a free decode row.
        rows = max_seqs + prefilled_rows + 1 
        self.block_table_gpu = torch.full((rows, max_blocks_per_seq), SCRATCH,
                                          dtype=torch.long, device=device)
        self.pos_gpu = torch.zeros(rows, dtype=torch.long, device=device)
        self.pad_row = rows - 1

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.width = max_blocks_per_seq * block_size
        self.batch_size = max_seqs
        self.device = device

        # On CUDA, attention is always a Triton kernel, no gather, no mask. Decode: the
        # kernel (decode_attention.py) reads each row's K/V through the block table in
        # place. Prefill: FlashAttention over the packed prompts (prefill_attention.py) --
        # a prompt's keys are its own fresh k/v. Triton runs only on NVIDIA GPUs, so on
        # CPU / Apple GPUs the same attention runs as SDPA.
        self.use_kernel = str(device).startswith('cuda')
        self.needs_mask = not self.use_kernel          # the kernels bound each row themselves

        # per-forward indices (begin_forward), shared by every layer's write and attention
        self._decode_rows = torch.arange(max_seqs, device=device)
        self._slots = None                             # flat pool index of each token written
        self._decode_seq_lens = None                   # (max_seqs,) pos + 1, decode only

    # -- block table (host decisions → device) -------------------------------

    def set_blocks(self, req_pool_idx, blocks):
        """Record which physical blocks `req_pool_idx` owns, in logical order.

        Unused entries point at scratch. One small host→device copy, made only when
        a row's blocks change (admit, grow, retire) -- never on an ordinary step.
        """
        if len(blocks) > self.max_blocks_per_seq:
            raise ValueError(f'{len(blocks)} blocks exceed max_blocks_per_seq={self.max_blocks_per_seq}')
        padded = list(blocks) + [SCRATCH] * (self.max_blocks_per_seq - len(blocks))
        self.block_table_gpu[req_pool_idx] = torch.tensor(padded, dtype=torch.long, device=self.device)

    def slots(self, rows, positions):
        """Flat pool index for each (row, position) pair -- tensors in, tensor out."""
        physical = self.block_table_gpu[rows, positions // self.block_size]
        return physical * self.block_size + positions % self.block_size

    # -- writes and reads ------------------------------------------------------

    def begin_forward(self, prefill=None):
        """Once per forward, before the layers: the pool slot of every token this forward
        writes, and (decode) every row's sequence length. All layers write the same slots
        and attend over the same lengths, so computing them per layer (~10 small kernels
        each) was pure overhead.

        prefill=None   -- decode: decode row r writes at its pos, attends over pos + 1.
        prefill=batch  -- packed prompts: token i at prefill.rows[i], prefill.positions[i].
        """
        if prefill is not None:
            self._slots = self.slots(prefill.rows, prefill.positions)
        else:
            positions = self.pos_gpu[:self.max_seqs]
            self._slots = self.slots(self._decode_rows, positions)
            self._decode_seq_lens = positions + 1

    def write(self, layer_idx, key, value, prefill=None):
        """Store this layer's k/v in the pool at the slots begin_forward computed, one
        index_copy_ per tensor. Decode: B rows, one token each (rows [0, B)). Prefill: the
        packed prompts, (1, H, sum(lens), D). Nothing is read back."""
        assert layer_idx is not None, 'layer_idx is required when a cache is passed'
        assert self._slots is not None, 'begin_forward() before the layers write'
        B, H, T, D = key.shape
        assert prefill is not None or T == 1, f'decode writes one token per row, got {T}'
        slots = self._slots[:B * T]                               # (B*T,), in (row, t) order
        self._flat(self.key[layer_idx]).index_copy_(0, slots, self._tokens_first(key))
        self._flat(self.value[layer_idx]).index_copy_(0, slots, self._tokens_first(value))

    def read(self, layer_idx, num_rows):
        """Rows [0, num_rows) gathered to W in logical order: a copy (see module doc)."""
        rows = slice(0, num_rows)
        return self._gather(self.key[layer_idx], rows), self._gather(self.value[layer_idx], rows)

    # -- attention ----------------------------------------------------------

    def prefill_attend(self, layer_idx, query, key, value, prefill):
        """Store the packed prompts, then attend within each prompt.

        Every prompt starts at position 0 (no cached prefix), so its keys are exactly its
        own packed k/v -- attention needs nothing from the pool. CUDA: one FlashAttention-
        style Triton launch over the packed q/k/v (prefill_attention.py): tiles of queries
        x tiles of keys on tensor cores, each query bounded to its own prompt. Elsewhere:
        one SDPA per prompt.
        """
        self.write(layer_idx, key, value, prefill)
        if not self.use_kernel:
            return self.sdpa_prefill(query, key, value, prefill)
        from inference.continuous_batch.paged.prefill_attention import prefill_attention
        out = prefill_attention(query[0].transpose(0, 1), key[0].transpose(0, 1),
                                value[0].transpose(0, 1), prefill.positions)   # (sum(lens), H, D)
        return out.transpose(0, 1).unsqueeze(0)                         # (1, H, sum(lens), D)

    def decode_attend(self, layer_idx, query, key, value, attn_mask):
        """Store the new k/v, then attend over each row's history.

        CUDA: ONE kernel does both -- stores the new k/v at the slots begin_forward
        computed, then reads each row's real blocks in place (seq_len = pos + 1): no
        index_copy_, no gather, no mask. Elsewhere: write, gather to W, masked SDPA.
        Idle rows sit at pos 0 on the scratch block: they write and attend one scratch
        slot, finite, and their output is unused.
        """
        B = key.size(0)
        if not self.use_kernel:
            self.write(layer_idx, key, value)
            return self.sdpa_decode(layer_idx, query, attn_mask)
        from inference.continuous_batch.paged.decode_attention import decode_attention
        out = decode_attention(query[:, :, 0], key[:, :, 0], value[:, :, 0],      # the kernel
                               self.key[layer_idx], self.value[layer_idx],        # also writes
                               self.block_table_gpu[:B], self._decode_seq_lens[:B],
                               self._slots[:B])
        return out.unsqueeze(2)

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

    def advance(self, T, prefill=None, active=None):
        """Move the write pointer once per forward, never per layer."""
        if prefill is not None:
            self.pos_gpu[prefill.slots_gpu] = prefill.lens_gpu   # each prompt written from 0
        elif active is not None:
            self.pos_gpu[:self.max_seqs] += active.long() * T   # decode rows still generating
        else:
            self.pos_gpu[:self.max_seqs] += T

    def seq_len(self):
        """The decode rows' positions -- a view, so a captured graph sees every update."""
        return self.pos_gpu[:self.max_seqs]

    def promote(self, prefilled_row, decode_row):
        """Move a prefilled request from its prefilled row into a free decode row: copy its
        block-table row and position, then free the prefilled row. Small in-place writes --
        the K/V themselves stay where they are in the pool."""
        self.block_table_gpu[decode_row] = self.block_table_gpu[prefilled_row]
        self.pos_gpu[decode_row] = self.pos_gpu[prefilled_row]
        self.release(prefilled_row)

    def release(self, req_pool_idx):
        """Point the row back at scratch. The pool is NOT cleared: stale blocks are
        harmless because the mask hides everything at or past pos."""
        self.pos_gpu[req_pool_idx] = 0
        self.block_table_gpu[req_pool_idx] = SCRATCH

    def reset(self):
        self.pos_gpu.zero_()
        self.block_table_gpu.fill_(SCRATCH)
