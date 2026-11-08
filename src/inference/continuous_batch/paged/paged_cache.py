"""A paged KV cache with the same interface as RowKVCache.

RowKVCache reserves a full-width row (W slots) per request up front. Here the KV
memory is one pool of fixed-size blocks per layer,

    key[layer], value[layer]   (num_blocks, block_size, n_head, head_dim)

and each request owns only the blocks it was given, recorded in a block table:

    block_table   (max_seqs, max_blocks_per_seq)   row r, logical block j -> physical block

It implements BaseKVCache like RowKVCache does, so gpt.py cannot tell them apart:

  write: every new token goes to   slot = block_table[row, pos // bs] * bs + pos % bs,
         computed on the device from tensors (no int(), no sync), then one
         index_copy_ into a flat (num_blocks * bs, H, D) view of the pool.
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
from inference.continuous_batch.paged.block_allocator import SCRATCH


class PagedKVCache(BaseKVCache):

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

        # On CUDA, attention is always the Triton paged kernel: it reads K/V through the
        # block table in place -- no gather, no mask. Decode: one row per request; prefill:
        # one row per prompt token, all prompts in one launch. Triton runs only on NVIDIA
        # GPUs, so on CPU / Apple GPUs the same attention runs as SDPA.
        self.use_kernel = str(device).startswith('cuda')

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

    def write(self, layer_idx, key, value, prefill=None):
        """Store this layer's k/v in the pool, one index_copy_ per tensor.

        prefill=None   -- decode: every row writes one token at its own pos.
        prefill=batch  -- packed prompts: token i goes to slot prefill.rows[i],
                          position prefill.positions[i]. Nothing is read back.
        """
        assert layer_idx is not None, 'layer_idx is required when a cache is passed'
        B, H, T, D = key.shape
        if prefill is not None:
            rows, positions = prefill.rows, prefill.positions
        else:
            assert T == 1, f'decode writes one token per row, got {T}'
            rows, positions = torch.arange(B, device=self.device), self.pos_gpu[:B]
        slots = self.slots(rows, positions)                       # (B*T,), in (row, t) order
        self._flat(self.key[layer_idx]).index_copy_(0, slots, self._tokens_first(key))
        self._flat(self.value[layer_idx]).index_copy_(0, slots, self._tokens_first(value))

    def read(self, layer_idx, num_rows):
        """Rows [0, num_rows) gathered to W in logical order: a copy (see module doc)."""
        rows = slice(0, num_rows)
        return self._gather(self.key[layer_idx], rows), self._gather(self.value[layer_idx], rows)

    # -- attention ----------------------------------------------------------

    def prefill_attend(self, layer_idx, query, key, value, prefill):
        """Store the packed prompts, then attend within each prompt.

        CUDA: ONE kernel launch in which every prompt token is a row -- token t of a
        prompt reads its own request's blocks up to position t (seq_len = t + 1): causal,
        never across prompts. Elsewhere: one SDPA per prompt.
        """
        self.write(layer_idx, key, value, prefill)
        if not self.use_kernel:
            return self.sdpa_prefill(query, key, value, prefill)
        from inference.continuous_batch.paged.paged_attention import paged_attention
        out = paged_attention(query[0].transpose(0, 1),                 # (sum(lens), H, D)
                              self.key[layer_idx], self.value[layer_idx],
                              self.block_table_gpu[prefill.rows],       # each token's request
                              prefill.positions + 1)                    # causal
        return out.transpose(0, 1).unsqueeze(0)                         # (1, H, sum(lens), D)

    def decode_attend(self, layer_idx, query, key, value, attn_mask):
        """Store the new k/v, then attend over each row's history.

        CUDA: the kernel reads each row's real blocks in place (seq_len = pos + 1) -- no
        gather, no mask. Elsewhere: gather to W and masked SDPA.
        Idle rows sit at pos 0 on the scratch block: they attend one scratch slot, finite,
        and their output is unused.
        """
        B = key.size(0)
        self.write(layer_idx, key, value)
        if not self.use_kernel:
            return self.sdpa_decode(layer_idx, query, attn_mask)
        from inference.continuous_batch.paged.paged_attention import paged_attention
        out = paged_attention(query[:, :, 0], self.key[layer_idx], self.value[layer_idx],
                              self.block_table_gpu[:B], self.pos_gpu[:B] + 1)
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
            self.pos_gpu += active.long() * T          # only rows still generating
        else:
            self.pos_gpu += T

    def seq_len(self):
        return self.pos_gpu

    def release(self, req_pool_idx):
        """Point the row back at scratch. The pool is NOT cleared: stale blocks are
        harmless because the mask hides everything at or past pos."""
        self.pos_gpu[req_pool_idx] = 0
        self.block_table_gpu[req_pool_idx] = SCRATCH

    def reset(self):
        self.pos_gpu.zero_()
        self.block_table_gpu.fill_(SCRATCH)
