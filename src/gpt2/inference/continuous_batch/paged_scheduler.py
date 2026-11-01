"""Continuous batching over a paged KV cache.

A subclass of Scheduler, so the diff between the two classes is exactly what paging
changes. The model, the mask, the decode forward and sampling are inherited; only
memory decisions differ:

  Scheduler        every slot owns a full-width row (W tokens), reserved for life.
  PagedScheduler   every request owns ceil((L + max_new) / block_size) blocks from a
                   shared pool -- memory proportional to the request -- and returns
                   them on retire.

Policy: reserve at admission. A request gets every block it can ever need when it
is admitted, so nothing can run out mid-generation and no preemption is needed.
(On-demand growth -- one block each time pos crosses a block boundary -- is the
next step, and is what makes preemption necessary.)

Decisions use host mirrors (row_blocks, row_len, row_active, num_active), so the
scheduler never reads GPU tensors to decide anything.

Logging: every admit, wait and retire is reported through the standard `logging`
module (logger "gpt2.inference.continuous_batch.paged_scheduler") -- INFO for each request's lifecycle,
DEBUG for every decode step. The library never configures logging itself; an app
turns it on (see scripts/chat_server.py --log-level). Log calls read host values only,
so they never force a GPU sync.
"""

import logging
from math import ceil

import torch

from gpt2.inference.continuous_batch.block_allocator import BlockAllocator
from gpt2.inference.continuous_batch.paged_cache import PagedKVCache
from gpt2.inference.continuous_batch.scheduler import Scheduler

log = logging.getLogger(__name__)


class PagedScheduler(Scheduler):

    def __init__(self, model, enc, config, max_seqs=4, block_size=16, num_blocks=None,
                 max_tokens=None, watermark=0, device='cpu', dtype=torch.float32):
        # Not calling super().__init__: it would allocate a RowKVCache we never use.
        self.model = model
        self.enc = enc
        self.config = config
        self.device = device

        self.B = max_seqs
        self.W = max_tokens if max_tokens is not None else config.block_size
        if self.W % block_size:
            # the gathered width must equal W, or the inherited mask no longer lines up
            raise ValueError(f'block_size {block_size} must divide max_tokens {self.W}')
        self.bs = block_size
        if num_blocks is None:                    # same memory as a RowKVCache of max_seqs rows
            num_blocks = 1 + max_seqs * (self.W // block_size)
        self.watermark = watermark

        self.allocator = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(config, num_blocks, block_size, max_seqs,
                                  self.W // block_size, device, dtype)

        # device state the inherited decode path uses
        self.active = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.next_token = torch.zeros(self.B, 1, dtype=torch.long, device=device)
        self.slot_req = [None] * self.B
        self.generated = {}

        # host mirrors: every scheduling decision reads these, never the GPU
        self.row_blocks = [[] for _ in range(self.B)]
        self.row_len = [0] * self.B
        self.row_active = [False] * self.B
        self.num_active = 0

        self.peak_active = 0                      # most requests resident at once
        self.peak_blocks = 0                      # most blocks in use at once
        self.decode_steps = 0
        self._waiting_req = None                  # log a wait once, not every step

    # -- the three operations --------------------------------------------------

    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Admit queued requests, in order, while free slots AND free blocks allow."""
        for row in range(self.B):
            if not queue:
                return
            if self.row_active[row]:
                continue

            req, _ = queue[0]
            ids = prompt_ids[req]
            L = len(ids)
            if L == 0 or L + max_new_tokens > self.W:
                raise ValueError(f'request {req}: prompt of {L} tokens does not fit '
                                 f'(cache width {self.W} leaves room for {self.W - max_new_tokens})')

            need = ceil((L + max_new_tokens) / self.bs)
            if need > self.allocator.capacity:
                # would wait forever: even an empty pool cannot hold it
                raise ValueError(f'request {req}: needs {need} blocks, the pool can never '
                                 f'provide more than {self.allocator.capacity} -- it will never fit')

            # Headroom only matters when something is running that could free blocks.
            # With nothing active the whole pool is free and the check above guarantees
            # the request fits -- so ignoring the watermark here prevents a spin.
            headroom = self.watermark if self.num_active else 0
            if self.allocator.num_free - need < headroom:
                if not self.num_active:           # unreachable given the checks above
                    raise RuntimeError('no progress: the head request cannot be admitted '
                                       'into an empty pool')
                if self._waiting_req != req:
                    self._waiting_req = req
                    log.info('wait   req %d needs %d blocks, %d free (headroom %d) -- '
                             'holding the queue until a request retires',
                             req, need, self.allocator.num_free, headroom)
                return                            # FCFS: wait; later requests may not skip ahead

            queue.popleft()
            blocks = self.allocator.allocate(need)
            self.row_blocks[row] = blocks
            self.cache.set_blocks(row, blocks)

            # prefill -- identical to Scheduler._admit
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            pos_ids = torch.arange(L, device=self.device).unsqueeze(0)
            mask = (torch.arange(self.W, device=self.device) < L).unsqueeze(0)
            logits, _ = self.model(idx, kv_cache=self.cache, pos_ids=pos_ids,
                                   attn_mask=mask, cache_row=row)

            tok = sampler(logits[:, -1, :self.enc.n_vocab])
            self.next_token[row] = tok
            self.active[row] = True
            self.slot_req[row] = req
            self.generated[req] = [int(tok)]

            self.row_active[row] = True
            self.row_len[row] = L
            self.num_active += 1
            self.peak_active = max(self.peak_active, self.num_active)
            self.peak_blocks = max(self.peak_blocks,
                                   self.allocator.capacity - self.allocator.num_free)
            self._waiting_req = None
            log.info('admit  req %d -> row %d: prompt %d tokens, %d blocks %s, '
                     '%d/%d blocks free, %d active',
                     req, row, L, need, _short(blocks), self.allocator.num_free,
                     self.allocator.capacity, self.num_active)

    def _decode(self, sampler):
        super()._decode(sampler)
        self.decode_steps += 1
        for row in range(self.B):
            if self.row_active[row]:
                self.row_len[row] += 1                # one token written per active row
        if log.isEnabledFor(logging.DEBUG):
            log.debug('decode step %d: %d active rows, lengths %s',
                      self.decode_steps, self.num_active,
                      [self.row_len[r] for r in range(self.B) if self.row_active[r]])

    def _retire(self, results, prompt_ids, max_new_tokens):
        eot = self.enc.eot_token
        for row in range(self.B):
            if not self.row_active[row]:
                continue

            req = self.slot_req[row]
            gen = self.generated[req]
            full = self.row_len[row] + 1 >= self.W      # host mirror, not cache.pos

            if gen[-1] == eot or len(gen) >= max_new_tokens or full:
                results[req] = gen
                reason = ('end-of-text' if gen[-1] == eot else
                          'max_new_tokens' if len(gen) >= max_new_tokens else 'cache width')
                freed = len(self.row_blocks[row])
                self.allocator.free(self.row_blocks[row])
                self.row_blocks[row] = []
                self.cache.release(row)               # back to scratch; pool not cleared
                self.active[row] = False
                self.slot_req[row] = None
                self.row_active[row] = False
                self.row_len[row] = 0
                self.num_active -= 1
                log.info('retire req %d <- row %d: %s after %d tokens, freed %d blocks, '
                         '%d/%d blocks free, %d active',
                         req, row, reason, len(gen), freed, self.allocator.num_free,
                         self.allocator.capacity, self.num_active)

    def _reset(self):
        super()._reset()                              # cache, active, next_token, slot_req, generated
        self.allocator.reset()
        self.row_blocks = [[] for _ in range(self.B)]
        self.row_len = [0] * self.B
        self.row_active = [False] * self.B
        self.num_active = 0
        self.peak_active = 0
        self.peak_blocks = 0
        self.decode_steps = 0
        self._waiting_req = None


def _short(blocks, n=6):
    """[3, 4, 5, …] -- block lists can be long; the first few say enough."""
    return str(blocks) if len(blocks) <= n else f'[{", ".join(map(str, blocks[:n]))}, …]'
