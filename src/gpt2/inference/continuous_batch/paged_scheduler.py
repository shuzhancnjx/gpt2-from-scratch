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

Decisions use host mirrors (row_blocks_cpu, row_len_cpu, active_cpu, num_active_cpu), so the
scheduler never reads GPU tensors to decide anything.

Logging: every admit, wait and retire is reported through the standard `logging`
module (logger "gpt2.inference.continuous_batch.paged_scheduler") -- INFO for each request's lifecycle,
DEBUG for every decode step. The library never configures logging itself; an app
turns it on (see scripts/chat_server.py --log-level). Log calls read host values only,
so they never force a GPU sync.
"""

import logging
from collections import deque
from math import ceil

import torch

from gpt2.inference.continuous_batch.block_allocator import BlockAllocator
from gpt2.inference.continuous_batch.paged_cache import PagedKVCache
from gpt2.inference.continuous_batch.scheduler import Scheduler

log = logging.getLogger(__name__)


class PagedScheduler(Scheduler):

    def __init__(self, model, enc, config, max_seqs=4, block_size=16, num_blocks=None,
                 max_tokens=None, watermark=0, device='cpu', dtype=torch.float32,
                 decode_kernel=False):
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
                                  self.W // block_size, device, dtype,
                                  decode_kernel=decode_kernel)   # Triton decode attention

        # device state the inherited decode path uses
        self.active_gpu = torch.zeros(self.B, dtype=torch.bool, device=device)   # read by the model (mask, advance)
        self.next_token_gpu = torch.zeros(self.B, 1, dtype=torch.long, device=device)
        self.slot_req_cpu = [None] * self.B
        self.generated_cpu = {}

        # host mirrors: every scheduling decision reads these, never the GPU
        self.row_blocks_cpu = [[] for _ in range(self.B)]
        self.row_len_cpu = [0] * self.B
        self.active_cpu = [False] * self.B                 # CPU mirror of active_gpu: every decision reads this
        self.num_active_cpu = 0

        self.peak_active = 0                      # most requests resident at once
        self.peak_blocks = 0                      # most blocks in use at once
        self.decode_steps = 0
        self._waiting_req = None                  # log a wait once, not every step
        self.graph = None                         # set by capture_graph(); None = eager decode
        self.graph_sampler = None                 # the sampler recorded inside the graph

    # -- the main loop: decide from host mirrors --------------------------------

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens=32, sampler=None):
        """Same loop as Scheduler.run, but every decision reads a host int, never the GPU."""
        if self.graph is not None:
            # the graph replays the sampler it was captured with; it cannot switch
            if sampler is not None and sampler is not self.graph_sampler:
                raise ValueError('this graph was captured with a different sampler: pass the same '
                                 'one (or none), or capture_graph(sampler) again')
            sampler = self.graph_sampler
        sampler = sampler or _greedy

        queue = deque(enumerate(prompts))
        prompt_ids = [self.enc.encode(p) for p in prompts]
        results = {}

        self._reset()
        while queue or self.num_active_cpu:            # was bool(self.active_gpu.any()): a sync
            self._admit(queue, prompt_ids, sampler, max_new_tokens)
            if self.num_active_cpu:                    # was a second sync
                self._decode(sampler)
            self._retire(results, prompt_ids, max_new_tokens)

        return [prompts[i] + self.enc.decode(results[i]) for i in range(len(prompts))]

    # -- the three operations --------------------------------------------------

    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Admit queued requests, in order, while free slots AND free blocks allow."""
        for row in range(self.B):
            if not queue:
                return
            if self.active_cpu[row]:
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
            headroom = self.watermark if self.num_active_cpu else 0
            if self.allocator.num_free - need < headroom:
                if not self.num_active_cpu:           # unreachable given the checks above
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
            self.row_blocks_cpu[row] = blocks
            self.cache.set_blocks(row, blocks)

            # prefill -- identical to Scheduler._admit
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            pos_ids = torch.arange(L, device=self.device).unsqueeze(0)
            mask = (torch.arange(self.W, device=self.device) < L).unsqueeze(0)
            logits, _ = self.model(idx, kv_cache=self.cache, pos_ids=pos_ids,
                                   attn_mask=mask, cache_row=row)

            tok = sampler(logits[:, -1, :self.enc.n_vocab])
            self.next_token_gpu[row] = tok
            self.active_gpu[row] = True
            self.slot_req_cpu[row] = req
            self.generated_cpu[req] = [int(tok)]

            self.active_cpu[row] = True
            self.row_len_cpu[row] = L
            self.num_active_cpu += 1
            self.peak_active = max(self.peak_active, self.num_active_cpu)
            self.peak_blocks = max(self.peak_blocks,
                                   self.allocator.capacity - self.allocator.num_free)
            self._waiting_req = None
            log.info('admit  req %d -> row %d: prompt %d tokens, %d blocks %s, '
                     '%d/%d blocks free, %d active',
                     req, row, L, need, _short(blocks), self.allocator.num_free,
                     self.allocator.capacity, self.num_active_cpu)

    def _decode(self, sampler):
        """One token for every row. Graph replay if captured, else the eager forward."""
        if self.graph is not None:
            self.graph.replay()                       # one launch replaces hundreds
        else:
            logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                                   pos_ids=self.cache.pos_gpu.unsqueeze(1),
                                   attn_mask=self._mask(), cache_active=self.active_gpu)
            tok = sampler(logits[:, -1, :self.enc.n_vocab])    # (B, 1), still on the GPU
            self.next_token_gpu.copy_(tok)                # in place: the same buffer every step
        self._record_tokens()

    def _record_tokens(self):
        """The one device -> host copy per step, then host bookkeeping only."""
        toks = self.next_token_gpu.view(-1).tolist()      # THE sync: all B tokens in one copy
        for row in range(self.B):
            if self.active_cpu[row]:                  # host mirror, was `if self.active_gpu[row]`
                self.generated_cpu[self.slot_req_cpu[row]].append(toks[row])
                self.row_len_cpu[row] += 1                # one token written per active row
        self.decode_steps += 1
        if log.isEnabledFor(logging.DEBUG):
            log.debug('decode step %d: %d active rows, lengths %s',
                      self.decode_steps, self.num_active_cpu,
                      [self.row_len_cpu[r] for r in range(self.B) if self.active_cpu[r]])

    def _retire(self, results, prompt_ids, max_new_tokens):
        eot = self.enc.eot_token
        for row in range(self.B):
            if not self.active_cpu[row]:
                continue

            req = self.slot_req_cpu[row]
            gen = self.generated_cpu[req]
            full = self.row_len_cpu[row] + 1 >= self.W      # host mirror, not cache.pos_gpu

            if gen[-1] == eot or len(gen) >= max_new_tokens or full:
                results[req] = gen
                reason = ('end-of-text' if gen[-1] == eot else
                          'max_new_tokens' if len(gen) >= max_new_tokens else 'cache width')
                freed = len(self.row_blocks_cpu[row])
                self.allocator.free(self.row_blocks_cpu[row])
                self.row_blocks_cpu[row] = []
                self.cache.release(row)               # back to scratch; pool not cleared
                self.active_gpu[row] = False
                self.slot_req_cpu[row] = None
                self.active_cpu[row] = False
                self.row_len_cpu[row] = 0
                self.num_active_cpu -= 1
                log.info('retire req %d <- row %d: %s after %d tokens, freed %d blocks, '
                         '%d/%d blocks free, %d active',
                         req, row, reason, len(gen), freed, self.allocator.num_free,
                         self.allocator.capacity, self.num_active_cpu)

    def _reset(self):
        super()._reset()                              # cache, active, next_token, slot_req, generated
        self.allocator.reset()
        self.row_blocks_cpu = [[] for _ in range(self.B)]
        self.row_len_cpu = [0] * self.B
        self.active_cpu = [False] * self.B
        self.num_active_cpu = 0
        self.peak_active = 0
        self.peak_blocks = 0
        self.decode_steps = 0
        self._waiting_req = None

    # -- CUDA graph: record the decode step once, replay it every step -----------

    def capture_graph(self, sampler=None):
        """Record one decode step -- forward, mask and `sampler` (default greedy). Call
        once after __init__, on CUDA. run() then always uses this sampler.

        The sampler must be capturable: tensor ops only, no reads back to the CPU.
        topk / softmax / multinomial (scripts/chat_server.py make_sampler) qualify.

        Safe to capture with no requests: every row is inactive, so the forward writes
        only to the scratch block and advance() adds 0 to pos. Prefill stays eager
        (prompt lengths vary), but writes the same pool, table and pos buffers the
        graph reads -- so a replay sees newly admitted rows automatically.
        """
        assert str(self.device).startswith('cuda'), 'CUDA graphs need a CUDA device'
        self.graph_sampler = sampler or _greedy
        with torch.inference_mode():
            self._reset()
            # warm up on a side stream: cuBLAS/SDPA choose kernels and allocate
            # workspaces BEFORE capture, so no first-time setup is recorded
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._graph_body()
            torch.cuda.current_stream().wait_stream(side)
            self._reset()                             # undo the warm-up, in place

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):        # record, don't run
                self._graph_body()

    def _graph_body(self):
        """The captured work. Reads and writes only fixed buffers: next_token, active,
        cache.pos_gpu, cache.block_table_gpu and the K/V pools -- all updated in place."""
        logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                               pos_ids=self.cache.pos_gpu.unsqueeze(1),
                               attn_mask=self._mask(), cache_active=self.active_gpu)
        tok = self.graph_sampler(logits[:, -1, :self.enc.n_vocab])        # on the GPU
        self.next_token_gpu.copy_(tok)


def _greedy(logits):
    return logits.argmax(-1, keepdim=True)


def _short(blocks, n=6):
    """[3, 4, 5, …] -- block lists can be long; the first few say enough."""
    return str(blocks) if len(blocks) <= n else f'[{", ".join(map(str, blocks[:n]))}, …]'
