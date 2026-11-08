"""Continuous batching over a paged KV cache.

Both engines subclass BaseScheduler (../base_scheduler.py), which owns the loop, the mask
and the reset. What differs is memory and how decisions are made:

  Scheduler        every slot owns a full-width row (W tokens), reserved for life.
  PagedScheduler   every request owns ceil((L + max_new) / block_size) blocks from a
                   shared pool -- memory proportional to the request -- and returns
                   them on retire.

High-level flow -- one loop iteration (BaseScheduler.run) is one step:

  ┌───────┐   ┌────────────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐
  │ queue │──►│ _admit_prefill │──►│ prefilled │──►│ _promote │──►│ _decode │──►│ _retire │
  └───────┘   └────────┬───────┘   └───────────┘   └──────────┘   └─────────┘   └────┬────┘
                       ▲                                                             │
                       └──────────────────── rows + blocks freed ────────────────────┘

  - queue: requests not yet admitted, served first come, first served.
  - _admit_prefill (producer): takes requests off the queue while there is a free
    prefilled row, enough free blocks (the queue waits rather than skip ahead) and room
    in max_prefill_tokens; then prefills them all in ONE packed forward.
  - prefilled (bounded buffer, at most prefilled_rows): requests whose blocks are written
    and first token sampled, waiting for a decode row. Each holds its blocks meanwhile.
  - _promote (consumer): a free decode row takes the oldest prefilled request -- its
    block-table row and pos are copied over; the K/V stay where they are in the pool.
  - _decode: ONE forward for all max_seqs decode rows, one new token each (CUDA: graph
    replay + the Triton paged kernel). One device -> host copy of the tokens per step.
  - _retire: a request that hit end-of-text, max_new_tokens or a full cache returns its
    text and gives its row and blocks back -- feeding the next _promote / _admit_prefill.
  - Order within a step: _promote, _admit_prefill, _promote, _decode, _retire. The second
    _promote lets requests prefilled this step start decoding at once if rows are free.

Memory layout of the block table / pos (PagedKVCache):

    rows [0, max_seqs)                          decode rows: decoded every step
    rows [max_seqs, max_seqs + prefilled_rows)  prefilled rows: waiting for a decode row

Policy: reserve at admission. A request gets every block it can ever need when it
is admitted, so nothing can run out mid-generation and no preemption is needed.
(On-demand growth -- one block each time pos crosses a block boundary -- is the
next step, and is what makes preemption necessary.)

Decisions use host mirrors (row_blocks_cpu, row_len_cpu, active_cpu, num_active_cpu), so the
scheduler never reads GPU tensors to decide anything.

Logging: every admit, wait and retire is reported through the standard `logging`
module (logger "inference.continuous_batch.paged.paged_scheduler") -- INFO for each request's lifecycle,
DEBUG for every decode step. The library never configures logging itself; an app
turns it on (see scripts/chat_server.py --log-level). Log calls read host values only,
so they never force a GPU sync.
"""

import logging
from collections import deque
from math import ceil

import torch

from inference.continuous_batch.paged.block_allocator import BlockAllocator
from inference.continuous_batch.paged.paged_cache import PagedKVCache
from inference.continuous_batch.base_scheduler import BaseScheduler
from inference.sampling import greedy

log = logging.getLogger(__name__)


class PagedScheduler(BaseScheduler):

    def __init__(self, model, enc, config, max_seqs=4, block_size=16, num_blocks=None,
                 max_tokens=None, watermark=0, device='cpu', dtype=torch.float32,
                 cuda_graph=False, prefilled_rows=None, max_prefill_tokens=None):
        """
        model, enc, config  the loaded GPT, its tokenizer, its GPTConfig
        max_seqs            decode rows: the most requests decoding at once (the batch size
                            of every decode step)
        block_size          tokens per KV block (16)
        num_blocks          size of the shared KV pool in blocks, incl. the scratch block.
                            Default: as much memory as a RowKVCache of max_seqs rows
        max_tokens          W, the most tokens one request may hold (prompt + generated).
                            Default: the model's context length
        watermark           blocks kept free while other requests hold blocks, so running
                            requests are not starved by a burst of admissions
        device, dtype       where the pool lives and its precision
        cuda_graph          generate(): replay the decode step as a CUDA graph, one graph
                            per sampler setting (CUDA only)
        prefilled_rows      capacity of the prefilled buffer: how many requests may be
                            prefilled and waiting for a decode row at once. Each holds its
                            blocks while it waits. Default: max_seqs (a full next batch).
                            At least 1: every request passes through this buffer
        max_prefill_tokens  prompt tokens one step may prefill. Bounds the stall a burst of
                            new requests causes the running ones; the first prompt of a step
                            always goes, however long. None = no cap
        """
        super().__init__(model, enc, config, max_seqs, max_tokens, device)
        
        if self.W % block_size:
            # the gathered width must equal W, or the shared mask no longer lines up
            raise ValueError(f'block_size {block_size} must divide max_tokens {self.W}')
        self.bs = block_size
        if num_blocks is None:                    # same memory as a RowKVCache of max_seqs rows
            num_blocks = 1 + max_seqs * (self.W // block_size)
        self.watermark = watermark

        self.allocator = BlockAllocator(num_blocks)
        self.prefilled_rows = max_seqs if prefilled_rows is None else prefilled_rows
        if self.prefilled_rows < 1:
            raise ValueError('prefilled_rows must be >= 1: every request is prefilled into '
                             'the prefilled buffer before it takes a decode row')
        self.max_prefill_tokens = max_prefill_tokens      # None: no cap per step
        self.cache = PagedKVCache(config, num_blocks, block_size, max_seqs,
                                  self.W // block_size, device, dtype,
                                  prefilled_rows=self.prefilled_rows)

        # host mirrors: every scheduling decision reads these, never the GPU
        self.row_blocks_cpu = [[] for _ in range(self.B)]
        self.row_len_cpu = [0] * self.B
        self.active_cpu = [False] * self.B                 # CPU mirror of active_gpu: every decision reads this
        self.num_active_cpu = 0
        # prefilled requests waiting for a decode row, oldest first, and free prefilled rows
        self.prefilled = deque()
        self.free_prefilled_rows = list(range(self.B, self.B + self.prefilled_rows))

        self.peak_active = 0                      # most requests resident at once
        self.peak_blocks = 0                      # most blocks in use at once
        self.decode_steps = 0
        self._waiting_req = None                  # log a wait once, not every step
        self.graph = None                         # set by capture_graph(); None = eager decode
        self.graph_sampler = None                 # the sampler recorded inside the graph
        self.cuda_graph = cuda_graph              # generate(): capture one graph per SamplingParams
        self._graphs = {}                         # (temperature, top_k, top_p) -> (graph, sampler)

    # -- decisions from host mirrors ------------------------------------------

    def _has_active(self):
        # host ints, no sync. Prefilled requests count: they still need rows and decoding.
        return self.num_active_cpu > 0 or bool(self.prefilled)

    def _sampler_for(self, params):
        if not self.cuda_graph:
            return self._choose_sampler(params.sampler())
        key = (params.temperature, params.top_k, params.top_p)   # max_new_tokens: not in the graph
        if key not in self._graphs:                    # first call with these settings
            self.capture_graph(params.sampler())
            self._graphs[key] = (self.graph, self.graph_sampler)
        self.graph, self.graph_sampler = self._graphs[key]
        return self.graph_sampler

    def _choose_sampler(self, sampler):
        if self.graph is not None:
            # the graph replays the sampler it was captured with; it cannot switch
            if sampler is not None and sampler is not self.graph_sampler:
                raise ValueError('this graph was captured with a different sampler: pass the same '
                                 'one (or none), or capture_graph(sampler) again')
            return self.graph_sampler
        return sampler or greedy

    # -- the three operations --------------------------------------------------

    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Two stages: prefilled requests take free decode rows; waiting requests are
        prefilled (one packed forward) into the prefilled buffer; then promote again so fresh
        prefills start decoding this step if rows are free."""
        self._promote()
        self._admit_prefill(queue, prompt_ids, sampler, max_new_tokens)
        self._promote()

    def _admit_prefill(self, queue, prompt_ids, sampler, max_new_tokens):
        """Waiting -> prefilled, in order, while prefilled rows, free blocks and the token
        budget allow; then ONE packed prefill for all of them."""
        slots, reqs, tokens = [], [], 0
        while queue and len(slots) < len(self.free_prefilled_rows):
            req, _ = queue[0]
            L = len(prompt_ids[req])
            if L == 0 or L + max_new_tokens > self.W:
                raise ValueError(f'request {req}: prompt of {L} tokens does not fit '
                                 f'(cache width {self.W} leaves room for {self.W - max_new_tokens})')
            if self.max_prefill_tokens and slots and tokens + L > self.max_prefill_tokens:
                break                                 # budget spent; the first prompt always goes

            need = ceil((L + max_new_tokens) / self.bs)
            if need > self.allocator.capacity:
                # would wait forever: even an empty pool cannot hold it
                raise ValueError(f'request {req}: needs {need} blocks, the pool can never '
                                 f'provide more than {self.allocator.capacity} -- it will never fit')

            # Headroom only matters when something holds blocks that will come back. With
            # nothing running, prefilled or being admitted, the whole pool is free and the check
            # above guarantees the request fits -- so ignoring the watermark prevents a spin.
            busy = self.num_active_cpu + len(self.prefilled) + len(slots)
            headroom = self.watermark if busy else 0
            if self.allocator.num_free - need < headroom:
                if not busy:                          # unreachable given the checks above
                    raise RuntimeError('no progress: the head request cannot be admitted '
                                       'into an empty pool')
                if self._waiting_req != req:
                    self._waiting_req = req
                    log.info('wait   req %d needs %d blocks, %d free (headroom %d) -- '
                             'holding the queue until a request retires',
                             req, need, self.allocator.num_free, headroom)
                break                                 # FCFS: later requests may not skip ahead

            queue.popleft()
            slot = self.free_prefilled_rows[len(slots)]
            blocks = self.allocator.allocate(need)
            self.cache.set_blocks(slot, blocks)       # before the prefill writes into them
            slots.append(slot)
            reqs.append((req, blocks, L))
            tokens += L
            self._waiting_req = None
        if not slots:
            return

        del self.free_prefilled_rows[:len(slots)]
        _, first = self._prefill(slots, [prompt_ids[r] for r, _, _ in reqs], sampler)
        for slot, (req, blocks, L), tok in zip(slots, reqs, first):
            self.generated_cpu[req] = [tok]           # the first token exists from here on
            self.prefilled.append(dict(slot=slot, req=req, blocks=blocks, length=L, token=tok))
        self.peak_blocks = max(self.peak_blocks, self.allocator.capacity - self.allocator.num_free)
        log.info('prefill %d prompt(s), %d tokens in one forward -> %d prefilled, waiting '
                 'for rows', len(slots), tokens, len(self.prefilled))

    def _promote(self):
        """Prefilled -> decoding: each free decode row takes the oldest prefilled request."""
        for row in range(self.B):
            if not self.prefilled:
                break
            if self.active_cpu[row]:
                continue
            s = self.prefilled.popleft()
            self.cache.promote(s['slot'], row)        # block-table row + pos, in place
            self.free_prefilled_rows.append(s['slot'])
            self.next_token_gpu[row] = s['token']
            self.active_gpu[row] = True
            self.slot_req_cpu[row] = s['req']
            self.active_cpu[row] = True
            self.row_len_cpu[row] = s['length']
            self.row_blocks_cpu[row] = s['blocks']
            self.num_active_cpu += 1
            log.info('admit  req %d -> row %d: prompt %d tokens, %d blocks %s, '
                     '%d/%d blocks free, %d active',
                     s['req'], row, s['length'], len(s['blocks']), _short(s['blocks']),
                     self.allocator.num_free, self.allocator.capacity, self.num_active_cpu)
        self.peak_active = max(self.peak_active, self.num_active_cpu)

    def _decode(self, sampler):
        """One token for every row. Graph replay if captured, else the eager forward."""
        if self.graph is not None:
            self.graph.replay()                       # one launch replaces hundreds
        else:
            logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                                   pos_ids=self.cache.seq_len().unsqueeze(1),
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
        super()._reset()                              # cache, active_gpu, next_token_gpu, slot_req, generated
        self.allocator.reset()
        self.row_blocks_cpu = [[] for _ in range(self.B)]
        self.row_len_cpu = [0] * self.B
        self.active_cpu = [False] * self.B
        self.num_active_cpu = 0
        self.prefilled = deque()
        self.free_prefilled_rows = list(range(self.B, self.B + self.prefilled_rows))
        self.peak_active = 0
        self.peak_blocks = 0
        self.decode_steps = 0
        self._waiting_req = None

    # -- CUDA graph: record the decode step once, replay it every step -----------

    def capture_graph(self, sampler=None):
        """Record one decode step -- forward, mask and `sampler` (default greedy). Call
        once after __init__, on CUDA. run() then always uses this sampler.

        The sampler must be capturable: tensor ops only, no reads back to the CPU.
        Everything in inference/sampling.py qualifies.

        Safe to capture with no requests: every row is inactive, so the forward writes
        only to the scratch block and advance() adds 0 to pos. Prefill stays eager
        (prompt lengths vary), but writes the same pool, table and pos buffers the
        graph reads -- so a replay sees newly admitted rows automatically.
        """
        assert str(self.device).startswith('cuda'), 'CUDA graphs need a CUDA device'
        self.graph_sampler = sampler or greedy
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
                               pos_ids=self.cache.seq_len().unsqueeze(1),
                               attn_mask=self._mask(), cache_active=self.active_gpu)
        tok = self.graph_sampler(logits[:, -1, :self.enc.n_vocab])        # on the GPU
        self.next_token_gpu.copy_(tok)


def _short(blocks, n=6):
    """[3, 4, 5, …] -- block lists can be long; the first few say enough."""
    return str(blocks) if len(blocks) <= n else f'[{", ".join(map(str, blocks[:n]))}, …]'
