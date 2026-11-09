"""PagedScheduler: who runs, when -- the paged engine's policy (CPU only, no model, no GPU).

A request moves through three places; the scheduler decides every move:

    waiting --schedule()--> prefilled buffer --promote()--> a decode row --update()--> finished
    (FCFS)    blocks + a      (prefill done,     (decoded every     (end-of-text, max_new_tokens
              prefilled row    first token out)   step)               or a full cache)

    schedule(max_new)        -> SchedulerOutput: requests promoted now, requests to prefill now
    prefilled(admissions, first_tokens)    after the runner's prefill: into the buffer
    promote()                -> requests given a decode row (the second chance in a step)
    update(tokens, max_new)  -> requests finished this step (after the runner's decode)

Admission policy (what stops _choose):
  - no free prefilled row (the buffer is full)
  - max_prefill_tokens: one step's prompt tokens are capped; the first prompt always goes
  - too few free blocks, keeping `watermark` spare while others hold blocks -- and then the
    whole queue waits (FCFS: no skipping ahead, so long prompts cannot starve)
  - a prompt that can NEVER fit raises

Logging: admit / wait / retire at INFO (logger "inference.continuous_batch.paged.paged_scheduler").
"""

import logging
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Admission:
    """One request, carried through every stage."""
    req: int                                    # its index in the run's prompts
    ids: list                                   # prompt token ids
    need: int = 0                               # blocks reserved (prompt + max_new_tokens)
    slot: int = None                            # its prefilled row, while in the buffer
    blocks: list = None                         # its blocks in the pool
    row: int = None                             # its decode row, once promoted
    tokens: list = field(default_factory=list)  # generated tokens (the first one from prefill)
    written: int = 0                            # tokens in its KV cache: prompt + decode steps


@dataclass
class SchedulerOutput:
    """What one step does before decoding, decided by PagedScheduler.schedule()."""
    promote: list                               # Admissions given a decode row now
    prefill: list                               # Admissions to prefill now (blocks reserved)


@dataclass
class Stats:
    peak_active: int = 0                        # most requests decoding at once
    peak_blocks: int = 0                        # most blocks in use at once
    decode_steps: int = 0


class PagedScheduler:

    def __init__(self, blocks, max_seqs, max_tokens, eot, watermark=0, max_prefill_tokens=None):
        """
        blocks              the BlockManager (asked for room; keeps the block table)
        max_seqs            decode rows
        max_tokens          W: the most tokens one request may hold
        eot                 the end-of-text token id
        watermark           blocks kept free while others hold blocks
        max_prefill_tokens  prompt tokens one step may prefill (None: no cap)
        """
        self.blocks, self.B, self.W, self.eot = blocks, max_seqs, max_tokens, eot
        self.watermark, self.max_prefill_tokens = watermark, max_prefill_tokens
        self.reset()

    def reset(self):
        self.waiting = deque()                  # Admissions not yet prefilled, oldest first
        self.prefilled = deque()                # prefilled, waiting for a decode row
        self.rows = [None] * self.B             # rows[r]: the Admission decoding in row r
        self.finished = {}                      # req -> generated tokens
        self.stats = Stats()
        self._waiting_req = None                # log a wait once, not every step

    def add(self, prompt_ids):
        for req, ids in enumerate(prompt_ids):
            self.waiting.append(Admission(req=req, ids=ids))

    # -- state ----------------------------------------------------------------------

    @property
    def decoding(self):
        return [a for a in self.rows if a is not None]

    @property
    def num_decoding(self):
        return sum(a is not None for a in self.rows)

    def done(self):
        return not self.waiting and not self.prefilled and not self.num_decoding

    # -- the step -------------------------------------------------------------------

    def schedule(self, max_new_tokens):
        """Promote what is ready, then choose (and reserve room for) this step's prefills."""
        return SchedulerOutput(promote=self.promote(), prefill=self._choose(max_new_tokens))

    def prefilled_with(self, admissions, first_tokens):
        """The runner prefilled them: each has its first token; into the buffer."""
        for a, tok in zip(admissions, first_tokens):
            a.tokens, a.written = [tok], len(a.ids)
            self.prefilled.append(a)
        self.stats.peak_blocks = max(self.stats.peak_blocks, self.blocks.num_used)
        log.info('prefill %d prompt(s), %d tokens in one forward -> %d prefilled, waiting '
                 'for rows', len(admissions), sum(len(a.ids) for a in admissions), len(self.prefilled))

    def promote(self):
        """Each free decode row takes the oldest prefilled request."""
        promoted = []
        for row in range(self.B):
            if not self.prefilled:
                break
            if self.rows[row] is not None:
                continue
            a = self.prefilled.popleft()
            self.blocks.move_to_decode(a.slot, row)
            a.row, a.slot, self.rows[row] = row, None, a
            promoted.append(a)
            log.info('admit  req %d -> row %d: prompt %d tokens, %d blocks %s, %d/%d blocks free, '
                     '%d active', a.req, row, len(a.ids), len(a.blocks), _short(a.blocks),
                     self.blocks.num_free, self.blocks.capacity, self.num_decoding)
        self.stats.peak_active = max(self.stats.peak_active, self.num_decoding)
        return promoted

    def update(self, tokens, max_new_tokens):
        """After a decode step: each decoding request takes its token (tokens[row]); those
        that are done retire, their blocks and rows freed. Returns the retired Admissions."""
        self.stats.decode_steps += 1
        retired = []
        for a in self.decoding:
            a.tokens.append(tokens[a.row])
            a.written += 1                      # the token fed this step is now in the cache
            reason = ('end-of-text' if a.tokens[-1] == self.eot else
                      'max_new_tokens' if len(a.tokens) >= max_new_tokens else
                      'cache width' if a.written + 1 >= self.W else None)
            if reason:
                self.blocks.release(a.row, a.blocks)
                self.rows[a.row] = None
                self.finished[a.req] = a.tokens
                retired.append(a)
                log.info('retire req %d <- row %d: %s after %d tokens, freed %d blocks, '
                         '%d/%d blocks free, %d active', a.req, a.row, reason, len(a.tokens),
                         len(a.blocks), self.blocks.num_free, self.blocks.capacity,
                         self.num_decoding)
        return retired

    # -- admission policy -----------------------------------------------------------

    def _choose(self, max_new_tokens):
        """Waiting -> to prefill, in order, while the rules allow; reserves their room."""
        chosen, tokens = [], 0
        while self.waiting and self.blocks.free_prefilled_rows:
            a = self.waiting[0]
            self._check_fits(a, max_new_tokens)
            if self.max_prefill_tokens and chosen and tokens + len(a.ids) > self.max_prefill_tokens:
                break                               # budget spent; the first prompt always goes
            if not self._has_room(a.need, busy=self.num_decoding + len(self.prefilled) + len(chosen)):
                self._note_wait(a)
                break                               # FCFS: later requests may not skip ahead
            self.waiting.popleft()
            a.slot, a.blocks = self.blocks.reserve(a.need)
            chosen.append(a)
            tokens += len(a.ids)
            self._waiting_req = None
        return chosen

    def _check_fits(self, a, max_new_tokens):
        """Size the request; raise if it can never fit."""
        L = len(a.ids)
        if L == 0 or L + max_new_tokens > self.W:
            raise ValueError(f'request {a.req}: prompt of {L} tokens does not fit '
                             f'(cache width {self.W} leaves room for {self.W - max_new_tokens})')
        a.need = self.blocks.blocks_for(L + max_new_tokens)
        if a.need > self.blocks.capacity:
            raise ValueError(f'request {a.req}: needs {a.need} blocks, the pool can never '
                             f'provide more than {self.blocks.capacity} -- it will never fit')

    def _has_room(self, need, busy):
        """Enough free blocks, keeping `watermark` spare while anything else holds blocks.
        With nothing busy the whole pool is free and _check_fits guarantees a fit -- so the
        watermark is ignored there, which prevents a spin."""
        headroom = self.watermark if busy else 0
        if self.blocks.num_free - need >= headroom:
            return True
        if not busy:                                # unreachable given _check_fits
            raise RuntimeError('no progress: the head request cannot be admitted into an empty pool')
        return False

    def _note_wait(self, a):
        if self._waiting_req != a.req:
            self._waiting_req = a.req
            log.info('wait   req %d needs %d blocks, %d free -- holding the queue until a '
                     'request retires', a.req, a.need, self.blocks.num_free)


def _short(blocks, n=6):
    """[3, 4, 5, …] -- block lists can be long; the first few say enough."""
    return str(blocks) if len(blocks) <= n else f'[{", ".join(map(str, blocks[:n]))}, …]'
