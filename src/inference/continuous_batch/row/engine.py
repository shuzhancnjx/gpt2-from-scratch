"""RowEngine: continuous batching over full-width KV rows -- the simple baseline.

The batched sampler in static_batch runs one batch to completion. A batch of 8 where
seven prompts stop at 20 tokens and one runs to 500 spends 480 steps computing seven rows
of masked filler. Here the batch width is fixed but its *occupants* rotate: a row that
finishes is released and refilled from the queue, so the tensors keep their shape while
the work in them stays real.

    while queued or busy:
        admit()     # fill free rows: prefill the new requests (one packed forward)
        decode()    # one (B, 1) forward for every row: one new token each
        retire()    # eot or cap -> emit, release the row

Two things fall out of RowKVCache's per-row pointers:

  - No left-padding. Each row owns columns [0, pos[r]) of its own cache row, so a column
    index is a position again and pos_ids is just cache.pos_gpu.
  - The mask is derived, not maintained (ModelRunner.mask), so it cannot drift from pos.

The model calls are the shared ModelRunner's (model_runner.py). The policy is inline and
reads the GPU to decide (active rows, positions, tokens): a sync per read, kept on purpose
-- PagedEngine decides from CPU state instead.
"""

from collections import deque

import torch

from inference.continuous_batch.continuous_engine import ContinuousEngine
from inference.continuous_batch.model_runner import ModelRunner
from inference.continuous_batch.row.row_cache import RowKVCache


class RowEngine(ContinuousEngine):

    def __init__(self, model, enc, config, batch_size=4, max_tokens=None,
                 device='cpu', dtype=torch.float32):
        self.enc = enc
        self.B = batch_size
        self.W = max_tokens if max_tokens is not None else config.block_size
        cache = RowKVCache(batch_size, device, config, max_tokens=self.W, dtype=dtype)
        self.runner = ModelRunner(model, enc, cache, batch_size, self.W, device)
        self.slot_req_cpu = [None] * self.B          # which request occupies each row
        self.generated = {}                          # request id -> list[int]

    def _run(self, prompts, sampling_params):
        queue = deque(enumerate(prompts))
        prompt_ids = [self.enc.encode(p) for p in prompts]
        results = {}

        self.runner.start(sampling_params)
        self.slot_req_cpu, self.generated = [None] * self.B, {}
        while queue or self._has_active():
            self._admit(queue, prompt_ids, sampling_params)
            if self._has_active():
                self.runner.decode(sampling_params)
                self._record_tokens()
            self._retire(results, sampling_params.max_new_tokens)
        return results

    def _has_active(self):
        return bool(self.runner.active_gpu.any())     # a sync, twice per loop: on purpose

    def _admit(self, queue, prompt_ids, sampling_params):
        """Fill every free row from the queue, then prefill them all in one forward."""
        rows, reqs = [], []
        for row in range(self.B):
            if not queue:
                break
            if self.runner.active_gpu[row]:
                continue
            req, _ = queue.popleft()
            L = len(prompt_ids[req])
            if L == 0 or L + sampling_params.max_new_tokens > self.W:
                raise ValueError(f'request {req}: prompt of {L} tokens does not fit '
                                 f'(cache width {self.W} leaves room for {self.W - sampling_params.max_new_tokens})')
            rows.append(row)
            reqs.append(req)
        if not rows:
            return

        first = self.runner.prefill(rows, [prompt_ids[r] for r in reqs], sampling_params)
        first = first.view(-1).tolist()                          # the one sync for all of them
        self.runner.apply(started=list(zip(rows, first)))
        for row, req, tok in zip(rows, reqs, first):
            self.slot_req_cpu[row] = req
            self.generated[req] = [tok]

    def _record_tokens(self):
        """Each active row's new token to its request -- a sync per row, on purpose."""
        for row in range(self.B):
            if self.runner.active_gpu[row]:
                self.generated[self.slot_req_cpu[row]].append(int(self.runner.next_token_gpu[row]))

    def _retire(self, results, max_new_tokens):
        eot, cache, stopped = self.enc.eot_token, self.runner.cache, []
        for row in range(self.B):
            if not self.runner.active_gpu[row]:
                continue
            req = self.slot_req_cpu[row]
            gen = self.generated[req]
            full = int(cache.pos_gpu[row]) + 1 >= self.W         # no room for another write
            if gen[-1] == eot or len(gen) >= max_new_tokens or full:
                results[req] = gen
                cache.release(row)
                self.slot_req_cpu[row] = None
                stopped.append(row)
        self.runner.apply(stopped=stopped)
