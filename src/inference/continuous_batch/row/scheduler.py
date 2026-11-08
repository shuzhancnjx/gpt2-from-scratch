"""Continuous batching: keep every slot busy instead of padding to the slowest row.

The batched sampler in inference runs one batch to completion. A batch of 8
where seven prompts stop at 20 tokens and one runs to 500 spends 480 steps
computing seven rows of masked filler.

Here the batch width is fixed but its *occupants* rotate. A slot that finishes
is released and immediately refilled from the queue, so the tensors stay the
same shape while the work in them stays real.

    while queued or busy:
        admit()     # fill free slots: prefill one row, sample its first token
        decode()    # one (B, 1) forward for every occupied row
        retire()    # eot or cap -> emit, release the slot

Two things fall out of RowKVCache's per-row pointers and are worth noticing:

  - No left-padding. Each row owns slots [0, pos[r]) of its own cache row, so a
    slot index is a position again and pos_ids is just cache.pos_gpu.

  - The mask is derived, not maintained. Recomputing it from pos each step means
    the mask and the cache cannot drift out of sync, which is the failure mode
    that incremental mask updates kept producing.
"""

import torch

from inference.continuous_batch.base_scheduler import BaseScheduler
from inference.continuous_batch.row.row_cache import RowKVCache


class Scheduler(BaseScheduler):

    def __init__(self, model, enc, config, batch_size=4, max_tokens=None,
                 device='cpu', dtype=torch.float32):
        super().__init__(model, enc, config, batch_size, max_tokens, device)
        self.cache = RowKVCache(batch_size, device, config, max_tokens=self.W, dtype=dtype)

    def _has_active(self):
        # Reads the GPU: a sync, twice per loop. Kept on purpose -- this engine is the
        # simple baseline; PagedScheduler answers from a host counter instead.
        return bool(self.active_gpu.any())

    # -- the three operations ----------------------------------------------

    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Fill every free slot from the queue, then prefill them all in one forward."""
        rows, reqs = [], []
        for row in range(self.B):
            if not queue:
                break
            if self.active_gpu[row]:
                continue
            req, _ = queue.popleft()
            L = len(prompt_ids[req])
            if L == 0 or L + max_new_tokens > self.W:
                raise ValueError(f'request {req}: prompt of {L} tokens does not fit '
                                 f'(cache width {self.W} leaves room for {self.W - max_new_tokens})')
            rows.append(row)
            reqs.append(req)
        if not rows:
            return

        first = self._prefill(rows, [prompt_ids[r] for r in reqs], sampler)
        for row, req, tok in zip(rows, reqs, first):
            self.slot_req_cpu[row] = req
            self.generated_cpu[req] = [tok]

    def _decode(self, sampler):
        """One token for every row, each writing at its own pos."""
        logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                               pos_ids=self.cache.pos_gpu.unsqueeze(1),
                               attn_mask=self._mask(), cache_active=self.active_gpu)

        tok = sampler(logits[:, -1, :self.enc.n_vocab])
        self.next_token_gpu = tok

        for row in range(self.B):
            if self.active_gpu[row]:
                self.generated_cpu[self.slot_req_cpu[row]].append(int(tok[row]))

    def _retire(self, results, prompt_ids, max_new_tokens):
        eot = self.enc.eot_token
        for row in range(self.B):
            if not self.active_gpu[row]:
                continue

            req = self.slot_req_cpu[row]
            gen = self.generated_cpu[req]
            full = int(self.cache.pos_gpu[row]) + 1 >= self.W      # no room for another write

            if gen[-1] == eot or len(gen) >= max_new_tokens or full:
                results[req] = gen
                self.cache.release(row)
                self.active_gpu[row] = False
                self.slot_req_cpu[row] = None
