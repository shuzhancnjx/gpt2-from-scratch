"""Continuous batching: keep every slot busy instead of padding to the slowest row.

The batched sampler in gpt2/inference runs one batch to completion. A batch of 8
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
    slot index is a position again and pos_ids is just cache.pos.

  - The mask is derived, not maintained. Recomputing it from pos each step means
    the mask and the cache cannot drift out of sync, which is the failure mode
    that incremental mask updates kept producing.
"""

from collections import deque

import torch

from gpt2.inference.continuous_batch.row_cache import RowKVCache


class Scheduler:

    def __init__(self, model, enc, config, batch_size=4, max_tokens=None,
                 device='cpu', dtype=torch.float32):
        self.model = model
        self.enc = enc
        self.config = config
        self.device = device

        self.B = batch_size
        self.W = max_tokens if max_tokens is not None else config.block_size
        self.cache = RowKVCache(batch_size, device, config, max_tokens=self.W, dtype=dtype)

        self.active = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.next_token = torch.zeros(self.B, 1, dtype=torch.long, device=device)
        self.slot_req = [None] * self.B          # which request occupies each slot
        self.generated = {}                      # request id -> list[int]

    # -- public ------------------------------------------------------------

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens=32, sampler=None):
        """Generate for every prompt, keeping the batch full. Returns text in order."""
        sampler = sampler or (lambda logits: logits.argmax(-1, keepdim=True))

        queue = deque(enumerate(prompts))
        prompt_ids = [self.enc.encode(p) for p in prompts]
        results = {}

        self._reset()
        while queue or bool(self.active.any()):
            self._admit(queue, prompt_ids, sampler, max_new_tokens)
            if bool(self.active.any()):
                self._decode(sampler)
            self._retire(results, prompt_ids, max_new_tokens)

        return [prompts[i] + self.enc.decode(results[i]) for i in range(len(prompts))]

    # -- the three operations ----------------------------------------------

    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Prefill queued requests into free slots, one batch-of-1 forward each.

        Prompts of different lengths never share a tensor, which is exactly why
        no padding is needed anywhere in this file.
        """
        for row in range(self.B):
            if self.active[row] or not queue:
                continue

            req, _ = queue.popleft()
            ids = prompt_ids[req]
            L = len(ids)
            if L == 0 or L + max_new_tokens > self.W:
                results_len = self.W - max_new_tokens
                raise ValueError(f'request {req}: prompt of {L} tokens does not fit '
                                 f'(cache width {self.W} leaves room for {results_len})')

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

    def _decode(self, sampler):
        """One token for every row, each writing at its own pos."""
        logits, _ = self.model(self.next_token, kv_cache=self.cache,
                               pos_ids=self.cache.pos.unsqueeze(1),
                               attn_mask=self._mask(), cache_active=self.active)

        tok = sampler(logits[:, -1, :self.enc.n_vocab])
        self.next_token = tok

        for row in range(self.B):
            if self.active[row]:
                self.generated[self.slot_req[row]].append(int(tok[row]))

    def _retire(self, results, prompt_ids, max_new_tokens):
        eot = self.enc.eot_token
        for row in range(self.B):
            if not self.active[row]:
                continue

            req = self.slot_req[row]
            gen = self.generated[req]
            full = int(self.cache.pos[row]) + 1 >= self.W      # no room for another write

            if gen[-1] == eot or len(gen) >= max_new_tokens or full:
                results[req] = gen
                self.cache.release(row)
                self.active[row] = False
                self.slot_req[row] = None

    # -- helpers -----------------------------------------------------------

    def _mask(self):
        """Which cache columns each row may attend to, derived from pos.

        `<=` not `<`: the token written this step lands at column pos[r] and has
        to see itself. `& active` blanks free slots, whose rows are all zeros.
        """
        cols = torch.arange(self.W, device=self.device)
        return (cols <= self.cache.pos.unsqueeze(1)) & self.active.unsqueeze(1)

    def _reset(self):
        self.cache.reset()
        self.active.zero_()
        self.next_token.zero_()
        self.slot_req = [None] * self.B
        self.generated = {}
