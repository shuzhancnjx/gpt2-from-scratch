"""What every continuous-batching engine shares.

    while queued or busy:
        admit()     # fill free rows: prefill a request, sample its first token
        decode()    # one forward for every row: one new token each
        retire()    # finished -> emit its text, free the row

BaseScheduler owns that loop, the per-row device state the model reads, the mask and
the reset. A subclass decides how KV memory is laid out (its `cache`) and how each of
the operations works:

    row/scheduler.py         Scheduler       one full-width row per request; decisions
                                             read the GPU (the simple baseline)
    paged/paged_scheduler.py PagedScheduler  blocks from a shared pool; decisions read
                                             host mirrors, optional CUDA-graph decode
"""

from abc import abstractmethod
from collections import deque

import torch

from inference.base_engine import BaseEngine
from inference.base_kv_cache import PrefillBatch
from inference.sampling import SamplingParams, greedy


class BaseScheduler(BaseEngine):

    def __init__(self, model, enc, config, num_rows, max_tokens, device):
        self.model = model
        self.enc = enc
        self.config = config
        self.device = device

        self.B = num_rows
        self.W = max_tokens if max_tokens is not None else config.block_size
        self.cache = None                            # set by the subclass: its KV layout

        # device state the model reads every decode step
        self.active_gpu = torch.zeros(self.B, dtype=torch.bool, device=device)   # mask, advance
        self.next_token_gpu = torch.zeros(self.B, 1, dtype=torch.long, device=device)

        self.slot_req_cpu = [None] * self.B          # which request occupies each row
        self.generated_cpu = {}                      # request id -> list[int]

    # -- the loop ------------------------------------------------------------

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens=32, sampler=None):
        """Generate for every prompt, keeping the rows full. Returns text in order."""
        results = self._run(prompts, max_new_tokens, self._choose_sampler(sampler))
        return [prompts[i] + self.enc.decode(results[i]) for i in range(len(prompts))]

    @torch.inference_mode()
    def generate(self, prompts, params=None):
        params = params or SamplingParams()
        prompts = list(prompts)
        results = self._run(prompts, params.max_new_tokens, self._sampler_for(params))
        return [self._completion(results[i]) for i in range(len(prompts))]

    def _sampler_for(self, params):
        """The sampler a generate() call uses. PagedScheduler overrides this to pick
        (or capture) the CUDA graph recorded with these params."""
        return self._choose_sampler(params.sampler())

    def _run(self, prompts, max_new_tokens, sampler):
        """The loop itself. Returns {prompt index: generated token ids}."""
        queue = deque(enumerate(prompts))
        prompt_ids = [self.enc.encode(p) for p in prompts]
        results = {}

        self._reset()
        while queue or self._has_active():
            self._admit(queue, prompt_ids, sampler, max_new_tokens)
            if self._has_active():
                self._decode(sampler)
            self._retire(results, prompt_ids, max_new_tokens)
        return results

    # -- what each engine decides --------------------------------------------

    @abstractmethod
    def _has_active(self):
        """Is any row generating? Called twice per loop iteration."""

    @abstractmethod
    def _admit(self, queue, prompt_ids, sampler, max_new_tokens):
        """Move queued requests into free rows and prefill them."""

    @abstractmethod
    def _decode(self, sampler):
        """One new token for every active row."""

    @abstractmethod
    def _retire(self, results, prompt_ids, max_new_tokens):
        """Emit finished requests and free their rows."""

    def _choose_sampler(self, sampler):
        return sampler or greedy

    # -- shared helpers --------------------------------------------------------

    def _prefill(self, rows, prompts, sampler):
        """Prefill several fresh requests in ONE forward: their prompts packed end to
        end, no padding. rows: each request's slot. prompts: their token-id lists.

        Writes each request's k/v into its slot and samples every first token. Returns
        them twice: on the device (N, 1), and as a list -- one device -> host copy for the
        whole batch.
        """
        batch = PrefillBatch.build(rows, [len(p) for p in prompts], self.device)
        idx = torch.tensor([t for p in prompts for t in p], dtype=torch.long,
                           device=self.device).unsqueeze(0)               # (1, sum(lens))
        logits, _ = self.model(idx, kv_cache=self.cache, pos_ids=batch.positions.unsqueeze(0),
                               prefill=batch)                            # (1, N, vocab): last tokens
        first = sampler(logits[0, :, :self.enc.n_vocab])                 # (N, 1)
        return first, first.view(-1).tolist()                             # the one sync

    def _mask(self):
        """Which cache columns each row may attend to, derived from pos.

        `<=` not `<`: the token written this step lands at column pos[r] and has
        to see itself. `& active` blanks free rows.
        """
        cols = torch.arange(self.W, device=self.device)
        return (cols <= self.cache.seq_len().unsqueeze(1)) & self.active_gpu.unsqueeze(1)

    def _reset(self):
        """Back to empty, IN PLACE: a captured CUDA graph holds these buffers' addresses."""
        self.cache.reset()
        self.active_gpu.zero_()
        self.next_token_gpu.zero_()
        self.slot_req_cpu = [None] * self.B
        self.generated_cpu = {}
