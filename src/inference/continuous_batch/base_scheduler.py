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
from inference.sampling import SamplingParams


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
    def run(self, prompts, max_new_tokens=32):
        """Greedy convenience: prompt + completion for every prompt, in order."""
        results = self._run(prompts, SamplingParams(max_new_tokens=max_new_tokens))
        return [prompts[i] + self.enc.decode(results[i]) for i in range(len(prompts))]

    @torch.inference_mode()
    def generate(self, prompts, sampling_params=None):
        prompts = list(prompts)
        results = self._run(prompts, sampling_params or SamplingParams())
        return [self._completion(results[i]) for i in range(len(prompts))]

    def _run(self, prompts, sampling_params):
        """The loop itself. Returns {prompt index: generated token ids}. Every stage gets
        the SamplingParams and derives what it needs (a sampler, a graph key)."""
        queue = deque(enumerate(prompts))
        prompt_ids = [self.enc.encode(p) for p in prompts]
        results = {}

        self._prepare(sampling_params)
        self._reset()
        while queue or self._has_active():
            self._admit(queue, prompt_ids, sampling_params)
            if self._has_active():
                self._decode(sampling_params)         # the computation: next token per row
                self._record_tokens()                 # the bookkeeping: each engine's own
            self._retire(results, prompt_ids, sampling_params.max_new_tokens)
        return results

    # -- what each engine decides --------------------------------------------

    @abstractmethod
    def _has_active(self):
        """Is any row generating? Called twice per loop iteration."""

    @abstractmethod
    def _admit(self, queue, prompt_ids, sampling_params):
        """Move queued requests into free rows and prefill them."""

    @abstractmethod
    def _record_tokens(self):
        """After _decode: hand each active row's new token (in next_token_gpu) to its
        request -- the engine's own bookkeeping."""

    @abstractmethod
    def _retire(self, results, prompt_ids, max_new_tokens):
        """Emit finished requests and free their rows."""

    def _prepare(self, sampling_params):
        """Before a run, with no request active. PagedScheduler captures its decode graph
        for these settings here."""

    # -- shared helpers --------------------------------------------------------

    def _prefill(self, rows, prompts, sampling_params):
        """Prefill several fresh requests in ONE forward: their prompts packed end to
        end, no padding. rows: each request's slot. prompts: their token-id lists.

        Writes each request's k/v into its slot, samples every first token and returns them
        on the device (N, 1). No sync and no bookkeeping -- like _decode; the caller does
        the one device -> host copy. PagedScheduler overrides it to replay a prefill graph.
        """
        batch = PrefillBatch.build(rows, [len(p) for p in prompts], self.device)
        idx = torch.tensor([t for p in prompts for t in p], dtype=torch.long,
                           device=self.device).unsqueeze(0)               # (1, sum(lens))
        logits = self._prefill_logits(idx, batch)                        # (1, N, vocab)
        return sampling_params.sampler()(logits[0, :, :self.enc.n_vocab])         # (N, 1)

    def _prefill_logits(self, idx, batch):
        """The packed prefill's model call: token ids (1, sum(lens)) and their PrefillBatch
        in, logits at each prompt's last token (1, N, vocab) out. Writes the prompts' k/v.

        Tensors in, tensors out, no CPU reads -- so PrefillGraphs records exactly this as
        its body, on its fixed buffers.
        """
        logits, _ = self.model(idx, kv_cache=self.cache, pos_ids=batch.positions.unsqueeze(0),
                               prefill=batch)
        return logits

    def _decode(self, sampling_params):
        """One decode step for every row: run the model on the scheduler's buffers, sample
        the last position, write the tokens INTO next_token_gpu (in place) and return it.
        No bookkeeping -- that is _record_tokens.

        Tensor ops only, no CPU reads -- so DecodeGraphs records exactly this as its body.
        PagedScheduler overrides it to replay that graph.
        """
        logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                               pos_ids=self.cache.seq_len().unsqueeze(1),
                               attn_mask=self._mask(), cache_active=self.active_gpu)
        sampler = sampling_params.sampler()
        return self.next_token_gpu.copy_(sampler(logits[:, -1, :self.enc.n_vocab]))   # (B, 1)

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
