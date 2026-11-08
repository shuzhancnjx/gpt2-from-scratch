"""The decode step as CUDA graphs -- one per sampler setting (CUDA only).

Mirrors PrefillGraphs (graph/prefill_graph.py); the scheduler holds one of each:

                  key                       buffers                     run() samples
    prefill   padded token count     its own, one set per bucket    after the replay
    decode    sampler setting        the scheduler's live state     inside the graph

The decode step has fixed shapes already: the batch is always max_seqs rows, and idle
rows write only to the scratch block -- so no padding. Its buffers are the scheduler's
own state, which admission, promotion and retirement update in place:

    sched.next_token_gpu   (B, 1)  input tokens; the graph writes the sampled ones back
    sched.active_gpu       (B,)    which rows decode (mask, advance)
    sched.cache.pos_gpu, sched.cache.block_table_gpu, and the K/V pools

so decode() only replays. The sampler is recorded inside the graph, which is why the
key is the sampler setting: (temperature, top_k, top_p) from the run's SamplingParams --
the scheduler passes those sampling_params down its loop, so the key is derived where it is used.

Capture happens in prepare(), before a run, with every row inactive: the warm-up runs
the step for real, and its advance() must not move any live row's position.
"""

import torch

from inference.cuda_graph import BaseGraphs
from inference.continuous_batch.base_scheduler import BaseScheduler
from inference.sampling import SamplingParams


class DecodeGraphs(BaseGraphs):
    """Decode graphs for one PagedScheduler: key = sampler setting, run(sampling_params) -> tokens."""

    GREEDY = (0.0, None, None)                  # the key of plain argmax decoding

    def __init__(self, sched):
        super().__init__(sched)
        self._captured = set()

    def key(self, sampling_params):
        """(temperature, top_k, top_p): what the graph records. max_new_tokens is not in it."""
        return (sampling_params.temperature, sampling_params.top_k, sampling_params.top_p)

    def prepare(self, key):
        """Capture between runs, never mid-run: the scheduler is reset around the capture so
        the warm-up (which runs the step for real) touches only scratch."""
        if key in self._captured:
            return
        with torch.inference_mode():
            self.sched._reset()                 # every row inactive: warm-up advances nothing
            self.graphs.get(key)
            self.sched._reset()                 # undo whatever the warm-up wrote, in place
        self._captured.add(key)

    def _build(self, key):
        """The base scheduler's eager decode step, with this setting's sampler. Called as
        BaseScheduler._decode explicitly: PagedScheduler._decode replays THIS graph, so
        recording it would record the graph itself."""
        sched, settings = self.sched, SamplingParams(temperature=key[0], top_k=key[1], top_p=key[2])
        return lambda: BaseScheduler._decode(sched, settings)

    def run(self, sampling_params):
        """One decode step for every row; returns sched.next_token_gpu, now the sampled
        tokens. The graph for these settings must have been prepared before the run."""
        key = self.key(sampling_params)
        assert key in self._captured, 'prepare(key) before the run'
        return self.graphs.get(key).replay()