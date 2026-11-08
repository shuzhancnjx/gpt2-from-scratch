"""The decode step as CUDA graphs -- one per sampler setting (CUDA only).

Mirrors PrefillGraphs (graph/prefill_graph.py); the ModelRunner holds one of each:

                  key                       buffers                     run() samples
    prefill   padded token count     its own, one set per bucket    after the replay
    decode    sampler setting        the runner's live state        inside the graph

The decode step has fixed shapes already: the batch is always max_seqs rows, and idle
rows write only to the scratch block -- so no padding. Its buffers are the runner's
own state, which admission, promotion and retirement update in place:

    runner.next_token_gpu  (B, 1)  input tokens; the graph writes the sampled ones back
    runner.active_gpu      (B,)    which rows decode (mask, advance)
    runner.cache.pos_gpu, runner.cache.block_table_gpu, and the K/V pools

so decode() only replays. The sampler is recorded inside the graph, which is why the
key is the sampler setting: (temperature, top_k, top_p) from the run's SamplingParams --
the engine passes those sampling_params down each step, so the key is derived where it is used.

Capture happens in prepare(), before a run, with every row inactive: the warm-up runs
the step for real, and its advance() must not move any live row's position.
"""

import torch

from inference.cuda_graph import BaseGraphs
from inference.sampling import SamplingParams


class DecodeGraphs(BaseGraphs):
    """Decode graphs for one PagedEngine: key = sampler setting, run(sampling_params) -> tokens."""

    GREEDY = (0.0, None, None)                  # the key of plain argmax decoding

    def __init__(self, runner):
        super().__init__(runner)
        self._captured = set()

    def key(self, sampling_params):
        """(temperature, top_k, top_p): what the graph records. max_new_tokens is not in it."""
        return (sampling_params.temperature, sampling_params.top_k, sampling_params.top_p)

    def prepare(self, key):
        """Capture between runs, never mid-run: the runner is reset around the capture so
        the warm-up (which runs the step for real) touches only scratch."""
        if key in self._captured:
            return
        with torch.inference_mode():
            self.runner.reset()                 # every row inactive: warm-up advances nothing
            self.graphs.get(key)
            self.runner.reset()                 # undo whatever the warm-up wrote, in place
        self._captured.add(key)

    def _build(self, key):
        """The runner's eager decode step (decode_eager) with this setting's sampler -- the
        graph records exactly what the eager path runs."""
        runner, settings = self.runner, SamplingParams(temperature=key[0], top_k=key[1], top_p=key[2])
        return lambda: runner.decode_eager(settings)

    def run(self, sampling_params):
        """One decode step for every row; returns runner.next_token_gpu, now the sampled
        tokens. The graph for these settings must have been prepared before the run."""
        key = self.key(sampling_params)
        assert key in self._captured, 'prepare(key) before the run'
        return self.graphs.get(key).replay()