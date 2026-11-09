"""ContinuousEngine: what both continuous-batching engines share -- the public calls.

    run(prompts, max_new_tokens)        greedy; prompt + completion per prompt
    generate(prompts, sampling_params)  BaseEngine's call; completions only

Each engine supplies _run(prompts, sampling_params) -> {prompt index: generated ids}, its
loop. Both loops drive the same ModelRunner (model_runner.py); they differ in policy and
KV layout:

    RowEngine    (row/engine.py)    one full-width row per request; decides inline
    PagedEngine  (paged/engine.py)  KV blocks; PagedScheduler + BlockManager decide
"""

from abc import abstractmethod

import torch

from inference.base_engine import BaseEngine
from inference.sampling import SamplingParams


class ContinuousEngine(BaseEngine):

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

    @abstractmethod
    def _run(self, prompts, sampling_params):
        """The engine's loop. Returns {prompt index: generated token ids}."""
