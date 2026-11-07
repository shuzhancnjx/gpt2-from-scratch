"""BaseEngine: the one interface every inference engine implements.

    engine.generate(prompts, SamplingParams(max_new_tokens=64, temperature=0.8))
        -> one completion per prompt (generated text only: no prompt, no <|endoftext|>)

Every engine is built the same way, from an already-loaded model
(gpt2.model.loading.load_model):

    static_batch.GptInference(model, enc, config, device)
    continuous_batch.Scheduler(model, enc, config, device=..., batch_size=...)
    continuous_batch.PagedScheduler(model, enc, config, device=..., max_seqs=..., ...)

so an app (gpt2.app.LLM, the chat server) can drive any of them through generate()
alone. Each engine keeps its own lower-level call too (GptInference.sample,
BaseScheduler.run), which returns prompt + completion as before.
"""

from abc import ABC, abstractmethod


class BaseEngine(ABC):

    @abstractmethod
    def generate(self, prompts, params=None):
        """prompts: list[str]. params: SamplingParams (default: greedy, 32 tokens).
        Returns list[str], the completions, in prompt order."""

    def _completion(self, tokens):
        """Generated ids -> text, dropping <|endoftext|> (it ends a request, it isn't text)."""
        eot = self.enc.eot_token
        return self.enc.decode([t for t in tokens if t != eot])
