"""LLM: the one front door for generation.

    llm = LLM('runs/keep/gpt2_124M_step013147.pt', engine='paged', device='cuda',
              decode_kernel=True, cuda_graph=True)
    llm.generate(['The capital of France is'], SamplingParams(max_new_tokens=64, temperature=0.8))

LLM loads the model once (gpt2.model.loading), builds the chosen engine, and forwards
generate() to it. Every engine implements BaseEngine, so the call is the same for all:

    engine='static'  GptInference     one batch, start to finish
    engine='row'     Scheduler        continuous batching, a full-width KV row per request
    engine='paged'   PagedScheduler   continuous batching over KV blocks (the default)

Engine options are passed through: max_seqs / block_size / num_blocks / decode_kernel /
cuda_graph for 'paged', batch_size for 'row'. max_tokens (the KV width per request)
defaults to the model's context length.
"""

from gpt2.inference.continuous_batch import PagedScheduler, Scheduler
from gpt2.inference.sampling import SamplingParams
from gpt2.inference.static_batch.inference import GptInference
from gpt2.model.loading import load_model
from gpt2.utils.device import pick_device

ENGINES = ('static', 'row', 'paged')


class LLM:

    def __init__(self, source, engine='paged', device='auto', dtype=None, **engine_options):
        if engine not in ENGINES:
            raise ValueError(f'engine must be one of {ENGINES}, got {engine!r}')
        self.device = pick_device(device)
        self.model, self.enc, self.config = load_model(source, self.device, dtype)
        self.engine_name = engine
        self.engine = self._build(engine, engine_options)

    def _build(self, engine, options):
        args = (self.model, self.enc, self.config)
        if engine == 'static':
            if options:
                raise TypeError(f'the static engine takes no options, got {sorted(options)}')
            return GptInference(*args, device=self.device)
        dtype = next(self.model.parameters()).dtype
        if engine == 'row':
            return Scheduler(*args, device=self.device, dtype=dtype, **options)
        return PagedScheduler(*args, device=self.device, dtype=dtype, **options)

    def generate(self, prompts, params=None, **settings):
        """prompts: a string or a list of strings. Settings go in `params` or as keywords
        (max_new_tokens=, temperature=, top_k=, top_p=). Returns the completion(s),
        matching the input: a string for a string, a list for a list."""
        if params is None:
            params = SamplingParams(**settings)
        elif settings:
            raise TypeError('pass either params or keyword settings, not both')
        single = isinstance(prompts, str)
        out = self.engine.generate([prompts] if single else list(prompts), params)
        return out[0] if single else out

    def __repr__(self):
        c = self.config
        return (f'LLM(engine={self.engine_name!r}, device={self.device!r}, '
                f'{c.n_layer} layers x {c.n_embd} wide)')
