"""Load a GPT and its tokenizer from wherever the weights live.

    load_model('runs/keep/gpt2_124M_step013147.pt', 'cuda')   # one of our checkpoints
    load_model('gpt2', 'cuda')                                 # OpenAI's weights (needs transformers)
    load_model(GPTConfig(n_layer=2, ...), 'cpu')               # random weights, for tests/benchmarks

Returns (model, enc, config): the model in eval mode on `device`, cast to `dtype`
(default: fp16 on CUDA, fp32 elsewhere), and the GPT-2 tiktoken encoding.
"""

import os

import tiktoken
import torch

from model.gpt2.gpt import GPT, GPTConfig
from utils.device import default_dtype

PRETRAINED = ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')


def load_model(source, device='cpu', dtype=None):
    if isinstance(source, GPTConfig):
        config, model = source, GPT(source)
    elif source in PRETRAINED:
        model = GPT.from_pretrained(source)
        config = model.config
    elif os.path.exists(str(source)):
        ck = torch.load(str(source), map_location='cpu', weights_only=True)
        config = GPTConfig(**ck['config'])
        model = GPT(config)
        model.load_state_dict(ck['model'])
    else:
        raise FileNotFoundError(f'no checkpoint at {source!r} (and not one of {PRETRAINED})')

    dtype = dtype or default_dtype(device)
    model = model.eval().to(device=device, dtype=dtype)
    return model, tiktoken.get_encoding('gpt2'), config
