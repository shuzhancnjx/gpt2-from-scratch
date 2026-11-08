"""The unified interface: every engine behind BaseEngine.generate(), and LLM in front.

The invariant: the same prompts and the same greedy settings give the same
completions whichever engine runs them -- static batching, the row scheduler or
the paged scheduler. Only speed and memory may differ.
"""

import pytest
import torch

from app import LLM, SamplingParams
from inference.base_engine import BaseEngine
from inference.continuous_batch import PagedScheduler, Scheduler
from inference.static_batch.inference import GptInference
from model.gpt2 import GPT, GPTConfig
from model.gpt2.loading import load_model

TINY = GPTConfig(vocab_size=50304, n_layer=2, n_head=2, n_embd=32, block_size=128)
PROMPTS = ['Hello', 'The capital of France is', 'A', 'It was a bright cold day in April']
GREEDY = SamplingParams(max_new_tokens=6)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / 'tiny.pt'
    torch.save({'config': vars(TINY), 'model': GPT(TINY).state_dict()}, path)
    return str(path)


def test_every_engine_is_a_base_engine():
    for cls in (GptInference, Scheduler, PagedScheduler):
        assert issubclass(cls, BaseEngine), cls.__name__
    with pytest.raises(TypeError):
        BaseEngine()


def test_all_engines_agree_through_generate(checkpoint):
    model, enc, cfg = load_model(checkpoint, 'cpu')
    engines = [GptInference(model, enc, cfg, device='cpu'),
               Scheduler(model, enc, cfg, batch_size=2, max_tokens=64),
               PagedScheduler(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=64)]
    outs = [e.generate(PROMPTS, GREEDY) for e in engines]
    assert outs[0] == outs[1] == outs[2]
    assert all(isinstance(o, str) for o in outs[0])


def test_generate_returns_the_completion_only(checkpoint):
    """run() keeps returning prompt + completion; generate() returns just the new text."""
    model, enc, cfg = load_model(checkpoint, 'cpu')
    sched = Scheduler(model, enc, cfg, batch_size=2, max_tokens=64)
    full = sched.run(PROMPTS, max_new_tokens=6)
    completions = sched.generate(PROMPTS, GREEDY)
    for p, f, c in zip(PROMPTS, full, completions):
        assert f.replace('<|endoftext|>', '') == p + c


@pytest.mark.parametrize('engine', ['static', 'row', 'paged'])
def test_llm_front_door(checkpoint, engine):
    options = {'static': {}, 'row': {'batch_size': 2, 'max_tokens': 64},
               'paged': {'max_seqs': 2, 'block_size': 4, 'max_tokens': 64}}[engine]
    llm = LLM(checkpoint, engine=engine, device='cpu', **options)
    reference = LLM(checkpoint, engine='static', device='cpu').generate(PROMPTS, GREEDY)
    assert llm.generate(PROMPTS, GREEDY) == reference
    assert isinstance(llm.generate('Hello', max_new_tokens=3), str)          # str in, str out


def test_llm_rejects_bad_arguments(checkpoint):
    with pytest.raises(ValueError, match='engine'):
        LLM(checkpoint, engine='turbo', device='cpu')
    with pytest.raises(TypeError, match='options'):
        LLM(checkpoint, engine='static', device='cpu', max_seqs=4)
    llm = LLM(checkpoint, engine='row', device='cpu', max_tokens=64)
    with pytest.raises(TypeError, match='either'):
        llm.generate('Hi', GREEDY, temperature=0.5)


def test_load_model_from_a_config():
    model, enc, cfg = load_model(TINY, 'cpu')
    assert cfg is TINY and not model.training and enc.n_vocab == 50257


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_paged_graph_per_sampling_setting(checkpoint):
    """cuda_graph=True captures one graph per sampler setting, reused across lengths."""
    llm = LLM(checkpoint, engine='paged', device='cuda', max_seqs=2, block_size=4,
              max_tokens=64, cuda_graph=True)
    eager = LLM(checkpoint, engine='paged', device='cuda', max_seqs=2, block_size=4, max_tokens=64)
    assert llm.generate(PROMPTS, GREEDY) == eager.generate(PROMPTS, GREEDY)
    llm.generate(PROMPTS, SamplingParams(max_new_tokens=3))                  # same sampler: no new graph
    assert len(llm.engine.decode_graphs) == 1
    llm.generate(PROMPTS, SamplingParams(max_new_tokens=3, temperature=0.8, top_k=40))
    assert len(llm.engine.decode_graphs) == 2
