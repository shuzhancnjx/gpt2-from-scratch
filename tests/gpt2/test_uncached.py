"""generate_uncached: the no-cache generation loop (training samples, reference path)."""

import pytest
import torch

from inference.sampling import greedy, make_sampler
from inference.uncached import generate_uncached
from model.gpt2.gpt import GPT, GPTConfig

CFG = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=16, block_size=32)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG)


def test_extends_the_sequence_and_keeps_the_prefix(model):
    idx = torch.zeros((2, 4), dtype=torch.long)
    out = generate_uncached(model, idx, 6, make_sampler(top_k=8))
    assert out.shape == (2, 10) and torch.equal(out[:, :4], idx)


def test_respects_the_vocab_limit(model):
    """The head is padded past the tokenizer's real vocab; those ids must never be sampled."""
    out = generate_uncached(model, torch.zeros((4, 2), dtype=torch.long), 12,
                            make_sampler(top_k=5), vocab_limit=10)
    assert out[:, 2:].max().item() < 10


def test_restores_training_mode_and_builds_no_graph(model):
    model.train()
    out = generate_uncached(model, torch.zeros((1, 2), dtype=torch.long), 2, greedy)
    assert model.training and not out.requires_grad


def test_greedy_is_the_argmax_of_a_full_forward(model):
    idx = torch.randint(0, 64, (1, 5))
    out = generate_uncached(model, idx, 1, greedy)
    logits, _ = model.eval()(idx)
    assert out[0, -1].item() == logits[0, -1].argmax().item()
