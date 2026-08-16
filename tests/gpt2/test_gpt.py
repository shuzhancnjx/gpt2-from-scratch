import math

import pytest
import torch

from gpt2.gpt import GPT, GPTConfig


def test_forward_shapes(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (3, 16))
    logits, loss = tiny_model(idx)
    assert logits.shape == (3, 16, tiny_config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_loss(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (3, 16))
    _, loss = tiny_model(idx, targets=idx)
    assert loss.ndim == 0 and loss.item() > 0


def test_initial_loss_is_near_uniform(tiny_config):
    """An untrained model should sit near ln(vocab_size) -- catches a broken init.

    The tolerance is loose because a 32-dim toy deviates much more than a real
    model does (the 768-dim GPT-2 measures 11.00 against ln(50304) = 10.83). This
    is a smoke check against an init that is wrong by an order of magnitude, not a
    precise assertion.
    """
    torch.manual_seed(0)
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (8, 32))
    _, loss = model(idx, targets=idx)
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 0.75


def test_weights_are_tied(tiny_model):
    assert tiny_model.transformer.wte.weight is tiny_model.lm_head.weight


def test_block_size_is_enforced(tiny_model, tiny_config):
    too_long = torch.zeros((1, tiny_config.block_size + 1), dtype=torch.long)
    with pytest.raises(AssertionError):
        tiny_model(too_long)


def test_optimizer_splits_decay_by_dimension(tiny_model):
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    decay, no_decay = opt.param_groups
    assert decay['weight_decay'] == 0.1 and no_decay['weight_decay'] == 0.0
    # matrices decay, biases and LayerNorm gains do not
    assert all(p.dim() >= 2 for p in decay['params'])
    assert all(p.dim() < 2 for p in no_decay['params'])


def test_generate_extends_sequence(tiny_model, tiny_config):
    idx = torch.zeros((2, 4), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=6, top_k=8)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], idx)      # prefix preserved


def test_generate_respects_vocab_limit(tiny_config):
    """The head is padded past the tokenizer's real vocab; those rows must not
    be sampled or the decode raises."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    limit = 10
    out = model.generate(torch.zeros((4, 2), dtype=torch.long),
                         max_new_tokens=12, top_k=5, vocab_limit=limit)
    assert out[:, 2:].max().item() < limit


def test_generate_restores_training_mode(tiny_model):
    tiny_model.train()
    tiny_model.generate(torch.zeros((1, 2), dtype=torch.long), max_new_tokens=2)
    assert tiny_model.training


def test_generate_does_not_build_a_graph(tiny_model):
    out = tiny_model.generate(torch.zeros((1, 2), dtype=torch.long), max_new_tokens=2)
    assert not out.requires_grad
