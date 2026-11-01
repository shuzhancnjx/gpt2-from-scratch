"""Log-likelihood evaluation. The checks that matter are the ones a wrong
implementation passes silently: an off-by-one shift, a misaligned mask, or padding
leaking into a score would all still produce plausible-looking accuracies."""

import json
import math

import pytest
import tiktoken
import torch
import torch.nn.functional as F

from gpt2.evaluation.hellaswag import evaluate_hellaswag, render, score
from gpt2.evaluation.likelihood import val_nll
from gpt2.gpt import GPT, GPTConfig

CFG = dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=32, block_size=64, dropout=0.0, bias=True)
EXAMPLES = [
    {'ctx': 'A man is standing in a kitchen. He', 'label': 2,
     'endings': ['flies away.', 'reads the moon quietly to a chair', 'starts cooking dinner.', 'is.']},
    {'ctx': 'The children ran onto the field and', 'label': 0,
     'endings': ['began to play soccer.', 'melted.', 'were a spoon in the sky of words', 'sang']},
    {'ctx': 'She opened the book', 'label': 1,
     'endings': ['and flew.', 'and started reading the first chapter.', 'blue.', 'into the cloud of soup']},
]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(**CFG)).eval()


@pytest.fixture
def enc():
    return tiktoken.get_encoding('gpt2')


# -- validation likelihood -----------------------------------------------------

def test_val_nll_matches_the_models_own_loss(model):
    """The training loss is the same quantity -- if these differ, the shift is wrong."""
    tokens = torch.randint(0, 50257, (65,), generator=torch.Generator().manual_seed(1))
    out = val_nll(model, tokens, T=64)
    _, loss = model(tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0))
    assert out['nll'] == pytest.approx(loss.item(), rel=1e-5)


def test_perplexity_and_bits_are_the_same_number_in_other_units(model):
    tokens = torch.randint(0, 50257, (64 * 3 + 1,), generator=torch.Generator().manual_seed(2))
    out = val_nll(model, tokens, T=64)
    assert out['perplexity'] == pytest.approx(math.exp(out['nll']))
    assert out['bits_per_token'] == pytest.approx(out['nll'] / math.log(2))
    assert out['tokens'] == 3 * 64


def test_val_nll_is_repeatable_and_batch_size_independent(model):
    tokens = torch.randint(0, 50257, (64 * 5 + 1,), generator=torch.Generator().manual_seed(3))
    a = val_nll(model, tokens, T=64, batch_size=1)
    b = val_nll(model, tokens, T=64, batch_size=4)
    assert a['nll'] == pytest.approx(b['nll'], rel=1e-5)


def test_position_curve_averages_back_to_the_total(model, device):
    tokens = torch.randint(0, 50257, (64 * 2 + 1,), generator=torch.Generator().manual_seed(4))
    out = val_nll(model.to(device), tokens, T=64, per_position=True, device=device)
    model.to('cpu')
    assert len(out['position_nll']) == 64
    assert sum(out['position_nll']) / 64 == pytest.approx(out['nll'], rel=1e-5)


# -- HellaSwag ---------------------------------------------------------------

def test_render_masks_exactly_the_ending_tokens(enc):
    rows, masks, label = render(EXAMPLES[0], enc)
    ctx = enc.encode(EXAMPLES[0]['ctx'])
    assert label == 2 and len(rows) == 4
    for row, mask, ending in zip(rows, masks, EXAMPLES[0]['endings']):
        assert row[:len(ctx)] == ctx
        assert row[len(ctx):] == enc.encode(' ' + ending), 'ending must carry its leading space'
        assert mask == [0] * len(ctx) + [1] * (len(row) - len(ctx))


def test_scores_match_a_hand_computation(model, enc):
    """Score one ending the slow, obvious way and compare."""
    rows, masks, _ = render(EXAMPLES[0], enc)
    total, mean = score(model, [render(EXAMPLES[0], enc)])
    for k in range(4):
        row = torch.tensor([rows[k]])
        logits, _ = model(row)
        logp = F.log_softmax(logits[0].float(), -1)
        n_ctx = masks[k].index(1)
        # token at position p (an ending token) is predicted by the logits at p - 1
        expected = sum(logp[p - 1, rows[k][p]].item() for p in range(n_ctx, len(rows[k])))
        assert total[0, k].item() == pytest.approx(expected, rel=1e-4)
        assert mean[0, k].item() == pytest.approx(expected / (len(rows[k]) - n_ctx), rel=1e-4)


def test_batching_and_padding_do_not_change_any_score(model, enc):
    """Examples of different lengths, scored alone and batched together."""
    rendered = [render(e, enc) for e in EXAMPLES]
    alone = [score(model, [r]) for r in rendered]
    together_total, together_mean = score(model, rendered)
    for i, (t, m) in enumerate(alone):
        assert torch.allclose(together_total[i], t[0], atol=1e-4), f'example {i}: padding leaked'
        assert torch.allclose(together_mean[i], m[0], atol=1e-4)


def test_accuracy_is_the_argmax_of_the_scores(model, enc, tmp_path):
    path = tmp_path / 'hs.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in EXAMPLES))
    out = evaluate_hellaswag(model, enc, path=str(path), batch_examples=2)

    total, mean = score(model, [render(e, enc) for e in EXAMPLES])
    labels = torch.tensor([e['label'] for e in EXAMPLES])
    assert out['n'] == 3
    assert out['acc'] == pytest.approx((total.argmax(1) == labels).float().mean().item())
    assert out['acc_norm'] == pytest.approx((mean.argmax(1) == labels).float().mean().item())


def test_hellaswag_scoring_runs_on_every_device(model, enc, device):
    on_cpu = score(model, [render(e, enc) for e in EXAMPLES])
    on_dev = score(model.to(device), [render(e, enc) for e in EXAMPLES], device=device)
    model.to('cpu')
    assert torch.allclose(on_dev[1], on_cpu[1], atol=1e-3)
