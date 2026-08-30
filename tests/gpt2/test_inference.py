"""Tests for the KV-cached inference path (gpt2/inference/).

Every bug this file targets is silent: a wrong causal mask, a stale position
offset or a sheared cache all produce fluent-looking text rather than an
exception. So the assertions are mostly *equivalences* -- the cached path must
agree with the obvious, slow, obviously-correct one.

The model is a 2-layer toy, but it keeps GPT-2's real vocab_size so that
tiktoken ids (and eot_token = 50256) are valid indices into wte. Weights are
random; that is fine, because nothing here asserts anything about text quality.
"""

import pytest
import tiktoken
import torch

from gpt2.gpt import GPT, GPTConfig
from gpt2.inference.inference import GptInference
from gpt2.inference.kv_cache import KVCache

# Real vocab (so tiktoken ids are in range), toy everything else.
TINY_GPT2 = dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=32,
                 block_size=64, dropout=0.0, bias=True)


@pytest.fixture
def gpt2_config():
    return GPTConfig(**TINY_GPT2)


@pytest.fixture
def inference(tmp_path):
    """A GptInference over a randomly initialised toy model.

    GptInference loads from a checkpoint, so we write one rather than reaching
    past its constructor -- that keeps the real load path under test too.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY_GPT2))
    ckpt = tmp_path / 'tiny.pt'
    torch.save({'config': TINY_GPT2, 'model': model.state_dict()}, ckpt)
    return GptInference(str(ckpt))


def greedy_reference(model, enc, prompt, max_new_tokens):
    """The slow path: no cache, no batching, recompute the whole prefix each step.

    Deliberately dumb. This is the oracle the cached path is checked against, so
    it must not share any of its machinery.
    """
    ids = enc.encode(prompt)
    for _ in range(max_new_tokens):
        logits, _ = model(torch.tensor([ids], dtype=torch.long))
        nxt = int(logits[0, -1, :enc.n_vocab].argmax())
        ids.append(nxt)
        if nxt == enc.eot_token:
            break
    return ids


# ---- the equivalences ------------------------------------------------------

def test_cached_greedy_matches_uncached(inference):
    """The headline invariant.

    Catches: is_causal applied to a single query (token sees only cache[0]),
    position embeddings that ignore the cache offset, and a cache that is
    written at the wrong slot. None of those raise on their own.
    """
    prompt = 'The capital of France is'
    expected = greedy_reference(inference.model, inference.enc, prompt, 8)

    got = inference.sample(prompt, max_new_tokens=8, temperature=0)

    assert got == inference.enc.decode(expected)


def test_batching_does_not_change_a_row(inference):
    """A prompt must generate the same text alone as it does beside a longer one.

    Catches everything left-padding can break: per-row position ids derived from
    the wrong axis, pad slots leaking into attention, rows sharing a write
    pointer incorrectly. The two prompts differ in length on purpose -- with
    equal lengths there is no padding and the test proves nothing.
    """
    short = 'Hello'
    long = 'It was a bright cold day in April and the clocks were striking'

    alone_short = inference.sample(short, max_new_tokens=6, temperature=0)
    alone_long = inference.sample(long, max_new_tokens=6, temperature=0)
    batched = inference.sample([short, long], max_new_tokens=6, temperature=0)

    assert batched == [alone_short, alone_long]


def test_duplicate_prompts_in_one_batch_agree(inference):
    """Same prompt twice in a batch, padded to a third, longer one.

    A weaker assertion than the one above (it cannot catch an error shared by
    both copies) but it isolates cross-row contamination specifically.
    """
    out = inference.sample(['Hello', 'Hello', 'A much longer prompt than the others'],
                           max_new_tokens=6, temperature=0)
    assert out[0] == out[1]


def test_cached_logits_match_uncached(gpt2_config):
    """Decoding from the cache must give the same logits as recomputing the prefix.

    Compares logits rather than sampled tokens on purpose. argmax over a randomly
    initialised model is remarkably insensitive -- it will happily pick the same
    token whether or not attention is working -- so token equality passes even
    with the mask fully broken. Logits do not.

    sample() always passes a mask, so this is also the only coverage of the other
    two branches in forward(): is_causal=(q_len == kv_len) and the
    arange(past, past + T) fallback.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    ids = torch.randint(0, 1000, (1, 5))

    with torch.inference_mode():
        cache = KVCache(batch_size=1, device='cpu', config=gpt2_config)
        logits, _ = model(ids, kv_cache=cache)

        for step in range(6):
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)

            cached, _ = model(nxt, kv_cache=cache)   # one token, rest from cache
            full, _ = model(ids)                     # whole prefix, no cache

            assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-5), \
                f'cached logits diverged from recomputed at step {step}'
            logits = cached


def test_padded_row_logits_match_unpadded(gpt2_config):
    """A short row inside a left-padded batch must behave as if it were alone.

    The logits-level counterpart of test_batching_does_not_change_a_row: catches
    position ids that count padding, and pad slots leaking into attention, even
    when both leave the sampled token unchanged.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    short = torch.randint(0, 1000, (1, 3))
    long = torch.randint(0, 1000, (1, 7))
    L = long.size(1)

    idx = torch.full((2, L), 50256, dtype=torch.long)
    mask = torch.zeros(2, L, dtype=torch.bool)
    idx[0, L - 3:], mask[0, L - 3:] = short[0], True
    idx[1], mask[1] = long[0], True
    pos_ids = (mask.cumsum(1) - 1).clamp(min=0)

    with torch.inference_mode():
        cache = KVCache(batch_size=2, device='cpu', config=gpt2_config)
        batched, _ = model(idx, kv_cache=cache, pos_ids=pos_ids, attn_mask=mask)
        alone, _ = model(short)

    assert torch.allclose(batched[0, -1], alone[0, -1], atol=1e-5), \
        'padding changed the short row'


# ---- the cache itself ------------------------------------------------------

def test_all_layers_write_the_same_slots(gpt2_config):
    """pos must advance once per forward, not once per layer.

    If advance() lived inside update(), layer i would write at slot i*T and the
    layers would shear apart -- valid shapes, garbage attention. Here every
    layer must have written [0, T) and touched nothing beyond it.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    cache = KVCache(batch_size=1, device='cpu', config=gpt2_config)
    T = 5

    with torch.inference_mode():
        model(torch.randint(0, 1000, (1, T)), kv_cache=cache)

    assert cache.seq_len() == T
    for layer in range(gpt2_config.n_layer):
        assert cache.key[layer][:, :, :T].abs().sum() > 0, f'layer {layer} wrote nothing'
        assert torch.all(cache.key[layer][:, :, T:] == 0), f'layer {layer} wrote past {T}'
        assert torch.all(cache.value[layer][:, :, T:] == 0)


def test_pos_advances_by_token_count(gpt2_config):
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    cache = KVCache(batch_size=1, device='cpu', config=gpt2_config)

    with torch.inference_mode():
        model(torch.randint(0, 1000, (1, 5)), kv_cache=cache)
        assert cache.seq_len() == 5
        model(torch.randint(0, 1000, (1, 1)), kv_cache=cache)
        assert cache.seq_len() == 6


def test_uncached_forward_still_works(gpt2_config):
    """Regression guard: the training path passes no cache.

    An unguarded kv_cache.advance() in forward() breaks every training step
    while leaving inference perfectly healthy.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config)
    idx = torch.randint(0, gpt2_config.vocab_size, (2, 8))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 8, gpt2_config.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


# ---- the sample() contract -------------------------------------------------

def test_padding_never_reaches_the_output(inference):
    """Left-pad filler must be stripped, and the prompt must survive intact."""
    short = 'Hi'
    out = inference.sample([short, 'A considerably longer prompt goes here'],
                           max_new_tokens=4, temperature=0)

    assert '<|endoftext|>' not in out[0]
    assert out[0].startswith(short)


def test_return_type_follows_input_type(inference):
    assert isinstance(inference.sample('Hello', max_new_tokens=2, temperature=0), str)
    assert isinstance(inference.sample(['Hello'], max_new_tokens=2, temperature=0), list)


def test_generation_stops_at_eot(inference, monkeypatch):
    """Reaching eot must end the loop early rather than running to the cap."""
    calls = {'n': 0}

    def fake_sample(logits, **kwargs):
        calls['n'] += 1
        eot = inference.enc.eot_token
        return torch.full((logits.size(0), 1), eot, dtype=torch.long)

    monkeypatch.setattr(inference, 'sample_next_token', fake_sample)
    inference.sample('Hello', max_new_tokens=50, temperature=0)

    assert calls['n'] == 1, 'should have stopped on the first eot, not run to 50'


def test_output_length_respects_max_new_tokens(inference):
    prompt = 'Hello'
    n_prompt = len(inference.enc.encode(prompt))
    out = inference.sample(prompt, max_new_tokens=7, temperature=0)

    assert len(inference.enc.encode(out)) <= n_prompt + 7
