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

    The cache returns its whole W-wide buffer, so the mask is the only thing
    hiding the not-yet-written slots -- which is why a cached forward now
    requires one. The uncached reference on the right needs no mask: it is fed
    exactly the real tokens and nothing else.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    P, N = 5, 6
    W = P + N

    buf = torch.zeros(1, W, dtype=torch.long)
    buf[0, :P] = torch.randint(0, 1000, (P,))
    mask = torch.zeros(1, W, dtype=torch.bool)
    mask[0, :P] = True

    with torch.inference_mode():
        cache = KVCache(batch_size=1, device='cpu', config=gpt2_config, max_tokens=W)
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        logits, _ = model(buf[:, :P], kv_cache=cache, pos_ids=pos[:, :P], attn_mask=mask)

        for step in range(N):
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            col = P + step
            buf[0, col], mask[0, col] = nxt[0, 0], True     # mark it BEFORE the forward

            cached, _ = model(nxt, kv_cache=cache,
                              pos_ids=torch.tensor([[col]]), attn_mask=mask)
            full, _ = model(buf[:, :col + 1])               # whole prefix, no cache

            assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-5), \
                f'cached logits diverged from recomputed at step {step}'
            logits = cached


def test_cached_forward_without_a_mask_is_rejected(gpt2_config):
    """The buffer is full-width, so unwritten slots are real zeros in k and v.

    Without a mask they are attended to like any other key -- silently, since the
    shapes are valid. The guard turns that into an error at the call site.
    """
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    cache = KVCache(batch_size=1, device='cpu', config=gpt2_config, max_tokens=8)

    with pytest.raises(AssertionError):
        model(torch.randint(0, 1000, (1, 5)), kv_cache=cache)


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
        cache = KVCache(batch_size=2, device='cpu', config=gpt2_config, max_tokens=L)
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
    T, W = 5, 12                      # W > T so there is room to detect a stray write
    cache = KVCache(batch_size=1, device='cpu', config=gpt2_config, max_tokens=W)

    mask = torch.zeros(1, W, dtype=torch.bool)
    mask[0, :T] = True

    with torch.inference_mode():
        model(torch.randint(0, 1000, (1, T)), kv_cache=cache, attn_mask=mask)

    assert cache.seq_len() == T
    for layer in range(gpt2_config.n_layer):
        assert cache.key[layer][:, :, :T].abs().sum() > 0, f'layer {layer} wrote nothing'
        assert torch.all(cache.key[layer][:, :, T:] == 0), f'layer {layer} wrote past {T}'
        assert torch.all(cache.value[layer][:, :, T:] == 0)


def test_pos_advances_by_token_count(gpt2_config):
    torch.manual_seed(0)
    model = GPT(gpt2_config).eval()
    W = 12
    cache = KVCache(batch_size=1, device='cpu', config=gpt2_config, max_tokens=W)
    mask = torch.zeros(1, W, dtype=torch.bool)

    with torch.inference_mode():
        mask[0, :5] = True
        model(torch.randint(0, 1000, (1, 5)), kv_cache=cache, attn_mask=mask)
        assert cache.seq_len() == 5

        mask[0, 5] = True                      # the slot this forward will write
        model(torch.randint(0, 1000, (1, 1)), kv_cache=cache, attn_mask=mask)
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
    """Reaching eot must end the loop early rather than running to the cap.

    Stopping is no longer immediate: reading `done` forces a device sync, so the
    loop only checks every STOP_CHECK_EVERY steps and trades a bounded amount of
    wasted work for far fewer stalls.

    Rather than pin the interval, this asserts the property that matters -- the
    step count is governed by the eot, not by max_new_tokens. Quadrupling the cap
    must not change how far it runs.
    """
    def run(max_new_tokens):
        calls = {'n': 0}

        def fake_sample(logits, **kwargs):
            calls['n'] += 1
            eot = inference.enc.eot_token
            return torch.full((logits.size(0), 1), eot, dtype=torch.long)

        monkeypatch.setattr(inference, 'sample_next_token', fake_sample)
        inference.sample('Hello', max_new_tokens=max_new_tokens, temperature=0)
        return calls['n']

    short, long = run(100), run(400)

    assert long < 400, 'ran to the cap -- eot never ended the loop'
    assert short == long, (f'step count follows the cap ({short} vs {long}), '
                           'so stopping is not eot-driven')


def test_top_p_returns_vocab_ids_not_ranks(inference):
    """top_p sorts the logits, so it must map the sample back to a vocab id.

    Returning the sorted *position* instead is silent: rank 0 is a valid token id,
    so generation just quietly produces low-id punctuation.
    """
    logits = torch.full((1, 50257), -10.0)
    logits[0, 40000] = 10.0                       # one overwhelming favourite

    out = inference.sample_next_token(logits, temperature=1.0, top_p=0.9)

    assert out.item() == 40000


def test_top_p_keeps_the_smallest_set_covering_p(inference):
    """probs are ~[.826, .112, .041, .015, .006]; p=0.9 needs the first two."""
    logits = torch.tensor([[5.0, 3.0, 2.0, 1.0, 0.0]])
    seen = {inference.sample_next_token(logits, temperature=1.0, top_p=0.9).item()
            for _ in range(300)}

    assert seen == {0, 1}


def test_top_p_degenerate_values_still_sample(inference):
    """p=0 would drop every token (exclusive mass of the top token is 0)."""
    logits = torch.tensor([[5.0, 3.0, 2.0, 1.0, 0.0]])

    assert inference.sample_next_token(logits, temperature=1.0, top_p=0).item() == 0
    assert inference.sample_next_token(logits, temperature=1.0, top_p=1.0).item() in range(5)


def test_output_length_respects_max_new_tokens(inference):
    prompt = 'Hello'
    n_prompt = len(inference.enc.encode(prompt))
    out = inference.sample(prompt, max_new_tokens=7, temperature=0)

    assert len(inference.enc.encode(out)) <= n_prompt + 7
