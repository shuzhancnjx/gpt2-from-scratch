"""Tests for the per-row KV cache behind continuous batching.

These exercise the cache directly rather than through a model. The property
under test is that rows are *independent* -- writing, advancing or retiring one
row must leave every other row untouched -- and that is much easier to assert
on the buffers than to infer from generated text.
"""

import pytest
import torch

from gpt2.gpt import GPTConfig
from gpt2.inference.continuous_batch import RowKVCache

TINY = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=32,
            dropout=0.0, bias=True)


@pytest.fixture
def config():
    return GPTConfig(**TINY)


@pytest.fixture
def cache(config):
    return RowKVCache(batch_size=3, device='cpu', config=config, max_tokens=16)


def kv(B, T, config, fill):
    """A (B, n_head, T, head_size) block whose every element is `fill`."""
    hs = config.n_embd // config.n_head
    return torch.full((B, config.n_head, T, hs), float(fill))


# -- prefill -----------------------------------------------------------------

def test_prefill_writes_one_row_from_slot_zero(cache, config):
    cache.update(0, kv(1, 5, config, 1.0), kv(1, 5, config, 1.0), row=1)
    cache.advance(5, row=1)

    assert cache.pos_gpu.tolist() == [0, 5, 0], 'only row 1 should have advanced'
    assert torch.all(cache.key[0][1, :, :5] == 1.0)
    assert torch.all(cache.key[0][1, :, 5:] == 0.0), 'wrote past the prompt'
    assert torch.all(cache.key[0][0] == 0.0), 'row 0 was touched'
    assert torch.all(cache.key[0][2] == 0.0), 'row 2 was touched'


def test_prefill_returns_a_view_not_a_copy(cache, config):
    """Attention reads the returned buffer; a copy would cost ~10MB per layer."""
    k, _ = cache.update(0, kv(1, 4, config, 2.0), kv(1, 4, config, 2.0), row=2)

    assert k.shape == (1, config.n_head, cache.width, config.n_embd // config.n_head)
    assert k._base is cache.key[0], 'returned a copy -- use basic slicing, not fancy indexing'


def test_prefill_into_an_occupied_row_is_rejected(cache, config):
    cache.update(0, kv(1, 5, config, 1.0), kv(1, 5, config, 1.0), row=0)
    cache.advance(5, row=0)

    with pytest.raises(AssertionError):
        cache.update(0, kv(1, 3, config, 1.0), kv(1, 3, config, 1.0), row=0)


# -- decode ------------------------------------------------------------------

def test_decode_writes_each_row_at_its_own_pos(cache, config):
    """The core of continuous batching: rows sit at different lengths."""
    cache.pos_gpu = torch.tensor([2, 7, 0])

    cache.update(0, kv(3, 1, config, 9.0), kv(3, 1, config, 9.0))

    for row, col in enumerate([2, 7, 0]):
        assert torch.all(cache.key[0][row, :, col] == 9.0), f'row {row} missed slot {col}'
        others = [c for c in range(cache.width) if c != col]
        assert torch.all(cache.key[0][row, :, others] == 0.0), f'row {row} wrote elsewhere'


def test_decode_advances_every_row_by_one(cache):
    cache.pos_gpu = torch.tensor([2, 7, 0])
    cache.advance(1)
    assert cache.pos_gpu.tolist() == [3, 8, 1]


def test_advance_skips_inactive_rows(cache):
    """A retired slot awaiting a new request must not drift forward."""
    cache.pos_gpu = torch.tensor([2, 7, 0])
    cache.advance(1, active=torch.tensor([True, False, True]))
    assert cache.pos_gpu.tolist() == [3, 7, 1]


# -- retirement --------------------------------------------------------------

def test_release_frees_one_row_only(cache, config):
    for row in range(3):
        cache.update(0, kv(1, 4, config, row + 1.0), kv(1, 4, config, row + 1.0), row=row)
        cache.advance(4, row=row)

    cache.release(1)

    assert cache.pos_gpu.tolist() == [4, 0, 4]
    assert torch.all(cache.key[0][1] == 0.0), 'row 1 should be cleared'
    assert torch.all(cache.key[0][0, :, :4] == 1.0), 'row 0 was disturbed'
    assert torch.all(cache.key[0][2, :, :4] == 3.0), 'row 2 was disturbed'


def test_released_row_accepts_a_new_request(cache, config):
    cache.update(0, kv(1, 6, config, 1.0), kv(1, 6, config, 1.0), row=0)
    cache.advance(6, row=0)
    cache.release(0)

    cache.update(0, kv(1, 2, config, 5.0), kv(1, 2, config, 5.0), row=0)
    cache.advance(2, row=0)

    assert int(cache.pos_gpu[0]) == 2
    assert torch.all(cache.key[0][0, :, :2] == 5.0)
    assert torch.all(cache.key[0][0, :, 2:] == 0.0), 'stale KV from the previous request'


# -- layer lockstep ----------------------------------------------------------

def test_all_layers_stay_in_lockstep(cache, config):
    """advance() belongs to the caller, once per forward -- never inside update()."""
    for layer in range(len(cache.key)):
        cache.update(layer, kv(1, 3, config, 1.0), kv(1, 3, config, 1.0), row=0)
    cache.advance(3, row=0)

    assert int(cache.pos_gpu[0]) == 3, 'pos moved per layer instead of per forward'
    for layer in range(len(cache.key)):
        assert torch.all(cache.key[layer][0, :, :3] == 1.0), f'layer {layer} wrote nothing'
        assert torch.all(cache.key[layer][0, :, 3:] == 0.0), f'layer {layer} wrote past 3'


# -- scheduler ---------------------------------------------------------------

GPT2_TINY = dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=32,
                 block_size=128, dropout=0.0, bias=True)


@pytest.fixture
def sched_parts():
    """A toy model with GPT-2's real vocab, so tiktoken ids are valid indices."""
    import tiktoken
    from gpt2.gpt import GPT
    torch.manual_seed(0)
    cfg = GPTConfig(**GPT2_TINY)
    return GPT(cfg).eval(), tiktoken.get_encoding('gpt2'), cfg


def greedy_reference(model, enc, prompt, max_new_tokens):
    """No cache, no batching: recompute the whole prefix each step."""
    ids = enc.encode(prompt)
    out = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _ = model(torch.tensor([ids + out], dtype=torch.long))
            nxt = int(logits[0, -1, :enc.n_vocab].argmax())
            out.append(nxt)
            if nxt == enc.eot_token:
                break
    return prompt + enc.decode(out)


PROMPTS = ['Hello', 'The capital of France is', 'A', 'It was a bright cold day in April']


def test_scheduler_matches_the_uncached_path(sched_parts):
    """The headline invariant: rotating slots must not change any row's text.

    Catches per-row position errors, mask drift, and one request seeing another's
    keys -- none of which raise on their own.
    """
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    got = Scheduler(model, enc, cfg, batch_size=2, max_tokens=64).run(PROMPTS, max_new_tokens=6)
    expected = [greedy_reference(model, enc, p, 6) for p in PROMPTS]

    assert got == expected


def test_slots_are_reused_across_requests(sched_parts):
    """More prompts than slots: the batch width is fixed, the occupants rotate."""
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    sched = Scheduler(model, enc, cfg, batch_size=2, max_tokens=64)
    out = sched.run(PROMPTS * 2, max_new_tokens=4)

    assert len(out) == 8
    assert not bool(sched.active_gpu.any()), 'slots left occupied after the run'
    assert sched.cache.pos_gpu.tolist() == [0, 0], 'released rows should be back at zero'


def test_batch_width_does_not_change_output(sched_parts):
    """Same prompts through 1, 2 and 4 slots must produce identical text."""
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    runs = [Scheduler(model, enc, cfg, batch_size=b, max_tokens=64)
            .run(PROMPTS, max_new_tokens=5) for b in (1, 2, 4)]

    assert runs[0] == runs[1] == runs[2]


def test_a_short_request_does_not_wait_for_a_long_one(sched_parts):
    """A row that hits its cap is released while its neighbour keeps going."""
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    sched = Scheduler(model, enc, cfg, batch_size=2, max_tokens=64)
    out = sched.run(['Hello', 'The capital of France is'], max_new_tokens=3)

    assert len(out) == 2
    assert all(o.startswith(p) for o, p in zip(out, ['Hello', 'The capital of France is']))


def test_prompt_too_long_for_the_cache_is_rejected(sched_parts):
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    sched = Scheduler(model, enc, cfg, batch_size=1, max_tokens=8)
    with pytest.raises(ValueError, match='does not fit'):
        sched.run(['It was a bright cold day in April and the clocks'], max_new_tokens=6)


def test_mask_covers_written_slots_and_blanks_idle_rows(sched_parts):
    """The mask is derived from pos, so assert its exact contents.

    End-to-end text comparison is too weak here: argmax over a randomly
    initialised model picks the same token even with the mask badly wrong.
    """
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    sched = Scheduler(model, enc, cfg, batch_size=3, max_tokens=6)
    sched.cache.pos_gpu = torch.tensor([2, 4, 0])
    sched.active_gpu = torch.tensor([True, True, False])

    m = sched._mask().int().tolist()

    # `<=` pos: the token written this step lands at column pos and sees itself
    assert m[0] == [1, 1, 1, 0, 0, 0]
    assert m[1] == [1, 1, 1, 1, 1, 0]
    assert m[2] == [0, 0, 0, 0, 0, 0], 'an idle slot must be fully masked'


def test_decode_feeds_per_row_positions_and_leaves_idle_slots_alone(sched_parts):
    """Two invariants that text comparison cannot see.

    Every decode forward must hand each row its own position, and a slot with no
    request must keep pos == 0 so the next admission finds it free.
    """
    from gpt2.inference.continuous_batch import Scheduler
    model, enc, cfg = sched_parts

    sched = Scheduler(model, enc, cfg, batch_size=2, max_tokens=64)
    seen = []
    real = sched.model

    def spy(idx, **kwargs):
        # cache_row, not idx width: a one-token prompt prefills with T == 1 too
        if kwargs.get('cache_row') is None:        # decode, not prefill
            seen.append((kwargs['pos_ids'].flatten().tolist(),
                         sched.cache.pos_gpu.tolist(),
                         sched.active_gpu.tolist()))
        return real(idx, **kwargs)

    sched.model = spy
    sched.run(PROMPTS[:3], max_new_tokens=4)

    assert seen, 'no decode step was observed'
    for pos_ids, pos, active in seen:
        assert pos_ids == pos, f'decode fed {pos_ids}, rows are at {pos}'
        for row, is_active in enumerate(active):
            if not is_active:
                assert pos[row] == 0, f'idle slot {row} drifted to {pos[row]}'
