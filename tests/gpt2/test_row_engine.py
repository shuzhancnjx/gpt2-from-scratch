"""RowEngine end to end: continuous batching over full-width rows, on a toy model with
GPT-2's real vocab. The text must match the uncached path whatever the batch width, and
rows must be reused, positions per row, idle rows left alone."""

import pytest
import torch

from model.gpt2.gpt import GPTConfig

GPT2_TINY = dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=32,
                 block_size=128, dropout=0.0, bias=True)


@pytest.fixture
def sched_parts():
    """A toy model with GPT-2's real vocab, so tiktoken ids are valid indices."""
    import tiktoken
    from model.gpt2.gpt import GPT
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
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    got = RowEngine(model, enc, cfg, batch_size=2, max_tokens=64).run(PROMPTS, max_new_tokens=6)
    expected = [greedy_reference(model, enc, p, 6) for p in PROMPTS]

    assert got == expected


def test_slots_are_reused_across_requests(sched_parts):
    """More prompts than slots: the batch width is fixed, the occupants rotate."""
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    sched = RowEngine(model, enc, cfg, batch_size=2, max_tokens=64)
    out = sched.run(PROMPTS * 2, max_new_tokens=4)

    assert len(out) == 8
    assert not bool(sched.runner.active_gpu.any()), 'slots left occupied after the run'
    assert sched.runner.cache.pos_gpu.tolist() == [0, 0], 'released rows should be back at zero'


def test_batch_width_does_not_change_output(sched_parts):
    """Same prompts through 1, 2 and 4 slots must produce identical text."""
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    runs = [RowEngine(model, enc, cfg, batch_size=b, max_tokens=64)
            .run(PROMPTS, max_new_tokens=5) for b in (1, 2, 4)]

    assert runs[0] == runs[1] == runs[2]


def test_a_short_request_does_not_wait_for_a_long_one(sched_parts):
    """A row that hits its cap is released while its neighbour keeps going."""
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    sched = RowEngine(model, enc, cfg, batch_size=2, max_tokens=64)
    out = sched.run(['Hello', 'The capital of France is'], max_new_tokens=3)

    assert len(out) == 2
    assert all(o.startswith(p) for o, p in zip(out, ['Hello', 'The capital of France is']))


def test_prompt_too_long_for_the_cache_is_rejected(sched_parts):
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    sched = RowEngine(model, enc, cfg, batch_size=1, max_tokens=8)
    with pytest.raises(ValueError, match='does not fit'):
        sched.run(['It was a bright cold day in April and the clocks'], max_new_tokens=6)


def test_mask_covers_written_slots_and_blanks_idle_rows(sched_parts):
    """The mask is derived from pos, so assert its exact contents.

    End-to-end text comparison is too weak here: argmax over a randomly
    initialised model picks the same token even with the mask badly wrong.
    """
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    sched = RowEngine(model, enc, cfg, batch_size=3, max_tokens=6)
    sched.runner.cache.pos_gpu = torch.tensor([2, 4, 0])
    sched.runner.active_gpu = torch.tensor([True, True, False])

    m = sched.runner.mask().int().tolist()

    # `<=` pos: the token written this step lands at column pos and sees itself
    assert m[0] == [1, 1, 1, 0, 0, 0]
    assert m[1] == [1, 1, 1, 1, 1, 0]
    assert m[2] == [0, 0, 0, 0, 0, 0], 'an idle slot must be fully masked'


def test_decode_feeds_per_row_positions_and_leaves_idle_slots_alone(sched_parts):
    """Two invariants that text comparison cannot see.

    Every decode forward must hand each row its own position, and a slot with no
    request must keep pos == 0 so the next admission finds it free.
    """
    from inference.continuous_batch import RowEngine
    model, enc, cfg = sched_parts

    sched = RowEngine(model, enc, cfg, batch_size=2, max_tokens=64)
    seen = []
    real = sched.runner.model

    def spy(idx, **kwargs):
        # the prefill batch, not idx width: a one-token prompt prefills with T == 1 too
        if kwargs.get('prefill') is None:             # decode, not prefill
            seen.append((kwargs['pos_ids'].flatten().tolist(),
                         sched.runner.cache.pos_gpu.tolist(),
                         sched.runner.active_gpu.tolist()))
        return real(idx, **kwargs)

    sched.runner.model = sched.runner.graph_model = spy      # decode calls graph_model
    sched.run(PROMPTS[:3], max_new_tokens=4)

    assert seen, 'no decode step was observed'
    for pos_ids, pos, active in seen:
        assert pos_ids == pos, f'decode fed {pos_ids}, rows are at {pos}'
        for row, is_active in enumerate(active):
            if not is_active:
                assert pos[row] == 0, f'idle slot {row} drifted to {pos[row]}'


def test_engines_share_the_runner_and_the_loop_contract(sched_parts):
    """Both continuous engines are ContinuousEngines (run/generate over an abstract _run)
    and drive the same ModelRunner -- one copy of the model calls."""
    from inference.continuous_batch import ContinuousEngine, ModelRunner, PagedEngine, RowEngine
    model, enc, cfg = sched_parts
    with pytest.raises(TypeError):
        ContinuousEngine()
    for engine in (RowEngine(model, enc, cfg, batch_size=2, max_tokens=64),
                   PagedEngine(model, enc, cfg, max_seqs=2, block_size=16, max_tokens=64)):
        assert isinstance(engine, ContinuousEngine) and type(engine.runner) is ModelRunner
