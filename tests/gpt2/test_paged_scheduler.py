"""Steps d and e of P1.4: the paged scheduler, end to end.

The headline invariant is the same as for Scheduler: rotating requests through a
paged pool must not change any request's text. On top of that, paging adds
bookkeeping invariants -- every block returned, no block owned twice, requests
wait (not fail) when the pool is tight -- that text comparison cannot see.
"""

import pytest
import torch

from gpt2.gpt import GPT, GPTConfig
from gpt2.inference.continuous_batch import PagedScheduler, Scheduler

GPT2_TINY = dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=32,
                 block_size=128, dropout=0.0, bias=True)
PROMPTS = ['Hello', 'The capital of France is', 'A', 'It was a bright cold day in April']
W = 64


@pytest.fixture
def parts():
    """A toy model with GPT-2's real vocab, so tiktoken ids are valid indices."""
    import tiktoken
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


# -- step e: identical outputs ---------------------------------------------------

def test_paged_matches_the_uncached_path(parts):
    model, enc, cfg = parts
    got = PagedScheduler(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(
        PROMPTS, max_new_tokens=6)
    assert got == [greedy_reference(model, enc, p, 6) for p in PROMPTS]


@pytest.mark.parametrize('max_seqs,block_size', [(1, 4), (2, 8), (4, 16), (3, 4)])
def test_paged_matches_the_row_scheduler(parts, max_seqs, block_size):
    """Same prompts, any slot count and block size: identical text to the baseline."""
    model, enc, cfg = parts
    expected = Scheduler(model, enc, cfg, batch_size=2, max_tokens=W).run(PROMPTS, max_new_tokens=5)
    got = PagedScheduler(model, enc, cfg, max_seqs=max_seqs, block_size=block_size,
                         max_tokens=W).run(PROMPTS, max_new_tokens=5)
    assert got == expected


def test_paged_runs_on_every_available_device(parts, device):
    model, enc, cfg = parts
    expected = Scheduler(model.to('cpu'), enc, cfg, batch_size=2, max_tokens=W).run(
        PROMPTS, max_new_tokens=4)
    got = PagedScheduler(model.to(device), enc, cfg, max_seqs=2, block_size=8,
                         max_tokens=W, device=device).run(PROMPTS, max_new_tokens=4)
    model.to('cpu')
    assert got == expected


# -- step d: memory decisions --------------------------------------------------

def test_more_requests_fit_than_rows_in_the_same_memory(parts):
    """The point of paging: short requests no longer pay for a full W-wide row.

    The pool holds exactly what a 2-row RowKVCache would (2 * W tokens), yet short
    requests -- each needing a few blocks, not a whole row -- fit 8 at a time.
    """
    model, enc, cfg = parts
    bs = 8
    sched = PagedScheduler(model, enc, cfg, max_seqs=8, block_size=bs, max_tokens=W,
                           num_blocks=1 + 2 * W // bs)
    sched.run(PROMPTS * 2, max_new_tokens=4)
    assert sched.peak_active > 2, f'only {sched.peak_active} resident -- no better than 2 rows'


def test_a_tight_pool_makes_requests_wait_not_fail(parts):
    """Blocks for exactly one request at a time: everything still completes, in order."""
    model, enc, cfg = parts
    max_new = 4
    longest = max(len(enc.encode(p)) for p in PROMPTS)
    bs = 4
    one_request = -(-(longest + max_new) // bs)               # ceil
    sched = PagedScheduler(model, enc, cfg, max_seqs=4, block_size=bs, max_tokens=W,
                           num_blocks=1 + one_request)

    got = sched.run(PROMPTS, max_new_tokens=max_new)

    assert got == [greedy_reference(model, enc, p, max_new) for p in PROMPTS]
    assert sched.peak_blocks <= one_request


def test_every_block_is_returned_after_a_run(parts):
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W)
    sched.run(PROMPTS * 2, max_new_tokens=4)

    assert sched.allocator.num_free == sched.allocator.capacity, 'blocks leaked'
    assert sched.num_active == 0
    assert all(b == [] for b in sched.row_blocks)
    assert not bool(sched.active.any())


def test_two_runs_in_a_row_give_identical_results(parts):
    """_reset must reset the allocator too, or the second run starts with a leaked pool."""
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W)
    first = sched.run(PROMPTS, max_new_tokens=4)
    second = sched.run(PROMPTS, max_new_tokens=4)
    assert first == second


def test_no_block_is_owned_by_two_live_requests(parts):
    """Checked at every decode step, against both the host mirror and the device table."""
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W)
    real_decode = sched._decode
    checked = []

    def checking_decode(sampler):
        live = [b for row in range(sched.B) if sched.row_active[row] for b in sched.row_blocks[row]]
        assert len(live) == len(set(live)), f'a block is owned twice: {live}'
        for row in range(sched.B):
            n = len(sched.row_blocks[row])
            assert sched.cache.block_table[row, :n].tolist() == sched.row_blocks[row], \
                f'device block table out of sync for row {row}'
        checked.append(True)
        real_decode(sampler)

    sched._decode = checking_decode
    sched.run(PROMPTS * 2, max_new_tokens=4)
    assert checked, 'no decode step was observed'


def test_a_request_that_can_never_fit_is_rejected(parts):
    """It would otherwise wait forever: even an empty pool cannot hold it."""
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                           num_blocks=3)                       # 2 usable blocks = 8 tokens
    with pytest.raises(ValueError, match='never fit'):
        sched.run(['It was a bright cold day in April'], max_new_tokens=6)


def test_a_prompt_too_long_for_the_width_is_rejected(parts):
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=1, block_size=4, max_tokens=8)
    with pytest.raises(ValueError, match='does not fit'):
        sched.run(['It was a bright cold day in April and the clocks'], max_new_tokens=6)


def test_block_size_must_divide_the_width(parts):
    model, enc, cfg = parts
    with pytest.raises(ValueError, match='must divide'):
        PagedScheduler(model, enc, cfg, max_seqs=2, block_size=5, max_tokens=W)


def test_a_watermark_never_blocks_an_empty_pool(parts):
    """With nothing running, the watermark is ignored -- otherwise the loop would spin."""
    model, enc, cfg = parts
    sched = PagedScheduler(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                           num_blocks=4, watermark=100)        # watermark > the whole pool
    got = sched.run(['Hello', 'A'], max_new_tokens=3)
    assert got == [greedy_reference(model, enc, p, 3) for p in ['Hello', 'A']]


# -- logging -------------------------------------------------------------------

def test_each_request_logs_one_admit_and_one_retire(parts, caplog):
    import logging
    model, enc, cfg = parts
    caplog.set_level(logging.INFO, logger='gpt2.inference.continuous_batch.paged_scheduler')
    PagedScheduler(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(
        PROMPTS, max_new_tokens=3)

    msgs = [r.getMessage() for r in caplog.records]
    for req in range(len(PROMPTS)):
        assert sum(m.startswith(f'admit  req {req} ') for m in msgs) == 1, f'req {req} admit'
        assert sum(m.startswith(f'retire req {req} ') for m in msgs) == 1, f'req {req} retire'


def test_a_wait_is_logged_once_not_every_step(parts, caplog):
    import logging
    model, enc, cfg = parts
    caplog.set_level(logging.INFO, logger='gpt2.inference.continuous_batch.paged_scheduler')
    longest = max(len(enc.encode(p)) for p in PROMPTS)
    sched = PagedScheduler(model, enc, cfg, max_seqs=4, block_size=4, max_tokens=W,
                           num_blocks=1 + -(-(longest + 4) // 4))      # one request at a time
    sched.run(PROMPTS, max_new_tokens=4)

    waits = [r.getMessage() for r in caplog.records if r.getMessage().startswith('wait')]
    assert waits, 'a tight pool should make requests wait'
    assert len(waits) == len(set(waits)), 'the same wait was logged repeatedly'
