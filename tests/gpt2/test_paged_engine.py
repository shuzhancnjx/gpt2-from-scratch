"""The paged engine end to end (PagedScheduler + BlockManager + ModelRunner, wired by PagedEngine).

The headline invariant is the same as for RowEngine: rotating requests through a
paged pool must not change any request's text. On top of that, paging adds
bookkeeping invariants -- every block returned, no block owned twice, requests
wait (not fail) when the pool is tight -- that text comparison cannot see.
"""

import pytest
import torch

from model.gpt2.gpt import GPT, GPTConfig
from inference.continuous_batch import PagedEngine, RowEngine

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
    got = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(
        PROMPTS, max_new_tokens=6)
    assert got == [greedy_reference(model, enc, p, 6) for p in PROMPTS]


@pytest.mark.parametrize('max_seqs,block_size', [(1, 4), (2, 8), (4, 16), (3, 4)])
def test_paged_matches_the_row_scheduler(parts, max_seqs, block_size):
    """Same prompts, any slot count and block size: identical text to the baseline."""
    model, enc, cfg = parts
    expected = RowEngine(model, enc, cfg, batch_size=2, max_tokens=W).run(PROMPTS, max_new_tokens=5)
    got = PagedEngine(model, enc, cfg, max_seqs=max_seqs, block_size=block_size,
                         max_tokens=W).run(PROMPTS, max_new_tokens=5)
    assert got == expected


def test_paged_runs_on_every_available_device(parts, device):
    model, enc, cfg = parts
    expected = RowEngine(model.to('cpu'), enc, cfg, batch_size=2, max_tokens=W).run(
        PROMPTS, max_new_tokens=4)
    got = PagedEngine(model.to(device), enc, cfg, max_seqs=2, block_size=8,
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
    sched = PagedEngine(model, enc, cfg, max_seqs=8, block_size=bs, max_tokens=W,
                           num_blocks=1 + 2 * W // bs)
    sched.run(PROMPTS * 2, max_new_tokens=4)
    assert sched.stats.peak_active > 2, f'only {sched.stats.peak_active} resident -- no better than 2 rows'


def test_a_tight_pool_makes_requests_wait_not_fail(parts):
    """Blocks for exactly one request at a time: everything still completes, in order."""
    model, enc, cfg = parts
    max_new = 4
    longest = max(len(enc.encode(p)) for p in PROMPTS)
    bs = 4
    one_request = -(-(longest + max_new) // bs)               # ceil
    sched = PagedEngine(model, enc, cfg, max_seqs=4, block_size=bs, max_tokens=W,
                           num_blocks=1 + one_request)

    got = sched.run(PROMPTS, max_new_tokens=max_new)

    assert got == [greedy_reference(model, enc, p, max_new) for p in PROMPTS]
    assert sched.stats.peak_blocks <= one_request


def test_every_block_is_returned_after_a_run(parts):
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W)
    sched.run(PROMPTS * 2, max_new_tokens=4)

    assert sched.blocks.allocator.num_free == sched.blocks.allocator.capacity, 'blocks leaked'
    assert sched.scheduler.num_decoding == 0
    assert all(a is None for a in sched.scheduler.rows)
    assert not bool(sched.runner.active_gpu.any()), "a retired row is still active on the GPU"


def test_two_runs_in_a_row_give_identical_results(parts):
    """_reset must reset the allocator too, or the second run starts with a leaked pool."""
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W)
    first = sched.run(PROMPTS, max_new_tokens=4)
    second = sched.run(PROMPTS, max_new_tokens=4)
    assert first == second


def test_no_block_is_owned_by_two_live_requests(parts):
    """Checked at every decode step, against both the scheduler's records and the device table."""
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W)
    real_decode = sched.runner.decode
    checked = []

    def checking_decode(sampling_params):
        decoding = sched.scheduler.decoding
        live = [b for a in decoding for b in a.blocks]
        assert len(live) == len(set(live)), f'a block is owned twice: {live}'
        for a in decoding:
            assert sched.runner.cache.block_table_gpu[a.row, :len(a.blocks)].tolist() == a.blocks, \
                f'device block table out of sync for row {a.row}'
        checked.append(True)
        return real_decode(sampling_params)

    sched.runner.decode = checking_decode
    sched.run(PROMPTS * 2, max_new_tokens=4)
    assert checked, 'no decode step was observed'


def test_a_request_that_can_never_fit_is_rejected(parts):
    """It would otherwise wait forever: even an empty pool cannot hold it."""
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                           num_blocks=3)                       # 2 usable blocks = 8 tokens
    with pytest.raises(ValueError, match='never fit'):
        sched.run(['It was a bright cold day in April'], max_new_tokens=6)


def test_a_prompt_too_long_for_the_width_is_rejected(parts):
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=1, block_size=4, max_tokens=8)
    with pytest.raises(ValueError, match='does not fit'):
        sched.run(['It was a bright cold day in April and the clocks'], max_new_tokens=6)


def test_block_size_must_divide_the_width(parts):
    model, enc, cfg = parts
    with pytest.raises(ValueError, match='must divide'):
        PagedEngine(model, enc, cfg, max_seqs=2, block_size=5, max_tokens=W)


def test_a_watermark_never_blocks_an_empty_pool(parts):
    """With nothing running, the watermark is ignored -- otherwise the loop would spin."""
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                           num_blocks=4, watermark=100)        # watermark > the whole pool
    got = sched.run(['Hello', 'A'], max_new_tokens=3)
    assert got == [greedy_reference(model, enc, p, 3) for p in ['Hello', 'A']]


# -- logging -------------------------------------------------------------------

def test_each_request_logs_one_admit_and_one_retire(parts, caplog):
    import logging
    model, enc, cfg = parts
    caplog.set_level(logging.INFO, logger='inference.continuous_batch.paged.paged_scheduler')
    PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(
        PROMPTS, max_new_tokens=3)

    msgs = [r.getMessage() for r in caplog.records]
    for req in range(len(PROMPTS)):
        assert sum(m.startswith(f'admit  req {req} ') for m in msgs) == 1, f'req {req} admit'
        assert sum(m.startswith(f'retire req {req} ') for m in msgs) == 1, f'req {req} retire'


def test_a_wait_is_logged_once_not_every_step(parts, caplog):
    import logging
    model, enc, cfg = parts
    caplog.set_level(logging.INFO, logger='inference.continuous_batch.paged.paged_scheduler')
    longest = max(len(enc.encode(p)) for p in PROMPTS)
    sched = PagedEngine(model, enc, cfg, max_seqs=4, block_size=4, max_tokens=W,
                           num_blocks=1 + -(-(longest + 4) // 4))      # one request at a time
    sched.run(PROMPTS, max_new_tokens=4)

    waits = [r.getMessage() for r in caplog.records if r.getMessage().startswith('wait')]
    assert waits, 'a tight pool should make requests wait'
    assert len(waits) == len(set(waits)), 'the same wait was logged repeatedly'


# -- CUDA graph decode ------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_graph_decode_matches_eager(parts):
    """Requests join and leave mid-run (more prompts than rows): replays must pick up
    each newly admitted row through the in-place buffers."""
    model, enc, cfg = parts
    model = model.to('cuda')
    prompts = PROMPTS * 3
    eager = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W, device='cuda')
    graphed = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                             device='cuda', cuda_graph=True)
    assert graphed.run(prompts, max_new_tokens=6) == eager.run(prompts, max_new_tokens=6)
    assert len(graphed.runner.decode_graphs) == 1, 'greedy run() should replay the greedy graph'
    assert graphed.blocks.allocator.num_free == graphed.blocks.allocator.capacity


def test_cuda_graph_is_ignored_off_cuda(parts):
    model, enc, cfg = parts
    s = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W, cuda_graph=True)
    assert s.runner.decode_graphs is None
    assert s.run(PROMPTS, max_new_tokens=3) == PagedEngine(
        model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(PROMPTS, max_new_tokens=3)


# -- the Triton kernel (always on CUDA) ------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton needs CUDA')
@pytest.mark.parametrize('graph', [False, True])
def test_cuda_paged_engine_matches_the_row_scheduler(parts, graph):
    """On CUDA the paged engine attends with the Triton kernel (prefill and decode, eager
    and graph-captured): same text as the row engine, requests joining mid-run."""
    model, enc, cfg = parts
    model = model.to('cuda')
    prompts = PROMPTS * 3
    expected = RowEngine(model, enc, cfg, batch_size=2, max_tokens=W, device='cuda').run(
        prompts, max_new_tokens=6)
    sched = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W,
                           device='cuda', cuda_graph=graph)
    assert sched.run(prompts, max_new_tokens=6) == expected
    assert sched.blocks.allocator.num_free == sched.blocks.allocator.capacity


# -- P1.7a: prefill attends over the prompt only ----------------------------------------

def test_prefill_never_gathers(parts, monkeypatch):
    """Prefill reads nothing back from the pool: only decode (all rows) may gather."""
    model, enc, cfg = parts
    from inference.continuous_batch.paged.paged_cache import PagedKVCache
    widths, original = [], PagedKVCache._gather

    def spy(self, pool, sel):
        widths.append(sel.stop - sel.start)
        return original(self, pool, sel)

    monkeypatch.setattr(PagedKVCache, '_gather', spy)
    PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W).run(PROMPTS, max_new_tokens=3)
    assert widths and all(w == 3 for w in widths), 'a 1-row gather means prefill read its row back'


def test_prefill_still_stores_the_prompt_for_decode(parts):
    """The prompt's k/v must land in the pool even though prefill doesn't read them:
    decode attends over them. Identical text to the uncached path proves it."""
    model, enc, cfg = parts
    got = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).run(
        PROMPTS, max_new_tokens=8)
    assert got == [greedy_reference(model, enc, p, 8) for p in PROMPTS]


# -- two-stage admission: prefill decoupled from decode rows --------------------------

def test_prefill_runs_ahead_of_free_rows(parts, monkeypatch):
    """With every row busy, waiting requests are prefilled into prefilled rows and promoted
    when a row frees -- same text as the row engine."""
    model, enc, cfg = parts
    prompts = PROMPTS * 3
    sched = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W)
    seen_prefilled = []
    original = sched.scheduler.promote

    def spy():
        seen_prefilled.append(len(sched.scheduler.prefilled))
        return original()

    monkeypatch.setattr(sched.scheduler, 'promote', spy)
    got = sched.run(prompts, max_new_tokens=6)
    expected = RowEngine(model, enc, cfg, batch_size=2, max_tokens=W).run(prompts, max_new_tokens=6)
    assert got == expected
    assert max(seen_prefilled) > 0, 'nothing was ever prefilled ahead of a free row'
    assert not sched.scheduler.prefilled and sorted(sched.blocks.free_prefilled_rows) == list(range(2, 4))
    assert sched.blocks.allocator.num_free == sched.blocks.allocator.capacity


def test_prefill_token_budget_splits_prefill_but_not_the_text(parts, monkeypatch):
    """max_prefill_tokens caps one step's prefill: more, smaller prefill forwards."""
    model, enc, cfg = parts
    counts = {}
    for budget in (None, 8):
        sched = PagedEngine(model, enc, cfg, max_seqs=4, block_size=4, max_tokens=W,
                               max_prefill_tokens=budget)
        calls, original = [], sched.runner.prefill

        def spy(rows, prompts, sampling_params, calls=calls, original=original):
            calls.append(sum(len(p) for p in prompts))
            return original(rows, prompts, sampling_params)

        monkeypatch.setattr(sched.runner, 'prefill', spy)
        counts[budget] = (sched.run(PROMPTS, max_new_tokens=5), calls)
    assert counts[8][0] == counts[None][0]
    assert len(counts[8][1]) > len(counts[None][1])
    longest = max(len(enc.encode(p)) for p in PROMPTS)
    assert all(n <= max(8, longest) for n in counts[8][1]), 'a step prefilled past its budget'


def test_promote_moves_the_block_table_row_and_pos(parts):
    from inference.continuous_batch.paged.block.block_allocator import SCRATCH
    model, enc, cfg = parts
    sched = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W)
    cache = sched.runner.cache
    cache.set_blocks(3, [5, 9])                         # prefilled row 3
    cache.pos_gpu[3] = 6
    cache.promote(3, 0)
    assert cache.block_table_gpu[0, :2].tolist() == [5, 9] and int(cache.pos_gpu[0]) == 6
    assert int(cache.pos_gpu[3]) == 0 and torch.all(cache.block_table_gpu[3] == SCRATCH)


def test_prefilled_buffer_needs_at_least_one_row(parts):
    """With no prefilled rows nothing could ever be admitted -- refuse it up front."""
    model, enc, cfg = parts
    with pytest.raises(ValueError, match='prefilled_rows'):
        PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W, prefilled_rows=0)


def test_one_prefilled_row_still_serves_everything(parts):
    """The smallest buffer: one prefilled request at a time, same text as the row engine."""
    model, enc, cfg = parts
    prompts = PROMPTS * 2
    got = PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                         prefilled_rows=1).run(prompts, max_new_tokens=5)
    expected = RowEngine(model, enc, cfg, batch_size=2, max_tokens=W).run(prompts, max_new_tokens=5)
    assert got == expected


# -- prefill CUDA graphs ----------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_prefill_graphs_match_eager(parts):
    """Packed prefill replayed from bucket graphs: same text as the eager engine, padding
    confined to the pad row (still all scratch afterwards)."""
    from inference.continuous_batch.paged.block.block_allocator import SCRATCH
    model, enc, cfg = parts
    model = model.to('cuda')
    prompts = PROMPTS * 3                                  # several waves, varied prompt counts
    eager = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W, device='cuda')
    graphed = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W,
                             device='cuda', cuda_graph=True)
    assert graphed.run(prompts, max_new_tokens=6) == eager.run(prompts, max_new_tokens=6)
    assert len(graphed.runner.prefill_graphs) == len(graphed.runner.prefill_graphs.buckets), \
        'every bucket should be captured before the run'
    assert torch.all(graphed.runner.cache.block_table_gpu[graphed.runner.cache.pad_row] == SCRATCH)
    assert graphed.blocks.allocator.num_free == graphed.blocks.allocator.capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_oversized_prefill_falls_back_to_eager(parts):
    """A batch bigger than every bucket runs the eager prefill -- same text, nothing captured."""
    model, enc, cfg = parts
    model = model.to('cuda')
    graphed = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W,
                             device='cuda', cuda_graph=True)
    graphed.runner.prefill_graphs.buckets = (4,)                  # every real batch overflows it
    eager = PagedEngine(model, enc, cfg, max_seqs=3, block_size=4, max_tokens=W, device='cuda')
    assert graphed.run(PROMPTS, max_new_tokens=4) == eager.run(PROMPTS, max_new_tokens=4)
    assert len(graphed.runner.prefill_graphs) == 1, 'the one bucket is captured up front, never replayed'


def test_graph_objects_are_not_tested_for_truth(parts):
    """BaseGraphs has __len__, so an empty one is falsy: the scheduler must check `is not
    None`, or graphs are never captured (they start empty) and everything runs eager."""
    import inspect
    from inference.continuous_batch import model_runner
    from inference.continuous_batch.paged import engine
    src = inspect.getsource(model_runner) + inspect.getsource(engine)
    assert 'if self.decode_graphs:' not in src and 'if self.prefill_graphs:' not in src
    assert '.prefill_graphs is not None' in src


def test_prefill_budget_auto_is_uncapped_without_graphs(parts):
    """'auto': no cap when prefill runs eagerly (each forward pays a fixed cost)."""
    model, enc, cfg = parts
    assert PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W).scheduler.max_prefill_tokens is None
    assert PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                          max_prefill_tokens=300).scheduler.max_prefill_tokens == 300


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_prefill_budget_auto_caps_with_graphs(parts):
    model, enc, cfg = parts
    s = PagedEngine(model.to('cuda'), enc, cfg, max_seqs=2, block_size=4, max_tokens=W,
                       device='cuda', cuda_graph=True)
    assert s.scheduler.max_prefill_tokens == 1024
    assert PagedEngine(model, enc, cfg, max_seqs=2, block_size=4, max_tokens=W, device='cuda',
                          cuda_graph=True, max_prefill_tokens=None).scheduler.max_prefill_tokens is None


# -- the components on their own: policy without a model -------------------------------

class _FakeCache:
    """Just enough of PagedKVCache for the BlockManager: records table writes."""
    def __init__(self):
        self.table = {}
    def set_blocks(self, row, blocks):
        self.table[row] = list(blocks)
    def promote(self, src, dst):
        self.table[dst] = self.table.pop(src)
    def release(self, row):
        self.table.pop(row, None)


def _scheduler(max_seqs=2, num_blocks=9, block_size=4, prefilled_rows=2, **kw):
    from inference.continuous_batch.paged.block.block_manager import BlockManager
    from inference.continuous_batch.paged.paged_scheduler import PagedScheduler as PagedPolicy
    blocks = BlockManager(_FakeCache(), num_blocks, block_size, max_seqs, prefilled_rows)
    return PagedPolicy(blocks, max_seqs, max_tokens=32, eot=0, **kw), blocks


def test_scheduler_policy_runs_without_a_model():
    """The policy is pure CPU: requests in, decisions out -- no model, no GPU."""
    sched, blocks = _scheduler()
    sched.add([[1] * 5, [1] * 5, [1] * 5])
    out = sched.schedule(max_new_tokens=3)               # 2 prefilled rows -> 2 admitted
    assert [a.req for a in out.prefill] == [0, 1] and out.promote == []
    sched.prefilled_with(out.prefill, [7, 7])
    assert [a.req for a in sched.promote()] == [0, 1]    # 2 decode rows free
    assert sched.num_decoding == 2 and [a.req for a in sched.waiting] == [2]
    retired = sched.update(tokens=[0, 9], max_new_tokens=3)   # row 0 hits end-of-text
    assert [a.req for a in retired] == [0] and sched.finished[0] == [7, 0]
    assert blocks.num_used == blocks.blocks_for(5 + 3)        # only request 1 still holds blocks


def test_scheduler_budget_and_fcfs_without_a_model():
    sched, _ = _scheduler(prefilled_rows=4, num_blocks=20, max_prefill_tokens=6)
    sched.add([[1] * 4, [1] * 4, [1] * 4])
    assert [a.req for a in sched.schedule(max_new_tokens=2).prefill] == [0]   # 4 + 4 > 6
    tight, _ = _scheduler(prefilled_rows=4, num_blocks=1 + 3)    # 3 usable blocks
    tight.add([[1] * 8, [1] * 2])                                  # needs 3, then 1
    first = tight.schedule(max_new_tokens=2).prefill
    assert [a.req for a in first] == [0]
    tight.prefilled_with(first, [5])                               # as the engine does next
    assert tight.schedule(max_new_tokens=2).prefill == []          # no room: req 1 waits
