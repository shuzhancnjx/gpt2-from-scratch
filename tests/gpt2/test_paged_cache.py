"""Steps b and c of P1.4: the paged cache's write path and gather path.

Like test_serving.py these drive the cache directly, not through a model: the
properties under test -- every token lands at its slot, gathering reproduces what
RowKVCache would hold, idle rows touch only scratch -- are exact facts about the
buffers, much easier to assert there than to infer from generated text.
"""

import pytest
import torch

from model.gpt2.gpt import GPTConfig
from inference.continuous_batch import RowKVCache
from inference.continuous_batch.paged.block_allocator import SCRATCH
from inference.continuous_batch.paged.paged_cache import PagedKVCache

TINY = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=32,
            dropout=0.0, bias=True)
BS = 4                     # block size
W = 16                     # max_blocks_per_seq * BS == the gathered width


@pytest.fixture
def config():
    return GPTConfig(**TINY)


def make_cache(config, device, max_seqs=3, num_blocks=10):
    return PagedKVCache(config, num_blocks=num_blocks, block_size=BS, max_seqs=max_seqs,
                        max_blocks_per_seq=W // BS, device=device)


def tokens(B, T, config, start, device):
    """(B, n_head, T, head_size) where token t of batch row b is filled with
    start + 100*b + t -- every token distinguishable from every other."""
    hs = config.n_embd // config.n_head
    vals = start + 100 * torch.arange(B).view(B, 1, 1, 1) + torch.arange(T).view(1, 1, T, 1)
    return vals.expand(B, config.n_head, T, hs).float().to(device)


def pool_at(cache, layer, physical, offset):
    """What the K pool holds for one slot (n_head, head_size)."""
    return cache.key[layer][physical, offset]


# -- step b: the write path ------------------------------------------------------

def test_prefill_writes_each_token_at_its_slot(config, device):
    cache = make_cache(config, device)
    cache.set_blocks(1, [7, 3])                       # scattered, out of order on purpose
    k = tokens(1, 6, config, start=1, device=device)

    cache.update(0, k, k, row=1)

    for p in range(6):
        physical, offset = [7, 3][p // BS], p % BS
        assert torch.all(pool_at(cache, 0, physical, offset) == 1 + p), f'position {p} misplaced'


def test_decode_writes_each_row_at_its_own_pos(config, device):
    cache = make_cache(config, device)
    cache.set_blocks(0, [2, 5])
    cache.set_blocks(2, [9])
    cache.pos_gpu = torch.tensor([5, 0, 2], device=device)     # row 0 in its 2nd block

    cache.update(0, tokens(3, 1, config, 1, device), tokens(3, 1, config, 1, device))

    assert torch.all(pool_at(cache, 0, 5, 1) == 1)         # row 0, pos 5 → block 5, offset 1
    assert torch.all(pool_at(cache, 0, 9, 2) == 201)       # row 2, pos 2 → block 9, offset 2


def test_slots_of_live_rows_are_unique(config, device):
    """Two real requests sharing a slot would silently corrupt one of them."""
    cache = make_cache(config, device)
    cache.set_blocks(0, [1, 2])
    cache.set_blocks(1, [3, 4])
    cache.set_blocks(2, [5])
    rows = torch.tensor([0, 0, 1, 1, 2], device=device)
    positions = torch.tensor([3, 4, 3, 4, 0], device=device)

    slots = cache.slots(rows, positions)

    assert len(set(slots.tolist())) == len(slots)


def test_idle_rows_write_only_to_scratch(config, device):
    """Decode writes every row; rows with no blocks must land in scratch and nowhere else."""
    cache = make_cache(config, device)
    cache.set_blocks(1, [6])
    cache.pos_gpu = torch.tensor([0, 1, 0], device=device)      # rows 0 and 2 idle

    cache.update(0, tokens(3, 1, config, 1, device), tokens(3, 1, config, 1, device))

    written = (cache.key[0] != 0).any(dim=(1, 2, 3)).nonzero().flatten().tolist()
    assert set(written) == {SCRATCH, 6}, f'idle rows wrote into real blocks: {written}'


def test_all_layers_write_at_the_same_slots(config, device):
    cache = make_cache(config, device)
    cache.set_blocks(0, [4])
    k = tokens(1, 3, config, 1, device)
    for layer in range(config.n_layer):
        cache.update(layer, k, k, row=0)
    cache.advance(3, row=0)

    assert int(cache.pos_gpu[0]) == 3, 'pos moved per layer instead of per forward'
    for layer in range(config.n_layer):
        assert torch.all(cache.key[layer][4, :3] != 0), f'layer {layer} wrote nothing'
        assert torch.all(cache.key[layer][4, 3:] == 0), f'layer {layer} wrote past 3'


# -- step c: the gather path -----------------------------------------------------

def test_gather_matches_row_cache(config, device):
    """Same token stream into both caches: the attended history must be identical."""
    row_cache = RowKVCache(batch_size=3, device=device, config=config, max_tokens=W)
    paged = make_cache(config, device)
    paged.set_blocks(0, [8, 2, 5])                        # scattered on purpose
    paged.set_blocks(2, [1, 9])

    def step(cache, layer, k, row=None):
        return cache.update(layer, k, k, row=row)

    # prefill rows 0 and 2 with different lengths
    for row, T, start in [(0, 6, 1), (2, 3, 500)]:
        k = tokens(1, T, config, start, device)
        for layer in range(config.n_layer):
            step(row_cache, layer, k, row)
            step(paged, layer, k, row)
        row_cache.advance(T, row=row)
        paged.advance(T, row=row)

    # three decode steps for the active rows
    active = torch.tensor([True, False, True], device=device)
    for i in range(3):
        k = tokens(3, 1, config, 1000 + 10 * i, device)
        for layer in range(config.n_layer):
            got_row = step(row_cache, layer, k)
            got_paged = step(paged, layer, k)
        row_cache.advance(1, active=active)
        paged.advance(1, active=active)

    assert got_paged[0].shape == got_row[0].shape == (3, config.n_head, W, config.n_embd // config.n_head)
    for row in (0, 2):
        n = int(paged.pos_gpu[row])                           # written history; beyond it is masked
        assert torch.equal(got_paged[0][row, :, :n], got_row[0][row, :, :n]), f'row {row} K differs'
        assert torch.equal(got_paged[1][row, :, :n], got_row[1][row, :, :n]), f'row {row} V differs'


def test_prefill_gather_returns_only_that_row(config, device):
    cache = make_cache(config, device)
    cache.set_blocks(2, [3])
    k, _ = cache.update(0, tokens(1, 2, config, 1, device), tokens(1, 2, config, 1, device), row=2)
    assert k.shape[0] == 1


def test_release_points_the_row_back_at_scratch(config, device):
    cache = make_cache(config, device)
    cache.set_blocks(1, [4, 6])
    cache.pos_gpu[1] = 5
    cache.release(1)

    assert int(cache.pos_gpu[1]) == 0
    assert torch.all(cache.block_table_gpu[1] == SCRATCH)


def test_too_many_blocks_for_a_row_is_rejected(config):
    cache = make_cache(config, 'cpu')
    with pytest.raises(ValueError):
        cache.set_blocks(0, list(range(1, W // BS + 2)))


def test_pools_start_finite(config):
    """Masked columns still flow through attention as 0 * value: NaN would poison them."""
    cache = make_cache(config, 'cpu')
    assert all(torch.isfinite(t).all() for t in cache.key + cache.value)
