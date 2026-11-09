"""Tests for the per-row KV cache behind continuous batching.

These exercise the cache directly rather than through a model. The property
under test is that rows are *independent* -- writing, advancing or retiring one
row must leave every other row untouched -- and that is much easier to assert
on the buffers than to infer from generated text.
"""

import pytest
import torch

from model.gpt2.gpt import GPTConfig
from inference.prefill_batch import PrefillBatch
from inference.continuous_batch import RowKVCache

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


def prefill(cache, layer, row, k, advance=True):
    """Prefill one prompt into `row` the way the scheduler does: a 1-prompt PrefillBatch."""
    batch = PrefillBatch.build([row], [k.size(2)], 'cpu')
    cache.write(layer, k, k, batch)
    if advance:
        cache.advance(k.size(2), prefill=batch)


# -- prefill -----------------------------------------------------------------

def test_prefill_writes_one_row_from_slot_zero(cache, config):
    prefill(cache, 0, 1, kv(1, 5, config, 1.0))

    assert cache.pos_gpu.tolist() == [0, 5, 0], 'only row 1 should have advanced'
    assert torch.all(cache.key[0][1, :, :5] == 1.0)
    assert torch.all(cache.key[0][1, :, 5:] == 0.0), 'wrote past the prompt'
    assert torch.all(cache.key[0][0] == 0.0), 'row 0 was touched'
    assert torch.all(cache.key[0][2] == 0.0), 'row 2 was touched'


def test_packed_prefill_writes_each_prompt_into_its_own_row(cache, config):
    """Two prompts packed end to end: each lands in its row from slot 0, nowhere else."""
    k = torch.cat([kv(1, 3, config, 1.0), kv(1, 5, config, 2.0)], dim=2)     # (1, H, 8, D)
    batch = PrefillBatch.build([2, 0], [3, 5], 'cpu')
    cache.write(0, k, k, batch)
    cache.advance(8, prefill=batch)

    assert cache.pos_gpu.tolist() == [5, 0, 3]
    assert torch.all(cache.key[0][2, :, :3] == 1.0) and torch.all(cache.key[0][2, :, 3:] == 0.0)
    assert torch.all(cache.key[0][0, :, :5] == 2.0) and torch.all(cache.key[0][0, :, 5:] == 0.0)
    assert torch.all(cache.key[0][1] == 0.0), 'row 1 was not in the batch'


# -- decode ------------------------------------------------------------------

def test_decode_writes_each_row_at_its_own_pos(cache, config):
    """The core of continuous batching: rows sit at different lengths."""
    cache.pos_gpu = torch.tensor([2, 7, 0])

    cache.write(0, kv(3, 1, config, 9.0), kv(3, 1, config, 9.0))

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
        prefill(cache, 0, row, kv(1, 4, config, row + 1.0))

    cache.release(1)

    assert cache.pos_gpu.tolist() == [4, 0, 4]
    assert torch.all(cache.key[0][1] == 0.0), 'row 1 should be cleared'
    assert torch.all(cache.key[0][0, :, :4] == 1.0), 'row 0 was disturbed'
    assert torch.all(cache.key[0][2, :, :4] == 3.0), 'row 2 was disturbed'


def test_released_row_accepts_a_new_request(cache, config):
    prefill(cache, 0, 0, kv(1, 6, config, 1.0))
    cache.release(0)
    prefill(cache, 0, 0, kv(1, 2, config, 5.0))

    assert int(cache.pos_gpu[0]) == 2
    assert torch.all(cache.key[0][0, :, :2] == 5.0)
    assert torch.all(cache.key[0][0, :, 2:] == 0.0), 'stale KV from the previous request'


# -- layer lockstep ----------------------------------------------------------

def test_all_layers_stay_in_lockstep(cache, config):
    """advance() belongs to the caller, once per forward -- never inside write()."""
    batch = PrefillBatch.build([0], [3], 'cpu')
    for layer in range(len(cache.key)):
        cache.write(layer, kv(1, 3, config, 1.0), kv(1, 3, config, 1.0), batch)
    cache.advance(3, prefill=batch)

    assert int(cache.pos_gpu[0]) == 3, 'pos moved per layer instead of per forward'
    for layer in range(len(cache.key)):
        assert torch.all(cache.key[layer][0, :, :3] == 1.0), f'layer {layer} wrote nothing'
        assert torch.all(cache.key[layer][0, :, 3:] == 0.0), f'layer {layer} wrote past 3'
