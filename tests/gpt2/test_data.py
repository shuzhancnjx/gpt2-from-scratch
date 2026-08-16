import os

import pytest

from gpt2.data import DataLoaderLite

from conftest import requires_data

pytestmark = requires_data


def make(start=0, n=3, split='train', rank=0, procs=1, B=2, T=64):
    return DataLoaderLite(B=B, T=T, device='cpu', process_rank=rank, num_processes=procs,
                          split=split, master_process=False, start_shard=start, max_shards=n)


def names(loader):
    return [os.path.basename(s) for s in loader.shards]


def test_window_selects_by_offset_and_count():
    assert len(names(make(0, 3))) == 3
    first, second = names(make(0, 3)), names(make(3, 3))
    assert not set(first) & set(second)          # successive windows are disjoint
    assert make(0, 3).shards[3:] == []


def test_window_bounds_total_tokens():
    one, three = make(0, 1), make(0, 3)
    assert three.total_num_tokens == 3 * one.total_num_tokens


def test_batch_shapes_and_offset():
    loader = make(B=2, T=64)
    x, y = loader.next_batch()
    assert x.shape == (2, 64) and y.shape == (2, 64)
    # y is x shifted by one: the model predicts the next token
    assert (x[0, 1:] == y[0, :-1]).all()


def test_ranks_read_disjoint_slices():
    a, b = make(rank=0, procs=2), make(rank=1, procs=2)
    xa, _ = a.next_batch()
    xb, _ = b.next_batch()
    assert not (xa == xb).all()


def test_resume_in_same_window_is_exact():
    a = make(3, 3)
    for _ in range(5):
        a.next_batch()
    state = a.state_dict()

    b = make(3, 3)
    b.load_state_dict(state)
    assert (b.current_shard, b.current_position) == (a.current_shard, a.current_position)
    assert state['shard_name'] == os.path.basename(a.shards[a.current_shard])


def test_resume_into_moved_window_restarts_there():
    """A shard index means nothing once the window moves -- resuming onto fresh
    data must not silently rewind into data the model already saw."""
    a = make(3, 3)
    for _ in range(5):
        a.next_batch()
    state = a.state_dict()

    b = make(6, 3)
    b.load_state_dict(state)
    assert b.current_shard == 0
    assert os.path.basename(b.shards[0]) not in state['shard_name']


def test_old_format_checkpoint_still_loads():
    b = make(0, 3)
    b.load_state_dict({'current_shard': 1, 'base_position': 4096})   # pre-shard_name
    assert (b.current_shard, b.current_position) == (1, 4096)


def test_rank_offset_is_not_baked_into_saved_state():
    """Rank 0 saves the checkpoint; every rank must restore its own offset."""
    a = make(0, 3, rank=1, procs=2, B=2, T=64)
    state = a.state_dict()
    assert state['base_position'] == a.current_position - 1 * 2 * 64


@pytest.mark.parametrize('start,n', [(10 ** 6, 2), (0, 10 ** 6)])
def test_invalid_windows_fail_loudly(start, n):
    with pytest.raises(AssertionError):
        make(start, n)


def test_missing_data_root_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match='prepare_data'):
        DataLoaderLite(B=1, T=8, device='cpu', process_rank=0, num_processes=1,
                       split='train', master_process=False, data_root='/nope/not/here')
