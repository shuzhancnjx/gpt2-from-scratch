"""Step a of P1.4: the free list. Pure host logic -- no device involved."""

import pytest

from gpt2.inference.continuous_batch.paged.block_allocator import SCRATCH, BlockAllocator


def test_capacity_excludes_scratch():
    a = BlockAllocator(8)
    assert a.capacity == 7
    assert a.num_free == 7


def test_allocate_returns_distinct_ids_and_never_scratch():
    a = BlockAllocator(8)
    blocks = a.allocate(7)

    assert len(set(blocks)) == 7, 'handed out the same block twice'
    assert SCRATCH not in blocks, 'scratch must never be handed out'
    assert a.num_free == 0


def test_allocating_more_than_free_raises_and_takes_nothing():
    a = BlockAllocator(4)
    a.allocate(2)

    with pytest.raises(RuntimeError, match='out of blocks'):
        a.allocate(2)
    assert a.num_free == 1, 'a failed allocation must not consume blocks'


def test_free_restores_the_count_and_blocks_are_reused():
    a = BlockAllocator(4)
    first = a.allocate(3)
    a.free(first)

    assert a.num_free == 3
    assert sorted(a.allocate(3)) == sorted(first)


def test_double_free_is_rejected_without_side_effects():
    a = BlockAllocator(4)
    blocks = a.allocate(2)
    a.free(blocks)

    with pytest.raises(ValueError, match='not allocated'):
        a.free(blocks)
    assert a.num_free == 3


def test_freeing_scratch_or_an_unknown_id_is_rejected():
    a = BlockAllocator(4)
    a.allocate(1)
    with pytest.raises(ValueError):
        a.free([SCRATCH])
    with pytest.raises(ValueError):
        a.free([99])


def test_a_bad_free_leaves_valid_blocks_allocated():
    """Validation happens before mutation: one bad id frees nothing."""
    a = BlockAllocator(4)
    good = a.allocate(1)
    with pytest.raises(ValueError):
        a.free(good + [99])
    assert a.num_free == 2, 'the valid block was freed despite the error'


def test_reset_frees_everything():
    a = BlockAllocator(5)
    a.allocate(4)
    a.reset()
    assert a.num_free == 4


def test_too_few_blocks_is_rejected():
    with pytest.raises(ValueError):
        BlockAllocator(1)
