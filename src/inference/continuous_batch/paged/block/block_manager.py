"""BlockManager: who owns which KV blocks and block-table rows (CPU bookkeeping).

Memory only -- no policy, no model. The PagedScheduler asks it for room; it answers from the
BlockAllocator and keeps the cache's block table in step:

    reserve(need)              a free prefilled row + `need` blocks; writes the row's table
    move_to_decode(slot, row)  the request's table row and pos move into a decode row
    release(row, blocks)       blocks back to the pool, the row back to scratch

Block-table rows (PagedKVCache):

    [0, max_seqs)                          decode rows
    [max_seqs, max_seqs + prefilled_rows)  prefilled rows: blocks written, waiting for a decode row
    the last row                           pad row: always SCRATCH (prefill-graph padding)
"""

from math import ceil

from inference.continuous_batch.paged.block.block_allocator import BlockAllocator


class BlockManager:

    def __init__(self, cache, num_blocks, block_size, max_seqs, prefilled_rows):
        self.cache = cache
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.prefilled_rows = prefilled_rows
        self.allocator = BlockAllocator(num_blocks)
        self.reset()

    def reset(self):
        """Every block free, every prefilled row free. (The cache's table is reset by the
        ModelRunner, which owns the device tensors.)"""
        self.allocator.reset()
        self.free_prefilled_rows = list(range(self.max_seqs, self.max_seqs + self.prefilled_rows))

    # -- questions ------------------------------------------------------------------

    def blocks_for(self, tokens):
        return ceil(tokens / self.block_size)

    @property
    def capacity(self):
        return self.allocator.capacity

    @property
    def num_free(self):
        return self.allocator.num_free

    @property
    def num_used(self):
        return self.allocator.capacity - self.allocator.num_free

    # -- changes --------------------------------------------------------------------

    def reserve(self, need):
        """A free prefilled row and `need` blocks, recorded in that row's block table
        (before the prefill writes into them). Returns (slot, blocks)."""
        slot = self.free_prefilled_rows.pop(0)
        blocks = self.allocator.allocate(need)
        self.cache.set_blocks(slot, blocks)
        return slot, blocks

    def move_to_decode(self, slot, row):
        """Prefilled row -> decode row: copy the table row and pos; free the prefilled row.
        The K/V stay where they are in the pool."""
        self.cache.promote(slot, row)
        self.free_prefilled_rows.append(slot)

    def release(self, row, blocks):
        """A finished request: its blocks back to the pool, its decode row back to scratch."""
        self.allocator.free(blocks)
        self.cache.release(row)
