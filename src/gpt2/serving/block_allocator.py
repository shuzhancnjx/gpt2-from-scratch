"""Host-side bookkeeping for the paged KV pool: which physical blocks are free.

The pool itself lives on the device (PagedKVCache). This class never touches it --
"allocating a block" is handing out an integer id that kernels later turn into an
address inside memory PyTorch allocated once, at startup.

Block 0 is reserved as SCRATCH and never handed out. Rows with no request point
their whole block-table row at it, so the decode step can write every row
unconditionally (no data-dependent filter, no GPU sync) and the garbage lands
somewhere nobody reads unmasked.
"""

from collections import deque

SCRATCH = 0


class BlockAllocator:

    def __init__(self, num_blocks):
        if num_blocks < 2:
            raise ValueError(f'need at least 2 blocks (one is scratch), got {num_blocks}')
        self.num_blocks = num_blocks
        self.reset()

    @property
    def capacity(self):
        """Blocks that can ever be handed out -- everything except scratch."""
        return self.num_blocks - 1

    @property
    def num_free(self):
        return len(self._free)

    def allocate(self, n):
        """Pop n free block ids. All-or-nothing: on failure nothing is taken."""
        if n < 0:
            raise ValueError(f'cannot allocate {n} blocks')
        if n > len(self._free):
            raise RuntimeError(f'out of blocks: need {n}, {len(self._free)} free')
        blocks = [self._free.popleft() for _ in range(n)]
        self._used.update(blocks)
        return blocks

    def free(self, blocks):
        """Return blocks to the pool. Validated before any mutation, so a bad call
        (double free, scratch, unknown id) leaves the allocator untouched."""
        if len(set(blocks)) != len(blocks):
            raise ValueError(f'duplicate ids in free: {blocks}')
        for b in blocks:
            if b not in self._used:
                raise ValueError(f'block {b} is not allocated (double free, or scratch?)')
        for b in blocks:
            self._used.remove(b)
            self._free.append(b)

    def reset(self):
        self._free = deque(range(1, self.num_blocks))
        self._used = set()
