"""KV block bookkeeping for the paged engine (CPU only):

    block_allocator.py  BlockAllocator  the free list of blocks; block 0 is scratch
    block_manager.py    BlockManager    which blocks and block-table rows each request holds
"""
