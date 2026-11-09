"""Paged engine: requests own fixed-size blocks from a shared pool (PagedKVCache).

    engine.py           PagedEngine: the orchestrator
    paged_scheduler.py  PagedScheduler: policy
    block/              BlockManager, BlockAllocator: memory
    paged_cache.py      PagedKVCache: the pools, block table, positions
    attention/          the Triton attention kernels (decode, prefill) -- CUDA only, so
                        not imported here; PagedKVCache imports them by module path
    graph/              PrefillGraphs, DecodeGraphs: CUDA graphs
"""
