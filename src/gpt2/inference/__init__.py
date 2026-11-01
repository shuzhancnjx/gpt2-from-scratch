"""Text generation, by batching mode.

static_batch      one fixed batch: every row starts and finishes together, one shared
                  write pointer (KVCache), decode captured in a CUDA graph.
continuous_batch  requests join and leave every step: a scheduler over per-row
                  (RowKVCache) or paged (PagedKVCache) KV memory.
"""
