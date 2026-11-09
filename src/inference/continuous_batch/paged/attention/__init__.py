"""The paged engine's attention kernels (Triton, CUDA only):

    decode_attention.py   decode: one query per row; writes the new token's k/v into the
                          pool, then reads the row's history through the block table
    prefill_attention.py  prefill: FlashAttention over the packed prompts (tiles of
                          queries x tiles of keys on tensor cores), each query bounded to
                          its own prompt

Neither is imported here: Triton needs CUDA, so importing them would break this package on
the Mac. PagedKVCache imports them by module path, on CUDA only.
"""
