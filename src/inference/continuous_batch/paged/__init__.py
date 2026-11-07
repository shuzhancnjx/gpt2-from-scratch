"""Paged engine: requests own fixed-size blocks from a shared pool (PagedKVCache).

paged_attention (the Triton decode kernel) is not imported here: Triton needs CUDA,
so importing it would break this package on the Mac. Import it by its module path.
"""
