"""Paged engine: requests own fixed-size blocks from a shared pool (PagedKVCache).

decode_attention and prefill_attention (the Triton kernels) are not imported here: Triton
needs CUDA, so importing them would break this package on the Mac. Import them by module path.
"""
