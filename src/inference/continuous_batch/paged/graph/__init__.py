"""CUDA graphs for the paged engine -- one class per phase, both implementing
inference/cuda_graph.py BaseGraphs:

    prefill_graph.py  PrefillGraphs  one graph per padded token count; samples after replay
    decode_graph.py   DecodeGraphs   one graph per sampler setting; samples inside

Not imported here: they need CUDA. Import them by module path.
"""
