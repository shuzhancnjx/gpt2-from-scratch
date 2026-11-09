"""CUDA graphs of the ModelRunner's steps -- one class per phase, both implementing
inference/cuda_graph.py BaseGraphs. The runner owns them and they record its own step
bodies (decode_eager, prefill_logits), so they live beside it, not inside one engine:

    prefill_graph.py  PrefillGraphs  one graph per padded token count; samples after replay
    decode_graph.py   DecodeGraphs   one graph per sampler setting; samples inside

Cache requirement: PrefillGraphs pads into the cache's pad_row (a block-table row that
is always scratch), which only PagedKVCache has -- so graphs run with the paged engine.

Not imported here: they need CUDA. Import them by module path.
"""
