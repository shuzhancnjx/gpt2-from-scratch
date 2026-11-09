"""What the benchmark and profiling scripts share (not a script itself).

    MODELS           the model configs the scripts can build: GPT-2 124M and a tiny one
    random_prompts   reproducible random prompts (ids below 50000: never <|endoftext|>)
    graph_us         GPU time of a callable, captured in a CUDA graph and replayed

Imported as `from _common import ...`: `python scripts/x.py` puts scripts/ on the path.
Nothing from src/ is imported at module level, so bench_admission.py can still measure
older checkouts (their src/ first on PYTHONPATH).
"""

import random

MODELS = {
    '124m': dict(vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=1024),
    'tiny': dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=64, block_size=256),
}


def random_prompts(enc, n, lo, hi, seed):
    """n prompts of lo..hi random tokens, decoded to text. The same seed gives the same
    prompts as every script has always used (length first, then the ids)."""
    rng = random.Random(seed)
    return [enc.decode([rng.randrange(50000) for _ in range(rng.randint(lo, hi))])
            for _ in range(n)]


def graph_us(fn, reps=200):
    """GPU microseconds per call of fn: captured once in a CUDA graph (CapturedGraph, the
    engine's own recipe), then timed over `reps` replays -- no launch overhead."""
    import torch
    from inference.cuda_graph import CapturedGraph

    graph = CapturedGraph(fn, warmup=3)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3
