"""Where one engine step's time goes: decode and prefill, kernel by kernel.

    python scripts/profile_step.py                         # 124M, CUDA, 16 decode rows
    python scripts/profile_step.py --max-seqs 64           # a wider decode batch
    python scripts/profile_step.py --trace-dir results/traces   # + Chrome traces (Perfetto)

bench_admission / bench_paging time whole steps; this one opens them up. Three parts:

  1. decode, eager    every kernel of one decode step (no graph): how many, how long, by
                      category -- matmul, attention, layernorm, indexing, copies, ...
  2. decode, graph    GPU time of one graph replay (CUDA events) vs the wall time of a
                      full engine step, split into its CPU parts (schedule, apply, launch,
                      update) -- whatever is left is the CPU waiting on the GPU
  3. prefill          one packed prefill of --prefill-tokens tokens: kernels by category
                      (eager), then graph replay time and the bucket padding it pays

On CUDA the engine runs the model under torch.compile (its graphs record the compiled
model), so the kernel tables show Inductor's fused kernels
(triton_per_fused_add_native_layer_norm_...), and the end reports Dynamo's graph breaks.

The engine is filled the normal way (prompts admitted until every decode row is busy)
and nothing finishes while it is measured, so every decode step is a full-batch step.
CUDA for parts 2 and 3's graphs and the Triton kernel; on a CPU it runs the eager parts
only (a smoke test: op counts, not kernels).
"""

import argparse
import json
import os
import random
import statistics
import time
from collections import defaultdict

import tiktoken
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from inference.base_kv_cache import PrefillBatch
from inference.continuous_batch import PagedEngine
from inference.sampling import SamplingParams
from model.gpt2 import GPT, GPTConfig
from utils.device import default_dtype, pick_device, synchronize
from _common import MODELS


# substring -> category, first match wins (order matters: a GELU or an index kernel is
# also an "elementwise" kernel by name)
CATEGORIES = [
    ('attention', ('decode_attention', 'prefill_attention', 'flash', 'fmha', 'attention')),
    ('matmul', ('gemm', 'gemv', 'cutlass', 'cublas', 'splitk', 'sm80_', 'sm90_', 'ampere_',
                'aten::addmm', 'aten::mm', 'aten::bmm', 'aten::linear', 'aten::matmul')),
    ('layernorm', ('layer_norm', 'layernorm')),
    ('gelu', ('gelu',)),
    ('memcpy/memset', ('memcpy', 'memset')),
    ('indexing', ('index', 'gather', 'scatter', 'embedding')),
    ('sampling/reduce', ('argmax', 'argmin', 'reduce', 'topk', 'softmax', 'sort')),
    ('copy', ('copy', 'contiguous', 'clone', 'reshape')),
    ('elementwise', ('elementwise', 'vectorized', 'aten::add', 'aten::mul', 'aten::div',
                     'aten::remainder', 'aten::arange', 'aten::lt', 'aten::le', 'aten::and')),
]


def category(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return 'other'


def kernels(prof, cuda):
    """(name, microseconds) of every GPU kernel / copy (CUDA), or of every top-level op
    (CPU smoke test)."""
    if cuda:
        return [(e.name, e.time_range.elapsed_us()) for e in prof.events()
                if e.device_type == DeviceType.CUDA]
    return [(e.name, e.time_range.elapsed_us()) for e in prof.events()
            if e.device_type == DeviceType.CPU and e.cpu_parent is None
            and e.name.startswith('aten::')]


def table(events, reps, title, top=12):
    """Print per-rep counts and times by category and the top kernels. Returns a summary."""
    by_cat, by_name = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    for name, us in events:
        for d, key in ((by_cat, category(name)), (by_name, name)):
            d[key][0] += 1
            d[key][1] += us
    count, us = len(events) / reps, sum(u for _, u in events) / reps
    print(f'\n{title}: {count:.0f} kernels, {us / 1e3:.3f} ms of kernel time per call')
    print(f'  {"category":18s} {"kernels":>8s} {"ms":>8s} {"share":>6s} {"us/kernel":>10s}')
    for cat, (n, t) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        print(f'  {cat:18s} {n / reps:8.1f} {t / reps / 1e3:8.3f} {t / reps / us:6.0%} '
              f'{t / n:10.1f}')
    print(f'  top {top} by time:')
    for name, (n, t) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f'    {t / reps / 1e3:7.3f} ms  {n / reps:5.1f}x  {name[:90]}')
    return dict(kernels_per_call=count, kernel_ms_per_call=us / 1e3,
                by_category={c: dict(kernels=n / reps, ms=t / reps / 1e3) for c, (n, t) in by_cat.items()})


def profiled(fn, reps, cuda, trace=None):
    """Run fn reps times under the profiler (after one untraced call); return its kernels."""
    fn()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts) as prof:
        for _ in range(reps):
            fn()
        synchronize('cuda' if cuda else 'cpu')
    if trace:
        prof.export_chrome_trace(trace)
        print(f'  trace: {trace}')
    return kernels(prof, cuda)


def wall_ms(fn, reps, device):
    """Wall time per call, with a sync after every call (what a step that reads its
    result pays)."""
    fn()
    synchronize(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def gpu_ms(fn, reps):
    """GPU time per call: CUDA events around reps back-to-back calls. Valid when the CPU
    launches faster than the GPU runs (true for a graph replay)."""
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


class Timers:
    """Wraps methods to add up their CPU wall time (no sync: what the CPU spends)."""

    def __init__(self):
        self.ms, self.calls = defaultdict(float), defaultdict(int)

    def wrap(self, obj, attr, label):
        fn = getattr(obj, attr)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.ms[label] += (time.perf_counter() - t0) * 1e3
            self.calls[label] += 1
            return out
        setattr(obj, attr, timed)
        return fn


def fill(engine, prompt_ids, sp):
    """Start a run and step until every decode row holds a request (and nothing waits)."""
    engine.runner.start(sp)
    engine.blocks.reset()
    engine.scheduler.reset()
    engine.scheduler.add(prompt_ids)
    steps = 0
    while engine.scheduler.num_decoding < engine.runner.B:
        engine.step(sp)
        steps += 1
        assert steps < 1000, 'decode rows never filled'
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--model', choices=MODELS, default='124m')
    p.add_argument('--max-seqs', type=int, default=16, help='decode rows')
    p.add_argument('--context', type=int, default=160, help='prompt tokens per decode row')
    p.add_argument('--prefill-tokens', type=int, default=1024)
    p.add_argument('--reps', type=int, default=20, help='calls per measurement')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--trace-dir', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    cuda = device.startswith('cuda')
    dtype = default_dtype(device)
    cfg = GPTConfig(dropout=0.0, bias=True, **MODELS[args.model])
    torch.manual_seed(args.seed)
    model = GPT(cfg).eval().to(device=device, dtype=dtype)
    enc = tiktoken.get_encoding('gpt2')
    rng = random.Random(args.seed)
    B, W = args.max_seqs, cfg.block_size
    trace = (lambda name: os.path.join(args.trace_dir, f'step_{name}_{device.split(":")[0]}.json')) \
        if args.trace_dir else (lambda name: None)
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    engine = PagedEngine(model, enc, cfg, max_seqs=B, block_size=16, device=device,
                         dtype=dtype, cuda_graph=cuda)
    runner, sched = engine.runner, engine.scheduler
    sp = SamplingParams(max_new_tokens=W - args.context - 16)      # nothing finishes
    prompt_ids = [[rng.randrange(50000) for _ in range(args.context)] for _ in range(B)]
    print(f'device {device}  model {args.model}  dtype {dtype}  decode rows {B}  '
          f'context ~{args.context}  graphs {runner.decode_graphs is not None}  '
          f'compiled {runner.graph_model is not model}')
    t_start = time.perf_counter()
    out = dict(args=vars(args), device=device)

    with torch.inference_mode():
        # -- 1. decode, eager -------------------------------------------------------
        fill(engine, prompt_ids, sp)
        events = profiled(lambda: runner.decode_eager(sp), args.reps, cuda, trace('decode_eager'))
        out['decode_eager'] = table(events, args.reps, f'[1] decode step, eager ({B} rows)')
        out['decode_eager']['wall_ms'] = wall_ms(lambda: runner.decode_eager(sp), args.reps, device)
        print(f'  wall per eager step (synced): {out["decode_eager"]["wall_ms"]:.3f} ms  '
              f'-> {out["decode_eager"]["wall_ms"] - out["decode_eager"]["kernel_ms_per_call"]:.3f} ms '
              f'not covered by kernels (launch overhead, gaps)')

        # -- 2. decode, graph vs the full engine step ---------------------------------
        if cuda:
            fill(engine, prompt_ids, sp)
            replay = gpu_ms(lambda: runner.decode(sp), args.reps)
            synced = wall_ms(lambda: runner.decode(sp), args.reps, device)

            timers = Timers()
            originals = [(obj, attr, timers.wrap(obj, attr, label)) for obj, attr, label in
                         ((sched, 'schedule', 'schedule'), (runner, 'apply', 'apply'),
                          (runner, 'decode', 'decode launch'), (sched, 'update', 'update'),
                          (runner, 'prefill', 'prefill'))]
            steps = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                engine.step(sp)
                steps.append((time.perf_counter() - t0) * 1e3)
            for obj, attr, fn in originals:
                setattr(obj, attr, fn)
            assert timers.calls['prefill'] == 0, 'a decode-only step ran a prefill'
            step = statistics.median(steps)
            parts = {k: v / args.reps for k, v in timers.ms.items()}
            cpu = sum(v for k, v in parts.items() if k != 'decode launch')
            print(f'\n[2] decode step, graph ({B} rows)')
            print(f'  GPU time per replay (events) {replay:7.3f} ms')
            print(f'  replay + sync (wall)         {synced:7.3f} ms')
            print(f'  full engine step (wall)      {step:7.3f} ms   = what bench_admission calls a step')
            for k in ('schedule', 'apply', 'decode launch', 'update'):
                print(f'    {k:15s} {parts.get(k, 0.0):7.3f} ms  CPU')
            print(f'    rest            {step - sum(parts.values()):7.3f} ms  mostly .tolist() '
                  f'waiting for the GPU')
            print(f'  CPU work outside the replay: {cpu:.3f} ms of {step:.3f} '
                  f'({cpu / step:.0%}) -- what overlap scheduling could hide')
            out['decode_graph'] = dict(gpu_ms=replay, replay_synced_ms=synced, step_ms=step,
                                       cpu_parts_ms=parts)

        # -- 3. prefill ---------------------------------------------------------------
        runner.reset()
        engine.blocks.reset()
        T, lens = args.prefill_tokens, []
        while sum(lens) < T and len(lens) < runner.max_prefill_prompts:
            lens.append(min(rng.randint(16, 256), T - sum(lens)))
        prompts = [[rng.randrange(50000) for _ in range(n)] for n in lens]
        rows = [engine.blocks.reserve(engine.blocks.blocks_for(n))[0] for n in lens]
        batch = PrefillBatch.build(rows, lens, device)
        idx = torch.tensor([t for q in prompts for t in q], dtype=torch.long, device=device)[None]
        title = f'[3] prefill, eager ({len(lens)} prompts, {sum(lens)} tokens)'
        events = profiled(lambda: runner.prefill_logits(idx, batch), args.reps, cuda, trace('prefill_eager'))
        out['prefill_eager'] = table(events, args.reps, title)
        out['prefill_eager']['wall_ms'] = wall_ms(lambda: runner.prefill_logits(idx, batch), args.reps, device)
        print(f'  wall per eager prefill (synced): {out["prefill_eager"]["wall_ms"]:.3f} ms')

        if cuda:
            graphs = runner.prefill_graphs
            bucket = graphs.key(sum(lens))
            if bucket is None:
                print(f'  graph: {sum(lens)} tokens exceed the largest bucket -- eager only')
            else:
                full = gpu_ms(lambda: graphs.run(rows, prompts, sp), args.reps)
                graph = graphs.graphs.get(bucket)
                only = gpu_ms(graph.replay, args.reps)
                print(f'  graph, bucket {bucket} ({1 - sum(lens) / bucket:.0%} padding): '
                      f'replay {only:.3f} ms, with buffer fill + sampling {full:.3f} ms')
                out['prefill_graph'] = dict(bucket=bucket, real_tokens=sum(lens),
                                            replay_ms=only, run_ms=full)

    if runner.graph_model is not model:
        from torch._dynamo.utils import counters
        breaks = dict(counters['graph_break'])
        print(f'\ncompile: {sum(breaks.values())} graph breaks'
              + ''.join(f'\n  {n}x  {reason[:110]}' for reason, n in breaks.items()))
        print(f'total run time incl. compilation: {time.perf_counter() - t_start:.0f} s')
        out['graph_breaks'] = breaks

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
