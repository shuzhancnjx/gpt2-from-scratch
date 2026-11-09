"""Profile the row and paged schedulers -- tooling for P1.2 and the paging open question.

    python scripts/profile_serving.py                       # 124M, auto device
    python scripts/profile_serving.py --device cuda --trace-dir results/traces

docs/benchmarks/paging.md asks why 8x more concurrency gave no throughput gain, and
lists three hypotheses. This script measures what separates them; explaining the
numbers is the analysis left for you.

Part A -- time breakdown. Wraps the scheduler's three operations with synchronised
timers and reports, per engine and per max_seqs:
    prefill share of wall time                        (hypothesis 3)
    decode steps, and milliseconds per decode step    (hypotheses 1 and 2:
                                                       does per-step cost grow
                                                       with max_seqs × W?)
The timers synchronise the device, which is exactly what a production engine avoids;
that is fine here because the goal is attribution, not peak speed.

Part B -- torch.profiler on one paged run: the top operators by device time, and a
Chrome trace (open in https://ui.perfetto.dev) showing the decode-step timeline.
"""

import argparse
import os
import time

import tiktoken
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from utils.device import default_dtype, pick_device, synchronize
from model.gpt2.gpt import GPT, GPTConfig
from inference.continuous_batch import PagedEngine, RowEngine
from _common import MODELS, random_prompts


def instrument(sched, device):
    """Wrap each phase with synchronised timers; return the totals dict.

    PagedEngine: admit = runner.prefill, decode = runner.decode, retire = scheduler.update.
    RowEngine: its _admit, runner.decode, its _retire."""
    t = dict(admit=0.0, decode=0.0, retire=0.0, decode_steps=0)

    def wrap(name, fn):
        def timed(*args, **kwargs):
            synchronize(device)
            t0 = time.perf_counter()
            with record_function(f'sched.{name}'):       # labels the region in traces
                out = fn(*args, **kwargs)
            synchronize(device)
            t[name] += time.perf_counter() - t0
            if name == 'decode':
                t['decode_steps'] += 1
            return out
        return timed

    if hasattr(sched, 'scheduler'):
        hooks = [(sched.runner, 'prefill', 'admit'), (sched.runner, 'decode', 'decode'),
                 (sched.scheduler, 'update', 'retire')]
    else:
        hooks = [(sched, '_admit', 'admit'), (sched.runner, 'decode', 'decode'),
                 (sched, '_retire', 'retire')]
    for obj, attr, name in hooks:
        setattr(obj, attr, wrap(name, getattr(obj, attr)))
    return t


def breakdown(sched, prompts, max_new, device):
    t = instrument(sched, device)
    synchronize(device)
    t0 = time.perf_counter()
    sched.run(prompts, max_new_tokens=max_new)
    synchronize(device)
    wall = time.perf_counter() - t0
    generated = sum(len(g) for g in sched.generated.values())
    return dict(wall=wall, tok_s=generated / wall, prefill_share=t['admit'] / wall,
                decode_steps=t['decode_steps'],
                ms_per_step=1e3 * t['decode'] / max(t['decode_steps'], 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--model', choices=MODELS, default='124m')
    p.add_argument('--row-slots', type=int, default=4)
    p.add_argument('--max-seqs', default='4,8,16,32', help='paged concurrency caps to sweep')
    p.add_argument('--block-size', type=int, default=16)
    p.add_argument('--num-prompts', type=int, default=32)
    p.add_argument('--prompt-min', type=int, default=16)
    p.add_argument('--prompt-max', type=int, default=128)
    p.add_argument('--max-new', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--trace-dir', default='results/traces')
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = default_dtype(device)
    cfg = GPTConfig(dropout=0.0, bias=True, **MODELS[args.model])
    W = cfg.block_size
    torch.manual_seed(args.seed)
    model = GPT(cfg).eval().to(device=device, dtype=dtype)
    enc = tiktoken.get_encoding('gpt2')
    prompts = random_prompts(enc, args.num_prompts, args.prompt_min, args.prompt_max, args.seed)
    num_blocks = 1 + args.row_slots * W // args.block_size           # same KV memory as the row cache

    name = torch.cuda.get_device_name() if device.startswith('cuda') else device
    print(f'device: {device} ({name})  dtype: {dtype}  model: {args.model}  W={W}  '
          f'prompts={args.num_prompts}  max_new={args.max_new}\n')

    # -- Part A: where does the time go? -----------------------------------------
    print('Part A — time breakdown (synchronised timers)')
    print(f'{"engine":18s} {"wall s":>7s} {"tok/s":>8s} {"prefill %":>10s} '
          f'{"decode steps":>13s} {"ms/step":>8s}')
    runs = [('row', args.row_slots,
             lambda: RowEngine(model, enc, cfg, batch_size=args.row_slots, max_tokens=W,
                               device=device, dtype=dtype))]
    for m in (int(x) for x in args.max_seqs.split(',')):
        runs.append((f'paged', m, lambda m=m: PagedEngine(
            model, enc, cfg, max_seqs=m, block_size=args.block_size, num_blocks=num_blocks,
            max_tokens=W, device=device, dtype=dtype)))
    for label, slots, build in runs:
        sched = build()
        sched.run(prompts[:2], max_new_tokens=2)                      # warm-up
        r = breakdown(sched, prompts, args.max_new, device)
        print(f'{label + f" ({slots} seqs)":18s} {r["wall"]:7.2f} {r["tok_s"]:8.1f} '
              f'{100 * r["prefill_share"]:9.1f}% {r["decode_steps"]:13d} {r["ms_per_step"]:8.2f}')

    # -- Part B: which kernels dominate? ------------------------------------------
    m = int(args.max_seqs.split(',')[-1])
    sched = PagedEngine(model, enc, cfg, max_seqs=m, block_size=args.block_size,
                           num_blocks=num_blocks, max_tokens=W, device=device, dtype=dtype)
    sched.run(prompts[:2], max_new_tokens=2)
    instrument(sched, device)                                         # labels admit/decode in the trace
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith('cuda') else [])
    with profile(activities=activities, record_shapes=True) as prof:
        sched.run(prompts[:min(len(prompts), m)], max_new_tokens=8)

    sort_key = 'cuda_time_total' if device.startswith('cuda') else 'cpu_time_total'
    print(f'\nPart B — top operators, paged ({m} seqs), sorted by {sort_key}')
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    os.makedirs(args.trace_dir, exist_ok=True)
    trace = os.path.join(args.trace_dir, f'paged_{device}_{m}seqs.json')
    prof.export_chrome_trace(trace)
    print(f'trace: {trace}  (open in https://ui.perfetto.dev)')


if __name__ == '__main__':
    main()
