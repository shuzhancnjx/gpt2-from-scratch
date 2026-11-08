"""Admission latency: time to first token and time between tokens under a queue.

    python scripts/bench_admission.py                       # 124M, auto device
    python scripts/bench_admission.py --budgets 0 256 512   # also cap prefill tokens per step

Throughput benchmarks (bench_paging.py) see total work; this one sees WAITING. With more
requests than decode rows, requests queue, and two latencies matter:

    TTFT          run start -> a request's first token (its prefill finished)
    step time     wall time of one engine step = the gap between two tokens of a running
                  request -- prefill done in that step stalls every running request

Nothing in the engine is changed: step times are taken at _record_tokens, which already
syncs once per step (its .tolist()), and TTFT when a request's first token is recorded.
The same script measures older code too (put that code's src/ first on PYTHONPATH);
options the older scheduler lacks are skipped.
"""

import argparse
import inspect
import json
import os
import random
import statistics
import time

import tiktoken
import torch

from inference.continuous_batch import PagedScheduler
from model.gpt2 import GPT, GPTConfig
from utils.device import default_dtype, pick_device, synchronize

MODELS = {
    '124m': dict(vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=1024),
    'tiny': dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=64, block_size=256),
}


class FirstTokenClock(dict):
    """generated_cpu replacement: notes when each request's first token appears."""

    def __init__(self, t0):
        super().__init__()
        self.t0, self.ttft = t0, {}

    def __setitem__(self, req, tokens):
        if req not in self:
            self.ttft[req] = time.perf_counter() - self.t0
        super().__setitem__(req, tokens)


def instrument(sched):
    """Wrap the scheduler's own hooks; returns the dict the timings land in."""
    rec = dict(steps=[], ttft={}, prefills=[])
    reset, record = sched._reset, sched._record_tokens

    def timed_reset():
        reset()
        rec['t0'] = rec['last'] = time.perf_counter()
        sched.generated_cpu = rec['clock'] = FirstTokenClock(rec['t0'])

    def timed_record():
        record()                                         # its .tolist() is the step's sync
        now = time.perf_counter()
        rec['steps'].append(now - rec['last'])
        rec['last'] = now

    sched._reset, sched._record_tokens = timed_reset, timed_record
    if hasattr(sched, '_prefill'):
        prefill = sched._prefill

        def counted_prefill(rows, prompts, sampler):
            rec['prefills'].append(sum(len(p) for p in prompts))
            return prefill(rows, prompts, sampler)
        sched._prefill = counted_prefill
    return rec


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--model', choices=MODELS, default='124m')
    p.add_argument('--num-prompts', type=int, default=128)
    p.add_argument('--max-seqs', type=int, default=16, help='decode rows')
    p.add_argument('--prompt-min', type=int, default=16)
    p.add_argument('--prompt-max', type=int, default=256)
    p.add_argument('--max-new', type=int, default=64)
    p.add_argument('--budgets', type=int, nargs='*', default=[0],
                   help='max_prefill_tokens values to try (0 = no cap)')
    p.add_argument('--graph', action='store_true', help='CUDA: capture the decode step')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--label', default='run')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = default_dtype(device)
    cfg = GPTConfig(dropout=0.0, bias=True, **MODELS[args.model])
    torch.manual_seed(args.seed)
    model = GPT(cfg).eval().to(device=device, dtype=dtype)
    enc = tiktoken.get_encoding('gpt2')
    rng = random.Random(args.seed)
    prompts = [enc.decode([rng.randrange(50000) for _ in range(rng.randint(args.prompt_min, args.prompt_max))])
               for _ in range(args.num_prompts)]

    accepts = inspect.signature(PagedScheduler.__init__).parameters
    print(f'[{args.label}] device {device}  {args.num_prompts} prompts '
          f'({args.prompt_min}-{args.prompt_max} tok), {args.max_seqs} decode rows, '
          f'max_new {args.max_new}, graph {args.graph}')

    results = {}
    for budget in args.budgets:
        kwargs = dict(max_seqs=args.max_seqs, block_size=16, device=device, dtype=dtype)
        if 'max_prefill_tokens' in accepts:
            kwargs['max_prefill_tokens'] = budget or None     # 0 = explicitly no cap
        elif budget:
            print(f'  budget {budget}: this scheduler has no max_prefill_tokens -- skipped')
            continue
        if args.graph and not hasattr(PagedScheduler, 'capture_graph'):
            kwargs['cuda_graph'] = True                  # graphs captured per sampler setting
        sched = PagedScheduler(model, enc, cfg, **kwargs)
        if args.graph and hasattr(sched, 'capture_graph'):
            sched.capture_graph()                        # older code: one explicit capture
        sched.run(prompts[:2], max_new_tokens=2)                          # warm-up
        rec = instrument(sched)
        synchronize(device)
        sched.run(prompts, max_new_tokens=args.max_new)
        synchronize(device)
        seconds = time.perf_counter() - rec['t0']
        tokens = sum(len(g) for g in sched.generated_cpu.values())
        steps, ttft = [s * 1e3 for s in rec['steps']], [t * 1e3 for t in rec['clock'].ttft.values()]
        r = dict(tokens_per_s=tokens / seconds, seconds=seconds,
                 ttft_ms=dict(p50=pct(ttft, .5), p95=pct(ttft, .95), max=max(ttft)),
                 step_ms=dict(p50=pct(steps, .5), p99=pct(steps, .99), max=max(steps),
                              mean=statistics.mean(steps)),
                 prefill_forwards=len(rec['prefills']), decode_steps=len(steps))
        results[budget or 'none'] = r
        print(f"  budget {budget or 'none':>5}: {r['tokens_per_s']:7.0f} tok/s | "
              f"TTFT p50 {r['ttft_ms']['p50']:7.1f} p95 {r['ttft_ms']['p95']:7.1f} "
              f"max {r['ttft_ms']['max']:7.1f} ms | step p50 {r['step_ms']['p50']:5.2f} "
              f"p99 {r['step_ms']['p99']:6.2f} max {r['step_ms']['max']:6.2f} ms | "
              f"{r['prefill_forwards']} prefills")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(dict(label=args.label, device=device, args=vars(args), results=results), f, indent=2)


if __name__ == '__main__':
    main()
