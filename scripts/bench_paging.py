"""P1.4 benchmark: paged vs row KV cache at the same KV memory.

    python scripts/bench_paging.py                          # 124M config, auto device
    python scripts/bench_paging.py --model tiny --device cpu
    python scripts/bench_paging.py --graph                  # CUDA: also the CUDA-graph decode

Both engines get the same KV budget: RowKVCache with `--row-slots` full-width rows,
and a block pool holding exactly as many tokens (+1 scratch block) shared by up to
`--paged-seqs` requests. Weights are random -- speed and memory do not depend on
them -- so the same prompts produce the same work in both engines.

Reports, per engine: wall time, generated tokens/s, most requests resident at once,
KV memory reserved, and (CUDA only) peak allocated memory -- which includes the
gather copy the paged path makes every step. Performance numbers belong to the
device that produced them; the device is printed and saved with every result.
"""

import argparse
import json
import os
import random
import time

import tiktoken
import torch

from gpt2.device import default_dtype, pick_device, synchronize
from gpt2.gpt import GPT, GPTConfig
from gpt2.inference.continuous_batch import PagedScheduler, Scheduler

MODELS = {
    '124m': dict(vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=1024),
    'tiny': dict(vocab_size=50304, n_layer=2, n_head=2, n_embd=64, block_size=256),
}


def make_prompts(enc, n, lo, hi, seed):
    rng = random.Random(seed)
    # random ids below 50000 avoid <|endoftext|> (50256) inside a prompt
    return [enc.decode([rng.randrange(50000) for _ in range(rng.randint(lo, hi))])
            for _ in range(n)]


def device_name(device):
    if device.startswith('cuda'):
        return torch.cuda.get_device_name()
    return {'mps': 'Apple GPU (MPS)', 'cpu': 'CPU'}.get(device, device)


def timed_run(sched, prompts, max_new, device):
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    t0 = time.perf_counter()
    sched.run(prompts, max_new_tokens=max_new)
    synchronize(device)
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
    generated = sum(len(g) for g in sched.generated_cpu.values())
    return seconds, generated, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--model', choices=MODELS, default='124m')
    p.add_argument('--width', type=int, default=None, help='max tokens per request (default: block_size)')
    p.add_argument('--row-slots', type=int, default=4)
    p.add_argument('--paged-seqs', type=int, default=16)
    p.add_argument('--block-size', type=int, default=16)
    p.add_argument('--num-prompts', type=int, default=32)
    p.add_argument('--prompt-min', type=int, default=16)
    p.add_argument('--prompt-max', type=int, default=128)
    p.add_argument('--max-new', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--graph', action='store_true',
                   help='CUDA only: also run the paged engine with its decode step in a CUDA graph')
    p.add_argument('--out', default='results/bench_paging.json')
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = default_dtype(device)
    cfg = GPTConfig(dropout=0.0, bias=True, **MODELS[args.model])
    W = args.width or cfg.block_size

    torch.manual_seed(args.seed)
    model = GPT(cfg).eval().to(device=device, dtype=dtype)
    enc = tiktoken.get_encoding('gpt2')
    prompts = make_prompts(enc, args.num_prompts, args.prompt_min, args.prompt_max, args.seed)

    bytes_per_token = 2 * cfg.n_layer * cfg.n_embd * torch.finfo(dtype).bits // 8     # K and V
    num_blocks = 1 + args.row_slots * W // args.block_size                           # same memory

    engines = {
        'row':   Scheduler(model, enc, cfg, batch_size=args.row_slots, max_tokens=W,
                           device=device, dtype=dtype),
        'paged': PagedScheduler(model, enc, cfg, max_seqs=args.paged_seqs,
                                block_size=args.block_size, num_blocks=num_blocks,
                                max_tokens=W, device=device, dtype=dtype),
    }
    reserved = {'row': args.row_slots * W * bytes_per_token,
                'paged': num_blocks * args.block_size * bytes_per_token}
    if args.graph:
        if not device.startswith('cuda'):
            raise SystemExit('--graph needs a CUDA device')
        engines['paged+graph'] = PagedScheduler(model, enc, cfg, max_seqs=args.paged_seqs,
                                                block_size=args.block_size, num_blocks=num_blocks,
                                                max_tokens=W, device=device, dtype=dtype)
        engines['paged+graph'].capture_graph()
        reserved['paged+graph'] = reserved['paged']

    print(f'device: {device} ({device_name(device)})  dtype: {dtype}  model: {args.model}  '
          f'W={W}  prompts={args.num_prompts} ({args.prompt_min}-{args.prompt_max} tok)  '
          f'max_new={args.max_new}')

    results = {}
    for name, sched in engines.items():
        sched.run(prompts[:2], max_new_tokens=2)                      # warm-up
        seconds, generated, peak = timed_run(sched, prompts, args.max_new, device)
        paged = name.startswith('paged')
        resident = sched.peak_active if paged else args.row_slots
        results[name] = dict(seconds=seconds, generated_tokens=generated,
                             decode_steps=sched.decode_steps if paged else None,
                             tokens_per_s=generated / seconds, max_resident=resident,
                             kv_reserved_mb=reserved[name] / 2**20,
                             peak_allocated_mb=None if peak is None else peak / 2**20)
        r = results[name]
        print(f'{name:11s} {r["seconds"]:7.2f} s  {r["tokens_per_s"]:8.1f} tok/s  '
              f'resident {r["max_resident"]:3d}  KV reserved {r["kv_reserved_mb"]:7.1f} MB  '
              f'peak {"n/a" if peak is None else f"{peak / 2**20:.1f} MB"}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(dict(device=device, device_name=device_name(device), dtype=str(dtype),
                       args=vars(args), width=W, num_blocks=num_blocks, results=results), f, indent=2)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
