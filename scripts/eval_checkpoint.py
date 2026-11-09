"""Evaluate a model's quality by log-likelihood -- any checkpoint, or OpenAI's GPT-2.

    python scripts/eval_checkpoint.py --checkpoint runs/keep/gpt2_124M_step013147.pt
    python scripts/eval_checkpoint.py --hf gpt2                       # the reference
    python scripts/eval_checkpoint.py --checkpoint X --hellaswag-limit 1000   # quick

Reports, on the same data for every model:
  val     NLL per token, perplexity, bits/token on the first --val-tokens tokens of
          the FineWeb-Edu val shard (a fixed window: no run trains on it), plus loss by
          position at a few context lengths
  hellaswag  acc (summed log-likelihood) and acc_norm (per-token), 10,042 examples

What the numbers mean is in docs/benchmarks/training.md.
"""

import argparse
import json
import os
import time

import torch

from utils.data import DEFAULT_DATA_ROOT, load_tokens
from utils.device import pick_device, synchronize
from evaluation import evaluate_hellaswag, val_nll
from model.gpt2.loading import load_model

POSITIONS = (0, 1, 3, 15, 63, 255, 1023)        # loss at these context positions


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint')
    src.add_argument('--hf', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    p.add_argument('--device', default='auto')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--val-tokens', type=int, default=2**20, help='fixed window size (default 1,048,576)')
    p.add_argument('--hellaswag-limit', type=int, default=None, help='first N examples (default all 10,042)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--out', default=None, help='JSON results file')
    args = p.parse_args()

    device = pick_device(args.device)
    # bf16 autocast where it is fast and native; fp32 elsewhere (MPS/CPU) keeps results exact
    amp = torch.bfloat16 if device.startswith('cuda') else None
    # fp32 weights (bf16 autocast on CUDA above); the file name carries the step
    model, enc, _ = load_model(args.hf or args.checkpoint, device, dtype=torch.float32)
    name = f'openai/{args.hf}' if args.hf else os.path.basename(args.checkpoint)
    print(f'model: {name}  device: {device}  autocast: {amp}')

    val_shard = next(os.path.join(args.data_root, f) for f in sorted(os.listdir(args.data_root)) if 'val' in f)
    tokens = load_tokens(val_shard)
    t0 = time.perf_counter()
    val = val_nll(model, tokens, T=model.config.block_size, batch_size=args.batch_size,
                  max_tokens=args.val_tokens, device=device, autocast_dtype=amp, per_position=True)
    synchronize(device)
    t_val = time.perf_counter() - t0
    position = {p: val['position_nll'][p] for p in POSITIONS if p < len(val['position_nll'])}
    print(f'val     nll {val["nll"]:.4f}  perplexity {val["perplexity"]:.2f}  '
          f'bits/token {val["bits_per_token"]:.3f}  ({val["tokens"]:,} tokens, {t_val:.0f}s)')
    print('        loss by position: ' + '  '.join(f'{p}:{v:.2f}' for p, v in position.items()))

    t0 = time.perf_counter()
    hs = evaluate_hellaswag(model, enc, limit=args.hellaswag_limit, device=device, autocast_dtype=amp)
    t_hs = time.perf_counter() - t0
    print(f'hellaswag  acc {hs["acc"]:.4f}  acc_norm {hs["acc_norm"]:.4f}  '
          f'({hs["n"]:,} examples, {t_hs:.0f}s)')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        val.pop('position_nll')
        with open(args.out, 'w') as f:
            json.dump(dict(model=name, device=device, val_shard=os.path.basename(val_shard),
                           val=val, loss_by_position=position, hellaswag=hs), f, indent=2)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
