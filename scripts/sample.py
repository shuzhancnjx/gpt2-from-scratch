#!/usr/bin/env python
"""Generate from a trained checkpoint through the KV-cached inference path.

    python scripts/sample.py "The capital of France is"
    python scripts/sample.py -n 60 --temperature 0.8 "first prompt" "second prompt"
    python scripts/sample.py --check "The capital of France is"

--check is the one that matters. It generates greedily twice -- once through the
cache, once by recomputing the whole prefix every step -- and compares. The two
must produce identical tokens; if they don't, the cache is wrong. It also times
both, which is the point of the cache in the first place.
"""

import argparse
import time
from pathlib import Path

import torch

from gpt2.inference.inference import GptInference

# Anchored to the repo, not the working directory, so the script runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / 'runs/keep/gpt2_124M_step013147.pt'


def greedy_uncached(inf, prompt, max_new_tokens):
    """The slow path: no cache, recompute the entire prefix at every step."""
    ids = inf.enc.encode(prompt)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx = torch.tensor([ids], dtype=torch.long, device=inf.device)
            logits, _ = inf.model(idx[:, -inf.config.block_size:])
            nxt = int(logits[0, -1, :inf.enc.n_vocab].argmax())
            ids.append(nxt)
            if nxt == inf.enc.eot_token:
                break
    return ids


def check(inf, prompts, max_new_tokens):
    ok = True
    for prompt in prompts:
        t0 = time.perf_counter()
        cached = inf.sample(prompt, max_new_tokens=max_new_tokens, temperature=0)
        t_cached = time.perf_counter() - t0

        t0 = time.perf_counter()
        reference = inf.enc.decode(greedy_uncached(inf, prompt, max_new_tokens))
        t_uncached = time.perf_counter() - t0

        match = cached == reference
        ok &= match
        print(f'{"MATCH" if match else "MISMATCH"}  {prompt!r}')
        print(f'    cached   {t_cached:6.2f}s')
        print(f'    uncached {t_uncached:6.2f}s   ({t_uncached / t_cached:.1f}x slower)')
        if not match:
            print(f'    cached:   {cached!r}')
            print(f'    expected: {reference!r}')
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prompts', nargs='*', default=['The capital of France is'])
    p.add_argument('--checkpoint', default=DEFAULT_CKPT)
    p.add_argument('--device', default='cpu')
    p.add_argument('-n', '--max-new-tokens', type=int, default=60)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--check', action='store_true',
                   help='verify the cached path against the uncached one (greedy)')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f'no checkpoint at {ckpt}\n'
                         f'pass one with --checkpoint, or train first with scripts/train.py')

    torch.manual_seed(args.seed)
    inf = GptInference(str(ckpt), device=args.device)

    if args.check:
        raise SystemExit(0 if check(inf, args.prompts, args.max_new_tokens) else 1)

    # one batched call -- prompts of different lengths exercise the padding path
    outputs = inf.sample(args.prompts, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    for prompt, out in zip(args.prompts, outputs):
        print(f'--- {prompt!r}')
        print(out)
        print()


if __name__ == '__main__':
    main()
