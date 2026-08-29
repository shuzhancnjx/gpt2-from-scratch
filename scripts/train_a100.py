#!/usr/bin/env python
"""Fresh 124M pretraining run on a single A100 -- the GPT-2 reference recipe.

    python scripts/train_a100.py

This is NOT a continuation: it trains from random init, with the batch size and
warmup the published hyperparameters were actually tuned for. The Mac runs used
total_batch_size=16384 (32x smaller) purely because 16GB of unified memory forced
it, which is why their loss curves were so noisy and why max_lr=6e-4 was arguably
mismatched there.

Measured on the A100: ~116,000 tok/s, so 2.5B tokens is ~6 hours.

Before a long run, sweep B -- 40GB fits B=16 at ~32% and B=32 at ~59%; the
logits tensor (B x T x 50304, fp32 for cross_entropy) dominates, not the layers:

    for B in (8, 16, 32):
        train(TrainConfig(B=B, total_batch_size=524288, max_steps=6,
                          eval_every=10**9, sample_prompt='', run_dir=f'/tmp/bs_{B}'))
"""

from gpt2.training import TrainConfig, train

CONFIG = TrainConfig(
    # --- model: GPT-2 124M, unchanged ---
    n_layer=12, n_head=12, n_embd=768,

    # --- the batch the published LR was tuned for ---
    B=16,                        # micro-batch; 32 accumulation steps
    total_batch_size=524288,     # 2^19 = 512 seqs x 1024 tokens, the GPT-2 figure

    # --- data: Chinchilla-optimal for 124M is ~2.5B tokens ---
    first_shard=0,
    num_shards=250,              # 2.5B tokens -> 4768 steps -> ~6h

    # --- schedule ---
    max_lr=6e-4,
    warmup_steps=715,            # ~375M tokens, matching GPT-2. The default 100 was
                                 # sized for 16k-token steps and is far too short here.
    lr_schedule_steps=None,      # single run: defaults to this run's end
    max_steps=None,              # derived from the data window

    # --- fresh, not a continuation ---
    resume_from=None,

    eval_every=250,              # ~19 evals over the run
    run_dir='runs_a100',
)

if __name__ == '__main__':
    train(CONFIG)
