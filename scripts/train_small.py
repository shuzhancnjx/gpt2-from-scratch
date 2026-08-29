#!/usr/bin/env python
"""Chinchilla-sized companion to the 124M run -- continuation.

Phase 1 (done): shards 0..21, 13,148 steps, 215M tokens -> val 4.2659 in 9.6h.
                The 124M model needed 31h for 4.1741 on the same data; at equal
                COMPUTE (9.6h) it was only at 5.0025.
Phase 2 (this): shards 22..36, +150M tokens.

  10.6M non-embd params, 365M tokens -> 34 tokens/param (past Chinchilla's 20).
"""

from gpt2.training import TrainConfig, train

CONFIG = TrainConfig(
    n_layer=6, n_head=6, n_embd=384,
    B=2,                         # measured optimal; B=4 ties, B=8 collapses

    first_shard=22,
    num_shards=15,               # 150M tokens = 9155 steps ~= 6.7h at 2.62 s/step

    resume_from='latest',        # runs_small/model_013147.pt
    run_dir='runs_small',

    # Phase 1's cosine already bottomed out at min_lr, so extending necessarily
    # warm-restarts: LR climbs back to ~2.6e-4 before decaying again. Unavoidable
    # when a completed schedule is extended, and usually helpful off a plateau.
    lr_schedule_steps=22303,
)

if __name__ == '__main__':
    train(CONFIG)
