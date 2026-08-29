#!/usr/bin/env python
"""Entry point for a pretraining run (the 124M model).

    python scripts/train.py                       # single device
    torchrun --standalone --nproc_per_node=8 scripts/train.py
"""

from gpt2.training import TrainConfig, train

# --- run 8: continue the 124M model on the next window -----------------------------
#   runs 1-7  shards  0..21   steps     0..13147   215M tokens   val 4.1741   ~31h
#   run 8     shards 22..27   steps 13148..16531   ~55M tokens   <- this one
# Six-shard window (3662 steps) capped at 8h = 3384 steps at 8.51 s/step end-to-end.
# Cumulative after this run: ~271M tokens (~3.2 tokens/param non-embedding).
#
# For reference, the 30M model reached val 4.11 on 365M tokens in 16h total, and its
# fitted curve asymptotes near 3.86 -- this model showed no such floor, which is what
# makes continuing it worthwhile.
CONFIG = TrainConfig(
    first_shard=22,
    num_shards=6,

    resume_from='latest',        # runs/model_013147.pt

    # Cumulative across every run: 13148 done + 3662 in this window. Spans the FULL
    # window, not the capped end, so the next run continues the decay.
    lr_schedule_steps=16810,

    # 8-hour wall-clock cap; the shard position is checkpointed so run 9 resumes
    # mid-window with nothing skipped.
    max_steps=16532,
)

if __name__ == '__main__':
    train(CONFIG)
