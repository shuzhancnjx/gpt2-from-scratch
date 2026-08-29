#!/usr/bin/env python
"""Entry point for a pretraining run.

    python scripts/train.py                       # single device
    torchrun --standalone --nproc_per_node=8 scripts/train.py

Edit CONFIG below, or import TrainConfig and drive it yourself.
"""

from gpt2.training import TrainConfig, train

# --- run 7: continue the model on the next window of shards ------------------------
#   run 1  shards  0..2    steps     0..1830   30M tokens   val 5.4248
#   run 2  shard   3       steps  1831..2440   10M tokens   val 5.3196
#   run 3  shards  4..5    steps  2441..3660   20M tokens   val 5.1134
#   run 4  shards  6..7    steps  3661..4880   20M tokens   val 4.8537
#   run 5  shards  8..11   steps  4881..7321   40M tokens   val 4.5305
#   run 6  shards 12..15   steps  7322..9762   40M tokens   val 4.3515
#   run 7  shards 16..21   steps 9763..13147   ~55M tokens  <- this one
# Six-shard window (3662 steps) capped at 8 hours = 3385 steps at 8.51 s/step
# end-to-end. Cumulative after this run: ~215M tokens (~1.7 tokens/param).
CONFIG = TrainConfig(
    first_shard=16,
    num_shards=6,

    resume_from='latest',        # newest checkpoint in runs/ (model_009762.pt)

    # The cosine spans the FULL 6-shard window (9763 + 3662), not the capped end, so
    # the next run picks the schedule up mid-decay instead of warm-restarting.
    lr_schedule_steps=13425,

    # 8-hour wall-clock cap. Stops ~277 steps short of finishing shard 21; the shard
    # position is checkpointed, so the next run resumes exactly there.
    max_steps=13148,
)

if __name__ == '__main__':
    train(CONFIG)
