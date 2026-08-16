#!/usr/bin/env python
"""Entry point for a pretraining run.

    python scripts/train.py                       # single device
    torchrun --standalone --nproc_per_node=8 scripts/train.py

Edit CONFIG below, or import TrainConfig and drive it yourself.
"""

from gpt2.training import TrainConfig, train

# --- run 2: continue the model from run 1 on the next window of shards -------------
# Run 1 trained shards 0..2 (30M tokens) for 1831 steps, ending at val 5.37.
# This picks that model up and trains it on shard 3 -- new data, same weights.
# One shard = 10M tokens = 610 steps ~= 1.4h at the measured 8.2 s/step.
CONFIG = TrainConfig(
    first_shard=3,
    num_shards=1,

    resume_from='latest',        # newest checkpoint in runs/ (model_001830.pt)
    # That checkpoint predates shard names, so its stored position cannot be checked
    # against this window and would land at an arbitrary offset into the new data.
    # Start at the front of shard 3 instead. Not needed for checkpoints written from
    # now on -- those record the shard by name and detect a moved window themselves.
    reset_data_position=True,

    # One continuous cosine across both runs: 1831 already done + 610 in this window.
    # Without this the schedule re-spans to this run's end and the LR jumps back up
    # (a warm restart) instead of continuing to decay.
    lr_schedule_steps=2441,
)

if __name__ == '__main__':
    train(CONFIG)
