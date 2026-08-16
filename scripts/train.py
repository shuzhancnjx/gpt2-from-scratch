#!/usr/bin/env python
"""Entry point for a pretraining run.

    python scripts/train.py                       # single device
    torchrun --standalone --nproc_per_node=8 scripts/train.py

Edit CONFIG below, or import TrainConfig and drive it yourself.
"""

from gpt2.training import TrainConfig, train

CONFIG = TrainConfig(
    # data window: shards[first_shard : first_shard + num_shards].
    # To continue a finished run on fresh data, bump first_shard past the last
    # window and set resume_from -- the model keeps training, the data is new.
    first_shard=0,
    num_shards=3,
    resume_from=None,
)

if __name__ == '__main__':
    train(CONFIG)
