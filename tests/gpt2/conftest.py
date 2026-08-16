import os

import pytest
import torch

from gpt2.data import DEFAULT_DATA_ROOT
from gpt2.gpt import GPT, GPTConfig

# Everything here runs on CPU with a toy model so the suite stays fast.
TINY = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=32,
            dropout=0.0, bias=True)


@pytest.fixture
def tiny_config():
    return GPTConfig(**TINY)


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return GPT(tiny_config)


class FakeLoader:
    """Deterministic (x, y) batches -- lets the loop be tested without the dataset."""

    def __init__(self, B, T, vocab_size, seed=0):
        self.B, self.T = B, T
        self.total_num_tokens = B * T * 64
        g = torch.Generator().manual_seed(seed)
        # one long stream, carved into batches, so different B values covering the
        # same token count see exactly the same tokens in the same order
        self.stream = torch.randint(0, vocab_size, (B * T * 64 + 1,), generator=g)
        self.pos = 0

    def next_batch(self):
        n = self.B * self.T
        buf = self.stream[self.pos:self.pos + n + 1]
        self.pos = (self.pos + n) % (len(self.stream) - n - 1)
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)

    def state_dict(self):
        return {'pos': self.pos}


requires_data = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_DATA_ROOT),
    reason=f'no token shards at {DEFAULT_DATA_ROOT}')
