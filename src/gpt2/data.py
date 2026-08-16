"""Token shard loading.

Shards are raw uint16 token dumps written by scripts/prepare_data.py -- no header,
so np.fromfile is the correct reader (np.load would choke).
"""

import os

import numpy as np
import torch

# repo_root/data -- data lives outside the source tree, so this walks up out of
# src/gpt2/ to find it. Override with DataLoaderLite(data_root=...).
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'edu_fineweb10B')


def load_tokens(filename, dtype=np.uint16):
    npt = np.fromfile(filename, dtype=dtype)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    """Streams (x, y) batches over a window of token shards.

    The window is shards[start_shard : start_shard + max_shards]. next_batch()
    wraps modulo the window, so the window is what bounds the data a run ever
    touches -- pointing successive runs at successive windows continues one model
    over fresh data.
    """

    def __init__(self, B, T, device, process_rank, num_processes, split,
                 master_process=True, start_shard=0, max_shards=None, data_root=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process

        data_root = data_root or DEFAULT_DATA_ROOT
        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f'no data directory at {data_root} -- run scripts/prepare_data.py first')
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        shards = [os.path.join(data_root, s) for s in shards]

        found = len(shards)
        assert found > 0, f'no shards found for split {split} in {data_root}'
        assert 0 <= start_shard < found, \
            f'start_shard {start_shard} outside 0..{found - 1} for split {split}'
        if max_shards is not None:
            assert start_shard + max_shards <= found, (
                f'asked for {max_shards} shards from index {start_shard}, but split '
                f'{split} only has {found} (would need {start_shard + max_shards})')
            shards = shards[start_shard:start_shard + max_shards]
        else:
            shards = shards[start_shard:]
        self.shards = shards
        self.start_shard = start_shard          # window offset, for reporting/state

        self.current_shard = 0                  # index *within* the window
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = process_rank * B * T

        self.device = device
        self.tokens_per_shard = len(self.tokens)
        self.total_num_tokens = self.tokens_per_shard * len(self.shards)

        if master_process:
            last = start_shard + len(self.shards) - 1
            print(f'found {found} shards for split {split}, '
                  f'using {len(self.shards)} (index {start_shard}..{last})')
            print(f'data_loader: {self.total_num_tokens} tokens in the working set')
            print(f'data_loader: 1 epoch = {self.total_num_tokens // (B * T)} micro batches')

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T).to(self.device, non_blocking=True)
        y = buf[1:].view(B, T).to(self.device, non_blocking=True)

        self.current_position += B * T * self.num_processes

        # every rank must flip shards on the same iteration, so the headroom we
        # require is a full global batch, not just this rank's slice
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y

    def state_dict(self):
        # Store the shard by NAME, not by index: an index only means something
        # relative to the window it was saved under, and the whole point of the
        # window is that the next run may move it.
        # base_position is rank-independent so rank 0's checkpoint restores on all ranks.
        return {'shard_name': os.path.basename(self.shards[self.current_shard]),
                'current_shard': self.current_shard,      # kept for older checkpoints
                'base_position': self.current_position - self.process_rank * self.B * self.T}

    def load_state_dict(self, state):
        names = [os.path.basename(s) for s in self.shards]
        name = state.get('shard_name')

        if name in names:                       # same window: resume exactly where we left off
            self.current_shard = names.index(name)
            self.current_position = state['base_position'] + self.process_rank * self.B * self.T
        elif name is None and state.get('current_shard', 0) < len(self.shards):
            self.current_shard = state['current_shard']       # pre-shard_name checkpoint
            self.current_position = state['base_position'] + self.process_rank * self.B * self.T
        else:                                   # window moved: start of the new data
            self.current_shard = 0
            self.current_position = self.process_rank * self.B * self.T
            if self.master_process:
                print(f'data_loader: checkpoint shard {name} is outside this window '
                      f'({names[0]}..{names[-1]}) -- starting at {names[0]}')

        self.tokens = load_tokens(self.shards[self.current_shard])
