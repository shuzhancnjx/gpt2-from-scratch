"""Run artifacts: the loss log and checkpoints.

This module is a leaf on purpose -- it writes and reads whatever it is handed and
never imports the model or the data loader. Anything that needs to *execute* a
checkpoint belongs in training/ or inference/, not here.
"""

import glob
import os
from dataclasses import asdict, is_dataclass

import torch

# repo_root/runs
DEFAULT_RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'runs')


class Logger:
    """Owns a run directory: one append-only loss log plus rolling checkpoints."""

    def __init__(self, run_dir=None, master_process=True, resuming=False,
                 keep_last_n=3, log_name='log.txt'):
        self.run_dir = run_dir or DEFAULT_RUN_DIR
        self.master_process = master_process
        self.keep_last_n = keep_last_n
        self.log_file = os.path.join(self.run_dir, log_name)

        if master_process:
            os.makedirs(self.run_dir, exist_ok=True)
            if not resuming:
                # only a fresh run starts a fresh log -- resuming must append or it
                # erases the very history it is continuing
                with open(self.log_file, 'w'):
                    pass

    def log(self, step, split, value):
        """Append one '<step> <split> <value>' record. Only the master writes."""
        if not self.master_process:
            return
        with open(self.log_file, 'a') as f:
            f.write(f'{step} {split} {value:.6f}\n')

    def checkpoint_path(self, step):
        return os.path.join(self.run_dir, f'model_{step:06d}.pt')

    def save_checkpoint(self, step, model, optimizer, loader, config=None):
        """Write weights + optimizer + loader position, then prune old ones.

        `model` must be the raw module (not DDP- or compile-wrapped) so the
        state_dict keys stay clean. Config is stored as a plain dict, never a
        pickled class -- that is what lets checkpoints survive a refactor.
        """
        if not self.master_process:
            return None

        model_config = getattr(model, 'config', config)
        if is_dataclass(model_config):
            model_config = asdict(model_config)

        path = self.checkpoint_path(step)
        torch.save({
            'step': step,
            'config': model_config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loader': loader.state_dict(),
        }, path)
        print(f'=> saved checkpoint {path}')

        self.prune(keep=self.keep_last_n)
        return path

    def prune(self, keep):
        """Delete all but the newest `keep` checkpoints (zero-padded names sort)."""
        if not keep or not self.master_process:
            return []
        stale = sorted(glob.glob(os.path.join(self.run_dir, 'model_*.pt')))[:-keep]
        for old in stale:
            os.remove(old)
        if stale:
            print(f'=> pruned {len(stale)} old checkpoint(s)')
        return stale

    def latest_checkpoint(self):
        found = sorted(glob.glob(os.path.join(self.run_dir, 'model_*.pt')))
        return found[-1] if found else None


def load_checkpoint(path, map_location='cpu'):
    """Read a checkpoint dict. weights_only=False because it carries a config dict."""
    return torch.load(path, map_location=map_location, weights_only=False)


def read_log(path):
    """Parse a loss log into {'train': [(step, loss)], 'val': [...]}.

    Tolerates a torn final line from a run that is still writing.
    """
    out = {'train': [], 'val': []}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3 or parts[1] not in out:
                continue
            try:
                out[parts[1]].append((int(parts[0]), float(parts[2])))
            except ValueError:
                continue
    return out
