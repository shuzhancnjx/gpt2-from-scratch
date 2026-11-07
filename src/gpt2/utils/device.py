"""Device choice, kept in one place so library code never decides it.

Classes take `device` as an argument; only entry points (scripts) and tests pick
one. Features that exist on a single backend -- CUDA graphs, torch.compile,
Triton -- are gated on capability, not on which machine the code runs on.
"""

import torch


def pick_device(requested='auto'):
    """'auto' prefers cuda, then mps (Apple GPU), then cpu."""
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def synchronize(device):
    """Wait for queued GPU work -- for timing only; correctness never needs it."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elif str(device) == 'mps':
        torch.mps.synchronize()


def default_dtype(device):
    """fp16 where it pays (CUDA); fp32 elsewhere, which keeps CPU tests exact."""
    return torch.float16 if str(device).startswith('cuda') else torch.float32
