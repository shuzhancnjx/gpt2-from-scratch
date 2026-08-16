"""Device, distributed, and mixed-precision setup.

Everything that used to be a module-level global -- device, master_process,
use_amp -- lives on Runtime and gets passed explicitly.
"""

import contextlib
import os
from dataclasses import dataclass

import torch
from torch.distributed import init_process_group, destroy_process_group


@dataclass
class Runtime:
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    device: str            # 'cuda:0' / 'mps' / 'cpu'  -- what tensors move to
    device_type: str       # 'cuda'   / 'mps' / 'cpu'  -- what autocast wants
    master_process: bool
    use_compile: bool
    use_amp: bool

    def amp_ctx(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def synchronize(self):
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        elif self.device_type == 'mps':
            torch.mps.synchronize()

    def print0(self, *args, **kwargs):
        if self.master_process:
            print(*args, **kwargs)

    def shutdown(self):
        if self.ddp:
            destroy_process_group()


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_runtime(use_compile=None, use_amp=None, seed=42):
    """Initialise distributed (if launched under torchrun) and pick the device."""
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), 'need cuda for ddp'
        init_process_group('nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        master_process = True
        device = pick_device()

    # 'cuda:0' is a device, 'cuda' is a device type -- autocast and fused AdamW
    # want the type
    device_type = 'cuda' if device.startswith('cuda') else device

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    rt = Runtime(
        ddp=ddp, rank=rank, local_rank=local_rank, world_size=world_size,
        device=device, device_type=device_type, master_process=master_process,
        use_compile=(device_type == 'cuda') if use_compile is None else use_compile,
        use_amp=(device_type == 'cuda') if use_amp is None else use_amp,
    )

    rt.print0(f'=> using ddp: {rt.ddp}')
    rt.print0(f'=> device: {rt.device}')
    rt.print0(f'=> ddp_world_size: {rt.world_size}')
    rt.print0(f'=> compile: {rt.use_compile}, amp(bf16): {rt.use_amp}')
    return rt
