"""PrefillBatch: several prompts packed end to end into one prefill forward.

Metadata only -- the packed data is the hidden state itself. Built by the ModelRunner
(eager prefill) or held as fixed buffers by PrefillGraphs; read by gpt.py (positions,
each prompt's last token), by every cache's prefill_attend and write, and by the prefill
attention kernel. Its own module because all of them use it, not only the caches.
"""

from dataclasses import dataclass

import torch


@dataclass
class PrefillBatch:
    """Several fresh requests' prompts, packed end to end into one forward.

        input ids      (1, sum(lens))   prompt 0 | prompt 1 | ... | prompt N-1
        slots          [N]              each request's req_pool_idx (its row)
        lens           [N]              each prompt's length
        rows           (sum(lens),)     the slot of every packed token
        positions      (sum(lens),)     every token's position in its own prompt (from 0)

    No padding anywhere. Attention stays inside each prompt (prefill_attend), the
    cache writes each token to (rows[i], positions[i]), and the output layer runs only on
    each prompt's last token (last). This object is metadata only: the packed data is
    the hidden state x itself.
    """
    slots: list
    lens: list
    slots_gpu: torch.Tensor
    lens_gpu: torch.Tensor
    rows: torch.Tensor
    positions: torch.Tensor
    last: torch.Tensor

    @classmethod
    def build(cls, slots, lens, device):
        lens_gpu = torch.tensor(lens, dtype=torch.long, device=device)
        slots_gpu = torch.tensor(slots, dtype=torch.long, device=device)
        starts = torch.cumsum(lens_gpu, 0) - lens_gpu                  # where each prompt begins
        total = int(sum(lens))
        rows = torch.repeat_interleave(slots_gpu, lens_gpu, output_size=total)
        positions = torch.arange(total, device=device) - torch.repeat_interleave(
            starts, lens_gpu, output_size=total)
        return cls(list(slots), list(lens), slots_gpu, lens_gpu, rows, positions,
                   last=starts + lens_gpu - 1)
