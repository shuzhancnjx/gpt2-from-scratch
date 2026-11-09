"""The static engine's decode step as one CUDA graph (CUDA only).

Fixed buffers -- a KVCache of B rows x W columns, the next tokens, their positions and
the attention mask -- and one CapturedGraph (inference/cuda_graph.py) of the model's
decode forward over them. GptInference fills the buffers in place and replays; one
DecodeStep per (B, W) bucket.
"""

import torch

from inference.cuda_graph import CapturedGraph
from inference.static_batch.kv_cache import KVCache


class DecodeStep:

    def __init__(self, model, config, B, W, device, dtype):
        self.model, self.B, self.W = model, B, W
        self.cache = KVCache(B, device, config, max_tokens=W, dtype=dtype)
        self.token = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.mask = torch.zeros(B, W, dtype=torch.bool, device=device)
        self.graph = CapturedGraph(self._forward, warmup=3)
        self.reset()                                    # undo what the warm-up wrote

    def _forward(self):
        """The recorded body: one decode forward over the fixed buffers -> logits."""
        logits, _ = self.model(self.token, kv_cache=self.cache, pos_ids=self.pos,
                               attn_mask=self.mask)
        return logits

    def replay(self):
        """One decode step from the buffers as they are now -> logits (B, 1, vocab)."""
        return self.graph.replay()

    def reset(self):
        """Empty, in place: the graph holds these buffers' addresses."""
        self.cache.reset()
        self.mask.zero_()
