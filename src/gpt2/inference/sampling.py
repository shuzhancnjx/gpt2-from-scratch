"""Choosing the next token from logits -- shared by every engine.

    greedy(logits)                               argmax
    sample(logits, temperature, top_k, top_p)    temperature, then top-k, then top-p
    make_sampler(temperature, top_k, top_p)      a one-argument sampler for the schedulers

Everything here is tensor ops only: no .item(), no int(), no `if` on a tensor value,
no boolean-mask indexing (that needs the count of True values on the CPU -- a sync).
So any sampler built here can be captured inside a CUDA graph. The `if`s below test
Python numbers (the settings), which are fixed when the sampler is built.

logits: (B, V). Returns (B, 1) token ids.
"""

import torch


def greedy(logits):
    return logits.argmax(-1, keepdim=True)


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature <= 0:
        return greedy(logits)

    logits = logits.float() / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))

    if top_p is not None:
        return _top_p(logits, top_p)

    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def _top_p(logits, top_p):
    # The top token's exclusive mass is 0, so `0 >= top_p` is True once top_p hits 0
    # and every token gets dropped. Treat it as greedy, like temperature == 0.
    if top_p <= 0:
        return greedy(logits)

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    drop = probs.cumsum(-1) - probs >= top_p           # mass strictly before this token
    probs = probs.masked_fill(drop, 0.0)               # not probs[drop] = 0: that syncs
    pick = torch.multinomial(probs, num_samples=1)     # position in the sorted order
    return torch.gather(sorted_idx, -1, pick)


def make_sampler(temperature=1.0, top_k=None, top_p=None):
    """A sampler with its settings fixed -- what a scheduler or a captured graph takes."""
    if temperature <= 0:
        return greedy

    def sampler(logits):
        return sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    return sampler
