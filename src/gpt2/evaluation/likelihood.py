"""Validation log-likelihood: how much probability the model gives held-out text.

The training loss is an average negative log-likelihood (NLL) per token, so these
are one quantity in three units:

    nll             nats per token: the average of  -log p(true next token)
    perplexity      exp(nll): the model is "effectively choosing among this many tokens"
    bits_per_token  nll / ln 2: the compression view -- bits to encode each token

Every model is scored on the SAME fixed window at the start of the val shard -- text
no training run ever sees -- so differences between numbers come from the models,
not the data.

Loss by position (per_position=True) is the fourth view: the average NLL of the
token at each position 0..T-1 of a chunk. Early positions have little context, so
their loss is high; a model that uses its context well drives the curve down as
position grows. A flat curve would mean the model ignores what came before.
"""

import contextlib
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def val_nll(model, tokens, T=1024, batch_size=8, max_tokens=None, device='cpu',
            autocast_dtype=None, per_position=False):
    """Score the first `max_tokens` of a 1-D token stream in non-overlapping T-token chunks.

    Returns nll, perplexity, bits_per_token, tokens; plus position_nll (a list of T
    averages) when per_position is set.
    """
    model.eval()
    n_chunks = (len(tokens) - 1) // T
    if max_tokens is not None:
        n_chunks = min(n_chunks, max_tokens // T)
    if n_chunks == 0:
        raise ValueError(f'need at least {T + 1} tokens, got {len(tokens)}')

    ctx = (torch.autocast(device_type=str(device).split(':')[0], dtype=autocast_dtype)
           if autocast_dtype is not None else contextlib.nullcontext())
    total, count = 0.0, 0
    by_position = torch.zeros(T, dtype=torch.float64)

    for start in range(0, n_chunks, batch_size):
        b = min(batch_size, n_chunks - start)
        chunk = torch.stack([tokens[(start + i) * T:(start + i) * T + T + 1] for i in range(b)])
        chunk = chunk.to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]              # position t predicts token t+1
        with ctx:
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction='none').view(b, T)
        total += nll.sum().item()
        count += nll.numel()
        if per_position:
            by_position += nll.sum(0).cpu().double()      # MPS has no float64: convert on the CPU

    nll = total / count
    out = dict(nll=nll, perplexity=math.exp(nll), bits_per_token=nll / math.log(2), tokens=count)
    if per_position:
        out['position_nll'] = (by_position / (count // T)).tolist()
    return out
