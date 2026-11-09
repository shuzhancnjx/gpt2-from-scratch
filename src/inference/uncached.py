"""Generation without a KV cache: re-run the whole prefix for every new token.

The slow, obviously-correct path -- what the training loop samples with (short samples,
a model mid-training) and the reference the cached engines are checked against. It
picks tokens with any sampler from inference/sampling.py, so sampling lives in one place.
"""

import torch


@torch.no_grad()
def generate_uncached(model, idx, max_new_tokens, sampler, vocab_limit=None):
    """idx (B, T) -> (B, T + max_new_tokens), one full forward per new token.

    sampler      logits (B, vocab) -> tokens (B, 1), e.g. make_sampler(top_k=50)
    vocab_limit  logits past it are dropped: the head is padded past the tokenizer's
                 real vocab, and those ids are not decodable
    The model is put in eval mode for the loop and restored after.
    """
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        idx = torch.cat([idx, sampler(logits[:, -1, :vocab_limit])], dim=1)
    if was_training:
        model.train()
    return idx
