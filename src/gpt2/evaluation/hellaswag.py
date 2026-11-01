"""HellaSwag: does higher likelihood pick the sensible continuation?

Each example is a context and four candidate endings; one is what actually happened,
three are machine-written distractors that are fluent but wrong. A language model
answers by likelihood alone: score every ending's log-probability given the context,
and pick the highest. No fine-tuning, no prompting tricks -- it tests whether the
probabilities the model learned carry *meaning*, on text unlike its training data.

Two scores per ending:

    acc       sum of the ending's token log-probs. Penalises long endings simply for
              having more tokens -- each token adds a negative number.
    acc_norm  mean log-prob per ending token (length-normalised). The standard figure.

Random guessing scores 25%. OpenAI's GPT-2 124M scores roughly 29-30% acc_norm.

Mechanics. Each ending is rendered as one row, context + ending, with a mask that is 1
only over the ending's tokens. Logits at position t predict token t+1, so the mask is
shifted along with the targets. Rows are right-padded to batch several examples
together; with causal attention the padding (after every real token) cannot change any
real token's prediction, and the mask excludes it from the score.
"""

import contextlib
import json
import os
import urllib.request

import torch
import torch.nn.functional as F

# The original GitHub copy (rowanz/hellaswag) no longer resolves; the same 10,042-example
# validation set is published on Hugging Face as parquet.
URL = 'https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet'
DEFAULT_CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'gpt2-hellaswag')


def download(cache_dir=DEFAULT_CACHE):
    path = os.path.join(cache_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(path):
        import pyarrow.parquet as pq                 # ships with `datasets`; present on Colab
        os.makedirs(cache_dir, exist_ok=True)
        parquet = os.path.join(cache_dir, 'validation.parquet')
        urllib.request.urlretrieve(URL, parquet)
        rows = pq.read_table(parquet, columns=['ctx', 'endings', 'label']).to_pylist()
        tmp = path + '.part'
        with open(tmp, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        os.replace(tmp, path)                        # atomic: never a half-written cache
    return path


def iterate_examples(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def render(example, enc):
    """→ (rows, masks, label): four token lists, four masks over the ending tokens."""
    ctx = enc.encode(example['ctx'])
    rows, masks = [], []
    for ending in example['endings']:
        end = enc.encode(' ' + ending)              # GPT-2 tokens carry their leading space
        rows.append(ctx + end)
        masks.append([0] * len(ctx) + [1] * len(end))
    return rows, masks, int(example['label'])


@torch.no_grad()
def score(model, rendered, device='cpu', autocast_dtype=None):
    """Log-likelihood of every ending in a list of rendered examples.

    Returns two (n_examples, 4) tensors: summed log-prob, and mean log-prob per token.
    """
    rows = [r for rs, _, _ in rendered for r in rs]
    masks = [m for _, ms, _ in rendered for m in ms]
    L = max(len(r) for r in rows)
    tok = torch.zeros(len(rows), L, dtype=torch.long)
    msk = torch.zeros(len(rows), L)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tok[i, :len(r)] = torch.tensor(r)
        msk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    tok, msk = tok.to(device), msk.to(device)

    ctx = (torch.autocast(device_type=str(device).split(':')[0], dtype=autocast_dtype)
           if autocast_dtype is not None else contextlib.nullcontext())
    with ctx:
        logits, _ = model(tok)
    targets, mask = tok[:, 1:], msk[:, 1:]          # shift: position t predicts token t+1
    logp = -F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                            targets.reshape(-1), reduction='none').view(len(rows), L - 1)
    total = (logp * mask).sum(1)
    mean = total / mask.sum(1)
    return total.view(-1, 4).cpu(), mean.view(-1, 4).cpu()


def evaluate_hellaswag(model, enc, path=None, limit=None, batch_examples=4, device='cpu',
                       autocast_dtype=None):
    """→ dict(acc, acc_norm, n). `limit` scores only the first N examples (for quick runs)."""
    model.eval()
    path = path or download()
    correct = correct_norm = n = 0
    batch = []

    def flush():
        nonlocal correct, correct_norm, n
        total, mean = score(model, batch, device, autocast_dtype)
        labels = torch.tensor([label for _, _, label in batch])
        correct += (total.argmax(1) == labels).sum().item()
        correct_norm += (mean.argmax(1) == labels).sum().item()
        n += len(batch)
        batch.clear()

    for example in iterate_examples(path, limit):
        batch.append(render(example, enc))
        if len(batch) == batch_examples:
            flush()
    if batch:
        flush()
    return dict(acc=correct / n, acc_norm=correct_norm / n, n=n)
