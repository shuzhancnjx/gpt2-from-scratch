# gpt2

A small GPT-2 built from scratch — tokenizer, pretraining, and (soon) inference and
fine-tuning. Written to learn how LLMs work end to end, and trained on a MacBook Pro.

## Layout

```
src/gpt2/
  gpt.py            GPTConfig, attention/MLP/blocks, GPT, generate()
  data.py           token shard loading, DataLoaderLite
  tokenizer.py      byte-pair encoding: BPETokenizer + BPETrainer
  log.py            loss log, checkpoints, pruning
  training/
    config.py       TrainConfig — every knob in one dataclass
    runtime.py      device / DDP / mixed-precision setup
    loop.py         train(), evaluate(), get_lr()
scripts/            prepare_data.py, train.py, plot_loss.py
tests/              pytest suite
learning/           earlier study code, kept as a record
```

`gpt.py`, `data.py`, `tokenizer.py`, and `log.py` are leaves that import nothing from
each other. `training/` is the application built on top of them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
```

## Getting the data

Training uses [**FineWeb-Edu**](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(the `sample-10BT` subset), tokenized with the GPT-2 BPE and written as raw `uint16`
shards of 10M tokens each.

```bash
python scripts/prepare_data.py
```

That streams the dataset from Hugging Face and writes `data/edu_fineweb10B/`:

| | |
|---|---|
| shards | 995 train + 1 val |
| tokens per shard | 10,000,000 |
| on disk | 20 MB per shard, ~19 GB total |
| format | headerless `uint16` — read with `np.fromfile`, **not** `np.load` |

The download is large; the tokenization is parallel across half your cores and takes a
while. You don't need all of it — a run reads only the shard window it is configured
for, so a handful of shards is enough to train on.

`data/` and `runs/` are gitignored. Nothing here ships datasets or checkpoints.

## Training

```bash
python scripts/train.py
```

Edit `CONFIG` in that file, or drive `TrainConfig` yourself:

```python
from gpt2.training import TrainConfig, train

train(TrainConfig(first_shard=0, num_shards=3))
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py
```

### Training across shard windows

A run reads `shards[first_shard : first_shard + num_shards]`. To continue the *same
model* on *new data*, move the window and resume:

```python
TrainConfig(
    first_shard=3,           # next window
    num_shards=1,
    resume_from='latest',    # newest checkpoint in runs/
    lr_schedule_steps=2441,  # see below
)
```

Checkpoints record the shard by name, so a moved window is detected and the loader
starts at the front of the new data instead of silently rewinding into data the model
has already seen.

`resume_from` accepts `'latest'`, a bare filename, or a path. `'latest'` starts fresh
if the run directory is empty; an explicit path that is missing raises.

**`lr_schedule_steps` is cumulative.** It is the span the cosine is stretched over, not
the length of this run — so it is *every step trained so far, plus this window*. A first
run of 1831 steps followed by a 610-step window gives `2441`. Leave it unset and the
schedule re-spans to the current run's end, which pushes the learning rate back up
toward `max_lr` — a warm restart. That is a legitimate technique, but it should be a
choice rather than a side effect.

Two knobs pair with a continuation:

- `reset_data_position=True` — restore weights and optimizer but start at the front of
  the window. Needed only for checkpoints written before shard names were recorded,
  since those cannot be checked against the window.
- `keep_last_n_checkpoints` — checkpoints are ~1.5 GB each; the default keeps 3.

## Plotting

```bash
python scripts/plot_loss.py --show          # writes runs/loss.png
```

`notebooks/loss_curve.ipynb` renders the same curve inline and can overlay several runs.

## Tests

```bash
pytest
```

Data-dependent tests skip automatically if `data/` is absent.

## Notes from training on Apple silicon

Measured on an M1 Pro (16 GB), `B=2 T=1024`, 16384-token batches:

| setting | tok/s |
|---|---|
| `torch.compile` + bf16 autocast | 985 |
| eager + bf16 autocast | 1560 |
| **eager + fp32** | **1820** |
| eager + fp16 autocast | 2030 |

Both `torch.compile` and bf16 autocast are *losses* on MPS — Metal has no native bf16
path, so autocast emulates it (a raw 2048³ matmul runs 1.76× slower in bf16 than fp32),
and the inductor MPS backend loses to Apple's own kernels. `use_compile` and `use_amp`
default to CUDA-only for this reason; both can be forced on.

Micro-batch size is memory-bound rather than compute-bound here: `B=2` measured fastest,
`B=4` was *slower* under allocator pressure, and `B=8` ran out of memory.

## Acknowledgements

This project follows **Andrej Karpathy's**
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course, in
particular [*Let's reproduce GPT-2 (124M)*](https://github.com/karpathy/build-nanogpt)
and [nanoGPT](https://github.com/karpathy/nanoGPT). The model architecture, the
weight-tying and scaled-residual initialisation, the gradient-accumulation loop, and the
FineWeb-Edu data pipeline all come from that material.

The earlier steps of the course are kept in `learning/` — the attention exercise, the
bigram-to-transformer progression, and the byte-pair encoder that became
`src/gpt2/tokenizer.py`.

Everything past that point — the package structure, shard windows for continuing a run
on fresh data, checkpoint/resume handling, the test suite, and the Apple-silicon
benchmarking — is my own work on top.
