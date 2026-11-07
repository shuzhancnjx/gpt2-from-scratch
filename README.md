# gpt2

A small GPT-2 built from scratch: tokenizer, pretraining, evaluation, and an inference
engine that grows from a fixed batch to paged, continuously batched serving with a
Triton attention kernel. Written to learn how LLMs work end to end; trained on a
MacBook Pro and an A100.

## Layout

```
src/                       one top-level package per layer (no umbrella package)
  model/
    gpt2/                  GPT-2; a second model would sit beside it
      gpt.py               GPTConfig, attention / MLP / blocks, GPT, generate()
      loading.py           load_model(): a checkpoint, 'gpt2' (OpenAI weights), or a config
  tokenizer/
    tokenizer.py           byte-pair encoding: BPETokenizer + BPETrainer
  utils/
    data.py                token shard loading, DataLoaderLite
    device.py              pick_device(), synchronize(), default_dtype()
    log.py                 loss log, checkpoints, pruning
  training/
    config.py              TrainConfig: every knob in one dataclass
    runtime.py             device / DDP / mixed-precision setup
    loop.py                train(), evaluate(), get_lr(), in-loop HellaSwag
  evaluation/
    likelihood.py          validation NLL, perplexity, bits per token, loss by position
    hellaswag.py           HellaSwag acc / acc_norm by likelihood
  inference/
    base_engine.py         BaseEngine: generate(prompts, SamplingParams), every engine
    base_kv_cache.py       BaseKVCache: the contract every KV cache implements
    sampling.py            greedy, temperature / top-k / top-p, SamplingParams (graph safe)
    static_batch/          one fixed batch, start to finish
      kv_cache.py            KVCache: one write pointer for the whole batch
      decode_step.py         the decode step captured in a CUDA graph
      inference.py           GptInference
    continuous_batch/      requests join and leave every step
      base_scheduler.py      BaseScheduler: the admit -> decode -> retire loop
      row/                   one full-width KV row per request
        row_cache.py           RowKVCache
        scheduler.py           Scheduler (the simple baseline)
      paged/                 fixed-size KV blocks from a shared pool
        block_allocator.py     BlockAllocator: free list, scratch block
        paged_cache.py         PagedKVCache: block table, gather, fused decode
        paged_scheduler.py     PagedScheduler: host-mirror decisions, CUDA-graph decode
        paged_attention.py     Triton decode attention over the block table (CUDA)
  app/
    llm.py                 LLM: the front door -- load once, pick an engine, generate()
scripts/        prepare_data, train*, plot_loss, eval_checkpoint,
                sample, chat_server, bench_paging, profile_serving
notebooks/      loss curve; Colab notebooks for A100 training and GPU profiling
docs/           request lifecycle, forward pass and sampling, learning plan, benchmarks
tests/          pytest suite (CUDA-only tests skip elsewhere)
learning/       earlier study code, kept as a record
```

`model/`, `tokenizer/` and `utils/` are the base layers. `training/`, `evaluation/` and
`inference/` build on them, and `app/` builds on `inference/`. Imports name the layer
directly: `from model.gpt2 import GPT`, `from app import LLM`.

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
from training import TrainConfig, train

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

## Inference

Three engines, from simplest to fastest. All three call the same `GPT.forward`, pass
it a KV cache (a `BaseKVCache`), and pick tokens with `inference/sampling.py`.

| engine | batch | KV memory per request | used by |
|---|---|---|---|
| `static_batch.GptInference` | fixed: every row starts and ends together | a full-width row | `scripts/sample.py` |
| `continuous_batch.Scheduler` | rows refill as requests finish | a full-width row (`RowKVCache`) | baseline in `bench_paging.py` |
| `continuous_batch.PagedScheduler` | rows refill as requests finish | only the blocks it needs (`PagedKVCache`) | `chat_server.py`, `bench_paging.py` |

### One request through the paged engine

```
PagedScheduler.run(prompts)                       BaseScheduler: the loop
  while requests are queued or running:
    admit()     CPU  find a free row, allocate ceil((prompt + max_new) / 16) blocks,
                     write the row's block table
                GPU  prefill: one forward over the whole prompt, writing its K/V into
                     those blocks; sample the first token
    decode()    GPU  one forward for every row, one new token each:
                       per layer: QKV projection -> write the new K/V into the pool
                       -> attention over the row's real blocks -> MLP
                     then sample, and copy the B new tokens to the CPU (the one sync)
    retire()    CPU  end-of-text or max length -> return the text, free the blocks
```

Decisions read **CPU mirrors** (`active_cpu`, `row_len_cpu`, `row_blocks_cpu`); the
model reads the **GPU copies** (`active_gpu`, `pos_gpu`, `block_table_gpu`). Admit and
retire write both; decode advances both by the same rule, so the CPU never reads the
GPU except for the sampled tokens.

### The decode step's two fast paths (CUDA)

```python
sched = PagedScheduler(model, enc, cfg, max_seqs=32, device='cuda', dtype=torch.float16,
                       decode_kernel=True)   # Triton paged attention: no gather, no mask
sched.capture_graph()                        # record the decode step; replay it each step
sched.run(prompts, max_new_tokens=128)
```

| decode attention | how |
|---|---|
| default | gather each row's blocks into a full-width copy, then masked SDPA |
| `decode_kernel=True` | `paged_attention.py` walks the block table up to the real length with an online softmax, reading K/V straight from the pool |

`capture_graph()` replays the whole decode step (forward, mask, sampler) with one
launch. Prefill, admission and retirement stay outside the graph; they write the same
fixed buffers the graph reads. Measured on an A100 (32 requests, gather path):
10.3 -> 3.9 ms per decode step. Details and the remaining bottlenecks are in
`docs/benchmarks/paging.md` and `docs/learning-plan.md`.

### Running it

```bash
python scripts/sample.py --check "The capital of France is"   # static engine, cached vs uncached
python scripts/chat_server.py                                 # local chat UI, paged engine
python scripts/bench_paging.py --graph --kernel               # row vs paged vs graph vs kernel (CUDA)
```

How a request moves through a server in general (scheduling, paging, prefix caching)
is in `docs/request-lifecycle.md`; one forward pass on the GPU, kernel by kernel, is in
`docs/forward-and-sampling.md`.

## Tests

```bash
pytest
```

Data-dependent tests skip automatically if `data/` is absent, and CUDA / Triton tests
skip without an NVIDIA GPU.

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
`src/tokenizer/tokenizer.py`.

Everything past that point — the package structure, shard windows for continuing a run
on fresh data, checkpoint/resume handling, the test suite, and the Apple-silicon
benchmarking — is my own work on top.
