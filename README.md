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
      continuous_engine.py   ContinuousEngine: run() / generate() over each engine's _run() loop
      model_runner.py        ModelRunner: execution (GPU) -- model calls, decode buffers, CUDA
                             graphs; cache-agnostic, shared by both engines
      row/                   one full-width KV row per request
        row_cache.py           RowKVCache
        engine.py              RowEngine (the simple baseline)
      paged/                 fixed-size KV blocks from a shared pool
        engine.py              PagedEngine: the orchestrator -- one step = schedule, run, update
        paged_scheduler.py     PagedScheduler: policy (CPU) -- admit, promote, retire -> SchedulerOutput
        block/                 memory (CPU)
          block_manager.py       BlockManager: blocks + block-table rows per request
          block_allocator.py     BlockAllocator: the free list of blocks, scratch block
        paged_cache.py         PagedKVCache: K/V pools, block table, positions
        decode_attention.py    Triton decode attention through the block table (CUDA)
        prefill_attention.py   Triton FlashAttention over packed prompts: prefill (CUDA)
        graph/                 PrefillGraphs, DecodeGraphs (CUDA graphs)
  app/
    llm.py                 LLM: the front door -- load once, pick an engine, generate()
    chat_server.py         FastAPI: chat page + OpenAI-compatible /v1 API, one engine thread
    chat.html              the chat page
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
pip install -e ".[viz,serve,dev]"
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
| `continuous_batch.RowEngine` | rows refill as requests finish | a full-width row (`RowKVCache`) | baseline in `bench_paging.py` |
| `continuous_batch.PagedEngine` | rows refill as requests finish | only the blocks it needs (`PagedKVCache`) | `chat_server.py`, `bench_paging.py` |

### One request through the paged engine

`PagedEngine` wires three components, each owning one kind of state:

| component | owns | does |
|---|---|---|
| `PagedScheduler` (policy, CPU) | waiting queue, prefilled buffer, decode rows | decides who is admitted, promoted, retired -> `SchedulerOutput` |
| `BlockManager` (memory, CPU) | which blocks and block-table rows each request holds | reserve, move to a decode row, release |
| `ModelRunner` (execution, GPU) | model, `PagedKVCache`, CUDA graphs, `next_token_gpu` | prefill, decode, sampling -- the same class `RowEngine` runs on |

```
PagedEngine.step()
  out = scheduler.schedule()        CPU  promote prefilled requests into free rows; choose this
                                         step's prefills (FCFS, block room, token budget) and
                                         reserve their blocks and prefilled rows
  runner.prefill(rows, prompts)     GPU  ONE packed forward over the new prompts (graph or eager);
                                         their K/V into their blocks; sample each first token
  scheduler.prefilled_with(...)     CPU  -> the prefilled buffer; promote again into free rows
  runner.apply(started=promoted)    GPU  promoted rows: first token in, row active
  runner.decode()                   GPU  one forward for every row, one new token each
  scheduler.update(tokens)          CPU  tokens to requests; end-of-text / max length -> retire,
                                         blocks back to the pool
  runner.apply(stopped=retired)     GPU  retired rows idle
```

The scheduler decides from **CPU state only**; the runner owns the **GPU state**
(`next_token_gpu`, `active_gpu`, `pos_gpu`, `block_table_gpu`). The engine is the only
place that copies from the GPU to the CPU: the first tokens after a prefill and the new
tokens after a decode. Because the policy needs no model, it is tested without one.

### The decode step's two fast paths (CUDA)

```python
engine = PagedEngine(model, enc, cfg, max_seqs=32, device='cuda', dtype=torch.float16,
                     cuda_graph=True)       # prefill and decode replayed as CUDA graphs
engine.run(prompts, max_new_tokens=128)
```

| attention (paged cache) | how |
|---|---|
| CUDA (always) | Decode: `decode_attention.py`, the Triton decode kernel: walks the block table up to each row's real length with an online softmax, reading K/V straight from the pool, one program per (request, head). Prefill: `prefill_attention.py`, FlashAttention over the packed prompts: tiles of queries x tiles of keys on tensor cores, each query bounded to its own prompt (prompts start at 0, so no pool reads) |
| CPU / Apple GPU | the same attention as SDPA: per-prompt causal for prefill, gather + mask for decode (Triton runs only on NVIDIA GPUs) |

With `cuda_graph=True`, `DecodeGraphs` (`paged/graph/decode_graph.py`) replays the whole
decode step (forward, mask, sampler) with one launch -- one graph per sampler setting,
captured before the first run that needs it. Prefill, admission and retirement stay
outside the graph; they write the same fixed buffers the graph reads. Measured on an A100 (32 requests, gather path):
10.3 -> 3.9 ms per decode step. Details and the remaining bottlenecks are in
`docs/benchmarks/paging.md` and `docs/learning-plan.md`.

### Running it

```bash
python scripts/sample.py --check "The capital of France is"   # static engine, cached vs uncached
python scripts/chat_server.py                                 # chat UI + OpenAI API (FastAPI), paged engine
python scripts/bench_paging.py --graph                        # row vs paged vs paged + graph (CUDA)
```

The chat server (`pip install -e ".[serve]"`) also speaks the OpenAI API, so standard
clients work against it; interactive docs are at `http://127.0.0.1:8765/docs`:

```bash
curl -s localhost:8765/v1/completions -H 'Content-Type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

FastAPI handles requests concurrently, but only one engine thread ever touches the
scheduler: endpoints queue a job and await its result, and the engine runs requests that
arrive together as one batch (grouped by sampling settings).

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
