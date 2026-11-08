"""PagedEngine: continuous batching over a paged KV cache -- the orchestrator.

Three components, each with one kind of state, and a loop that wires them together:

    PagedScheduler  (paged_scheduler.py)      policy, CPU: who is admitted, promoted, retired
    BlockManager    (block/block_manager.py)  memory, CPU: which blocks and table rows each request holds
    ModelRunner     (../model_runner.py)      execution, GPU: model, KV cache, graphs, sampling
                                              (shared with RowEngine)

One step:

    out = scheduler.schedule(max_new)          promote the ready ones; choose + reserve prefills
    if out.prefill:
        first = runner.prefill(rows, prompts)  one packed prefill (graph or eager)
        scheduler.prefilled_with(..., first)   -> the prefilled buffer        (sync: first tokens)
        promoted += scheduler.promote()        fresh prefills take rows still free
    runner.apply(started=promoted)             their first tokens in, the rows decode
    if anything decodes:
        tokens = runner.decode()               one decode step (graph or eager)
        retired = scheduler.update(tokens)     tokens to requests, finished ones out  (sync)
        runner.apply(stopped=retired rows)

Only the engine copies from the GPU to the CPU: the first tokens after a prefill, the new
tokens after a decode. Everything the scheduler decides, it decides from CPU state.
"""

import torch

from inference.continuous_batch.continuous_engine import ContinuousEngine
from inference.continuous_batch.model_runner import ModelRunner
from inference.continuous_batch.paged.block.block_manager import BlockManager
from inference.continuous_batch.paged.paged_cache import PagedKVCache
from inference.continuous_batch.paged.paged_scheduler import PagedScheduler


class PagedEngine(ContinuousEngine):

    def __init__(self, model, enc, config, max_seqs=4, block_size=16, num_blocks=None,
                 max_tokens=None, watermark=0, device='cpu', dtype=torch.float32,
                 cuda_graph=False, prefilled_rows=None, max_prefill_tokens='auto'):
        """
        model, enc, config  the loaded GPT, its tokenizer, its GPTConfig
        max_seqs            decode rows: the most requests decoding at once
        block_size          tokens per KV block (16)
        num_blocks          size of the shared KV pool in blocks, incl. the scratch block.
                            Default: as much memory as a RowKVCache of max_seqs rows
        max_tokens          W, the most tokens one request may hold. Default: the context length
        watermark           blocks kept free while other requests hold blocks
        device, dtype       where the pool lives and its precision
        cuda_graph          CUDA graphs for prefill (one per token bucket) and decode (one
                            per sampler setting), captured before each run (CUDA only)
        prefilled_rows      capacity of the prefilled buffer (>= 1, default max_seqs)
        max_prefill_tokens  prompt tokens one step may prefill. 'auto' (default): 1024 with
                            CUDA graphs (A100: vs no cap, +1% tok/s, -8% TTFT, worst step
                            18.8 -> 8.4 ms), no cap without (each eager prefill pays ~10 ms of
                            fixed cost). None = no cap.
        """
        self.enc = enc
        self.W = max_tokens if max_tokens is not None else config.block_size
        if self.W % block_size:
            raise ValueError(f'block_size {block_size} must divide max_tokens {self.W}')
        prefilled_rows = max_seqs if prefilled_rows is None else prefilled_rows
        if prefilled_rows < 1:
            raise ValueError('prefilled_rows must be >= 1: every request is prefilled into '
                             'the prefilled buffer before it takes a decode row')
        if num_blocks is None:                         # same memory as a RowKVCache of max_seqs rows
            num_blocks = 1 + max_seqs * (self.W // block_size)

        cache = PagedKVCache(config, num_blocks, block_size, max_seqs, self.W // block_size,
                             device, dtype, prefilled_rows=prefilled_rows)
        self.runner = ModelRunner(model, enc, cache, max_seqs, self.W, device, cuda_graph,
                                  max_prefill_prompts=prefilled_rows)
        self.blocks = BlockManager(cache, num_blocks, block_size, max_seqs, prefilled_rows)
        if max_prefill_tokens == 'auto':
            max_prefill_tokens = 1024 if self.runner.prefill_graphs is not None else None
        self.scheduler = PagedScheduler(self.blocks, max_seqs, self.W, enc.eot_token,
                                        watermark=watermark, max_prefill_tokens=max_prefill_tokens)

    # -- what callers read ---------------------------------------------------------------

    @property
    def stats(self):
        """peak_active, peak_blocks, decode_steps of the last run."""
        return self.scheduler.stats

    @property
    def generated(self):
        """{prompt index: generated token ids} of the last run."""
        return self.scheduler.finished

    # -- the loop ------------------------------------------------------------------------

    def _run(self, prompts, sampling_params):
        """The loop. Returns {prompt index: generated token ids}."""
        prompt_ids = [self.enc.encode(p) for p in prompts]    # before the run starts
        self.runner.start(sampling_params)             # graphs captured before, never during
        self.blocks.reset()
        self.scheduler.reset()
        self.scheduler.add(prompt_ids)
        while not self.scheduler.done():
            self.step(sampling_params)
        return self.scheduler.finished

    def step(self, sampling_params):
        """One engine step: admit (promote, prefill, promote again), then decode."""
        max_new = sampling_params.max_new_tokens
        out = self.scheduler.schedule(max_new)
        promoted = out.promote
        if out.prefill:
            first = self.runner.prefill([a.slot for a in out.prefill],
                                        [a.ids for a in out.prefill], sampling_params)
            self.scheduler.prefilled_with(out.prefill, first.view(-1).tolist())   # sync
            promoted = promoted + self.scheduler.promote()
        self.runner.apply(started=[(a.row, a.tokens[-1]) for a in promoted])
        if self.scheduler.num_decoding:
            tokens = self.runner.decode(sampling_params).view(-1).tolist()       # sync
            retired = self.scheduler.update(tokens, max_new)
            self.runner.apply(stopped=[a.row for a in retired])
