"""The packed prefill as CUDA graphs -- one per padded token count (CUDA only).

An eager prefill forward costs ~10 ms of kernel launches whatever its size. A graph
replays all of them with one launch. A graph needs fixed shapes, so:

    tokens   pad sum(lens) up to a bucket T (BUCKETS)
    prompts  always N = runner.max_prefill_prompts entries; unused entries are dummies
    padding  padding tokens use the cache's pad row (all SCRATCH): they write only the
             scratch block and attend one scratch slot -- their output is never read

Real tokens are computed exactly as in the eager prefill: each one reads only its own
request's blocks, up to its own position, through the Triton kernel.

Needs from the rest of the engine (to add alongside this file):
  - PagedKVCache.pad_row: one extra block-table row that is always SCRATCH
  - PagedEngine._prefill: try PrefillGraphs.run first; None -> the eager forward
  - PagedEngine: build PrefillGraphs when the decode graph is captured (CUDA)
"""
import torch 
from inference.base_kv_cache import PrefillBatch
from inference.cuda_graph import BaseGraphs

# Small buckets: prefill graphs pay off on small, launch-bound prefills (the token budget
# keeps them <= 1024 tokens); a bigger batch is GPU-bound -- it runs eagerly, padding-free.
BUCKETS = (64, 128, 256, 512, 1024)


class PrefillGraphs(BaseGraphs):
    """Prefill graphs for one PagedEngine.

    runner:   the ModelRunner (its model, cache, device and max_prefill_prompts are used)
    buckets:  padded token counts to capture, smallest first
    """

    def __init__(self, runner, buckets=BUCKETS):
        super().__init__(runner)                      # CUDA check, self.runner, self.graphs
        self.buckets = tuple(sorted(buckets))
        self._bufs = {}

    def prepare(self, key):
        """Capture bucket key's graph if new. Safe at any time, even mid-run: the warm-up
        runs on padding-only buffers, so it writes only the scratch block."""
        self.graphs.get(key)

    def key(self, total_tokens):
        """The smallest bucket >= total_tokens, or None if none is big enough."""

        return next((T for T in self.buckets if T >= total_tokens), None)

    def _buffers(self, T):
        """The fixed tensors bucket T's graph reads, all on the device:

            idx        (1, T)  packed token ids (padding: 0)
            batch      a PrefillBatch whose tensors are fixed buffers:
                         slots_gpu (N,)  lens_gpu (N,)  last (N,)
                         rows (T,)       positions (T,)
                       start as padding: rows / slots_gpu = cache.pad_row, positions /
                       lens_gpu / last = 0. Its CPU lists (slots, lens) are dummies --
                       the kernel path reads only the tensors.
        Returns them as a dict.
        """

        cache, device, N = self.runner.cache, self.runner.device, self.runner.max_prefill_prompts
        pad = cache.pad_row 

        long = dict(dtype=torch.long, device=device)
        return dict(
            idx=torch.zeros(1, T, **long), 
            batch=PrefillBatch(
                slots=[pad] * N, lens=[0] * N,
                slots_gpu=torch.full((N,), pad, **long),
                lens_gpu=torch.zeros(N, **long),
                rows=torch.full((T,), pad, **long),
                positions=torch.zeros(T, **long), 
                last=torch.zeros(N, **long)))

    def _build(self, T):
        """GraphCache's build(): make bucket T's buffers (kept in self._bufs[T], filled by
        run() before each replay) and return the body: the runner's prefill model
        call (prefill_logits) on those buffers -- logits (1, N, vocab)."""
        buf = self._bufs[T] = self._buffers(T)
        runner = self.runner
        return lambda: runner.prefill_logits(buf['idx'], buf['batch'])

    def run(self, slots, prompts, sampling_params):
        """Prefill these prompts into these slots with one graph replay, then sample.

        slots:            each request's block-table row (blocks already set)
        prompts:          their token-id lists
        sampling_params:  how to pick each first token (sampled after the replay: the
                          graph itself returns logits, so one graph serves every setting)
        Returns the first tokens on the device (n, 1) -- like DecodeGraphs.run -- or None
        if the batch is bigger than the largest bucket (the caller runs the eager prefill).
        """
        lens = [len(p) for p in prompts]
        total, n = sum(lens), len(prompts)

        T = self.key(total)
        if T is None:
            return None 

        self.prepare(T)                               # capture if new (padding-only warm-up)
        graph = self.graphs.get(T)

        buf, pad = self._bufs[T], self.runner.cache.pad_row
        real = PrefillBatch.build(slots, lens, self.runner.device)
        b = buf['batch']

        ids = torch.tensor([t for p in prompts for t in p], dtype=torch.long, device=self.runner.device)
        buf['idx'][0, :total].copy_(ids);          buf['idx'][0, total:].zero_()
        b.rows[:total].copy_(real.rows);           b.rows[total:].fill_(pad)
        b.positions[:total].copy_(real.positions); b.positions[total:].zero_()
        b.slots_gpu[:n].copy_(real.slots_gpu);      b.slots_gpu[n:].fill_(pad)
        b.lens_gpu[:n].copy_(real.lens_gpu);       b.lens_gpu[n:].zero_()
        b.last[:n].copy_(real.last);               b.last[n:].zero_() 

        logits = graph.replay()[0, :n]                            # (n, vocab): real prompts
        return sampling_params.sampler()(logits[:, :self.runner.enc.n_vocab])     # (n, 1)
    