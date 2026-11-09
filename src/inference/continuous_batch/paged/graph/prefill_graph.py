"""The packed prefill as CUDA graphs -- one per padded token count (CUDA only).

An eager prefill forward costs ~10 ms of kernel launches whatever its size. A graph
replays all of them with one launch. A graph needs fixed shapes, so:

    tokens   pad sum(lens) up to a bucket T (BUCKETS)
    prompts  always N = runner.max_prefill_prompts entries; unused entries are dummies
    padding  padding tokens use the cache's pad row (all SCRATCH): they write only the
             scratch block and attend one scratch slot -- their output is never read

Real tokens are computed exactly as in the eager prefill: each attends only within its
own prompt (prefill_attention.py) and writes its k/v into its own request's blocks.

The indices themselves are built on the CPU in a pinned buffer and moved with one async
copy per prefill (_buffers, run).
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
        """The fixed tensors bucket T's graph reads, all views of ONE device tensor `flat`:

            idx        (1, T)  packed token ids (padding: 0)
            batch      a PrefillBatch whose tensors are views of flat:
                         rows (T,)   positions (T,)   slots_gpu (N,)  lens_gpu (N,)  last (N,)
                       padding: rows / slots_gpu = cache.pad_row, the rest 0. Its CPU
                       lists (slots, lens) are dummies -- the kernels read only tensors.
            flat       the device tensor: [idx | rows | positions | slots | lens | last]
            staging    flat's pinned host twin: run() builds a batch's indices there and
                       moves them with ONE async copy (not ~25 small kernels + 3 blocking
                       copies, ~0.8 ms per prefill on the A100)
            copied     an event after that copy: staging is not rewritten before it lands
        """
        cache, device, N = self.runner.cache, self.runner.device, self.runner.max_prefill_prompts
        pad = cache.pad_row

        flat = torch.zeros(3 * T + 3 * N, dtype=torch.long, device=device)
        idx, rows, positions, slots_gpu, lens_gpu, last = flat.split([T, T, T, N, N, N])
        rows.fill_(pad)
        slots_gpu.fill_(pad)
        return dict(
            idx=idx.view(1, T),
            batch=PrefillBatch(slots=[pad] * N, lens=[0] * N, slots_gpu=slots_gpu,
                               lens_gpu=lens_gpu, rows=rows, positions=positions, last=last),
            flat=flat,
            staging=torch.empty(flat.shape, dtype=torch.long, pin_memory=True),
            copied=torch.cuda.Event())

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

        buf, pad, N = self._bufs[T], self.runner.cache.pad_row, self.runner.max_prefill_prompts

        # build every index on the CPU, in the pinned staging buffer (padding included) ...
        buf['copied'].synchronize()                   # the previous batch's copy has landed
        idx, rows, positions, slots_cpu, lens_cpu, last = buf['staging'].split([T, T, T, N, N, N])
        lens_t, slots_t = torch.tensor(lens), torch.tensor(slots)
        starts = torch.cumsum(lens_t, 0) - lens_t                       # where each prompt begins
        idx[:total] = torch.tensor([t for p in prompts for t in p]);   idx[total:] = 0
        rows[:total] = torch.repeat_interleave(slots_t, lens_t);       rows[total:] = pad
        positions[:total] = torch.arange(total) - torch.repeat_interleave(starts, lens_t)
        positions[total:] = 0
        slots_cpu[:n] = slots_t;                 slots_cpu[n:] = pad
        lens_cpu[:n] = lens_t;                   lens_cpu[n:] = 0
        last[:n] = starts + lens_t - 1;          last[n:] = 0

        # ... then one async host -> device copy into the graph's buffers, and replay
        buf['flat'].copy_(buf['staging'], non_blocking=True)
        buf['copied'].record()
        logits = graph.replay()[0, :n]                            # (n, vocab): real prompts
        return sampling_params.sampler()(logits[:, :self.runner.enc.n_vocab])     # (n, 1)
    