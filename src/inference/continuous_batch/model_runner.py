"""ModelRunner: runs the model -- execution (GPU) for both continuous-batching engines.

No policy, no queue, and no idea which KV layout it drives: it calls the model on
whatever cache the engine hands it (RowKVCache or PagedKVCache). It owns everything the
model touches each step:

    the model and the cache
    next_token_gpu (B, 1), active_gpu (B,)      the decode rows' inputs
    PrefillGraphs / DecodeGraphs                 CUDA graphs (paged cache, cuda_graph=True, CUDA)

and offers the steps an engine calls, all in terms of cache rows:

    start(sampling_params)          before a run: capture every graph it may replay, then empty
    prefill(rows, prompts, sp)      one packed prefill into these rows -> first tokens (n, 1)
    apply(started, stopped)         decode rows that took a request (row, first token) / finished
    decode(sp)                      one decode step for every row -> next tokens (B, 1)

Nothing here reads the GPU back to the CPU: the engine does the one copy per step.
"""

import torch

from inference.base_kv_cache import PrefillBatch
from inference.continuous_batch.paged.graph.decode_graph import DecodeGraphs
from inference.continuous_batch.paged.graph.prefill_graph import PrefillGraphs


class ModelRunner:

    def __init__(self, model, enc, cache, max_seqs, max_tokens, device, cuda_graph=False,
                 max_prefill_prompts=None):
        """
        cache                the engine's KV cache; decode rows are its rows [0, max_seqs)
        max_tokens           W, the cache width the decode mask spans
        cuda_graph           capture prefill + decode as CUDA graphs (needs CUDA and a
                             PagedKVCache: the prefill graphs pad into its pad_row)
        max_prefill_prompts  most prompts one prefill packs (the prefill graphs' width)
        """
        self.model, self.enc, self.cache = model, enc, cache
        self.B, self.W, self.device = max_seqs, max_tokens, device
        self.max_prefill_prompts = max_prefill_prompts

        self.next_token_gpu = torch.zeros(self.B, 1, dtype=torch.long, device=device)
        self.active_gpu = torch.zeros(self.B, dtype=torch.bool, device=device)

        use_graphs = cuda_graph and str(device).startswith('cuda')
        self.decode_graphs = DecodeGraphs(self) if use_graphs else None
        self.prefill_graphs = PrefillGraphs(self) if use_graphs else None

    def start(self, sampling_params):
        """Before a run: capture every graph it may replay -- so no capture lands inside a
        served (or timed) step -- then back to empty. Decode: these sampler settings.
        Prefill: every bucket."""
        if self.decode_graphs is not None:
            self.decode_graphs.prepare(self.decode_graphs.key(sampling_params))
        if self.prefill_graphs is not None:
            for T in self.prefill_graphs.buckets:
                self.prefill_graphs.prepare(T)
        self.reset()

    def reset(self):
        """Back to empty, IN PLACE: captured graphs hold these buffers' addresses."""
        self.cache.reset()
        self.active_gpu.zero_()
        self.next_token_gpu.zero_()

    # -- prefill ----------------------------------------------------------------------

    def prefill(self, rows, prompts, sampling_params):
        """One packed prefill: prompts[i] (token ids) into cache row rows[i]. A graph replay
        when a bucket fits, else eager. -> first tokens (n, 1) on the device."""
        if self.prefill_graphs is not None:
            first = self.prefill_graphs.run(rows, prompts, sampling_params)
            if first is not None:                       # None: too big for any bucket
                return first
        batch = PrefillBatch.build(rows, [len(p) for p in prompts], self.device)
        idx = torch.tensor([t for p in prompts for t in p], dtype=torch.long,
                           device=self.device).unsqueeze(0)                 # (1, sum(lens))
        logits = self.prefill_logits(idx, batch)                           # (1, n, vocab)
        return sampling_params.sampler()(logits[0, :, :self.enc.n_vocab])  # (n, 1)

    def prefill_logits(self, idx, batch):
        """The packed prefill's model call: logits at each prompt's last token (1, n, vocab).
        Tensors in, tensors out -- PrefillGraphs records exactly this as its body."""
        logits, _ = self.model(idx, kv_cache=self.cache, pos_ids=batch.positions.unsqueeze(0),
                               prefill=batch)
        return logits

    # -- decode -----------------------------------------------------------------------

    def apply(self, started=(), stopped=()):
        """The decode rows' changes since the last step, in place.
        started: (row, first token) -- the token becomes the row's next input, the row decodes.
        stopped: rows whose request finished -- the row goes idle."""
        if started:
            rows, tokens = zip(*started)
            rows = torch.tensor(rows, device=self.device)
            self.next_token_gpu[rows, 0] = torch.tensor(tokens, device=self.device)
            self.active_gpu[rows] = True
        if stopped:
            self.active_gpu[torch.tensor(list(stopped), device=self.device)] = False

    def decode(self, sampling_params):
        """One decode step for every row: the graph for these settings, or eager.
        -> next_token_gpu (B, 1), the sampled tokens."""
        if self.decode_graphs is not None:
            return self.decode_graphs.run(sampling_params)
        return self.decode_eager(sampling_params)

    def decode_eager(self, sampling_params):
        """The decode step itself: model on the buffers, sample, tokens INTO next_token_gpu.
        Tensor ops only -- DecodeGraphs records exactly this as its body."""
        logits, _ = self.model(self.next_token_gpu, kv_cache=self.cache,
                               pos_ids=self.cache.seq_len().unsqueeze(1),
                               attn_mask=self.mask(), cache_active=self.active_gpu)
        sampler = sampling_params.sampler()
        return self.next_token_gpu.copy_(sampler(logits[:, -1, :self.enc.n_vocab]))

    def mask(self):
        """Which cache columns each decode row may attend to, derived from pos (never
        maintained, so it cannot drift from the cache). `<=`: the token written this step
        lands at column pos[r] and must see itself. `& active` blanks idle rows."""
        cols = torch.arange(self.W, device=self.device)
        return (cols <= self.cache.seq_len().unsqueeze(1)) & self.active_gpu.unsqueeze(1)
