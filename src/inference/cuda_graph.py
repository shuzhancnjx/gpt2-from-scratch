"""CUDA graphs: record a step once, replay it with one launch.

Shared by every captured step in the engines:

    decode   DecodeGraphs        one graph per sampler setting   (paged/graph/decode_graph.py)
    prefill  PrefillGraphs       one graph per padded token count (paged/graph/prefill_graph.py)
    (static_batch.DecodeStep follows the same recipe)

Three layers, all here:

    CapturedGraph   record a body once, replay it with one launch
    GraphCache      one CapturedGraph per key, captured on first use
    BaseGraphs      the contract a phase's graphs follow -- key / prepare / run -- over a
                    GraphCache. PrefillGraphs and DecodeGraphs implement it:

              key                    run(...)                                returns
    prefill   padded token count     run(slots, prompts, sampling_params)    first tokens, or None
    decode    sampler setting        run(sampling_params)                    next tokens

Both return sampled tokens. Decode records its sampler inside the graph (so the key is the
sampler setting); prefill's graph returns logits and run() samples them after the replay
(so one graph per bucket serves every setting).

Each user keeps its own fixed buffers and the body that reads them.

Rules for a body (what makes it capturable):
  - reads and writes only FIXED tensors, updated in place between replays
    (copy_ / fill_ / zero_ / index writes -- never rebinding a name to a new tensor)
  - no CPU reads of GPU values inside (.item(), int(t), .tolist(), `if` on a tensor)
  - fixed shapes: everything the body touches has the same shape on every replay
  - returns its output tensor(s); they live in the graph's memory pool and are
    overwritten by the next replay of the same graph
"""

from abc import ABC, abstractmethod

import torch


class CapturedGraph:
    """One recorded body.

    body:    a zero-argument callable obeying the rules above; returns the output.
    warmup:  how many times to run body on a side stream before recording, so kernel
             selection, Triton compilation and workspace allocation happen outside it.
    """

    def __init__(self, body, warmup=2):
        #  1. warm up: run body() `warmup` times on a side CUDA stream (the side stream
        #     waits for the current one first; the current one waits for it after)
        #  2. record: inside torch.cuda.graph(...), call body() once and keep its result
        #     as self.output
        #  3. keep the torch.cuda.CUDAGraph as self.graph
        #  both runs under torch.inference_mode()
        
        side = torch.cuda.Stream() 
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode(): 
            for _ in range(warmup): 
                body() 
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph() 
        with torch.cuda.graph(self.graph), torch.inference_mode(): 
            self.output = body() 
        

    def replay(self):
        """Run the recorded work with one launch; return self.output (now updated)."""
        self.graph.replay()
        return self.output 
    

class GraphCache:
    """One CapturedGraph per key, captured the first time the key is asked for.

    build(key) returns the body for that key (and creates any buffers it needs).
    Keys: decode uses the sampler setting, prefill the padded token count.
    """

    def __init__(self, build):
        # TODO: keep build; start with no graphs
        self.build = build 
        self.graphs = {}

    def get(self, key):
        """The graph for key -- captured now if this is its first use."""

        if key not in self.graphs:
            self.graphs[key] = CapturedGraph(self.build(key))

        return self.graphs[key]

    def __len__(self):
        """How many graphs have been captured (tests use it)."""
        return len(self.graphs)


class BaseGraphs(ABC):

    def __init__(self, runner):
        assert str(runner.device).startswith('cuda'), f'{type(self).__name__} needs CUDA'
        self.runner = runner                         # the ModelRunner: model, cache, buffers
        self.graphs = GraphCache(self._build)

    @abstractmethod
    def key(self, *args):
        """The key of the graph a step needs (hashable)."""

    @abstractmethod
    def _build(self, key):
        """GraphCache's build(): create the key's buffers if it has its own, return the
        zero-argument body to capture."""

    @abstractmethod
    def run(self, *args):
        """One step: fill the buffers (if any), replay the key's graph, return its output."""

    @abstractmethod
    def prepare(self, key):
        """Capture the graph for key if it is new. Each phase states when that is safe:
        prefill at any time (its warm-up runs on padding only); decode between runs, with
        the runner reset around it (its warm-up runs the real step)."""

    def __len__(self):
        return len(self.graphs)
