import time

import torch

from inference.base_engine import BaseEngine
from inference.sampling import SamplingParams, sample
from inference.static_batch.decode_step import DecodeStep
from inference.static_batch.kv_cache import KVCache


class GptInference(BaseEngine):
    """Static batching: every prompt in one batch, left-padded, run to the end together."""

    def __init__(self, model, enc, config, device='cpu'):
        self.model = model.eval()
        self.enc = enc
        self.config = config
        self.device = device
        self.dtype = next(self.model.parameters()).dtype

        self.last_stats = None
        self.decode_steps = {}
        self.use_graph = str(device).startswith('cuda')

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Load a checkpoint and build the engine. fp32, as this engine always ran."""
        from model.gpt2.loading import load_model
        model, enc, config = load_model(path, device, dtype=torch.float32)
        return cls(model, enc, config, device)

    @torch.inference_mode()
    def generate(self, prompts, params=None):
        params = params or SamplingParams()
        self.sample(list(prompts), max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature, top_k=params.top_k, top_p=params.top_p)
        return [self._completion(t) for t in self._last_generated]

    def _decode_step(self, B, W):

        key = (B, W)
        if key not in self.decode_steps: 
            self.decode_steps[key] = DecodeStep(
                self.model, self.config, B, W, self.device, self.dtype) 
        return self.decode_steps[key]

    @torch.inference_mode()
    def sample(self, context, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):

        if context is None:
            return  
        
        single = isinstance(context, str)
        if single: 
            context = [context]

        B = len(context)
        eot = self.enc.eot_token

        tokens = [self.enc.encode(c) for c in context]
        L = max(len(s) for s in tokens)

        if self.use_graph: 
            B_cap = next(b for b in (1, 2, 4, 8, 16) if b >= B)
            W_cap = next(w for w in (128, 256, 512, 1024) if w >= L + max_new_tokens)
            step = self._decode_step(B_cap, W_cap)
            step.reset() 
            cache, mask = step.cache, step.mask 
        else: 
            step = None 
            B_cap, W_cap = B, L + max_new_tokens
            cache = KVCache(batch_size=B_cap, device=self.device, config=self.config, max_tokens=W_cap, dtype=self.dtype)
            mask = torch.zeros(B_cap, W_cap, dtype=torch.bool, device=self.device)


        idx = torch.full((B_cap, W_cap), eot, dtype=torch.long, device=self.device)

        for i, s in enumerate(tokens):
            idx[i, L-len(s):L] = torch.tensor(s, device=self.device)
            mask[i, L-len(s):L] = True 
       
        self._sync() 
        t0 = time.perf_counter()

        prefill_pos = (mask.cumsum(1)-1).clamp(min=0)
        logits, _ = self.model(idx[:, :L], kv_cache=cache, pos_ids=prefill_pos[:, :L], attn_mask=mask)

        self._sync() 
        prefill_s = time.perf_counter() - t0 

        done = torch.zeros(B_cap, dtype=torch.bool, device=self.device)
        done[B:] = True 
        row_pos = mask.sum(1) - 1
        steps = 0 

        self._sync() 
        t0 = time.perf_counter() 

        STOP_CHECK_FRE = 16
        for i in range(max_new_tokens): 

            next_digits = logits[:, -1, :self.enc.n_vocab]

            next_token = self.sample_next_token(next_digits, temperature=temperature, top_k=top_k, top_p=top_p)
            next_token = next_token.masked_fill(done.unsqueeze(1), eot)

            cols = L + steps 

            mask[:, cols] =(~done)
            idx[:, cols]= next_token.squeeze(1)

            steps +=1 

            done = done | (next_token.squeeze(1) == eot)
            if i == max_new_tokens -1: 
                break 

            if (i + 1) % STOP_CHECK_FRE ==0 and done.all(): 
                break 

            row_pos +=1 

            if step is None: 
                logits, _ = self.model(next_token, kv_cache=cache, pos_ids=row_pos.unsqueeze(1), attn_mask=mask)
            else: 
                step.token.copy_(next_token)
                step.pos.copy_(row_pos.unsqueeze(1))
                step.graph.replay() 
                logits = step.logits

        self._sync() 
        decode_s = time.perf_counter() - t0 

        self.last_stats = {
            'batch': B, 
            'prompt_tokens': B * L, 
            'generate_tokens': B * steps, 
            'prefill_s': prefill_s, 
            'decode_s': decode_s, 
            'prefill_token_s': B * L / prefill_s if prefill_s else float('inf'), 
            'decode_token_s': B * steps / decode_s if decode_s else float('inf'), 
            'ms_per_step':  1000 * decode_s / steps if steps else 0.0
        }

        outputs = []
        self._last_generated = []
        for i in range(B):

            row = idx[i][mask[i]]
            outputs.append(self.enc.decode(row.tolist())) 
            gen = idx[i, L:L + steps][mask[i, L:L + steps]]
            self._last_generated.append(gen.tolist())

        return outputs[0] if single else outputs


    def sample_next_token(self, logits, temperature=1.0, top_k=None, top_p=None):
        return sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    def _sync(self): 

        if self.device.startswith('cuda'):
            torch.cuda.synchronize() 
        elif self.device.startswith('mps'):
            torch.mps.synchronize() 
