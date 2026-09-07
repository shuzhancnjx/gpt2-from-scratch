import torch 
import tiktoken
from gpt2.inference.kv_cache import KVCache
from gpt2.gpt import GPT, GPTConfig
from gpt2.inference.decode_step import DecodeStep
import time 

class GptInference: 

    def __init__(self, model_url, device='cpu'):

        # load model
        self.device = device
        checkpoint = torch.load(model_url,
                                map_location=device, weights_only=True)

        self.config = GPTConfig(**checkpoint['config'])
        self.model = GPT(self.config)

        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)

        self.dtype = next(self.model.parameters()).dtype
        
        self.model.eval()

        self.enc = tiktoken.get_encoding('gpt2')

        self.last_stats = None

        self.decode_steps = {}
        self.use_graph = str(device).startswith('cuda')

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
        for i in range(B):

            row = idx[i][mask[i]]
            outputs.append(self.enc.decode(row.tolist())) 

        return outputs[0] if single else outputs


    def sample_next_token(self, logits, temperature=1.0, top_k=None, top_p=None): 

        if temperature == 0: 
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None: 
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)

        if top_p is not None: 
            return self.top_p_sampling(logits=logits, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1)


    def top_p_sampling(self, logits, top_p):

        # The top token's exclusive mass is 0, so `0 >= top_p` is True once top_p
        # hits 0 and every token gets dropped. Treat it as greedy, the way
        # temperature == 0 is handled.
        if top_p <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        sorted_logits, sorted_idx = torch.sort(logits,  dim=-1, descending=True)

        probs = sorted_logits.softmax(dim=-1)

        drop = probs.cumsum(-1) - probs >= top_p

        probs[drop] = 0 

        next_token_pos = torch.multinomial(probs, num_samples=1)

        return torch.gather(sorted_idx, -1, next_token_pos)

    def _sync(self): 

        if self.device.startswith('cuda'):
            torch.cuda.synchronize() 
        elif self.device.startswith('mps'):
            torch.mps.synchronize() 
