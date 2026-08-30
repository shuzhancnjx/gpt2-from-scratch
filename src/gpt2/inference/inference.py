import torch 
import tiktoken
from gpt2.inference.kv_cache import KVCache
from gpt2.gpt import GPT, GPTConfig

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
        self.model.eval() 
        self.enc = tiktoken.get_encoding('gpt2')

    @torch.inference_mode()
    def sample(self, context, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):

        if context is None:
            return  
        single = isinstance(context, str)
        if isinstance(context, str): 
            context = [context]

        B = len(context)

        tokens = [self.enc.encode(c) for c in context]
        L = max(len(s) for s in tokens)

        idx = torch.full((B, L), self.enc.eot_token, dtype=torch.long, device=self.device)
        mask = torch.zeros(B, L, dtype=torch.bool, device=self.device)

        for i, s in enumerate(tokens):
            idx[i, L-len(s):] = torch.tensor(s, device=self.device)
            mask[i, L-len(s):] = True 
       
        cache = KVCache(batch_size=idx.size(0), config=self.config, device=self.device)

        prefill_pos = (mask.cumsum(1)-1).clamp(min=0)
        logits, _ = self.model(idx, kv_cache=cache, pos_ids=prefill_pos, attn_mask=mask)

        eot = self.enc.eot_token
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        row_pos = mask.sum(1) - 1
        for i in range(max_new_tokens): 

            next_digits = logits[:, -1, :self.enc.n_vocab]

            next_token = self.sample_next_token(next_digits, temperature=temperature, top_k=top_k, top_p=top_p)
            next_token = next_token.masked_fill(done.unsqueeze(1), eot)

            mask = torch.cat([mask, (~done).unsqueeze(1)], dim=1)
            idx = torch.cat([idx, next_token], dim=1)

            done = done | (next_token.squeeze(1) == eot)
            if done.all() or i == max_new_tokens - 1: 
                break 

            row_pos +=1 
            logits, _ = self.model(next_token, kv_cache=cache, pos_ids=row_pos.unsqueeze(1), attn_mask=mask)

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

        probs = torch.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1)
















        