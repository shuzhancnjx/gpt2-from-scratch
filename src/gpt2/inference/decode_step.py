import torch 
from gpt2.inference.kv_cache import KVCache

class DecodeStep: 

    def __init__(self, model, config, B, W, device, dtype): 

        self.B, self.W = B, W 

        self.cache = KVCache(B, device, config, max_tokens= W, dtype=dtype)

        self.token = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.mask = torch.zeros(B, W, dtype=torch.bool, device=device)

        s = torch.cuda.Stream() 
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s): 

            for _ in range(3): 
                model(self.token, kv_cache=self.cache, pos_ids=self.pos, attn_mask=self.mask)

        torch.cuda.current_stream().wait_stream(s)

        self.cache.reset() 

        self.graph = torch.cuda.CUDAGraph() 
        with torch.cuda.graph(self.graph): 

            self.logits, _ = model(self.token, kv_cache=self.cache, 
                                   pos_ids=self.pos, attn_mask=self.mask)

    def reset(self): 
        self.cache.reset()
        self.mask.zero_() 
