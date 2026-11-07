import torch 
from model.gpt2.gpt import GPTConfig
from inference.base_kv_cache import BaseKVCache

class KVCache(BaseKVCache):

    def __init__(self, batch_size, device, config: GPTConfig, max_tokens=None, dtype=torch.float32): 

        W = max_tokens if max_tokens != None else config.block_size
        shape = (batch_size, config.n_head, W, config.n_embd // config.n_head)
        self.key = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.value = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.width = W 

        self.pos = torch.zeros(1,  dtype=torch.long, device=device)

    def update(self, layer_idx,  key, value, row=None): 

        # This cache has ONE shared pointer; per-row writes need RowKVCache.
        assert row is None, 'KVCache writes every row at one shared pos; use RowKVCache'

        assert layer_idx is not None, "layer_idx should not be none when kv_cache is passed"

        T = key.size(2)

        cols = self.pos if T == 1 else self.pos + torch.arange(T, device=self.pos.device)
      
        self.key[layer_idx].index_copy_(2, cols, key)
        self.value[layer_idx].index_copy_(2, cols, value)

        return self.key[layer_idx], self.value[layer_idx]

    def seq_len(self): 
        return int(self.pos) 

    def advance(self, T, row=None, active=None): 
        assert row is None and active is None, \
            'KVCache advances every row together; use RowKVCache for per-row pointers'
        self.pos += T

    def reset(self): 
        self.pos.zero_()
        for k, v in zip(self.key, self.value): 
            k.zero_(); v.zero_() 
    

        