import torch 
from gpt2.gpt import GPTConfig

class KVCache:

    def __init__(self, batch_size, device, config: GPTConfig, dtype=torch.float32): 

        shape = (batch_size, config.n_head, config.block_size, config.n_embd // config.n_head)
        self.key = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.value = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

        self.pos = 0 

    def update(self, layer_idx,  key, value): 

        assert layer_idx is not None, "layer_idx should not be none when kv_cache is passed"

        T = key.size(2)

        start, end = self.pos, self.pos + T
      
        self.key[layer_idx][:, :, start:end] = key
        self.value[layer_idx][:, :, start:end] = value

        return self.key[layer_idx][:, :, :end], self.value[layer_idx][:, :, :end]

    def seq_len(self): 
        return self.pos 

    def advance(self, token_size): 
        self.pos += token_size

        