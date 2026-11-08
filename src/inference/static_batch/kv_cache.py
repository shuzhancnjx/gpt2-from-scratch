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

    def write(self, layer_idx, key, value, prefill=None):
        # ONE shared pointer: every row writes at the same position (left-padded batch).
        assert prefill is None, 'KVCache writes every row at one shared pos; packed prefill needs a per-row cache'
        T = key.size(2)
        cols = self.pos if T == 1 else self.pos + torch.arange(T, device=self.pos.device)
        self.key[layer_idx].index_copy_(2, cols, key)
        self.value[layer_idx].index_copy_(2, cols, value)

    def read(self, layer_idx, num_rows):
        return self.key[layer_idx], self.value[layer_idx]

    def prefill_attend(self, layer_idx, query, key, value, prefill):
        raise NotImplementedError('KVCache has one shared pointer: it prefills a left-padded '
                                  'batch through decode_attend with a mask, not packed prompts')

    def decode_attend(self, layer_idx, query, key, value, attn_mask):
        """Store the new tokens at the shared position, attend over the batch's buffers
        (the mask hides padding and unwritten columns)."""
        self.write(layer_idx, key, value)
        return self.sdpa_decode(layer_idx, query, attn_mask)

    def seq_len(self):
        return int(self.pos)

    def advance(self, T, prefill=None, active=None):
        assert prefill is None and active is None, \
            'KVCache advances every row together; use RowKVCache for per-row pointers'
        self.pos += T

    def reset(self): 
        self.pos.zero_()
        for k, v in zip(self.key, self.value): 
            k.zero_(); v.zero_() 
    

        