import torch 
from torch import nn
from torch.nn import functional as F 

block_size = 8 

class MultiHeadAttention(nn.Module): 

    def __init__(self, n_head, head_size, n_embedding, dropout): 
        super().__init__() 

        self.n_head = n_head
        self.head_size = head_size
        self.n_embedding = n_embedding

        self.k_proj = nn.Linear(n_embedding, head_size * n_head, bias=False)
        self.q_proj = nn.Linear(n_embedding, head_size * n_head, bias=False)
        self.v_proj = nn.Linear(n_embedding, head_size * n_head, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size * n_head, n_embedding)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

    def forward(self, x,  kv_cache=None, use_cache=False): 

        B, T, C = x.shape 

        key = self.k_proj(x)
        query = self.q_proj(x)
        value = self.v_proj(x)

        key = key.view(B, T, self.n_head, self.head_size).transpose(1, 2) # B, n_head, T, head_size
        query = query.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if use_cache: 
            if kv_cache != None:
                k, v = kv_cache
                k = torch.cat([k, key], dim=2)
                v = torch.cat([v, value], dim=2)
            else: 
                k, v = key, value 

            new_cache = (k, v)
            wei = query @ k.transpose(-1, -2) # B, n_head, T, T 
            wei = wei / self.head_size**0.5 
            
            if T > 1: 
                past_length = k.size(2) - T
                mask = self.tril[past_length:past_length + T, :k.size(2)]
                wei = wei.mask_fill(mask==0, float('-inf'))
            
            

        else:
            wei = query @ key.transpose(-1, -2)
            wei = wei / self.head_size**0.5
            wei = wei.mask_fill(self.tril[:T, :T]==0, float('-inf'))
            new_cache = None 

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        value = v if use_cache else value 
        out = wei @ value 

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.head_size * self.n_head)

        out = self.proj(out)

        return out, new_cache