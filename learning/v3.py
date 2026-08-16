import torch 
from torch import nn
from torch.nn import functional as F 

bach_size = 64
block_size = 256
dropout = 0.2 
device = 'cpu'


class AttentionHead(nn.Module): 

    def __init__(self, head_size, n_embedding): 
        super().__init__() 

        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)

        self.register_buffer('tril',  torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 

        B, T, C = x.shape 

        key = self.key(x) # B, T, C @ C, H -> B T H
        query = self.query(x) # B, T, C @ C, H -> B T H

        wei = query @ key.transpose(-2, -1)  # B T H @ B H T -> B T T 
        wei = wei * (key.size(-1)**-0.5)  

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # B T C -> B T H
        out = wei @ v  # B T T @  B  T H -> B T H 
        return out 

class MultiHeadsAttention(nn.Module): 

    def __init__(self, n_head, head_size, n_embedding): 
        super().__init__() 

        self.attention_heads = nn.ModuleList([AttentionHead(head_size, n_embedding) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 

        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out 

class MLP(nn.Module): 

    def __init__(self, n_embedding): 
        super().__init__()

        self.net = nn.Sequential(
            SwiGLU(n_embedding, 4 * n_embedding), 
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        out = self.net(x)
        return out 

class SwiGLU(nn.Module): 

    def __init__(self, n_embedding, hidden_dim): 
        super().__init__() 

        self.w1 = nn.Linear(n_embedding, hidden_dim)
        self.w2 = nn.Linear(n_embedding, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, n_embedding)

    def forward(self, x): 
        gate = self.w1(x)
        value = self.w2(x)

        x = F.silu(gate) * value
        out = self.w3(x)
        return out 

class Block(nn.Module):

    def __init__(self, n_head, head_size, n_embedding): 
        super().__init__() 

        self.multi_heads = MultiHeadsAttention(n_head=n_head, head_size=head_size, n_embedding=n_embedding)
        self.ffd = MLP(n_embedding)

        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x): 
        x = x + self.multi_heads(self.ln1(x)) 
        x = x + self.ffd(self.ln2(x)) 
        return x 

class Transformer(nn.Module): 

    def __init__(self, n_embedding, head_size, n_head, n_layer, vocab_size, block_size): 

        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding, device=device)
        self.pos_embedding_table = nn.Embedding(block_size, n_embedding, device=device)

        self.blocks = nn.Sequential(*[Block(n_head, head_size, n_embedding) for _ in range(n_layer)])

        self.ln = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None): 

        B, T = idx.shape

        token = self.token_embedding_table(idx) # B T C
        pos = self.pos_embedding_table(torch.arange(T, device=device)) # T C 
        x = token + pos

        logits = self.blocks(x) # B T C 
        logits = self.ln(logits) # B T C 
        logits = self.lm_head(logits) # B T C @ B C V -> B T V 

        if targets is None: 
            loss = None 
        else: 
            B, T, V = logits.shape 

            logits_flat = logits.view(B *T, V)
            targets_flat  = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_view_token, temperature, topk=None): 

        B, T = idx.shape

        cond_idx = idx if idx.size(-1) < max_view_token else idx[:, -block_size:]

        logits, _ = self(cond_idx) # B T V 

        logits = logits[:, -1, :] # B V 

        if topk is not None: 

            v, _ = torch.topk(logits, topk) # B topK
            logits[logits < v[:, [-1]]] = float('-inf') # B V 

        prob = F.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(prob, num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)

        return idx 



        


