import tiktoken
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)

batch_size = 32
block_size = 8 
head_size = 256 
max_iters = 100
learning_rate = 1e-3
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
eval_iters = 200 
n_embd = 32
n_layer = 6
n_head = 6
dropout = 0.2

# read data
with open('/Users/zhanshu/Desktop/llm learning/data/input.txt', 'r', encoding='utf-8') as f: 
    text = f.read() 

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) 

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split): 

    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size, ), device=device)
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size + 1] for  i in ix])
    return x, y 

@torch.no_grad()
def estimate_loss(): 

    out = {}
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean() 
    model.train() 
    return out 
 
class Head(nn.Module): 

    def __init__(self, head_size): 

        super().__init__() 

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 

        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        wei = query @ key.transpose(-2, -1) / key.size(-1)**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) 
        return out

class FeedForward(nn.Module): 

    def __init__(self, n_embd): 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head): 

        super().__init__() 
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x 
        
class BigramLanguageModel(nn.Module): 
    def __init__(self):

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(block_size,  n_embd, device=device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape 

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None 
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss 

    @torch.no_grad()
    def generate(self, idx, max_view_tokens, temperature=1.0, top_k=None): 

        for _ in range(max_view_tokens): 

            idx_cond = idx if idx.size(1) < block_size else idx[:, -block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            if top_k is not None: 
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx  

model = BigramLanguageModel()

m = model.to(device)

optimzer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters): 

    if iter % eval_iters == 0: 
        losses = estimate_loss() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimzer.zero_grad(set_to_none=True)
    loss.backward() 
    optimzer.step() 

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_view_tokens=500)[0].tolist()))