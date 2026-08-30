"""The GPT-2 model: config, blocks, and the top-level module."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.head_size = config.n_embd // config.n_head
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * self.head_size, bias=config.bias)
        self.c_proj = nn.Linear(config.n_head * self.head_size, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache=None,  layer_idx=None, attn_mask=None):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)

        k = k.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)

        if kv_cache is not None: 
            k, v = kv_cache.update(layer_idx, k, v)

        drop_out = self.config.dropout if self.training else 0.0
        if attn_mask is not None: 
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=drop_out,
            )
        else: 
            # flash attention: fuses the mask/softmax/dropout and never materializes T x T
            y = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=(q.size(2) == k.size(2)), # to distinguish prefill or one token decoding
                dropout_p=drop_out,
            )  # B, NH, T, H

        y = y.transpose(1, 2).contiguous().view(B, T, self.head_size * self.config.n_head)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, layer_idx=None, attn_mask=None):
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight  # sharing weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None, pos_ids=None, attn_mask=None):
        B, T = idx.shape

        past = 0 if kv_cache is None else kv_cache.seq_len() 
        assert past + T <= self.config.block_size, \
            f'cannot forward sequence of length {past + T}, block size {self.config.block_size}'

        token_embd = self.transformer.wte(idx)

        if pos_ids is None: 
            pos_ids = torch.arange(past, past + T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos_ids)

        sdpa_mask = None 
        if attn_mask is not None: 
            sdpa_mask = attn_mask[:, None, None, :]
            if past == 0: 
                tri = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
                sdpa_mask = sdpa_mask & tri

        x = token_embd + pos_embd

        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_cache=kv_cache, layer_idx=i, attn_mask=sdpa_mask)
        if kv_cache is not None: 
            kv_cache.advance(T)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, logits.size(-1)), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, vocab_limit=None):
        """Sample max_new_tokens continuations of idx (B, T) -> (B, T + max_new_tokens).

        vocab_limit truncates the logits: the head is usually padded past the
        tokenizer's real vocab, and those extra rows are not decodable.

        No KV cache yet -- this re-runs the full prefix each step, which is fine for
        short samples during training. The cached path lands in gpt2/inference/.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            window = idx[:, -self.config.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :vocab_limit] if vocab_limit else logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_pos = torch.topk(probs, top_k, dim=-1)
            choice = torch.multinomial(topk_probs, num_samples=1)
            idx = torch.cat([idx, torch.gather(topk_pos, -1, choice)], dim=1)

        if was_training:
            self.train()
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        non_decay = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                 fused=(device_type == 'cuda'))

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print('Loading weights from pretrained gpt: %s' % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['dropout'] = 0.0
        config_args['bias'] = True

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), \
            f'mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}'

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
