import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class LayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def _apply_rope(self, x, seq_len, device):
        # x shape: (B, n_head, T, head_dim)
        dim = x.shape[-1]
        # Calculate frequencies: 10000^(-2i/dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)  # (T, dim/2)
        
        # Split into real and imaginary parts (simulated via pairs)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, T, dim)
        
        # Rotary transformation: x_rot = x * cos + x_perp * sin
        # where x_perp = [-x1, x0, -x3, x2, ...]
        x_perp = torch.empty_like(x)
        x_perp[..., 0::2] = -x[..., 1::2]
        x_perp[..., 1::2] = x[..., 0::2]
        
        return (x * cos) + (x_perp * sin)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Split heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Queries and Keys
        q = self._apply_rope(q, T, x.device)
        k = self._apply_rope(k, T, x.device)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

from transformers import GenerationMixin, GenerationConfig

class GPT2Modded(nn.Module, GenerationMixin):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(
                [
                    TransformerBlock(config.n_embd, config.n_head)
                    for _ in range(config.n_layer)
                ]
            ),
            "ln_f": LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig()
        
        # Proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        B, T = input_ids.size()
        
        # Token embeddings only (RoPE is applied inside Attention blocks)
        x = self.transformer.wte(input_ids)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tensors
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )
