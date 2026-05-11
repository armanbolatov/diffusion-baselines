"""Decoder-only transformer with AdaLN time conditioning, for discrete diffusion.

Inputs:
    x: (B, T) int64 token ids (may contain MASK token)
    t: (B,) float in [0, 1] — diffusion time
Outputs:
    logits: (B, T, V)

Architecture:
    - Token embedding (V -> D)
    - Sinusoidal positional embedding (T -> D)
    - Sinusoidal time embedding -> 2-layer MLP -> AdaLN params
    - N transformer blocks: pre-AdaLN → self-attention (causal=False) → AdaLN → MLP
    - Final LayerNorm → linear head to V

Bidirectional self-attention (no causal mask): standard for masked diffusion.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0, 1] → (B, dim) sinusoidal features."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )
    a = t[:, None] * freqs[None, :] * 1000.0  # scale t up so sinusoids vary meaningfully
    return torch.cat([a.sin(), a.cos()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive LayerNorm: scale, shift = MLP(time_emb)."""
    def __init__(self, d_model, time_emb_dim):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(time_emb_dim, 2 * d_model)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, t_emb):
        # x: (B, T, D), t_emb: (B, time_emb_dim)
        scale, shift = self.to_scale_shift(t_emb).chunk(2, dim=-1)
        return self.ln(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, time_emb_dim, dropout=0.0):
        super().__init__()
        self.attn_norm = AdaLN(d_model, time_emb_dim)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.mlp_norm = AdaLN(d_model, time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, t_emb):
        h = self.attn_norm(x, t_emb)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.mlp_norm(x, t_emb)
        x = x + self.mlp(h)
        return x


class DiffTransformer(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 256,
                 n_layer: int = 12, n_head: int = 6, d_model: int = 384,
                 dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        time_emb_dim = 4 * d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self._d_model = d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, time_emb_dim, dropout)
            for _ in range(n_layer)
        ])
        self.final_norm = AdaLN(d_model, time_emb_dim)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.token_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: (B, T) int, t: (B,) float
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :T]
        t_features = _sinusoidal_time_embedding(t, self._d_model)
        t_emb = self.time_mlp(t_features)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.final_norm(h, t_emb)
        return self.head(h)
