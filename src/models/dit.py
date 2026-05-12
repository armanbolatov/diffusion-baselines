"""DiT — Diffusion Transformer (Peebles & Xie 2023), slim CIFAR-32 variant.

Returns a wrapper with the same interface as diffusers UNet2DModel:
    model(x, t)            -> namespace with .sample tensor of same shape as x
where x is (B, C, H, W) and t is (B,) of long or float.

DiT-S/2 defaults: hidden=384, depth=12, heads=6, patch_size=2.
On 32x32 input this gives 256 tokens.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass


def _sin_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None, :]
    return torch.cat([a.sin(), a.cos()], dim=-1)


def _modulate(x, scale, shift):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s_msa, g_msa, sc_msa, s_mlp, g_mlp, sc_mlp = self.adaLN(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), s_msa, g_msa)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + sc_msa[:, None, :] * a
        h = _modulate(self.norm2(x), s_mlp, g_mlp)
        x = x + sc_mlp[:, None, :] * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=True))
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), scale, shift))


@dataclass
class _UNetLike:
    sample: torch.Tensor


class DiT(nn.Module):
    def __init__(self, sample_size=32, in_channels=3, out_channels=3,
                 patch_size=2, hidden=384, depth=12, heads=6):
        super().__init__()
        assert sample_size % patch_size == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.n_tokens = (sample_size // patch_size) ** 2

        # patchify via Conv2d
        self.patch_embed = nn.Conv2d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # time conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.final  = FinalLayer(hidden, patch_size, out_channels)
        self._hidden = hidden
        self._sample_size = sample_size

    def _unpatchify(self, x):
        p = self.patch_size
        gh = gw = self._sample_size // p
        # x: (B, gh*gw, p*p*C) -> (B, C, H, W)
        return rearrange(x, "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
                         gh=gh, gw=gw, ph=p, pw=p, c=self.out_channels)

    def forward(self, x: torch.Tensor, t) -> _UNetLike:
        # x: (B, C, H, W); t: (B,) int or float
        if t.dtype != x.dtype:
            t = t.float()
        h = self.patch_embed(x)                        # (B, hidden, H/p, W/p)
        h = rearrange(h, "b c h w -> b (h w) c")        # (B, n_tokens, hidden)
        h = h + self.pos_emb
        c = self.t_mlp(_sin_time_embedding(t, self._hidden))
        for blk in self.blocks:
            h = blk(h, c)
        h = self.final(h, c)                            # (B, n_tokens, p*p*C)
        x_out = self._unpatchify(h)
        return _UNetLike(sample=x_out)


def build_dit(sample_size=32, in_channels=3, out_channels=3,
              patch_size=2, hidden=384, depth=12, heads=6):
    return DiT(sample_size=sample_size, in_channels=in_channels,
               out_channels=out_channels, patch_size=patch_size,
               hidden=hidden, depth=depth, heads=heads)
