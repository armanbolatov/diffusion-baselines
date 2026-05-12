"""Consistency Training (CT) — Song et al. 2023, one-step image generation.

CT trains a network f_θ(x, σ) to satisfy the boundary condition
    f_θ(x, σ_min) = x
and the self-consistency property
    f_θ(x_{σ_i}, σ_i) ≈ f_θ̃(x_{σ_{i-1}}, σ_{i-1})
where θ̃ is an EMA of θ and x_{σ_i} = x_0 + σ_i · noise.

EDM-style sigma schedule:
    σ_i = (σ_max^{1/ρ} + i/(N-1) · (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ,  i = 0..N-1
Forward:
    x_σ = x_0 + σ · noise

Loss:
    L_CT(θ, θ̃) = E [ d( f_θ(x_{σ_i}, σ_i), f_θ̃(x_{σ_{i-1}}, σ_{i-1}) ) ]
with d = pseudo-Huber (sqrt(‖·‖² + c²) − c).

Sampling (one step):
    x_hat = f_θ(σ_max · noise, σ_max)
"""
import math

import torch
import torch.nn.functional as F

from ..utils.ema import make_ema, ema_update

__all__ = [
    "SCHEME", "sigma_schedule", "consistency_fn", "pseudo_huber",
    "loss", "eval_loss", "sample_one_step", "make_ema", "ema_update",
]

SCHEME = "consistency"


def sigma_schedule(sigma_min: float, sigma_max: float, rho: float, n: int, device):
    """EDM noise schedule on n+1 levels (σ_0 = σ_min, σ_n = σ_max)."""
    i = torch.arange(n + 1, device=device, dtype=torch.float32)
    s = (sigma_max ** (1 / rho) + i / n * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return s


def consistency_fn(model, x, sigma, sigma_min: float, sigma_data: float = 0.5):
    """EDM preconditioning so f_θ(x, σ) satisfies the boundary at σ=σ_min."""
    sigma = sigma.view(-1, *([1] * (x.dim() - 1)))
    c_skip = sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
    c_out  = sigma_data * (sigma - sigma_min) / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_in   = 1.0 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = 0.25 * torch.log(sigma.clamp(min=1e-8))
    # cast to UNet's integer time-embedding domain
    t_int = (c_noise.flatten() * 1000.0 + 500.0).long().clamp(0, 999)
    out = model(c_in * x, t_int).sample
    return c_skip * x + c_out * out


def pseudo_huber(x, y, c: float = 0.03):
    return torch.sqrt(((x - y) ** 2).mean(dim=list(range(1, x.dim()))) + c * c) - c


def loss(model, ema_model, x0, sigma_min: float, sigma_max: float, rho: float, n_steps: int) -> torch.Tensor:
    """One CT step: pick adjacent (σ_{i-1}, σ_i) on the schedule, train consistency."""
    B = x0.size(0)
    device = x0.device
    sigmas = sigma_schedule(sigma_min, sigma_max, rho, n_steps, device)
    # pick i ~ Uniform(1, n) per sample
    idx = torch.randint(1, n_steps + 1, (B,), device=device)
    s_hi = sigmas[idx]
    s_lo = sigmas[idx - 1]
    noise = torch.randn_like(x0)
    s_hi_b = s_hi.view(-1, *([1] * (x0.dim() - 1)))
    s_lo_b = s_lo.view(-1, *([1] * (x0.dim() - 1)))
    x_hi = x0 + s_hi_b * noise
    x_lo = x0 + s_lo_b * noise
    f_hi = consistency_fn(model, x_hi, s_hi, sigma_min)
    with torch.no_grad():
        f_lo = consistency_fn(ema_model, x_lo, s_lo, sigma_min)
    return pseudo_huber(f_hi, f_lo).mean()


@torch.no_grad()
def eval_loss(model, ema_model, eval_iter, n_batches: int,
              sigma_min: float, sigma_max: float, rho: float, n_steps: int) -> float:
    losses = []
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        losses.append(loss(model, ema_model, x0, sigma_min, sigma_max, rho, n_steps).item())
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def sample_one_step(model, shape, sigma_max: float, sigma_min: float, device="cuda"):
    """One-step generation: f_θ(σ_max · noise, σ_max)."""
    noise = torch.randn(shape, device=device) * sigma_max
    sigma = torch.full((shape[0],), sigma_max, device=device)
    out = consistency_fn(model, noise, sigma, sigma_min)
    return out.clamp(-1, 1)
