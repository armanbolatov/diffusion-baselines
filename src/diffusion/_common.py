"""Shared utilities for discrete diffusion schemes."""
import torch


def sample_t_antithetic(n: int, device, eps: float) -> torch.Tensor:
    """Stratified (antithetic) t ~ Uniform(eps, 1) across a batch of size n.

    Each example sees t in its own evenly-spaced stratum of [eps, 1]; the
    marginal is unchanged, only per-batch variance drops by ~n×. Used by
    MDLM loss/eval.
    """
    u = torch.rand(n, device=device)
    offset = torch.arange(n, device=device, dtype=u.dtype) / n
    u = (u / n + offset) % 1.0
    return eps + (1.0 - eps) * u
