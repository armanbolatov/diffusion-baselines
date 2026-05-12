"""MDLM (Masked Diffusion Language Model) — absorbing-state, Sahoo et al. 2024.

Forward: q(x_t | x_0): each position independently replaced by MASK with prob t.
Loss:    L(x_0, t) = (1/t) · CE(model(x_t, t)[masked], x_0[masked])
"""
import math
import torch
import torch.nn.functional as F

from ..data.tokenizer import MASK_ID, VOCAB_SIZE


SCHEME = "mdlm"


def add_mask_noise(x: torch.Tensor, t: torch.Tensor):
    """Mask each position w.p. t. Returns (x_t, mask). x: (B,T) int; t: (B,) in (0,1)."""
    p = t[:, None].expand_as(x)
    mask = torch.rand_like(p) < p
    return torch.where(mask, torch.full_like(x, MASK_ID), x), mask


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    B, T = x0.shape
    device = x0.device
    t = torch.rand(B, device=device) * (1 - eps) + eps
    x_t, mask = add_mask_noise(x0, t)
    logits = model(x_t, t)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    masked_nll = (nll * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1.0)
    return (masked_nll / t).mean()


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_t_grid: int = 16,
             eps: float = 1e-3, device="cuda") -> float:
    t_grid = torch.linspace(eps, 1.0, n_t_grid, device=device)
    total, n_seen = 0.0, 0
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        B, T = x0.shape
        per_seq = torch.zeros(B, device=device)
        for t_val in t_grid:
            t = torch.full((B,), t_val.item(), device=device)
            x_t, mask = add_mask_noise(x0, t)
            logits = model(x_t, t)
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            masked_nll = (nll * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1.0)
            per_seq += masked_nll / t_val
        per_seq /= n_t_grid
        bpc = per_seq / math.log(2)
        total += float(bpc.sum().item())
        n_seen += B
    return total / max(1, n_seen)
