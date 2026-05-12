"""UDLM (Uniform Discrete Language Model) — uniform forward kernel.

Forward: each token transitions uniformly to any vocab item with prob t/(V-1),
         else stays. (No MASK token; just uniform corruption.)

The forward kernel:
    q(x_t = j | x_0 = i) = (1 - α_t) * 𝟙[i=j] + α_t / V
where α_t = t (we use continuous t ∈ [eps, 1]).

Loss (negative-ELBO weighted CE on corrupted positions):
    L(x_0, t) = (1/t) · CE(model(x_t, t)[corrupted], x_0[corrupted])
"""
import math
import torch
import torch.nn.functional as F

from ..data.tokenizer import VOCAB_SIZE


SCHEME = "udlm"


def add_uniform_noise(x: torch.Tensor, t: torch.Tensor):
    """Each position w.p. t replaced by a uniformly-random token. Returns (x_t, mask)."""
    p = t[:, None].expand_as(x)
    mask = torch.rand_like(p) < p
    rand = torch.randint(0, VOCAB_SIZE, x.shape, device=x.device, dtype=x.dtype)
    return torch.where(mask, rand, x), mask


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    B, T = x0.shape
    device = x0.device
    t = torch.rand(B, device=device) * (1 - eps) + eps
    x_t, mask = add_uniform_noise(x0, t)
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
            x_t, mask = add_uniform_noise(x0, t)
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
