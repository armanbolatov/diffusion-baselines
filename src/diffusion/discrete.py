"""MDLM (Masked Diffusion Language Model) — absorbing-state discrete diffusion.

Forward process (continuous-time):
    Given clean tokens x0 and a time t ~ U(eps, 1):
    each position is independently masked with probability t,
    i.e. x_t[i] = MASK with probability t else x0[i].

Loss (per Sahoo et al. 2024, MDLM):
    L(x0, t) = (1/t) · CE(model(x_t, t)[masked positions], x0[masked positions])
The 1/t weighting is the ELBO / NELBO ratio for the absorbing process.
We integrate t ~ U(eps, 1) by sampling, where eps avoids the 1/t -> inf singularity.

Bits-per-character is computed at eval as -log p(x_0) / log(2) / seq_len,
estimated by averaging the per-token CE over a fixed t-grid (Riemann sum).
"""
import math
import torch
import torch.nn.functional as F

from ..data.tokenizer import MASK_ID, VOCAB_SIZE


def add_mask_noise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mask each position independently w.p. t. t broadcasts over the seq dim.

    x: (B, T) int64
    t: (B,) float in [eps, 1]
    returns: (B, T) int64 with some positions replaced by MASK_ID
    """
    p = t[:, None].expand_as(x)
    mask = torch.rand_like(p) < p
    return torch.where(mask, torch.full_like(x, MASK_ID), x), mask


def mdlm_loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """One MC sample of MDLM negative ELBO.

    x0: (B, T) int64 clean tokens (no MASK).
    eps: lower bound on t to avoid 1/t blow-up.
    """
    B, T = x0.shape
    device = x0.device
    t = torch.rand(B, device=device) * (1 - eps) + eps   # in [eps, 1]
    x_t, mask = add_mask_noise(x0, t)                    # mask: bool (B, T)
    logits = model(x_t, t)                               # (B, T, V)
    # CE only on masked positions; reduce per-batch then weight by 1/t
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, T)
    masked_nll = (nll * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1.0)  # (B,)
    weighted = masked_nll / t   # (B,)
    return weighted.mean()


@torch.no_grad()
def mdlm_eval_bpc(model, eval_iter, n_batches: int, n_t_grid: int = 16,
                  eps: float = 1e-3, device="cuda") -> float:
    """Estimate bits-per-character on the eval iter via t-grid integration.

    bpc = E_x [ -log p(x) ] / (T * log 2)
        ≈ (1/N) sum_x integral_t [ (1/t) · CE(masked) ] dt
        where integral is over t ~ U(eps, 1).
    """
    t_grid = torch.linspace(eps, 1.0, n_t_grid, device=device)
    total = 0.0
    n_seen = 0
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        B, T = x0.shape
        elbo_per_seq = torch.zeros(B, device=device)
        for t_val in t_grid:
            t = torch.full((B,), t_val.item(), device=device)
            x_t, mask = add_mask_noise(x0, t)
            logits = model(x_t, t)
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            masked_nll = (nll * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1.0)
            elbo_per_seq += masked_nll / t_val
        # average over t-grid → expectation
        elbo_per_seq /= n_t_grid
        # bits-per-char
        bpc = elbo_per_seq / math.log(2)
        total += float(bpc.sum().item())
        n_seen += B
    return total / max(1, n_seen)
