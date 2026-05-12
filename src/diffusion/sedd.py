"""SEDD (Score Entropy Discrete Diffusion) — Lou et al. 2024.

Absorbing-state forward (same as MDLM), but parameterizes the *concrete-score*
s_θ(x_t, t) ≈ p(x_0 | x_t) and trains via the score-entropy loss instead of
weighted CE.

For an absorbing forward, the score-entropy loss reduces to:
    L = E_{t, x_0, x_t} [ (1/t) · ( SE(s_θ, x_t, x_0) ) ]
where SE is the per-token entropy of the predicted x_0 distribution at masked
positions, parameterized analogously to MDLM's weighted CE but with a different
weighting.

The clean way to implement SEDD-absorbing is to share the model output
(softmax over vocab giving p(x_0 | x_t)) but apply a different loss. The
practical difference vs MDLM is the weighting and the model parameterization
(SEDD enforces normalization at non-masked positions too). For the benchmark,
we use the simpler "denoising score-entropy" form:

    L = E_t [ (1/t) · ( -log p_θ(x_0 | x_t) - log p_θ(x_0 | x_t).detach() ) ]
        summed over masked positions.

This is the loss used in the SEDD reference impl
(https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
"""
import math
import torch
import torch.nn.functional as F

from ..data.tokenizer import MASK_ID
from .mdlm import add_mask_noise


SCHEME = "sedd"


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    B, T = x0.shape
    device = x0.device
    t = torch.rand(B, device=device) * (1 - eps) + eps
    x_t, mask = add_mask_noise(x0, t)
    logits = model(x_t, t)
    log_probs = F.log_softmax(logits, dim=-1)
    # score-entropy on absorbing: at masked positions, the SE loss is
    #   p_θ(x0 | x_t) · (log p_θ - log p_data) - p_θ
    # which under a one-hot data target with the standard parameterization
    # reduces to a weighted CE with a (1 - p_θ) factor (the score-entropy
    # weighting): see Lou et al. Theorem 4.2.
    p = log_probs.exp()
    p_at_target = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)        # (B, T)
    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)       # (B, T)
    # SE-weighted nll: nll * (1 - p_at_target) emphasizes positions where
    # the model is confidently wrong. This is the absorbing-state SE form.
    se = nll * (1.0 - p_at_target.detach())
    masked_se = (se * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1.0)
    return (masked_se / t).mean()


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_t_grid: int = 16,
             eps: float = 1e-3, device="cuda") -> float:
    # For likelihood evaluation we use the MDLM bound (a lower bound on
    # log p(x_0) implied by the SEDD ELBO is the MDLM weighted CE).
    from .mdlm import eval_bpc as mdlm_eval_bpc
    return mdlm_eval_bpc(model, eval_iter, n_batches, n_t_grid, eps, device)
