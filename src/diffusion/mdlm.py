"""MDLM (Masked Diffusion Language Model) — absorbing-state diffusion.

Ported from s-sahoo/duo/algo.py::MaskedAbsorbingState.

Forward process (log-linear schedule)::

    q(x_t | x_0): each token of x_0 independently replaced by MASK with
    probability t, where t ~ Uniform(eps, 1).

Loss (continuous-time SUBS-parameterized ELBO)::

    L(x_0) = E_{t ~ U}[E_{x_t|x_0}[ (1/t) * mean_over_seq( -log p_theta(x_0_i | x_t) * 1[i was masked] ) ]]

Three details that distinguish this from a naive implementation:

1. **SUBS parameterization** — the model's logit at the MASK index is
   forced to -inf before the softmax, so the denoiser never wastes
   probability mass on predicting MASK.
2. **Per-token (not per-masked-token) averaging** — we divide the
   masked-position NLL sum by the *sequence length*, not by the number
   of masked positions. The masked count grows linearly with t, so the
   `(1/t)` factor and the sum together stay bounded as t → 0. Dividing
   by the masked count instead would produce a `log(1/eps)` divergence
   that dominates the gradient.
3. **MC estimate at eval** — bpc is reported as an empirical mean over
   ``n_mc`` independent t draws, mirroring the loss formula exactly.

Random init reference: per-token NLL ≈ log(VOCAB_SIZE - 1) ≈ log(27) ≈
3.30 nats, i.e. eval_bpc ≈ log2(27) ≈ 4.75. Trained MDLM on text8
reaches ~1.4 bpc at scale.
"""
import math

import torch
import torch.nn.functional as F

from ..data.tokenizer import MASK_ID, VOCAB_SIZE


SCHEME = "mdlm"
_NEG_INF = -1e9


def _sample_t_antithetic(n: int, device, eps: float) -> torch.Tensor:
    """Stratified (antithetic) t ~ Uniform(eps, 1) across a batch of size n.

    Each example sees a t inside its own evenly-spaced stratum of [eps, 1].
    Within a stratum the t is still uniform, so the marginal distribution
    is unchanged; only the per-batch variance drops by ~n×.
    """
    u = torch.rand(n, device=device)
    offset = torch.arange(n, device=device, dtype=u.dtype) / n
    u = (u / n + offset) % 1.0
    return eps + (1.0 - eps) * u


def q_xt(x0: torch.Tensor, t: torch.Tensor):
    """Forward process: mask each position of x0 independently with prob t.

    Args:
        x0: (B, L) int tokens.
        t:  (B,)  float in (0, 1) per example.

    Returns:
        (x_t, mask) where ``x_t[b, i] = MASK`` iff ``mask[b, i] = True``.
    """
    p = t[:, None].expand_as(x0)
    mask = torch.rand_like(p, dtype=torch.float) < p
    x_t = torch.where(mask, torch.full_like(x0, MASK_ID), x0)
    return x_t, mask


def _log_probs(model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward the model, force MASK logit to -inf, return log-softmax.

    The MDLM paper (Sahoo et al., 2024; kuleshov-group/mdlm reference impl)
    sets ``time_conditioning=False`` by default: the optimal prediction at
    masked positions depends only on the unmasked context, not on t, so
    feeding t to the network adds a feature the model must learn to ignore.
    That learning is the dominant cause of AdamW's multi-thousand-iter
    plateau on this task. We follow the reference and pass a zero "time"
    signal — the AdaLN MLPs in the backbone collapse to fixed affine
    parameters, equivalent to ordinary LayerNorm.

    Returns: (B, L, VOCAB_SIZE) log-probs whose distribution over the
    last dim sums to 1 with p(MASK) = 0.
    """
    zero_t = torch.zeros_like(t)
    logits = model(x_t, zero_t)
    # Clone before mutation so autograd can backprop through the forward.
    logits = logits.clone()
    logits[..., MASK_ID] = _NEG_INF
    return F.log_softmax(logits, dim=-1)


def _per_example_nll(log_p: torch.Tensor, x0: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Average per-token NLL over masked positions, divided by SEQ LEN.

    The "divided by seq len" (rather than by num_masked) is what keeps
    the ``(1/t)`` weighting bounded as t → 0.

    Returns: (B,) tensor of nats/token contributions.
    """
    nll = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
    L = x0.shape[-1]
    return (nll * mask.float()).sum(dim=-1) / L


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Continuous-time MDLM ELBO loss (per-token nats), averaged over batch.

    Uses antithetic / stratified t-sampling: per-batch variance drops by
    ~B×, which dramatically smooths the training-loss curve.
    """
    B = x0.shape[0]
    t = _sample_t_antithetic(B, x0.device, eps)             # (B,)
    x_t, mask = q_xt(x0, t)
    log_p = _log_probs(model, x_t, t)
    return (_per_example_nll(log_p, x0, mask) / t).mean()


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_mc: int = 32,
             eps: float = 1e-3, device="cuda") -> float:
    """Monte-Carlo estimate of the MDLM ELBO in bits-per-character.

    Returns ``mean_per_token_nats / log(2)`` averaged across
    ``n_batches`` eval batches × ``n_mc`` independent t samples.
    """
    total_bpc, total_examples = 0.0, 0
    log2 = math.log(2)
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        B = x0.shape[0]
        nats = torch.zeros(B, device=device)
        for _ in range(n_mc):
            t = _sample_t_antithetic(B, device, eps)
            x_t, mask = q_xt(x0, t)
            log_p = _log_probs(model, x_t, t)
            nats += _per_example_nll(log_p, x0, mask) / t
        nats /= n_mc
        total_bpc += float((nats / log2).sum().item())
        total_examples += B
    return total_bpc / max(1, total_examples)
