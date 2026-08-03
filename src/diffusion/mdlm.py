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

The absorbing MASK token is the last vocab index by convention (``mask_id =
model_vocab - 1``), derived from the model so the scheme is tokenizer-agnostic:
text8 (vocab 28 → mask 27) and LM1B (bert vocab 30522 + 1 → mask 30522) both
work unchanged. Padded datasets (LM1B) pass ``pad_id`` so pad positions are
excluded from the per-token average, matching the paper's masked perplexity.

Random init reference (text8): per-token NLL ≈ log(vocab - 1) ≈ log(27) ≈
3.30 nats, i.e. eval_bpc ≈ log2(27) ≈ 4.75. Trained MDLM on text8
reaches ~1.4 bpc at scale.
"""
import math

import torch
import torch.nn.functional as F


SCHEME = "mdlm"
_NEG_INF = -1e9


def _mask_id(model) -> int:
    """The absorbing MASK token index = last row of the model's output head."""
    return model.head.out_features - 1


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


def q_xt(x0: torch.Tensor, t: torch.Tensor, mask_id: int):
    """Forward process: mask each position of x0 independently with prob t.

    Args:
        x0: (B, L) int tokens.
        t:  (B,)  float in (0, 1) per example.
        mask_id: the absorbing MASK token index.

    Returns:
        (x_t, mask) where ``x_t[b, i] = mask_id`` iff ``mask[b, i] = True``.
    """
    p = t[:, None].expand_as(x0)
    mask = torch.rand_like(p, dtype=torch.float) < p
    x_t = torch.where(mask, torch.full_like(x0, mask_id), x0)
    return x_t, mask


def _log_probs(model, x_t: torch.Tensor, t: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Forward the model, force MASK logit to -inf, return log-softmax.

    The MDLM paper (Sahoo et al., 2024; kuleshov-group/mdlm reference impl)
    sets ``time_conditioning=False`` by default: the optimal prediction at
    masked positions depends only on the unmasked context, not on t, so
    feeding t to the network adds a feature the model must learn to ignore.
    That learning is the dominant cause of AdamW's multi-thousand-iter
    plateau on this task. We follow the reference and pass a zero "time"
    signal — the AdaLN MLPs in the backbone collapse to fixed affine
    parameters, equivalent to ordinary LayerNorm.

    Returns: (B, L, vocab) log-probs whose distribution over the
    last dim sums to 1 with p(MASK) = 0.
    """
    zero_t = torch.zeros_like(t)
    logits = model(x_t, zero_t)
    # Clone before mutation so autograd can backprop through the forward.
    logits = logits.clone()
    logits[..., mask_id] = _NEG_INF
    return F.log_softmax(logits, dim=-1)


def _per_example_nll(log_p: torch.Tensor, x0: torch.Tensor,
                     mask: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Average per-token NLL over masked positions, divided by REAL-token count.

    Dividing by a fixed per-example length (not by num_masked) is what keeps
    the ``(1/t)`` weighting bounded as t → 0. For unpadded data ``real`` is all
    ones so this is exactly "divided by seq len"; for padded data (LM1B) pad
    positions are dropped from both the masked-NLL sum and the denominator,
    matching the paper's masked perplexity.

    Returns: (B,) tensor of nats/token contributions.
    """
    nll = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
    keep = (mask & real).float()
    denom = real.float().sum(dim=-1).clamp(min=1.0)
    return (nll * keep).sum(dim=-1) / denom


def loss(model, x0: torch.Tensor, eps: float = 1e-3, pad_id=None) -> torch.Tensor:
    """Continuous-time MDLM ELBO loss (per-token nats), averaged over batch.

    Uses antithetic / stratified t-sampling: per-batch variance drops by
    ~B×, which dramatically smooths the training-loss curve. ``pad_id`` (if
    given) excludes pad positions from the per-token average.
    """
    B = x0.shape[0]
    mask_id = _mask_id(model)
    real = torch.ones_like(x0, dtype=torch.bool) if pad_id is None else (x0 != pad_id)
    t = _sample_t_antithetic(B, x0.device, eps)             # (B,)
    x_t, mask = q_xt(x0, t, mask_id)
    log_p = _log_probs(model, x_t, t, mask_id)
    return (_per_example_nll(log_p, x0, mask, real) / t).mean()


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_mc: int = 32,
             eps: float = 1e-3, device="cuda", pad_id=None) -> float:
    """Monte-Carlo estimate of the MDLM ELBO in bits-per-character.

    Returns ``mean_per_token_nats / log(2)`` averaged across
    ``n_batches`` eval batches × ``n_mc`` independent t samples. ``pad_id``
    (if given) excludes pad positions, matching the paper's masked perplexity.
    """
    mask_id = _mask_id(model)
    total_bpc, total_examples = 0.0, 0
    log2 = math.log(2)
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        B = x0.shape[0]
        real = torch.ones_like(x0, dtype=torch.bool) if pad_id is None else (x0 != pad_id)
        nats = torch.zeros(B, device=device)
        for _ in range(n_mc):
            t = _sample_t_antithetic(B, device, eps)
            x_t, mask = q_xt(x0, t, mask_id)
            log_p = _log_probs(model, x_t, t, mask_id)
            nats += _per_example_nll(log_p, x0, mask, real) / t
        nats /= n_mc
        total_bpc += float((nats / log2).sum().item())
        total_examples += B
    return total_bpc / max(1, total_examples)
