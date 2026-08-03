"""DUO — uniform-state discrete diffusion trained through its Gaussian dual.

Ported from Sahoo et al., 2025 ("The Diffusion Duality", arXiv:2506.10892) and
the reference implementation s-sahoo/duo (``algo.py``, curriculum mode
``simple`` — the variant their LM1B script uses).

How DUO differs from the repo's UDLM scheme (same uniform-state family):

1. **Curriculum phase** (first half of training): the discrete latent is built
   from a *Gaussian* diffusion latent ``w = a~*onehot(x0) + s~*eps`` via the
   duality ``z_t = argmax(w)``; the model input is ``softmax(w/tau)`` (tau=1e-3,
   an almost-one-hot relaxation) and the loss target is ``z_t``. Sampling the
   argmax of a Gaussian instead of a categorical gives lower-variance gradients
   (the paper's speedup). The discrete noise level is the duality integral
   ``alpha = T(gamma)`` (paper Eq. 10), evaluated here by direct quadrature —
   the reference caches it in a pickle only because it backprops through it,
   which we never do. After the curriculum (and always at eval) training is
   plain uniform-state diffusion under the log-linear schedule ``alpha = 1-t``.
2. **Rao-Blackwellized NELBO** — ``_nll_per_token`` is their lower-variance
   estimator of the same uniform-state ELBO UDLM integrates (appendix A.4).

LM1B recipe constants (their ``train_lm1b_duo.sh``): gamma in [-3.5, -1.75],
tau fixed at 1e-3, curriculum over the first 50% of steps.
"""
import math

import torch
import torch.nn.functional as F

SCHEME = "duo"

GAMMA_MIN, GAMMA_MAX = -3.5, -1.75
TAU_INV = 1.0e3                  # 1/tau; gumbel_tau_log10 = -3 in the reference
CURRICULUM_FRAC = 0.5            # 500k of 1M steps in the paper


def _vocab_size(model) -> int:
    m = getattr(model, "_orig_mod", model)
    if hasattr(m, "head"):
        return m.head.out_features
    return m.token_emb.num_embeddings


def _sample_t(n: int, device, eps: float) -> torch.Tensor:
    """Stratified (antithetic) t ~ Uniform(eps, 1), as in the reference."""
    u = torch.rand(n, device=device)
    offset = torch.arange(n, device=device, dtype=u.dtype) / n
    u = (u / n + offset) % 1.0
    return eps + (1.0 - eps) * u


def _alpha_from_gamma(gamma: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Duality operator T (paper Eq. 10): discrete alpha for Gaussian noise gamma.

    alpha = (K*I - 1)/(K - 1) with I = integral phi(z - e^(-gamma/2)) Phi(z)^(K-1) dz,
    by trapezoid quadrature in float64 (integrand support is z in ~[3, c+8]).
    """
    K = vocab_size
    g = gamma.to(torch.float64)
    z = torch.arange(0.0, 25.0, 1e-3, dtype=torch.float64, device=gamma.device)
    c = torch.exp(-g / 2)[..., None]                       # (..., 1)
    log_pdf = -0.5 * (z - c) ** 2 - 0.5 * math.log(2 * math.pi)
    log_cdf = (K - 1) * torch.special.log_ndtr(z)
    integral = torch.exp(log_pdf + log_cdf).sum(-1) * 1e-3
    return ((K * integral - 1) / (K - 1)).to(gamma.dtype)


def _alpha_dalpha_from_t(t: torch.Tensor, vocab_size: int):
    """(alpha_t, dalpha/dt) on the curriculum path; finite difference like the
    reference (``_gamma_to_alpha_dalpha_cached``, dgamma = 1e-3)."""
    gamma = GAMMA_MIN + t * (GAMMA_MAX - GAMMA_MIN)
    both = _alpha_from_gamma(torch.cat([gamma, gamma + 1e-3]), vocab_size)
    alpha, alpha_plus = both.chunk(2)
    dalpha = (GAMMA_MAX - GAMMA_MIN) * (alpha_plus - alpha) / 1e-3
    return alpha, dalpha


def _nll_per_token(log_p, xt, x0, alpha_t, dalpha_t, vocab_size):
    """DUO's Rao-Blackwellized uniform-state NELBO integrand (per-token nats).

    Exact port of ``DUO_BASE.nll_per_token``; alpha_t/dalpha_t are (B, 1).
    """
    V = vocab_size
    x_bar_theta = V * alpha_t[:, :, None] * log_p.exp() + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (V * alpha_t)                              # (B, 1), negative
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + V * alpha_t * x_eq_xt
    xbar_theta_xt = x_bar_theta.gather(-1, xt[..., None]).squeeze(-1)
    xbar_theta_x = x_bar_theta.gather(-1, x0[..., None]).squeeze(-1)
    term1 = V * (1 / xbar_xt - 1 / xbar_theta_xt)

    const = (1 - alpha_t) / (V * alpha_t + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((V - 1) * const * x_eq_xt - (1 / const) * x_neq_xt) * const.log()
    term2 = -term2_coefs * (x_bar_theta.log().sum(-1) - V * xbar_theta_xt.log())
    term2 = term2 - V * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt
    term2 = term2 + term2_offset
    return coeff * (term1 - term2)


def _discrete_nll(model, x0, t, vocab_size, eps):
    """Post-curriculum / eval path: log-linear uniform-state diffusion
    (reference ``LogLinear``: alpha = 1 - (1-eps)*t, dalpha = -(1-eps))."""
    alpha = (1.0 - (1.0 - eps) * t)[:, None]                      # (B, 1)
    dalpha = -(1.0 - eps) * torch.ones_like(alpha)
    move = torch.rand(*x0.shape, device=x0.device) < (1 - alpha)
    rand = torch.randint(0, vocab_size, x0.shape, device=x0.device, dtype=x0.dtype)
    xt = torch.where(move, rand, x0)
    sigma = -alpha.squeeze(-1).log()
    log_p = F.log_softmax(model(xt, sigma).float(), dim=-1)
    return _nll_per_token(log_p, xt, x0, alpha, dalpha, vocab_size)


def loss(model, x0: torch.Tensor, step: int, curriculum_end: int,
         eps: float = 1e-3, pad_id=None) -> torch.Tensor:
    """DUO training loss (per-token nats over non-pad positions)."""
    B = x0.shape[0]
    V = _vocab_size(model)
    t = _sample_t(B, x0.device, eps)
    real = torch.ones_like(x0, dtype=torch.bool) if pad_id is None else (x0 != pad_id)

    if step < curriculum_end:
        alpha, dalpha = _alpha_dalpha_from_t(t, V)
        alpha, dalpha = alpha[:, None], dalpha[:, None]           # (B, 1)
        gamma = GAMMA_MIN + t * (GAMMA_MAX - GAMMA_MIN)
        a_tilde = torch.sigmoid(-gamma).sqrt()[:, None, None]
        s_tilde = torch.sigmoid(gamma).sqrt()[:, None, None]
        w = a_tilde * F.one_hot(x0, V).float() + s_tilde * torch.randn(
            (B, x0.shape[1], V), dtype=torch.float32, device=x0.device)
        xt = w.argmax(-1)
        probs = F.softmax(w.mul_(TAU_INV), dim=-1)                # ~one-hot relaxation
        sigma = -alpha.squeeze(-1).log()
        log_p = F.log_softmax(model(probs, sigma).float(), dim=-1)
        per_tok = _nll_per_token(log_p, xt, x0, alpha, dalpha, V)
    else:
        per_tok = _discrete_nll(model, x0, t, V, eps)
    return (per_tok * real).sum() / real.sum().clamp(min=1)


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_mc: int = 32,
             eps: float = 1e-3, device="cuda", pad_id=None) -> float:
    """Monte-Carlo DUO NELBO in bits per token, pad-masked (matches the
    reference eval, which always uses the discrete estimator)."""
    V = _vocab_size(model)
    log2 = math.log(2)
    total_nats, total_tokens = 0.0, 0.0
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        mask = ((x0 != pad_id) if pad_id is not None
                else torch.ones_like(x0, dtype=torch.bool)).float()
        per_pos = torch.zeros_like(mask)
        for _ in range(n_mc):
            t = _sample_t(x0.shape[0], device, eps)
            per_pos += _discrete_nll(model, x0, t, V, eps)
        per_pos /= n_mc
        total_nats += float((per_pos * mask).sum().item())
        total_tokens += float(mask.sum().item())
    return (total_nats / max(1.0, total_tokens)) / log2
