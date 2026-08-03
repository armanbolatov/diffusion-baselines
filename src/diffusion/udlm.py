"""UDLM (Uniform Diffusion Language Model) — uniform-state discrete diffusion.

Ported from Schiff et al., 2024 ("Simple Guidance Mechanisms for Discrete
Diffusion Models", arXiv:2412.10193) and the reference implementation
kuleshov-group/discrete-diffusion-guidance ``diffusion.py`` (``self.diffusion
== 'uniform'``).

How UDLM differs from the repo's MDLM (absorbing/masked) scheme:

1. **Forward kernel** — each token is, w.p. ``1 - alpha_t``, replaced by a
   *uniformly random* token (stationary distribution ``pi = 1/N``). Unlike the
   absorbing MASK, a corrupted token can keep changing, which is what makes
   uniform diffusion more amenable to guidance (the paper's motivation).
2. **Mean parameterization** — the network predicts a distribution ``x_theta``
   over the *clean* token (a log-softmax over the full vocab); there is no MASK
   index to force to -inf (no SUBS).
3. **Time conditioning is ON** — the optimal denoiser depends on ``t`` because
   the latent is continually edited, so we pass the real ``t`` (MDLM, by
   contrast, zeroes it).

Objective — the continuous-time NELBO (paper Eq. 19), with the reconstruction
and prior terms vanishing analytically as ``T -> inf``. Using the log-linear
schedule ``alpha_t = 1 - t`` (so ``alpha_t' = -1``), the per-token integrand is
the uniform-posterior KL of the repo's ``diffusion_loss``; sampling ``t ~ U``
makes its expectation the integral. Random-init reference: per-token NLL ≈
``log(N)``, i.e. ``eval_bpc ≈ log2(N)`` (≈ 5.3 for the QM9 vocab of ~40).
"""
import math

import torch
import torch.nn.functional as F


SCHEME = "udlm"


def _vocab_size(model) -> int:
    """Infer N from the model so the scheme stays tokenizer-agnostic."""
    m = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    if hasattr(m, "head"):
        return m.head.out_features
    return m.token_emb.num_embeddings


def _sample_t(n: int, device, eps: float) -> torch.Tensor:
    """Stratified (antithetic) t ~ Uniform(eps, 1); ~n× lower per-batch variance."""
    u = torch.rand(n, device=device)
    offset = torch.arange(n, device=device, dtype=u.dtype) / n
    u = (u / n + offset) % 1.0
    return eps + (1.0 - eps) * u


def q_xt(x0: torch.Tensor, t: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Forward kernel: each position is replaced by a uniform token w.p. t.

    With ``alpha_t = 1 - t`` the move probability is exactly ``t``.
    """
    p = t[:, None].expand_as(x0).to(torch.float)
    move = torch.rand_like(p) < p
    rand = torch.randint(0, vocab_size, x0.shape, device=x0.device, dtype=x0.dtype)
    return torch.where(move, rand, x0)


def _nelbo_per_token(log_p: torch.Tensor, x0: torch.Tensor, x_t: torch.Tensor,
                     t: torch.Tensor, vocab_size: int,
                     eps: float = 1e-3) -> torch.Tensor:
    """Per-token UDLM NELBO integrand at the sampled t (paper Eq. 19).

    Args:
        log_p: (B, L, V) model log-probs over the clean token (mean param).
        x0:    (B, L) clean token ids.
        x_t:   (B, L) noisy token ids.
        t:     (B,) diffusion time.
    Returns:
        (B, L) tensor of per-token nats.
    """
    N = vocab_size
    alpha_t = (1.0 - t)[:, None, None].clamp(min=eps)            # (B, 1, 1)
    one_hot_x0 = F.one_hot(x0, N).to(log_p.dtype)               # (B, L, V)

    # x_bar = N*alpha_t*x + (1 - alpha_t); same with x_theta = exp(log_p).
    x_bar = N * alpha_t * one_hot_x0 + (1.0 - alpha_t)
    x_bar_theta = N * alpha_t * log_p.exp() + (1.0 - alpha_t)

    coeff = -1.0 / (N * alpha_t)                                # alpha_t' / (N*alpha_t)

    x_bar_zt = x_bar.gather(-1, x_t[..., None])                 # (B, L, 1)
    x_bar_theta_zt = x_bar_theta.gather(-1, x_t[..., None])     # (B, L, 1)

    term1 = (N / x_bar_zt) - (N / x_bar_theta_zt)               # (B, L, 1)
    term2 = (x_bar / x_bar_zt) * (
        x_bar_theta_zt.log() - x_bar_theta.log()
        + x_bar.log() - x_bar_zt.log()
    )
    term2 = term2.sum(dim=-1, keepdim=True)                     # (B, L, 1)
    return (coeff * (term1 - term2)).squeeze(-1)                # (B, L)


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Continuous-time UDLM NELBO (per-token nats), averaged over batch & length."""
    B = x0.shape[0]
    V = _vocab_size(model)
    t = _sample_t(B, x0.device, eps)
    x_t = q_xt(x0, t, V)
    log_p = F.log_softmax(model(x_t, t).float(), dim=-1)
    return _nelbo_per_token(log_p, x0, x_t, t, V, eps=eps).mean()


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_mc: int = 32,
             eps: float = 1e-3, device="cuda", pad_id: int = None) -> float:
    """Monte-Carlo UDLM NELBO in bits per token.

    When ``pad_id`` is given, the average is over NON-pad tokens only (the mask
    is ``x0 != pad_id``, so bos/eos still count). This matches the UDLM paper's
    attention-masked perplexity, where padding contributes to neither the
    numerator nor the denominator — so the reported number is comparable to
    theirs. With ``pad_id=None`` it averages over all positions.
    """
    V = _vocab_size(model)
    log2 = math.log(2)
    total_nats, total_tokens = 0.0, 0.0
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        mask = ((x0 != pad_id) if pad_id is not None
                else torch.ones_like(x0, dtype=torch.bool)).float()
        per_pos = torch.zeros_like(mask)                     # (B, L) per-token nats
        for _ in range(n_mc):
            t = _sample_t(x0.shape[0], device, eps)
            x_t = q_xt(x0, t, V)
            log_p = F.log_softmax(model(x_t, t).float(), dim=-1)
            per_pos += _nelbo_per_token(log_p, x0, x_t, t, V, eps=eps)
        per_pos /= n_mc
        total_nats += float((per_pos * mask).sum().item())
        total_tokens += float(mask.sum().item())
    return (total_nats / max(1.0, total_tokens)) / log2


def _posterior(x_theta: torch.Tensor, x_t: torch.Tensor,
               alpha_s: float, alpha_t: float, vocab_size: int) -> torch.Tensor:
    """Uniform reverse posterior p_theta(z_s | z_t) (paper Eq. 15).

    ``x_theta``: (B, L, V) predicted clean distribution; ``x_t``: (B, L) ids.
    Returns (B, L, V) categorical probabilities for z_s.
    """
    N = vocab_size
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(x_t, N).to(x_theta.dtype)
    limiting = 1.0 / N
    num = (alpha_t * N * x_theta * xt_one_hot
           + (alpha_ts - alpha_t) * xt_one_hot
           + d_alpha * x_theta
           + (1.0 - alpha_ts) * (1.0 - alpha_s) * limiting)
    den = alpha_t * N * x_theta.gather(-1, x_t[..., None]) + (1.0 - alpha_t)
    return num / den


@torch.no_grad()
def sample(model, n_samples: int, length: int, n_steps: int = 128,
           vocab_size: int = None, device="cuda") -> torch.Tensor:
    """Ancestral sampling under the uniform reverse process. Returns (n, L) ids."""
    if vocab_size is None:
        vocab_size = _vocab_size(model)
    x = torch.randint(0, vocab_size, (n_samples, length), device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t, s = float(ts[i]), float(ts[i + 1])
        alpha_t = max(1.0 - t, 1e-3)
        alpha_s = max(1.0 - s, 1e-3)
        t_batch = torch.full((n_samples,), t, device=device)
        x_theta = F.softmax(model(x, t_batch).float(), dim=-1)
        probs = _posterior(x_theta, x, alpha_s, alpha_t, vocab_size)
        probs = probs.clamp(min=0).reshape(-1, vocab_size)
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12)
        x = torch.multinomial(probs, 1).reshape(n_samples, length)
    return x
