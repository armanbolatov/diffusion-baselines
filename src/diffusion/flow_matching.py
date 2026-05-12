"""Flow Matching with the conditional-OT path (Lipman et al. 2023).

Forward (linear interpolation in time, a.k.a. straight-line / rectified flow):
    x_t = (1 - t) · noise + t · x_0,   t ~ U(0, 1)
Target velocity:
    v*(x_t, t) = x_0 - noise
Loss:
    L = E_{x_0, noise, t} ‖ v_θ(x_t, t) − (x_0 − noise) ‖²

Sampling: integrate dx/dt = v_θ(x, t) from t=0 (pure noise) to t=1.
"""
import torch
import torch.nn.functional as F


SCHEME = "flow_matching"


def add_fm_noise(x0: torch.Tensor):
    """Return (x_t, target_velocity, t). t in [0, 1]."""
    B = x0.size(0)
    device = x0.device
    noise = torch.randn_like(x0)
    t = torch.rand(B, device=device)
    t_b = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1.0 - t_b) * noise + t_b * x0
    v_target = x0 - noise
    return x_t, v_target, t


def loss(model, batch) -> torch.Tensor:
    """Batch already prepared by the FM train iter as (x_t, v_target, t).
    `model` is expected to be a UNet2DModel-style network returning .sample with shape == x_t."""
    x_t, v_target, t = batch
    # diffusers UNet expects integer-typed timesteps for default time-embedding.
    # We scale continuous t to [0, 1000) integer range so the existing
    # sinusoidal time-embedding gives a meaningful signal.
    t_int = (t * 999.0).long().clamp(0, 999)
    pred = model(x_t, t_int).sample
    return F.mse_loss(pred, v_target)


@torch.no_grad()
def eval_loss(model, eval_iter, n_batches: int) -> float:
    losses = []
    for i, batch in enumerate(eval_iter):
        if i >= n_batches:
            break
        x_t, v_target, t = batch
        t_int = (t * 999.0).long().clamp(0, 999)
        pred = model(x_t, t_int).sample
        losses.append(F.mse_loss(pred, v_target).item())
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def sample(model, shape, n_steps: int = 50, device="cuda"):
    """Euler ODE sample x_1 starting from x_0 ~ N(0, I)."""
    x = torch.randn(shape, device=device)
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    for i in range(n_steps):
        t_lo = ts[i]
        dt = ts[i + 1] - ts[i]
        t_b = t_lo.expand(shape[0])
        t_int = (t_b * 999.0).long().clamp(0, 999)
        v = model(x, t_int).sample
        x = x + v * dt
    return x.clamp(-1, 1)
