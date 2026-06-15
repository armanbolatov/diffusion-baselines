"""MARS-M: MARS variance-reduction applied to Muon (matrix-level).

Ported from ~/MARS/MARS_M/optimizers/mars_m.py (arXiv:2510.21800).

For matrix-valued parameters:
  c_t = (g - last_grad) * (gamma * mu / (1-mu)) + g     # variance-reduced grad
  if clip_c: c_t /= max(1, ||c_t||_F)
  m_t = mu * m_{t-1} + (1-mu) * c_t                      # momentum
  u_t = NewtonSchulz5(m_t)                               # orthogonalize
  p   <- (1 - lr * wd) * p - adjusted_lr * u_t           # decoupled wd
  adjusted_lr = lr * 0.2 * sqrt(max(dim0, dim1))

For 1D / embedding params: plain AdamW with bias correction.

Notes:
- ``is_approx=True`` (the default) uses the *prior step's* g as last_grad
  with no extra forward pass; the exact variant requires two backward
  passes per step (see paper). We expose only the approximate variant
  to keep the integration with our trainer simple.
- ``clip_c`` is exposed but is a near no-op once NewtonSchulz5 normalizes
  the update; defaults to False.
"""
import math

import torch


@torch.compile
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz iteration for matrix orthogonalization."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X


class MARS_M(torch.optim.Optimizer):
    def __init__(self, muon_params, adamw_params=None,
                 lr=1e-3, wd=0.1,
                 momentum=0.95, ns_steps=5,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 gamma=0.025, clip_c=False):
        mars_factor = gamma * momentum / (1 - momentum)
        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, ns_steps=ns_steps,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            gamma=gamma, mars_factor=mars_factor, clip_c=clip_c,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        # Exposed for the trainer/test harness; populated each step().
        self.last_metrics: dict = {}

    @staticmethod
    def _muon_adjusted_lr(lr: float, param_shape) -> float:
        A, B = param_shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            # ---- Muon branch (matrix params) ----
            muon_params = [p for p in group["params"]
                           if self.state[p].get("use_muon", False)]
            lr = group["lr"]; wd = group["wd"]
            mu = group["momentum"]; mars_factor = group["mars_factor"]
            clip_c = group["clip_c"]; ns_steps = group["ns_steps"]
            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "last_grad" not in state:
                    state["last_grad"] = torch.zeros_like(g)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                c_t = (g - state["last_grad"]).mul(mars_factor).add(g)
                if clip_c:
                    n = c_t.norm()
                    if n > 1:
                        c_t.div_(n)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(c_t, alpha=(1 - mu))
                u = _zeropower_via_newtonschulz5(buf, steps=ns_steps)
                adjusted_lr = self._muon_adjusted_lr(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)
                state["last_grad"] = g.detach()

            # ---- AdamW backup (1D / embedding params) ----
            adamw_params = [p for p in group["params"]
                            if not self.state[p].get("use_muon", False)]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                b1 = state["moment1"]; b2 = state["moment2"]
                b1.lerp_(g, 1 - beta1)
                b2.lerp_(g.square(), 1 - beta2)
                num = b1 / (eps + b2.sqrt())
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                scale = bc1 / (bc2 ** 0.5)
                p.data.mul_(1 - lr * wd)
                p.data.add_(num, alpha=-lr / scale)
        return loss
