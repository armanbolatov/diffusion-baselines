"""Muon optimizer: momentum + NewtonSchulz5 spectral-norm post-processing,
with AdamW fallback for non-matrix (1-D) parameters.

The Newton-Schulz iteration operates on the last 2 dims of the gradient tensor.
For 4-D conv weights of shape (out, in, kH, kW), this orthogonalizes along
(kH, kW) per (out, in) pair.
"""
import math
import torch
import torch._dynamo

# Fall back to eager if torch.compile's backend can't build (e.g. CPU inductor
# or a torch version mismatch) instead of crashing the run.
torch._dynamo.config.suppress_errors = True


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """Newton-Schulz quintic iteration. Handles ndim >= 2 via last-2-dim batched matmul."""
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


class Muon(torch.optim.Optimizer):
    """
    Muon: momentum + Newton-Schulz orthogonalization + adjusted-LR scaling
    (Moonlight-style: AdamW backup for non-matrix params).

    Args
    ----
    muon_params : iterable[Parameter] — params with ndim >= 2 (matrix-like)
    adamw_params : iterable[Parameter] — params with ndim < 2 (norms/biases)
    lr : Muon spectral learning rate (typical 3e-4 .. 3e-3 for diffusion/LM)
    wd : weight decay (decoupled, applied multiplicatively)
    momentum : SGD-momentum for the matrix branch (0.95 default)
    nesterov : Nesterov-style momentum
    ns_steps : Newton-Schulz iterations (5 default)
    adamw_lr : lr for the AdamW backup branch (None = same as the Muon lr)
    adamw_betas / adamw_eps : for the AdamW backup branch
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        lr=1e-3,
        wd=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_lr=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_lr=lr if adamw_lr is None else adamw_lr,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        self.last_metrics: dict = {}

    def adjust_lr_for_muon(self, lr, param_shape):
        # Use the first 2 dims as the conceptual matrix shape.
        A, B = param_shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ----- Muon branch (matrix params) -----
            mparams = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            for p in mparams:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                upd = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # ----- AdamW backup branch (1-D params) -----
            aparams = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            alr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in aparams:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(g)
                    state["m2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                state["m1"].lerp_(g, 1 - beta1)
                state["m2"].lerp_(g.square(), 1 - beta2)
                g_eff = state["m1"] / (eps + state["m2"].sqrt())
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                scale = bc1 / bc2 ** 0.5
                p.data.mul_(1 - alr * wd)
                p.data.add_(g_eff, alpha=-alr / scale)
        return loss
