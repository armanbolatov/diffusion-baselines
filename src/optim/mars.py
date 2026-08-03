"""MARS-Adam: variance-reduced AdamW (no Muon).

The `mars-adamw` variant:
    c_t = g_t + gamma * beta1/(1-beta1) * (g_t - g_{t-1})
    AdamW step on c_t.

When gamma = 0, this reduces to standard AdamW (small overhead from caching g_{t-1}).
"""
import math
import torch
from typing import Callable, Union

GammaArg = Union[float, Callable[[int], float]]


def _resolve_gamma(g, step):
    if callable(g):
        return float(g(step))
    return float(g)


class MARSAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, gamma: GammaArg = 0.025,
                 log_smoothness: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gamma=gamma)
        super().__init__(list(params), defaults)
        self.log_smoothness = log_smoothness
        self.step_count = 0
        self.last_metrics: dict = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        sum_grad_sq = 0.0
        sum_step_sq = 0.0
        sum_diff_sq = 0.0
        sum_c_sq = 0.0
        sum_c_dot_g = 0.0
        last_gamma = 0.0
        last_mars = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            gamma_t = _resolve_gamma(group["gamma"], self.step_count)
            last_gamma = gamma_t
            mars_factor = gamma_t * beta1 / (1.0 - beta1)
            last_mars = mars_factor

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["last_grad"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]

                if self.log_smoothness:
                    sum_grad_sq += float(g.float().pow(2).sum().item())
                    sum_diff_sq += float((g - state["last_grad"]).float().pow(2).sum().item())

                # MARS-Adam correction
                c = (g - state["last_grad"]).mul(mars_factor).add(g)

                if self.log_smoothness:
                    sum_c_sq    += float(c.float().pow(2).sum().item())
                    sum_c_dot_g += float((c.float() * g.float()).sum().item())

                # AdamW step on c
                m1 = state["m1"]; m2 = state["m2"]
                m1.mul_(beta1).add_(c, alpha=(1 - beta1))
                m2.mul_(beta2).addcmul_(c, c, value=(1 - beta2))
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (m2.sqrt() / math.sqrt(bc2)).add_(eps)
                update = (m1 / denom) * (lr / bc1)

                if self.log_smoothness:
                    sum_step_sq += float(update.float().pow(2).sum().item())

                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-1.0)

                state["last_grad"] = g.detach().clone()

        if self.log_smoothness:
            grad_norm = sum_grad_sq ** 0.5
            step_norm = sum_step_sq ** 0.5
            diff_norm = sum_diff_sq ** 0.5
            c_norm = sum_c_sq ** 0.5
            cos_cg = (sum_c_dot_g / (c_norm * grad_norm)) if (c_norm > 0 and grad_norm > 0) else 0.0
            self.last_metrics = {
                "grad_norm_F": grad_norm,
                "step_norm_F": step_norm,
                "grad_diff_norm_F": diff_norm,
                "c_norm_F": c_norm,
                "cos_c_g": cos_cg,
                "gamma_t": last_gamma,
                "mars_factor_t": last_mars,
            }
        self.step_count += 1
        return loss
