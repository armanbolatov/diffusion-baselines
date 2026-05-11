"""Prodigy wrapper. Uses prodigyopt's Prodigy class.

Prodigy is learning-rate-free: `lr` should be 1.0 (the default).
The optimizer infers a learning rate `d` internally; we log `opt.param_groups[0]['d']`
to `last_metrics["prodigy_d"]` so it shows up in jsonl logs.
"""
from prodigyopt import Prodigy as _Prodigy


class Prodigy(_Prodigy):
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decouple=True, d_coef=1.0,
                 safeguard_warmup=True):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, decouple=decouple,
                         d_coef=d_coef, safeguard_warmup=safeguard_warmup)
        self.last_metrics: dict = {}

    def step(self, closure=None):
        loss = super().step(closure)
        d = self.param_groups[0].get("d", None)
        if d is not None:
            self.last_metrics["prodigy_d"] = float(d)
        return loss
