"""AdEMAMix wrapper. Two-timescale momentum EMA (Pagliardini et al. 2024).

Uses pytorch-optimizer's reference implementation.
"""
from pytorch_optimizer import AdEMAMix as _AdEMAMix


class AdEMAMix(_AdEMAMix):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999),
                 weight_decay=0.0, weight_decouple=True, alpha=5.0, eps=1e-8):
        super().__init__(
            params, lr=lr, betas=betas, weight_decay=weight_decay,
            weight_decouple=weight_decouple, alpha=alpha, eps=eps,
        )
        self.last_metrics: dict = {}
