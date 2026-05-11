"""Adafactor wrapper. Uses transformers.optimization.Adafactor.

This is the standard Adafactor implementation (Shazeer & Stern 2018) used in
T5/PaLM. Memory-efficient: no second-moment buffer for 1-D parameters.

For comparability with other optimizers we set `relative_step=False` and
require an explicit lr from the CLI.
"""
from transformers.optimization import Adafactor as _Adafactor


class Adafactor(_Adafactor):
    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3),
                 clip_threshold=1.0, decay_rate=-0.8, beta1=None,
                 weight_decay=0.0, scale_parameter=True,
                 relative_step=False, warmup_init=False):
        super().__init__(
            params, lr=lr, eps=eps, clip_threshold=clip_threshold,
            decay_rate=decay_rate, beta1=beta1, weight_decay=weight_decay,
            scale_parameter=scale_parameter, relative_step=relative_step,
            warmup_init=warmup_init,
        )
        self.last_metrics: dict = {}
