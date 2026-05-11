"""AdamW wrapper using torch.optim.AdamW with a uniform `last_metrics` attribute."""
import torch


class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.last_metrics: dict = {}
