"""SGD with momentum wrapper for uniform interface."""
import torch


class SGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=nesterov)
        self.last_metrics: dict = {}
