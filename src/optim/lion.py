"""Lion wrapper. Uses lion-pytorch reference implementation."""
from lion_pytorch import Lion as _Lion


class Lion(_Lion):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.last_metrics: dict = {}
