"""SOAP wrapper. Shampoo-style preconditioner in eigenbasis + AdamW (Vyas et al. 2024).

Uses pytorch-optimizer's reference implementation. Default `precondition_frequency=10`
means the Shampoo eigendecomposition is updated every 10 steps; raise it for cheaper
training, lower it for stronger preconditioning.
"""
from pytorch_optimizer import SOAP as _SOAP


class SOAP(_SOAP):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=None,
                 weight_decay=0.01, precondition_frequency=10,
                 max_precondition_dim=10000, precondition_1d=False, eps=1e-8):
        super().__init__(
            params, lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dim=max_precondition_dim,
            precondition_1d=precondition_1d, eps=eps,
        )
        self.last_metrics: dict = {}
