"""All 11 optimizers instantiate and take one step on a toy net."""
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from src.optim import build_optimizer, OPTIMIZERS


def _toy():
    return nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 1))


def _default_args(opt: str):
    return Namespace(
        optimizer=opt, lr=1e-3, weight_decay=0.0,
        beta1=0.9, beta2=0.999, eps=1e-8,
        momentum=0.9, nesterov=False,
        scale_parameter=False,
        beta3=0.9999, alpha=5.0,
        precondition_frequency=10, max_precondition_dim=10000, shampoo_beta=None,
        d_coef=1.0, warmup_iters=0, ns_steps=5,
        gamma=0.025, mars_type="mars-adamw", optimize_1d=False,
        lr_1d=3e-3, beta1_1d=0.9, beta2_1d=0.95, weight_decay_1d=0.0,
    )


@pytest.mark.parametrize("opt_name", OPTIMIZERS)
def test_optimizer_one_step(opt_name):
    torch.manual_seed(0)
    model = _toy()
    args = _default_args(opt_name)
    opt, summary = build_optimizer(args, model)
    if hasattr(opt, "train"):
        opt.train()
    x = torch.randn(4, 8)
    loss = (model(x) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    assert isinstance(opt.last_metrics, dict)
    assert summary["n_params"] > 0
