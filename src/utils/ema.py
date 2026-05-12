"""Exponential moving average of model parameters."""
import copy

import torch
import torch.nn as nn


def make_ema(model: nn.Module) -> nn.Module:
    """Deep-copy a model and freeze its parameters."""
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA: ema ← decay·ema + (1-decay)·model."""
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
