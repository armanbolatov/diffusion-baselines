"""Learning-rate schedules. Adapted from ~/MARS/MARS_M/utils/cv_utils.py."""
import math
from typing import Optional


def cosine_with_warmup(it: int, lr: float, warmup_iters: int,
                       total_iters: int, min_lr: float = 0.0) -> float:
    """Linear warmup → cosine decay to min_lr."""
    if it < warmup_iters:
        return lr * it / max(1, warmup_iters)
    if it >= total_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, total_iters - warmup_iters)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def constant_with_warmup(it: int, lr: float, warmup_iters: int,
                         total_iters: Optional[int] = None,
                         min_lr: float = 0.0) -> float:
    """Linear warmup → constant lr."""
    if it < warmup_iters:
        return lr * it / max(1, warmup_iters)
    return lr


SCHEDULES = {
    "cosine": cosine_with_warmup,
    "constant": constant_with_warmup,
}


def build_schedule(name: str):
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule: {name}; choose from {list(SCHEDULES)}")
    return SCHEDULES[name]
