"""Learning-rate schedules (cosine, constant, warmup-stable-decay), each with linear warmup."""
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


def wsd_with_warmup(it: int, lr: float, warmup_iters: int,
                    total_iters: int, min_lr: float = 0.0,
                    decay_frac: float = 0.2) -> float:
    """Warmup-Stable-Decay (llm-baselines default).

    Linear warmup over ``warmup_iters`` → constant ``lr`` plateau →
    linear decay to ``min_lr`` over the final ``decay_frac`` of training.
    """
    if it < warmup_iters:
        return lr * it / max(1, warmup_iters)
    if it >= total_iters:
        return min_lr
    decay_start = int(total_iters * (1.0 - decay_frac))
    if it < decay_start:
        return lr
    frac = (it - decay_start) / max(1, total_iters - decay_start)
    frac = min(1.0, max(0.0, frac))
    return lr + (min_lr - lr) * frac


SCHEDULES = {
    "cosine": cosine_with_warmup,
    "constant": constant_with_warmup,
    "wsd": wsd_with_warmup,
}


def build_schedule(name: str):
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule: {name}; choose from {list(SCHEDULES)}")
    return SCHEDULES[name]
