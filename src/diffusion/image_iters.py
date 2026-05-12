"""Per-scheme image-batch iters.

Each iter wraps a CIFAR-10 DataLoader and emits whatever the matching
training loss needs: (x_noisy, noise, t) for DDPM, (x_t, v, t) for FM,
or raw x_0 for Consistency Training.

The iters are intentionally thin — each one is ~12 LOC; the heavy lifting
(noise schedule, target computation) lives in the scheme modules.
"""
from itertools import cycle
from typing import Tuple

import torch

from . import flow_matching as fmmod


class DDPMTrainIter:
    def __init__(self, loader, scheduler, device):
        self._it = cycle(loader)
        self._scheduler = scheduler
        self._device = device
        self._T = scheduler.config.num_train_timesteps

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self._it)
        x = x.to(self._device, non_blocking=True)
        bsz = x.size(0)
        t = torch.randint(0, self._T, (bsz,), device=self._device).long()
        noise = torch.randn_like(x)
        return self._scheduler.add_noise(x, noise, t), noise, t


class DDPMEvalIter:
    def __init__(self, loader, scheduler, device):
        self._loader = loader
        self._scheduler = scheduler
        self._device = device
        self._T = scheduler.config.num_train_timesteps

    def __iter__(self):
        for x, _ in self._loader:
            x = x.to(self._device, non_blocking=True)
            bsz = x.size(0)
            t = torch.randint(0, self._T, (bsz,), device=self._device).long()
            noise = torch.randn_like(x)
            yield self._scheduler.add_noise(x, noise, t), noise, t


class FMTrainIter:
    def __init__(self, loader, device):
        self._it = cycle(loader); self._device = device

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self._it)
        x = x.to(self._device, non_blocking=True)
        return fmmod.add_fm_noise(x)


class FMEvalIter:
    def __init__(self, loader, device):
        self._loader = loader; self._device = device

    def __iter__(self):
        for x, _ in self._loader:
            x = x.to(self._device, non_blocking=True)
            yield fmmod.add_fm_noise(x)


class RawImageTrainIter:
    """For Consistency Training: just yield x_0 from a cyclic DataLoader."""
    def __init__(self, loader, device):
        self._it = cycle(loader); self._device = device

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self._it)
        return x.to(self._device, non_blocking=True)


class RawImageEvalIter:
    def __init__(self, loader, device):
        self._loader = loader; self._device = device

    def __iter__(self):
        for x, _ in self._loader:
            yield x.to(self._device, non_blocking=True)
