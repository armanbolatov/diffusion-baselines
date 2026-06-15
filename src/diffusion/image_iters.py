"""DDPM image-batch iters. Each iter wraps a DataLoader and emits (x_noisy, noise, t)."""
from itertools import cycle

import torch


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
