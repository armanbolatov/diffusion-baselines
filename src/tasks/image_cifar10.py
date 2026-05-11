"""CIFAR-10 continuous DDPM task."""
from itertools import cycle
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..data.cifar10 import cifar10_loaders
from ..diffusion.continuous import build_ddpm_scheduler
from ..models.unet2d import build_unet2d
from .base import DiffusionTask


class ImageCIFAR10(DiffusionTask):
    name = "image_cifar10"

    def __init__(self):
        self.scheduler = None

    def build_model(self, args):
        return build_unet2d(sample_size=32, in_channels=3, out_channels=3)

    def build_data(self, args) -> Tuple:
        self.scheduler = build_ddpm_scheduler(
            num_train_timesteps=args.num_diffusion_steps,
            beta_schedule=args.beta_schedule,
        )
        train_loader, eval_loader = cifar10_loaders(
            args.data_dir, args.batch_size, args.eval_batch_size, args.num_workers
        )
        self._device = torch.device(args.device)
        return _DDPMTrainIter(train_loader, self.scheduler, self._device), \
               _DDPMEvalIter(eval_loader, self.scheduler, self._device)

    def loss_step(self, model, batch, step):
        x_noisy, noise, _t = batch
        pred = model(x_noisy, _t).sample
        loss = F.mse_loss(pred, noise)
        return loss, {}

    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        losses = []
        for i, batch in enumerate(eval_iter):
            if i >= n_batches:
                break
            x_noisy, noise, t = batch
            pred = model(x_noisy, t).sample
            losses.append(F.mse_loss(pred, noise).item())
        return {"eval_loss": float(sum(losses) / max(1, len(losses)))}


class _DDPMTrainIter:
    def __init__(self, loader, scheduler, device):
        self.loader = loader
        self.it = cycle(loader)
        self.scheduler = scheduler
        self.device = device
        self.T = scheduler.config.num_train_timesteps

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self.it)
        x = x.to(self.device, non_blocking=True)
        bsz = x.size(0)
        t = torch.randint(0, self.T, (bsz,), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, t)
        return x_noisy, noise, t


class _DDPMEvalIter:
    def __init__(self, loader, scheduler, device):
        self.loader = loader
        self.scheduler = scheduler
        self.device = device
        self.T = scheduler.config.num_train_timesteps

    def __iter__(self):
        for x, _ in self.loader:
            x = x.to(self.device, non_blocking=True)
            bsz = x.size(0)
            t = torch.randint(0, self.T, (bsz,), device=self.device).long()
            noise = torch.randn_like(x)
            x_noisy = self.scheduler.add_noise(x, noise, t)
            yield x_noisy, noise, t
