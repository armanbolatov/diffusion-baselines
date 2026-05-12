"""CIFAR-10 image-diffusion task.

Dispatches on ``args.diffusion`` (ddpm | flow_matching | consistency) and
``args.model`` (unet2d | dit). The task class is intentionally thin:
- per-scheme batch iters live in src.diffusion.image_iters
- per-scheme loss/sample math lives in src.diffusion.{ddpm,flow_matching,consistency}
- FID lives in src.evaluation.fid
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..data.cifar10 import cifar10_loaders
from ..diffusion import consistency as cmod
from ..diffusion import flow_matching as fmmod
from ..diffusion.continuous import build_ddpm_scheduler
from ..diffusion.image_iters import (
    DDPMTrainIter, DDPMEvalIter,
    FMTrainIter,  FMEvalIter,
    RawImageTrainIter, RawImageEvalIter,
)
from ..evaluation.fid import cifar10_fid
from ..models import build_image_model
from ..utils.ema import make_ema, ema_update
from .base import DiffusionTask


_SUPPORTED_SCHEMES = ("ddpm", "flow_matching", "consistency")


class ImageCIFAR10(DiffusionTask):
    name = "image_cifar10"

    def __init__(self) -> None:
        self.scheme: str = "ddpm"
        self._device = None
        self._args = None
        self._scheduler = None
        self._ema_model = None

    # ---- build ----
    def build_model(self, args):
        self._args = args
        self.scheme = getattr(args, "diffusion", "ddpm")
        if self.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(f"image_cifar10 supports {_SUPPORTED_SCHEMES}, got {self.scheme!r}")
        return build_image_model(args)

    def build_data(self, args) -> Tuple:
        self._device = torch.device(args.device)
        train_loader, eval_loader = cifar10_loaders(
            args.data_dir, args.batch_size, args.eval_batch_size, args.num_workers,
        )
        if self.scheme == "ddpm":
            self._scheduler = build_ddpm_scheduler(
                num_train_timesteps=args.num_diffusion_steps,
                beta_schedule=args.beta_schedule,
            )
            return (DDPMTrainIter(train_loader, self._scheduler, self._device),
                    DDPMEvalIter(eval_loader, self._scheduler, self._device))
        if self.scheme == "flow_matching":
            return (FMTrainIter(train_loader, self._device),
                    FMEvalIter(eval_loader, self._device))
        if self.scheme == "consistency":
            return (RawImageTrainIter(train_loader, self._device),
                    RawImageEvalIter(eval_loader, self._device))
        raise AssertionError("unreachable")

    # ---- step ----
    def loss_step(self, model, batch, step):
        if self.scheme == "ddpm":
            x_noisy, noise, t = batch
            pred = model(x_noisy, t).sample
            return F.mse_loss(pred, noise), {}
        if self.scheme == "flow_matching":
            return fmmod.loss(model, batch), {}
        if self.scheme == "consistency":
            ema = self._ensure_ema(model)
            x0 = batch
            l = cmod.loss(model, ema, x0,
                          self._args.ct_sigma_min, self._args.ct_sigma_max,
                          self._args.ct_rho,       self._args.ct_n_steps)
            ema_update(ema, model, self._args.ema_decay)
            return l, {}
        raise AssertionError("unreachable")

    # ---- eval ----
    @torch.no_grad()
    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        if self.scheme == "ddpm":
            metrics = self._ddpm_eval(model, eval_iter, n_batches)
        elif self.scheme == "flow_matching":
            metrics = self._fm_eval(model, eval_iter, n_batches)
        elif self.scheme == "consistency":
            metrics = self._consistency_eval(model, eval_iter, n_batches)
        else:
            raise AssertionError("unreachable")

        n_fid = int(getattr(self._args, "fid_samples", 0) or 0)
        if n_fid > 0:
            metrics["fid"] = self._compute_fid(model, n_fid,
                                               int(getattr(self._args, "fid_steps", 50)))
        return metrics

    # ---- per-scheme eval helpers ----
    def _ddpm_eval(self, model, eval_iter, n_batches):
        losses = []
        for i, (x_noisy, noise, t) in enumerate(eval_iter):
            if i >= n_batches: break
            pred = model(x_noisy, t).sample
            losses.append(F.mse_loss(pred, noise).item())
        return {"eval_loss": float(sum(losses) / max(1, len(losses))),
                "diffusion": "ddpm"}

    def _fm_eval(self, model, eval_iter, n_batches):
        losses = []
        for i, (x_t, v, t) in enumerate(eval_iter):
            if i >= n_batches: break
            pred = model(x_t, (t * 999.0).long().clamp(0, 999)).sample
            losses.append(F.mse_loss(pred, v).item())
        return {"eval_loss": float(sum(losses) / max(1, len(losses))),
                "diffusion": "flow_matching"}

    def _consistency_eval(self, model, eval_iter, n_batches):
        ema = self._ensure_ema(model)
        losses = []
        for i, x0 in enumerate(eval_iter):
            if i >= n_batches: break
            l = cmod.loss(model, ema, x0,
                          self._args.ct_sigma_min, self._args.ct_sigma_max,
                          self._args.ct_rho,       self._args.ct_n_steps)
            losses.append(l.item())
        return {"eval_loss": float(sum(losses) / max(1, len(losses))),
                "diffusion": "consistency"}

    # ---- FID via the matching sampler ----
    def _compute_fid(self, model, n_samples: int, n_steps: int) -> float:
        def sample_fn(shape):
            if self.scheme == "ddpm":
                return self._ddpm_sample(model, shape, n_steps)
            if self.scheme == "flow_matching":
                return fmmod.sample(model, shape, n_steps=n_steps, device=self._device)
            return cmod.sample_one_step(self._ensure_ema(model), shape,
                                        self._args.ct_sigma_max,
                                        self._args.ct_sigma_min,
                                        device=self._device)
        return cifar10_fid(sample_fn, n_samples=n_samples, device=self._device,
                           data_dir=self._args.data_dir)

    @torch.no_grad()
    def _ddpm_sample(self, model, shape, n_steps: int):
        """DDIM sampler over the existing DDPM scheduler config."""
        from diffusers import DDIMScheduler
        sched = DDIMScheduler(num_train_timesteps=self._scheduler.config.num_train_timesteps,
                              beta_schedule=self._scheduler.config.beta_schedule)
        sched.set_timesteps(n_steps, device=self._device)
        x = torch.randn(shape, device=self._device)
        for t in sched.timesteps:
            t_b = t.expand(shape[0]).long()
            eps = model(x, t_b).sample
            x = sched.step(eps, t, x).prev_sample
        return x

    # ---- EMA bookkeeping (lazy init) ----
    def _ensure_ema(self, model):
        if self._ema_model is None:
            self._ema_model = make_ema(model).to(self._device)
        return self._ema_model
