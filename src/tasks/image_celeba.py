"""CelebA DDPM image-diffusion task (64x64 in the benchmark)."""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..data.celeba import celeba_loaders
from ..diffusion.continuous import build_ddpm_scheduler
from ..diffusion.image_iters import DDPMTrainIter, DDPMEvalIter
from ..models import build_image_model
from .base import DiffusionTask


class ImageCelebA(DiffusionTask):
    name = "image_celeba"

    def __init__(self) -> None:
        self._device = None
        self._args = None
        self._scheduler = None
        self._eval_i = 0

    def build_model(self, args):
        self._args = args
        if getattr(args, "diffusion", "ddpm") != "ddpm":
            raise ValueError("image_celeba currently supports diffusion=ddpm only")
        return build_image_model(args, sample_size=int(getattr(args, "image_size", 64)))

    def build_data(self, args) -> Tuple:
        self._device = torch.device(args.device)
        train_loader, eval_loader = celeba_loaders(
            args.data_dir, args.batch_size, args.eval_batch_size, args.num_workers,
            image_size=int(getattr(args, "image_size", 64)),
        )
        self._scheduler = build_ddpm_scheduler(
            num_train_timesteps=args.num_diffusion_steps,
            beta_schedule=args.beta_schedule,
        )
        return (DDPMTrainIter(train_loader, self._scheduler, self._device),
                DDPMEvalIter(eval_loader, self._scheduler, self._device))

    def loss_step(self, model, batch, step):
        x_noisy, noise, t = batch
        pred = model(x_noisy, t).sample
        return F.mse_loss(pred, noise), {}

    @torch.no_grad()
    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        losses = []
        for i, (x_noisy, noise, t) in enumerate(eval_iter):
            if i >= n_batches: break
            pred = model(x_noisy, t).sample
            losses.append(F.mse_loss(pred, noise).item())
        out = {"eval_loss": float(sum(losses) / max(1, len(losses))),
               "diffusion": "ddpm"}
        # Optional in-training FID (opt-in via --fid_n; every --fid_stride-th eval).
        fid_n = int(getattr(self._args, "fid_n", 0) or 0)
        if fid_n > 0 and self._eval_i % max(1, int(getattr(self._args, "fid_stride", 1))) == 0:
            from ..utils.fid import fid_score
            out["eval_fid"] = fid_score(model, fid_n,
                                        int(getattr(self._args, "fid_steps", 50)),
                                        self._device,
                                        size=int(getattr(self._args, "image_size", 64)))
        self._eval_i += 1
        return out
