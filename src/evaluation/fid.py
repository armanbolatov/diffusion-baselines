"""FID computation against the CIFAR-10 training set.

``cifar10_fid(model_sample_fn, n_samples, device, data_dir)`` computes Frechet
Inception Distance between a model-generated sample set and a same-sized
random subset of the CIFAR-10 training images. ``model_sample_fn`` takes a
shape tuple and returns a Tensor of images in [-1, 1].
"""
from typing import Callable, Tuple

import torch
from torchvision import datasets, transforms


def cifar10_fid(model_sample_fn: Callable[[Tuple[int, int, int, int]], torch.Tensor],
                n_samples: int,
                device,
                data_dir: str,
                batch_size: int = 128,
                feature: int = 2048) -> float:
    # Lazy import: keep torchmetrics optional for callers that never compute FID.
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
    real_ds = datasets.CIFAR10(root=data_dir, train=True, download=True,
                               transform=transforms.ToTensor())
    idx = torch.randperm(len(real_ds))[:n_samples].tolist()
    for s in range(0, n_samples, batch_size):
        sub = idx[s:s + batch_size]
        real = torch.stack([real_ds[i][0] for i in sub]).to(device)
        fid.update(real, real=True)
    for s in range(0, n_samples, batch_size):
        shape = (min(batch_size, n_samples - s), 3, 32, 32)
        fake = model_sample_fn(shape).clamp(-1, 1)
        fake01 = (fake + 1) / 2
        fid.update(fake01, real=False)
    return float(fid.compute().item())
