"""In-training FID for CelebA DDPM.

Cheap enough to log periodically during a run: samples ``n`` images with DDIM
(``steps`` steps, far fewer than DDPM-1000) and scores them against a cached
folder of real CelebA images (at the run's image_size) via torch_fidelity. Used by the celeba task's
``eval_step`` (opt-in via ``--fid_n``); the full-budget FID lives in
``scripts/fid_celeba.py``.
"""
import shutil
import tempfile
from pathlib import Path

import torch
import torchvision.utils as vutils
from diffusers import DDIMScheduler


def ensure_real(n: int, size: int) -> None:
    """Cache ``n`` real CelebA eval images (at ``size``) as PNGs (once, atomically)."""
    real = Path(f"exps/fid/real_celeba_{size}")
    if real.exists() and len(list(real.glob("*.png"))) >= n:
        return real
    from ..data.celeba import celeba_loaders
    real.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"real_celeba_{size}_", dir=real.parent))
    _, eval_loader = celeba_loaders("data", 256, 256, num_workers=2, image_size=size)
    saved = 0
    for x, _ in eval_loader:
        x = (x.clamp(-1, 1) + 1) / 2
        for img in x:
            vutils.save_image(img, tmp / f"{saved:06d}.png")
            saved += 1
            if saved >= n:
                break
        if saved >= n:
            break
    shutil.rmtree(real, ignore_errors=True)   # stale or undersized cache
    try:
        tmp.rename(real)
    except OSError:                           # a concurrent run won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return real


@torch.no_grad()
def _sample(model, n: int, steps: int, device, out_dir: Path, size: int) -> None:
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    sched.set_timesteps(steps, device=device)
    saved = 0
    while saved < n:
        b = min(256, n - saved)
        x = torch.randn(b, 3, size, size, device=device)
        for t in sched.timesteps:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                eps = model(x, t.expand(b).long()).sample
            x = sched.step(eps.float(), t, x).prev_sample
        x = (x.clamp(-1, 1) + 1) / 2
        for img in x:
            vutils.save_image(img, out_dir / f"{saved:06d}.png")
            saved += 1


@torch.no_grad()
def fid_score(model, n: int, steps: int, device, size: int = 32) -> float:
    """Sample ``n`` images (DDIM ``steps``) and return FID vs the cached real set."""
    real = ensure_real(n, size)
    was_training = model.training
    model.eval()
    # per-call scratch: concurrent runs must not share a sample dir
    fake = Path(tempfile.mkdtemp(prefix="fid_fake_"))
    try:
        _sample(model, n, steps, device, fake, size)
        from torch_fidelity import calculate_metrics
        m = calculate_metrics(input1=str(fake), input2=str(real),
                              cuda=device.type == "cuda", isc=False, fid=True,
                              kid=False, verbose=False)
    finally:
        shutil.rmtree(fake, ignore_errors=True)
        if was_training:
            model.train()
    return float(m["frechet_inception_distance"])
