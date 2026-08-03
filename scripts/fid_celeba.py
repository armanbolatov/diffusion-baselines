"""FID for one CelebA DDPM checkpoint (the reportable generative metric).

Samples N images via DDPM-1000 (EMA weights if present), writes them as PNGs,
then runs torch_fidelity against a cached folder of real CelebA images at --size.
This is the headline FID — use --n 50000 for a publishable number; the in-training
eval_fid (N=1024, DDIM-50) is only a training monitor.

Usage:
    uv run python scripts/fid_celeba.py <ckpt> --n 50000 --size 64
"""
import argparse
import math
import shutil
from pathlib import Path

import torch
from diffusers import DDPMScheduler
import torchvision.utils as vutils

from src.data.celeba import celeba_loaders
from src.models import build_image_model


def real_dir(size):
    return Path(f"exps/fid/real_celeba_{size}")


@torch.no_grad()
def dump_real(n: int, size: int) -> None:
    d = real_dir(size)
    if d.exists() and len(list(d.iterdir())) >= n:
        return
    d.mkdir(parents=True, exist_ok=True)
    _, eval_loader = celeba_loaders(data_dir="data", batch_size=256,
                                    eval_batch_size=256, num_workers=0, image_size=size)
    saved = 0
    for x, _ in eval_loader:
        x = (x.clamp(-1, 1) + 1) / 2
        for img in x:
            vutils.save_image(img, d / f"{saved:06d}.png")
            saved += 1
            if saved >= n: return


@torch.no_grad()
def sample_to_dir(ckpt: Path, n: int, bs: int, out_dir: Path, size: int) -> None:
    device = torch.device("cuda")
    model = build_image_model(argparse.Namespace(dropout=0.0), sample_size=size).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    # Sample from EMA weights when available — the standard for stable DDPM FID.
    state = payload.get("ema_state") or payload["model_state"]
    model.load_state_dict(state); model.eval()

    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    sched.set_timesteps(1000, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < n:
        b = min(bs, n - saved)
        x = torch.randn(b, 3, size, size, device=device)
        for t in sched.timesteps:
            t_b = t.expand(b).long()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                eps = model(x, t_b).sample
            x = sched.step(eps.float(), t, x).prev_sample
        x = (x.clamp(-1, 1) + 1) / 2
        for img in x:
            vutils.save_image(img, out_dir / f"{saved:06d}.png")
            saved += 1
        print(f"  sampled {saved}/{n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n",  type=int, default=2048)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--size", type=int, default=64, help="image resolution (celeba64 = 64)")
    p.add_argument("--keep_samples", action="store_true")
    args = p.parse_args()

    dump_real(args.n, args.size)
    fake_dir = Path(f"exps/fid/fake/{Path(args.ckpt).parent.name}")
    if not (fake_dir.exists() and len(list(fake_dir.iterdir())) >= args.n):
        sample_to_dir(Path(args.ckpt), args.n, args.bs, fake_dir, args.size)

    from torch_fidelity import calculate_metrics
    m = calculate_metrics(input1=str(fake_dir), input2=str(real_dir(args.size)),
                          cuda=True, isc=False, fid=True, kid=False, verbose=False)
    print(f"FID = {m['frechet_inception_distance']:.2f}")

    if not args.keep_samples:
        shutil.rmtree(fake_dir)


if __name__ == "__main__":
    main()
