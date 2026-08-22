"""FID for one CelebA DDPM checkpoint (the reportable generative metric).

Samples N images via DDPM-1000 (EMA weights if present), writes them as PNGs,
then runs torch_fidelity against a cached folder of real CelebA images at --size.
This is the headline FID — use --n 50000 for a publishable number; the in-training
eval_fid (N=1024, DDIM-50) is only a training monitor.

Usage:
    uv run python scripts/fid_celeba.py <ckpt> --n 50000 --size 64
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from diffusers import DDPMScheduler
import torchvision.utils as vutils

from src.models import build_image_model


def real_dir(size):
    # train-split reference stats (standard FID protocol)
    return Path(f"exps/fid/real_celeba_train_{size}")


@torch.no_grad()
def dump_real(n: int, size: int) -> None:
    d = real_dir(size)
    if d.exists() and len(list(d.iterdir())) >= n:
        return
    d.mkdir(parents=True, exist_ok=True)
    from torch.utils.data import DataLoader
    from src.data.celeba import _CelebA, eval_transform
    loader = DataLoader(_CelebA("train", eval_transform(size)),
                        batch_size=256, num_workers=2)
    saved = 0
    for x, _ in loader:
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
    ck = Path(args.ckpt)
    # key on <opt>_<seed>: every optimizer's ckpt lives in a dir called "seed42",
    # so keying on the parent alone made concurrent runs share one sample dir.
    tag = f"{ck.parent.parent.name}_{ck.parent.name}"
    fake_dir = Path(f"exps/fid/fake/{tag}")
    if not (fake_dir.exists() and len(list(fake_dir.iterdir())) >= args.n):
        sample_to_dir(Path(args.ckpt), args.n, args.bs, fake_dir, args.size)

    from torch_fidelity import calculate_metrics
    m = calculate_metrics(input1=str(fake_dir), input2=str(real_dir(args.size)),
                          cuda=True, isc=False, fid=True, kid=False, verbose=False)
    fid = float(m["frechet_inception_distance"])
    print(f"FID = {fid:.2f}")

    # persist it: the reportable FID used to exist only in this print()
    out = Path("logs/fid_celeba.json")
    data = json.loads(out.read_text()) if out.exists() else {}
    data[tag] = {"fid": fid, "n": args.n, "size": args.size, "ckpt": str(ck)}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"wrote {out}")
    if not args.keep_samples:
        shutil.rmtree(fake_dir)


if __name__ == "__main__":
    main()
