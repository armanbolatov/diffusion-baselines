"""Sample faces from a CelebA-32 DDPM checkpoint (full DDPM-1000 reverse process). Defaults to raw weights; pass --ema to load EMA weights instead."""
import argparse
import math
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from src.models import build_image_model


@torch.no_grad()
def sample(ckpt_path: Path, n: int, out: Path, use_ema: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_image_model(argparse.Namespace(), sample_size=32).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_ema and payload.get("ema_state") is not None:
        weights = payload["ema_state"]; tag = "EMA"
    else:
        weights = payload["model_state"]; tag = "raw"
    model.load_state_dict(weights)
    model.eval()
    print(f"loaded {tag} weights at iter {payload['iter_num']}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    sched.set_timesteps(1000, device=device)
    x = torch.randn(n, 3, 32, 32, device=device)
    for t in sched.timesteps:
        t_b = t.expand(n).long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            eps = model(x, t_b).sample
        x = sched.step(eps.float(), t, x).prev_sample
    x = (x.clamp(-1, 1) + 1) / 2

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = torch.zeros(3, rows * 32, cols * 32, device=device)
    for i, img in enumerate(x):
        r, c = divmod(i, cols)
        grid[:, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = img

    import torchvision.utils as vutils
    vutils.save_image(grid, out)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", default="exps/samples/faces.png")
    p.add_argument("--ema", action="store_true",
                   help="use EMA weights (default: raw). Recommended only when training is long enough to wash out EMA init bias.")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample(Path(args.ckpt), args.n, Path(args.out), use_ema=args.ema)


if __name__ == "__main__":
    main()
