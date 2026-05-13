"""Plot all training-loss curves for every finished run in a sweep.

Useful for diagnosing whether training is actually happening across the
grid. Each line = one run, colored by lr.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


SWEEPS = {
    "prodigy":   "4hjyk7hh",
    "muon":      "pnrip3g7",
    "adamw":     "f520y6rg",
    "mars_adam": "zjdahinh",
    "soap":      "jt5rohiz",
}
ENTITY = "steeldream"
PROJECT = "diffusion-baselines-src"
OUT = Path("exps/image_phase_a")
OUT.mkdir(parents=True, exist_ok=True)


def plot_one_sweep(api, name: str, sid: str) -> int:
    sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
    lines = []
    for r in sw.runs:
        if r.state != "finished":
            continue
        rows = list(r.scan_history(keys=["loss", "_step"]))
        rows = [row for row in rows if row.get("loss") is not None]
        if len(rows) < 2:
            continue
        iters = [row["_step"] for row in rows]
        losses = [row["loss"] for row in rows]
        lines.append({"iters": iters, "losses": losses,
                      "lr": r.config.get("lr"),
                      "wd": r.config.get("weight_decay"),
                      "eval_loss": float(r.summary.get("eval_loss", float("nan")))})
    if not lines:
        return 0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lrs = sorted({l["lr"] for l in lines})
    cmap = plt.cm.viridis
    for ln in lines:
        # color by lr
        i = lrs.index(ln["lr"])
        c = cmap(i / max(1, len(lrs) - 1))
        ax.plot(ln["iters"], ln["losses"], color=c, alpha=0.6, lw=0.9,
                label=f"lr={ln['lr']:g} wd={ln['wd']:g} (final={ln['eval_loss']:.4f})")
    ax.set_xlabel("iter")
    ax.set_ylabel("training loss (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{name} — all {len(lines)} finished runs, colored by lr")
    # dedupe legend by lr
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep = []
    for h, l in zip(handles, labels):
        k = l.split(" ")[0]
        if k in seen:
            continue
        seen.add(k); keep.append((h, l))
    ax.legend([h for h, _ in keep], [l for _, l in keep],
              loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT / f"all_curves_{name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  wrote {out}  ({len(lines)} runs)")
    return len(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="comma-separated subset of optimizers")
    args = p.parse_args()
    api = wandb.Api()
    targets = list(SWEEPS) if args.only is None else args.only.split(",")
    for name in targets:
        sid = SWEEPS.get(name)
        if sid is None:
            print(f"!! {name}: unknown")
            continue
        print(f"=== {name} ===")
        plot_one_sweep(api, name, sid)


if __name__ == "__main__":
    main()
