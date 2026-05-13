"""Pull all Phase-A sweep runs from W&B and produce diagnostic plots.

Outputs four PNGs to exps/phase_a/:
  - dist_per_optimizer.png  : boxplot of eval_bpc per optimizer (216/144/etc. each)
  - lr_vs_bpc.png           : scatter of lr vs eval_bpc, colored by optimizer
  - heatmap_<opt>.png       : (lr × wd) grid mean for each optimizer
  - best_per_opt.png        : bar chart of min eval_bpc per optimizer

Also writes exps/phase_a/summary.csv (per-optimizer min/median/max + winning config).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


SWEEPS = {
    "adamw":     "pdr7wn47",
    "muon":      "w08ponn7",
    "mars_adam": "2bw9wvmk",
    "prodigy":   "1u6m8k4d",
    "soap":      "xyzdfad0",
}
ENTITY = "steeldream"
PROJECT = "diffusion-baselines-src"
OUT = Path("exps/phase_a")
OUT.mkdir(parents=True, exist_ok=True)


def fetch():
    api = wandb.Api()
    data = {}
    for name, sid in SWEEPS.items():
        rows = []
        sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        for r in sw.runs:
            if r.state != "finished":
                continue
            v = r.summary.get("eval_bpc")
            if v is None:
                continue
            rows.append({"name": r.name, "bpc": float(v), **dict(r.config)})
        data[name] = rows
        print(f"{name:12s} {len(rows):>4d} runs")
    return data


def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(SWEEPS)
    vals = [[r["bpc"] for r in data[k]] for k in labels]
    bp = ax.boxplot(vals, labels=labels, showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10.colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_ylabel("final eval_bpc (lower is better)")
    ax.set_title("Phase A — eval_bpc distribution per optimizer (text8 MDLM, small, 1500 iters)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "dist_per_optimizer.png", dpi=120)
    plt.close()


def plot_lr_vs_bpc(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    for color, (name, rows) in zip(plt.cm.tab10.colors, data.items()):
        if not rows or "lr" not in rows[0]:
            continue
        lrs = [r["lr"] for r in rows]
        bpcs = [r["bpc"] for r in rows]
        ax.scatter(lrs, bpcs, alpha=0.45, label=name, color=color, s=18)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final eval_bpc")
    ax.set_title("Phase A — lr vs eval_bpc, colored by optimizer")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "lr_vs_bpc.png", dpi=120)
    plt.close()


def plot_heatmaps(data):
    # one heatmap per optimizer: average bpc over (lr, wd) cells.
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(4.5 * len(SWEEPS), 4.2), squeeze=False)
    for i, (name, rows) in enumerate(data.items()):
        cells = defaultdict(list)
        for r in rows:
            lr = r.get("lr"); wd = r.get("weight_decay")
            if lr is None or wd is None:
                continue
            cells[(lr, wd)].append(r["bpc"])
        if not cells:
            continue
        lrs = sorted({k[0] for k in cells})
        wds = sorted({k[1] for k in cells})
        Z = np.full((len(wds), len(lrs)), np.nan)
        for (lr, wd), vs in cells.items():
            Z[wds.index(wd), lrs.index(lr)] = float(np.mean(vs))
        ax = axes[0, i]
        im = ax.imshow(Z, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{x:g}" for x in lrs], rotation=45)
        ax.set_yticks(range(len(wds)))
        ax.set_yticklabels([f"{x:g}" for x in wds])
        ax.set_xlabel("lr")
        ax.set_ylabel("weight_decay")
        ax.set_title(name)
        for y in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                v = Z[y, x]
                if np.isfinite(v):
                    ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > np.nanmean(Z) else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, label="mean eval_bpc")
    fig.suptitle("Phase A — mean eval_bpc across (lr × wd) (other dims averaged)")
    plt.tight_layout()
    plt.savefig(OUT / "heatmaps_lr_wd.png", dpi=120)
    plt.close()


def plot_best_per_opt(data):
    names, mins = [], []
    for name in SWEEPS:
        rows = data[name]
        if not rows:
            continue
        names.append(name)
        mins.append(min(r["bpc"] for r in rows))
    order = np.argsort(mins)
    names = [names[i] for i in order]
    mins = [mins[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(names)), mins, color=plt.cm.tab10.colors[: len(names)], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("min eval_bpc (winner of grid)")
    ax.set_title("Phase A — best eval_bpc per optimizer (1500 iters, small model)")
    ax.grid(axis="x", alpha=0.3)
    for b, v in zip(bars, mins):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.4f}",
                va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "best_per_opt.png", dpi=120)
    plt.close()


def write_summary(data):
    rows = [["optimizer", "n", "min", "median", "max", "winner_config"]]
    for name in SWEEPS:
        rs = data[name]
        if not rs:
            continue
        bpcs = sorted(r["bpc"] for r in rs)
        winner = min(rs, key=lambda r: r["bpc"])
        sig_keys = [k for k in ["lr","weight_decay","grad_clip","warmup_iters",
                                "betas_pair","momentum","d_coef","beta2","gamma",
                                "precondition_frequency","shampoo_beta"] if k in winner]
        sig = " ".join(f"{k}={winner[k]}" for k in sig_keys)
        rows.append([name, len(rs),
                     f"{bpcs[0]:.4f}", f"{bpcs[len(bpcs)//2]:.4f}", f"{bpcs[-1]:.4f}",
                     sig])
    with (OUT / "summary.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("--- summary.csv ---")
    for row in rows:
        print("  ".join(str(c) for c in row))


def main() -> None:
    data = fetch()
    write_summary(data)
    plot_distribution(data)
    plot_lr_vs_bpc(data)
    plot_heatmaps(data)
    plot_best_per_opt(data)
    print(f"\nplots written under {OUT}/")


if __name__ == "__main__":
    main()
