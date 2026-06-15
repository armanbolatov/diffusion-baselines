"""Plot training-loss curves for each optimizer's tuning-phase winner.

One figure per task, saved to exps/tuning_plots/<task>_train_loss.png.
Sweep IDs come from scripts/benchmark_winners.py:SWEEPS so the two stay in sync.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from benchmark_winners import ENTITY, PROJECT, SWEEPS, METRIC

OUT = Path("exps/tuning_plots")
OUT.mkdir(parents=True, exist_ok=True)


def winner(api, sids, metric):
    best, best_v = None, float("inf")
    for sid in sids:
        for r in api.sweep(f"{ENTITY}/{PROJECT}/{sid}").runs:
            if r.state != "finished":
                continue
            v = r.summary.get(metric)
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f < best_v:
                best_v, best = f, r
    return best, best_v


def history(run, key):
    rows = [(x["_step"], x[key]) for x in run.scan_history(keys=[key, "_step"])
            if x.get(key) is not None]
    seen, out = set(), []
    for s, v in rows:
        if s in seen:
            continue
        seen.add(s)
        out.append((s, v))
    return out


def plot_task(task: str) -> None:
    sweeps = SWEEPS.get(task)
    if not sweeps:
        print(f"!! {task}: no sweep IDs registered in benchmark_winners.SWEEPS"); return
    api = wandb.Api()
    metric = METRIC[task]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10.colors
    for color, (opt, sids) in zip(cmap, sweeps.items()):
        r, v = winner(api, sids, metric)
        if r is None:
            print(f"!! {task}/{opt}: no finished run"); continue
        loss_h = history(r, "loss")
        if not loss_h:
            print(f"!! {task}/{opt}: no training loss history"); continue
        steps = [s for s, _ in loss_h]
        losses = [l for _, l in loss_h]
        ax.plot(steps, losses, color=color, lw=1.1, alpha=0.85,
                label=f"{opt} (final {metric}={v:.4f})")
        print(f"  {task}/{opt:14s}  final {metric}={v:.4f}  loss_pts={len(losses)}")
    ax.set_xlabel("iter")
    ax.set_ylabel("training loss (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{task} tuning: training loss for each optimizer's best config")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = OUT / f"{task}_train_loss.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}\n")


def main(argv=None) -> int:
    tasks = (argv or sys.argv)[1:] or list(SWEEPS)
    for t in tasks:
        plot_task(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
