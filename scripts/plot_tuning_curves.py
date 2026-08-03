"""Plot the training-loss curve of each optimizer's tuning-phase winner.

Reads logs/tune_<task>_winners.json (the winning trial number) and that trial's
exps_tuning/<task>/<opt>/trial_<n>/log.jsonl (no network). One figure per task
-> exps/tuning_plots/<task>_train_loss.png.

    uv run python scripts/plot_tuning_curves.py qm9 [text8 celeba64]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark_winners import METRIC

OUT = Path("exps/tuning_plots")
OUT.mkdir(parents=True, exist_ok=True)


def loss_curve(log_path):
    steps, losses = [], []
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "iter" and e.get("loss") is not None:
            steps.append(e["iter"]); losses.append(e["loss"])
    return steps, losses


def plot_task(task: str) -> None:
    wins = Path("logs") / f"tune_{task}_winners.json"
    if not wins.exists():
        print(f"!! {wins} not found — run `bash scripts/tune.sh <gpu> {task}` first")
        return
    winners = json.loads(wins.read_text())
    metric = METRIC[task]
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, (opt, w) in zip(plt.cm.tab10.colors, winners.items()):
        log = Path("exps_tuning", task, opt, f"trial_{w['trial']}", "log.jsonl")
        steps, losses = loss_curve(log) if log.exists() else ([], [])
        if not losses:
            print(f"!! {task}/{opt}: no loss history at {log}"); continue
        ax.plot(steps, losses, color=color, lw=1.1, alpha=0.85,
                label=f"{opt} (best {metric}={w['value']:.4f})")
    ax.set_xlabel("iter"); ax.set_ylabel("training loss (log scale)"); ax.set_yscale("log")
    ax.set_title(f"{task} tuning: training loss of each optimizer's best trial")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = OUT / f"{task}_train_loss.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}")


def main(argv=None) -> int:
    for t in (argv or sys.argv[1:]) or ["qm9", "text8", "celeba64"]:
        plot_task(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
