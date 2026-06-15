"""Plot the 3-seed benchmark: per-step + per-wall-time curves + final bar chart.

For each task we make a 3-panel figure:
  [ step vs metric ]  [ wall time vs metric ]  [ final bar chart ]

Wall time is the trainer's ``elapsed_s`` (started after model construction,
data loaders, optimizer init). Each curve is mean +/- std over 3 seeds.

Usage:
    uv run python scripts/plot_final.py mdlm
    uv run python scripts/plot_final.py celeba
    uv run python scripts/plot_final.py mdlm celeba
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


ENTITY = "steeldream"
TASKS = {
    "text8":  {"project": "diffusion-baselines-bench-mdlm",   "metric": "eval_bpc",  "ylabel": "eval bpc",  "log_y": True},
    "celeba": {"project": "diffusion-baselines-bench-celeba", "metric": "eval_loss", "ylabel": "eval MSE", "log_y": False},
}

OPT_ORDER = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]

OUT = Path("exps/final_plots")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_triples(run, metric_key):
    """Pull (step, elapsed_s, metric) from a run's eval events, dedup on step."""
    keys = [metric_key, "elapsed_s", "_step"]
    rows = []
    for x in run.scan_history(keys=keys):
        if x.get(metric_key) is None or x.get("elapsed_s") is None:
            continue
        rows.append((x["_step"], float(x["elapsed_s"]), float(x[metric_key])))
    seen, out = set(), []
    for s, t, v in rows:
        if s in seen:
            continue
        seen.add(s)
        out.append((s, t, v))
    return out


def aggregate(runs, metric_key):
    """Stack (step, elapsed_s, metric) across seeds aligned by eval index.

    Returns (steps, times_mean, times_std, vals_mean, vals_std) — five 1-D
    arrays of length = number of common eval points.
    """
    series = [fetch_triples(r, metric_key) for r in runs]
    if not series or not all(series):
        return None
    n_common = min(len(s) for s in series)
    if n_common == 0:
        return None
    steps = np.array([series[0][i][0] for i in range(n_common)])
    times = np.array([[s[i][1] for i in range(n_common)] for s in series])
    vals  = np.array([[s[i][2] for i in range(n_common)] for s in series])
    return steps, times.mean(0), times.std(0), vals.mean(0), vals.std(0)


def plot_task(task: str) -> None:
    cfg = TASKS[task]
    api = wandb.Api()
    runs_by_opt = defaultdict(list)
    for r in api.runs(f"{ENTITY}/{cfg['project']}"):
        if r.state != "finished":
            continue
        opt = r.config.get("optimizer")
        if opt not in OPT_ORDER:
            continue
        # Skip diverged image runs (final metric near init = ~1.0).
        v = r.summary.get(cfg["metric"])
        if cfg["metric"] == "eval_loss" and v is not None and float(v) > 0.1:
            print(f"!! skipping diverged {opt} run {r.id}: {cfg['metric']}={v:.3f}")
            continue
        runs_by_opt[opt].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    ax_step, ax_time, ax_bar = axes
    cmap = plt.cm.tab10.colors

    finals = []  # (opt, mean, std, total_time_mean, color)
    for color, opt in zip(cmap, OPT_ORDER):
        runs = runs_by_opt.get(opt, [])
        if not runs:
            print(f"!! {task}/{opt}: no finished runs"); continue
        agg = aggregate(runs, cfg["metric"])
        if agg is None:
            print(f"!! {task}/{opt}: no common eval points"); continue
        steps, t_mean, _, v_mean, v_std = agg

        ax_step.plot(steps, v_mean, color=color, lw=1.6, label=opt)
        ax_step.fill_between(steps, v_mean - v_std, v_mean + v_std,
                             color=color, alpha=0.18)

        ax_time.plot(t_mean / 60, v_mean, color=color, lw=1.6, label=opt)
        ax_time.fill_between(t_mean / 60, v_mean - v_std, v_mean + v_std,
                             color=color, alpha=0.18)

        finals.append((opt, float(v_mean[-1]), float(v_std[-1]),
                       float(t_mean[-1]) / 60, color))

    for ax, xlabel, title in [
        (ax_step, "iter", f"{task}: {cfg['metric']} vs iter"),
        (ax_time, "wall time (min, post model+data init)",
         f"{task}: {cfg['metric']} vs wall time"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(cfg["ylabel"])
        if cfg["log_y"]:
            ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, which="both")

    finals.sort(key=lambda x: x[1])
    names  = [x[0] for x in finals]
    means  = [x[1] for x in finals]
    stds   = [x[2] for x in finals]
    times  = [x[3] for x in finals]
    colors = [x[4] for x in finals]
    ax_bar.barh(names[::-1], means[::-1], xerr=stds[::-1],
                color=colors[::-1], alpha=0.85, capsize=4)
    ax_bar.set_xlabel(f"final {cfg['metric']}")
    ax_bar.set_title("final mean +/- std (best -> worst)")
    for i, (m, s, t) in enumerate(zip(means[::-1], stds[::-1], times[::-1])):
        ax_bar.text(m, i, f"  {m:.4f}  ({t:.0f} min)", va="center", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out = OUT / f"{task}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")
    print(f"{'opt':<14}{'final':>10}  {'std':>8}   {'time(min)':>10}")
    for opt, m, s, t, _ in sorted(finals, key=lambda x: x[1]):
        print(f"{opt:<14}{m:>10.4f}  {s:>8.4f}   {t:>10.1f}")


def main(argv=None) -> int:
    tasks = (argv or sys.argv)[1:] or ["text8", "celeba"]
    for t in tasks:
        if t not in TASKS:
            print(f"unknown task: {t}; choose from {list(TASKS)}"); return 1
        plot_task(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
