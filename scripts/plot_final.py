"""Plot the 3-seed benchmark: metric vs iter + vs training FLOPs + final bar chart.

Reads exps_bench/<task>/<opt>/*/log.jsonl (no network). For each task a 3-panel
figure: [ iter vs metric ] [ training FLOPs vs metric ] [ final bar chart ], each
curve mean +/- std over the seeds. FLOPs come from logs/flops.json (measured
forward x3 + optimizer-step overhead). The bar reports the metric at the LAST
eval (not the trajectory minimum) and annotates each optimizer's total FLOPs and
wall-clock; wall-clock is only comparable across runs on the same dedicated GPU.

    uv run python scripts/plot_final.py qm9 [text8 celeba64]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = {
    "text8":  {"metric": "eval_bpc",  "ylabel": "eval bpc", "log_y": True},
    "qm9":    {"metric": "eval_bpc",  "ylabel": "eval bpc", "log_y": True},
    "celeba64": {"metric": "eval_fid", "ylabel": "FID", "log_y": False},
    "lm1b":   {"metric": "eval_bpc",  "ylabel": "eval bpc", "log_y": True},
}
OPT_ORDER = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
OUT = Path("exps/final_plots")
OUT.mkdir(parents=True, exist_ok=True)


def read_triples(log_path, metric):
    """(iter, elapsed_s, metric) from a run's eval events, deduped on iter."""
    seen, out = set(), []
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") not in ("eval", "final_eval"):
            continue
        m, t, it = e.get(metric), e.get("elapsed_s"), e.get("iter")
        if m is None or t is None or it in seen:
            continue
        seen.add(it)
        out.append((it, float(t), float(m)))
    return out


def flops_table(task):
    """Per-optimizer FLOPs/iter from logs/flops.json (scripts/compute_flops.py):
    measured backbone + per-optimizer overhead (Newton-Schulz, SOAP rotations)."""
    p = Path("logs/flops.json")
    if not p.exists():
        print("!! logs/flops.json missing — run scripts/compute_flops.py for the FLOPs axis")
        return {}
    return json.loads(p.read_text()).get(task, {})


def aggregate(series):
    """Stack a list of (iter, elapsed_s, metric) sequences (one per seed)."""
    series = [s for s in series if s]
    n = min((len(s) for s in series), default=0)
    if n == 0:
        return None
    steps = np.array([series[0][i][0] for i in range(n)])
    times = np.array([[s[i][1] for i in range(n)] for s in series])
    vals = np.array([[s[i][2] for i in range(n)] for s in series])
    return steps, times.mean(0), times.std(0), vals.mean(0), vals.std(0)


def plot_task(task: str) -> None:
    cfg = TASKS[task]
    metric = cfg["metric"]
    by_opt = defaultdict(list)
    for log in sorted(Path("exps_bench", task).glob("*/*/log.jsonl")):
        opt = log.parts[-3]                                   # exps_bench/<task>/<opt>/<run>/
        if opt not in OPT_ORDER:
            continue
        tr = read_triples(log, metric)
        if not tr:
            continue
        if metric == "eval_loss" and tr[-1][2] > 0.1:         # skip diverged image runs
            print(f"!! skipping diverged {opt}: {metric}={tr[-1][2]:.3f}")
            continue
        by_opt[opt].append(tr)

    fpi_by_opt = flops_table(task)

    fig, (ax_step, ax_flop, ax_bar) = plt.subplots(
        1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    cmap = plt.cm.tab10.colors
    finals = []
    for color, opt in zip(cmap, OPT_ORDER):
        agg = aggregate(by_opt.get(opt, []))
        if agg is None:
            print(f"!! {task}/{opt}: no runs"); continue
        steps, t_mean, _, v_mean, v_std = agg
        ax_step.plot(steps, v_mean, color=color, lw=1.6, label=opt)
        ax_step.fill_between(steps, v_mean - v_std, v_mean + v_std, color=color, alpha=0.18)
        fpi = fpi_by_opt.get(opt)
        if fpi:
            pflops = steps * fpi / 1e15
            ax_flop.plot(pflops, v_mean, color=color, lw=1.6, label=opt)
            ax_flop.fill_between(pflops, v_mean - v_std, v_mean + v_std, color=color, alpha=0.18)
        finals.append((opt, float(v_mean[-1]), float(v_std[-1]),    # FINAL eval, not trajectory min
                       float(steps[-1] * fpi / 1e15) if fpi else 0.0,
                       float(t_mean[-1]) / 60, color))

    for ax, xlabel, title in [
        (ax_step, "iter", f"{task}: {metric} vs iter"),
        (ax_flop, "training PFLOPs (measured fwd·3 + optimizer step)", f"{task}: {metric} vs FLOPs"),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel(cfg["ylabel"]); ax.set_xlim(left=0)
        if cfg["log_y"]:
            ax.set_yscale("log")
        ax.set_title(title); ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, which="both")

    finals.sort(key=lambda x: x[1])
    names, means, stds, pflops, mins, colors = zip(*finals) if finals else ([],) * 6
    ax_bar.barh(names[::-1], means[::-1], xerr=stds[::-1], color=colors[::-1], alpha=0.85, capsize=4)
    ax_bar.set_xlabel(f"final {metric}")
    ax_bar.set_title("final mean +/- std (best -> worst)")
    for i, (m, pf, mn) in enumerate(zip(means[::-1], pflops[::-1], mins[::-1])):
        ax_bar.text(m, i, f"  {m:.4f}  ({pf:.0f} PFLOPs, {mn:.0f} min)", va="center", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUT / f"{task}.png", dpi=120)
    plt.close()
    print(f"wrote {OUT / f'{task}.png'}")
    for opt, m, s, pf, mn, _ in finals:
        print(f"{opt:<14}{m:>10.4f}  +/-{s:>7.4f}   {pf:>7.1f} PFLOPs  {mn:>6.0f} min")


def main(argv=None) -> int:
    for t in (argv or sys.argv[1:]) or ["qm9", "text8", "celeba64", "lm1b"]:
        if t not in TASKS:
            print(f"unknown task: {t}; choose from {list(TASKS)}")
            return 1
        plot_task(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
