"""Scatter per task: training FLOPs (x) vs peak GPU memory (y), color = final metric.
Overlapping points (same FLOPs+memory, e.g. adamw/schedule_free) are drawn as
split half-circles so both stay visible.

    uv run python scripts/plot_scatter.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle

TASKS = {"qm9": ("eval_bpc", 25000), "text8": ("eval_bpc", 30000),
         "celeba64": ("eval_fid", 100000)}
OPT_ORDER = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
FLOPS = json.loads(Path("logs/flops.json").read_text())
OUT = Path("exps/final_plots/scatter_flops_mem.png")


def run_stats(log, metric, iters):
    mem, final, done = 0.0, None, False
    for line in log.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "iter":
            mem = max(mem, e.get("peak_mem_mb") or 0)
        elif e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
            final = e[metric]                          # FINAL eval, not trajectory min
            done |= e["iter"] >= iters
    return (mem, final) if done and final is not None else None


fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
for ax, (task, (metric, iters)) in zip(axes, TASKS.items()):
    pts = []
    for opt in OPT_ORDER:
        runs = [run_stats(f, metric, iters)
                for f in Path("exps_bench", task, opt).glob("seed*/log.jsonl")]
        runs = [r for r in runs if r]
        if not runs or opt not in FLOPS.get(task, {}):
            continue
        pts.append((opt,
                    FLOPS[task][opt] * iters / 1e15,             # PFLOPs
                    float(np.mean([r[0] for r in runs])) / 1024,  # GB
                    float(np.mean([r[1] for r in runs]))))        # metric
    if not pts:
        continue
    norm = Normalize(min(p[3] for p in pts), max(p[3] for p in pts))
    cmap = plt.cm.viridis_r

    groups = defaultdict(list)                          # cluster near-coincident points
    for p in pts:
        groups[(round(p[1], 1), round(p[2], 2))].append(p)
    for grp in groups.values():
        fills = ["full"] if len(grp) == 1 else (["left", "right"] if len(grp) == 2
                 else ["full"] * len(grp))
        for k, (opt, x, y, val) in enumerate(grp):
            xk = x if len(grp) <= 2 else x * (1 + 0.006 * (k - len(grp) / 2))
            ax.plot(xk, y, linestyle="none", markersize=17,
                    marker=MarkerStyle("o", fillstyle=fills[k]),
                    markerfacecolor=cmap(norm(val)), markeredgecolor="k", zorder=3)
            dy = 12 if k % 2 == 0 else -16
            ax.annotate(opt, (x, y), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=9)
    fig.colorbar(ScalarMappable(norm, cmap), ax=ax,
                 label=f"final {metric} (darker = better)")
    ax.set_xlabel("training PFLOPs"); ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title(task); ax.grid(alpha=0.3); ax.margins(0.15)

fig.suptitle("Compute vs memory vs quality (color) — split circles = overlapping points", y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"wrote {OUT}")
