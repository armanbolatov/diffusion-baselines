"""Results figure: 2x3 eval curves (rows = vs iter, vs FLOPs; cols = tasks) + a
summary table of final mean+/-std. No histograms.

    uv run python scripts/plot_grid.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OPT_ORDER = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
COLORS = dict(zip(OPT_ORDER, plt.cm.tab10.colors))
# task -> (metric, iters, ylabel, logy, ylim)
TASKS = {
    "qm9":    ("eval_bpc", 25000, "eval bpc", True,  None),
    "text8":  ("eval_bpc", 30000, "eval bpc", True,  None),
    "celeba64": ("eval_fid", 100000, "FID", False, None),
}
FLOPS = json.loads(Path("logs/flops.json").read_text())
OUT = Path("exps/final_plots/grid.png")


def curves(task, opt, metric, iters):
    runs = []
    for f in Path("exps_bench", task, opt).glob("seed*/log.jsonl"):
        pts = {}
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
                pts[e["iter"]] = e[metric]
        if pts and max(pts) >= iters:
            runs.append(pts)
    if not runs:
        return None
    steps = sorted(set.intersection(*[set(r) for r in runs]))
    mat = np.array([[r[s] for s in steps] for r in runs])
    return np.array(steps), mat.mean(0), mat.std(0)


fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.28, wspace=0.22)
finals = {}                                            # (task, opt) -> (mean, std)

for col, (task, (metric, iters, ylabel, logy, ylim)) in enumerate(TASKS.items()):
    ax_it, ax_fl = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
    for opt in OPT_ORDER:
        c = curves(task, opt, metric, iters)
        if c is None:
            continue
        steps, mean, std = c
        finals[(task, opt)] = (mean[-1], std[-1])      # FINAL eval, not trajectory min
        fpi = FLOPS[task][opt]
        for ax, x in [(ax_it, steps), (ax_fl, steps * fpi / 1e15)]:
            ax.plot(x, mean, color=COLORS[opt], lw=1.6, label=opt)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[opt], alpha=0.15)
    for ax, xlabel in [(ax_it, "iter"), (ax_fl, "training PFLOPs")]:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_xlim(left=0)
        ax.grid(alpha=0.3, which="both")
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
    ax_it.set_title(task, fontsize=12)

fig.legend([plt.Line2D([], [], color=COLORS[o], lw=2) for o in OPT_ORDER],
           OPT_ORDER, loc="upper center", ncol=7, fontsize=10, bbox_to_anchor=(0.5, 0.955))

# ---- summary table ----
ax_t = fig.add_subplot(gs[2, :]); ax_t.axis("off")
cols = ["QM9  bpc ↓", "text8  bpc ↓", "CelebA  FID ↓"]
rows, cells = [], []
for opt in OPT_ORDER:
    rows.append(opt)
    row = []
    for task, nd in [("qm9", 4), ("text8", 4), ("celeba64", 2)]:
        m, s = finals.get((task, opt), (None, None))
        row.append("—" if m is None else f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s else ""))
    cells.append(row)
tbl = ax_t.table(cellText=cells, rowLabels=rows, colLabels=cols,
                 cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
# highlight best (lowest mean) per column
for j, task in enumerate(["qm9", "text8", "celeba64"]):
    vals = [(i, finals[(task, o)][0]) for i, o in enumerate(OPT_ORDER) if (task, o) in finals]
    best_i = min(vals, key=lambda x: x[1])[0]
    tbl[(best_i + 1, j)].set_facecolor("#c8e6c9")
for i, o in enumerate(OPT_ORDER):
    tbl[(i + 1, -1)].set_facecolor(COLORS[o] + (0.25,))

fig.suptitle("Optimizer benchmark — eval vs iter (top) and vs FLOPs (mid); final metric (table)",
             y=0.99, fontsize=13)
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"wrote {OUT}")
