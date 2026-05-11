"""Ingest exps/*/log.jsonl → per-(task,optimizer) mean ± std table + plot."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_run(run_dir: Path):
    log = run_dir / "log.jsonl"
    if not log.exists():
        return None
    cfg = None
    final_eval = None
    last_eval = None
    iters = []
    with open(log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ev = r.get("event")
            if ev == "config":
                cfg = r.get("args", {})
            elif ev == "eval":
                last_eval = r
            elif ev == "final_eval":
                final_eval = r
            elif ev == "iter":
                iters.append((r.get("iter"), r.get("loss")))
    return {"cfg": cfg, "last_eval": last_eval, "final_eval": final_eval, "iters": iters}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exps_dir", default="exps")
    p.add_argument("--out", default="exps/summary.png")
    args = p.parse_args()

    runs = []
    for d in sorted(Path(args.exps_dir).iterdir()):
        if not d.is_dir():
            continue
        r = parse_run(d)
        if r is None or r["cfg"] is None:
            continue
        runs.append((d.name, r))

    # group by (task, optimizer)
    groups = defaultdict(list)
    for name, r in runs:
        task = r["cfg"].get("task")
        opt = r["cfg"].get("optimizer")
        finalr = r.get("final_eval") or r.get("last_eval")
        if finalr is None:
            continue
        # use whichever metric exists
        v = finalr.get("eval_loss", finalr.get("eval_bpc"))
        if v is None:
            continue
        groups[(task, opt)].append(v)

    # table
    print(f"{'task':18s}  {'optimizer':14s}  {'n':>3s}  {'mean':>10s}  {'std':>10s}")
    print("-" * 65)
    rows = []
    for (task, opt), vs in sorted(groups.items()):
        m = float(np.mean(vs))
        s = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
        rows.append((task, opt, len(vs), m, s))
        print(f"{task:18s}  {opt:14s}  {len(vs):>3d}  {m:>10.4f}  {s:>10.4f}")

    # bar plot per task
    tasks = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    for ti, task in enumerate(tasks):
        sub = [r for r in rows if r[0] == task]
        sub.sort(key=lambda r: r[3])  # best first
        labels = [r[1] for r in sub]
        means = [r[3] for r in sub]
        stds = [r[4] for r in sub]
        x = np.arange(len(labels))
        axes[0, ti].barh(x, means, xerr=stds, color="steelblue", alpha=0.8, capsize=4)
        axes[0, ti].set_yticks(x)
        axes[0, ti].set_yticklabels(labels)
        axes[0, ti].invert_yaxis()
        axes[0, ti].set_title(task)
        axes[0, ti].set_xlabel("final eval metric (lower better)")
        axes[0, ti].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
