"""Aggregate Phase-D benchmark runs into a (size, optimizer) bpc table + plot.

Reads ``exps/v3_benchmark/*/log.jsonl``, infers the size label from
``n_layer`` (4=small, 8=medium, 12=large), groups by (size, optimizer),
and produces:

- ``exps/v3_summary.csv`` — mean ± std eval_bpc per cell
- ``exps/v3_summary.png`` — grouped bar chart, one cluster per size,
  one bar per optimizer, with std error bars
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIZE_FROM_LAYER = {4: "small", 8: "medium", 12: "large"}
SIZE_ORDER = ["small", "medium", "large"]


def _parse_run(run_dir: Path):
    log = run_dir / "log.jsonl"
    if not log.exists():
        return None
    cfg = None
    last_bpc = None
    final_bpc = None
    with open(log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ev = rec.get("event")
            if ev == "config":
                cfg = rec.get("args", {})
            elif ev == "eval" and rec.get("eval_bpc") is not None:
                last_bpc = rec["eval_bpc"]
            elif ev == "final_eval" and rec.get("eval_bpc") is not None:
                final_bpc = rec["eval_bpc"]
    bpc = final_bpc if final_bpc is not None else last_bpc
    return {"cfg": cfg, "bpc": bpc}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exps_dir", default="exps/v3_benchmark")
    p.add_argument("--out_png", default="exps/v3_summary.png")
    p.add_argument("--out_csv", default="exps/v3_summary.csv")
    args = p.parse_args(argv)

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for d in sorted(Path(args.exps_dir).glob("*")):
        if not d.is_dir():
            continue
        rec = _parse_run(d)
        if rec is None or rec["cfg"] is None or rec["bpc"] is None:
            continue
        cfg = rec["cfg"]
        n_layer = cfg.get("n_layer")
        opt = cfg.get("optimizer")
        size = SIZE_FROM_LAYER.get(n_layer, f"L{n_layer}")
        groups[(size, opt)].append(float(rec["bpc"]))

    if not groups:
        print(f"no runs found under {args.exps_dir}")
        return 1

    optimizers = sorted({k[1] for k in groups})
    sizes = [s for s in SIZE_ORDER if any(k[0] == s for k in groups)]

    # CSV
    rows = ["size,optimizer,n_seeds,mean_eval_bpc,std_eval_bpc"]
    print(f"{'size':8s} {'optimizer':12s} {'n':>3s} {'mean':>8s} {'std':>8s}")
    print("-" * 45)
    for size in sizes:
        for opt in optimizers:
            vs = groups.get((size, opt), [])
            if not vs:
                continue
            m = float(np.mean(vs))
            s = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
            rows.append(f"{size},{opt},{len(vs)},{m:.6f},{s:.6f}")
            print(f"{size:8s} {opt:12s} {len(vs):>3d} {m:>8.4f} {s:>8.4f}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).write_text("\n".join(rows) + "\n")
    print(f"# wrote {args.out_csv}")

    # Plot: x = size, hue = optimizer
    width = 0.8 / max(1, len(optimizers))
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(1.6 + 2.5 * len(sizes), 5))
    for i, opt in enumerate(optimizers):
        means = []
        stds = []
        for size in sizes:
            vs = groups.get((size, opt), [])
            means.append(float(np.mean(vs)) if vs else np.nan)
            stds.append(float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, means,
               width=width, yerr=stds, capsize=3, label=opt, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("eval_bpc (lower is better)")
    ax.set_title("text8 MDLM — final eval_bpc by (size, optimizer)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=120)
    print(f"# wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
