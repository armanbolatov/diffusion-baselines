"""Validate the 20%-proxy tuning: does the proxy ranking of configs survive to full budget?

For one (task, optimizer): read the finished Optuna study (each completed trial's objective
IS its 20%-proxy metric), pick K configs spanning best->worst proxy value, retrain each at the
full benchmark budget with the same seed 0 tuning used, and report Kendall-tau between the proxy
and full-budget rankings. tau near 1 => training 5x longer ranks these configs the same way, so
selecting the winner at 20% is sound. Also reports where the proxy-best config lands at full
budget (the number that actually matters for HP selection).

    uv run python scripts/proxy_transfer.py text8 adamw [--k 8] [--gpu 0]

Cost: K full-budget runs on ONE task. Use text8 (the cheapest) unless you need another.
Run after tuning has populated optuna_db/optuna_<task>_<opt>.db.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_winners import BENCH_ITERS, PASSTHRU   # noqa: E402
from tune_local import PENALTY, search_space          # noqa: E402


def unique_by_params(trials):
    """Drop dedup-cached duplicates (same params) — keep the first, they share a value."""
    seen, out = set(), []
    for t in sorted(trials, key=lambda t: t.value):
        key = tuple(sorted(t.params.items()))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def pick(trials, k):
    """K trials spanning best->worst proxy value (endpoints always included)."""
    if len(trials) <= k:
        return trials
    idx = sorted({round(i * (len(trials) - 1) / (k - 1)) for i in range(k)})
    return [trials[i] for i in idx]


def full_metric(cfg, task, opt, i, iters, metric, env):
    """Run one config at full budget (seed 0); return best-over-curve metric (as tuned)."""
    d = Path("exps_transfer") / task / opt / f"cfg{i}"
    d.mkdir(parents=True, exist_ok=True)
    argv = [s for k, v in cfg.items() if k in PASSTHRU and v is not None
            for s in (f"--{k}", str(v))]
    argv += ["--seed", "0", "--max_iters", str(iters),
             "--warmup_iters", str(max(50, iters // 10)),
             "--eval_interval", str(max(50, iters // 20)),
             "--eval_batches", "5", "--ema_decay", "0.9999",
             "--out_dir", str(d)]
    subprocess.call([sys.executable, "-u", "-m", "src.main", *argv], env=env,
                    stderr=subprocess.STDOUT)
    best = math.inf
    for line in (d / "log.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
            best = min(best, e[metric])
    return best


def kendall_tau(a, b):
    """tau-b over concordant/discordant pairs (== tau-a when no ties; eval metrics rarely tie)."""
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            conc += s > 0
            disc += s < 0
    d = conc + disc
    return (conc - disc) / d if d else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("optimizer")
    p.add_argument("--k", type=int, default=8, help="configs to re-run at full budget")
    p.add_argument("--iters", type=int, default=None,
                   help="full-budget iters (default: per-task BENCH_ITERS; lower = cheaper check)")
    p.add_argument("--gpu")
    a = p.parse_args()

    fixed, _, metric, _ = search_space(a.task, a.optimizer)
    iters = a.iters or BENCH_ITERS[a.task]
    study = optuna.load_study(
        study_name=f"{a.task}_{a.optimizer}",
        storage=f"sqlite:///optuna_db/optuna_{a.task}_{a.optimizer}.db")
    done = unique_by_params([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE
                             and t.value is not None and t.value < PENALTY])
    if len(done) < 3:
        print(f"!! only {len(done)} usable complete trials — run tuning first")
        return 1
    chosen = pick(done, a.k)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu

    print(f"=== proxy-transfer {a.task}/{a.optimizer}: {len(chosen)} configs, "
          f"proxy vs full {iters} iters ({metric}) ===", flush=True)
    proxy, full = [], []
    for i, t in enumerate(chosen):
        t0 = time.time()
        m = full_metric({**fixed, **t.params}, a.task, a.optimizer, i, iters, metric, env)
        proxy.append(t.value)
        full.append(m)
        print(f"  cfg{i}: proxy={t.value:.4f}  full={m:.4f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    tau = kendall_tau(proxy, full)
    best_proxy = min(range(len(proxy)), key=lambda i: proxy[i])
    full_rank = sorted(range(len(full)), key=lambda i: full[i]).index(best_proxy) + 1
    print(f"\nKendall tau (proxy vs full) = {tau:+.3f}   (1 = perfect transfer)")
    print(f"proxy-best config ranks #{full_rank}/{len(full)} at full budget", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
