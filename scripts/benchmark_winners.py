"""Re-train each tuning-phase winner with 3 seeds.

For each (task, optimizer), pulls the best-eval-metric run from W&B
(across one or more sweep IDs), then runs ``src.main`` at seeds
{42, 69, 1337} with the winning config and ``--max_iters=BENCH_ITERS``.

Outputs land under ``exps_bench/<task>/<optimizer>/<run-name>/``.

Usage:
    uv run python scripts/benchmark_winners.py mdlm
    uv run python scripts/benchmark_winners.py mdlm --iters 100000
    uv run python scripts/benchmark_winners.py celeba
    uv run python scripts/benchmark_winners.py mdlm celeba
"""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import wandb


ENTITY = "steeldream"
PROJECT = "diffusion-baselines-src"
SEEDS = [42, 69, 1337]
BENCH_ITERS = 32000   # ~50 epochs on CelebA at bs=256 (matches KimRass/DDPM training scale).

SWEEPS = {
    "text8": {
        "adamw":         ["igof33it", "tgtbc95h"],
        "lion":          ["pamki85z"],
        "muon":          ["7dnh1zp5"],
        "schedule_free": ["fg27uabp"],
        "soap":          ["spak46n9"],
        "mars":          ["8qjdpiup"],
        "mars_m":        ["qfjs75ib"],
    },
    "celeba": {
        "adamw":         ["56m31eq5"],
        "lion":          ["j1ebowue"],
        "muon":          ["as9nta5h"],
        "schedule_free": ["oic5tul6"],
        "soap":          ["djv8ygo3"],
        "mars":          ["fkqfvxry"],
        "mars_m":        ["dw5olk1r"],
    },
}

METRIC = {"text8": "eval_bpc", "celeba": "eval_loss"}

# Task -> W&B project for the 3-seed bench runs. The text8 project name keeps
# its historical "mdlm" suffix so the W&B runs already deposited there remain
# queryable; only the local task identifier changed.
WANDB_PROJECTS = {
    "text8":  "diffusion-baselines-bench-mdlm",
    "celeba": "diffusion-baselines-bench-celeba",
}

PASSTHRU = {
    "task", "diffusion", "optimizer",
    "n_layer", "n_head", "d_model", "block_size",
    "batch_size", "schedule", "mdlm_eps", "dropout", "eps",
    "num_diffusion_steps", "beta_schedule",
    "ns_steps", "grad_clip",
    "lr", "weight_decay", "momentum", "gamma",
    "beta1", "beta2",
    "precondition_frequency", "max_precondition_dim", "shampoo_beta",
    "nesterov",
}


def winner(api, sids, metric):
    best, best_v = None, float("inf")
    for sid in sids:
        sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        for r in sw.runs:
            if r.state != "finished":
                continue
            v = r.summary.get(metric)
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                continue
            if f < best_v:
                best_v, best = f, r
    return best, best_v


def _argv(cfg, seed, task, opt, iters):
    argv = []
    if "betas_pair" in cfg and cfg["betas_pair"]:
        b1, b2 = cfg["betas_pair"]
        argv += ["--beta1", str(b1), "--beta2", str(b2)]
    for k, v in cfg.items():
        if k in PASSTHRU and v is not None:
            argv += [f"--{k}", str(v)]
    argv += [
        "--seed", str(seed),
        "--max_iters", str(iters),
        "--warmup_iters", str(max(50, iters // 10)),
        "--eval_interval", str(max(50, iters // 20)),
        "--eval_batches", "5",
        "--exp_root", f"exps_bench/{task}/{opt}",
        "--save_ckpt", "--ema_decay", "0.9999",
        "--wandb",
        "--wandb_project", WANDB_PROJECTS[task],
    ]
    return argv


def run_task(task: str, iters: int) -> None:
    api = wandb.Api()
    metric = METRIC[task]
    sweeps = SWEEPS.get(task, {})
    if not sweeps:
        print(f"!! no sweep IDs registered for {task} — fill in SWEEPS[{task!r}]")
        return
    print(f"\n=== {task} benchmark (3 seeds x {iters} iters) ===")
    Path("logs").mkdir(exist_ok=True)
    for opt, sids in sweeps.items():
        r, v = winner(api, sids, metric)
        if r is None:
            print(f"!! {opt}: no finished trials in {sids} — skip")
            continue
        cfg = dict(r.config)
        print(f"\n  {opt}: tune-best {metric}={v:.4f}  run={r.id}  "
              f"lr={cfg.get('lr')}  wd={cfg.get('weight_decay')}")
        for seed in SEEDS:
            log = Path(f"logs/bench_{task}_{opt}_s{seed}.log")
            argv = _argv(cfg, seed, task, opt, iters)
            print(f"    seed {seed} -> {log}", flush=True)
            t0 = time.time()
            with log.open("w") as f:
                rc = subprocess.call(
                    ["uv", "run", "python", "-m", "src.main", *argv],
                    stdout=f, stderr=subprocess.STDOUT,
                )
            dt = (time.time() - t0) / 60
            print(f"      exit={rc}  duration={dt:.1f} min", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*",
                   help="tasks to benchmark (default: all in SWEEPS)")
    p.add_argument("--iters", type=int, default=BENCH_ITERS,
                   help=f"max_iters per bench run (default {BENCH_ITERS})")
    p.add_argument("--only", nargs="+", default=None,
                   help="bench only these optimizers (skip the rest)")
    args = p.parse_args(argv)

    tasks = args.tasks or list(SWEEPS)
    for t in tasks:
        if t not in SWEEPS:
            print(f"unknown task: {t}; choose from {list(SWEEPS)}")
            return 1
    if args.only:
        for t in tasks:
            SWEEPS[t] = {k: v for k, v in SWEEPS[t].items() if k in args.only}
    for t in tasks:
        run_task(t, args.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
