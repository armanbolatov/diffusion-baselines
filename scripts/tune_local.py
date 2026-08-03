"""Local Optuna hyperparameter tuning.

Per (task, optimizer): an Optuna study (TPE sampler + median pruning, resumable
SQLite, no server) samples the grid from configs/<task>/{common,<opt>}.yaml,
runs `src.main` as a subprocess, tails its log.jsonl to prune weak trials live,
and writes the winner to logs/tune_<task>_winners.json.

    uv run python scripts/tune_local.py <task> <optimizer> [--gpu G] [--n_trials N]
"""
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
import yaml

PENALTY = 1e9   # objective value for a trial that produced no finite eval


def search_space(task, opt):
    """Return (fixed, swept, metric_name, n_trials) from the merged config."""
    base = Path("configs") / task
    common = yaml.safe_load((base / "common.yaml").read_text())
    extra = yaml.safe_load((base / f"{opt}.yaml").read_text())
    params = {**common.get("parameters", {}), **extra.get("parameters", {})}
    fixed, swept = {}, {}
    for k, spec in params.items():
        if k in ("exp_root", "seed"):                  # out_dir/seed are set per trial
            continue
        (swept if "values" in spec else fixed)[k] = spec.get("values", spec.get("value"))
    metric = (extra.get("metric") or common["metric"])["name"]
    return fixed, swept, metric, common.get("run_cap", 100)


def read_events(path, pos):
    """Yield new complete JSON events from `path`, advancing byte offset pos[0]."""
    if not path.exists():
        return
    with path.open() as f:
        f.seek(pos[0]); data = f.read(); pos[0] = f.tell()
    if data and not data.endswith("\n"):               # hold back a partial last line
        cut = data.rfind("\n") + 1
        pos[0] -= len(data) - cut; data = data[:cut]
    for line in data.splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass


def terminate(proc, grace=10):
    """SIGTERM then SIGKILL the trial's process group; frees its GPU memory."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
            proc.wait(grace)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def run_trial(trial, fixed, swept, metric, task, opt, env):
    """Launch src.main, tail its log, prune live, return the best (min) eval."""
    sampled = {k: trial.suggest_categorical(k, v) for k, v in swept.items()}
    for t in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if t.number != trial.number and t.params == trial.params:
            return t.value                     # duplicate config already evaluated — reuse, no GPU
    out_dir = Path("exps_tuning") / task / opt / f"trial_{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [s for k, v in {**fixed, **sampled}.items() for s in (f"--{k}", str(v))]
    argv += ["--seed", "0", "--out_dir", str(out_dir)]
    print(f"--- {opt} trial {trial.number}: "
          + "  ".join(f"{k}={v}" for k, v in sampled.items()), flush=True)
    # inherit stdout/stderr so the trainer's live `iter | loss | lr` prints show
    proc = subprocess.Popen([sys.executable, "-m", "src.main", *argv], env=env,
                            stderr=subprocess.STDOUT, start_new_session=True)
    log, pos, best, seen = out_dir / "log.jsonl", [0], math.inf, set()
    try:
        while True:
            done = proc.poll() is not None
            for ev in read_events(log, pos):
                if ev.get("event") not in ("eval", "final_eval"):
                    continue
                m, step = ev.get(metric), int(ev.get("iter", 0))
                if m is None or not math.isfinite(m):
                    continue
                best = min(best, m)
                if step not in seen:                   # report the raw curve to the pruner
                    seen.add(step); trial.report(m, step)
                if trial.should_prune():
                    terminate(proc); raise optuna.TrialPruned()
            if done:
                return best if math.isfinite(best) else PENALTY
            time.sleep(2)
    finally:
        terminate(proc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", help="config dir: qm9 | text8 | celeba64")
    p.add_argument("optimizer")
    p.add_argument("--gpu")
    p.add_argument("--n_trials", type=int, help="default: run_cap from the config")
    a = p.parse_args()

    fixed, swept, metric, run_cap = search_space(a.task, a.optimizer)
    Path("optuna_db").mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu

    study = optuna.create_study(
        study_name=f"{a.task}_{a.optimizer}",
        storage=f"sqlite:///optuna_db/optuna_{a.task}_{a.optimizer}.db",   # per-opt DB: parallel-safe
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=int(fixed.get("eval_interval", 1)) or 1,
            max_resource=int(fixed.get("max_iters", 1)), reduction_factor=3))
    target = a.n_trials or run_cap
    done = sum(t.state.is_finished() for t in study.trials)        # resume: skip finished
    study.optimize(lambda t: run_trial(t, fixed, swept, metric, a.task, a.optimizer, env),
                   n_trials=max(0, target - done), gc_after_trial=True)

    wins = Path("logs") / f"tune_{a.task}_winners.json"
    wins.parent.mkdir(exist_ok=True)
    with open(wins, "a+") as f:                    # lock: other opts of this task write here too
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); cur = f.read()
        data = json.loads(cur) if cur.strip() else {}
        data[a.optimizer] = {"value": study.best_value,
                             "params": {**fixed, **study.best_params},
                             "trial": study.best_trial.number}
        f.seek(0); f.truncate(); f.write(json.dumps(data, indent=2))
    print(f"{a.optimizer}: best {metric}={study.best_value:.4f}  ->  {wins}")


if __name__ == "__main__":
    main()
