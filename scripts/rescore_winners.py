"""Re-derive tune_<task>_winners.json from the raw trial logs.

A trial's log.jsonl is append-only and its directory can outlive a study reset,
so a log may hold runs from several generations (or, for a trial skipped as a
duplicate, someone else's run entirely). Scoring rule, same for every task: a
trial scores the best eval from the runs whose logged config matches the config
that trial was given. The winner is the lowest-scoring COMPLETE trial.

    uv run python scripts/rescore_winners.py text8 [celeba64 qm9 lm1b] [--write]

Without --write it only prints the diff against the current winners file.
"""
import argparse
import fcntl
import json
import math
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_local import search_space  # noqa: E402

MIN_TRIALS = 10   # below this a study is too young to overwrite a mature winner
OPTS = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]


def same(a, b):
    """Equal as config values: numerically if both are numbers, else as text
    (0 vs 0.0 from YAML/argparse must compare equal)."""
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def own_score(task, opt, n, metric, cfg):
    """Min `metric` over the runs in trial n's log whose config matches `cfg`.

    A trial directory can hold runs from more than one study generation, and a
    trial skipped as a duplicate never runs at all — so the log is only this
    trial's evidence where the logged args match the config it was given.
    """
    p = Path("exps_tuning", task, opt, f"trial_{n}", "log.jsonl")
    if not p.exists():
        return None
    best, run, mine = None, [], False
    for line in p.read_text().splitlines() + ['{"event": "config", "args": {}}']:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "config":
            if mine and run:                          # close the previous run
                best = min(run) if best is None else min(best, min(run))
            args = e.get("args", {})
            mine = all(same(args[k], v) for k, v in cfg.items() if k in args)
            run = []
        elif mine and e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
            run.append(e[metric])
    return best


def rescore(task, opt):
    """(value, trial_number, params) for the true best COMPLETE trial, or None."""
    db = Path("optuna_db", f"optuna_{task}_{opt}.db")
    if not db.exists():
        return None
    fixed, _, metric, _ = search_space(task, opt)
    study = optuna.load_study(study_name=f"{task}_{opt}", storage=f"sqlite:///{db}")
    best, n_scored = None, 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        cfg = {**fixed, **t.params}
        v = own_score(task, opt, t.number, metric, cfg)
        if v is None:
            continue
        n_scored += 1
        if best is None or v < best[0]:
            best = (v, t.number, cfg)
    return (*best, n_scored) if best else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="+")
    p.add_argument("--write", action="store_true", help="update logs/tune_<task>_winners.json")
    a = p.parse_args()

    for task in a.tasks:
        wins = Path("logs", f"tune_{task}_winners.json")
        old = json.loads(wins.read_text()) if wins.exists() else {}
        new, changed = {}, 0
        print(f"\n=== {task} ===")
        for opt in OPTS:
            r = rescore(task, opt)
            if r is None:
                continue
            value, n, params, n_scored = r
            prev = old.get(opt, {}).get("value")
            if not math.isfinite(value):
                print(f"  {opt:<14} {'skip':>10}  (best is non-finite)"); continue
            if prev is not None and n_scored < MIN_TRIALS and value > prev:
                print(f"  {opt:<14} {value:>10.4f}  skip: only {n_scored} scored trials "
                      f"(study still running?), keeping {prev:.4f}"); continue
            new[opt] = {"value": value, "params": params, "trial": n}
            if prev is None:
                print(f"  {opt:<14} {value:>10.4f}  (new)")
            elif abs(prev - value) < 1e-9:
                print(f"  {opt:<14} {value:>10.4f}  unchanged")
            else:
                changed += 1
                print(f"  {opt:<14} {value:>10.4f}  CORRECTED from {prev:.4f} "
                      f"({value - prev:+.4f}), trial {n}")
        print(f"  -> {changed} corrected of {len(new)}")
        if a.write and new:
            wins.parent.mkdir(exist_ok=True)
            with open(wins, "a+") as f:               # same lock the tuner takes
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0); cur = f.read()
                data = json.loads(cur) if cur.strip() else {}
                data.update(new)                      # merge: never drop an optimizer
                f.seek(0); f.truncate()
                f.write(json.dumps(data, indent=2))
            print(f"  wrote {wins} ({len(new)} entries merged, {len(data)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
