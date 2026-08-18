"""Edge audit: do tuned winners sit on the boundary of their search grid?

For each finished (task, optimizer) study, look at the top-K completed trials
and report, per swept hyperparameter, where the best value sits in its grid.
A winner on a grid edge (min or max of an extendable axis) means the true
optimum may lie outside the grid — widen that axis (new study) before
benchmarking. Binary toggles (2-value grids) are skipped: they have no
"outside".

    uv run python scripts/audit_edges.py                # audit every study in optuna_db/
    uv run python scripts/audit_edges.py text8 --top 5
"""
import argparse
import glob
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_local import PENALTY, search_space  # noqa: E402


def audit_study(task, opt, top_k):
    db = f"optuna_db/optuna_{task}_{opt}.db"
    try:
        study = optuna.load_study(study_name=f"{task}_{opt}",
                                  storage=f"sqlite:///{db}")
    except Exception:
        return None
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and t.value < PENALTY]
    if not done:
        return None
    # Use what the study ACTUALLY searched, not today's config: a grid edited
    # after the fact would make a boundary winner look interior.
    swept = {k: list(d.choices) for k, d in done[-1].distributions.items()
             if hasattr(d, "choices")}
    top = sorted(done, key=lambda t: t.value)[:top_k]
    best = top[0]

    rows, edges = [], 0
    for param, grid in sorted(swept.items()):
        vals = sorted(v for v in grid if isinstance(v, (int, float)))
        if len(vals) <= 2:                          # binary toggle: no outside
            continue
        if param == "weight_decay" and best.params.get(param) == vals[0] == 0:
            continue                                # wd=0 is a domain floor, not a grid limit
        b = best.params.get(param)
        at_min = sum(t.params.get(param) == vals[0] for t in top)
        at_max = sum(t.params.get(param) == vals[-1] for t in top)
        flag = ""
        if b == vals[0]:
            flag = f"EDGE-MIN ({at_min}/{len(top)} of top-{len(top)} at {vals[0]:g})"
        elif b == vals[-1]:
            flag = f"EDGE-MAX ({at_max}/{len(top)} of top-{len(top)} at {vals[-1]:g})"
        edges += bool(flag)
        rows.append((param, b, vals[0], vals[-1], flag))
    return len(done), best.value, rows, edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*", help="default: every task found in optuna_db/")
    p.add_argument("--top", type=int, default=5, help="top-K trials for the edge count")
    a = p.parse_args()

    pairs = []
    for db in sorted(glob.glob("optuna_db/optuna_*.db")):
        task, opt = Path(db).stem.removeprefix("optuna_").split("_", 1)
        if not a.tasks or task in a.tasks:
            pairs.append((task, opt))
    if not pairs:
        print("no studies found in optuna_db/")
        return 1

    total_edges = 0
    for task, opt in pairs:
        res = audit_study(task, opt, a.top)
        if res is None:
            print(f"\n{task}/{opt}: no completed trials")
            continue
        n, best, rows, edges = res
        total_edges += edges
        print(f"\n{task}/{opt}  ({n} completed, best={best:.4f})")
        for param, b, lo, hi, flag in rows:
            print(f"  {param:<14} best={b:<8g} grid=[{lo:g} .. {hi:g}]  {flag or 'interior'}")
    print(f"\n{'!! ' if total_edges else ''}{total_edges} edge winner(s)"
          + (" — consider widening those axes (fresh study per affected optimizer)"
             if total_edges else " — all winners interior"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
