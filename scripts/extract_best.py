"""Extract the winning hyperparameters from a completed W&B sweep.

Queries the W&B public API for all finished runs in a sweep, picks the
one with the lowest final ``eval_bpc``, and writes its config to a JSON
file in ``configs/best/`` (or stdout).

Usage:
    uv run python scripts/extract_best.py \\
        --sweep <entity>/<project>/<sweep_id> \\
        --out configs/best/text8_small_adamw.json
    uv run python scripts/extract_best.py \\
        --sweep <sweep_id> --project diffusion-baselines  # entity from wandb default

Also supports ``--top K`` to print the top-K runs.
"""
import argparse
import json
import sys
from pathlib import Path


# Hyperparameters we care about persisting from wandb.config. Other
# fields (e.g. derived flags, system info) are dropped.
TUNED_KEYS = [
    "lr", "weight_decay", "grad_clip", "warmup_iters",
    "beta1", "beta2", "betas_pair", "momentum",
    "precondition_frequency", "shampoo_beta",
    "gamma", "d_coef",
    # context (so downstream scripts know the cell):
    "task", "diffusion", "optimizer",
    "n_layer", "n_head", "d_model", "block_size",
    "batch_size", "schedule", "ns_steps", "mdlm_eps", "dropout", "eps",
]


def _final_metric(run, metric: str) -> float | None:
    try:
        hist = run.history(keys=[metric], pandas=False)
    except Exception:
        return None
    vals = [row[metric] for row in hist if row.get(metric) is not None]
    if not vals:
        return run.summary.get(metric)
    return float(vals[-1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True,
                   help="sweep id, optionally prefixed with entity/project/")
    p.add_argument("--project", default="diffusion-baselines",
                   help="used only if --sweep is the bare id")
    p.add_argument("--entity", default=None)
    p.add_argument("--metric", default="eval_bpc")
    p.add_argument("--top", type=int, default=1,
                   help="Print top-K runs; only the #1 run is written to --out.")
    p.add_argument("--out", default=None,
                   help="Path to write the winning config JSON. "
                        "If omitted, print to stdout.")
    args = p.parse_args(argv)

    # Lazy import: keep wandb optional for callers that never query.
    import wandb
    api = wandb.Api()

    sid = args.sweep
    if sid.count("/") < 2:
        entity = args.entity or api.default_entity
        sid = f"{entity}/{args.project}/{sid}"
    sweep = api.sweep(sid)

    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        print(f"no finished runs in sweep {sid}", file=sys.stderr)
        return 1

    scored = []
    for r in runs:
        val = _final_metric(r, args.metric)
        if val is None or val != val:  # skip NaN / missing
            continue
        scored.append((val, r))
    scored.sort(key=lambda t: t[0])

    if not scored:
        print(f"no runs with metric {args.metric} in sweep {sid}", file=sys.stderr)
        return 1

    print(f"# top {min(args.top, len(scored))} runs by {args.metric} (lower=better):")
    for i, (val, r) in enumerate(scored[: args.top]):
        print(f"  {i+1}. {r.name}  {args.metric}={val:.4f}  id={r.id}")

    best_val, best_run = scored[0]
    cfg = {k: v for k, v in best_run.config.items() if k in TUNED_KEYS}
    cfg["_sweep_id"] = sid
    cfg["_run_id"] = best_run.id
    cfg[args.metric] = best_val

    payload = json.dumps(cfg, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(f"# wrote {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
