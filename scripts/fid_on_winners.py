"""After image-DDPM sweep finishes, re-run each optimizer's top config with FID.

For each (optimizer, top-1 config) pulled from W&B:
- Resume the same hyperparameters
- Train for the same iter budget (3000)
- At final eval, compute FID with --fid_samples 2048 --fid_steps 50

Writes one JSONL log per run under exps/image_fid_eval/ and a final
exps/image_fid_eval/summary.csv comparing optimizers.

Usage:
    uv run python scripts/fid_on_winners.py
"""
import json
import subprocess
import sys
from pathlib import Path

import wandb


SWEEPS = {
    "prodigy":   "4r2y0zqn",
    "muon":      "ek8s7g65",
    "adamw":     "284pbx2p",
    "mars_adam": "mi6mupw3",
    "soap":      "4y7odkbk",
}
ENTITY = "steeldream"
PROJECT = "diffusion-baselines-src"

# Hyperparams that need to be passed back as CLI flags. Everything else
# is either fixed (model, schedule, iters) or excluded (run metadata).
TUNED_KEYS = {
    "lr", "weight_decay", "warmup_iters", "grad_clip",
    "beta1", "beta2", "momentum",
    "precondition_frequency", "shampoo_beta",
    "gamma", "d_coef",
    # AdamW betas_pair gets expanded into beta1/beta2 below.
}
SHARED_KEYS = {
    "task", "diffusion", "optimizer", "model",
    "num_diffusion_steps", "beta_schedule",
    "batch_size", "max_iters", "schedule",
    "eval_interval", "eval_batches", "dropout", "eps",
}


def _winner(api, sid: str) -> dict:
    sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
    best = None
    best_v = float("inf")
    for r in sw.runs:
        if r.state != "finished":
            continue
        v = r.summary.get("eval_loss")
        if v is None:
            continue
        if float(v) < best_v:
            best_v = float(v); best = r
    if best is None:
        return None
    cfg = dict(best.config)
    cfg["_run_id"] = best.id
    cfg["_eval_loss"] = best_v
    return cfg


def _argv_from_cfg(cfg: dict, seed: int) -> list[str]:
    argv: list[str] = []
    keys = (TUNED_KEYS | SHARED_KEYS) - {"task", "diffusion", "optimizer", "model"}
    # Always-on shape
    argv += ["--task", str(cfg["task"]),
             "--diffusion", str(cfg["diffusion"]),
             "--optimizer", str(cfg["optimizer"]),
             "--model", str(cfg["model"])]
    if "betas_pair" in cfg:
        b1, b2 = cfg["betas_pair"]
        argv += ["--beta1", str(b1), "--beta2", str(b2)]
    for k in keys:
        if k in cfg and cfg[k] is not None:
            argv += [f"--{k}", str(cfg[k])]
    # FID at final eval
    argv += ["--fid_samples", "2048", "--fid_steps", "50",
             "--seed", str(seed),
             "--exp_root", "exps/image_fid_eval"]
    return argv


def main() -> int:
    api = wandb.Api()
    Path("exps/image_fid_eval").mkdir(parents=True, exist_ok=True)
    summary = []
    for name, sid in SWEEPS.items():
        cfg = _winner(api, sid)
        if cfg is None:
            print(f"!! no finished runs for {name}/{sid}; skipping")
            continue
        argv = _argv_from_cfg(cfg, seed=0)
        print(f"=== {name}: best eval_loss={cfg['_eval_loss']:.4f} ===")
        print("  argv:", " ".join(argv))
        cmd = ["uv", "run", "python", "-m", "src.main", *argv]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! {name} run exit={rc}")
            summary.append((name, cfg["_eval_loss"], None, None))
            continue
        # parse the resulting log.jsonl for final_eval fid
        run_dirs = sorted(Path("exps/image_fid_eval").glob(f"*_{name}_*"))
        log = (run_dirs[-1] / "log.jsonl") if run_dirs else None
        fid_val, eval_loss = None, None
        if log and log.exists():
            for line in log.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == "final_eval":
                    fid_val = rec.get("fid")
                    eval_loss = rec.get("eval_loss")
        summary.append((name, cfg["_eval_loss"], eval_loss, fid_val))

    # Print final table
    print()
    print(f"{'optimizer':12s} {'sweep_eval_loss':>16s} {'final_eval_loss':>16s} {'fid':>8s}")
    print("-" * 60)
    for name, sl, el, fid in summary:
        sl_s = f"{sl:.4f}" if sl is not None else "-"
        el_s = f"{el:.4f}" if el is not None else "-"
        fid_s = f"{fid:.2f}" if fid is not None else "-"
        print(f"{name:12s} {sl_s:>16s} {el_s:>16s} {fid_s:>8s}")

    # CSV
    Path("exps/image_fid_eval/summary.csv").write_text(
        "optimizer,sweep_eval_loss,final_eval_loss,fid\n"
        + "\n".join(f"{n},{sl},{el if el is not None else ''},{fid if fid is not None else ''}"
                    for n, sl, el, fid in summary)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
