"""Continue every bench run further by setting --resume and a higher --max_iters.

For each exps_bench/<task>/<opt>/<run-dir>/ that has ckpt.pt, re-launch src.main
with the SAME config from log.jsonl but max_iters=NEW_MAX (only if NEW_MAX > current iter_num).

Usage:
    uv run python scripts/continue_bench.py celeba64 100000
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PASSTHRU = {
    "task", "diffusion", "optimizer",
    "n_layer", "n_head", "d_model", "block_size",
    "batch_size", "schedule", "mdlm_eps", "dropout", "eps",
    "num_diffusion_steps", "beta_schedule",
    "ns_steps", "grad_clip",
    "lr", "weight_decay", "momentum", "gamma",
    "beta1", "beta2",
    "precondition_frequency", "max_precondition_dim", "shampoo_beta",
    "nesterov", "ema_decay",
}


BOOL_FLAGS = {"nesterov"}


def _argv(cfg, new_max, out_dir, task):
    argv = []
    for k, v in cfg.items():
        if k not in PASSTHRU or v is None:
            continue
        if k in BOOL_FLAGS:
            if v: argv += [f"--{k}"]
        else:
            argv += [f"--{k}", str(v)]
    argv += [
        "--seed", str(cfg["seed"]),
        "--max_iters", str(new_max),
        "--warmup_iters", str(cfg["warmup_iters"]),
        "--eval_interval", str(cfg["eval_interval"]),
        "--eval_batches", str(cfg.get("eval_batches", 5)),
        "--out_dir", str(out_dir),
        "--save_ckpt", "--resume",
    ]
    return argv


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    task = sys.argv[1]; new_max = int(sys.argv[2])
    root = Path(f"exps_bench/{task}")
    if not root.exists():
        print(f"no {root}"); sys.exit(1)

    Path("logs").mkdir(exist_ok=True)
    for opt_dir in sorted(root.iterdir()):
        if not opt_dir.is_dir(): continue
        for run_dir in sorted(opt_dir.iterdir()):
            stable = run_dir / "ckpt_stable.pt"
            ckpt = run_dir / "ckpt.pt"
            log  = run_dir / "log.jsonl"
            if not (ckpt.exists() and log.exists()): continue
            # Resuming from the WSD stable-phase ckpt avoids the LR jump that a
            # post-decay ckpt would cause when total_iters is increased.
            if stable.exists():
                import shutil; shutil.copy(stable, ckpt)
                print(f"  using stable-phase ckpt for {run_dir.name}")
            with log.open() as f: cfg = json.loads(f.readline())["args"]
            import torch
            saved_iter = int(torch.load(ckpt, map_location="cpu", weights_only=False)["iter_num"])
            if saved_iter >= new_max:
                print(f"  skip {run_dir.name}  (iter={saved_iter} >= {new_max})"); continue
            last_eval = None
            for ln in log.open():
                try: x = json.loads(ln)
                except: continue
                if x.get("event") in ("eval", "final_eval"):
                    last_eval = x.get("eval_loss")
            if last_eval is not None and last_eval > 0.1:
                print(f"  skip diverged {run_dir.name}  (eval={last_eval:.3f})"); continue
            argv = _argv(cfg, new_max, run_dir, task)
            tag  = f"{opt_dir.name}_s{cfg['seed']}"
            log_path = Path(f"logs/cont_{task}_{tag}.log")
            print(f"\n  {tag}: {saved_iter} -> {new_max} iters, log={log_path}")
            t0 = time.time()
            with log_path.open("w") as f:
                rc = subprocess.call(["uv", "run", "python", "-m", "src.main", *argv],
                                     stdout=f, stderr=subprocess.STDOUT)
            print(f"    exit={rc}  duration={(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
