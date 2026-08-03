"""Benchmark the tuned winners: retrain each optimizer's best config at the full
budget. Reads winners from logs/tune_<task>_winners.json (written by
scripts/tune_local.py); writes each run to exps_bench/<task>/<opt>/seed<seed>/.
Defaults to one seed; add more later with --seeds — finished seeds are skipped.

    uv run python scripts/benchmark_winners.py qm9 [--iters N] [--seeds 42 69 1337] [--only adamw lion]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42]   # default: one seed. Add more with --seeds 42 69 1337 (finished seeds are skipped)
BENCH_ITERS = {"text8": 30000, "qm9": 25000, "lm1b": 30000, "celeba64": 100000}
METRIC = {"text8": "eval_bpc", "qm9": "eval_bpc", "lm1b": "eval_bpc",
          "celeba64": "eval_loss"}
OPT_ORDER = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]

# config keys forwarded to src.main; budget/seed/out are derived per bench run below
PASSTHRU = {
    "task", "diffusion", "optimizer", "lr", "adamw_lr", "weight_decay", "beta1", "beta2",
    "momentum", "gamma", "ns_steps", "precondition_frequency", "max_precondition_dim",
    "shampoo_beta", "nesterov", "eps", "grad_clip", "grad_accum", "schedule",
    "batch_size", "block_size", "n_layer", "n_head", "d_model", "time_emb_dim",
    "dropout", "mdlm_eps", "num_diffusion_steps", "beta_schedule",
    "image_size", "ema_eval",
}


def out_dir(task, opt, seed):
    return Path("exps_bench") / task / opt / f"seed{seed}"


def is_done(d, iters):
    """A run is finished once it logged an eval at (or past) the target iters.

    (The trainer only emits a distinct ``final_eval`` when max_iters isn't a
    multiple of eval_interval; otherwise the terminal eval is a plain ``eval``.)
    """
    log = d / "log.jsonl"
    if not log.exists():
        return False
    for line in log.read_text().splitlines():
        if '"event": "eval"' in line or '"event": "final_eval"' in line:
            try:
                if json.loads(line).get("iter", 0) >= iters:
                    return True
            except json.JSONDecodeError:
                pass
    return False


def build_argv(cfg, seed, task, opt, iters, extra=None):
    out = [s for k, v in cfg.items() if k in PASSTHRU and v is not None
           for s in (f"--{k}", str(v))]
    out += [
        "--seed", str(seed),
        "--max_iters", str(iters),
        "--warmup_iters", str(max(50, iters // 10)),
        "--eval_interval", str(max(50, iters // 20)),
        "--eval_batches", "5",
        "--out_dir", str(out_dir(task, opt, seed)),
        "--save_ckpt", "--ema_decay", "0.9999",
    ]
    return out + list(extra or [])


def run_task(task, iters, seeds, only=None, extra=None):
    wins = Path("logs") / f"tune_{task}_winners.json"
    if not wins.exists():
        print(f"!! {wins} not found — run `bash scripts/tune.sh <gpu> {task}` first")
        return
    winners = json.loads(wins.read_text())
    print(f"\n=== {task} benchmark ({len(seeds)} seed(s) {seeds} x {iters} iters) ===")
    # seed-outer / optimizer-inner: finish every optimizer at one seed before
    # moving to the next, so stopping early still yields a full 1-seed sweep.
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        for opt in OPT_ORDER:
            if opt not in winners or (only and opt not in only):
                continue
            cfg = winners[opt]["params"]
            d = out_dir(task, opt, seed)
            if is_done(d, iters):
                print(f"  {opt} seed {seed} -> already done, skipping", flush=True)
                continue
            if d.exists():                     # partial run: wipe (the logger appends)
                shutil.rmtree(d, ignore_errors=True)
            print(f"  {opt} seed {seed}: tune-best {METRIC[task]}={winners[opt]['value']:.4f}  "
                  f"lr={cfg.get('lr')}  wd={cfg.get('weight_decay')} -> {d}/log.jsonl", flush=True)
            t0 = time.time()
            # inherit stdout (-u = unbuffered) so the trainer's `iter | loss | lr`
            # streams live; benchmark.sh tees the whole stream to logs/bench_<task>.log
            rc = subprocess.call(
                [sys.executable, "-u", "-m", "src.main",
                 *build_argv(cfg, seed, task, opt, iters, extra)],
                stderr=subprocess.STDOUT)
            print(f"      exit={rc}  {(time.time() - t0) / 60:.1f} min", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*", help="default: all")
    p.add_argument("--iters", type=int, default=None, help="full-budget iters (default: per task)")
    p.add_argument("--only", nargs="+", help="bench only these optimizers")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                   help="seeds to run (default: one seed; finished seeds are skipped)")
    p.add_argument("--fid_n", type=int, default=0,
                   help="CelebA: sample N images for in-training FID (0 = off)")
    p.add_argument("--fid_steps", type=int, default=50, help="CelebA FID: DDIM steps")
    p.add_argument("--fid_stride", type=int, default=1, help="CelebA FID: every Nth eval")
    a = p.parse_args(argv)
    extra = (["--fid_n", str(a.fid_n), "--fid_steps", str(a.fid_steps),
              "--fid_stride", str(a.fid_stride)] if a.fid_n > 0 else None)
    Path("logs").mkdir(exist_ok=True)
    for t in (a.tasks or list(METRIC)):
        if t not in METRIC:
            print(f"unknown task: {t}; choose from {list(METRIC)}")
            return 1
        run_task(t, a.iters or BENCH_ITERS[t], a.seeds, a.only, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
