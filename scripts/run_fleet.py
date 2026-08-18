"""Run every tune + benchmark job across a set of GPUs, one job per GPU.

Independent jobs (task × optimizer for tuning; × seed for benchmarking) are
spread over the given GPUs — a GPU picks up the next job when it frees up.
Resumable: the tuner resumes its per-optimizer study, the benchmark skips
finished seeds. Logs land in logs/fleet/.

    uv run python scripts/run_fleet.py --gpus 0 1 2 3
    uv run python scripts/run_fleet.py --gpus 0 1 --phase tune --tasks text8
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

OPTS  = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
TASKS = ["qm9", "text8", "celeba64", "lm1b"]
SEEDS = [42, 69, 1337]
FID   = {"celeba64": ["--fid_n", "1024", "--fid_steps", "50"]}
PREP  = {"qm9":   [sys.executable, "-m", "src.data.qm9", "--block_size", "32"],
         "text8": [sys.executable, "-m", "src.data.text8"],
         "lm1b":  [sys.executable, "-m", "src.data.lm1b", "--block_size", "128"]}
# celeba downloads itself on first use
LOGDIR = Path("logs/fleet")


def jobs_for(phase, tasks, only=None):
    opts = [o for o in OPTS if not only or o in only]
    if phase == "tune":
        return [(f"tune_{t}_{o}", [sys.executable, "scripts/tune_local.py", t, o])
                for t in tasks for o in opts]
    return [(f"bench_{t}_{o}_s{s}",
             [sys.executable, "scripts/benchmark_winners.py", t, "--only", o,
              "--seeds", str(s), *FID.get(t, [])])
            for t in tasks for o in opts for s in SEEDS]


def worker(gpu, q):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    while True:
        try:
            name, cmd = q.get_nowait()
        except queue.Empty:
            return
        log = LOGDIR / f"{name}.log"
        print(f"[gpu{gpu}] start {name}", flush=True)
        with log.open("a") as f:                  # append: a restart keeps history
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:              # tee: full log to file, prefixed to console
                f.write(line)
                f.flush()
                print(f"[gpu{gpu}] {line}", end="", flush=True)
            rc = proc.wait()
        print(f"[gpu{gpu}] done  {name} (exit {rc}) -> {log}", flush=True)


def run(phase, tasks, gpus, only=None):
    q = queue.Queue()
    for job in jobs_for(phase, tasks, only):
        q.put(job)
    print(f"=== {phase}: {q.qsize()} jobs on GPUs {gpus} ===", flush=True)
    threads = [threading.Thread(target=worker, args=(g, q)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--phase", choices=["tune", "bench", "all"], default="all")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--only", nargs="+", help="restrict to these optimizers")
    a = p.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    phases = ["tune", "bench"] if a.phase == "all" else [a.phase]
    if "tune" in phases:                          # one-time data prep before parallel tuning
        for t in a.tasks:
            if t in PREP:
                print(f"prep {t}", flush=True); subprocess.call(PREP[t])
    for phase in phases:
        run(phase, a.tasks, a.gpus, a.only)
    print("fleet complete", flush=True)


if __name__ == "__main__":
    main()
