#!/usr/bin/env bash
# Full run. Retrains each tuned winner at the full budget, on a single GPU.
# Defaults to ONE seed; add more later with --seeds (finished seeds are skipped).
# Run this AFTER scripts/tune.sh. Progress is printed live AND saved to
# logs/bench_<task>.log; checkpoints/metrics land in exps_bench/<task>/<opt>/seed<seed>/.
#
#   bash scripts/benchmark.sh <gpu> <task>              # task: qm9 | text8 | celeba64 | lm1b
#   bash scripts/benchmark.sh 2 qm9                     # one seed (42)
#   bash scripts/benchmark.sh 2 qm9 --seeds 42 69 1337  # add seeds later; done ones skipped
#   bash scripts/benchmark.sh 2 celeba64 --fid_n 1024   # extra args pass through
#
# It reads each optimizer's winning config from logs/tune_<task>_winners.json
# (written by scripts/tune.sh) — no manual step.
set -uo pipefail
cd "$(dirname "$0")/.."

gpu=${1:?usage: bash scripts/benchmark.sh <gpu> <task>}
task=${2:?usage: bash scripts/benchmark.sh <gpu> <task>}

export PATH="$HOME/.local/bin:$PATH"               # find uv
export CUDA_DEVICE_ORDER=PCI_BUS_ID                # number GPUs like nvidia-smi
export CUDA_VISIBLE_DEVICES=$gpu                   # use ONLY this GPU
mkdir -p logs

uv run python scripts/benchmark_winners.py "$task" "${@:3}" 2>&1 | tee "logs/bench_${task}.log"
