#!/usr/bin/env bash
# Full v1 sweep: 2 tasks × 11 optimizers × 3 seeds = 66 runs.
# Sequential single-GPU; ~10–20 min/run → 12–22 GPU-hours total.
#
# Per-(task, optimizer) flags live in configs/sweeps/{task}_{opt}.flags
# (one-line `--lr ... --weight_decay ...` overrides). The skeleton ships
# defaults sufficient for the smoke tests; tune via scripts/lr_tune.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=(image_cifar10 text_text8)
OPTS=(adamw sgd lion adafactor adema soap prodigy schedule_free muon mars mars_adam)
SEEDS=(0 1 2)

# Per-task base iters/batch
declare -A BASE_FLAGS
BASE_FLAGS["image_cifar10"]="--max_iters 3000 --eval_interval 500 --batch_size 64 --warmup_iters 200"
BASE_FLAGS["text_text8"]="--max_iters 3000 --eval_interval 500 --batch_size 32 --block_size 256 --warmup_iters 200"

run_one() {
    local task=$1 opt=$2 seed=$3
    local flags_file="configs/sweeps/${task}_${opt}.flags"
    local extra=""
    if [[ -f $flags_file ]]; then
        extra=$(cat "$flags_file")
    fi
    local cmd="uv run python -m src.main --task $task --optimizer $opt --seed $seed ${BASE_FLAGS[$task]} $extra"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "$cmd"
    else
        echo "=== $task / $opt / seed=$seed ==="
        eval "$cmd"
    fi
}

for task in "${TASKS[@]}"; do
    for opt in "${OPTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$task" "$opt" "$seed"
        done
    done
done
echo "sweep_v1 done."
