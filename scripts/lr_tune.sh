#!/usr/bin/env bash
# LR-tuning sub-sweep. For each (task, optimizer), runs one seed across
# lr ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. Writes the best lr (lowest final
# eval metric) to configs/sweeps/{task}_{opt}.flags so the main sweep
# can lock it in.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=(image_cifar10 text_text8)
OPTS=(adamw sgd lion adafactor adema soap prodigy schedule_free muon mars mars_adam)
LRS=(3e-5 1e-4 3e-4 1e-3 3e-3)

ITERS=1500

for task in "${TASKS[@]}"; do
    for opt in "${OPTS[@]}"; do
        for lr in "${LRS[@]}"; do
            tag="lrtune_${task}_${opt}_lr${lr}"
            if [[ "${DRY_RUN:-0}" == "1" ]]; then
                echo "uv run python -m src.main --task $task --optimizer $opt --lr $lr --max_iters $ITERS --seed 0 --out_dir exps/lrtune/$tag"
            else
                echo "=== $tag ==="
                uv run python -m src.main --task "$task" --optimizer "$opt" --lr "$lr" \
                    --max_iters "$ITERS" --seed 0 --out_dir "exps/lrtune/$tag"
            fi
        done
    done
done
echo "lr_tune done. Use scripts/select_best_lr.py to write per-opt .flags files."
