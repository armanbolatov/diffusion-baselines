#!/usr/bin/env bash
# Phase D: final benchmark sweep using best hyperparameters from configs/best/.
#
# Reads configs/best/text8_<size>_<opt>.json for each (size, optimizer)
# cell, then runs 3 seeds at the appropriate iter count per size.
# Total: 3 sizes * 5 opts * 3 seeds = 45 runs.
#
# Usage:
#   bash scripts/benchmark_main.sh
#   DRY_RUN=1 bash scripts/benchmark_main.sh   # print commands, do not execute
#
# Requires `jq` for JSON parsing (apt: jq, brew: jq).

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v jq >/dev/null 2>&1; then
    echo "this script requires jq (sudo apt install jq)" >&2
    exit 1
fi

OPTIMIZERS=(adamw muon soap mars_adam prodigy)
SEEDS=(0 1 2)

# size → (n_layer n_head d_model max_iters)
declare -A SIZE_SPEC=(
    [small]="4 4 128 5000"
    [medium]="8 6 256 10000"
    [large]="12 6 384 20000"
)

run_or_echo() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "$@"
    else
        echo "==> $*"
        "$@"
    fi
}

for SIZE in small medium large; do
    read -r N_LAYER N_HEAD D_MODEL MAX_ITERS <<< "${SIZE_SPEC[$SIZE]}"
    for OPT in "${OPTIMIZERS[@]}"; do
        CFG="configs/best/text8_${SIZE}_${OPT}.json"
        if [ ! -f "$CFG" ]; then
            echo "skipping ${SIZE}/${OPT}: no best config at $CFG" >&2
            continue
        fi
        # Pull tuned hyperparameters as a series of --flag value pairs.
        # Drop:
        #  - metadata keys (start with _)
        #  - the eval_bpc summary metric
        #  - shape/control keys we explicitly pass below (task/diffusion/optimizer/
        #    n_layer/n_head/d_model/max_iters/seed/block_size/batch_size/schedule)
        # Special-case betas_pair → --beta1/--beta2.
        EXTRA_FLAGS=$(jq -r '
            to_entries
            | map(select(
                .key | test(
                  "^_|^eval_|^task$|^diffusion$|^optimizer$|^n_layer$|^n_head$|^d_model$|^max_iters$|^seed$|^block_size$|^batch_size$|^schedule$"
                ) | not
              ))
            | map(
                if .key == "betas_pair" then
                    "--beta1 \(.value[0]) --beta2 \(.value[1])"
                elif .value == null then
                    empty
                else
                    "--\(.key) \(.value)"
                end
              )
            | join(" ")
        ' "$CFG")

        for SEED in "${SEEDS[@]}"; do
            # shellcheck disable=SC2086
            run_or_echo uv run python -m src.main \
                --task text_text8 --diffusion mdlm \
                --optimizer "$OPT" \
                --n_layer "$N_LAYER" --n_head "$N_HEAD" --d_model "$D_MODEL" \
                --block_size 256 --batch_size 64 \
                --max_iters "$MAX_ITERS" --schedule wsd \
                --seed "$SEED" \
                --exp_root "exps/v3_benchmark" \
                --wandb --wandb_project diffusion-baselines-v3-bench \
                $EXTRA_FLAGS
        done
    done
done
