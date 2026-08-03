#!/usr/bin/env bash
# Local Optuna tuning: tune all 7 optimizers for one task on one GPU.
# Resumable: the SQLite study (optuna_db/optuna_<task>.db) means a restart
# continues where it stopped and never redoes finished trials.
#
#   bash scripts/tune.sh <gpu> <task>      # task: qm9 | text8 | celeba64 | lm1b
#
# Winners -> logs/tune_<task>_winners.json; then: bash scripts/benchmark.sh <gpu> <task>
set -uo pipefail
cd "$(dirname "$0")/.."

gpu=${1:?usage: bash scripts/tune.sh <gpu> <task> [optimizers...]}
task=${2:?usage: bash scripts/tune.sh <gpu> <task> [optimizers...]}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs

# optional: a custom subset/order of optimizers; default runs all 7,
# schedule_free & lion first (they need the most eval attention).
opts=("${@:3}"); [ ${#opts[@]} -eq 0 ] && opts=(schedule_free lion adamw muon soap mars mars_m)

# one-time data prep (celeba downloads itself when the task first builds data)
[ "$task" = qm9 ]   && uv run python -m src.data.qm9 --block_size 32
[ "$task" = text8 ] && uv run python -m src.data.text8
[ "$task" = lm1b ]  && uv run python -m src.data.lm1b --block_size 128

for opt in "${opts[@]}"; do
    echo "===== tuning $task / $opt  (GPU $gpu) ====="
    uv run python scripts/tune_local.py "$task" "$opt" --gpu "$gpu" \
        2>&1 | tee -a "logs/tune_${task}_${opt}.log"
done
echo "done. winners -> logs/tune_${task}_winners.json"
