#!/usr/bin/env bash
# 7 CelebA-64 Bayes sweeps -> 3-seed bench -> plot. 3h watchdog per sweep.
# WANDB_AGENT_DISABLE_FLAPPING=true: agent survives transient OOMs from other GPU users.
# IDs persisted in logs/celeba_ids.env so re-runs resume the same sweeps.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=10000   # ignore OOM bursts from other GPU users

ENTITY=steeldream
PROJECT=diffusion-baselines-src
# When called with no args, drain all 7 sweeps and auto-bench + plot at the end.
# When called with args (subset), drain just those and skip the auto-bench
# (run benchmark_winners.py --only ... manually after).
if [ $# -eq 0 ]; then
    OPTS=(adamw lion muon schedule_free soap mars mars_m)
    AUTOBENCH=1
else
    OPTS=("$@")
    AUTOBENCH=0
fi

touch logs/celeba_ids.env
source logs/celeba_ids.env
# Register any sweep IDs missing from celeba_ids.env (per-opt backfill).
for opt in "${OPTS[@]}"; do
    if [[ -z "${!opt:-}" ]]; then
        sid=$(uv run wandb sweep "configs/sweeps/celeba_${opt}.yaml" 2>&1 \
              | grep -oE "with ID: [a-z0-9]+" | awk '{print $3}')
        echo "${opt}=${sid}" >> logs/celeba_ids.env
        export "${opt}=${sid}"
        echo "registered celeba_${opt} -> $sid"
    fi
done

for opt in "${OPTS[@]}"; do
    sid="${!opt}"
    echo "[$(date -Iseconds)] starting celeba_${opt} ($sid)" | tee -a logs/tune.log
    timeout 10800 uv run wandb agent "${ENTITY}/${PROJECT}/${sid}" \
        >> "logs/tune_celeba_${opt}.log" 2>&1 || true
done
echo "[$(date -Iseconds)] Phase D complete" | tee -a logs/tune.log

if [ "$AUTOBENCH" = "1" ]; then
    uv run python scripts/benchmark_winners.py celeba 2>&1 | tee logs/bench_celeba.log
    uv run python scripts/plot_final.py celeba 2>&1 | tee logs/plot_celeba.log
fi
