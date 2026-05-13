#!/usr/bin/env bash
# Drain all five Phase-A sweeps sequentially. Designed to run in the background;
# logs to phase_a.log. Each `wandb agent` exits once its grid is exhausted.
set -uo pipefail
cd "$(dirname "$0")/.."

ENTITY=steeldream
PROJECT=diffusion-baselines-src
SWEEPS=(
    "adamw:pdr7wn47"
    "muon:w08ponn7"
    "mars_adam:2bw9wvmk"
    "prodigy:1u6m8k4d"
    "soap:xyzdfad0"
)

mkdir -p logs
for entry in "${SWEEPS[@]}"; do
    NAME="${entry%%:*}"
    SID="${entry##*:}"
    echo "[$(date -Iseconds)] starting sweep $NAME ($SID)" | tee -a logs/phase_a.log
    uv run wandb agent "${ENTITY}/${PROJECT}/${SID}" >> "logs/phase_a_${NAME}.log" 2>&1
    echo "[$(date -Iseconds)] sweep $NAME finished" | tee -a logs/phase_a.log
done
echo "[$(date -Iseconds)] Phase A complete" | tee -a logs/phase_a.log
