#!/usr/bin/env bash
# Drain all five image-DDPM sweeps sequentially.
set -uo pipefail
cd "$(dirname "$0")/.."

ENTITY=steeldream
PROJECT=diffusion-baselines-src
SWEEPS=(
    "prodigy:4hjyk7hh"
    "muon:pnrip3g7"
    "adamw:f520y6rg"
    "mars_adam:zjdahinh"
    "soap:jt5rohiz"
)

mkdir -p logs
for entry in "${SWEEPS[@]}"; do
    NAME="${entry%%:*}"
    SID="${entry##*:}"
    echo "[$(date -Iseconds)] starting sweep $NAME ($SID)" | tee -a logs/image2.log
    uv run wandb agent "${ENTITY}/${PROJECT}/${SID}" >> "logs/image2_${NAME}.log" 2>&1
    echo "[$(date -Iseconds)] sweep $NAME finished" | tee -a logs/image2.log
done
echo "[$(date -Iseconds)] image sweeps complete" | tee -a logs/image2.log
