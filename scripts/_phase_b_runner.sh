#!/usr/bin/env bash
# Drain all five Phase-B (medium) sweeps sequentially in the background.
# Logs to logs/phase_b.log + logs/phase_b_<opt>.log.
set -uo pipefail
cd "$(dirname "$0")/.."

ENTITY=steeldream
PROJECT=diffusion-baselines-src
SWEEPS=(
    "prodigy:6f5b7984"
    "muon:gsa2u4o4"
    "adamw:07h1kx84"
    "mars_adam:hvx1prd1"
    "soap:nmhlioeh"
)

mkdir -p logs
for entry in "${SWEEPS[@]}"; do
    NAME="${entry%%:*}"
    SID="${entry##*:}"
    echo "[$(date -Iseconds)] starting sweep $NAME ($SID)" | tee -a logs/phase_b.log
    uv run wandb agent "${ENTITY}/${PROJECT}/${SID}" >> "logs/phase_b_${NAME}.log" 2>&1
    echo "[$(date -Iseconds)] sweep $NAME finished" | tee -a logs/phase_b.log
done
echo "[$(date -Iseconds)] Phase B complete" | tee -a logs/phase_b.log
