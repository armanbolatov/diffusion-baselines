#!/usr/bin/env bash
# Launch a W&B grid sweep and start one agent in this shell.
#
# Usage:
#   bash scripts/launch_sweep.sh configs/sweeps/text8_small_adamw.yaml
#   COUNT=10 bash scripts/launch_sweep.sh configs/sweeps/text8_small_adamw.yaml
#
# Notes:
# - `wandb sweep` prints "wandb agent <user>/<project>/<id>" on stderr;
#   we parse that to get the agent invocation.
# - With method:grid, omitting --count makes the agent run every grid
#   point exactly once and then exit.
# - To run multiple agents in parallel (e.g. across GPUs), run this
#   script multiple times after the sweep is created, passing the
#   resulting sweep id to `wandb agent` directly.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <sweep.yaml> [extra args to wandb agent]" >&2
    exit 1
fi

YAML="$1"
shift

if [ ! -f "$YAML" ]; then
    echo "sweep yaml not found: $YAML" >&2
    exit 1
fi

# Capture stderr (where wandb prints the agent command) while letting
# the user see it too.
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
uv run wandb sweep "$YAML" 2>&1 | tee "$TMP"

AGENT_CMD=$(grep -oE 'wandb agent [^ ]+' "$TMP" | tail -1)
if [ -z "$AGENT_CMD" ]; then
    echo "could not parse sweep id from wandb output" >&2
    exit 1
fi
SWEEP_ID=${AGENT_CMD##wandb agent }

echo
echo "==> sweep created: $SWEEP_ID"
echo "==> starting agent (Ctrl-C to stop; sweep can be resumed with: wandb agent $SWEEP_ID)"
echo

if [ -n "${COUNT:-}" ]; then
    uv run wandb agent --count "$COUNT" "$SWEEP_ID" "$@"
else
    uv run wandb agent "$SWEEP_ID" "$@"
fi
