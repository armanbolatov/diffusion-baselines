#!/usr/bin/env bash
# 500-iter text8 MDLM smoke test, plain AdamW. Downloads + tokenizes text8 if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.data.text8   # downloads + tokenizes if not already present
uv run python -m src.main \
    --task text_text8 \
    --optimizer adamw \
    --lr 5e-4 \
    --weight_decay 0.0 \
    --batch_size 16 \
    --block_size 128 \
    --n_layer 4 --n_head 4 --d_model 128 \
    --max_iters 500 \
    --warmup_iters 100 \
    --eval_interval 250 \
    --eval_batches 5 \
    --log_interval 50 \
    --seed 0 \
    --out_dir exps/smoke_text
echo "smoke_text done. log: exps/smoke_text/log.jsonl"
