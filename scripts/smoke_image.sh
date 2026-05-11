#!/usr/bin/env bash
# 200-iter CIFAR-10 DDPM smoke test, plain AdamW.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.main \
    --task image_cifar10 \
    --optimizer adamw \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --batch_size 32 \
    --max_iters 200 \
    --warmup_iters 50 \
    --eval_interval 100 \
    --log_interval 25 \
    --seed 0 \
    --out_dir exps/smoke_image
echo "smoke_image done. log: exps/smoke_image/log.jsonl"
