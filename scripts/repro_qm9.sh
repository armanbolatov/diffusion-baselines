#!/usr/bin/env bash
# Paper-faithful QM9 UDLM reproduction — a SINGLE run (not a sweep).
#
# Sizes our (DiT-style) DiffTransformer to the paper's DiT-small
# (hidden 768 / 12 layers / 12 heads / dropout 0.1) and trains with their QM9
# hyperparameters (UDLM uniform diffusion, AdamW, lr 3e-4, 25k steps, length 32,
# yairschiff/qm9-tokenizer vocab 40). The reported eval_ppl is the pad-masked,
# per-real-token perplexity — directly comparable to the paper's QM9 UDLM 2.02.
#
#   bash scripts/repro_qm9.sh <gpu>
#
# This is separate from the optimizer benchmark (scripts/tune.sh), which stays
# small/tractable. Use it to validate our UDLM implementation against the paper.
#
# Note: batch 2048 at length 32 may OOM on a single GPU. If so, lower
# --batch_size (the trainer has no gradient accumulation); lr was tuned for
# 2048, so drop lr roughly proportionally if you change the batch a lot.
set -uo pipefail
cd "$(dirname "$0")/.."

gpu=${1:?usage: bash scripts/repro_qm9.sh <gpu>}

export PATH="$HOME/.local/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p logs

uv run python -m src.data.qm9 --block_size 32        # one-time, cached

uv run python -m src.main \
    --task mol_qm9 --diffusion udlm --optimizer adamw \
    --lr 3e-4 --weight_decay 0.0 --beta1 0.9 --beta2 0.999 --eps 1e-8 \
    --grad_clip 1.0 --schedule wsd \
    --batch_size 2048 --block_size 32 \
    --n_layer 12 --n_head 12 --d_model 768 --time_emb_dim 128 --dropout 0.1 \
    --max_iters 25000 --warmup_iters 2500 \
    --eval_interval 1000 --eval_batches 5 \
    --log_interval 100 --seed 0 \
    --save_ckpt --out_dir exps/repro_qm9 \
    2>&1 | tee logs/repro_qm9.log

echo "done. compare the final '[eval @ ...] eval_ppl' in logs/repro_qm9.log to the paper's 2.02"
