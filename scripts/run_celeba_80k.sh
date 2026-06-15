#!/usr/bin/env bash
# Fresh 80K-iter bench for all 7 opts. 6 use benchmark_winners (tune-best HPs).
# mars uses safe HPs (lr=1e-3, not the diverging 3e-3) hardcoded below.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

uv run python scripts/benchmark_winners.py celeba --iters 80000 \
    --only adamw lion muon schedule_free soap mars_m \
    2>&1 | tee logs/bench_celeba_80k.log

for seed in 42 69 1337; do
    echo "[$(date -Iseconds)] mars (safe) seed $seed"
    uv run python -m src.main \
        --task image_celeba --diffusion ddpm --optimizer mars \
        --lr 1e-3 --weight_decay 0.1 --beta1 0.9 --beta2 0.99 --eps 1e-8 \
        --grad_clip 1.0 --gamma 0.025 --schedule wsd --batch_size 256 \
        --num_diffusion_steps 1000 --beta_schedule linear \
        --seed "$seed" --max_iters 80000 --warmup_iters 8000 \
        --eval_interval 1600 --eval_batches 5 \
        --exp_root exps_bench/celeba/mars \
        --save_ckpt --ema_decay 0.9999 \
        --wandb --wandb_project diffusion-baselines-bench-celeba \
        > "logs/bench_celeba_mars_safe_s${seed}.log" 2>&1
done

echo "[$(date -Iseconds)] 80K bench chain complete"
