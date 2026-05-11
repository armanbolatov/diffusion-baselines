# diffusion-baselines

Optimizer benchmark across diffusion tasks. Compares 11 optimizers on
two task families: continuous image diffusion (CIFAR-10 / DDPM) and
discrete text diffusion (text8 / MDLM).

Modeled after [epfml/llm-baselines](https://github.com/epfml/llm-baselines).

## Quick start

```bash
# bootstrap env (uses uv)
uv sync

# data prep (one time)
uv run python -m src.data.text8         # downloads + tokenizes text8

# smoke tests
bash scripts/smoke_image.sh
bash scripts/smoke_text.sh

# single run
uv run python -m src.main \
    --task image_cifar10 \
    --optimizer mars \
    --lr 3e-4 \
    --max_iters 3000 \
    --seed 0

# full sweep (11 optimizers × 2 tasks × 3 seeds = 66 runs)
bash scripts/sweep_v1.sh
```

## Supported optimizers (v1)

| family | name | CLI flag |
|---|---|---|
| Adam family | AdamW | `--optimizer adamw` |
| Adam family | Lion | `--optimizer lion` |
| Adam family | AdEMAMix | `--optimizer adema` |
| Adam family | Adafactor | `--optimizer adafactor` |
| Matrix preconditioner | Muon | `--optimizer muon` |
| Matrix preconditioner | SOAP | `--optimizer soap` |
| Variance reduction | MARS | `--optimizer mars` |
| Variance reduction | MARS-Adam | `--optimizer mars_adam` |
| Schedule-free | Schedule-Free AdamW | `--optimizer schedule_free` |
| Schedule-free | Prodigy | `--optimizer prodigy` |
| Classic | SGD + momentum | `--optimizer sgd` |

## Tasks (v1)

- `image_cifar10` — continuous DDPM (ε-prediction MSE) with diffusers `UNet2DModel`, 17M params, 32×32 RGB.
- `text_text8` — discrete diffusion (MDLM absorbing-state, weighted-CE) with a custom transformer (12L × d=384, AdaLN time-cond), char-level vocab.

## Layout

```
src/
├── main.py             # entry point
├── trainer.py          # task-agnostic train/eval loop
├── config/base.py      # argparse spec
├── tasks/              # one file per task
├── models/             # one file per architecture
├── data/               # one file per dataset
├── diffusion/          # forward processes + samplers
├── optim/              # one file per optimizer + dispatcher
├── logger/             # JSONL + optional wandb
└── utils/              # seed, exp_name, checkpoint
scripts/                # smoke + sweep bash scripts
tests/                  # pytest suite
exps/                   # auto-generated run output (gitignored)
```

## Reproducing earlier MARS_M results

Three parity checks (see `tests/test_parity.py` or `scripts/parity.sh`):

1. CIFAR-10 + AdamW + 3000 iters should reach eval MSE ≈ 0.039.
2. CIFAR-10 + Muon + 3000 iters should reach eval MSE ≈ 0.032 (18% relative win over AdamW).
3. CIFAR-10 + MARS-Adam + 3000 iters should match plain AdamW within seed noise.
