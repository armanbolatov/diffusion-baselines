# diffusion-baselines

Optimizer benchmark across diffusion tasks, modeled after
[epfml/llm-baselines](https://github.com/epfml/llm-baselines).

Compares **11 optimizers** on **multiple diffusion formulations** across
two task families:

- **Continuous image diffusion** on CIFAR-10 — DDPM, Flow Matching, Consistency Training
- **Discrete text diffusion** on text8 — MDLM, UDLM, SEDD, ARM (AR baseline)

## Quick start

```bash
# bootstrap env (uses uv)
uv sync

# data prep (one time)
uv run python -m src.data.text8        # downloads + tokenizes text8

# smoke tests
bash scripts/smoke_image.sh
bash scripts/smoke_text.sh

# single run
uv run python -m src.main \
    --task image_cifar10 \
    --diffusion ddpm \
    --model unet2d \
    --optimizer mars \
    --lr 3e-4 \
    --max_iters 3000 \
    --seed 0

# full sweep (11 optimizers × 2 tasks × 3 seeds = 66 runs)
bash scripts/sweep_v1.sh
```

## Supported optimizers

| family | name | CLI |
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

## Supported diffusion schemes

| family | name | CLI |
|---|---|---|
| Continuous (image) | DDPM (ε-prediction) | `--diffusion ddpm` |
| Continuous (image) | Flow Matching (cond-OT) | `--diffusion flow_matching` |
| Continuous (image) | Consistency Training | `--diffusion consistency` |
| Discrete (text) | MDLM (absorbing) | `--diffusion mdlm` |
| Discrete (text) | UDLM (uniform corruption) | `--diffusion udlm` |
| Discrete (text) | SEDD (score-entropy) | `--diffusion sedd` |
| Discrete (text) | ARM (AR baseline) | `--diffusion arm` |

## Models

| name | CLI | sizes |
|---|---|---|
| diffusers `UNet2DModel` | `--model unet2d` | ~17M params (CIFAR-32) |
| DiT-S/2 | `--model dit` | ~30M params at default config |
| Custom decoder-only transformer (AdaLN time-cond) | implicit for `text_text8` | ~30M params (12L × d=384) |

## Layout

```
src/
├── main.py              # entry point
├── trainer.py           # task-agnostic train/eval loop (logs dt_iter_s, peak_mem_mb)
├── config/base.py       # argparse spec
├── tasks/               # one file per task (image_cifar10, text_text8)
├── models/              # one file per architecture (unet2d, dit, diff_transformer)
├── data/                # one file per dataset (cifar10, text8) + tokenizer
├── diffusion/           # forward processes + losses + samplers, one file per scheme
│   ├── continuous.py    # DDPM scheduler wrapper
│   ├── flow_matching.py
│   ├── consistency.py
│   ├── discrete.py      # dispatcher → mdlm | udlm | arm | sedd
│   ├── mdlm.py / udlm.py / arm.py / sedd.py
│   └── image_iters.py   # per-scheme batch iters for image tasks
├── evaluation/fid.py    # CIFAR-10 FID via torchmetrics
├── optim/               # one file per optimizer + dispatcher + param-split + schedule
├── logger/              # JSONL + optional wandb
└── utils/               # seed, exp_name, checkpoint, ema
scripts/                 # smoke + sweep bash + analysis
tests/                   # pytest suite
exps/                    # auto-generated run output (gitignored)
```

## Logging

Each run writes `exps/{auto_name}/log.jsonl`, one JSON per line, with an
`event` field. Schemas:

- `{"event":"config", "args":{...}}` — once at start
- `{"event":"params", "optimizer":..., "n_params":..., "n_muon_params":...}` — once at start
- `{"event":"iter", "iter":i, "loss":..., "lr":..., "dt_iter_s":..., "peak_mem_mb":..., "opt_metrics":{...}}` — every `--log_interval`
- `{"event":"eval", "iter":i, "elapsed_s":..., "eval_loss":..., "diffusion":..., "fid":...?}` — every `--eval_interval`
- `{"event":"final_eval", "iter":i, "eval_loss":..., ...}` — at end
- `{"event":"diverged", "iter":i}` — only if non-finite for ≥50 consecutive iters

If `--wandb` is passed, the same events are forwarded to W&B.

## FID (optional)

Pass `--fid_samples 2048 --fid_steps 50` to compute FID against a 2048-image
CIFAR-10 train subset each eval. Requires `torch-fidelity` (`uv add torch-fidelity`).

## Sweeping

`scripts/sweep_v1.sh` runs 2 tasks × 11 optimizers × 3 seeds = 66 runs sequentially.
`scripts/lr_tune.sh` runs a 5-LR scan per (task, optimizer) for picking sweep-time LRs.

Both honor `DRY_RUN=1` to print commands without launching.

## Tests

```bash
uv run pytest tests/ -q
```

Coverage:
- All 11 optimizers instantiate + take one step on a toy net.
- All 7 diffusion schemes execute a single loss step + backward.
- DiT forward returns shape-matched output to UNet2D.
- EMA helpers behave correctly.
- `--dry_run` prints resolved config and exits 0.

## Reference numbers (from sibling MARS_M experiments)

For parity checks during development:

1. `image_cifar10 --diffusion ddpm --optimizer adamw --lr 1e-4 --max_iters 3000`
   → eval MSE ≈ **0.039**
2. `image_cifar10 --diffusion ddpm --optimizer muon --lr 3e-4 --max_iters 3000`
   → eval MSE ≈ **0.032** (18% relative win over AdamW)
3. `image_cifar10 --diffusion ddpm --optimizer mars_adam --lr 1e-4 --max_iters 3000`
   → eval MSE ≈ AdamW within seed noise
