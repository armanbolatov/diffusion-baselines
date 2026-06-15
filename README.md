# diffusion-baselines

Head-to-head benchmark of seven optimizers on two diffusion training tasks:
**MDLM** (masked discrete diffusion language model) on text8, and **DDPM** on
CelebA at 32x32.

## Optimizers

| family | name | CLI | implementation |
|---|---|---|---|
| Adam-family | AdamW | `--optimizer adamw` | `torch.optim.AdamW` |
| Adam-family | Lion | `--optimizer lion` | `lion-pytorch` |
| Matrix preconditioner | Muon | `--optimizer muon` | hand-port |
| Matrix preconditioner | SOAP | `--optimizer soap` | `pytorch-optimizer` |
| Variance reduction | MARS | `--optimizer mars` | hand-port |
| Variance reduction | MARS-M | `--optimizer mars_m` | hand-port |
| Schedule-free | Schedule-Free AdamW | `--optimizer schedule_free` | `schedulefree` |

## Workflow

```bash
# bootstrap env
uv sync

# one-time data prep for text8
uv run python -m src.data.text8

# generate the 14 sweep YAMLs (7 optimizers x 2 tasks)
uv run python scripts/gen_sweep_yamls.py

# tune one (task, optimizer) sweep on Weights & Biases
uv run wandb sweep configs/sweeps/celeba_adamw.yaml
uv run wandb agent <sweep_id>

# tune all 7 CelebA sweeps in series (registers IDs, drains them, watchdog per sweep)
bash scripts/run_celeba_phase.sh

# 3-seed benchmark at the full budget
uv run python scripts/benchmark_winners.py text8
uv run python scripts/benchmark_winners.py celeba --iters 80000

# plots
uv run python scripts/plot_final.py             # mean curves + bar chart per task
uv run python scripts/plot_tuning_curves.py     # tuning-winner training curve per optimizer

# sample CelebA faces and compute FID from a benchmark checkpoint
uv run python scripts/sample_celeba.py exps_bench/celeba/<opt>/<run>/ckpt.pt --n 16
uv run python scripts/fid_celeba.py    exps_bench/celeba/<opt>/<run>/ckpt.pt --n 2048
```

## Sweep design

Each YAML is a Bayesian sweep over a discrete grid: learning-rate ladder per
optimizer (the ladder is centered per-family on values reported by the original
authors), `weight_decay ∈ {0.01, 0.1, 0.3}`, `grad_clip ∈ {0, 1.0}`, plus
optimizer-specific axes (β1, β2 for Adam-family; momentum for Muon and MARS-M;
γ for MARS / MARS-M; preconditioner update frequency for SOAP). The sweep agent
runs up to 100 trials at a short proxy budget; the lowest-validation-metric
trial is then re-trained at the full budget.

| task | proxy iters | full-budget iters | warmup | early-kill iter / threshold |
|---|---:|---:|---:|---:|
| MDLM (text) | 10000 | 30000 | 1000 | 3000 / 3.5 bpc |
| CelebA-32 (DDPM) | 500 | 80000 | 100 | 200 / 0.5 MSE |

Trials whose eval metric still exceeds the threshold at the early-kill iter are
aborted to free the sweep budget for promising configurations. Mixed precision
(bfloat16), a warmup-stable-decay learning-rate schedule with a 20% linear
cooldown, and a non-finite-loss divergence guard are shared by both tasks.

## Folder layout

```
src/
  main.py                    # CLI entry
  trainer.py                 # task-agnostic train/eval loop (bf16 autocast, EMA, WSD)
  config/base.py             # argparse spec
  tasks/{image_celeba,text_text8}.py
  data/{celeba,text8,tokenizer}.py
  diffusion/{mdlm,continuous,image_iters,discrete,_common}.py
  models/{unet2d,diff_transformer}.py
  optim/{adamw,lion,muon,soap,schedule_free,mars,mars_m}.py + schedule.py
  logger/{jsonl,wandb_logger}.py
  utils/{exp_name,checkpoint,seed}.py
  wandb_entrypoint.py        # bridges wandb sweep config -> argparse argv

scripts/
  gen_sweep_yamls.py         # writes the 14 configs/sweeps/*.yaml
  run_celeba_phase.sh        # registers + drains 7 CelebA sweeps in series
  run_celeba_80k.sh          # full 80k-iter bench chain for CelebA
  benchmark_winners.py       # 3-seed retrain of each sweep winner at the full budget
  continue_bench.py          # resume an existing bench run at a longer iter count
  plot_final.py              # mean curves + bar chart per task
  plot_tuning_curves.py      # training-loss curve per optimizer (sweep winners)
  sample_celeba.py           # DDPM-1000 sample 16 faces from a ckpt
  fid_celeba.py              # 2048-sample FID-2048 from a ckpt
  smoke_text.sh              # short MDLM sanity check

configs/sweeps/              # 14 wandb sweep YAMLs (regen via gen_sweep_yamls)
tests/                       # pytest suite

exps_tuning/<task>/<opt>/    # sweep trial outputs (one dir per trial)
exps_bench/<task>/<opt>/     # 3-seed benchmark outputs (log.jsonl + ckpt.pt + ckpt_stable.pt)
exps/{samples,final_plots,tuning_plots}/  # PNGs (gitignored)
```

## Logging

Every run writes `<out_dir>/log.jsonl` with one JSON event per line:

- `config` — args namespace, once at start
- `params` — optimizer name + parameter counts
- `iter` — every `--log_interval`: loss, lr, dt_iter_s, peak_mem_mb
- `eval` / `final_eval` — every `--eval_interval`: eval metric (`eval_bpc` for MDLM, `eval_loss` for CelebA)
- `early_killed` — when killed by `--kill_if_above`
- `diverged` — when the loss is non-finite for an extended interval

With `--wandb` the same events are forwarded to Weights & Biases.

## Checkpointing

The trainer writes `ckpt.pt` (the latest state) and `ckpt_stable.pt` (a single
snapshot captured at the boundary of the WSD stable-and-decay phases). The
stable checkpoint lets a future run extend training at a new total iteration
count without a learning-rate discontinuity at the resume point. For
Schedule-Free, the trainer toggles the optimizer into its eval (running
average) mode before saving so the saved weights can be used for inference
without reconstructing the optimizer state.

For sampling and FID, the scripts default to the **raw** (final-iter) weights
rather than the EMA, because the EMA at decay 0.9999 keeps a non-trivial
trajectory-average bias at iteration counts well below the paper-scale 800k.
Pass `--ema` to the sampler to override.

## Tests

```bash
uv run pytest tests/ -q
```

Coverage: every optimizer instantiates and takes a step on a tiny model; the
discrete diffusion scheme runs one forward + backward; `--dry_run` resolves a
valid config and exits cleanly.

## Reproducibility

- Each trial's output directory is named
  `<task>_<opt>_lr<...>_wd<...>_seed<...>_iters<...>_<8-char-hash>`,
  where the hash covers every tuned axis to guarantee unique directories
  across sibling sweep trials.
- Sweep IDs are minted by `wandb sweep` and persisted locally so that a
  re-run of the phase script resumes the same sweeps rather than minting
  fresh ones.
- Checkpoints (`*.pt`) are not committed to the repository (see
  `.gitignore`). The `log.jsonl` files under `exps_bench/` and
  `exps_tuning/` retain every iteration's loss and metric and are
  sufficient to reproduce all plots in the writeup.
