# diffusion-baselines

Head-to-head benchmark of seven optimizers across four diffusion training tasks —
one per diffusion paradigm:

- **text8** — MDLM (masked/absorbing discrete diffusion), character-level
- **QM9** — UDLM (uniform-state discrete diffusion; Schiff et al. 2024), SMILES
- **CelebA-64** — DDPM (Gaussian diffusion; Ho et al. 2020) on 64x64 face images
- **LM1B** — DUO (uniform-state diffusion trained through its Gaussian dual;
  Sahoo et al. 2025), bert-base-uncased tokenizer, context 128

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
uv sync                                     # bootstrap (torch pinned to cu121)

# one-time data prep (celeba downloads itself on first run)
uv run python -m src.data.text8
uv run python -m src.data.qm9  --block_size 32
uv run python -m src.data.lm1b --block_size 128

# tune all 7 optimizers for one task on one GPU (Optuna, resumable)
bash scripts/tune.sh <gpu> <task>           # task: qm9 | text8 | celeba64 | lm1b
# -> logs/tune_<task>_winners.json

# retrain the winners at the full budget (one seed by default; add more anytime)
bash scripts/benchmark.sh <gpu> <task>
bash scripts/benchmark.sh <gpu> <task> --seeds 42 69 1337   # finished seeds are skipped
bash scripts/benchmark.sh <gpu> celeba64 --fid_n 1024       # log in-training FID (celeba64)

# plots (read local log.jsonl; no network)
uv run python scripts/compute_flops.py      # -> logs/flops.json (per-optimizer FLOPs/iter)
uv run python scripts/plot_final.py qm9     # per-task: metric vs iter, vs FLOPs, final bar
uv run python scripts/plot_grid.py          # all tasks in one figure + summary table
uv run python scripts/plot_scatter.py       # FLOPs vs memory, colored by final metric

# generative scores from a benchmark checkpoint
uv run python scripts/fid_celeba.py exps_bench/celeba64/<opt>/seed42/ckpt.pt --n 50000 --size 64
uv run python scripts/sample_qm9.py exps_bench/qm9/<opt>/seed42/ckpt.pt --n 1000
```

## Running in parallel

The tune and benchmark jobs are fully independent — one per (task, optimizer) for
tuning, one per (task, optimizer, seed) for benchmarking — so they spread across as
many GPUs as you have. `scripts/run_fleet.py` dispatches them: one thread per GPU
pulls the next job from a queue and pins it with `CUDA_VISIBLE_DEVICES`, running the
tune phase first, then the benchmark phase. It is resumable (the tuner resumes each
study; the benchmark skips finished seeds), so re-running the same command continues
where it stopped. Per-job logs land in `logs/fleet/`.

```bash
# whole pipeline across every GPU on the machine
uv run python scripts/run_fleet.py --gpus 0 1 2 3 4 5 6 7

# just tuning, or just one task
uv run python scripts/run_fleet.py --gpus 0 1 2 3 --phase tune
uv run python scripts/run_fleet.py --gpus 0 1 --tasks text8
```

More GPUs cut wall-clock, not total GPU-hours. Keep the same GPU type within a task
so its wall-clock numbers stay comparable.

## Tuning

One [Optuna](https://optuna.org) study per (task, optimizer). The search space is the
grid in `configs/<task>/{common,<opt>}.yaml` — shared learning rate, weight decay, and
grad clip, plus optimizer-specific axes (`value:` = fixed, `values:` = swept). Trials
run at a ~20% proxy budget with early pruning; each winner is then retrained at the
full budget (`BENCH_ITERS` in `benchmark_winners.py`). Studies live in
`optuna_db/optuna_<task>_<opt>.db` and resume on restart. Tuning selects on a cheap
proxy metric (`eval_bpc` for text/molecules, denoising `eval_loss` for images); for
CelebA the reported quality metric is FID, computed separately (see below).

Winners are re-derived from the raw trial logs before benchmarking, for every task:

```bash
uv run python scripts/rescore_winners.py text8 celeba64 qm9 lm1b --write
```

Each trial is scored by the minimum eval logged after the last `config` event in its
own `log.jsonl` — i.e. only the run belonging to that trial. This makes the winners
reproducible from the logs alone rather than trusted from the study database.

## Layout

```
src/
  main.py                    # CLI entry: dispatch by --task and --optimizer
  trainer.py                 # task-agnostic train/eval loop (bf16, EMA, WSD, divergence guard)
  config/base.py             # argparse spec
  tasks/{image_celeba,text_text8,text_lm1b,mol_qm9}.py
  data/{celeba,text8,lm1b,qm9,tokenizer,mol_tokenizer}.py
  diffusion/{mdlm,udlm,duo,discrete,continuous,image_iters}.py
  models/{unet2d,diff_transformer}.py
  optim/{adamw,lion,muon,soap,schedule_free,mars,mars_m}.py + schedule.py, param_groups.py
  logger/jsonl.py
  utils/{fid,exp_name,checkpoint,seed}.py

scripts/
  run_fleet.py                      # dispatch all tune+bench jobs across GPUs (parallel)
  tune.sh / tune_local.py           # Optuna tuning over all 7 optimizers
  benchmark.sh / benchmark_winners.py  # retrain winners at full budget (seed-outer)
  compute_flops.py                  # per-optimizer FLOPs/iter -> logs/flops.json
  plot_final.py / plot_grid.py / plot_scatter.py   # result figures
  fid_celeba.py / sample_celeba.py / sample_qm9.py # generative metrics
  repro_qm9.sh                      # paper-faithful single QM9/UDLM run

configs/<task>/              # common.yaml + one <optimizer>.yaml each (value: fixed, values: swept)
logs/tune_<task>_winners.json  # best config per optimizer (tuner -> benchmark)
exps_bench/<task>/<opt>/seed<seed>/   # benchmark outputs (log.jsonl + ckpt.pt)
exps/final_plots/            # figures (gitignored)
```

## Logging

Every run writes `<out_dir>/log.jsonl`, one JSON event per line:

- `config` — the args namespace, once at start
- `params` — optimizer name + parameter counts
- `iter` — every `--log_interval`: loss, lr, dt_iter_s, peak_mem_mb
- `eval` / `final_eval` — every `--eval_interval`: the eval metric (`eval_bpc`,
  `eval_loss`, and `eval_fid` for CelebA when `--fid_n > 0`)
- `diverged` — when the loss goes non-finite or explodes far above its best

`log.jsonl` is the single source of truth: tuning, benchmarking, and every plot
read it. There is no external tracker.

## Checkpoints and EMA

The trainer writes `ckpt.pt` (latest state, including the EMA weights) and
`ckpt_stable.pt` (a snapshot at the WSD stable/decay boundary, so training can be
extended without a learning-rate discontinuity). With `--ema_eval 1` the trainer
evaluates and samples from the EMA weights (decay `--ema_decay`), which is what
gives stable, low DDPM FID; `scripts/fid_celeba.py` also samples from the EMA
weights when present. For Schedule-Free, the trainer switches the optimizer into
its averaged-iterate mode around evaluation and checkpointing.

## Tests

```bash
uv run pytest tests/ -q
```

Every optimizer instantiates and takes a step on a tiny model; each discrete
diffusion scheme runs a forward/backward and a sampling/eval smoke; `--dry_run`
resolves a valid config and exits cleanly.
