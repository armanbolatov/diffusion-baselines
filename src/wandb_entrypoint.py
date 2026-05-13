"""W&B sweep entry point.

Reads ``wandb.config`` (populated by the sweep agent) and dispatches into
``src.main.main`` as if the same hyperparameters had been passed on the
command line. One trial per process; ``wandb.init`` is called once here
and reused by the trainer's WandbLogger.

Special-cased fields:
- ``betas_pair: [b1, b2]`` is expanded into ``--beta1`` / ``--beta2`` so
  the AdamW sweep can sweep (β1, β2) as a single grid axis.

Boolean flags use store_true semantics: pass ``true``/``false`` in the
YAML and we add ``--flag`` only when truthy.
"""
import sys

import wandb

from src.main import main as run_main

# Argparse flags that use ``action="store_true"`` in src/config/base.py.
# Listed explicitly so we can map ``key: true`` from wandb.config to a
# bare ``--key`` flag (and skip it when false).
BOOL_FLAGS = {
    "wandb", "compile", "dry_run", "nesterov", "optimize_1d",
    "scale_parameter", "resume",
}


def _config_to_argv(config: dict) -> list[str]:
    argv: list[str] = []
    for key, val in config.items():
        if key == "betas_pair":
            b1, b2 = val
            argv += ["--beta1", str(b1), "--beta2", str(b2)]
            continue
        if key in BOOL_FLAGS:
            if val:
                argv += [f"--{key}"]
            continue
        if val is None:
            continue
        argv += [f"--{key}", str(val)]
    # Always forward to W&B from inside the trainer too.
    if "--wandb" not in argv:
        argv += ["--wandb"]
    return argv


def main() -> int:
    run = wandb.init()
    config = dict(run.config)
    argv = _config_to_argv(config)
    return run_main(argv)


if __name__ == "__main__":
    sys.exit(main())
