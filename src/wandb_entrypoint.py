"""W&B sweep entry point: reads wandb.config and dispatches into src.main.main."""
import sys

import wandb

from src.main import main as run_main

BOOL_FLAGS = {
    "wandb", "compile", "dry_run", "nesterov", "optimize_1d",
    "scale_parameter", "resume", "save_ckpt",
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
