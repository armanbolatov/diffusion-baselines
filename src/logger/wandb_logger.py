"""Optional W&B forwarder. Gated on args.wandb."""
from typing import Any


class WandbLogger:
    def __init__(self, args, exp_name: str):
        import wandb
        self._wandb = wandb
        wandb.init(project=args.wandb_project, name=exp_name, config=vars(args))

    def log(self, event: str, **fields: Any) -> None:
        if event in ("config", "params"):
            return
        step = fields.pop("iter", None)
        # flatten any nested dict (opt_metrics) so wandb shows individual keys
        flat = {}
        for k, v in fields.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}/{kk}"] = vv
            else:
                flat[k] = v
        self._wandb.log(flat, step=step)

    def close(self) -> None:
        self._wandb.finish()
