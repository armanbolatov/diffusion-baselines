"""Auto-name an experiment directory from key hyperparameters."""
from pathlib import Path


def make_exp_name(args) -> str:
    parts = [args.task, args.optimizer,
             f"lr{args.lr:g}",
             f"wd{args.weight_decay:g}",
             f"seed{args.seed}",
             f"iters{args.max_iters}"]
    if hasattr(args, "gamma") and args.optimizer in ("mars", "mars_adam"):
        parts.append(f"g{args.gamma:g}")
    return "_".join(parts)


def resolve_out_dir(args) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    return Path(args.exp_root) / make_exp_name(args)
