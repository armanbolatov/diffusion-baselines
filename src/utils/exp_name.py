"""Auto-name an experiment directory from key hyperparameters.

The (task, optimizer, lr, wd, seed, iters) prefix isn't unique across
sweep trials that differ in betas/momentum/gamma/etc. — so we append an
8-char hash over all tuned dims to guarantee unique dirs.
"""
import hashlib
from pathlib import Path


_PER_OPT_EXTRAS = {
    "adamw":         ("beta1", "beta2", "eps"),
    "lion":          ("beta1", "beta2"),
    "muon":          ("momentum", "nesterov", "ns_steps"),
    "schedule_free": ("beta1", "beta2"),
    "soap":          ("beta1", "beta2", "eps",
                      "precondition_frequency", "max_precondition_dim",
                      "shampoo_beta"),
    "mars":          ("beta1", "beta2", "eps", "gamma"),
    "mars_m":        ("momentum", "gamma", "ns_steps", "beta1", "beta2", "eps"),
}

_SHARED_EXTRAS = ("schedule", "warmup_iters", "grad_clip", "min_lr",
                  "batch_size", "block_size", "diffusion",
                  "n_layer", "n_head", "d_model", "dropout",
                  "num_diffusion_steps", "beta_schedule")


def _hash_extras(args) -> str:
    parts = []
    for k in _SHARED_EXTRAS + _PER_OPT_EXTRAS.get(args.optimizer, ()):
        if hasattr(args, k):
            parts.append(f"{k}={getattr(args, k)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


def make_exp_name(args) -> str:
    parts = [args.task, args.optimizer,
             f"lr{args.lr:g}", f"wd{args.weight_decay:g}",
             f"seed{args.seed}", f"iters{args.max_iters}"]
    if args.optimizer in ("mars", "mars_m") and hasattr(args, "gamma"):
        parts.append(f"g{args.gamma:g}")
    parts.append(_hash_extras(args))
    return "_".join(parts)


def resolve_out_dir(args) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    return Path(args.exp_root) / make_exp_name(args)
