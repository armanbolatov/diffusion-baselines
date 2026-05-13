"""Auto-name an experiment directory from key hyperparameters.

For sweeps with several tuned axes (betas, momentum, d_coef, ...), the
human-readable prefix alone collides. We append a short hash over every
tuned dim so dirs are guaranteed unique per config.
"""
import hashlib
from pathlib import Path


# Per-optimizer extra fields that affect the run but aren't in the
# (task, optimizer, lr, wd, seed, iters) prefix. Anything here goes into
# the hash so different configs never share a dir.
_PER_OPT_EXTRAS = {
    "adamw":         ("beta1", "beta2", "eps"),
    "lion":          ("beta1", "beta2"),
    "adafactor":     ("beta2", "scale_parameter"),
    "adema":         ("beta1", "beta2", "beta3", "alpha"),
    "soap":          ("beta1", "beta2", "eps",
                      "precondition_frequency", "max_precondition_dim",
                      "shampoo_beta"),
    "prodigy":       ("beta1", "beta2", "d_coef"),
    "schedule_free": ("beta1", "beta2"),
    "muon":          ("momentum", "nesterov", "ns_steps"),
    "mars":          ("beta1", "beta2", "eps", "gamma", "mars_type",
                      "optimize_1d"),
    "mars_adam":     ("beta1", "beta2", "eps", "gamma", "mars_type",
                      "optimize_1d"),
    "sgd":           ("momentum", "nesterov"),
}

# Shared fields that affect every run regardless of optimizer.
_SHARED_EXTRAS = ("schedule", "warmup_iters", "grad_clip", "min_lr",
                  "batch_size", "block_size",
                  "diffusion", "model",
                  "n_layer", "n_head", "d_model", "dropout",
                  "num_diffusion_steps", "beta_schedule")


def _hash_extras(args) -> str:
    parts = []
    for k in _SHARED_EXTRAS + _PER_OPT_EXTRAS.get(args.optimizer, ()):
        if hasattr(args, k):
            parts.append(f"{k}={getattr(args, k)}")
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return digest[:8]


def make_exp_name(args) -> str:
    parts = [args.task, args.optimizer,
             f"lr{args.lr:g}",
             f"wd{args.weight_decay:g}",
             f"seed{args.seed}",
             f"iters{args.max_iters}"]
    if hasattr(args, "gamma") and args.optimizer in ("mars", "mars_adam"):
        parts.append(f"g{args.gamma:g}")
    parts.append(_hash_extras(args))
    return "_".join(parts)


def resolve_out_dir(args) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    return Path(args.exp_root) / make_exp_name(args)
