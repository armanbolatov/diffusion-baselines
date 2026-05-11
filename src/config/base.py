"""Argparse spec for diffusion-baselines.

One CLI flag per hyperparameter. Task- and optimizer-specific flags are
added in conditional groups so unused defaults don't pollute the help output.
"""
import argparse


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffusion-baselines",
        description="Optimizer benchmark across diffusion tasks.",
    )

    # --- top-level dispatch ---
    p.add_argument("--task", required=True,
                   choices=["image_cifar10", "text_text8"])
    p.add_argument("--optimizer", required=True,
                   choices=["adamw", "sgd", "lion", "adafactor", "adema",
                            "soap", "prodigy", "schedule_free",
                            "muon", "mars", "mars_adam"])

    # --- output / logging ---
    p.add_argument("--out_dir", default=None,
                   help="Override auto-generated exp dir under exps/.")
    p.add_argument("--exp_root", default="exps")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="diffusion-baselines")

    # --- system ---
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (slows first iter, faster afterwards)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print resolved config and exit without training")

    # --- training horizon ---
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--min_lr", type=float, default=0.0)

    # --- data ---
    p.add_argument("--data_dir", default="data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)

    # --- optimizer (shared) ---
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="0 disables; otherwise clip grad-norm to this value")

    # SGD
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")

    # AdEMAMix
    p.add_argument("--beta3", type=float, default=0.9999)
    p.add_argument("--alpha", type=float, default=5.0)

    # SOAP
    p.add_argument("--precondition_frequency", type=int, default=10)
    p.add_argument("--max_precondition_dim", type=int, default=10000)
    p.add_argument("--shampoo_beta", type=float, default=None)

    # Prodigy
    p.add_argument("--d_coef", type=float, default=1.0)

    # Adafactor
    p.add_argument("--scale_parameter", action="store_true")

    # Muon
    p.add_argument("--ns_steps", type=int, default=5)

    # MARS / MARS-Adam
    p.add_argument("--gamma", type=float, default=0.025)
    p.add_argument("--mars_type", default="mars-adamw",
                   choices=["mars-adamw", "mars-lion", "mars-shampoo"])
    p.add_argument("--optimize_1d", action="store_true")
    p.add_argument("--lr_1d", type=float, default=3e-3)
    p.add_argument("--beta1_1d", type=float, default=0.9)
    p.add_argument("--beta2_1d", type=float, default=0.95)
    p.add_argument("--weight_decay_1d", type=float, default=0.0)

    # --- image_cifar10 task ---
    p.add_argument("--num_diffusion_steps", type=int, default=1000)
    p.add_argument("--beta_schedule", default="linear",
                   choices=["linear", "scaled_linear", "squaredcos_cap_v2"])

    # --- text_text8 task ---
    p.add_argument("--block_size", type=int, default=256,
                   help="Sequence length for text diffusion")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mdlm_eps", type=float, default=1e-3,
                   help="Lower bound on mask probability t (avoids 1/t→inf at t=0)")

    args = p.parse_args(argv)
    return args
