"""Argparse spec for diffusion-baselines."""
import argparse


def _str2bool(s) -> bool:
    return str(s).lower() in ("1", "true", "yes")


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffusion-baselines",
        description="Optimizer benchmark across diffusion tasks.",
    )

    p.add_argument("--task", required=True,
                   choices=["image_celeba", "text_text8", "text_lm1b", "mol_qm9"])
    p.add_argument("--optimizer", required=True,
                   choices=["adamw", "lion", "muon", "schedule_free",
                            "soap", "mars", "mars_m"])
    p.add_argument("--diffusion", default="ddpm",
                   choices=["ddpm", "mdlm", "udlm", "duo"])

    p.add_argument("--out_dir", default=None)
    p.add_argument("--exp_root", default="exps")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=10)

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--kill_at_iter", type=int, default=0)
    p.add_argument("--kill_if_above", type=float, default=0.0)

    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "constant", "wsd"])
    p.add_argument("--min_lr", type=float, default=0.0)

    p.add_argument("--data_dir", default="data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_docs", type=int, default=2_000_000,
                   help="LM1B only: number of train sentences to stream + cache")
    p.add_argument("--image_size", type=int, default=64,
                   help="CelebA only: square resolution (32 or 64)")
    p.add_argument("--ema_eval", type=int, default=0,
                   help="eval/sample from EMA weights instead of raw (1=on; needs ema_decay>0)")
    p.add_argument("--fid_n", type=int, default=0,
                   help="CelebA: images to sample for in-training FID (0 = off)")
    p.add_argument("--fid_steps", type=int, default=50,
                   help="CelebA: DDIM steps for in-training FID sampling")
    p.add_argument("--fid_stride", type=int, default=1,
                   help="CelebA: compute FID every Nth eval (keeps overhead down)")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="micro-batch accumulation steps (same effective batch)")

    p.add_argument("--precondition_frequency", type=int, default=10)
    p.add_argument("--max_precondition_dim", type=int, default=10000)
    p.add_argument("--shampoo_beta", type=float, default=None)

    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", type=_str2bool, default=True,
                   help="muon: Nesterov-style momentum (swept true/false)")
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--adamw_lr", type=float, default=None,
                   help="muon/mars_m: separate lr for the AdamW backup branch; None = same as --lr")

    p.add_argument("--gamma", type=float, default=0.025)

    p.add_argument("--num_diffusion_steps", type=int, default=1000)
    p.add_argument("--beta_schedule", default="linear",
                   choices=["linear", "scaled_linear", "squaredcos_cap_v2"])

    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--time_emb_dim", type=int, default=0,
                   help="AdaLN conditioning width; 0 = 4*d_model (default)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mdlm_eps", type=float, default=1e-3)
    p.add_argument("--sample_n", type=int, default=256)

    return p.parse_args(argv)
