"""Sample molecules from a QM9 UDLM checkpoint and report validity/uniqueness/novelty.

Defaults to raw (final-iter) weights; pass --ema to load EMA weights instead.

    uv run python scripts/sample_qm9.py exps_bench/qm9/<opt>/<run>/ckpt.pt --n 1000
"""
import argparse
from pathlib import Path

import torch

from src.data.qm9 import get_tokenizer
from src.diffusion import udlm
from src.models.diff_transformer import DiffTransformer
from src.tasks.mol_qm9 import mol_metrics


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--ema", action="store_true",
                   help="use EMA weights (default: raw)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(args.data_dir, args.block_size)
    model = DiffTransformer(vocab_size=tok.vocab_size, block_size=args.block_size,
                            n_layer=args.n_layer, n_head=args.n_head,
                            d_model=args.d_model).to(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    weights = (payload["ema_state"] if args.ema and payload.get("ema_state")
               else payload["model_state"])
    model.load_state_dict(weights)
    model.eval()

    ids = udlm.sample(model, n_samples=args.n, length=args.block_size,
                      n_steps=args.n_steps, vocab_size=tok.vocab_size, device=device)
    smiles = [tok.decode(row.tolist()) for row in ids.cpu()]
    metrics = mol_metrics(smiles)
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in metrics.items()})


if __name__ == "__main__":
    main()
