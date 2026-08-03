"""Per-(task, optimizer) training FLOPs/iter -> logs/flops.json (for plot_final).

backbone/iter = 3 * measured_forward_flops * batch   (backward ~= 2x forward),
measured with torch.utils.flop_counter on the real model — the 6*N*tokens
formula is wrong here (AdaLN conditioning is per-sample, convs reuse params).

optimizer overhead/iter, from the real ndim>=2 param shapes (elementwise
optimizers cost ~12 FLOPs/param — negligible; matrix optimizers are not):
  muon / mars_m : Newton-Schulz-5 -> 5 * (2*m^2*n + 2*m^3) * 2 per matrix
  soap          : grad/update eigenbasis rotations + GG^T accumulation
                  (~6 matmul passes) + eigh((m^3+n^3)*30) / precondition_frequency

    uv run python scripts/compute_flops.py
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.diff_transformer import DiffTransformer
from src.models import build_image_model

ELEMENTWISE = 12  # FLOPs/param: adamw, lion, schedule_free, mars
OPTS = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]


def fwd_flops(model, *inp):
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(*inp)
    return fc.get_total_flops()


def ns5(shapes, steps=5):
    # quintic Newton-Schulz per matrix (m<=n): ~3 matmuls of 2*m*m*n + one 2*m^3, per iter
    tot = 0
    for s in shapes:
        m, n = sorted(s[-2:])
        batch = 1
        for d in s[:-2]:
            batch *= d
        tot += batch * steps * (6 * m * m * n + 2 * m ** 3)
    return tot


def soap(shapes, freq, max_dim=10000):
    # per step: rotate grad + rotate update back + L,R accumulation = 6*m*n*d per
    # preconditioned factor dim d; amortized eigendecomposition ~25*d^3 every
    # `freq` steps. Factor dims > max_precondition_dim (the embedding/head 30k
    # side on lm1b) are skipped by the real optimizer, so skip them here too.
    tot = 0
    for s in shapes:
        m, n = s[-2], s[-1]
        kept = [d for d in (m, n) if d <= max_dim]
        tot += 6 * m * n * sum(kept) + 25 * sum(d ** 3 for d in kept) / freq
    return tot


def table(model, fwd_one, batch, soap_freq):
    from src.optim.param_groups import split_for_muon
    n = sum(p.numel() for p in model.parameters())
    shapes = [tuple(p.shape) for p in model.parameters() if p.ndim >= 2]
    # Newton-Schulz runs only on the muon branch (embeddings/head are AdamW)
    muon_shapes = [tuple(p.shape) for p in split_for_muon(model)[0]]
    backbone = 3 * fwd_one * batch
    ns = ns5(muon_shapes)
    over = {o: ELEMENTWISE * n for o in OPTS}
    over["muon"] = over["mars_m"] = ns + ELEMENTWISE * n
    over["soap"] = soap(shapes, soap_freq) + ELEMENTWISE * n
    return {o: backbone + over[o] for o in OPTS}


def main():
    out = {}
    m = DiffTransformer(vocab_size=40, block_size=32, n_layer=12, n_head=12,
                        d_model=768, time_emb_dim=128)
    out["qm9"] = table(m, fwd_flops(m, torch.zeros(1, 32, dtype=torch.long),
                                    torch.zeros(1)), batch=2048, soap_freq=10)
    m = DiffTransformer(vocab_size=28, block_size=256, n_layer=8, n_head=8, d_model=256)
    out["text8"] = table(m, fwd_flops(m, torch.zeros(1, 256, dtype=torch.long),
                                      torch.zeros(1)), batch=512, soap_freq=10)
    m = build_image_model(argparse.Namespace(), sample_size=64)
    out["celeba64"] = table(m, fwd_flops(m, torch.zeros(1, 3, 64, 64),
                                         torch.zeros(1, dtype=torch.long)),
                            batch=128, soap_freq=10)
    m = DiffTransformer(vocab_size=30522, block_size=128, n_layer=12, n_head=12,
                        d_model=768, time_emb_dim=128)
    out["lm1b"] = table(m, fwd_flops(m, torch.zeros(1, 128, dtype=torch.long),
                                     torch.zeros(1)), batch=512, soap_freq=10)

    Path("logs").mkdir(exist_ok=True)
    Path("logs/flops.json").write_text(json.dumps(out, indent=2))
    for task, t in out.items():
        base = t["adamw"]
        print(f"{task}: adamw {base/1e12:.2f} TFLOPs/iter | " +
              "  ".join(f"{o} +{100*(t[o]-base)/base:.1f}%"
                        for o in ("muon", "soap", "mars_m")))


if __name__ == "__main__":
    main()
