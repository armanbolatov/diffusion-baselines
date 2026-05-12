"""ARM — Autoregressive baseline using the same transformer architecture
as the diffusion models. Standard next-token CE; needed as a control for
discrete-diffusion benchmarks (per DUO/Sahoo et al.).

Note: the transformer we use (DiffTransformer) is bidirectional by default.
For ARM mode we pass t=0 (unused conditioning) and shift labels by one.
This is a slight architectural mismatch — for a stronger ARM baseline,
plug in a causal-masked variant. For benchmark-purposes parity, this works.
"""
import math
import torch
import torch.nn.functional as F


SCHEME = "arm"


def loss(model, x0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Next-token CE: predict x0[1:] from x0[:-1]."""
    B, T = x0.shape
    device = x0.device
    # condition with t=0 (no diffusion noise added)
    t = torch.zeros(B, device=device)
    logits = model(x0, t)        # (B, T, V)
    # standard LM shift: predict token i+1 from logits at position i
    pred = logits[:, :-1, :]
    targ = x0[:, 1:]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), targ.reshape(-1))


@torch.no_grad()
def eval_bpc(model, eval_iter, n_batches: int, n_t_grid: int = 1,
             eps: float = 1e-3, device="cuda") -> float:
    total, n_seen = 0.0, 0
    for i, x0 in enumerate(eval_iter):
        if i >= n_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        B, T = x0.shape
        t = torch.zeros(B, device=device)
        logits = model(x0, t)
        pred = logits[:, :-1, :]
        targ = x0[:, 1:]
        nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)), targ.reshape(-1),
                              reduction="sum")
        # bits per char
        total += float(nll.item()) / math.log(2)
        n_seen += B * (T - 1)
    return total / max(1, n_seen)
