"""Discrete-diffusion scheme dispatcher.

Each scheme exposes ``loss(model, x0, eps)`` and ``eval_bpc(model,
eval_iter, n_batches, ...)``.
"""
from . import mdlm as _mdlm


SCHEMES = {
    "mdlm": _mdlm,
}


def get_discrete_scheme(name: str):
    name = (name or "mdlm").lower()
    if name not in SCHEMES:
        raise ValueError(f"unknown discrete diffusion scheme: {name}; "
                         f"choose from {list(SCHEMES)}")
    return SCHEMES[name]
