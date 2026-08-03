"""Discrete-diffusion scheme dispatcher.

Each scheme exposes ``loss(model, x0, eps)`` and ``eval_bpc(model,
eval_iter, n_batches, ...)``.

Schemes are imported lazily so that selecting one (e.g. ``udlm`` for molecules)
does not drag in another scheme's task-specific dependencies (``mdlm`` imports
the text8 tokenizer).
"""
import importlib

SCHEMES = {
    "mdlm": "mdlm",
    "udlm": "udlm",
    "duo": "duo",
}


def get_discrete_scheme(name: str):
    name = (name or "mdlm").lower()
    if name not in SCHEMES:
        raise ValueError(f"unknown discrete diffusion scheme: {name}; "
                         f"choose from {list(SCHEMES)}")
    return importlib.import_module(f".{SCHEMES[name]}", __package__)
