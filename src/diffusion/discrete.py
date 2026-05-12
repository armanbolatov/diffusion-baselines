"""Discrete-diffusion scheme dispatcher.

`get_discrete_scheme(name)` returns a module with two callables:
    loss(model, x0, eps=...) -> Tensor
    eval_bpc(model, eval_iter, n_batches, n_t_grid, eps, device) -> float
"""
from . import mdlm as _mdlm
from . import udlm as _udlm
from . import arm as _arm
from . import sedd as _sedd

SCHEMES = {
    "mdlm": _mdlm,
    "udlm": _udlm,
    "arm":  _arm,
    "sedd": _sedd,
}


def get_discrete_scheme(name: str):
    name = (name or "mdlm").lower()
    if name not in SCHEMES:
        raise ValueError(f"unknown discrete diffusion scheme: {name}; "
                         f"choose from {list(SCHEMES)}")
    return SCHEMES[name]


# Back-compat shims so existing imports keep working.
add_mask_noise = _mdlm.add_mask_noise
mdlm_loss = _mdlm.loss
mdlm_eval_bpc = _mdlm.eval_bpc
