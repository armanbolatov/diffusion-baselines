"""Optimizer dispatcher.

`build_optimizer(args, model)` returns (optimizer, summary_dict).

Eleven optimizers, three integration patterns:
- Matrix-aware (Muon): needs param-split via param_groups.split_for_muon.
- Variance-reduced (MARS, MARS-Adam): flat params; expose --gamma.
- Plain Adam family + others: flat params.
"""
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

from .param_groups import split_for_muon


def _count_params(params) -> int:
    return sum(p.numel() for p in params if p.requires_grad)


def build_optimizer(args, model: nn.Module) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    name = args.optimizer.lower()
    lr = args.lr
    wd = args.weight_decay
    betas = (args.beta1, args.beta2)
    eps = args.eps
    summary: Dict[str, Any] = {"optimizer": name, "lr": lr}

    if name == "adamw":
        from .adamw import AdamW
        opt = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "sgd":
        from .sgd import SGD
        opt = SGD(model.parameters(), lr=lr, momentum=args.momentum,
                  weight_decay=wd, nesterov=args.nesterov)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "lion":
        from .lion import Lion
        opt = Lion(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "adafactor":
        from .adafactor import Adafactor
        opt = Adafactor(model.parameters(), lr=lr, weight_decay=wd,
                        scale_parameter=args.scale_parameter, relative_step=False)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "adema":
        from .adema import AdEMAMix
        betas3 = (args.beta1, args.beta2, args.beta3)
        opt = AdEMAMix(model.parameters(), lr=lr, betas=betas3,
                       weight_decay=wd, alpha=args.alpha, eps=eps)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "soap":
        from .soap import SOAP
        opt = SOAP(model.parameters(), lr=lr, betas=betas, weight_decay=wd,
                   precondition_frequency=args.precondition_frequency,
                   max_precondition_dim=args.max_precondition_dim,
                   shampoo_beta=args.shampoo_beta, eps=eps)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "prodigy":
        from .prodigy import Prodigy
        opt = Prodigy(model.parameters(), lr=lr, betas=betas, eps=eps,
                      weight_decay=wd, d_coef=args.d_coef, safeguard_warmup=True)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "schedule_free":
        from .schedule_free import ScheduleFreeAdamW
        opt = ScheduleFreeAdamW(model.parameters(), lr=lr, betas=betas, eps=eps,
                                weight_decay=wd, warmup_steps=args.warmup_iters)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "muon":
        from .muon import Muon
        muon_params, adamw_params = split_for_muon(model)
        opt = Muon(muon_params=muon_params, adamw_params=adamw_params,
                   lr=lr, wd=wd, momentum=args.momentum,
                   nesterov=args.nesterov, ns_steps=args.ns_steps,
                   adamw_betas=betas, adamw_eps=eps)
        summary["n_muon_params"] = _count_params(muon_params)
        summary["n_adamw_params"] = _count_params(adamw_params)
        summary["n_params"] = summary["n_muon_params"] + summary["n_adamw_params"]

    elif name == "mars":
        from .mars import MARS
        opt = MARS(model.parameters(), lr=lr, betas=betas, eps=eps,
                   weight_decay=wd, gamma=args.gamma,
                   mars_type=args.mars_type, optimize_1d=args.optimize_1d,
                   lr_1d=args.lr_1d, betas_1d=(args.beta1_1d, args.beta2_1d),
                   weight_decay_1d=args.weight_decay_1d)
        summary["n_params"] = _count_params(model.parameters())

    elif name == "mars_adam":
        from .mars_adam import MARSAdam
        opt = MARSAdam(model.parameters(), lr=lr, betas=betas, eps=eps,
                       weight_decay=wd, gamma=args.gamma, log_smoothness=False)
        summary["n_params"] = _count_params(model.parameters())

    else:
        raise ValueError(f"unknown optimizer: {name}")

    return opt, summary


OPTIMIZERS = (
    "adamw", "sgd", "lion", "adafactor", "adema", "soap",
    "prodigy", "schedule_free", "muon", "mars", "mars_adam",
)
