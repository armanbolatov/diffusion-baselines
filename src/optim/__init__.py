"""Optimizer dispatcher.

7 optimizers used in the benchmark:
  adamw, lion, muon, schedule_free, soap, mars, mars_m
"""
from typing import Any, Dict, Tuple

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

    elif name == "lion":
        from .lion import Lion
        opt = Lion(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

    elif name == "soap":
        from .soap import SOAP
        opt = SOAP(model.parameters(), lr=lr, betas=betas, weight_decay=wd,
                   precondition_frequency=args.precondition_frequency,
                   max_precondition_dim=args.max_precondition_dim,
                   shampoo_beta=args.shampoo_beta, eps=eps)

    elif name == "schedule_free":
        from .schedule_free import ScheduleFreeAdamW
        opt = ScheduleFreeAdamW(model.parameters(), lr=lr, betas=betas, eps=eps,
                                weight_decay=wd, warmup_steps=args.warmup_iters)

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
        return opt, summary

    elif name == "mars":
        from .mars import MARSAdam
        opt = MARSAdam(model.parameters(), lr=lr, betas=betas, eps=eps,
                       weight_decay=wd, gamma=args.gamma, log_smoothness=False)

    elif name == "mars_m":
        from .mars_m import MARS_M
        muon_params, adamw_params = split_for_muon(model)
        opt = MARS_M(muon_params=muon_params, adamw_params=adamw_params,
                     lr=lr, wd=wd, momentum=args.momentum, ns_steps=args.ns_steps,
                     adamw_betas=betas, adamw_eps=eps,
                     gamma=args.gamma, clip_c=False)
        summary["n_muon_params"] = _count_params(muon_params)
        summary["n_adamw_params"] = _count_params(adamw_params)
        summary["n_params"] = summary["n_muon_params"] + summary["n_adamw_params"]
        return opt, summary

    else:
        raise ValueError(f"unknown optimizer: {name}")

    summary["n_params"] = _count_params(model.parameters())
    return opt, summary


OPTIMIZERS = (
    "adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m",
)
