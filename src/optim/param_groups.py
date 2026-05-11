"""Canonical parameter-group split for matrix-preconditioner optimizers (Muon, SOAP).

Muon (and SOAP) operate on parameter tensors with ndim >= 2. 1-D parameters
(LayerNorm scales, biases, 1-D embeddings) fall back to a scalar optimizer
(typically AdamW).

This is the same split convention used in:
- ~/MARS/MARS_M/optimizers/moonlight.py
- ~/MARS/MARS_M/train_ddpm.py:107-110
- epfml/llm-baselines src/optim/muon.py
"""
from typing import List, Tuple

import torch.nn as nn


def split_for_muon(model: nn.Module,
                   exclude_names: Tuple[str, ...] = ("embed_tokens", "lm_head", "wte", "wpe")
                   ) -> Tuple[List, List]:
    """Split model parameters into (matrix_params, scalar_params).

    `matrix_params`: params with ndim >= 2, excluding any whose names contain
        any string in `exclude_names`. These are the targets for Muon/SOAP.
    `scalar_params`: everything else (LayerNorm, biases, embeddings, lm_head).
        These go to the AdamW backup branch.
    """
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_excluded = any(s in name for s in exclude_names)
        if p.ndim >= 2 and not is_excluded:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    return matrix_params, scalar_params
