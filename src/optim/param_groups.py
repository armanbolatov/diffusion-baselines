"""Parameter-group split for matrix-preconditioner optimizers (Muon, SOAP).

Muon (and SOAP) operate on parameter tensors with ndim >= 2. 1-D parameters
(LayerNorm scales, biases, 1-D embeddings) fall back to a scalar optimizer
(typically AdamW).
"""
from typing import List, Tuple

import torch.nn as nn


def split_for_muon(model: nn.Module,
                   exclude_names: Tuple[str, ...] = ("embed_tokens", "lm_head", "wte", "wpe",
                                                     "token_emb", "pos_emb", "head")
                   ) -> Tuple[List, List]:
    """Split model parameters into (matrix_params, scalar_params).

    `matrix_params`: params with ndim >= 2, excluding any whose names contain
        any string in `exclude_names`. These are the targets for Muon/SOAP.
    `scalar_params`: everything else (LayerNorm, biases, embeddings, the
        unembedding head) — the AdamW backup branch, per the Muon convention
        that embeddings/head are never orthogonalized.
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
