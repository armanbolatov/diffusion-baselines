"""LM1B discrete-diffusion task (MDLM or DUO).

Bert-base-uncased tokenizer, context 128, on the shared ``DiffTransformer``
backbone. With ``--diffusion mdlm`` the model vocab is the tokenizer vocab + 1
(the extra last index is the absorbing MASK token); DUO is uniform-state, so no
MASK slot. DUO's Gaussian curriculum covers the first half of ``max_iters``
(the paper's 500k-of-1M fraction). Eval PPL is pad-masked in both schemes.
"""
import math
from typing import Dict, Tuple

import torch

from ..data.lm1b import DEFAULT_MAX_DOCS, get_tokenizer, lm1b_loaders
from ..diffusion.discrete import get_discrete_scheme
from ..models.diff_transformer import DiffTransformer
from .base import DiffusionTask


class TextLM1B(DiffusionTask):
    name = "text_lm1b"

    def __init__(self):
        self._device = None
        self._scheme = None
        self._eps = 1e-3
        self._pad_id = None
        self._curriculum_end = 0

    def build_model(self, args):
        tok = get_tokenizer()
        self._pad_id = tok.pad_token_id
        mask_slot = getattr(args, "diffusion", "mdlm") == "mdlm"
        return DiffTransformer(
            vocab_size=tok.vocab_size + int(mask_slot),   # +1 = absorbing MASK (mdlm only)
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            dropout=args.dropout,
            time_emb_dim=getattr(args, "time_emb_dim", 0),
        )

    def build_data(self, args) -> Tuple:
        self._device = torch.device(args.device)
        self._scheme = get_discrete_scheme(getattr(args, "diffusion", "mdlm"))
        self._eps = getattr(args, "mdlm_eps", 1e-3)
        self._curriculum_end = args.max_iters // 2      # duo only; ignored by mdlm
        if self._pad_id is None:
            self._pad_id = get_tokenizer().pad_token_id
        return lm1b_loaders(
            args.data_dir, args.block_size,
            args.batch_size, args.eval_batch_size, args.num_workers,
            max_docs=getattr(args, "max_docs", DEFAULT_MAX_DOCS),
        )

    def loss_step(self, model, batch, step):
        x0 = batch.to(self._device, non_blocking=True)
        if self._scheme.SCHEME == "duo":
            loss = self._scheme.loss(model, x0, step=step,
                                     curriculum_end=self._curriculum_end,
                                     eps=self._eps, pad_id=self._pad_id)
        else:
            loss = self._scheme.loss(model, x0, eps=self._eps, pad_id=self._pad_id)
        return loss, {}

    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        bpc = self._scheme.eval_bpc(model, eval_iter, n_batches=n_batches,
                                    n_mc=32, eps=self._eps, device=self._device,
                                    pad_id=self._pad_id)
        return {"eval_bpc": float(bpc),
                "eval_ppl": float(math.exp(bpc * math.log(2))),
                "diffusion": self._scheme.SCHEME}
