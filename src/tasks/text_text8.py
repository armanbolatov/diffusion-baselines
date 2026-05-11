"""text8 discrete-diffusion task (MDLM)."""
from typing import Dict, Tuple

import torch

from ..data.text8 import text8_loaders
from ..data.tokenizer import VOCAB_SIZE
from ..diffusion.discrete import mdlm_loss, mdlm_eval_bpc
from ..models.diff_transformer import DiffTransformer
from .base import DiffusionTask


class TextText8(DiffusionTask):
    name = "text_text8"

    def __init__(self):
        self._device = None

    def build_model(self, args):
        return DiffTransformer(
            vocab_size=VOCAB_SIZE,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            dropout=args.dropout,
        )

    def build_data(self, args) -> Tuple:
        self._device = torch.device(args.device)
        train_iter, eval_iter = text8_loaders(
            args.data_dir, args.block_size,
            args.batch_size, args.eval_batch_size, args.num_workers,
        )
        return train_iter, eval_iter

    def loss_step(self, model, batch, step):
        x0 = batch.to(self._device, non_blocking=True)
        loss = mdlm_loss(model, x0, eps=getattr(model, "_mdlm_eps", 1e-3))
        return loss, {}

    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        bpc = mdlm_eval_bpc(model, eval_iter, n_batches=n_batches,
                            n_t_grid=16, eps=1e-3, device=self._device)
        return {"eval_bpc": float(bpc)}
