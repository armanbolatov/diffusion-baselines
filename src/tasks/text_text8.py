"""text8 discrete-diffusion task. Pluggable scheme via args.diffusion."""
from typing import Dict, Tuple

import torch

from ..data.text8 import text8_loaders
from ..data.tokenizer import VOCAB_SIZE
from ..diffusion.discrete import get_discrete_scheme
from ..models.diff_transformer import DiffTransformer
from .base import DiffusionTask


class TextText8(DiffusionTask):
    name = "text_text8"

    def __init__(self):
        self._device = None
        self._scheme = None
        self._eps = 1e-3

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
        self._scheme = get_discrete_scheme(getattr(args, "diffusion", "mdlm"))
        self._eps = getattr(args, "mdlm_eps", 1e-3)
        train_iter, eval_iter = text8_loaders(
            args.data_dir, args.block_size,
            args.batch_size, args.eval_batch_size, args.num_workers,
        )
        return train_iter, eval_iter

    def loss_step(self, model, batch, step):
        x0 = batch.to(self._device, non_blocking=True)
        loss = self._scheme.loss(model, x0, eps=self._eps)
        return loss, {}

    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        bpc = self._scheme.eval_bpc(model, eval_iter, n_batches=n_batches,
                                    n_t_grid=16, eps=self._eps, device=self._device)
        return {"eval_bpc": float(bpc), "diffusion": self._scheme.SCHEME}
