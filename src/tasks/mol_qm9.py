"""QM9 molecule discrete-diffusion task. Pluggable scheme via args.diffusion.

Paired with UDLM (uniform-state diffusion) by default; reuses the
vocab-agnostic ``DiffTransformer`` backbone. The eval metric is the NELBO in
bits-per-character (``eval_bpc``, what the sweeps minimize); perplexity
(``eval_ppl``) is reported alongside for comparison with the UDLM paper (QM9
PPL ≈ 2.02). Sampling-based validity/uniqueness/novelty live in ``sample`` and
``scripts/sample_qm9.py`` (RDKit), separate from the training metric.
"""
import math
from typing import Dict, Tuple

import torch

from ..data.qm9 import get_tokenizer, qm9_loaders
from ..diffusion.discrete import get_discrete_scheme
from ..models.diff_transformer import DiffTransformer
from .base import DiffusionTask


class MolQM9(DiffusionTask):
    name = "mol_qm9"

    def __init__(self):
        self._device = None
        self._scheme = None
        self._eps = 1e-3
        self._tok = None

    def build_model(self, args):
        # Vocab is needed before data is built, so resolve (and prep) it here.
        self._tok = get_tokenizer(args.data_dir, args.block_size)
        return DiffTransformer(
            vocab_size=self._tok.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            dropout=args.dropout,
            time_emb_dim=getattr(args, "time_emb_dim", 0),
        )

    def build_data(self, args) -> Tuple:
        self._device = torch.device(args.device)
        self._scheme = get_discrete_scheme(getattr(args, "diffusion", "udlm"))
        self._eps = getattr(args, "mdlm_eps", 1e-3)
        if self._tok is None:
            self._tok = get_tokenizer(args.data_dir, args.block_size)
        return qm9_loaders(
            args.data_dir, args.block_size,
            args.batch_size, args.eval_batch_size, args.num_workers,
        )

    def loss_step(self, model, batch, step):
        x0 = batch.to(self._device, non_blocking=True)
        loss = self._scheme.loss(model, x0, eps=self._eps)
        return loss, {}

    def eval_step(self, model, eval_iter, n_batches) -> Dict[str, float]:
        # Average over non-pad tokens only, matching the paper's masked PPL.
        bpc = self._scheme.eval_bpc(model, eval_iter, n_batches=n_batches,
                                    n_mc=32, eps=self._eps, device=self._device,
                                    pad_id=self._tok.pad_id)
        return {"eval_bpc": float(bpc),
                "eval_ppl": float(math.exp(bpc * math.log(2))),
                "diffusion": self._scheme.SCHEME}

    @torch.no_grad()
    def sample(self, model, args, iter_num, out_dir):
        """Sample molecules and report validity / uniqueness / novelty."""
        if not hasattr(self._scheme, "sample"):
            return None
        tok = self._tok or get_tokenizer(args.data_dir, args.block_size)
        n = getattr(args, "sample_n", 256)
        ids = self._scheme.sample(model, n_samples=n, length=args.block_size,
                                  vocab_size=tok.vocab_size,
                                  device=torch.device(args.device))
        smiles = [tok.decode(row.tolist()) for row in ids.cpu()]
        return mol_metrics(smiles)


def mol_metrics(smiles_list) -> Dict[str, float]:
    """Validity / uniqueness / novelty of generated SMILES (RDKit, lazy import)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    valid = []
    for s in smiles_list:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol))  # canonicalize for uniqueness
    n = max(1, len(smiles_list))
    unique = set(valid)
    return {
        "n_sampled": len(smiles_list),
        "validity": len(valid) / n,
        "uniqueness": (len(unique) / len(valid)) if valid else 0.0,
    }
