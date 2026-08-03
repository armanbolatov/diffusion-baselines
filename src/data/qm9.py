"""QM9 small-molecule dataset for discrete diffusion (paper-faithful setup).

Matches kuleshov-group/discrete-diffusion-guidance: canonical SMILES from the
`yairschiff/qm9` dataset, tokenized with `yairschiff/qm9-tokenizer` (vocab 40)
and padded to length 32. Prep is one-time and cached under `<data_dir>/qm9/`.

    uv run python -m src.data.qm9 --block_size 32

Loaders mirror ``text8_loaders``: ``train_iter`` is a one-epoch iterator the
trainer re-builds on ``StopIteration``; ``eval_iter`` is re-iterable.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .mol_tokenizer import SmilesTokenizer

DATASET = "yairschiff/qm9"
FIELD = "canonical_smiles"


def _qm9_dir(data_dir) -> Path:
    return Path(data_dir) / "qm9"


def _split_path(data_dir, split: str, block_size: int) -> Path:
    return _qm9_dir(data_dir) / f"{split}_L{block_size}.npy"


def prepare_qm9(data_dir, block_size: int, val_frac: float = 0.05,
                test_frac: float = 0.05, seed: int = 0) -> None:
    """Download + tokenize + split QM9. Idempotent (skips if already cached)."""
    if all(_split_path(data_dir, s, block_size).exists()
           for s in ("train", "val", "test")):
        return
    _qm9_dir(data_dir).mkdir(parents=True, exist_ok=True)

    import datasets
    ds = datasets.load_dataset(DATASET, split="train")
    smiles = [s for s in ds[FIELD] if s]
    tok = SmilesTokenizer()
    arr = np.asarray([tok.encode(s, block_size) for s in smiles], dtype=np.int64)

    perm = np.random.default_rng(seed).permutation(len(arr))
    arr = arr[perm]
    n = len(arr)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    splits = {
        "test":  arr[:n_test],
        "val":   arr[n_test:n_test + n_val],
        "train": arr[n_test + n_val:],
    }
    for name, a in splits.items():
        np.save(_split_path(data_dir, name, block_size), a)


def get_tokenizer(data_dir=None, block_size=None) -> SmilesTokenizer:
    """The QM9 tokenizer. Vocab is fixed by the HF tokenizer; no data needed."""
    return SmilesTokenizer()


class _Batches:
    """Re-iterable view over a DataLoader that yields the token tensor only."""
    def __init__(self, dl: DataLoader):
        self.dl = dl

    def __iter__(self):
        for (x,) in self.dl:
            yield x


def qm9_loaders(data_dir, block_size, batch_size, eval_batch_size, num_workers):
    prepare_qm9(data_dir, block_size)
    train = torch.from_numpy(np.load(_split_path(data_dir, "train", block_size)))
    val = torch.from_numpy(np.load(_split_path(data_dir, "val", block_size)))
    train_dl = DataLoader(TensorDataset(train), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(TensorDataset(val), batch_size=eval_batch_size,
                        shuffle=False, num_workers=num_workers)
    return iter(_Batches(train_dl)), _Batches(val_dl)


def main() -> None:
    p = argparse.ArgumentParser(description="One-time QM9 prep (paper tokenizer).")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--block_size", type=int, default=32)
    args = p.parse_args()
    prepare_qm9(args.data_dir, args.block_size)
    print(f"QM9 ready under {_qm9_dir(args.data_dir)}/ "
          f"(vocab_size={SmilesTokenizer().vocab_size}, block_size={args.block_size})")


if __name__ == "__main__":
    main()
