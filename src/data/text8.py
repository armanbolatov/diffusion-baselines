"""Download + char-tokenize text8 into np.memmap .bin files.

Usage (one-time prep):
    uv run python -m src.data.text8

Output:
    data/text8/train.bin   (90M uint8)
    data/text8/val.bin     ( 5M uint8)
    data/text8/test.bin    ( 5M uint8)
"""
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from .tokenizer import encode

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def _ensure_text8(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    bin_paths = [data_dir / s for s in ("train.bin", "val.bin", "test.bin")]
    if all(p.exists() for p in bin_paths):
        return data_dir

    zip_path = data_dir / "text8.zip"
    if not zip_path.exists():
        print(f"downloading text8 to {zip_path}…")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)
    print("extracting text8…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    raw_path = data_dir / "text8"
    with open(raw_path, "r") as f:
        raw = f.read()
    print(f"tokenizing text8 ({len(raw):,} chars)…")
    ids = encode(raw)
    n = len(ids)
    n_train, n_val = 90_000_000, 5_000_000
    splits = {
        "train.bin": ids[:n_train],
        "val.bin":   ids[n_train:n_train + n_val],
        "test.bin":  ids[n_train + n_val:],
    }
    for name, arr in splits.items():
        out = data_dir / name
        arr_mm = np.memmap(out, dtype=np.uint8, mode="w+", shape=(len(arr),))
        arr_mm[:] = arr
        arr_mm.flush()
        print(f"  wrote {out} ({len(arr):,} ids)")
    return data_dir


def text8_loaders(data_root: str, block_size: int, batch_size: int,
                  eval_batch_size: int, num_workers: int = 2):
    """Returns (train_iter, eval_iter) yielding (x: LongTensor[B, T]) on CPU.
    Train is an infinite random-window sampler; eval is a deterministic span sampler."""
    data_dir = _ensure_text8(Path(data_root) / "text8")
    train_arr = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    val_arr   = np.memmap(data_dir / "val.bin",   dtype=np.uint8, mode="r")

    train_iter = _TextRandomWindowIter(train_arr, block_size, batch_size)
    eval_iter  = _TextEvalIter(val_arr, block_size, eval_batch_size, num_batches=200)
    return train_iter, eval_iter


class _TextRandomWindowIter:
    def __init__(self, arr, block_size, batch_size):
        self.arr = arr
        self.block_size = block_size
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        ix = np.random.randint(0, len(self.arr) - self.block_size, size=(self.batch_size,))
        x = np.stack([self.arr[i:i + self.block_size].astype(np.int64) for i in ix])
        return torch.from_numpy(x)


class _TextEvalIter:
    """Deterministic eval: takes `num_batches * batch_size` consecutive windows."""
    def __init__(self, arr, block_size, batch_size, num_batches=200):
        self.arr = arr
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        N = len(self.arr)
        usable = (N // self.block_size) * self.block_size
        max_windows = self.num_batches * self.batch_size
        starts = np.arange(0, usable - self.block_size, self.block_size)[:max_windows]
        for i in range(0, len(starts), self.batch_size):
            batch_starts = starts[i:i + self.batch_size]
            if len(batch_starts) < self.batch_size:
                return
            x = np.stack([self.arr[s:s + self.block_size].astype(np.int64) for s in batch_starts])
            yield torch.from_numpy(x)


if __name__ == "__main__":
    _ensure_text8(Path("data") / "text8")
    print("text8 ready.")
