"""LM1B (One Billion Words) for masked diffusion — MDLM's paper-faithful setup.

Following Sahoo et al. (kuleshov-group/mdlm) and Appendix D.4: detokenize each
sentence, tokenize with the ``bert-base-uncased`` tokenizer, and pad/truncate to
a fixed length (128). Eval perplexity is computed pad-masked (see the task).

For an optimizer benchmark we don't need all ~30M train sentences, so prep
*streams* the first ``--max_docs`` (default 2M) and caches them — a few minutes
and <1 GB rather than the full multi-GB download.

    uv run python -m src.data.lm1b --block_size 128 --max_docs 2000000

Loaders mirror ``qm9_loaders``: ``train_iter`` is a one-epoch iterator the
trainer re-builds on ``StopIteration``; ``eval_iter`` is re-iterable.
"""
import os
# Hard-cap CPU before transformers/tokenizers load: the fast (rust) tokenizer
# otherwise spawns a rayon thread per core. setdefault lets the caller override.
os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import re
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

TOKENIZER = "bert-base-uncased"
# Parquet mirrors of LM1B (the canonical `lm1b` is a loading script, which
# recent `datasets` no longer supports). Both have train/test splits + a `text`
# column, matching the original One Billion Words benchmark.
DATASET_SOURCES = ("dvruette/lm1b", "jdeschena/lm1b")
DEFAULT_MAX_DOCS = 2_000_000
VAL_DOCS = 10_000
_CHUNK = 10_000            # sentences per tokenizer call during prep


def lm1b_detokenizer(x: str) -> str:
    """MDLM's LM1B detokenizer (kuleshov-group/mdlm dataloader.py), verbatim."""
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def get_tokenizer():
    """The bert-base-uncased tokenizer (vocab 30522, pad_id 0)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _lm1b_dir(data_dir) -> Path:
    return Path(data_dir) / "lm1b"


def _split_path(data_dir, split: str, block_size: int, max_docs=None) -> Path:
    tag = f"_n{max_docs}" if max_docs is not None else ""   # train is keyed by subset size
    return _lm1b_dir(data_dir) / f"{split}_L{block_size}{tag}.npy"


def _stream_split(hf_split: str):
    """Yield raw LM1B text lines from a streaming HF split (id-robust)."""
    import datasets
    last = None
    for name in DATASET_SOURCES:
        try:
            ds = datasets.load_dataset(name, split=hf_split, streaming=True)
            for row in ds:
                yield row["text"]
            return
        except Exception as e:                          # try the next mirror
            last = e
    raise RuntimeError(f"could not stream LM1B from {DATASET_SOURCES}: {last}")


def _tokenize_split(hf_split: str, n_docs: int, block_size: int, tok) -> np.ndarray:
    """Stream ``n_docs`` sentences, detokenize + tokenize, return uint16 [N, L]."""
    out = np.empty((n_docs, block_size), dtype=np.uint16)
    filled = 0
    stream = (lm1b_detokenizer(t) for t in islice(_stream_split(hf_split), n_docs))
    while filled < n_docs:
        chunk = list(islice(stream, _CHUNK))
        if not chunk:
            break
        enc = tok(chunk, max_length=block_size, padding="max_length",
                  truncation=True, add_special_tokens=True,
                  return_attention_mask=False)["input_ids"]
        arr = np.asarray(enc, dtype=np.uint16)
        out[filled:filled + len(arr)] = arr
        filled += len(arr)
        print(f"  {hf_split}: tokenized {filled:,}/{n_docs:,}", flush=True)
    return out[:filled]


def prepare_lm1b(data_dir, block_size: int = 128,
                 max_docs: int = DEFAULT_MAX_DOCS) -> None:
    """Stream + tokenize a bounded LM1B subset. Idempotent (skips if cached)."""
    train_p = _split_path(data_dir, "train", block_size, max_docs)
    val_p = _split_path(data_dir, "val", block_size)
    if train_p.exists() and val_p.exists():
        return
    _lm1b_dir(data_dir).mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    print(f"preparing LM1B (bert tokenizer, L={block_size}, max_docs={max_docs:,})…")
    train = _tokenize_split("train", max_docs, block_size, tok)
    val = _tokenize_split("test", VAL_DOCS, block_size, tok)
    np.save(train_p, train)
    np.save(val_p, val)
    print(f"  wrote {len(train):,} train / {len(val):,} val sequences")


class _Batches:
    """Re-iterable view over a DataLoader that yields the token tensor only."""
    def __init__(self, dl: DataLoader):
        self.dl = dl

    def __iter__(self):
        for (x,) in self.dl:
            yield x


def _load(path) -> torch.Tensor:
    return torch.from_numpy(np.load(path).astype(np.int64))


def lm1b_loaders(data_dir, block_size, batch_size, eval_batch_size,
                 num_workers, max_docs: int = DEFAULT_MAX_DOCS):
    prepare_lm1b(data_dir, block_size, max_docs)
    train = _load(_split_path(data_dir, "train", block_size, max_docs))
    val = _load(_split_path(data_dir, "val", block_size))
    train_dl = DataLoader(TensorDataset(train), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(TensorDataset(val), batch_size=eval_batch_size,
                        shuffle=False, num_workers=num_workers)
    return iter(_Batches(train_dl)), _Batches(val_dl)


def main() -> None:
    p = argparse.ArgumentParser(description="One-time LM1B prep (bert tokenizer).")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--max_docs", type=int, default=DEFAULT_MAX_DOCS)
    args = p.parse_args()
    prepare_lm1b(args.data_dir, args.block_size, args.max_docs)
    tok = get_tokenizer()
    print(f"LM1B ready under {_lm1b_dir(args.data_dir)}/ "
          f"(vocab_size={tok.vocab_size}, pad_id={tok.pad_token_id}, "
          f"block_size={args.block_size})")


if __name__ == "__main__":
    main()
