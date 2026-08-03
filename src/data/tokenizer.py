"""Char-level tokenizer for text8 (a-z + space).

Vocab: 27 valid chars (a-z=0..25, space=26) + 1 MASK token = 28 total.
Note: a-z = unicode 97..122; we map 'a' -> 0 etc. via subtraction.
"""
import numpy as np

VOCAB_SIZE = 28
MASK_ID = 27


def encode(s: str) -> np.ndarray:
    """Encode an ASCII a-z + space string into uint8 ids."""
    arr = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8).copy()
    out = np.empty_like(arr)
    is_letter = (arr >= 97) & (arr <= 122)
    is_space = (arr == 32)
    out[is_letter] = arr[is_letter] - 97
    out[is_space] = 26
    mask = is_letter | is_space
    return out[mask]


def decode(ids: np.ndarray) -> str:
    chars = []
    for i in ids:
        if i == 26:
            chars.append(" ")
        elif i < 26:
            chars.append(chr(i + 97))
        elif i == MASK_ID:
            chars.append("█")
        else:
            chars.append("?")
    return "".join(chars)
