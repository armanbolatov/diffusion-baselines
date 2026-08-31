#!/usr/bin/env python3
"""Export compact, weights-only LM1B checkpoints from benchmark runs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file


OPTIMIZERS = [
    "adamw",
    "lion",
    "muon",
    "schedule_free",
    "soap",
    "mars",
    "mars_m",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_one(source: Path, destination: Path, optimizer: str, seed: int) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    raw_state = payload["model_state"]
    non_fp32 = {
        name: str(tensor.dtype)
        for name, tensor in raw_state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    }
    if non_fp32:
        preview = dict(list(non_fp32.items())[:10])
        raise ValueError(f"{source} contains non-FP32 floating tensors: {preview}")

    # Clone every tensor so tied/shared PyTorch storages become independent,
    # which makes the state dictionary valid for SafeTensors serialization.
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in raw_state.items()
    }
    iteration = int(payload["iter_num"])
    del raw_state, payload
    gc.collect()

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        destination,
        metadata={
            "optimizer": optimizer,
            "seed": str(seed),
            "iteration": str(iteration),
            "precision": "fp32",
            "contents": "final model_state only; optimizer and EMA state excluded",
            "source": str(source),
        },
    )

    dtype_counts = Counter(str(tensor.dtype) for tensor in state.values())
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    tensor_elements = sum(tensor.numel() for tensor in state.values())
    result = {
        "optimizer": optimizer,
        "seed": seed,
        "iteration": iteration,
        "file": str(destination),
        "sha256": sha256(destination),
        "file_bytes": destination.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "tensor_elements": tensor_elements,
        "tensor_count": len(state),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "source_checkpoint": str(source),
    }
    del state
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("exps_bench/lm1b"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/lm1b"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=OPTIMIZERS)
    args = parser.parse_args()

    entries = []
    for optimizer in args.optimizers:
        source = args.root / optimizer / f"seed{args.seed}" / "ckpt.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = args.output / f"{optimizer}_seed{args.seed}_fp32.safetensors"
        print(f"Exporting {optimizer}: {source} -> {destination}", flush=True)
        entry = export_one(source, destination, optimizer, args.seed)
        entries.append(entry)
        print(
            f"  {entry['file_bytes'] / 2**20:.1f} MiB  sha256={entry['sha256']}",
            flush=True,
        )

    manifest = {
        "format": "safetensors",
        "precision": "fp32",
        "contents": "final model_state only; optimizer and EMA state excluded",
        "entries": entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
