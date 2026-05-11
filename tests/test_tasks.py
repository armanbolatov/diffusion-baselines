"""Both tasks build their model and execute one loss step on a tiny batch."""
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks import get_task


def _base_args(task):
    return Namespace(
        task=task, optimizer="adamw", lr=1e-3, weight_decay=0.0,
        beta1=0.9, beta2=0.999, eps=1e-8,
        device="cpu",
        data_dir="tests/_tmpdata", batch_size=2, eval_batch_size=2, num_workers=0,
        # image
        num_diffusion_steps=10, beta_schedule="linear",
        # text
        block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0, mdlm_eps=1e-3,
    )


def test_image_cifar10_loss_step(tmp_path):
    pytest.importorskip("torchvision")
    args = _base_args("image_cifar10")
    args.data_dir = str(tmp_path)
    task = get_task(args)
    model = task.build_model(args)
    train_iter, _ = task.build_data(args)
    batch = next(train_iter)
    loss, _ = task.loss_step(model, batch, 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_text_text8_loss_step(tmp_path, monkeypatch):
    # Build fake text8 .bin files so we don't need the real download.
    data_root = tmp_path / "data"
    text8_dir = data_root / "text8"
    text8_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, n in [("train.bin", 10000), ("val.bin", 2000), ("test.bin", 2000)]:
        arr = rng.integers(0, 27, size=n, dtype=np.uint8)
        mm = np.memmap(text8_dir / name, dtype=np.uint8, mode="w+", shape=(n,))
        mm[:] = arr
        mm.flush()

    args = _base_args("text_text8")
    args.data_dir = str(data_root)
    task = get_task(args)
    model = task.build_model(args)
    train_iter, _ = task.build_data(args)
    batch = next(train_iter)
    loss, _ = task.loss_step(model, batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
