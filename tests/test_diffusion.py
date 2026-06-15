"""Per-scheme loss_step smoke for the discrete (text) task family."""
from argparse import Namespace

import numpy as np
import pytest
import torch

from src.tasks import get_task


@pytest.fixture
def _text8_args(tmp_path):
    data_root = tmp_path / "data"
    text8_dir = data_root / "text8"
    text8_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, n in [("train.bin", 10000), ("val.bin", 2000), ("test.bin", 2000)]:
        arr = rng.integers(0, 27, size=n, dtype=np.uint8)
        mm = np.memmap(text8_dir / name, dtype=np.uint8, mode="w+", shape=(n,))
        mm[:] = arr
        mm.flush()
    return Namespace(
        task="text_text8", optimizer="adamw", device="cpu",
        data_dir=str(data_root), batch_size=2, eval_batch_size=2, num_workers=0,
        block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0,
        mdlm_eps=1e-3,
    )


@pytest.mark.parametrize("scheme", ["mdlm"])
def test_discrete_scheme_loss(scheme, _text8_args):
    args = _text8_args
    args.diffusion = scheme
    task = get_task(args)
    model = task.build_model(args)
    train_iter, _ = task.build_data(args)
    batch = next(train_iter)
    loss, _ = task.loss_step(model, batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
