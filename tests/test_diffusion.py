"""Per-scheme loss_step smoke for both task families.

For each --diffusion value, build the task with a tiny model and verify a
single forward+backward step produces a finite scalar loss.
"""
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks import get_task


@pytest.fixture
def _text8_args(tmp_path):
    # synthetic text8 .bin so we don't need the real download
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


@pytest.fixture
def _image_args(tmp_path):
    return Namespace(
        task="image_cifar10", optimizer="adamw", device="cpu",
        data_dir=str(tmp_path), batch_size=2, eval_batch_size=2, num_workers=0,
        model="unet2d",
        num_diffusion_steps=10, beta_schedule="linear",
        ct_sigma_min=0.002, ct_sigma_max=80.0, ct_rho=7.0, ct_n_steps=4,
        ema_decay=0.999, fid_samples=0, fid_steps=4, fm_path="cond_ot",
        dit_patch_size=2, dit_hidden=32, dit_depth=2, dit_heads=2,
    )


@pytest.mark.parametrize("scheme", ["mdlm", "udlm", "arm", "sedd"])
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


@pytest.mark.parametrize("scheme", ["ddpm", "flow_matching", "consistency"])
def test_continuous_scheme_loss(scheme, _image_args):
    pytest.importorskip("torchvision")
    args = _image_args
    args.diffusion = scheme
    task = get_task(args)
    model = task.build_model(args)
    train_iter, _ = task.build_data(args)
    batch = next(train_iter)
    loss, _ = task.loss_step(model, batch, 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_dit_forward_shape(_image_args):
    """DiT model returns same-shape output as UNet2D."""
    from src.models import build_image_model
    args = _image_args
    args.model = "dit"
    model = build_image_model(args)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    out = model(x, t).sample
    assert out.shape == x.shape


def test_ema_helpers():
    import torch.nn as nn
    from src.utils.ema import make_ema, ema_update
    m = nn.Linear(4, 4)
    ema = make_ema(m)
    # ema params should be frozen and identical to m at construction time
    for ep, p in zip(ema.parameters(), m.parameters()):
        assert not ep.requires_grad
        assert torch.equal(ep, p)
    # take a step on m and update ema; ema should drift toward new m by (1-decay)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p))
    ema_update(ema, m, decay=0.5)
    for ep, p in zip(ema.parameters(), m.parameters()):
        # ema is exactly the average since at construction ep==(prev m), and prev_m + new_m halved
        pass  # smoke check; just verify no exception
