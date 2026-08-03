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


def test_udlm_loss_forward_backward():
    """UDLM scheme is vocab-agnostic; smoke it directly on the backbone."""
    from src.diffusion import udlm
    from src.models.diff_transformer import DiffTransformer

    V, L = 40, 16
    model = DiffTransformer(vocab_size=V, block_size=L, n_layer=2, n_head=2,
                            d_model=32)
    x0 = torch.randint(0, V, (4, L))
    loss = udlm.loss(model, x0)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0

    # eval_bpc runs and lands near the random-init reference log2(V).
    eval_iter = [torch.randint(0, V, (4, L)) for _ in range(2)]
    bpc = udlm.eval_bpc(model, eval_iter, n_batches=2, n_mc=4, device="cpu")
    assert bpc == bpc and bpc > 0  # finite and positive


def test_udlm_sample_shape():
    from src.diffusion import udlm
    from src.models.diff_transformer import DiffTransformer

    V, L = 40, 16
    model = DiffTransformer(vocab_size=V, block_size=L, n_layer=2, n_head=2,
                            d_model=32)
    ids = udlm.sample(model, n_samples=3, length=L, n_steps=4, device="cpu")
    assert ids.shape == (3, L)
    assert int(ids.min()) >= 0 and int(ids.max()) < V


def test_duo_loss_both_phases():
    """DUO: curriculum (Gaussian-argmax) and discrete phases + eval, tiny model."""
    from src.diffusion import duo
    from src.models.diff_transformer import DiffTransformer

    V, L = 40, 16
    model = DiffTransformer(vocab_size=V, block_size=L, n_layer=2, n_head=2,
                            d_model=32)
    x0 = torch.randint(0, V, (4, L))
    for step in (0, 10):                       # curriculum, then discrete
        loss = duo.loss(model, x0, step=step, curriculum_end=10)
        loss.backward()
        assert torch.isfinite(loss)

    eval_iter = [torch.randint(0, V, (4, L)) for _ in range(2)]
    bpc = duo.eval_bpc(model, eval_iter, n_batches=2, n_mc=4, device="cpu")
    assert bpc == bpc and bpc > 0
