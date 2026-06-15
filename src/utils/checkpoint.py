"""Checkpoint save/load with auto-resume support."""
from pathlib import Path
from typing import Optional

import torch


def save_ckpt(path: Path, model, optimizer, iter_num: int,
              ema_state: Optional[dict] = None,
              extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iter_num": iter_num,
        "ema_state": ema_state,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_ckpt(path: Path, model, optimizer, device="cuda"):
    """Returns (iter_num, ema_state_or_None)."""
    if not Path(path).exists():
        return 0, None
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["iter_num"]), payload.get("ema_state")
