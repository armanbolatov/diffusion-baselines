"""Event-typed JSONL logger.

Single file per run: ``{out_dir}/log.jsonl``. Each line is a JSON object with
an ``event`` field discriminating among config / params / iter / eval /
sample / final_eval / diverged.
"""
import json
from pathlib import Path
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "log.jsonl"
        # line-buffered so logs flush as runs progress
        self.f = open(self.path, "a", buffering=1)

    def log(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"event": event}
        payload.update(fields)
        self.f.write(json.dumps(payload, default=_default) + "\n")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _default(o):
    # numpy / torch numbers
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "item"):
        return o.item()
    return str(o)
