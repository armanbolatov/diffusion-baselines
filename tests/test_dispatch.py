"""--dry_run prints resolved config without training."""
import json
import subprocess
import sys


def test_dry_run_text():
    cmd = [
        sys.executable, "-m", "src.main",
        "--task", "text_text8",
        "--diffusion", "mdlm",
        "--optimizer", "adamw",
        "--lr", "1e-4",
        "--max_iters", "10",
        "--dry_run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg["task"] == "text_text8"
    assert cfg["optimizer"] == "adamw"
    assert cfg["lr"] == 1e-4
