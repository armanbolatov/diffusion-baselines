"""--dry_run prints resolved config without training."""
import json
import subprocess
import sys


def test_dry_run_image():
    cmd = [
        sys.executable, "-m", "src.main",
        "--task", "image_cifar10",
        "--optimizer", "adamw",
        "--lr", "1e-4",
        "--max_iters", "10",
        "--dry_run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg["task"] == "image_cifar10"
    assert cfg["optimizer"] == "adamw"
    assert cfg["lr"] == 1e-4
