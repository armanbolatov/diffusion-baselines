"""Plot training-loss curves for the top-1 config per optimizer.

Pulls each sweep's best finished run by eval_loss, then downloads the full
``loss`` history and renders one overlay figure + one small-multiples figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


SWEEPS = {
    "prodigy":   "4hjyk7hh",
    "muon":      "pnrip3g7",
    "adamw":     "f520y6rg",
    "mars_adam": "zjdahinh",
    "soap":      "jt5rohiz",
}
ENTITY = "steeldream"
PROJECT = "diffusion-baselines-src"
OUT = Path("exps/image_phase_a")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_best(api, sid: str):
    """Pick the run with the lowest eval_loss *among runs that actually
    trained* (i.e. have at least 2 iter-level loss rows). Skips collision
    artifacts that only have a final_eval row."""
    sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
    best, best_v, best_hist = None, float("inf"), None
    for r in sw.runs:
        if r.state != "finished":
            continue
        v = r.summary.get("eval_loss")
        if v is None:
            continue
        rows = list(r.scan_history(keys=["loss", "_step"]))
        loss_rows = [row for row in rows if row.get("loss") is not None]
        if len(loss_rows) < 2:
            continue
        if float(v) < best_v:
            best_v, best, best_hist = float(v), r, loss_rows
    return best, best_v, best_hist


def main() -> None:
    api = wandb.Api()
    curves = {}
    summary_lines = []
    for name, sid in SWEEPS.items():
        run, ev, loss_rows = fetch_best(api, sid)
        if run is None:
            print(f"!! {name}: no trained runs yet")
            continue
        iters = [row["_step"] for row in loss_rows]
        losses = [row["loss"] for row in loss_rows]
        curves[name] = (iters, losses, ev, run.config)
        # Best config signature
        cfg = run.config
        sig_keys = [k for k in ["lr","weight_decay","betas_pair","momentum",
                                "d_coef","beta2","gamma",
                                "precondition_frequency","shampoo_beta"] if k in cfg]
        sig = " ".join(f"{k}={cfg[k]}" for k in sig_keys)
        summary_lines.append(f"{name:10s} eval_loss={ev:.4f}  {sig}")
        print(summary_lines[-1])

    # ----- overlay plot -----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.tab10.colors
    for color, (name, (it, ls, ev, _)) in zip(cmap, curves.items()):
        ax.plot(it, ls, label=f"{name} (best {ev:.4f})", color=color, alpha=0.9, lw=1.2)
    ax.set_xlabel("iter")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.set_title("CIFAR-10 DDPM — training loss for each optimizer's best config (3000 iters, UNet2D)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "loss_curves_overlay.png", dpi=120)
    plt.close()

    # ----- small multiples (linear) -----
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, color, (name, (it, ls, ev, cfg)) in zip(axes, cmap, curves.items()):
        ax.plot(it, ls, color=color, alpha=0.95, lw=1.1)
        ax.set_xlabel("iter")
        ax.set_title(f"{name}\nmin eval_loss={ev:.4f}", fontsize=11)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        # annotate the winning hyperparams on the plot
        sig_keys = [k for k in ["lr","weight_decay","betas_pair","momentum",
                                "d_coef","beta2","gamma",
                                "precondition_frequency","shampoo_beta"] if k in cfg]
        sig = "\n".join(f"{k}={cfg[k]}" for k in sig_keys)
        ax.text(0.98, 0.95, sig, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    axes[0].set_ylabel("training loss (log scale)")
    plt.tight_layout()
    plt.savefig(OUT / "loss_curves_small_multiples.png", dpi=120)
    plt.close()

    # also: zoom plot of the tail (iter >= 500)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for color, (name, (it, ls, ev, _)) in zip(cmap, curves.items()):
        ix = [i for i, j in enumerate(it) if j >= 500]
        ax.plot([it[i] for i in ix], [ls[i] for i in ix],
                label=f"{name} ({ev:.4f})", color=color, alpha=0.9, lw=1.2)
    ax.set_xlabel("iter")
    ax.set_ylabel("training loss")
    ax.set_title("CIFAR-10 DDPM — training loss (tail, iter>=500)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "loss_curves_tail.png", dpi=120)
    plt.close()

    (OUT / "winners.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nplots: {OUT}/loss_curves_*.png")


if __name__ == "__main__":
    main()
