"""Qualitative samples for the paper, one fixed seed per task.

Writes into the Overleaf checkout: figures/celeba_samples.pdf (7x4 face grid,
same noise per column), tables/text8_samples.tex, tables/qm9_samples.tex,
tables/qm9_quality.tex (validity/uniqueness over 1024 molecules).

    uv run python scripts/paper_samples.py <overleaf_dir> [--gpu 3]
"""
import argparse
import os
import sys
from pathlib import Path

OPTS = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
NAME = {"adamw": "AdamW", "lion": "Lion", "muon": "Muon",
        "schedule_free": "Schedule-Free", "soap": "SOAP",
        "mars": "MARS", "mars_m": "MARS-M"}
SEED = 42          # sampling seed AND the benchmark seed whose ckpt we load


def load(ckpt, model, device):
    import torch
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("ema_state") or payload["model_state"])
    return model.eval()


def celeba(out_pdf, device):
    import torch
    import matplotlib.pyplot as plt
    from diffusers import DDPMScheduler
    from src.models import build_image_model
    import argparse as ap

    n = 4
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    sched.set_timesteps(100, device=device)          # DDIM-100, headline protocol
    # one shared initial noise so columns are comparable across optimizers
    g = torch.Generator(device="cpu").manual_seed(SEED)
    noise0 = torch.randn(n, 3, 64, 64, generator=g).to(device)

    fig, axes = plt.subplots(len(OPTS), n, figsize=(n * 1.5, len(OPTS) * 1.5))
    for r, opt in enumerate(OPTS):
        ckpt = Path("exps_bench/celeba64", opt, f"seed{SEED}", "ckpt.pt")
        model = build_image_model(ap.Namespace(dropout=0.0), sample_size=64).to(device)
        load(ckpt, model, device)
        x = noise0.clone()
        with torch.no_grad():
            for t in sched.timesteps:
                t_b = t.expand(n).long()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    eps = model(x, t_b).sample
                x = sched.step(eps.float(), t, x).prev_sample
        x = ((x.clamp(-1, 1) + 1) / 2).cpu().permute(0, 2, 3, 1).numpy()
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(x[c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(NAME[opt], rotation=0, ha="right", va="center",
                              fontsize=9)
        del model
        torch.cuda.empty_cache()
        print(f"  celeba {opt} done", flush=True)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)


def mdlm_sample(model, n, length, n_steps, device):
    """Ancestral MDLM sampling: start all-mask, unmask with the linear schedule."""
    import torch
    import torch.nn.functional as F
    mask_id = model.head.out_features - 1            # mask token = last vocab index
    x = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    ts = torch.linspace(1.0, 1e-3, n_steps + 1, device=device)
    with torch.no_grad():
        for i in range(n_steps):
            t, s = float(ts[i]), float(ts[i + 1])
            t_b = torch.full((n,), t, device=device)
            logits = model(x, t_b).float()
            logits[..., mask_id] = -1e9              # never sample the mask token
            probs = F.softmax(logits, dim=-1)
            draw = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1
                                     ).reshape(n, length)
            # a masked position is revealed with prob (t - s) / t
            reveal = (torch.rand(n, length, device=device) < (t - s) / t)
            x = torch.where((x == mask_id) & reveal, draw, x)
        # anything still masked at the end is resolved from the last logits
        x = torch.where(x == mask_id, draw, x)
    return x


def text8(out_tex, device):
    import torch
    from src.models.diff_transformer import DiffTransformer
    from src.data.tokenizer import VOCAB_SIZE, decode

    L = [r"\begin{tabular}{ll}", r"\toprule",
         r"optimizer & sample (first 64 of 256 characters) \\", r"\midrule"]
    for opt in OPTS:
        ckpt = Path("exps_bench/text8", opt, f"seed{SEED}", "ckpt.pt")
        model = DiffTransformer(vocab_size=VOCAB_SIZE, block_size=256,
                                n_layer=8, n_head=8, d_model=256).to(device)
        load(ckpt, model, device)
        torch.manual_seed(SEED)
        ids = mdlm_sample(model, n=1, length=256, n_steps=256, device=device)
        txt = decode(ids[0].cpu().numpy())[:64]
        L.append(rf"{NAME[opt]} & \texttt{{{txt}}} \\")
        del model
        torch.cuda.empty_cache()
        print(f"  text8 {opt}: {txt[:40]}...", flush=True)
    L += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(L) + "\n")


def qm9(out_tex, quality_tex, device):
    """Sample molecules per optimizer: a few SMILES for the appendix, and a
    validity/uniqueness table (n=1024, fixed seed) for the results."""
    import torch
    from src.diffusion import udlm
    from src.models.diff_transformer import DiffTransformer
    from src.data.qm9 import get_tokenizer
    from src.tasks.mol_qm9 import mol_metrics

    tok = get_tokenizer("data", 32)
    L = [r"\begin{tabular}{lllr}", r"\toprule",
         r"optimizer & \multicolumn{2}{l}{samples (SMILES)} & valid \\",
         r"\midrule"]
    Q = [r"\begin{tabular}{lrr}", r"\toprule",
         r"optimizer & valid $\uparrow$ & unique $\uparrow$ \\",
         r"\midrule"]
    for opt in OPTS:
        ckpt = Path("exps_bench/qm9", opt, f"seed{SEED}", "ckpt.pt")
        model = DiffTransformer(vocab_size=tok.vocab_size, block_size=32,
                                n_layer=12, n_head=12, d_model=768).to(device)
        load(ckpt, model, device)
        torch.manual_seed(SEED)
        ids = udlm.sample(model, n_samples=1024, length=32, n_steps=128,
                          vocab_size=tok.vocab_size, device=device)
        smiles = [tok.decode(row.tolist()) for row in ids.cpu()]
        m = mol_metrics(smiles)
        def esc(x):
            return x.replace("#", r"\#").replace("%", r"\%").replace("_", r"\_")
        two = [esc(x) for x in smiles[:2]]
        L.append(rf"{NAME[opt]} & \texttt{{{two[0]}}} & \texttt{{{two[1]}}}"
                 rf" & {100 * m.get('validity', 0):.0f}\% \\")
        Q.append(rf"{NAME[opt]} & {100 * m['validity']:.1f}\%"
                 rf" & {100 * m['uniqueness']:.1f}\% \\")
        del model
        torch.cuda.empty_cache()
        print(f"  qm9 {opt}: {m}", flush=True)
    L += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(L) + "\n")
    Q += [r"\bottomrule", r"\end{tabular}"]
    quality_tex.write_text("\n".join(Q) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("overleaf")
    p.add_argument("--gpu", default="3")
    a = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    device = torch.device("cuda")
    out = Path(a.overleaf)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print("== text8 ==", flush=True)
    text8(out / "tables" / "text8_samples.tex", device)
    print("== qm9 ==", flush=True)
    qm9(out / "tables" / "qm9_samples.tex",
        out / "tables" / "qm9_quality.tex", device)
    print("== celeba ==", flush=True)
    celeba(out / "figures" / "celeba_samples.pdf", device)
    print("all samples written", flush=True)


if __name__ == "__main__":
    main()
