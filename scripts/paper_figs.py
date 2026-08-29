"""Generate every figure and table used in the paper, straight from the run logs.

    uv run python scripts/paper_figs.py <output_dir>

Writes <output_dir>/figures/*.pdf and <output_dir>/tables/*.tex. Nothing in the
paper is typed by hand: if a task has not finished, its cells come out as
\\pending{--} so the draft shows exactly what is still missing.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_winners import BENCH_ITERS

OPTS = ["adamw", "lion", "muon", "schedule_free", "soap", "mars", "mars_m"]
NAME = {"adamw": "AdamW", "lion": "Lion", "muon": "Muon",
        "schedule_free": "Schedule-Free", "soap": "SOAP",
        "mars": "MARS", "mars_m": "MARS-M"}
SEEDS = {"seed42", "seed69", "seed1337"}
# task -> (metric used for the headline table, pretty name)
TASKS = {"text8": ("eval_bpc", "text8 (MDLM)"),
         "qm9": ("eval_bpc", "QM9 (UDLM)"),
         "lm1b": ("eval_bpc", "LM1B (DUO)"),
         "celeba64": ("eval_fid", "CelebA-64 (DDPM)")}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (apply_style, PALETTE, MARKERS, LINESTYLES, LW, S_LINE,  # noqa: E402
                    minor_grid, shared_legend)

apply_style()
COLORS = PALETTE

PEND = r"\pending{--}"
# LaTeX macro names cannot contain digits, so task keys need letter-only aliases
MACRO = {"text8": "TextEight", "qm9": "QMNine",
         "lm1b": "LMOneB", "celeba64": "CelebaSixtyFour"}


def curves(task, metric):
    """{optimizer: {iter: [value per seed]}} over completed benchmark seeds."""
    out = defaultdict(lambda: defaultdict(list))
    target = BENCH_ITERS.get(task)
    for f in Path("exps_bench", task).glob("*/seed*/log.jsonl") if Path("exps_bench", task).exists() else []:
        opt, seed = f.parts[-3], f.parts[-2]
        if opt not in OPTS or seed not in SEEDS:
            continue
        pts, last = {}, 0
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
                pts[e["iter"]] = e[metric]
                last = max(last, e["iter"])
        if last >= target:                       # only fully finished seeds count
            for it, v in pts.items():
                out[opt][it].append(v)
    return out


def finals(task, metric):
    """{optimizer: [final value per seed]} at the full budget."""
    c = curves(task, metric)
    t = BENCH_ITERS.get(task)
    return {o: c[o][t] for o in c if c[o].get(t)}


def headline_fid():
    """{opt: [FID per seed]} from logs/fid_celeba.json (50k-sample headline;
    eval_fid in the run logs is the 10k training monitor)."""
    p = Path("logs/fid_celeba.json")
    if not p.exists():
        return {}
    out = defaultdict(list)
    for tag, v in json.loads(p.read_text()).items():
        opt = tag.rsplit("_seed", 1)[0]
        if v.get("n", 0) >= 50000:
            out[opt].append(v["fid"])
    return dict(out)


def kendall(a, b):
    con = dis = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            con += s > 0
            dis += s < 0
    return (con - dis) / (con + dis) if con + dis else float("nan")


def by_seed(task, metric):
    """{optimizer: {seed: final value}} so comparisons can be paired."""
    out = defaultdict(dict)
    target = BENCH_ITERS.get(task)
    base = Path("exps_bench", task)
    for f in (base.glob("*/seed*/log.jsonl") if base.exists() else []):
        opt, seed = f.parts[-3], f.parts[-2]
        if opt not in OPTS or seed not in SEEDS:
            continue
        v, last = None, 0
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") in ("eval", "final_eval") and e.get(metric) is not None:
                if e["iter"] >= last:
                    v, last = e[metric], e["iter"]
        if v is not None and last >= target:
            out[opt][seed] = v
    return out


def fmt(v, nd=4):
    return PEND if v is None else f"{v:.{nd}f}"


def paired(base, tab):
    """Within-seed difference against `base`: {opt: (mean, se, n)}."""
    out = {}
    for o in tab:
        if o == base:
            continue
        d = [tab[o][s] - tab[base][s] for s in tab[o] if s in tab.get(base, {})]
        if len(d) > 1:
            out[o] = (st.mean(d), st.stdev(d) / len(d) ** 0.5, len(d))
    return out


# ---------------------------------------------------------------- figures
def fig_curves(task, metric, ylabel, out, logy=None):
    """Metric vs step, mean over seeds with a +/-1 sd band. logy=None: log
    scale only when the data spans at least a decade."""
    c = curves(task, metric)
    if not c:
        return False
    vals = [v for o in c for i in c[o] for v in c[o][i] if v > 0]
    if logy is None:
        logy = bool(vals) and max(vals) / min(vals) >= 10
    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    for o in OPTS:
        if o not in c:
            continue
        its = sorted(c[o])
        m = [st.mean(c[o][i]) for i in its]
        sd = [st.stdev(c[o][i]) if len(c[o][i]) > 1 else 0 for i in its]
        ax.plot(its, m, color=COLORS[o], lw=LW, ls=LINESTYLES[o], label=NAME[o])
        ax.fill_between(its, [a - b for a, b in zip(m, sd)],
                        [a + b for a, b in zip(m, sd)],
                        color=COLORS[o], alpha=0.15, lw=0)
        # a few markers so the series is readable without relying on colour
        step = max(1, len(its) // 6)
        ax.scatter(its[::step], m[::step], color=COLORS[o],
                   marker=MARKERS[o], s=S_LINE, zorder=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if logy:
        ax.set_yscale("log")
    minor_grid(ax)
    shared_legend(fig, ax, ncols=4)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def _plain_log_ticks(ax):
    """Plain numbers on a log axis. Minor ticks are labeled only when the axis
    spans less than a decade, where majors alone leave it nearly unlabeled;
    on wider ranges labeled minors collide."""
    import math
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    f = ScalarFormatter(); f.set_scientific(False)
    ax.yaxis.set_major_formatter(f)
    lo, hi = ax.get_ylim()
    span = math.log10(hi / lo) if lo > 0 else 99
    fm = ScalarFormatter(); fm.set_scientific(False)
    if span < 1:                      # label every minor, majors are too sparse
        ax.yaxis.set_minor_formatter(fm)
    elif span < 2.5:                  # label only the 2x and 5x minors
        from matplotlib.ticker import LogLocator
        ax.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(fm)
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def _panel(ax, xs_ys, ylabel, xlabel, logy=False):
    for o, xs, ys, sd in xs_ys:
        ax.plot(xs, ys, color=COLORS[o], lw=LW, ls=LINESTYLES[o], label=NAME[o])
        if sd is not None:
            ax.fill_between(xs, [a - b for a, b in zip(ys, sd)],
                            [a + b for a, b in zip(ys, sd)],
                            color=COLORS[o], alpha=0.15, lw=0)
        step = max(1, len(xs) // 6)
        ax.scatter(xs[::step], ys[::step], color=COLORS[o],
                   marker=MARKERS[o], s=S_LINE, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    minor_grid(ax)
    if logy:
        _plain_log_ticks(ax)   # after minor_grid: minorticks_on() resets locators


def fig_task(task, metric, ylabel, out, second="flops", logy=None, ymax=None):
    """Two panels, one legend: metric vs steps, plus metric vs cumulative
    compute (second='flops') or a second metric (second=<metric>)."""
    c = curves(task, metric)
    if not c:
        return False
    if logy is None:
        vals = [v for o in c for i in c[o] for v in c[o][i] if v > 0]
        logy = bool(vals) and max(vals) / min(vals) >= 10
    fl = json.loads(Path("logs/flops.json").read_text()).get(task, {}) if second == "flops" else {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.1), constrained_layout=True)
    series = []
    for o in OPTS:
        if o not in c:
            continue
        its = sorted(c[o])
        m = [st.mean(c[o][i]) for i in its]
        sd = [st.stdev(c[o][i]) if len(c[o][i]) > 1 else 0 for i in its]
        series.append((o, its, m, sd))
    _panel(a1, series, ylabel, "Training step", logy)
    a1.set_xlim(left=0)
    if ymax is not None:
        a1.set_ylim(top=ymax)
    if second == "flops":
        conv = [(o, [i * fl[o] / 1e15 for i in xs], m, sd)
                for o, xs, m, sd in series if o in fl]
        _panel(a2, conv, ylabel, "Training compute (PFLOPs)", logy)
        if ymax is not None:
            a2.set_ylim(top=ymax)
    else:
        c2 = curves(task, second)
        s2 = []
        for o in OPTS:
            if o not in c2:
                continue
            its = sorted(c2[o])
            m = [st.mean(c2[o][i]) for i in its]
            sd = [st.stdev(c2[o][i]) if len(c2[o][i]) > 1 else 0 for i in its]
            s2.append((o, its, m, sd))
        _panel(a2, s2, {"eval_loss": "Denoising MSE"}.get(second, second),
               "Training step", second == "eval_loss")
        a2.set_xlim(left=0)
    shared_legend(fig, a1, ncols=7)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


def fig_placeholder(out, label):
    """Two empty styled panels holding the layout until the data lands."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation bpc")
        ax.set_xlim(0, 30000)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(0.5, 0.5, f"{label}\nresults pending", transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="#B03030")
        minor_grid(ax)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- tables
def tex_main_results(path):
    """Main table: mean +/- sd per (optimizer, task), arrows = paired-vs-AdamW
    difference beyond 2 SE."""
    cols = list(TASKS)
    vals = {t: finals(t, TASKS[t][0]) for t in cols}
    hf = headline_fid()
    if hf:
        vals["celeba64"] = hf
    marks = {}
    for t in cols:
        tab = by_seed(t, TASKS[t][0])
        if t == "celeba64" and hf and all(len(v) >= 3 for v in hf.values()):
            # headline FID has all seeds: pair on the reported metric itself
            import json as _json
            tab = {}
            for tag, v in _json.loads(Path("logs/fid_celeba.json").read_text()).items():
                if v.get("n", 0) >= 50000:
                    o, seed = tag.rsplit("_seed", 1)
                    tab.setdefault(o, {})[seed.split("_")[0]] = v["fid"]
        p = paired("adamw", tab)
        m = {}
        for o, (d, se, _) in p.items():
            if abs(d) > 2 * se:
                m[o] = r"$^{\uparrow}$" if d < 0 else r"$^{\downarrow}$"
        marks[t] = m
    best = {t: (min(st.mean(v) for v in vals[t].values())
                if vals[t] and len(vals[t]) == len(OPTS)
                and all(len(v) >= 3 for v in vals[t].values()) else None)
            for t in cols}
    L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         " & ".join([""] + [TASKS[t][1] for t in cols]) + r" \\",
         " & ".join([""] + [r"bpc $\downarrow$" if TASKS[t][0] == "eval_bpc"
                            else r"FID $\downarrow$" for t in cols]) + r" \\",
         r"\midrule"]
    order = sorted(OPTS, key=lambda o: st.mean(vals["text8"][o]) if o in vals.get("text8", {}) else 9e9)
    for o in order:
        row = [NAME[o]]
        for t in cols:
            v = vals[t].get(o)
            if not v:
                row.append(PEND)
                continue
            nd = 4 if TASKS[t][0] == "eval_bpc" else 2
            m = st.mean(v)
            if len(v) < 3:
                row.append(rf"\pending{{{m:.{nd}f}\,({len(v)} seed)}}")
                continue
            cell = f"{m:.{nd}f} $\\pm$ {st.stdev(v):.{nd}f}"
            if best[t] is not None and abs(m - best[t]) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            cell += marks.get(t, {}).get(o, "")
            row.append(cell)
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def tex_paired(path, task="text8", metric=None):
    """Paired within-seed comparison against AdamW."""
    metric = metric or TASKS[task][0]
    tab = by_seed(task, metric)
    if "adamw" not in tab:
        path.write_text(PEND + "\n")
        return
    p = paired("adamw", tab)
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r" & final " + ("bpc" if metric == "eval_bpc" else "FID") +
         r" & $\Delta$ vs AdamW & significant \\", r"\midrule"]
    nd = 4 if metric == "eval_bpc" else 2
    means = {o: st.mean(list(tab[o].values())) for o in tab}
    for o in sorted(tab, key=lambda o: means[o]):
        if o == "adamw":
            L.append(f"{NAME[o]} & {means[o]:.{nd}f} & --- & --- \\\\")
            continue
        if o not in p:
            continue
        m, se, _ = p[o]
        sig = "yes" if abs(m) > 2 * se else "no"
        L.append(f"{NAME[o]} & {means[o]:.{nd}f} & ${m:+.{nd}f} \\pm {se:.{nd}f}$ & {sig} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def tex_cost(path):
    """Per-step time and peak memory relative to AdamW, plus FLOPs/iter."""
    flops = json.loads(Path("logs/flops.json").read_text()) if Path("logs/flops.json").exists() else {}
    rows = defaultdict(dict)
    for task in TASKS:
        base = Path("exps_bench", task)
        dt, mem = defaultdict(list), defaultdict(list)
        for f in (base.glob("*/seed*/log.jsonl") if base.exists() else []):
            o, seed = f.parts[-3], f.parts[-2]
            if o not in OPTS or seed not in SEEDS:
                continue
            for line in f.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") == "iter" and e.get("iter", 0) > 500:
                    if e.get("dt_iter_s"):
                        dt[o].append(e["dt_iter_s"])
                    if e.get("peak_mem_mb"):
                        mem[o].append(e["peak_mem_mb"])
        if "adamw" in dt:
            b = st.median(dt["adamw"])
            bm = max(mem["adamw"]) if mem.get("adamw") else 1
            for o in dt:
                rows[o][task] = (st.median(dt[o]) / b,
                                 (max(mem[o]) / bm) if mem.get(o) else None)
    tasks = [t for t in TASKS if any(t in rows[o] for o in rows)]
    L = [r"\begin{tabular}{l" + "cc" * len(tasks) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{TASKS[t][1]}}}" for t in tasks) + r" \\",
         " & " + " & ".join(["time & mem"] * len(tasks)) + r" \\", r"\midrule"]
    for o in OPTS:
        cells = []
        for t in tasks:
            if t in rows.get(o, {}):
                s, m = rows[o][t]
                cells += [f"{s:.2f}", f"{m:.2f}" if m else PEND]
            else:
                cells += [PEND, PEND]
        L.append(" & ".join([NAME[o]] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def tex_winners(path):
    """Tuned hyperparameters actually used for every benchmarked run."""
    L = [r"\begin{tabular}{llrrrl}", r"\toprule",
         r"task & optimizer & lr & wd & clip & other \\", r"\midrule"]
    for t in TASKS:
        w = Path("logs", f"tune_{t}_winners.json")
        if not w.exists():
            L.append(rf"{TASKS[t][1]} & \multicolumn{{5}}{{l}}{{{PEND}}} \\")
            continue
        d = json.loads(w.read_text())
        for i, o in enumerate([o for o in OPTS if o in d]):
            p = d[o]["params"]
            extra = ", ".join(r"\texttt{" + f"{k.replace(chr(95), chr(92) + chr(95))}={p[k]}" + "}" for k in
                              ("adamw_lr", "momentum", "nesterov", "gamma",
                               "precondition_frequency", "beta1", "beta2") if k in p)
            L.append(f"{TASKS[t][1] if i == 0 else ''} & {NAME[o]} & {p['lr']:g} & "
                     f"{p['weight_decay']:g} & {p['grad_clip']:g} & \\scriptsize {extra} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    path.write_text("\n".join(L) + "\n")


def tex_stats(path):
    """Numbers quoted inline in the text, as LaTeX macros."""
    out = {}
    t8 = by_seed("text8", "eval_bpc")
    if t8:
        p = paired("adamw", t8)
        out["textEightNSig"] = str(sum(1 for m, se, _ in p.values() if abs(m) > 2 * se))
        out["textEightNTied"] = str(sum(1 for m, se, _ in p.values() if abs(m) <= 2 * se))
    ls, fd = finals("celeba64", "eval_loss"), finals("celeba64", "eval_fid")
    common = [o for o in OPTS if o in ls and o in fd]
    if len(common) >= 3:
        lm = [st.mean(ls[o]) for o in common]
        fm = [st.mean(fd[o]) for o in common]
        out["celebaTau"] = f"{kendall(lm, fm):+.2f}"
        out["celebaMseSpread"] = f"{100 * (max(lm) / min(lm) - 1):.1f}"
        out["celebaFidSpread"] = f"{100 * (max(fm) / min(fm) - 1):.0f}"
        sep = sum(1 for a, b in combinations(common, 2)
                  if abs(st.mean(fd[a]) - st.mean(fd[b])) >
                  2 * (st.stdev(fd[a]) ** 2 / 3 + st.stdev(fd[b]) ** 2 / 3) ** 0.5)
        out["celebaFidSep"] = str(sep)
        sepl = sum(1 for a, b in combinations(common, 2)
                   if abs(st.mean(ls[a]) - st.mean(ls[b])) >
                   2 * (st.stdev(ls[a]) ** 2 / 3 + st.stdev(ls[b]) ** 2 / 3) ** 0.5)
        out["celebaMseSep"] = str(sepl)
        out["celebaNPairs"] = str(len(common) * (len(common) - 1) // 2)
    # Proxy vs full budget rank agreement. Compare like with like: the tuning
    # objective is eval_loss on celeba64 and eval_bpc elsewhere, so the full-budget
    # side must use that same metric, not the headline FID.
    TUNE_METRIC = {"celeba64": "eval_loss"}
    for t in TASKS:
        w = Path("logs", f"tune_{t}_winners.json")
        f = finals(t, TUNE_METRIC.get(t, "eval_bpc"))
        if not w.exists():
            continue
        d = json.loads(w.read_text())
        common = [o for o in OPTS if o in d and o in f and len(f[o]) >= 3]
        if len(common) < 6:                       # too few to rank meaningfully
            continue
        tau = kendall([d[o]["value"] for o in common], [st.mean(f[o]) for o in common])
        out["proxyTau" + MACRO[t]] = f"{tau:+.2f}"
    path.write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in out.items()) + "\n")
    return out


def tex_tasks(path):
    """Task/model/data table, read from the configs so it cannot drift."""
    import yaml
    rows = []
    for t, (_, pretty) in TASKS.items():
        f = Path("configs", t, "common.yaml")
        if not f.exists():
            continue
        c = yaml.safe_load(f.read_text())["parameters"]
        g = lambda k: c.get(k, {}).get("value")
        nparam = {"text8": "16.6M", "qm9": "90.2M", "lm1b": "137.1M", "celeba64": "78.7M"}[t]
        shape = (f"{g('n_layer')}L/{g('d_model')}d/{g('n_head')}h"
                 if g("n_layer") else "UNet, ch 128")
        ctx = g("block_size") or f"{g('image_size')}px"
        rows.append((pretty, nparam, shape,
                     g("batch_size"), ctx, BENCH_ITERS.get(t),
                     c["max_iters"]["value"]))
    L = [r"\begin{tabular}{lrlrrrr}", r"\toprule",
         r"task & params & shape & batch & ctx & steps & tune steps \\",
         r"\midrule"]
    for r in rows:
        L.append(" & ".join(str(x) for x in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def _searched(task, opt):
    """Grid a study actually searched (trial distributions, not today's config,
    which may have been edited since the study ran)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        st = optuna.load_study(study_name=f"{task}_{opt}",
                               storage=f"sqlite:///optuna_db/optuna_{task}_{opt}.db")
    except Exception:
        return None
    done = [t for t in st.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and t.value < 1e9]
    if not done:
        return None
    return {k: list(d.choices) for k, d in done[-1].distributions.items()
            if hasattr(d, "choices")}


def _fmt_vals(vs):
    def one(v):
        f = float(v)
        if f == 0:
            return "0"
        if abs(f) >= 0.01:
            return f"{f:g}"
        mant, exp = f"{f:.0e}".split("e-")
        return rf"{mant}\!\cdot\!10^{{-{int(exp)}}}"
    # separate $...$ per value so a p{} column can break the line between them
    return (r"$\{$" + ", ".join(f"${one(v)}$" for v in sorted(float(x) for x in vs))
            + r"$\}$")


def tex_grid(path):
    """Optimizer-specific axes with the exact value sets each study searched.
    The three shared axes are stated in the appendix prose."""
    SHARED = {"lr", "weight_decay", "grad_clip"}
    L = [r"\begin{tabular}{llp{7.6cm}}", r"\toprule",
         r"optimizer & axis & values \\", r"\midrule"]
    for o in OPTS:
        sw = _searched("text8", o)
        if not sw:
            continue
        own = {k: v for k, v in sorted(sw.items()) if k not in SHARED}
        first = True
        for k, v in own.items():
            name = NAME[o] if first else ""
            first = False
            axis = r"\texttt{" + k.replace("_", r"\_") + "}"
            L.append(f"{name} & {axis} & {_fmt_vals(v)} \\\\")
        if not own:
            L.append(f"{NAME[o]} & --- & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def tex_grid_dev(path):
    """Table of every study whose shared axes differ from the common grid."""
    COMMON = {"lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
              "weight_decay": [0, 0.01, 0.1, 0.3, 1, 3],
              "grad_clip": [0, 1.0]}
    def fmt(v):
        f = float(v)
        return f"{f:g}"
    rows = []
    for task in ("text8", "qm9", "celeba64", "lm1b"):
        for o in OPTS:
            sw = _searched(task, o)
            if not sw:
                continue
            for k, common in COMMON.items():
                got = sw.get(k)
                if got is None:
                    continue
                a = sorted(float(x) for x in got)
                b = sorted(float(x) for x in common)
                if len(a) == len(b) and all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
                    continue
                rows.append((task, NAME[o], k.replace("_", r"\_"),
                             f"$\\{{{fmt(a[0])}, \\ldots, {fmt(a[-1])}\\}}$ ({len(a)})",
                             f"({len(b)})"))
    L = [r"\begin{tabular}{lllll}", r"\toprule",
         r"task & optimizer & axis & searched & common \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")
    return len(rows)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
    fig, tab = out / "figures", out / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    made = []
    if fig_task("text8", "eval_bpc", "Validation bpc", fig / "text8_curves.pdf",
                logy=True, ymax=2.5):
        made.append("text8_curves")
    if fig_task("qm9", "eval_bpc", "Validation bpc", fig / "qm9_curves.pdf",
                logy=True, ymax=1.3):
        made.append("qm9_curves")
    if fig_task("celeba64", "eval_fid", "FID", fig / "celeba_curves.pdf",
                second="eval_loss"):
        made.append("celeba_curves")
    if not fig_task("lm1b", "eval_bpc", "Validation bpc", fig / "lm1b_curves.pdf",
                    logy=True):
        fig_placeholder(fig / "lm1b_curves.pdf", "LM1B (DUO)")
        made.append("lm1b_curves (placeholder)")
    else:
        made.append("lm1b_curves")
    tex_main_results(tab / "main_results.tex")
    tex_paired(tab / "paired_text8.tex")
    tex_paired(tab / "paired_celeba.tex", "celeba64", "eval_fid")
    tex_paired(tab / "paired_qm9.tex", "qm9")
    ndev = tex_grid_dev(tab / "grid_dev.tex")
    print(f"grid deviations from the common axes: {ndev}")
    tex_cost(tab / "cost.tex")
    tex_winners(tab / "winners.tex")
    tex_tasks(tab / "tasks_table.tex")
    tex_grid(tab / "grid_table.tex")
    stats = tex_stats(tab / "stats.tex")
    print("figures:", ", ".join(made))
    print("tables:  main_results, paired_text8, cost, winners, stats")
    for k, v in stats.items():
        print(f"  \\{k} = {v}")


if __name__ == "__main__":
    main()
