"""Shared plot style for every figure in the paper.

Seaborn darkgrid on a soft blue-gray panel, white gridlines, no spines, no tick
marks, a shared legend under the figure, and the Color Universal Design palette.
Nothing is styled inline: scripts call apply_style() once and import the names
below.

    from _style import apply_style, PALETTE, MARKERS, LW, S_LINE, minor_grid, shared_legend
"""
import matplotlib
import seaborn as sns

LW = 2.2        # line width
S_LINE = 34     # scatter marker size for the dots on a line

# Colorblind-safe (Color Universal Design). Keyed by series name so a method
# keeps its colour in every figure; never pick a colour per plot.
PALETTE = {
    "adamw":         "#d55e00",   # vermilion, the baseline everything is read against
    "lion":          "#0173b2",   # blue
    "muon":          "#029e73",   # green
    "soap":          "#de8f05",   # orange
    "schedule_free": "#cc78bc",   # pink-purple
    "mars":          "#ca9161",   # tan
    "mars_m":        "#56b4e9",   # light blue
    "reference":     "#949494",   # gray, reserved for references and baselines
}

# One marker shape per method, fixed alongside the colour.
MARKERS = {
    "adamw": "o", "lion": "s", "muon": "^", "soap": "D",
    "schedule_free": "v", "mars": "P", "mars_m": "X",
}


def apply_style():
    sns.set_theme(style="darkgrid", context="notebook", rc={
        "axes.facecolor":   "#EAEAF2",   # light blue-gray panel
        "figure.facecolor": "white",
        "axes.edgecolor":   "white",     # no visible spines
        "axes.linewidth":   0.0,
        "grid.color":       "white",
        "grid.linewidth":   1.0,
        "axes.grid":        True,
        "xtick.bottom": False, "ytick.left": False,   # no tick marks
        "xtick.color": "#4A4A4A", "ytick.color": "#4A4A4A",
        "axes.labelcolor": "#2A2A2A", "text.color": "#2A2A2A",
        "legend.frameon":   False,
        "lines.linewidth":  LW,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42,        # editable text in the PDF
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        color=list(PALETTE.values()))


def alpha_colors(n, cmap="viridis"):
    """Sequential colours for an ordered sweep (e.g. values of one axis)."""
    cm = matplotlib.colormaps[cmap]
    return [cm(i / max(n - 1, 1)) for i in range(n)]


def minor_grid(ax, axis="both"):
    """Faint white minor gridlines; essential on log axes."""
    ax.minorticks_on()
    ax.grid(True, which="minor", axis=axis, color="white", linewidth=0.5, alpha=0.6)
    ax.grid(True, which="major", axis="both", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)


def shared_legend(fig, ax, ncols=None):
    """One legend for the whole figure, below it, never inside a panel."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncols=ncols or len(handles), frameon=False)
