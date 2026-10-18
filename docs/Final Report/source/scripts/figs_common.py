"""Shared matplotlib style for the final-report figures (palette validated with the dataviz skill)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, RULE = "#1c1c1e", "#5c5c66", "#d8d6d0"
OUT = r"C:\00-Shobaki\Data Science\Queen's MCs\Graduation Project\project\docs\Final Report\figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def hbars(ax, labels, values, colors, fmt, xmax, refs=()):
    """Horizontal bars, top to bottom, with direct value labels and optional dashed reference lines."""
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.55)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt(v), va="center", color=INK, fontsize=8.5, fontweight="bold")
    for x, label in refs:
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
        ax.text(x, len(labels) - 0.4, label, ha="center", va="bottom", color=MUTED, fontsize=7.5)
