"""Team-level figures for the final report. Every value is quoted from a source report (see FIGURES.md)."""
import os
import shutil

import numpy as np
from matplotlib.patches import FancyBboxPatch

from figs_common import INK, MUTED, OUT, RULE, S1, S2, S3, S4, S5, plt

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(WORK, "figures_src", "phase1")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(b.get_height()),
                ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")


# Phase 1 figures, copied as published (P1 Fig. 2 and Fig. 3)
shutil.copy(os.path.join(P1, "p12_fig2_class_balance.png"), os.path.join(OUT, "fig_class_balance.png"))
shutil.copy(os.path.join(P1, "p13_fig3_fault_kpi_heatmap.png"), os.path.join(OUT, "fig_fault_kpi_heatmap.png"))

# Model 1 on the five held-out sessions (FR §1, §4.1)
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2))
panels = [("False alarms", [1105, 116], lambda v: f"{v:,.0f}"),
          ("Precision", [0.196, 0.70], lambda v: f"{v:.2f}"),
          ("F1", [0.32, 0.78], lambda v: f"{v:.2f}")]
for ax, (title, vals, fmt) in zip(axes, panels):
    bars = ax.bar(["First\ndetector", "Fixed\ndetector"], vals, color=[MUTED, S1], width=0.6)
    label_bars(ax, bars, fmt)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.2)
fig.tight_layout()
save(fig, "fig_model1_heldout.png")

# Model 2 final candidates (FR §6.2)
cands = ["v3 recipe\n(CatBoost+v3+B+FFT)", "+ trend features\n(final)", "QUANT,\nreal only",
         "CatBoost+catch22,\nreal only", "Soft vote\nCatBoost & QUANT", "Soft vote\nof all three"]
cv = [0.374, 0.389, 0.426, 0.393, 0.453, 0.446]
test = [0.537, 0.591, 0.463, 0.608, 0.516, 0.574]
x = np.arange(len(cands))
fig, ax = plt.subplots(figsize=(6.6, 2.8))
b1 = ax.bar(x - 0.19, cv, 0.38, color=S1, label="CV macro-F1 (25 session folds)")
b2 = ax.bar(x + 0.19, test, 0.38, color=S2, label="Held-out macro-F1 (300 faults)")
label_bars(ax, b1, lambda v: f"{v:.3f}")
label_bars(ax, b2, lambda v: f"{v:.3f}")
ax.set_xticks(x, cands, fontsize=7)
ax.set_ylim(0, 0.72)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
save(fig, "fig_model2_candidates.png")

# Cascade false alarms and the CCI-Mild catch-all (FR §7, §8)
fig, ax = plt.subplots(figsize=(5.2, 2.4))
groups = ["First cascade\n(12 Sep 2026)", "Final cascade"]
total = [1105, 116]
catch = [908, 75]
x = np.arange(2)
bt = ax.bar(x - 0.18, total, 0.36, color=MUTED, label="False alarms")
bc = ax.bar(x + 0.18, catch, 0.36, color=S4, label="of which labelled CCI-Mild")
label_bars(ax, bt, lambda v: f"{v:,.0f}")
for b, pct in zip(bc, ["≈908 (82.2 %)", "75 (64.7 %)"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), pct, ha="center", va="bottom", fontsize=8,
            color=INK, fontweight="bold")
ax.set_xticks(x, groups)
ax.set_ylim(0, 1300)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, fontsize=7.5)
save(fig, "fig_false_alarms.png")

# Track C: first ablation vs final run (P1 Table 16; AS Table 5)
conds = ["C1: label only /\nlabel + SHAP", "C2: + retrieved\ntickets / incidents", "C3: + alignment\nrow + RCA evidence"]
first = [2 / 30 * 100, 3 / 30 * 100, 27 / 30 * 100]
final = [5 / 30 * 100, 5 / 30 * 100, 21 / 30 * 100]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(6.2, 2.7))
bf = ax.bar(x - 0.19, first, 0.38, color=MUTED, label="First ablation (first prototype): valid against the predicted fault's row")
bn = ax.bar(x + 0.19, final, 0.38, color=S1, label="Final run: sampled by true fault, correct vs true fault")
for b, t in zip(bf, ["2/30", "3/30", "27/30"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), t, ha="center", va="bottom", fontsize=8, fontweight="bold")
for b, t in zip(bn, ["5/30", "5/30", "21/30"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), t, ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(70, color=MUTED, linestyle="--", linewidth=1)
ax.text(-0.55, 71, "O4: 70 %", color=MUTED, fontsize=7.5, ha="left", va="bottom")
ax.set_xticks(x, conds, fontsize=8)
ax.set_ylim(0, 110)
ax.set_ylabel("Citations counted as valid (%)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "fig_trackc_evolution.png")

# Track B: G-Eval dimension means (RK Table 5.Y.1)
dims = ["Citation validity", "Fault specificity", "Actionability", "Causal soundness", "Composite"]
means = [3.63, 3.63, 3.50, 3.60, 3.59]
fig, ax = plt.subplots(figsize=(5.6, 2.2))
y = range(len(dims))[::-1]
ax.barh(list(y), means, color=[S1] * 4 + [S3], height=0.55)
ax.set_yticks(list(y), dims)
ax.set_xlim(1, 5)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
for yi, v in zip(y, means):
    ax.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("Mean score on the 1-5 rubric (n = 30 windows)")
save(fig, "fig_trackb_dimensions.png")

# Track D: TeleQnA (RK Tables 5.X.2 and 5.X.3)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
cats = ["Standards\nspecifications", "Standards\noverview", "Lexicon", "Overall"]
base = [67.0, 75.0, 87.0, 76.3]
rag = [61.0, 53.0, 41.5, 51.8]
x = np.arange(4)
bb = a1.bar(x - 0.19, base, 0.38, color=S1, label="Baseline LLM (no context)")
br = a1.bar(x + 0.19, rag, 0.38, color=S2, label="KPI-RAG, no confidence gate")
label_bars(a1, bb, lambda v: f"{v:.1f}")
label_bars(a1, br, lambda v: f"{v:.1f}")
a1.set_xticks(x, cats, fontsize=7)
a1.set_ylim(0, 128)
a1.set_yticks([0, 20, 40, 60, 80, 100])
a1.set_ylabel("Accuracy (%)")
a1.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0, 1.02))
a1.set_title("Accuracy by category (n = 600)", fontsize=8.5)
tau = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
acc = [57.3, 62.2, 66.7, 69.7, 73.8, 75.2, 76.0, 76.3, 76.3]
a2.plot(tau, acc, color=S2, marker="o", markersize=4, linewidth=2, label="Gated KPI-RAG")
a2.axhline(76.3, color=S1, linestyle="--", linewidth=1.2, label="Baseline 76.3 %")
a2.set_xlabel("Confidence threshold τ")
a2.set_ylim(50, 80)
a2.legend(frameon=False, fontsize=6.8, loc="lower right")
a2.set_title("Accuracy with a confidence gate", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_trackd.png")


# Architecture of the final system (P1 Fig. 1 updated with FR §8, RK, AS §4)
def box(ax, x, y, w, h, text, fc, ec=RULE, color=INK, bold=False, size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))


fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
rows = [  # (label, y, boxes)
    ("Data", 0.86, [("TelecomTS\n32,000 windows · 18 channels\n1,235 anomalous · 11 fault types", 0.17, 0.30),
                    ("Session split\n33 sessions (300 s gap)\n28 train / 5 test", 0.51, 0.20),
                    ("Synthetic faults (v3)\ntraining sessions only", 0.75, 0.22)]),
    ("Layer 1", 0.67, [("Model 1 features\nConfig A · 64 statistics", 0.17, 0.30),
                       ("Model 2 features\nConfig B + FFT + trend · 496", 0.51, 0.46)]),
    ("Layer 2", 0.48, [("Model 1\nCatBoost detector (t = 0.5)", 0.17, 0.23),
                       ("Model 2\nCatBoost 11-class classifier", 0.44, 0.24),
                       ("SHAP top-3\n→ Layer 2 JSON", 0.72, 0.25)]),
    ("Layer 3", 0.29, [("Retrieval: ChromaDB\n935 training narratives\nfiltered by predicted type", 0.17, 0.27),
                       ("Alignment table v1.2\n+ per-window\nRCA evidence", 0.475, 0.20),
                       ("Gemini 3.5 Flash Lite\nJSON explanation\n→ citation validator", 0.70, 0.27)]),
    ("Output", 0.10, [("Streamlit dashboard · six panels", 0.17, 0.42),
                      ("Evaluation · Tracks A, B, C, D", 0.63, 0.34)]),
]
H = 0.12
for label, y, boxes in rows:
    ax.text(0.01, y + H / 2, label, ha="left", va="center", fontsize=8, color=S1, fontweight="bold")
    for i, (text, x, w) in enumerate(boxes):
        key = (label, i) in {("Layer 3", 2), ("Layer 3", 1)}
        box(ax, x, y, w, H, text, fc="#dce6f7" if key else "#f8f7f5", ec=S1 if key else RULE)
for (_, y0, b0), (_, y1, _) in zip(rows[:-1], rows[1:]):
    arrow(ax, 0.55, y0, 0.55, y1 + H)
arrow(ax, 0.40, 0.48 + H / 2, 0.44, 0.48 + H / 2)
arrow(ax, 0.68, 0.48 + H / 2, 0.72, 0.48 + H / 2)
arrow(ax, 0.445, 0.29 + H / 2, 0.475, 0.29 + H / 2)
arrow(ax, 0.675, 0.29 + H / 2, 0.70, 0.29 + H / 2)
ax.text(0.17, 0.455, "Normal → stop (no explanation)", fontsize=6.5, color=MUTED)
save(fig, "fig_architecture.png")
print("ok")
