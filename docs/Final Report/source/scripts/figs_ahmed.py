"""Figures from the explanation-layer results (values from data/processed/track_c_v3 and the retrieval design experiment)."""
import os

from figs_common import OUT, S1, S2, S3, S4, S5, INK, MUTED, hbars, plt

os.makedirs(OUT, exist_ok=True)

# Track C: citation correct for the true fault (n = 30)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
labels = ["C1: label + SHAP", "C2: + past incidents", "C3: + alignment row + evidence",
          "No LLM: always TS 38.141-1", "No LLM: cite predicted row"]
vals = [5, 5, 21, 12, 21]
hbars(ax, labels, vals, [S1, S1, S1, S2, S2], lambda v: f"{v}/30", 30, refs=[(21, "70 % threshold")])
ax.set_xlabel("Windows whose citation is valid for the true fault (out of 30)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)],
          labels=["LLM condition", "Rule without an LLM"], loc="upper center", bbox_to_anchor=(0.5, -0.28),
          ncol=2, frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, "fig_trackc_citation.png"))
plt.close(fig)

# C3 outcome per window (stacked)
fig, ax = plt.subplots(figsize=(6.2, 1.25))
parts = [(19, "Classified correctly, citation correct", S1),
         (2, "Misclassified, same specification", S3),
         (3, "Misclassified, wrong row", S4),
         (6, "Missed by the detector", S5)]
left = 0
for n, label, color in parts:
    ax.barh([0], [n], left=left, color=color, height=0.6, edgecolor="white", linewidth=1.5, label=f"{label} ({n})")
    ax.text(left + n / 2, 0, str(n), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    left += n
ax.set_xlim(0, 30)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Track C windows under C3 (n = 30)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.7), ncol=2, frameon=False, fontsize=7.5)
fig.savefig(os.path.join(OUT, "fig_trackc_outcomes.png"))
plt.close(fig)

# Retrieval design comparison, precision@1 against the true class
fig, ax = plt.subplots(figsize=(6.2, 2.3))
labels = ["Tickets, label in query (prototype)", "Tickets, label removed",
          "Narratives, label-free (deployed)", "Narratives, paragraphs averaged"]
vals = [0.707, 0.102, 0.451, 0.049]
hbars(ax, labels, vals, [S1] * 4, lambda v: f"{v:.3f}", 1.0,
      refs=[(0.353, "majority 0.353"), (0.748, "classifier 0.748")])
ax.set_xlabel("Precision@1 against the true fault type (266 detected test faults)")
fig.savefig(os.path.join(OUT, "fig_retrieval_design.png"))
plt.close(fig)
print("ok")
