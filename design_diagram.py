"""Render design_diagram.png: the flow of the selector comparison.

Full-information selection design: the data-driven methods score the whole
training partition, the LLM prior stays data-free, one generic GE learner
trains on the k labelled rows, test ROC-AUC compared across the five
methods. Regenerate with: python design_diagram.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#2b2a27"
GREY = "#8a8a8a"
NAVY = "#1e3a5c"
GREEN, GREEN_EDGE, GREEN_FILL = "#1b6b45", "#7fae92", "#e9f2ec"
AMBER, AMBER_EDGE, AMBER_FILL = "#b06000", "#d9a441", "#fdf3e3"
BLUE_EDGE, BLUE_FILL = "#8ba3c7", "#e9eef7"
PILL_EDGE, PILL_FILL = "#b9c0cc", "#eef0f5"
YES, NO = "#1b7a4b", "#b2432f"

W, H = 13.6, 9.0
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, fill, edge, lw=1.4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=fill, edgecolor=edge, linewidth=lw))


def arrow(x0, y0, x1, y1, color="#44546a"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3,
                                 color=color, shrinkA=0, shrinkB=1))


def line(x0, y0, x1, y1, color="#44546a"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.3, solid_capstyle="round")


def text(x, y, s, size, color=INK, weight="normal", style="normal", family=None):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, fontstyle=style, family=family)


# title
text(W / 2, 8.62, "Where should the feature subset for a GE learner come from?",
     16.5, NAVY, "bold")
text(W / 2, 8.28, "Data-driven selection against a data-free prior and its semantic floor — "
     "the learner always sees only the k labelled rows · 24 datasets, classification and regression", 10.5, GREY)

# draw box
box(W / 2, 7.62, 5.2, 0.78, BLUE_FILL, BLUE_EDGE)
text(W / 2, 7.80, "FeatLLM split, one draw per seed (quartile-stratified for regression)", 11, NAVY, "bold")
text(W / 2, 7.58, "selectors score the whole 80% training partition · "
     "the learner gets k ∈ {4, 8, 16, 32, 64} rows · fixed budget K", 9)
text(W / 2, 7.38, "the 20% test partition is held out and never seen until scoring",
     8, GREY)

# five methods
CARDS = [
    ("corr", "correlation with label",
     ["binary: |point-biserial r|", "multiclass: correlation", "ratio η"],
     "labels · full train rows", YES, "linear supervised signal", False),
    ("mi", "mutual information",
     ["I(feature ; label)", "kNN estimator"],
     "labels · full train rows", YES, "captures non-linear dependence", False),
    ("pca", "PCA loadings",
     ["|loadings| weighted by", "explained variance,", "summed over components"],
     "no labels · full train rows", AMBER, "unsupervised — variance only",
     False),
    ("rf", "random-forest importance",
     ["impurity importance", "100-tree forest,", "fixed seed"],
     "labels · full train rows", YES, "non-linear, interaction-aware", False),
    ("llm", "knowledge prior",
     ["LLM reasons over", "feature names + descriptions", "+ task text"],
     "no labels · no rows", NO, "semantic / prior knowledge only", True),
    ("random", "semantic floor",
     ["size-K subset drawn", "at random per draw;", "= prior with anonymised names"],
     "no labels · no rows", NO, "cardinality-matched floor", True),
]
CW, CH, CY = 2.06, 1.86, 5.72
xs = [1.30 + i * 2.22 for i in range(6)]
for x, (key, sub, body, sees, sees_c, foot, is_llm) in zip(xs, CARDS):
    fill, edge, head = ((AMBER_FILL, AMBER_EDGE, AMBER) if is_llm else
                        (GREEN_FILL, GREEN_EDGE, GREEN))
    box(x, CY, CW, CH, fill, edge)
    text(x, CY + 0.70, key, 12, head, "bold")
    text(x, CY + 0.49, sub, 9, head, "bold")
    line(x - 0.75, CY + 0.36, x + 0.75, CY + 0.36, edge)
    for i, s in enumerate(body):
        text(x, CY + 0.17 - 0.20 * i, s, 8.4)
    text(x, CY - 0.49, sees, 8.6, sees_c, "bold")
    text(x, CY - 0.72, foot, 7.8, GREY)
    arrow(x, 7.20, x, CY + CH / 2 + 0.03)
line(xs[0], 7.20, xs[5], 7.20)
line(W / 2, 7.23, W / 2, 7.20)

# connectors first: one collector at pill mid-height, hidden where the pill
# sits on top of it; every control, llm included, ends at the same object
for x in xs:
    line(x, CY - CH / 2 - 0.02, x, 3.86)
line(xs[0], 3.86, xs[5], 3.86)

# group rails (white bbox so the captions sit over the connectors)
line(xs[0] - CW / 2, 4.56, xs[3] + CW / 2, 4.56, GREEN_EDGE)
ax.text((xs[0] + xs[3]) / 2, 4.38, "DATA-DRIVEN — each scores the whole "
        "training partition, never the test rows", ha="center", va="center",
        fontsize=9, color=GREEN, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
line(xs[4] - CW / 2, 4.56, xs[5] + CW / 2, 4.56, AMBER_EDGE)
ax.text((xs[4] + xs[5]) / 2, 4.38, "DATA-FREE", ha="center", va="center", fontsize=9,
        color=AMBER, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# aggregation pill, drawn over the collector
box(W / 2, 3.86, 6.4, 0.62, PILL_FILL, PILL_EDGE)
text(W / 2, 3.98, "score per column → group-max to features → rank "
     "→ top-K", 10, INK, "bold")
text(W / 2, 3.74, "every method ends at the same object: an unordered subset "
     "of K features", 8.2, GREY)

# inject box
box(W / 2, 2.84, 7.0, 0.80, BLUE_FILL, BLUE_EDGE)
text(W / 2, 3.03, "Inject the K features into the grammar — terminal rule only",
     11, NAVY, "bold")
text(W / 2, 2.81, "<v> ::= x[:,i₁] | … | x[:,i_K]", 9.5,
     family="monospace")
text(W / 2, 2.59, "every other rule — operators, recursion, coefficients "
     "— is identical across all methods", 8, GREY)
arrow(W / 2, 3.55, W / 2, 3.27)

# learner box
box(W / 2, 1.78, 7.0, 0.86, GREEN_FILL, "#4e9468")
text(W / 2, 2.02, "One identical GE learner (GRAPE)", 11.5, GREEN, "bold")
text(W / 2, 1.79, "same search, budget, population, generations, hall of fame "
     "for every method", 8.8)
text(W / 2, 1.60, "evolved on the k labelled rows only · fitness = 1 − training ROC-AUC "
     "(classification) or 1 − R² after linear rescaling (regression)", 8.8)
text(W / 2, 1.42, "the selection method is the only thing that changes", 8,
     GREY, style="italic")
arrow(W / 2, 2.42, W / 2, 2.24, "#2f7a4f")

# compare box
box(W / 2, 0.64, 6.6, 0.78, AMBER_FILL, AMBER_EDGE)
text(W / 2, 0.86, "Compare test ROC-AUC (Spearman ρ for regression) across the methods", 11.5,
     AMBER, "bold")
text(W / 2, 0.64, "does the data-free LLM prior match or beat corr / mi / pca / rf",
     8.6)
text(W / 2, 0.47, "and the random floor — and does the answer depend on k, task and named vs anonymous features?",
     8.6)
text(W / 2, 0.30, "blocked on dataset · Wilcoxon signed-rank · Holm "
     "· Cliff's δ · Friedman + Nemenyi", 7.8, GREY)
arrow(W / 2, 1.33, W / 2, 1.06)

fig.savefig("design_diagram.png", dpi=200, facecolor="white")
print("design_diagram.png")
