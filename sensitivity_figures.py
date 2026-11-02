"""Figures for the search-budget sensitivity -> figures/sensitivity/.

  fig_budget_effect        paired change 40 -> 80 generations per dataset and
                           level: training fitness beside test AUC, one marker
                           per method, draw SD as the grey band
  fig_budget_contrasts     llm minus selector (median over the hard tier)
                           under both budgets, one dumbbell per contrast
  fig_budget_ranks         five-method mean ranks on the hard tier under both
                           budgets with the Nemenyi CD

Run after analyse_budget_sensitivity.py. Style follows selector_figures.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, tabular  # noqa: E402
from selector_figures import STYLE, SHOT_LABELS  # noqa: E402

ANALYSIS = config.SENSITIVITY_DIR / "analysis"
FIGDIR = config.FIGURES_DIR / "sensitivity"
SHOTS = list(config.SENSITIVITY_SHOTS)
CONTROLS = [c for c in config.FEWSHOT_CONTROLS if c != "random"]
SELECTORS = ["corr", "mi", "pca", "rf"]
PRIMARY = list(config.FEWSHOT_PRIMARY_CONTROLS)
HARD = [k for k, t in config.SEARCH_TIER.items() if t == "hard"]
LABEL = {k: tabular.FEATLLM_DATASETS[k].label for k in HARD}


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/sensitivity/{name}.png")


def fig_budget_effect():
    cells = pd.read_csv(ANALYSIS / "paired_cells.csv", dtype={"shot": str})
    per = (cells.groupby(["dataset", "shot", "control"])
                .agg(d_train=("d_train", "mean"), d_test=("d_test", "mean")).reset_index())
    sd = cells.groupby(["dataset", "shot"]).test_roc_auc_40.std().reset_index(name="sd")
    order = sorted(HARD, key=lambda k: LABEL[k])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for ax, col, title in zip(axes, ("d_train", "d_test"),
                              ("(a) Training ROC-AUC on the k labelled rows", "(b) Test ROC-AUC")):
        for i, ds in enumerate(order):
            for j, shot in enumerate(SHOTS):
                y = i + (j - 1) * 0.27
                if col == "d_test":
                    s = float(sd[(sd.dataset == ds) & (sd.shot == shot)].sd.iloc[0]) * 100
                    ax.plot([-s, s], [y, y], color="#e3e1dc", lw=5, solid_capstyle="butt", zorder=1)
                sub = per[(per.dataset == ds) & (per.shot == shot)]
                for _, r in sub.iterrows():
                    c, m, _, _, _ = STYLE[r.control]
                    ax.scatter(r[col] * 100, y, color=c, marker=m, s=34, zorder=3,
                               edgecolor="white", linewidth=0.5)
                ax.text(-0.02, y, SHOT_LABELS[shot], transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7.6, color="#8a8780")
        ax.axvline(0, color="#8a8780", lw=0.9, zorder=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL[k] for k in order])
        ax.tick_params(axis="y", pad=22)
        ax.set_xlabel("Change, 80 minus 40 generations (AUC points)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylim(len(order) - 0.5, -0.5)
    handles = [plt.Line2D([], [], color=STYLE[c][0], marker=STYLE[c][1], ls="", ms=6,
                          label=STYLE[c][4]) for c in CONTROLS]
    handles.append(plt.Line2D([], [], color="#e3e1dc", lw=5, label="\u00b1 SD over draws at 40 generations"))
    fig.legend(handles=handles, fontsize=8.6, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig_budget_effect")


def fig_budget_contrasts():
    c = pd.read_csv(ANALYSIS / "confirmatory_hard_tier_both_budgets.csv", dtype={"shot": str})
    fig, axes = plt.subplots(1, len(SHOTS), figsize=(11, 3.6), sharey=True, sharex=True)
    for ax, shot in zip(axes, SHOTS):
        sub = c[c.shot == shot]
        for i, sel in enumerate(SELECTORS):
            a = float(sub[(sub.contrast == f"llm vs {sel}") & (sub.generations == 40)].median_diff_pts.iloc[0])
            b = float(sub[(sub.contrast == f"llm vs {sel}") & (sub.generations == 80)].median_diff_pts.iloc[0])
            col = STYLE[sel][0]
            ax.plot([a, b], [i, i], color=col, lw=2, alpha=0.5, zorder=1)
            ax.scatter(a, i, color="white", edgecolor=col, s=52, zorder=3, linewidth=1.6)
            ax.scatter(b, i, color=col, s=52, zorder=3, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="#8a8780", lw=0.9)
        ax.set_yticks(range(len(SELECTORS)))
        ax.set_yticklabels([f"LLM prior \u2212 {STYLE[s][4]}" for s in SELECTORS], fontsize=9.5)
        ax.set_title(f"k = {SHOT_LABELS[shot]}", fontsize=11)
        ax.invert_yaxis()
    axes[1].set_xlabel("Median paired difference over the hard tier (AUC points)")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#2b2a27", ms=7,
                          label="40 generations"),
               plt.Line2D([], [], marker="o", ls="", color="#2b2a27", ms=7,
                          label="80 generations")]
    axes[-1].legend(handles=handles, fontsize=8.6, loc="lower right")
    save(fig, "fig_budget_contrasts")


def fig_budget_ranks():
    o = pd.read_csv(ANALYSIS / "friedman_hard_tier_both_budgets.csv", dtype={"shot": str})
    fig, axes = plt.subplots(1, len(SHOTS), figsize=(11, 3.4), sharey=True)
    for ax, shot in zip(axes, SHOTS):
        sub = o[o.shot == shot].set_index("generations")
        for c in PRIMARY:
            col, m, _, _, lab = STYLE[c]
            ax.plot([0, 1], [sub.loc[40, f"rank_{c}"], sub.loc[80, f"rank_{c}"]],
                    color=col, marker=m, lw=1.8, ms=6, label=lab)
        cd = float(sub.loc[40, "nemenyi_cd"])
        ax.plot([1.18, 1.18], [1, 1 + cd], color="#5a5852", lw=2, solid_capstyle="butt")
        ax.text(1.22, 1 + cd / 2, f"CD = {cd:.2f}", fontsize=8, va="center", color="#5a5852")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["40 generations", "80 generations"], fontsize=9)
        ax.set_xlim(-0.3, 1.6)
        ax.set_title(f"k = {SHOT_LABELS[shot]}   (Spearman \u03c1 = {float(sub.loc[40, 'rank_spearman_40_vs_80']):.2f})",
                     fontsize=10.5)
        ax.invert_yaxis()
    axes[0].set_ylabel("Mean Friedman rank over 7 datasets (1 = best)")
    axes[-1].legend(fontsize=8.6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig_budget_ranks")


def main() -> None:
    fig_budget_effect()
    fig_budget_contrasts()
    fig_budget_ranks()


if __name__ == "__main__":
    main()
