"""Fitness convergence figures from the harvested logbooks -> figures/.

Median best-so-far fitness (1 - AUC on the labelled rows) per generation,
IQR band over the ten draws, log fitness axis. Main figure: the five
selection methods; the _all variant adds the all-features baseline.

Usage: python convergence_figure.py heart --shot 64
Needs results/convergence/<dataset>__shot<shot>.csv (harvest_convergence.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# DejaVu has no semibold weight; the fallback is fine, the warning is noise
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, tabular                                # noqa: E402
from harvest_convergence import out_path                           # noqa: E402
from selector_figures import CONTROLS, PRIMARY_CONTROLS, STYLE, save  # noqa: E402


def fig_convergence(dataset: str, shot: str, controls: list[str],
                    title: str, name: str) -> None:
    df = pd.read_csv(out_path(dataset, shot), dtype={"shot": str})
    # a run at fitness 0 has perfect training AUC on the k rows; once a method's
    # median floors there the log axis has nothing to show, so fall back to linear
    medians = df.groupby(["control", "gen"]).best_so_far.median()
    saturated = bool((medians.loc[controls] <= 0).any())

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    final_median = {}
    for i, control in enumerate(controls):
        g = df[df.control == control].groupby("gen").best_so_far
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        c, m, ls, lw, label = STYLE[control]
        ax.fill_between(med.index, q1, q3, color=c, alpha=0.14, linewidth=0)
        # marker phases staggered per method so coinciding lines stay tellable
        ax.plot(med.index, med, color=c, marker=m, linestyle=ls, linewidth=lw,
                markersize=5.5, markevery=(1 + i, 5), markeredgecolor="white",
                markeredgewidth=0.8, label=label,
                zorder=3 if control == "llm" else 2)
        final_median[control] = float(med.iloc[-1])

    if saturated:
        ax.set_ylim(bottom=0)
        ax.text(0.02, 0.03, "linear axis: fitness reaches 0 (perfect training AUC)",
                transform=ax.transAxes, fontsize=9, color="#5a5852")
    else:
        ax.set_yscale("log")
    ax.xaxis.grid(False)
    ax.margins(x=0.02)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far fitness, 1 \u2212 training ROC-AUC (median, IQR)")
    ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=10.5, ha="left", va="bottom")

    # legend in the order the lines end in, best (lowest fitness) first
    order = sorted(controls, key=final_median.get)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[STYLE[c][4]] for c in order],
              [STYLE[c][4] for c in order], ncol=2, fontsize=9.5,
              loc="upper right")
    save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=tabular.keys(task_type="classification"))
    ap.add_argument("--shot", default="64", choices=[*map(str, config.FEWSHOT_SHOTS), "full"])
    args = ap.parse_args()
    d, s = args.dataset, args.shot
    label = f"{tabular.DATASETS[d].label}, k = {s}"
    fig_convergence(d, s, PRIMARY_CONTROLS, label, f"fig_convergence_{d}_shot_{s}")
    fig_convergence(d, s, CONTROLS, label, f"fig_convergence_{d}_shot_{s}_all")


if __name__ == "__main__":
    main()
