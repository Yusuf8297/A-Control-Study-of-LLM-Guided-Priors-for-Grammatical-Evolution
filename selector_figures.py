"""Comparison figures -> figures/. One chart per file, PNG and PDF.

Colours are Okabe-Ito (colour-blind safe); the all-features baseline is a
neutral grey with its own dash pattern, the random subset a lighter grey,
and every method has its own marker, so identity never rests on colour
alone. Figures carry no titles; the captions in figures/CAPTIONS.md
describe each one.

Per analysis family (figures/<family>/): pooled score against k, mean rank
against k, critical-difference diagrams, paired per-dataset differences
against each selector and against the random subset (the semantic floor),
the confirmatory and all-pairs significance maps, stability and overlap.
Main figures compare the five selection methods at the five few-shot
levels; variants with the _all suffix add the all-features baseline, the
random subset and the capped full-data level.

Across families (figures/): compute time, the semantic-anchor figure (LLM
prior minus the best classical selector per dataset, by stratum), the
cross-task rank figure and one score-against-k figure per dataset under
figures/per_dataset/.
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

ANALYSIS_ROOT = config.RESULTS_DIR / "analysis"
FIGROOT = config.FIGURES_DIR
SHOTS = list(config.ALL_SHOTS)
SHOT_LABELS = {"4": "4", "8": "8", "16": "16", "32": "32", "64": "64", "full": "full*"}
CONTROLS = list(config.FEWSHOT_CONTROLS)
SELECTORS = ["corr", "mi", "pca", "rf"]
FLOOR = config.SEMANTIC_FLOOR_CONTROL
PRIMARY_CONTROLS = list(config.FEWSHOT_PRIMARY_CONTROLS)
PRIMARY_SHOTS = list(config.FEWSHOT_PRIMARY_SHOTS)
TIME_REF = config.TIME_REFERENCE_CONTROL

STYLE = {   # colour, marker, linestyle, linewidth, legend label
    "none":   ("#5b6676", "o", (0, (1, 1.2)), 1.6, "All features"),
    "random": ("#a3a9b3", "X", (0, (4, 2)),   1.6, "Random subset"),
    "corr":   ("#009E73", "s", "-",           1.8, "Correlation"),
    "mi":     ("#0072B2", "^", "-",           1.8, "Mutual information"),
    "pca":    ("#CC79A7", "D", "-",           1.8, "PCA loadings"),
    "rf":     ("#D55E00", "v", "-",           1.8, "RF importance"),
    "llm":    ("#E69F00", "o", "-",           2.4, "LLM prior"),
}
SHORT = {"none": "All", "random": "Rand", "corr": "Corr", "mi": "MI", "pca": "PCA", "rf": "RF", "llm": "LLM"}
# grey-to-ink ramp for the five labelling budgets in the difference plots
SHOT_SHADES = {"4": "#cfccc6", "8": "#aba7a0", "16": "#87837b",
               "32": "#5a5852", "64": "#2b2a27"}
STRATUM_COLOUR = {"named": "#E69F00", "anonymous": "#5b6676"}
INK, MUTED, GRID = "#2b2a27", "#5a5852", "#e3e1dc"
FULL_NOTE = "*full: training rows capped at 1000"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Inter", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.edgecolor": "#8a8780",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": True, "legend.facecolor": "white", "legend.framealpha": 0.88,
    "legend.edgecolor": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


class Family:
    """Paths and labels of one analysis family."""

    def __init__(self, name):
        self.name = name
        self.analysis = ANALYSIS_ROOT / name
        self.figdir = FIGROOT / name
        info = pd.read_csv(self.analysis / "family.csv").iloc[0]
        self.n = int(info.n_datasets)
        self.task = info.task
        self.metric = config.METRIC_LABEL[self.task]
        self.points = "points" if self.task == "classification" else "x 100"
        self.figdir.mkdir(parents=True, exist_ok=True)

    def read(self, name):
        return pd.read_csv(self.analysis / name, dtype={"shot": str})


def save(fig, path):
    path = Path(path) if isinstance(path, Path) else FIGROOT / str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(config.ROOT)}.png")


def panel_label(ax, letter, x=-0.08, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")


def legend_sorted(ax, frame, value, controls, shots, ascending=False, **kw):
    """Legend entries in the vertical order the lines end in."""
    def final(c):
        sub = frame[(frame.control == c) & (frame.shot == shots[-1])][value]
        return float(sub.iloc[0]) if len(sub) else -np.inf
    order = sorted(controls, key=final, reverse=not ascending)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[STYLE[c][4]] for c in order if STYLE[c][4] in by_label],
              [STYLE[c][4] for c in order if STYLE[c][4] in by_label], **kw)


def control_lines(ax, frame, value, controls=CONTROLS, shots=SHOTS):
    x = np.arange(len(shots))
    for control in controls:
        sub = frame[frame.control == control].set_index("shot").reindex(shots)
        c, m, ls, lw, label = STYLE[control]
        ax.plot(x, sub[value], color=c, marker=m, linestyle=ls, linewidth=lw,
                markersize=6.5, markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3 if control == "llm" else 2)
    ax.xaxis.grid(False)
    ax.margins(x=0.03)
    ax.set_xticks(x, [SHOT_LABELS[s] for s in shots])
    ax.set_xlabel("Labelled rows, k")
    if "full" in shots:
        ax.annotate(FULL_NOTE, xy=(1.0, -0.16), xycoords="axes fraction",
                    ha="right", fontsize=8.2, color=MUTED, annotation_clip=False)


def n_note(fn):
    """'N = 19' or 'N = 11-19 by level' from a Friedman table."""
    ns = sorted(set(fn.n_datasets))
    return f"N = {ns[0]}" if len(ns) == 1 else f"N = {ns[0]}-{ns[-1]} by level"


# ------------------------------------------------------------- per family

def fig_pooled(fam: Family, all_methods=False):
    pooled = fam.read("pooled_summary.csv")
    controls = CONTROLS if all_methods else PRIMARY_CONTROLS
    shots = [s for s in (SHOTS if all_methods else PRIMARY_SHOTS) if s in set(pooled.shot)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    control_lines(ax, pooled, "pooled_median", controls=controls, shots=shots)
    ax.set_ylabel(f"Median test {fam.metric} ({fam.n} datasets)")
    legend_sorted(ax, pooled, "pooled_median", controls, shots, ncol=2, fontsize=9.5,
                  loc="lower right")
    save(fig, fam.figdir / ("fig_pooled_score_vs_shot_all" if all_methods else "fig_pooled_score_vs_shot"))


def fig_ranks(fam: Family, design="primary"):
    controls = PRIMARY_CONTROLS if design == "primary" else CONTROLS
    fn = fam.read(f"friedman_nemenyi_{design}.csv")
    shots = [s for s in (PRIMARY_SHOTS if design == "primary" else SHOTS) if s in set(fn.shot)]
    long = fn.melt(id_vars=["shot", "nemenyi_cd"], value_vars=[f"rank_{a}" for a in controls],
                   var_name="control", value_name="rank")
    long["control"] = long.control.str.removeprefix("rank_")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    control_lines(ax, long, "rank", controls=controls, shots=shots)
    ax.invert_yaxis()
    cds = fn.nemenyi_cd
    cd_txt = f"CD = {cds.iloc[0]:.2f}" if cds.nunique() == 1 else f"CD = {cds.min():.2f}-{cds.max():.2f}"
    ax.text(0.02, 0.04, f"Nemenyi {cd_txt} ({len(controls)} methods, {n_note(fn)})",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_ylabel("Mean Friedman rank (1 = best)")
    legend_sorted(ax, long, "rank", controls, shots, ascending=True, fontsize=9.5,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, fam.figdir / ("fig_mean_rank_vs_shot" if design == "primary" else "fig_mean_rank_vs_shot_all"))


def fig_cd_diagram(fam: Family, shot, design="primary"):
    """Critical-difference diagram in the usual Demsar layout."""
    controls = PRIMARY_CONTROLS if design == "primary" else CONTROLS
    fn = fam.read(f"friedman_nemenyi_{design}.csv")
    rows = fn[fn.shot == shot]
    if rows.empty:
        raise ValueError(f"no omnibus at k = {shot}")
    row = rows.iloc[0]
    ranks = {a: row[f"rank_{a}"] for a in controls}
    cd = row.nemenyi_cd
    order = sorted(ranks, key=ranks.get)
    k = len(order)
    fig, ax = plt.subplots(figsize=(7.6, 2.8 + 0.2 * max(0, k - 5)))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-3.4 - 0.3 * max(0, k - 5), 2.1)
    ax.axis("off")
    ax.plot([lo, hi], [0, 0], color=INK, lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.12], color=INK, lw=1.2)
        ax.text(t, 0.3, str(t), ha="center", fontsize=9.5)
    ax.plot([lo, lo + cd], [1.15, 1.15], color=INK, lw=2.2)
    ax.text(lo + cd / 2, 1.4, f"CD = {cd:.2f}", ha="center", fontsize=9.5)
    level = f"k = {shot}" if shot != "full" else "full data (capped)"
    ax.text(hi + 0.3, 1.4, f"{level}, N = {int(row.n_datasets)}", ha="right", fontsize=9.5, color=MUTED)
    half = (k + 1) // 2
    for i, control in enumerate(order):
        left = i < half
        slot = i if left else i - half
        y = -0.65 - 0.62 * slot
        xend = lo - 0.25 if left else hi + 0.25
        c, _, _, _, label = STYLE[control]
        ax.plot([ranks[control], ranks[control]], [0, y], color="#8a8780", lw=0.9)
        ax.plot([ranks[control], xend], [y, y], color="#8a8780", lw=0.9)
        ax.text(xend + (-0.05 if left else 0.05), y,
                f"{label}  ({ranks[control]:.2f})", ha="right" if left else "left",
                va="center", fontsize=9.8, color=INK)
    ybar = -0.28
    drawn = []
    for control in order:
        group = [b for b in order if ranks[control] <= ranks[b] <= ranks[control] + cd]
        if len(group) > 1:
            span = (ranks[group[0]], ranks[group[-1]])
            if not any(s[0] <= span[0] and span[1] <= s[1] for s in drawn):
                ax.plot(span, [ybar, ybar], color=INK, lw=3.2, solid_capstyle="round")
                drawn.append(span)
                ybar -= 0.16
    save(fig, fam.figdir / (f"fig_cd_shot_{shot}" + ("" if design == "primary" else "_all")))


def fig_score_diff(fam: Family, other):
    """Paired per-dataset differences, LLM prior minus one method, at each
    few-shot level; dots moving towards zero show where the other method
    catches up with the prior."""
    per = fam.read("summary_per_dataset.csv")
    wide = per.pivot_table(index="dataset", columns=["shot", "control"], values="mean_score")
    shots = [s for s in PRIMARY_SHOTS if (s, "llm") in wide.columns]
    datasets = list(wide.index)
    mat = np.array([[100 * (wide.loc[d, (s, "llm")] - wide.loc[d, (s, other)])
                     for s in shots] for d in datasets])
    order = np.argsort(np.nanmean(mat, axis=1))
    mat, datasets = mat[order], [datasets[i] for i in order]
    y = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(7.0, 2.6 + 0.26 * len(datasets)))
    ax.grid(False)
    for i, yi in enumerate(y):
        row = mat[i][np.isfinite(mat[i])]
        if len(row):
            ax.plot([row.min(), row.max()], [yi, yi], color="#e9e7e2", lw=3.2,
                    solid_capstyle="round", zorder=1)
    ax.axvline(0, color="#8a8780", lw=1.1, ls=(0, (4, 2)), zorder=2)
    for j, s in enumerate(shots):
        ax.scatter(mat[:, j], y, s=52, color=SHOT_SHADES[s], edgecolor="white",
                   linewidth=0.9, zorder=3, label=f"k = {s}")
    ax.set_yticks(y, [tabular.DATASETS[d].label for d in datasets])
    ax.set_ylim(-0.7, len(datasets) - 0.3)
    span = max(4.0, np.nanmax(np.abs(mat)) * 1.12)
    ax.set_xlim(-span, span)
    label = STYLE[other][4]
    ax.set_xlabel(f"Test {fam.metric} difference, LLM prior − {label} ({fam.points})")
    ax.text(0.01, 1.01, f"← {label} higher", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.6, color=MUTED)
    ax.text(0.99, 1.01, "LLM prior higher →", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.6, color=MUTED)
    ax.legend(fontsize=8.6, loc="lower right", title="labelled rows", title_fontsize=8.6)
    save(fig, fam.figdir / f"fig_score_diff_llm_vs_{other}")


def fig_stability(fam: Family):
    st = fam.read("stability_pooled.csv")
    shots = [s for s in SHOTS if s in set(st.shot)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    control_lines(ax, st, "stability", controls=["corr", "mi", "pca", "rf", "llm", "random"], shots=shots)
    ax.set_ylabel(f"Mean pairwise Jaccard similarity ({config.FEWSHOT_DRAWS} draws)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=9.5, loc="lower right")
    save(fig, fam.figdir / "fig_stability_vs_shot")


def fig_overlap(fam: Family):
    ov = fam.read("overlap_with_llm_pooled.csv")
    shots = [s for s in SHOTS if s in set(ov.shot)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    control_lines(ax, ov, "jaccard_llm", controls=SELECTORS + [FLOOR], shots=shots)
    ax.set_ylabel("Mean Jaccard similarity with the LLM subset")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=9.5, loc="upper left")
    save(fig, fam.figdir / "fig_overlap_with_llm_vs_shot")


def _pair_lookup(frame, shot, a, b):
    m = frame[(frame.shot == shot) & (((frame.a == a) & (frame.b == b)) |
                                      ((frame.a == b) & (frame.b == a)))]
    return m.iloc[0] if len(m) else None


def fig_significance_confirmatory(fam: Family):
    """The confirmatory contrasts and the semantic floor on one page: LLM
    prior against each selector and against the random subset at every
    level. Tile fill marks a Holm-significant difference and its colour the
    direction; the number is the median paired difference with the
    Holm-adjusted p beneath."""
    conf = fam.read("confirmatory_wilcoxon.csv")
    floor = fam.read("semantic_floor_wilcoxon.csv")
    frame = pd.concat([conf, floor], ignore_index=True)
    shots = [s for s in SHOTS if s in set(frame.shot)]
    rows = SELECTORS + [FLOOR]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.grid(False)
    llm_c, sel_c = STYLE["llm"][0], STYLE["mi"][0]
    for j, s in enumerate(shots):
        for i, sel in enumerate(rows):
            m = frame[(frame.shot == s) & (frame.contrast == f"llm vs {sel}")]
            if m.empty:
                continue
            r = m.iloc[0]
            sig = bool(r.significant_05)
            direction = llm_c if r.median_diff_pts > 0 else sel_c
            face = direction if sig else "white"
            edge = direction if abs(r.median_diff_pts) > 0.05 else "#c9c6bf"
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, facecolor=face,
                                       edgecolor=edge, linewidth=1.4 if sig else 0.9,
                                       alpha=0.92 if sig else 1.0, zorder=2))
            txt = "white" if sig else INK
            ax.text(j, i + 0.1, f"{r.median_diff_pts:+.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if sig else "normal", color=txt, zorder=3)
            ax.text(j, i - 0.22, f"p = {r.holm_p:.3f}", ha="center", va="center",
                    fontsize=7.8, color=txt if sig else MUTED, zorder=3)
    ax.axhline(len(SELECTORS) - 0.5, color="#8a8780", lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(-0.6, len(shots) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xticks(range(len(shots)), [f"k = {SHOT_LABELS[s]}" if s != "full" else "full*" for s in shots])
    ax.set_yticks(range(len(rows)), [f"LLM prior vs {STYLE[a][4]}" for a in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=llm_c, edgecolor=llm_c),
               plt.Rectangle((0, 0), 1, 1, facecolor=sel_c, edgecolor=sel_c),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#c9c6bf")]
    ax.legend(handles, ["LLM prior higher, Holm p < 0.05", "Other method higher, Holm p < 0.05",
                        "Not significant"],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8.6, frameon=False)
    ax.annotate(f"{FULL_NOTE}; differences in {fam.metric} {fam.points}; the last row is the "
                "semantic floor, Holm across the few-shot levels",
                xy=(1.0, -0.27), xycoords="axes fraction", ha="right", fontsize=7.8, color=MUTED,
                annotation_clip=False)
    save(fig, fam.figdir / "fig_significance_confirmatory")


def fig_significance_allpairs(fam: Family, design="primary"):
    """Every pair, two post-hoc readings side by side per level. Row (a):
    all-pairs Wilcoxon over datasets with Holm across every pair in the
    level. Row (b): Nemenyi mean-rank gap against the critical difference."""
    controls = PRIMARY_CONTROLS if design == "primary" else CONTROLS
    wil = fam.read(f"exploratory_allpairs_wilcoxon_{design}.csv")
    nem = fam.read(f"nemenyi_pairwise_{design}.csv")
    shots = [s for s in (PRIMARY_SHOTS if design == "primary" else SHOTS) if s in set(wil.shot)]
    k = len(controls)
    fig, axes = plt.subplots(2, len(shots), figsize=(2.05 * len(shots) + 1.2, 2.1 * 2 + 1.4),
                             gridspec_kw={"wspace": 0.12, "hspace": 0.32}, squeeze=False)
    for col, s in enumerate(shots):
        for row, (frame, kind) in enumerate(((wil, "wilcoxon"), (nem, "nemenyi"))):
            ax = axes[row, col]
            ax.grid(False)
            ax.set_xlim(-0.5, k - 0.5)
            ax.set_ylim(k - 0.5, -0.5)
            ax.set_aspect("equal")
            for i, a in enumerate(controls):
                for j, b in enumerate(controls):
                    if i == j:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f3f2ef",
                                                   edgecolor="white", zorder=1))
                        continue
                    r = _pair_lookup(frame, s, a, b)
                    if r is None:
                        continue
                    sig = bool(r.significant_05)
                    winner = r.better
                    face = STYLE[winner][0] if sig and winner in STYLE else "white"
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                               edgecolor=GRID, linewidth=0.8,
                                               alpha=0.9 if sig else 1, zorder=2))
                    if kind == "wilcoxon":
                        label = f"{r.holm_p:.2f}" if r.holm_p >= 0.01 else f"{r.holm_p:.3f}"
                    else:
                        label = f"{r.rank_gap:.1f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=6.6,
                            color="white" if sig else "#8a8780",
                            fontweight="bold" if sig else "normal", zorder=3)
            ax.set_xticks(range(k), [SHORT[c] for c in controls], fontsize=7.6)
            ax.set_yticks(range(k), [SHORT[c] for c in controls] if col == 0 else [], fontsize=7.6)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(f"k = {SHOT_LABELS[s]}" if s != "full" else "full*", fontsize=10.5, pad=6)
    cds = nem.nemenyi_cd
    cd_txt = f"CD = {cds.iloc[0]:.2f}" if cds.nunique() == 1 else f"CD = {cds.min():.2f}-{cds.max():.2f}"
    axes[0, 0].text(-0.02, 1.26, "(a) Wilcoxon signed-rank, Holm across all pairs: adjusted p",
                    transform=axes[0, 0].transAxes, fontsize=9.2, color=INK, fontweight="bold")
    axes[1, 0].text(-0.02, 1.08, f"(b) Nemenyi post hoc: mean-rank gap, {cd_txt} "
                    f"({k} methods, {n_note(nem)})", transform=axes[1, 0].transAxes,
                    fontsize=9.2, color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[c][0]) for c in controls]
    fig.legend(handles, [f"{STYLE[c][4]} higher" for c in controls],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=8.2, frameon=False)
    save(fig, fam.figdir / ("fig_significance_allpairs" + ("" if design == "primary" else "_all")))


def family_figures(name):
    fam = Family(name)
    jobs = [("pooled", lambda: fig_pooled(fam)), ("pooled all", lambda: fig_pooled(fam, True)),
            ("ranks", lambda: fig_ranks(fam)), ("ranks all", lambda: fig_ranks(fam, "all")),
            ("significance confirmatory", lambda: fig_significance_confirmatory(fam)),
            ("significance all pairs", lambda: fig_significance_allpairs(fam)),
            ("significance all pairs, all methods", lambda: fig_significance_allpairs(fam, "all")),
            *[(f"cd {s}", lambda s=s: fig_cd_diagram(fam, s)) for s in PRIMARY_SHOTS],
            *[(f"cd {s} all", lambda s=s: fig_cd_diagram(fam, s, "all")) for s in SHOTS],
            *[(f"diff {a}", lambda a=a: fig_score_diff(fam, a)) for a in SELECTORS + [FLOOR]],
            ("stability", lambda: fig_stability(fam)), ("overlap", lambda: fig_overlap(fam))]
    return run_jobs(jobs, name)


# ----------------------------------------------------------- across families

def fig_semantic_anchor():
    """LLM prior minus the best data-driven selector per dataset, by
    stratum, at each few-shot level; the moderator figure."""
    frames = []
    for fam_name in ("classification", "regression"):
        path = ANALYSIS_ROOT / fam_name / "summary_per_dataset.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"shot": str}))
    per = pd.concat(frames, ignore_index=True)
    wide = per.pivot_table(index="dataset", columns=["shot", "control"], values="mean_score")
    shots = [s for s in PRIMARY_SHOTS if (s, "llm") in wide.columns]
    rows = []
    for d in wide.index:
        for s in shots:
            best = np.nanmax([wide.loc[d, (s, c)] for c in SELECTORS if (s, c) in wide.columns])
            rows.append(dict(dataset=d, shot=s, diff=100 * (wide.loc[d, (s, "llm")] - best),
                             stratum=tabular.DATASETS[d].semantic,
                             task=tabular.DATASETS[d].task_type))
    df = pd.DataFrame(rows).dropna()
    df.to_csv(ANALYSIS_ROOT / "semantic_anchor_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(0, color="#8a8780", lw=1.1, ls=(0, (4, 2)), zorder=1)
    rng = np.random.default_rng(0)
    x = np.arange(len(shots))
    for stratum in ("named", "anonymous"):
        sub = df[df.stratum == stratum]
        off = -0.17 if stratum == "named" else 0.17
        for j, s in enumerate(shots):
            vals = sub[sub.shot == s]
            for task, marker in (("classification", "o"), ("regression", "s")):
                v = vals[vals.task == task]
                if v.empty:
                    continue
                ax.scatter(x[j] + off + rng.uniform(-0.08, 0.08, len(v)), v["diff"], s=30,
                           color=STRATUM_COLOUR[stratum], edgecolor="white", linewidth=0.6,
                           alpha=0.85, zorder=3, marker=marker,
                           label=stratum if (j == 0 and task == "classification") else None)
            if len(vals):
                m = float(np.median(vals["diff"]))
                ax.plot([x[j] + off - 0.13, x[j] + off + 0.13], [m, m], color=INK, lw=1.8, zorder=4)
    ax.set_xticks(x, [f"k = {s}" for s in shots])
    ax.xaxis.grid(False)
    ax.set_ylabel("LLM prior − best data-driven selector\n(test score, points)")
    ax.legend(fontsize=9.2, loc="lower right", title="Feature names", title_fontsize=9)
    ax.text(0.01, 1.01, "circles: classification (ROC-AUC); squares: regression (Spearman ρ); bars: median",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, va="bottom")
    save(fig, FIGROOT / "fig_semantic_anchor")


def fig_cross_task_ranks():
    fn = pd.read_csv(ANALYSIS_ROOT / "cross_task_ranks.csv", dtype={"shot": str})
    shots = [s for s in PRIMARY_SHOTS if s in set(fn.shot)]
    long = fn.melt(id_vars=["shot", "nemenyi_cd"], value_vars=[f"rank_{a}" for a in PRIMARY_CONTROLS],
                   var_name="control", value_name="rank")
    long["control"] = long.control.str.removeprefix("rank_")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    control_lines(ax, long, "rank", controls=PRIMARY_CONTROLS, shots=shots)
    ax.invert_yaxis()
    ax.text(0.02, 0.04, f"Within-dataset ranks over both task types; Nemenyi CD = "
            f"{fn.nemenyi_cd.min():.2f}-{fn.nemenyi_cd.max():.2f} ({n_note(fn)}); exploratory",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_ylabel("Mean Friedman rank (1 = best)")
    legend_sorted(ax, long, "rank", PRIMARY_CONTROLS, shots, ascending=True, fontsize=9.5,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, FIGROOT / "fig_cross_task_mean_rank_vs_shot")


def _results():
    from geprior import fewshot
    df = fewshot._load(config.RESULTS_DIR / "fewshot.csv")
    return df[df.control.isin(CONTROLS) & df.shot.isin(SHOTS) & df.dataset.isin(tabular.DATASETS)]


def fig_compute_time():
    """(a) wall-clock seconds per GE cell, median with the interquartile range
    over all cells of a method, each selection method annotated with its
    paired median ratio to the all-features run and the Holm-adjusted p.
    (b) the selectors' own fitting cost per split, log scale, against one GE
    search on the same machine."""
    df = _results()
    contr = pd.read_csv(ANALYSIS_ROOT / "compute_time_contrasts.csv", dtype={"shot": str})
    contr = contr[contr.shot == "all"].set_index("contrast")
    timing_path = ANALYSIS_ROOT / "selector_timing.csv"
    have_timing = timing_path.exists()
    fig, axes = plt.subplots(1, 2 if have_timing else 1,
                             figsize=(12.6 if have_timing else 7.4, 5.0),
                             gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3} if have_timing else None)
    ax = axes[0] if have_timing else axes
    order = [TIME_REF] + [c for c in CONTROLS if c != TIME_REF]
    x = np.arange(len(order))
    med = np.array([df[df.control == c].seconds.median() for c in order])
    q1 = np.array([df[df.control == c].seconds.quantile(0.25) for c in order])
    q3 = np.array([df[df.control == c].seconds.quantile(0.75) for c in order])
    cols = [STYLE[c][0] for c in order]
    bars = ax.bar(x, med, width=0.66, color=cols, edgecolor="white", linewidth=0.8, zorder=3)
    for b in bars[1:]:
        b.set_alpha(0.82)
    ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none", ecolor=INK,
                elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)
    ax.axhline(med[0], color=STYLE[TIME_REF][0], lw=1.1, ls=(0, (4, 2)), zorder=2)
    for i, c in enumerate(order):
        if c == TIME_REF or f"{c} vs {TIME_REF}" not in contr.index:
            continue
        r = contr.loc[f"{c} vs {TIME_REF}"]
        ax.text(x[i], q3[i] + 0.6, f"×{r.median_ratio:.2f}\np = {r.holm_p:.2f}",
                ha="center", va="bottom", fontsize=8.4, color=INK)
    ax.set_xticks(x, [STYLE[c][4].replace(" ", "\n", 1) for c in order], fontsize=9)
    ax.set_ylabel("Wall-clock seconds per GE cell (median, IQR)")
    ax.set_ylim(0, max(q3) * 1.32)
    ax.xaxis.grid(False)
    if have_timing:
        panel_label(ax, "a")
        ax2 = axes[1]
        panel_label(ax2, "b")
        t = pd.read_csv(timing_path)
        ref_path = ANALYSIS_ROOT / "selector_timing_reference_cells.csv"
        ge_ref = pd.read_csv(ref_path).ge_cell_seconds.median() if ref_path.exists() else None
        sels = SELECTORS + ["llm"]
        per = t[t.control.isin(sels)].groupby(["dataset", "control"]).seconds.median().reset_index()
        floor = 1e-3
        xs = np.arange(len(sels))
        rng = np.random.default_rng(0)
        for i, c in enumerate(sels):
            vals = per[per.control == c].seconds.clip(lower=floor).to_numpy()
            jitter = rng.uniform(-0.17, 0.17, len(vals))
            ax2.scatter(xs[i] + jitter, vals, s=34, color=STYLE[c][0], edgecolor="white",
                        linewidth=0.7, zorder=3, alpha=0.9)
            m = float(np.median(vals))
            ax2.plot([xs[i] - 0.3, xs[i] + 0.3], [m, m], color=INK, lw=1.6, zorder=4)
        big = per[per.seconds > 1]
        for _, r in big.iterrows():
            ax2.annotate(tabular.DATASETS[r.dataset].label,
                         (sels.index(r.control) + 0.2, r.seconds), fontsize=7.4,
                         color=MUTED, va="center")
        if ge_ref:
            ax2.axhline(ge_ref, color="#5b6676", lw=1.1, ls=(0, (4, 2)), zorder=2)
            ax2.text(-0.4, ge_ref * 1.3, f"one GE search ≈ {ge_ref:.0f} s", ha="left",
                     fontsize=8.4, color="#5b6676")
        ax2.set_yscale("log")
        ax2.set_ylim(floor * 0.8, max(per.seconds.max(), ge_ref or 1) * 4)
        ax2.set_xticks(xs, [STYLE[c][4].replace(" ", "\n", 1) for c in sels], fontsize=9)
        ax2.set_ylabel("Selector fitting time per split, s (log scale)")
        ax2.xaxis.grid(False)
    save(fig, FIGROOT / "fig_compute_time")


def fig_compute_time_vs_shot():
    pooled = pd.read_csv(ANALYSIS_ROOT / "compute_time_pooled.csv", dtype={"shot": str})
    pooled = pooled[pooled.shot != "all"]
    shots = [s for s in SHOTS if s in set(pooled.shot)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(shots))
    for control in CONTROLS:
        sub = pooled[pooled.control == control].set_index("shot").reindex(shots)
        c, m, ls, lw, label = STYLE[control]
        ref = control == TIME_REF
        ax.plot(x, sub.pooled_median_seconds, color=c, marker=m, linestyle="-" if ref else ls,
                linewidth=3.0 if ref else 1.5, markersize=7.5 if ref else 5.5,
                markeredgecolor="white", markeredgewidth=0.8, label=label,
                zorder=4 if ref else 2, alpha=1 if ref else 0.85)
    ax.xaxis.grid(False)
    ax.margins(x=0.03)
    ax.set_xticks(x, [SHOT_LABELS[s] for s in shots])
    ax.set_xlabel("Labelled rows, k")
    ax.set_ylabel(f"Median seconds per GE cell ({int(pooled.n_datasets.max())} datasets)")
    ax.set_ylim(0, pooled.pooled_median_seconds.max() * 1.25)
    ax.annotate(FULL_NOTE, xy=(1.0, -0.16), xycoords="axes fraction",
                ha="right", fontsize=8.2, color=MUTED, annotation_clip=False)
    ax.legend(ncol=2, fontsize=9.2, loc="upper left")
    save(fig, FIGROOT / "fig_compute_time_vs_shot")


def fig_compute_time_ratio():
    """Per dataset, the median within-cell ratio of each selection method's
    time to the all-features run; 1 means no difference."""
    contr = pd.read_csv(ANALYSIS_ROOT / "compute_time_contrasts.csv", dtype={"shot": str})
    contr = contr[contr.shot == "all"].set_index("contrast")
    df = _results()
    ref = df[df.control == TIME_REF].set_index(["dataset", "shot", "seed"]).seconds
    df = df.join(ref.rename("ref"), on=["dataset", "shot", "seed"])
    df["ratio"] = df.seconds / df.ref
    sels = [c for c in CONTROLS if c != TIME_REF]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(sels))
    rng = np.random.default_rng(1)
    for i, c in enumerate(sels):
        vals = df[df.control == c].groupby("dataset").ratio.median().to_numpy()
        ax.scatter(vals, y[i] + rng.uniform(-0.16, 0.16, len(vals)), s=40, color=STYLE[c][0],
                   edgecolor="white", linewidth=0.7, zorder=3)
        m = float(np.median(vals))
        ax.plot([m, m], [y[i] - 0.3, y[i] + 0.3], color=INK, lw=1.8, zorder=4)
        if f"{c} vs {TIME_REF}" in contr.index:
            r = contr.loc[f"{c} vs {TIME_REF}"]
            ax.text(1.52, y[i], f"Holm p = {r.holm_p:.2f}",
                    va="center", fontsize=8.6, color=MUTED)
    ax.axvline(1.0, color=STYLE[TIME_REF][0], lw=1.4, ls=(0, (4, 2)), zorder=2)
    ax.set_yticks(y, [STYLE[c][4] for c in sels])
    ax.set_ylim(len(sels) - 0.5, -0.7)
    lo = min(0.75, float(df.groupby(["control", "dataset"]).ratio.median().min()) * 0.95)
    ax.set_xlim(lo, 1.62)
    ax.yaxis.grid(False)
    ax.set_xlabel("Median time ratio to the all-features run in the same cell")
    save(fig, FIGROOT / "fig_compute_time_ratio")


def fig_per_dataset(key):
    spec = tabular.DATASETS[key]
    fam_name = "classification" if spec.task_type == "classification" else "regression"
    per = pd.read_csv(ANALYSIS_ROOT / fam_name / "summary_per_dataset.csv", dtype={"shot": str})
    sub = per[per.dataset == key].rename(columns={"mean_score": "value"})
    if sub.empty:
        raise ValueError(f"no completed cells for {key}")
    shots = [s for s in SHOTS if s in set(sub.shot)]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    control_lines(ax, sub, "value", shots=shots)
    ax.set_ylabel(f"Mean test {config.METRIC_LABEL[spec.task_type]} ({config.FEWSHOT_DRAWS} draws)")
    ax.text(0.0, 1.02, f"{spec.label}, K = {spec.budget}, {spec.semantic} names", transform=ax.transAxes,
            fontsize=10.5, ha="left", va="bottom")
    ax.legend(ncol=2, fontsize=8.5, loc="lower right")
    save(fig, FIGROOT / "per_dataset" / f"fig_score_{key}")


def run_jobs(jobs, label):
    failed = []
    for name, job in jobs:
        try:
            job()
        except Exception as e:                      # usually an incomplete run
            failed.append(name)
            print(f"  [{label}] skipped {name}: {type(e).__name__}: {e}")
    return failed


def main():
    FIGROOT.mkdir(parents=True, exist_ok=True)
    failed = []
    for family in config.FAMILIES:
        if (ANALYSIS_ROOT / family / "family.csv").exists():
            failed += family_figures(family)
        else:
            print(f"  [{family}] no analysis tables; skipped")
    jobs = [("semantic anchor", fig_semantic_anchor), ("cross-task ranks", fig_cross_task_ranks),
            ("compute time", fig_compute_time), ("compute time vs shot", fig_compute_time_vs_shot),
            ("compute time ratio", fig_compute_time_ratio),
            *[(f"dataset {k}", lambda k=k: fig_per_dataset(k)) for k in tabular.DATASETS]]
    failed += run_jobs(jobs, "all")
    if failed:
        print(f"{len(failed)} figure(s) skipped; re-run once the campaign completes")


if __name__ == "__main__":
    main()
