"""Per-method views: each method and each selector on its own, no overlays.

Complements selector_figures.py (comparisons). Three families of output:

  figures/per_method/           one result sheet per method, the all-features
                                baseline included: dataset x shot matrix of
                                mean test AUC over the 10 draws, plus the
                                pooled median row.
  figures/selector_diagnostics/ per-dataset diagnostics: PCA loadings with
                                explained variance, MI profile, correlation
                                profile (|r| binary, eta multiclass), RF
                                importance profile, each with the budget-K
                                cutoff; and a selection-frequency map per
                                method from the logged subsets.
  results/analysis/per_method/  the matching tables.

Scores are recomputed with the run's own code paths on the rows the selectors
scored: the whole training partition of each of the ten splits under the
full-information design, or the capped full-level labelled rows under the
matched design (FEWSHOT_SELECTOR_ROWS). Bars carry across-draw whiskers and
match what each method handed the grammar. Selection frequencies come from
the `features` column of results/fewshot.csv; nothing is re-evolved."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, priors, selectors, tabular  # noqa: E402
from geprior.fewshot import Cell, fewshot_mutual_information, selector_rows  # noqa: E402

ANALYSIS = config.RESULTS_DIR / "analysis"
TABDIR = ANALYSIS / "per_method"
CONTROLDIR = config.FIGURES_DIR / "per_method"
DIAGDIR = config.FIGURES_DIR / "selector_diagnostics"
SHOTS = ["4", "8", "16", "32", "64", "full"]
SHOT_LABELS = {"4": "4", "8": "8", "16": "16", "32": "32", "64": "64", "full": "full*"}
CONTROLS = list(config.FEWSHOT_CONTROLS)
DRAWS = config.FEWSHOT_DRAWS
MAX_BARS = 25            # longest readable profile; the table carries the rest
MAX_LOADING_ROWS = 20    # loadings matrix rows
MAX_PCS = 8              # loadings matrix components (or 90% EVR, whichever first)

STYLE = {   # colour, legend label -- same identities as selector_figures.py
    "none":   ("#5b6676", "All features"),
    "random": ("#a3a9b3", "Random subset"),
    "corr":   ("#009E73", "Correlation"),
    "mi":     ("#0072B2", "Mutual information"),
    "pca":    ("#CC79A7", "PCA loadings"),
    "rf":     ("#D55E00", "RF importance"),
    "llm":    ("#E69F00", "LLM prior"),
}
LOADING_CMAP = LinearSegmentedColormap.from_list(
    "loading_div", ["#0072B2", "#f3f2ef", "#D55E00"])
FREQ_CMAP = LinearSegmentedColormap.from_list(
    "freq_seq", ["#f7f6f3", "#2b2a27"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Inter", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.color": "#e3e1dc", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.edgecolor": "#8a8780",
    "axes.labelcolor": "#2b2a27", "text.color": "#2b2a27",
    "xtick.color": "#5a5852", "ytick.color": "#5a5852",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": True, "legend.facecolor": "white", "legend.framealpha": 0.88,
    "legend.edgecolor": "none",
})


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {outdir.relative_to(config.ROOT)}/{name}.png")


# ---------------------------------------------------------------- control sheets

def control_cmap(control):
    return LinearSegmentedColormap.from_list(f"seq_{control}", ["#f7f6f3", STYLE[control][0]])


def cell_ink(rgba):
    """Dark ink on light cells, white on dark ones."""
    r, g, b = rgba[:3]
    return "#2b2a27" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def fig_control_sheet(control, per, family):
    """One method alone: dataset x shot matrix of mean score, pooled median
    below; one sheet per family (classification in ROC-AUC, regression in R^2)."""
    colour, label = STYLE[control]
    keys = [k for k in tabular.family_keys(family) if k in set(per.dataset)]
    shots = [s for s in SHOTS if s in set(per.shot)]
    sub = per[per.control == control]

    def cell(k, s):
        m = sub[(sub.dataset == k) & (sub.shot == s)].mean_score
        return 100 * m.iloc[0] if len(m) else np.nan
    mat = np.array([[cell(k, s) for s in shots] for k in keys])
    pooled = np.nanmedian(mat, axis=0)
    full = np.vstack([mat, pooled])
    lo, hi = (45, 100) if family == "classification" else (-100, 100)
    fig, ax = plt.subplots(figsize=(7.2, 2.2 + 0.32 * len(keys)))
    ax.grid(False)
    cmap = control_cmap(control)
    im = ax.imshow(full, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(range(len(shots)), [SHOT_LABELS[s] for s in shots])
    names = [tabular.DATASETS[k].label for k in keys] + [f"Median ({len(keys)} datasets)"]
    ax.set_yticks(range(len(names)), names)
    ax.get_yticklabels()[-1].set_fontstyle("italic")
    ax.axhline(len(keys) - 0.5, color="white", lw=3)
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            if np.isnan(full[i, j]):
                continue
            shade = cmap((full[i, j] - lo) / (hi - lo))
            ax.text(j, i, f"{full[i, j]:.1f}", ha="center", va="center",
                    fontsize=8.6, color=cell_ink(shade))
    ax.set_xlabel("Labelled rows, k   (*full: training rows capped at 1000)")
    ax.set_title(f"{label}, {family}", loc="left", fontsize=11.5, pad=10)
    metric = config.METRIC_LABEL[config.FAMILIES[family]["task"]]
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(f"Mean test {metric} \u00d7 100")
    save(fig, CONTROLDIR, f"fig_control_score_{control}_{family}")
    out = sub[sub.dataset.isin(keys)].copy()
    out["shot"] = pd.Categorical(out.shot, SHOTS, ordered=True)
    (out.sort_values(["dataset", "shot"])
        .to_csv(TABDIR / f"control_score_{control}_{family}.csv", index=False))


# ------------------------------------------------------- selector diagnostics

def draw_rows(enc, seed):
    """The rows the selectors scored on this draw."""
    if config.FEWSHOT_SELECTOR_ROWS == "train":
        return selector_rows(enc, seed)
    cell = Cell(enc, "full", seed)
    return cell.X_fit, cell.y_fit


def per_draw_scores(enc, rows):
    """Per-encoded-column scores on each draw's selector rows, per selector."""
    out = {}
    for control in ("corr", "mi", "pca", "rf"):
        per_draw = []
        for X, y in rows:
            if control == "mi":
                per_draw.append(fewshot_mutual_information(X, y, enc.n_classes))
            else:
                per_draw.append(selectors.column_scores(control, X, y, enc.n_classes))
        out[control] = np.vstack(per_draw)
    return out


def group_max(enc, col_scores):
    """Aggregate column scores to original features, the campaign's convention."""
    return np.array([[draw[enc.groups[n]].max() for n in enc.features]
                     for draw in col_scores])


def logged_freq(df, key, control, names, shot="full"):
    """Selection frequency from the run log, not recomputed: exact collinear
    ties (Blood: Monetary = 250 x Frequency) resolve by last-ulp noise that
    differs across NumPy builds, so the log is the ground truth."""
    sub = df[(df.dataset == key) & (df.control == control) & (df.shot == shot)]
    sets = [frozenset(f.split("|")) for f in sub.features]
    return np.array([np.mean([n in s for s in sets]) for n in names])


def score_name(control, n_classes):
    if control == "corr":
        return {0: "|Pearson r|", 2: "|point-biserial r|"}.get(n_classes, "correlation ratio")
    return {"mi": "mutual information (nats)", "pca": "variance-weighted |loading|",
            "rf": "impurity importance"}[control]


def fig_profile(key, enc, control, feat_scores, freq):
    """The traditional bar profile: one selector, one dataset, every feature."""
    colour, label = STYLE[control]
    spec = tabular.DATASETS[key]
    names = np.array(enc.features)
    mean, lo, hi = feat_scores.mean(0), feat_scores.min(0), feat_scores.max(0)
    order = np.argsort(-mean, kind="stable")
    shown = order[:MAX_BARS]
    y = np.arange(len(shown))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.34 * len(shown) + 1.6)))
    ax.yaxis.grid(False)
    cols = [colour if freq[i] >= 0.5 else "#c9c6c0" for i in shown]
    ax.barh(y, mean[shown], color=cols, edgecolor="white", linewidth=0.5)
    ax.errorbar(mean[shown], y, xerr=[np.clip(mean[shown] - lo[shown], 0, None), np.clip(hi[shown] - mean[shown], 0, None)],
                fmt="none", ecolor="#5a5852", elinewidth=1.0, capsize=2)
    ax.set_yticks(y, names[shown], fontsize=9)
    if spec.budget < len(shown):
        ax.axhline(y[spec.budget - 1] - 0.5, color="#2b2a27", lw=1.1, ls=(0, (4, 2)))
        ax.text(ax.get_xlim()[1], y[spec.budget - 1] - 0.5, f"  budget K = {spec.budget}",
                va="center", fontsize=9, color="#2b2a27")
    ax.set_xlabel(f"{score_name(control, enc.n_classes)} "
                  f"({DRAWS}-draw mean, whiskers min-max)")
    extra = f"  (top {MAX_BARS} of {len(names)})" if len(names) > MAX_BARS else ""
    ax.set_title(f"{spec.label}: {label}{extra}", loc="left", fontsize=11.5, pad=10)
    ax.text(0.995, 0.02, f"filled: selected in at least {DRAWS // 2} of {DRAWS} draws",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
            color="#5a5852")
    save(fig, DIAGDIR / key, f"fig_{control}_{key}")


def fig_pca_loadings(key, enc, rows, pca_col_scores, freq):
    """Loadings matrix (draw 0, signed, per encoded column) next to the
    aggregated variance-weighted ranking the method actually uses."""
    from sklearn.decomposition import PCA
    spec = tabular.DATASETS[key]
    colour, _ = STYLE["pca"]
    X0, _y0 = rows[0]
    keep = np.ptp(X0, axis=0) > 0
    n_comp = min(X0.shape[0] - 1, int(keep.sum()))
    pca = PCA(n_components=n_comp, svd_solver="full").fit(X0[:, keep])
    evr = pca.explained_variance_ratio_
    n_show = min(MAX_PCS, max(2, int(np.searchsorted(np.cumsum(evr), 0.90) + 1)))
    weighted = (np.abs(pca.components_) * evr[:, None]).sum(0)
    kept_cols = np.array(enc.columns)[keep]
    top = np.argsort(-weighted, kind="stable")[:min(MAX_LOADING_ROWS, len(kept_cols))]
    mat = pca.components_[:n_show, :][:, top].T

    feat = group_max(enc, pca_col_scores)
    mean, lo, hi = feat.mean(0), feat.min(0), feat.max(0)
    order = np.argsort(-mean, kind="stable")[:MAX_BARS]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.4, max(4.2, 0.34 * max(len(top), len(order)) + 1.8)),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42})
    axL.grid(False)
    span = np.abs(mat).max()
    im = axL.imshow(mat, cmap=LOADING_CMAP, norm=TwoSlopeNorm(0, -span, span),
                    aspect="auto")
    axL.set_xticks(range(n_show), [f"PC{i+1}\n{100*evr[i]:.0f}%" for i in range(n_show)],
                   fontsize=8.6)
    axL.set_yticks(range(len(top)), kept_cols[top], fontsize=8.6)
    axL.set_title(f"(a) Signed loadings, draw 0  ({n_show}/{n_comp} PCs, "
                  f"{100*np.cumsum(evr)[n_show-1]:.0f}% of variance)",
                  loc="left", fontsize=10.5, pad=8)
    fig.colorbar(im, ax=axL, shrink=0.75).set_label("loading")

    names = np.array(enc.features)
    y = np.arange(len(order))[::-1]
    axR.yaxis.grid(False)
    cols = [colour if freq[i] >= 0.5 else "#c9c6c0" for i in order]
    axR.barh(y, mean[order], color=cols, edgecolor="white", linewidth=0.5)
    axR.errorbar(mean[order], y, xerr=[mean[order] - lo[order], hi[order] - mean[order]],
                 fmt="none", ecolor="#5a5852", elinewidth=1.0, capsize=2)
    axR.set_yticks(y, names[order], fontsize=9)
    if spec.budget < len(order):
        axR.axhline(y[spec.budget - 1] - 0.5, color="#2b2a27", lw=1.1, ls=(0, (4, 2)))
        axR.text(axR.get_xlim()[1], y[spec.budget - 1] - 0.5,
                 f"  budget K = {spec.budget}", va="center", fontsize=9)
    axR.set_xlabel(f"variance-weighted |loading| ({DRAWS}-draw mean)")
    axR.set_title("(b) Ranking used for selection", loc="left", fontsize=10.5, pad=8)
    fig.suptitle(f"{spec.label}: PCA loadings", x=0.01, ha="left", fontsize=11.5)
    save(fig, DIAGDIR / key, f"fig_pca_loadings_{key}")


def fig_selection_freq(key, enc, df):
    """What each method selected: feature x shot frequency over draws, one
    panel per method. From the logged subsets; the all-features baseline is
    omitted as it always takes every feature."""
    spec = tabular.DATASETS[key]
    controls = [a for a in ("corr", "mi", "pca", "rf", "llm") if a in CONTROLS]
    sub = df[(df.dataset == key) & df.control.isin(controls)].copy()
    sub["fset"] = sub.features.str.split("|").map(frozenset)
    names = list(enc.features)
    freq = {a: np.array([[np.mean([n in s for s in
                                   sub[(sub.control == a) & (sub.shot == sh)].fset])
                          for sh in SHOTS] for n in names]) for a in controls}
    overall = np.mean([freq[a] for a in controls], axis=(0, 2))
    rows = np.argsort(-overall, kind="stable")[:MAX_BARS]
    fig, axes = plt.subplots(1, len(controls), figsize=(2.6 * len(controls) + 0.8,
                                                        max(4.2, 0.28 * len(rows) + 1.8)),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, control in zip(np.atleast_1d(axes).ravel(), controls):
        colour, label = STYLE[control]
        ax.grid(False)
        im = ax.imshow(freq[control][rows], cmap=FREQ_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SHOTS)), [SHOT_LABELS[s] for s in SHOTS], fontsize=8.4)
        ax.set_yticks(range(len(rows)), [names[i] for i in rows], fontsize=8)
        ax.set_title(label, loc="left", fontsize=10.5)
        ax.set_xlabel("k", fontsize=9)
    extra = f" (top {MAX_BARS} of {len(names)} features)" if len(names) > MAX_BARS else ""
    fig.suptitle(f"{spec.label}: selection frequency over {DRAWS} draws{extra}",
                 x=0.01, ha="left", fontsize=11.5)
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.015).set_label("fraction of draws")
    save(fig, DIAGDIR / key, f"fig_selection_freq_{key}")


def scores_table(key, enc, col_scores, df):
    rows = []
    llm_names = priors.LLM.select_names(key, list(enc.features), enc.budget, 0,
                                        enc.task, enc.descriptions)
    freqs = {a: logged_freq(df, key, a, list(enc.features))
             for a in ("corr", "mi", "pca", "rf")}
    for i, n in enumerate(enc.features):
        row = dict(feature=n, encoded_cols=len(enc.groups[n]),
                   llm_selected=n in llm_names)
        for control in ("corr", "mi", "pca", "rf"):
            feat = group_max(enc, col_scores[control])
            row[f"{control}_mean"] = float(feat.mean(0)[i])
            row[f"{control}_min"] = float(feat.min(0)[i])
            row[f"{control}_max"] = float(feat.max(0)[i])
            row[f"{control}_sel_freq_full"] = float(freqs[control][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABDIR / f"selector_scores_{key}.csv", index=False)


def main():
    for d in (CONTROLDIR, DIAGDIR, TABDIR):
        d.mkdir(parents=True, exist_ok=True)
    from geprior import fewshot
    df = fewshot._load(config.RESULTS_DIR / "fewshot.csv")
    print("per-method sheets:")
    for family in ("classification", "regression"):
        path = ANALYSIS / family / "summary_per_dataset.csv"
        if not path.exists():
            continue
        per = pd.read_csv(path, dtype={"shot": str})
        for control in CONTROLS:
            fig_control_sheet(control, per, family)
    print("selector diagnostics:")
    for key in tabular.DATASETS:
        if key not in set(df.dataset):
            continue
        (DIAGDIR / key).mkdir(exist_ok=True)
        enc = tabular.load(key)
        rows = [draw_rows(enc, s) for s in range(DRAWS)]
        col_scores = per_draw_scores(enc, rows)
        names = list(enc.features)
        for control in ("corr", "mi", "rf"):
            fig_profile(key, enc, control, group_max(enc, col_scores[control]),
                        logged_freq(df, key, control, names))
        fig_pca_loadings(key, enc, rows, col_scores["pca"],
                         logged_freq(df, key, "pca", names))
        fig_selection_freq(key, enc, df)
        scores_table(key, enc, col_scores, df)
    print(f"-> {CONTROLDIR}\n-> {DIAGDIR}\n-> {TABDIR}")


if __name__ == "__main__":
    main()
