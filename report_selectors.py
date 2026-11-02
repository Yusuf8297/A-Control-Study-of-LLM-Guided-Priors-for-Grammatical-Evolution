"""Write results/analysis/report.md from the analysis tables. Every number is
read from the executed outputs, never typed in, so a re-run propagates. The
wording is generated from the tables and never presumes a direction."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, tabular  # noqa: E402

A = config.RESULTS_DIR / "analysis"
SHOTS = list(config.ALL_SHOTS)
CONTROLS = list(config.FEWSHOT_CONTROLS)
SELECTORS = ["corr", "mi", "pca", "rf"]
FLOOR = config.SEMANTIC_FLOOR_CONTROL
PRIMARY_CONTROLS = list(config.FEWSHOT_PRIMARY_CONTROLS)
PRIMARY_SHOTS = list(config.FEWSHOT_PRIMARY_SHOTS)
NAMES = {"none": "all features", "random": "random subset",
         "corr": "correlation", "mi": "mutual information", "pca": "PCA loadings",
         "rf": "RF importance", "llm": "LLM prior"}
FAMILY_TITLE = {"classification": "Classification (primary family)",
                "regression": "Regression (descriptive family)",
                "classification_named": "Classification, named feature strata (moderator)",
                "classification_anonymous": "Classification, anonymous feature strata (descriptive)"}


def read(path):
    return pd.read_csv(path, dtype={"shot": str}) if path.exists() else None


def level(shot):
    return f"k = {shot}" if shot != "full" else "the full level"


def levels(frame):
    return ", ".join(f"k = {x}" if x != "full" else "full" for x in frame.shot) or "none"


def other_of(contrast):
    return contrast.split(" vs ")[1]


def cap(name):
    return name[0].upper() + name[1:]


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|---" * len(header) + "|",
                      *["| " + " | ".join(r) + " |" for r in rows]])


def pooled_tbl(pooled, controls, scale=100):
    wide = pooled.pivot(index="control", columns="shot", values="pooled_median").reindex(controls)
    shots = [s for s in SHOTS if s in wide.columns]
    ns = pooled.groupby("shot").n_datasets.max()
    head = ["Method"] + [f"k = {s} (N = {int(ns[s])})" if s != "full" else f"full (N = {int(ns[s])})" for s in shots]
    rows = [[cap(NAMES[c])] + [f"{wide.loc[c, s] * scale:.1f}" for s in shots] for c in controls]
    return md_table(head, rows)


def contrast_tbl(conf):
    rows = []
    for _, r in conf.iterrows():
        star = " \\*" if r.significant_05 else ""
        rows.append([r.shot, f"LLM prior vs {NAMES[other_of(r.contrast)]}", str(int(r.n_datasets)),
                     f"{r.median_diff_pts:+.1f}", f"{r.wilcoxon_p:.3f}", f"{r.holm_p:.3f}{star}",
                     f"{r.cliffs_delta:+.2f} ({r.magnitude})"])
    return md_table(["Level", "Contrast", "N", "Median difference", "Wilcoxon p", "Holm p",
                     "Cliff's delta"], rows)


def fried_tbl(frame, controls):
    rows = []
    for _, r in frame.iterrows():
        rows.append([r.shot, str(int(r.n_datasets)), f"{r.friedman_chi2:.1f}", f"{r.friedman_p:.3f}"]
                    + [f"{r[f'rank_{a}']:.2f}" for a in controls] + [f"{r.nemenyi_cd:.2f}"])
    return md_table(["Level", "N", "Friedman chi-squared", "p"] + [cap(NAMES[a]) for a in controls]
                    + ["CD"], rows)


def summ_tbl(summ):
    rows = []
    for _, r in summ.iterrows():
        if pd.isna(r.nemenyi_cd):
            nem, nv = "-", "not in omnibus"
        else:
            nem = f"{r.nemenyi_rank_gap:.2f} / {r.nemenyi_cd:.2f}"
            nv = "significant" if r.nemenyi_significant else "n.s."
        rows.append([r.shot, f"LLM prior vs {NAMES[other_of(r.contrast)]}", f"{r.median_diff_pts:+.1f}",
                     f"{r.holm_p:.3f}", r.outcome, nem, nv])
    return md_table(["Level", "Contrast", "Median difference", "Holm p", "Wilcoxon/Holm",
                     "Nemenyi gap / CD", "Nemenyi"], rows)


def pairs_list(frame, label_fn):
    sig = frame[frame.significant_05]
    if sig.empty:
        return "none"
    return "; ".join(label_fn(r) for _, r in sig.iterrows())


def direction_sentence(conf, what="contrasts"):
    """Which contrasts reached significance and in whose favour, from the table."""
    sig = conf[conf.significant_05]
    n, total = len(sig), len(conf)
    if n == 0:
        return (f"None of the {total} {what} reached significance after Holm correction.")
    parts = []
    for _, r in sig.iterrows():
        who = "the LLM prior" if r.median_diff_pts > 0 else NAMES[other_of(r.contrast)]
        parts.append(f"{NAMES[other_of(r.contrast)]} at {level(r.shot)} (favouring {who}, "
                     f"{r.magnitude} effect)")
    return (f"{n} of the {total} {what} reached significance after Holm correction: LLM prior vs "
            + "; ".join(parts) + ".")


def family_section(name):
    out = A / name
    if not (out / "family.csv").exists():
        return f"## {FAMILY_TITLE.get(name, name)}\n\nNo complete cells for this family yet.\n"
    info = pd.read_csv(out / "family.csv").iloc[0]
    metric = config.METRIC_LABEL[info.task]
    unit = "points" if info.task == "classification" else "x 100"
    pooled, conf, floor = read(out / "pooled_summary.csv"), read(out / "confirmatory_wilcoxon.csv"), \
        read(out / "semantic_floor_wilcoxon.csv")
    fried, fried_all = read(out / "friedman_nemenyi_primary.csv"), read(out / "friedman_nemenyi_all.csv")
    summ, allp, nemp = read(out / "significance_summary.csv"), \
        read(out / "exploratory_allpairs_wilcoxon_primary.csv"), read(out / "nemenyi_pairwise_primary.csv")
    nem_all = read(out / "nemenyi_pairwise_all.csv")
    stab, over = read(out / "stability_pooled.csv"), read(out / "overlap_with_llm_pooled.csv")
    datasets = ", ".join(tabular.DATASETS[k].label for k in info.datasets.split("|"))
    role = info.role
    text = [f"## {FAMILY_TITLE.get(name, name)}", "",
            f"{int(info.n_datasets)} datasets: {datasets}. Metric: test {metric}; differences are in {unit}. "
            f"Pre-specified role of the family: {role}."]
    if role == "descriptive":
        text.append("Tests in this family are reported for the direction and size of the differences; "
                    "with these few blocks the Holm-corrected Wilcoxon cannot reach alpha, so no "
                    "significance claim is made from it.")
    text += ["", f"### Pooled test {metric}", "",
             f"Median across the datasets of per-dataset means ({unit}); N is the number of datasets "
             "with that shot level.", "", pooled_tbl(pooled, CONTROLS)]
    sec = read(out / "pooled_summary_secondary.csv")
    if sec is not None:
        label = config.SECONDARY_METRIC[info.task][1]
        text += ["", f"Secondary metric, calibrated test {label} (x 100), same layout; the metric of "
                 "Memar et al. (2026), degenerate at small k (README, change 6):", "",
                 pooled_tbl(sec, CONTROLS)]
    if conf is not None and len(conf):
        text += ["", "### Confirmatory contrasts: LLM prior against each data-driven selector", "",
                 "Blocked on dataset (per-dataset means), two-sided Wilcoxon signed-rank, Holm across "
                 "the four contrasts within each level. Positive differences favour the LLM prior; an "
                 f"asterisk marks Holm p < {config.ALPHA}.", "", contrast_tbl(conf), "",
                 direction_sentence(conf, "selector contrasts")]
    if floor is not None and len(floor):
        text += ["", "### Semantic floor: LLM prior against the random subset", "",
                 "The same size-K subset drawn at random; for a prior that never sees data, "
                 "anonymised names leave no information about the columns, so this contrast is the "
                 "semantic contribution of the prior at matched cardinality. Holm across the few-shot "
                 "levels; the full level carries its raw p.", "", contrast_tbl(floor), "",
                 direction_sentence(floor, "semantic-floor contrasts")]
    if summ is not None and len(summ):
        n_dis = int((~summ.agree & summ.confirmatory_significant).sum())
        text += ["", "### Confirmatory contrasts next to the Nemenyi reading", "", summ_tbl(summ)]
        if n_dis:
            text.append(f"\nThe Nemenyi post hoc, which corrects over all pairs on mean ranks and is the "
                        f"more conservative reading, does not separate {n_dis} of the pairs the "
                        f"confirmatory contrasts separate.")
    if fried is not None and len(fried):
        sig = fried[fried.friedman_p < config.ALPHA]
        text += ["", "### Omnibus tests", "",
                 "Friedman over the five selection methods at the few-shot levels, blocked on dataset; "
                 "mean rank, 1 = best; Nemenyi critical difference for five methods and that level's N.",
                 "", fried_tbl(fried, PRIMARY_CONTROLS), "",
                 (f"The Friedman test is significant at {levels(sig)}." if len(sig) else
                  "The Friedman test is not significant at any level.")]
        if nemp is not None:
            text.append("Pairs the Nemenyi CD separates: " + pairs_list(
                nemp, lambda r: f"{r.shot}: {NAMES[r.better]} > {NAMES[r.b if r.better == r.a else r.a]} "
                                f"({r.rank_gap:.2f})") + ".")
        if allp is not None:
            text.append("Exploratory all-pairs Wilcoxon with Holm across the ten pairs: " + pairs_list(
                allp, lambda r: f"{r.shot}: {NAMES[r.better]} > {NAMES[r.b if r.better == r.a else r.a]} "
                                f"(Holm p = {r.holm_p:.3f})") + ".")
    if fried_all is not None and len(fried_all):
        text += ["", f"Omnibus over all {len(CONTROLS)} reported methods at all levels "
                 f"(CD for {len(CONTROLS)} methods):", "", fried_tbl(fried_all, CONTROLS)]
        if nem_all is not None:
            text.append("\nPairs the CD separates: " + pairs_list(
                nem_all, lambda r: f"{r.shot}: {NAMES[r.better]} > {NAMES[r.b if r.better == r.a else r.a]} "
                                   f"({r.rank_gap:.2f})") + ".")
    if stab is not None:
        st = stab.pivot(index="control", columns="shot", values="stability").reindex(
            [a for a in CONTROLS if a != "none"])
        st = st[[s for s in SHOTS if s in st.columns]]
        st.index = [cap(NAMES[c]) for c in st.index]
        st.index.name = "Method"
        text += ["", "### Selection stability across draws", "",
                 "Mean pairwise Jaccard similarity of the selected subsets, pooled across datasets. The "
                 "cached LLM prior is a fixed subset at these budgets, so its stability is 1 by "
                 "construction.", "", st.round(3).to_markdown()]
    if over is not None:
        ov = over.pivot(index="control", columns="shot", values="jaccard_llm")
        ov = ov[[s for s in SHOTS if s in ov.columns]]
        ov.index = [cap(NAMES[c]) for c in ov.index]
        ov.index.name = "Method"
        text += ["", "### Overlap with the LLM subset", "",
                 "Mean Jaccard similarity between each method's subset and the LLM subset in the same "
                 "cell, pooled across datasets.", "", ov.round(3).to_markdown()]
    return "\n".join(text) + "\n"


def compute_section():
    tpool, tcon = read(A / "compute_time_pooled.csv"), read(A / "compute_time_contrasts.csv")
    if tpool is None:
        return ""
    wide = tpool.pivot(index="control", columns="shot", values="pooled_median_seconds").reindex(CONTROLS)
    cols = [s for s in SHOTS + ["all"] if s in wide.columns]
    rows = [[cap(NAMES[c])] + [f"{wide.loc[c, s]:.1f}" for s in cols] for c in CONTROLS]
    tbl = md_table(["Method"] + [f"k = {s}" if s not in ("full", "all") else s for s in cols], rows)
    sub = tcon[tcon.shot == "all"]
    rows2 = [[f"{cap(NAMES[r.contrast.split(' vs ')[0]])} vs all features", f"{r.median_ratio:.3f}",
              f"{r.median_diff_seconds:+.2f}", f"{r.wilcoxon_p:.3f}", f"{r.holm_p:.3f}",
              f"{r.cliffs_delta:+.2f} ({r.magnitude})"] for _, r in sub.iterrows()]
    tbl2 = md_table(["Contrast (all levels pooled)", "Median ratio", "Median difference (s)",
                     "Wilcoxon p", "Holm p", "Cliff's delta"], rows2)
    n_sig = int(sub.significant_05.sum())
    n_data = int(sub.n_datasets.max()) if len(sub) else 0
    tsel = read(A / "selector_timing.csv")
    tref = read(A / "selector_timing_reference_cells.csv")
    if tsel is not None:
        per = tsel.groupby(["dataset", "control"]).seconds.median().unstack()
        sels = [c for c in ("corr", "mi", "pca", "rf", "llm") if c in per.columns]
        rows_n = tsel.groupby("dataset").n_train_rows.first()
        rows3 = [[tabular.DATASETS[d].label, str(int(rows_n[d]))] + [f"{per.loc[d, c]:.3f}" for c in sels]
                 for d in per.index if d in tabular.DATASETS]
        rows3.append(["Median", ""] + [f"{per[c].median():.3f}" for c in sels])
        sel_tbl = md_table(["Dataset", "Training rows"] + [cap(NAMES[c]) for c in sels], rows3)
        ref = (f" One GE search (all features, k = 16) took a median of {tref.ge_cell_seconds.median():.1f} s "
               "on the same machine." if tref is not None else "")
        sel_txt = sel_tbl + f"\n\nSeconds per split, median over the splits.{ref}"
    else:
        sel_txt = "The selectors' own fitting cost was not timed (`time_selectors.py` has not been run)."
    return f"""## Computational cost (all datasets)

Wall-clock seconds per GE cell (selection plus search) from the run log,
median across the {n_data} datasets of per-dataset medians over draws. The
all-features run gives the grammar every feature; each selection method
gives it K.

{tbl}

Each selection method is paired with the all-features run within the same
(dataset, shot, draw) cell, which cancels the shared machine load:
per-dataset median ratio, Wilcoxon over the datasets, Holm across the
contrasts. Ratios below 1 mean the selection method was faster.

{tbl2}

{n_sig} of {len(sub)} contrasts are significant. Figures: `fig_compute_time`,
`fig_compute_time_vs_shot`, `fig_compute_time_ratio`.

The selectors' own fitting cost is paid once per split (it is cached across
the shot levels of a draw). Measured separately with `time_selectors.py`:

{sel_txt}
"""


def cross_section():
    ct = read(A / "cross_task_ranks.csv")
    if ct is None:
        return ""
    return f"""## Cross-task ranks (exploratory)

Friedman on within-dataset ranks over every dataset of either task type,
five selection methods, few-shot levels. Ranks are metric-free, so the
blocks pool; the test is exploratory and reported for the ordering.

{fried_tbl(ct, PRIMARY_CONTROLS)}
"""


def contact_section():
    ct = read(A / "contact_memar_table1.csv")
    if ct is None or ct.empty:
        return ""
    rows = []
    for _, r in ct.iterrows():
        rows.append([tabular.DATASETS[r.dataset].label, r.our_level,
                     f"{r.ours_none:.3f}", f"{r.ours_best_classical:.3f}", f"{r.ours_llm:.3f}",
                     f"{r.get('memar_No Selection', float('nan')):.3f}",
                     f"{r.get('memar_XGB Importance', float('nan')):.3f}",
                     f"{r.get('memar_Zero-shot Llama-70B', float('nan')):.3f}",
                     f"{r.get('memar_RL-GRPO v1', float('nan')):.3f}"])
    tbl = md_table(["Dataset", "Our level", "Ours: all features", "Ours: best classical", "Ours: LLM prior",
                    "Memar: no selection", "Memar: XGB top-k", "Memar: Llama-70B", "Memar: RL-GRPO v1"], rows)
    return f"""## Point of contact with Memar et al. (2026), Table I

The datasets both studies use, our per-dataset means at k = 64 and at the
capped full level beside their reported means (XGBoost on a 60% training
split, five partitions). The protocols differ in learner, training-set size
and split, so the table locates the two studies relative to one another and
is not a comparison of methods.

{tbl}
"""


def floor_section():
    fc = read(A / "r2_floor_counts.csv")
    if fc is None or fc.empty:
        return ""
    tot = fc.groupby("dataset")[["cells", "floored"]].sum()
    rows = [[tabular.DATASETS[d].label, str(int(r.cells)), str(int(r.floored)),
             f"{100 * r.floored / r.cells:.1f}"] for d, r in tot.iterrows()]
    return f"""## Regression cells at the R² floor

The secondary calibrated test R² is floored at {config.REG_R2_FLOOR} (README). Cells per
regression dataset whose single rule or ensemble hit the floor, all methods
pooled; the per-method table is `r2_floor_counts.csv`. The primary regression
metric, Spearman ρ, has no floor.

{md_table(["Dataset", "Cells", "Floored", "%"], rows)}
"""


def main():
    agree_path = A / "reference_agreement.csv"
    agree = read(agree_path)
    if agree is not None and len(agree):
        lines = []
        for _, r in agree.iterrows():
            scope = ("the bundled matched-design reference on the FeatLLM eleven" if r.comparison == "reference"
                     else "the other campaign")
            lines.append(f"Against {scope} ({r.controls_compared.replace('|', ', ')}): "
                         f"{int(r.identical_score)} of {int(r.shared_cells)} shared cells have an identical "
                         f"score (maximum absolute difference {r.max_abs_diff:.1e}) and "
                         f"{int(r.identical_phenotypes)} an identical evolved phenotype.")
        check = " ".join(lines)
    else:
        check = "The run has not been compared with the bundled reference (`check_reference.py`)."
    if config.FEWSHOT_SELECTOR_ROWS == "train":
        design = ("Design: the four data-driven selectors (correlation, mutual information, PCA "
                  "loadings and random-forest importance) score the whole training partition of "
                  "each split; the LLM prior sees neither rows nor labels; the random subset is the "
                  "semantic floor; the GE learner trains on the k labelled rows only. Because the "
                  "data-driven subsets are fixed per split, any trend with k in the contrasts "
                  "reflects the learner, not the selectors.")
    else:
        design = ("Design (matched): the selectors are restricted to the same k labelled rows the "
                  "learner receives; the LLM prior sees neither rows nor labels; the random subset "
                  "is the semantic floor. The trend with k mixes the learner's and the selectors' "
                  "gains from additional labelled rows.")
    dt = read(A / "dataset_table.csv")
    ds_tbl = ""
    if dt is not None:
        rows = [[r.label, r.collection, r.task, r.semantic, r.memar_role, str(int(r.samples)),
                 str(int(r.features)), str(int(r.budget)), r.shots] for _, r in dt.iterrows()]
        ds_tbl = md_table(["Dataset", "Collection", "Task", "Names", "Role in Memar et al.", "Rows",
                           "Features", "K", "Shot levels"], rows)
    sections = [family_section(f) for f in config.FAMILIES]
    text = f"""# Results

Generated from the executed outputs in `{config.RESULTS_DIR.relative_to(config.ROOT)}` by
`report_selectors.py` after `analyse_selectors.py`. The design, the analysis
plan and the deviations from it are described in `README.md`. Classification
scores are test ROC-AUC and regression scores test R², both multiplied by
100 in the tables. The primary summary is the median across datasets of the
per-dataset mean over draws, for the single evolved rule.

{design}

## Reproducibility

{check}

## Datasets

{ds_tbl}

{chr(10).join(sections)}
{cross_section()}
{floor_section()}
{compute_section()}
{contact_section()}
## Notes

The ANOVA F statistic ranks binary tasks identically to |r|, so no F-test
method is included. PCA never consults the labels; its k - 1 component limit
and the small-k degeneracy of the correlation score bind only when the
selectors score the k labelled rows (the matched design). The NHANES label is
a threshold on a retained feature (RIDAGEYR), so every selector that keeps it
reaches near-perfect AUC. The full level (training rows capped at 1000) lies
outside the FeatLLM table and serves as a reference point rather than a
benchmark claim. A shot level runs on a multiclass dataset only when every
class can receive a labelled row, so N varies by level and is given in every
table.
"""
    (A / "report.md").write_text(text, encoding="utf-8")
    print(f"-> {A / 'report.md'}")


if __name__ == "__main__":
    main()
