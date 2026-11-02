"""Tables and tests for the selector comparison -> results/analysis/.

The campaign is analysed in families (config.FAMILIES), each on its own
metric and its own blocks, written to results/analysis/<family>/:

  classification            every classification dataset; the primary family
  regression                every regression dataset (R^2); pre-specified as
                            descriptive: with six or fewer blocks a
                            Holm-corrected Wilcoxon over four contrasts
                            cannot fall below alpha
  classification_named      moderator: datasets whose feature names carry
                            domain meaning
  classification_anonymous  descriptive: anonymous feature names

Within a family (pre-registered, see README):
  confirmatory     at each shot level, the LLM prior against each of corr,
                   mi, pca and rf, blocked on dataset (one value per dataset:
                   the mean over draws), two-sided Wilcoxon signed-rank,
                   Holm across the four contrasts within the level, Cliff's
                   delta alongside -> confirmatory_wilcoxon.csv
  semantic floor   the LLM prior against the size-K random subset at each
                   level, Holm across the few-shot levels; the data-free
                   analogue of an anonymised-names condition at matched
                   cardinality -> semantic_floor_wilcoxon.csv
  omnibus          Friedman across the five selection methods at the five
                   few-shot levels with the Nemenyi post hoc (primary), and
                   across every reported method at all six levels (all);
                   each with its pairwise Nemenyi table
  exploratory      all-pairs Wilcoxon with Holm across every pair within a
                   level; per-dataset Wilcoxon over the paired draws (Holm
                   across levels); selection stability (mean pairwise
                   Jaccard across draws); subset overlap with the LLM prior

Across families, at the top level of results/analysis/:
  dataset_table.csv          the datasets, their roles, budgets and shot ladders
  cross_task_ranks.csv       exploratory Friedman on within-dataset ranks over
                             every dataset, ranks being metric-free
  compute_time_*.csv         wall-clock analysis over every dataset, paired
                             within the (dataset, shot, draw) cell against the
                             all-features run
  r2_floor_counts.csv        regression cells whose secondary test R^2 hit the floor
  <family>/*_secondary.csv   pooled and per-dataset summaries on the secondary
                             metric (calibrated R^2 for regression)
  contact_memar_table1.csv   our scores next to Memar et al.'s Table I on the
                             datasets the two studies share (different
                             protocols; descriptive)

The number of blocks differs by level because a shot level runs on a
multiclass dataset only when every class can receive a labelled row; every
table carries n_datasets. Medians are the primary summary; means alongside.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, fewshot, tabular  # noqa: E402
from geprior.statistics import cliffs_delta, holm_correction, interpret_delta  # noqa: E402

CONTROLS = list(config.FEWSHOT_CONTROLS)
SELECTORS = ["corr", "mi", "pca", "rf"]
SHOTS = list(config.ALL_SHOTS)
PRIMARY_CONTROLS = list(config.FEWSHOT_PRIMARY_CONTROLS)
PRIMARY_SHOTS = list(config.FEWSHOT_PRIMARY_SHOTS)
FLOOR = config.SEMANTIC_FLOOR_CONTROL
METRIC = "test_score"          # the single evolved rule; the ensemble is secondary
OUT = config.RESULTS_DIR / "analysis"

# Demsar (2006), two-tailed Nemenyi q at alpha = 0.05, indexed by k
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}
TIME_REF = config.TIME_REFERENCE_CONTROL


def paired_wilcoxon(a, b):
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", method="auto").pvalue)


def load():
    df = fewshot._load(config.RESULTS_DIR / "fewshot.csv")
    df = df[df.control.isin(CONTROLS) & df.shot.isin(SHOTS) & df.dataset.isin(tabular.DATASETS)]
    # keep only cells where every method has run so comparisons stay paired
    counts = df.groupby(["dataset", "shot", "seed"]).control.nunique()
    ok = counts[counts == len(CONTROLS)].index
    df = df.set_index(["dataset", "shot", "seed"]).loc[ok].reset_index()
    df["task_type"] = df.dataset.map(lambda k: tabular.DATASETS[k].task_type)
    df["semantic"] = df.dataset.map(lambda k: tabular.DATASETS[k].semantic)
    return df


def per_dataset_means(df):
    """One value per (dataset, shot, method): the mean over draws, the blocking unit."""
    return (df.groupby(["dataset", "shot", "control"], as_index=False)
              .agg(score=(METRIC, "mean"), score_ens=("test_score_ensemble", "mean")))


def summaries(df, out, metric=METRIC, suffix=""):
    ens = "test_score_ensemble" if metric == METRIC else f"{metric}_ensemble"
    per = (df.groupby(["dataset", "shot", "control"])
             .agg(mean_score=(metric, "mean"), median_score=(metric, "median"),
                  std_score=(metric, "std"), mean_score_ensemble=(ens, "mean"),
                  n=("seed", "size"))
             .reset_index())
    per.to_csv(out / f"summary_per_dataset{suffix}.csv", index=False)
    blocks = (df.groupby(["dataset", "shot", "control"], as_index=False)
                .agg(score=(metric, "mean"), score_ens=(ens, "mean")))
    pooled = (blocks.groupby(["shot", "control"])
                    .agg(pooled_median=("score", "median"), pooled_mean=("score", "mean"),
                         pooled_median_ens=("score_ens", "median"), n_datasets=("score", "size"))
                    .reset_index())
    pooled.to_csv(out / f"pooled_summary{suffix}.csv", index=False)


def _contrast_rows(blocks, shot, pairs):
    sub = blocks[blocks.shot == shot].pivot(index="dataset", columns="control", values="score").dropna()
    rows = []
    for a_name, b_name in pairs:
        a, b = sub[a_name].to_numpy(), sub[b_name].to_numpy()
        rows.append(dict(shot=shot, contrast=f"{a_name} vs {b_name}", n_datasets=len(sub),
                         median_diff_pts=float(np.median(a - b) * 100),
                         mean_diff_pts=float(np.mean(a - b) * 100),
                         wilcoxon_p=paired_wilcoxon(a, b), cliffs_delta=cliffs_delta(a, b)))
    return rows


def _finish(rows, pvals):
    adj = holm_correction(np.array(pvals))
    for r, ap in zip(rows, adj):
        r["holm_p"] = float(ap)
        r["significant_05"] = bool(ap < config.ALPHA)
        r["magnitude"] = interpret_delta(r["cliffs_delta"])
    return rows


def confirmatory(df, out):
    """Holm across the four selector contrasts within each level."""
    blocks = per_dataset_means(df)
    rows = []
    for shot in [s for s in SHOTS if s in set(blocks.shot)]:
        raw = _contrast_rows(blocks, shot, [("llm", c) for c in SELECTORS])
        rows += _finish(raw, [r["wilcoxon_p"] for r in raw])
    res = pd.DataFrame(rows)
    res.to_csv(out / "confirmatory_wilcoxon.csv", index=False)
    return res


def semantic_floor(df, out):
    """LLM prior against the random subset; Holm across the few-shot levels,
    the full level reported alongside with its raw p."""
    blocks = per_dataset_means(df)
    levels = [s for s in SHOTS if s in set(blocks.shot)]
    raw = [r for s in levels for r in _contrast_rows(blocks, s, [("llm", FLOOR)])]
    few = [r for r in raw if r["shot"] in PRIMARY_SHOTS]
    rest = [r for r in raw if r["shot"] not in PRIMARY_SHOTS]
    rows = _finish(few, [r["wilcoxon_p"] for r in few]) if few else []
    for r in rest:
        r.update(holm_p=r["wilcoxon_p"], significant_05=bool(r["wilcoxon_p"] < config.ALPHA),
                 magnitude=interpret_delta(r["cliffs_delta"]))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "semantic_floor_wilcoxon.csv", index=False)
    return res


def friedman_nemenyi(df, controls, shots, name, out):
    """Friedman over `controls` blocked on dataset, per shot level; Nemenyi
    critical difference for len(controls) methods and that level's N."""
    blocks = per_dataset_means(df)
    k = len(controls)
    q = NEMENYI_Q05[k]
    rows = []
    for shot in [s for s in shots if s in set(blocks.shot)]:
        sub = (blocks[blocks.shot == shot]
               .pivot(index="dataset", columns="control", values="score")[controls].dropna())
        n = len(sub)
        if n < 3:
            continue
        stat, p = friedmanchisquare(*[sub[a].to_numpy() for a in controls])
        ranks = sub.rank(axis=1, ascending=False).mean()     # rank 1 = best score
        cd = q * np.sqrt(k * (k + 1) / (6.0 * n))   # Demsar: CD = q sqrt(k(k+1)/6N)
        row = dict(shot=shot, n_datasets=n, friedman_chi2=float(stat), friedman_p=float(p),
                   nemenyi_cd=float(cd), q_alpha=q, k=k)
        row.update({f"rank_{a}": float(ranks[a]) for a in controls})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"friedman_nemenyi_{name}.csv", index=False)
    pairwise_nemenyi(res, controls, name, out)
    return res


def pairwise_nemenyi(fn, controls, name, out):
    """Every pair under the omnibus: mean-rank gap against the CD. Reads the
    Friedman table so the ranks are exactly the reported ones."""
    rows = []
    for _, r in fn.iterrows():
        for a, b in combinations(controls, 2):
            gap = r[f"rank_{a}"] - r[f"rank_{b}"]
            rows.append(dict(shot=r.shot, a=a, b=b, rank_a=r[f"rank_{a}"], rank_b=r[f"rank_{b}"],
                             rank_gap=float(abs(gap)), nemenyi_cd=r.nemenyi_cd,
                             n_datasets=r.n_datasets,
                             better=(b if gap > 0 else a) if gap != 0 else "tie",
                             significant_05=bool(abs(gap) > r.nemenyi_cd),
                             friedman_p=r.friedman_p))
    pd.DataFrame(rows).to_csv(out / f"nemenyi_pairwise_{name}.csv", index=False)


def exploratory_allpairs(df, controls, shots, name, out):
    """Exploratory: every pair of methods, Wilcoxon over the per-dataset
    means, Holm across all pairs within a shot level."""
    blocks = per_dataset_means(df)
    rows = []
    for shot in [s for s in shots if s in set(blocks.shot)]:
        raw = _contrast_rows(blocks, shot, list(combinations(controls, 2)))
        for r in raw:
            a, b = r["contrast"].split(" vs ")
            r.update(a=a, b=b, better=a if r["mean_diff_pts"] > 0 else b)
        rows += _finish(raw, [r["wilcoxon_p"] for r in raw])
    res = pd.DataFrame(rows)
    res.to_csv(out / f"exploratory_allpairs_wilcoxon_{name}.csv", index=False)
    return res


def significance_summary(conf, out, name="primary"):
    """One row per confirmatory contrast and shot, with the Nemenyi reading
    of the same pair alongside."""
    nem = pd.read_csv(out / f"nemenyi_pairwise_{name}.csv", dtype={"shot": str})
    rows = []
    for _, r in conf.iterrows():
        other = r.contrast.split(" vs ")[1]
        m = nem[(nem.shot == r.shot) & (((nem.a == "llm") & (nem.b == other)) |
                                         ((nem.b == "llm") & (nem.a == other)))]
        nem_sig = bool(m.significant_05.iloc[0]) if len(m) else None
        gap = float(m.rank_gap.iloc[0]) if len(m) else np.nan
        cd = float(m.nemenyi_cd.iloc[0]) if len(m) else np.nan
        favours = "LLM prior" if r.median_diff_pts > 0 else (other if r.median_diff_pts < 0 else "neither")
        rows.append(dict(shot=r.shot, contrast=r.contrast, n_datasets=r.n_datasets, favours=favours,
                         median_diff_pts=r.median_diff_pts, holm_p=r.holm_p,
                         confirmatory_significant=bool(r.significant_05),
                         nemenyi_rank_gap=gap, nemenyi_cd=cd, nemenyi_significant=nem_sig,
                         agree=(nem_sig is None) or (nem_sig == bool(r.significant_05)),
                         outcome=("significant" if r.significant_05 else "not significant")))
    res = pd.DataFrame(rows)
    res.to_csv(out / "significance_summary.csv", index=False)
    return res


def exploratory_perdataset(df, out):
    rows = []
    for dataset, dsub in df.groupby("dataset"):
        for control in SELECTORS + [FLOOR]:
            pvals, raw = [], []
            for shot in [s for s in SHOTS if s in set(dsub.shot)]:
                cell = dsub[dsub.shot == shot].pivot(index="seed", columns="control", values=METRIC)
                a, b = cell["llm"].to_numpy(), cell[control].to_numpy()
                p = paired_wilcoxon(a, b)
                pvals.append(p)
                raw.append(dict(dataset=dataset, shot=shot, contrast=f"llm vs {control}",
                                n_draws=len(cell), median_diff_pts=float(np.median(a - b) * 100),
                                wilcoxon_p=p, cliffs_delta=cliffs_delta(a, b)))
            adj = holm_correction(np.array(pvals))   # Holm across shot levels
            for r, ap in zip(raw, adj):
                r["holm_p"] = float(ap)
                r["significant_05"] = bool(ap < config.ALPHA)
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "exploratory_perdataset_wilcoxon.csv", index=False)


def _subsets(df):
    sub = df[["dataset", "shot", "control", "seed", "features"]].copy()
    sub["fset"] = sub.features.str.split("|").map(frozenset)
    return sub


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def stability(df, out):
    """Mean pairwise Jaccard of a method's subsets across the draws."""
    sub = _subsets(df)
    rows = []
    for (dataset, shot, control), g in sub.groupby(["dataset", "shot", "control"]):
        sets = list(g.sort_values("seed").fset)
        pairs = [_jaccard(a, b) for a, b in combinations(sets, 2)]
        rows.append(dict(dataset=dataset, shot=shot, control=control,
                         stability=float(np.mean(pairs)) if pairs else 1.0))
    res = pd.DataFrame(rows)
    res.to_csv(out / "stability_per_dataset.csv", index=False)
    (res.groupby(["shot", "control"]).stability.mean().reset_index()
        .to_csv(out / "stability_pooled.csv", index=False))


def overlap_with_llm(df, out):
    """Jaccard overlap of each method's subset with the LLM subset, same cell."""
    sub = _subsets(df)
    wide = sub.pivot_table(index=["dataset", "shot", "seed"], columns="control",
                           values="fset", aggfunc="first")
    rows = []
    for control in [a for a in CONTROLS if a not in ("llm", "none")]:
        j = [_jaccard(r[control], r["llm"]) for _, r in wide.iterrows()]
        tmp = wide.reset_index()[["dataset", "shot"]].copy()
        tmp["control"], tmp["jaccard_llm"] = control, j
        rows.append(tmp)
    res = pd.concat(rows, ignore_index=True)
    (res.groupby(["dataset", "shot", "control"]).jaccard_llm.mean().reset_index()
        .to_csv(out / "overlap_with_llm_per_dataset.csv", index=False))
    (res.groupby(["shot", "control"]).jaccard_llm.mean().reset_index()
        .to_csv(out / "overlap_with_llm_pooled.csv", index=False))


def analyse_family(df, family):
    keys = tabular.family_keys(family)
    sub = df[df.dataset.isin(keys)]
    out = OUT / family
    out.mkdir(parents=True, exist_ok=True)
    if sub.empty or sub.dataset.nunique() < 2:
        print(f"[{family}] fewer than two datasets with complete cells; skipped")
        return None
    summaries(sub, out)
    task = config.FAMILIES[family]["task"]
    if task in config.SECONDARY_METRIC and config.SECONDARY_METRIC[task][0] in sub.columns:
        summaries(sub, out, metric=config.SECONDARY_METRIC[task][0], suffix="_secondary")
    conf = confirmatory(sub, out)
    floor = semantic_floor(sub, out)
    fn_primary = friedman_nemenyi(sub, PRIMARY_CONTROLS, PRIMARY_SHOTS, "primary", out)
    friedman_nemenyi(sub, CONTROLS, SHOTS, "all", out)
    summ = significance_summary(conf, out, "primary") if len(fn_primary) else None
    exploratory_allpairs(sub, PRIMARY_CONTROLS, PRIMARY_SHOTS, "primary", out)
    exploratory_allpairs(sub, CONTROLS, SHOTS, "all", out)
    exploratory_perdataset(sub, out)
    stability(sub, out)
    overlap_with_llm(sub, out)
    pd.DataFrame([dict(family=family, role=config.FAMILIES[family]["role"],
                       task=config.FAMILIES[family]["task"],
                       n_datasets=sub.dataset.nunique(), datasets="|".join(sorted(sub.dataset.unique())),
                       metric=config.METRIC[config.FAMILIES[family]["task"]])]
                 ).to_csv(out / "family.csv", index=False)
    return dict(conf=conf, floor=floor, fn=fn_primary, summ=summ)


# ------------------------------------------------------------- cross-family

def cross_task_ranks(df):
    """Exploratory Friedman on within-dataset ranks over every dataset of
    either task; ranks are metric-free so the blocks can be pooled."""
    out = OUT
    res = friedman_nemenyi(df, PRIMARY_CONTROLS, PRIMARY_SHOTS, "cross_task_primary", out)
    friedman_nemenyi(df, CONTROLS, SHOTS, "cross_task_all", out)
    res.to_csv(out / "cross_task_ranks.csv", index=False)
    return res


def compute_time(df):
    """Wall-clock seconds per GE cell from the run log. The paired unit is
    the (dataset, shot, draw) cell: its methods ran one after another in the
    same worker, so the within-cell ratio to the all-features run cancels
    machine load. Per-dataset medians over draws are the blocking unit."""
    t = df[["dataset", "shot", "seed", "control", "seconds", "n_terminals"]].copy()
    ref = (t[t.control == TIME_REF].set_index(["dataset", "shot", "seed"]).seconds
             .rename("seconds_ref"))
    t = t.join(ref, on=["dataset", "shot", "seed"])
    t["ratio_to_ref"] = t.seconds / t.seconds_ref
    per = (t.groupby(["dataset", "shot", "control"])
             .agg(median_seconds=("seconds", "median"), mean_seconds=("seconds", "mean"),
                  std_seconds=("seconds", "std"), median_ratio=("ratio_to_ref", "median"),
                  n_terminals=("n_terminals", "median"), n=("seed", "size"))
             .reset_index())
    per.to_csv(OUT / "compute_time_per_dataset.csv", index=False)
    pooled = (per.groupby(["shot", "control"])
                .agg(pooled_median_seconds=("median_seconds", "median"),
                     pooled_mean_seconds=("mean_seconds", "mean"),
                     pooled_median_ratio=("median_ratio", "median"),
                     n_datasets=("dataset", "size"))
                .reset_index())
    overall = (per.groupby("control")
                 .agg(pooled_median_seconds=("median_seconds", "median"),
                      pooled_mean_seconds=("mean_seconds", "mean"),
                      pooled_median_ratio=("median_ratio", "median"),
                      n_datasets=("dataset", "nunique"))
                 .reset_index().assign(shot="all"))
    pd.concat([pooled, overall], ignore_index=True).to_csv(OUT / "compute_time_pooled.csv", index=False)
    rows = []
    levels = [s for s in SHOTS if s in set(per.shot)] + ["all"]
    for shot in levels:
        if shot == "all":
            sub = (t.groupby(["dataset", "control"]).seconds.median().reset_index()
                     .pivot(index="dataset", columns="control", values="seconds"))
        else:
            sub = per[per.shot == shot].pivot(index="dataset", columns="control",
                                              values="median_seconds")
        sub = sub.dropna()
        pvals, raw = [], []
        for control in [a for a in CONTROLS if a != TIME_REF]:
            a, b = sub[control].to_numpy(), sub[TIME_REF].to_numpy()
            p = paired_wilcoxon(a, b)
            pvals.append(p)
            raw.append(dict(shot=shot, contrast=f"{control} vs {TIME_REF}", n_datasets=len(sub),
                            median_ratio=float(np.median(a / b)),
                            median_diff_seconds=float(np.median(a - b)),
                            wilcoxon_p=p, cliffs_delta=cliffs_delta(a, b)))
        rows += _finish(raw, pvals)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "compute_time_contrasts.csv", index=False)
    return res


def r2_floor_counts(df):
    reg = df[df.task_type == "regression"]
    if reg.empty:
        return
    (reg.groupby(["dataset", "control"]).agg(cells=("r2_floored", "size"),
                                              floored=("r2_floored", "sum")).reset_index()
        .to_csv(OUT / "r2_floor_counts.csv", index=False))


def contact_memar(df):
    """Our per-dataset means at k = 64 and the full level beside Memar et
    al.'s Table I on the shared datasets. Protocols differ (XGBoost on a 60%
    training split there; GE on k labelled rows here), so the table is a
    point of contact, not a comparison."""
    if not tabular.MEMAR_TABLE1_PATH.exists():
        return
    memar = tabular.memar_table1()
    shared = {"adult": "adult", "sonar": "sonar", "student_dropout": "dropout",
              "ames_housing": "ames", "parkinsons": "parkinsons",
              "student_performance": "student_performance"}
    # Memar et al. report R^2 on regression, so the contact uses the secondary
    # calibrated R^2 there and ROC-AUC on classification
    df = df.copy()
    if "test_r2" in df.columns:
        df[METRIC] = np.where(df.task_type == "regression", df.test_r2, df[METRIC])
    per = (df[df.dataset.isin(shared)].groupby(["dataset", "shot", "control"])[METRIC]
             .mean().reset_index())
    rows = []
    for key, col in shared.items():
        if col not in memar.columns or key not in set(per.dataset):
            continue
        for shot in ("64", "full"):
            sub = per[(per.dataset == key) & (per.shot == shot)].set_index("control")[METRIC]
            if sub.empty:
                continue
            row = dict(dataset=key, memar_column=col, our_level=shot,
                       metric=tabular.DATASETS[key].task_type)
            row.update({f"ours_{c}": float(sub.get(c, np.nan)) for c in CONTROLS})
            classical = [c for c in SELECTORS if c in sub.index]
            row["ours_best_classical"] = float(sub[classical].max()) if classical else np.nan
            for _, m in memar.iterrows():
                row[f"memar_{m.method}"] = m[col]
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "contact_memar_table1.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    expected = 0
    for key in tabular.DATASETS:
        try:
            expected += len(tabular.load(key).available_shots()) * len(CONTROLS) * config.FEWSHOT_DRAWS
        except FileNotFoundError:
            pass
    print(f"paired cells: {len(df)} / {expected}")
    if len(df) < expected:
        print("WARNING: run incomplete; analysis covers completed, fully paired cells only")
    tabular.describe([k for k in tabular.DATASETS if k in set(df.dataset)]
                     ).to_csv(OUT / "dataset_table.csv", index=False)
    results = {}
    for family in config.FAMILIES:
        results[family] = analyse_family(df, family)
    if df.dataset.nunique() >= 3:
        cross_task_ranks(df)
    times = compute_time(df)
    r2_floor_counts(df)
    contact_memar(df)
    prim = results.get(config.PRIMARY_FAMILY)
    if prim:
        print(f"\n[{config.PRIMARY_FAMILY}] confirmatory contrasts:")
        print(prim["conf"].to_string(index=False))
        print(f"\n[{config.PRIMARY_FAMILY}] semantic floor (LLM prior vs random subset):")
        print(prim["floor"].to_string(index=False))
        print(f"\n[{config.PRIMARY_FAMILY}] primary omnibus:")
        print(prim["fn"].to_string(index=False))
        if prim["summ"] is not None:
            print("\nsignificance summary (confirmatory contrasts next to the Nemenyi reading):")
            print(prim["summ"].to_string(index=False))
    reg = results.get("regression")
    if reg:
        print("\n[regression] contrasts (descriptive family):")
        print(reg["conf"].to_string(index=False))
    print(f"\ncompute time, each method against {TIME_REF} (all datasets):")
    print(times[times.shot == "all"].to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
