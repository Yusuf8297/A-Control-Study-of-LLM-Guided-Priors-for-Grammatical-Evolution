"""Search-budget sensitivity -> results/sensitivity/analysis/ and figures/sensitivity/.

Pairs every hard-tier cell of the campaign (40 generations) with the same
cell re-run at 80 generations (same draw, subset, grammar and search seed;
generations 0-40 identical). Three questions, in order:

  1. did the extra search move the training fitness, i.e. was the budget
     binding at all;
  2. did it move the test AUC, and by how much relative to the draw-to-draw
     spread;
  3. do the confirmatory llm-vs-selector contrasts and the five-method
     omnibus read the same on the hard tier under both budgets.

The hard tier has seven datasets, so a blocked Wilcoxon cannot go below
p = 1/64 two-sided; the contrasts are reported for the direction and size of
the differences rather than for a significance decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, tabular  # noqa: E402
from geprior.statistics import cliffs_delta, holm_correction, interpret_delta  # noqa: E402

KEYS = ["dataset", "shot", "control", "seed"]
CONTROLS = [c for c in config.FEWSHOT_CONTROLS if c != "random"]   # the sensitivity run excluded it
SELECTORS = ["corr", "mi", "pca", "rf"]
PRIMARY = list(config.FEWSHOT_PRIMARY_CONTROLS)
SHOTS = list(config.SENSITIVITY_SHOTS)
HARD = [k for k, t in config.SEARCH_TIER.items() if t == "hard"]
OUT = config.SENSITIVITY_DIR / "analysis"
NEMENYI_Q05 = {5: 2.728, 6: 2.850, 7: 2.949}


def load() -> pd.DataFrame:
    base = pd.read_csv(config.RESULTS_DIR / "fewshot.csv", dtype={"shot": str})
    ext = pd.read_csv(config.SENSITIVITY_DIR / "fewshot.csv", dtype={"shot": str})
    base = base[base.dataset.isin(HARD) & base.shot.isin(SHOTS) & base.control.isin(CONTROLS)]
    ext = ext[ext.control.isin(CONTROLS)]   # reported methods only
    cols = KEYS + ["test_roc_auc", "test_roc_auc_ensemble", "train_roc_auc",
                   "effective_length", "phenotype"]
    paired = base[cols].merge(ext[cols], on=KEYS, suffixes=("_40", "_80"), how="inner")
    # keep draws where every control ran at both budgets so contrasts stay paired
    n = paired.groupby(["dataset", "shot", "seed"]).control.nunique()
    ok = n[n == len(CONTROLS)].index
    paired = paired.set_index(["dataset", "shot", "seed"]).loc[ok].reset_index()
    paired["d_test"] = paired.test_roc_auc_80 - paired.test_roc_auc_40
    paired["d_train"] = paired.train_roc_auc_80 - paired.train_roc_auc_40
    paired["d_len"] = paired.effective_length_80 - paired.effective_length_40
    paired["changed"] = paired.phenotype_80 != paired.phenotype_40
    return paired


def _p(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def budget_effect(p: pd.DataFrame):
    """Paired 40 -> 80 change per (dataset, shot) over the 70 cells, and per
    (shot, control) pooled over the hard tier."""
    rows = []
    for (ds, shot), g in p.groupby(["dataset", "shot"]):
        rows.append(dict(dataset=ds, shot=shot, n_cells=len(g),
                         d_train_mean_pts=g.d_train.mean() * 100,
                         d_test_mean_pts=g.d_test.mean() * 100,
                         d_test_median_pts=g.d_test.median() * 100,
                         d_test_sd_pts=g.d_test.std() * 100,
                         draw_sd_pts=g.test_roc_auc_40.std() * 100,
                         frac_phenotype_changed=g.changed.mean(),
                         frac_train_improved=(g.d_train > 1e-9).mean(),
                         wilcoxon_p=_p(g.test_roc_auc_80, g.test_roc_auc_40),
                         cliffs_delta=cliffs_delta(g.test_roc_auc_80, g.test_roc_auc_40)))
    out = pd.DataFrame(rows)
    out["holm_p"] = holm_correction(out.wilcoxon_p)
    out["significant_05"] = out.holm_p < config.ALPHA
    out.to_csv(OUT / "budget_effect_per_dataset.csv", index=False)
    pooled = (p.groupby(["shot", "control"])
                .agg(d_train_mean_pts=("d_train", lambda s: s.mean() * 100),
                     d_test_mean_pts=("d_test", lambda s: s.mean() * 100),
                     d_test_median_pts=("d_test", lambda s: s.median() * 100),
                     frac_phenotype_changed=("changed", "mean"), n_cells=("seed", "size"))
                .reset_index())
    pooled.to_csv(OUT / "budget_effect_pooled.csv", index=False)
    return out, pooled


def _blocks(p: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return (p.groupby(["dataset", "shot", "control"], as_index=False)
              .agg(auc=(f"test_roc_auc_{suffix}", "mean")))


def contrasts_both_budgets(p: pd.DataFrame):
    """The confirmatory llm-vs-selector contrasts, hard tier only, under
    each budget; Holm across the four contrasts within a shot and budget."""
    rows = []
    for suffix in ("40", "80"):
        blocks = _blocks(p, suffix)
        for shot in SHOTS:
            sub = blocks[blocks.shot == shot].pivot(index="dataset", columns="control", values="auc")
            raw, pv = [], []
            for c in SELECTORS:
                a, b = sub["llm"].to_numpy(), sub[c].to_numpy()
                pv.append(_p(a, b))
                raw.append(dict(generations=int(suffix), shot=shot, contrast=f"llm vs {c}",
                                n_datasets=len(sub), median_diff_pts=np.median(a - b) * 100,
                                mean_diff_pts=np.mean(a - b) * 100, wilcoxon_p=pv[-1],
                                cliffs_delta=cliffs_delta(a, b)))
            for r, ap in zip(raw, holm_correction(pv)):
                r.update(holm_p=float(ap), significant_05=bool(ap < config.ALPHA),
                         magnitude=interpret_delta(r["cliffs_delta"]))
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "confirmatory_hard_tier_both_budgets.csv", index=False)
    return out


def omnibus_both_budgets(p: pd.DataFrame):
    """Friedman over the five selection methods, hard tier, per shot and
    budget; Nemenyi critical difference for five methods; Spearman agreement
    of the mean ranks."""
    rows = []
    k, q = len(PRIMARY), NEMENYI_Q05[len(PRIMARY)]
    for shot in SHOTS:
        ranks = {}
        for suffix in ("40", "80"):
            blocks = _blocks(p, suffix)
            sub = (blocks[blocks.shot == shot]
                   .pivot(index="dataset", columns="control", values="auc")[PRIMARY])
            n = len(sub)
            stat, pval = friedmanchisquare(*[sub[c].to_numpy() for c in PRIMARY])
            r = sub.rank(axis=1, ascending=False).mean()
            ranks[suffix] = r
            row = dict(generations=int(suffix), shot=shot, n_datasets=n, friedman_chi2=stat,
                       friedman_p=pval, nemenyi_cd=q * np.sqrt(k * (k + 1) / (6.0 * n)), k=k)
            row.update({f"rank_{c}": float(r[c]) for c in PRIMARY})
            rows.append(row)
        rho = spearmanr(ranks["40"][PRIMARY], ranks["80"][PRIMARY]).statistic
        for row in rows[-2:]:
            row["rank_spearman_40_vs_80"] = float(rho)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "friedman_hard_tier_both_budgets.csv", index=False)
    return out


def report(p, effect, pooled, contrasts, omni) -> None:
    labels = {k: tabular.FEATLLM_DATASETS[k].label for k in HARD}
    names = {"none": "All features", "corr": "Correlation", "mi": "Mutual information",
             "pca": "PCA loadings", "rf": "RF importance", "llm": "LLM prior"}
    L = []
    L.append("# Search-budget sensitivity analysis\n")
    L.append("Generated by `analyse_budget_sensitivity.py` from `results/fewshot.csv` "
             "(40 generations) and `results/sensitivity/fewshot.csv` (80 generations). "
             "Hard tier: " + ", ".join(labels[k] for k in sorted(HARD, key=labels.get)) +
             "; levels " + ", ".join(SHOTS) + f"; all {len(CONTROLS)} reported methods; ten "
             "draws. Every cell is paired with its 40-generation counterpart (same draw, "
             "subset, grammar and search seed, so generations 0-40 coincide). AUC in points "
             "(x 100).\n")
    L.append(f"Cells paired: {len(p)}.\n")
    L.append("## Effect of the extra search\n")
    L.append("Mean change in training fitness (ROC-AUC on the k labelled rows) and in test "
             "AUC over the 70 cells of each (dataset, level); the draw SD is the spread of "
             "the 40-generation test AUC over the same cells, for scale. Wilcoxon over the "
             "70 paired cells, Holm across the 21 tests.\n")
    t = effect.copy()
    t["dataset"] = t.dataset.map(labels)
    for c in ("d_train_mean_pts", "d_test_mean_pts", "d_test_median_pts", "draw_sd_pts"):
        t[c] = t[c].round(2)
    t["frac_phenotype_changed"] = t.frac_phenotype_changed.round(2)
    t["holm_p"] = t.holm_p.map(lambda v: f"{v:.3f}")
    t = t.rename(columns={"dataset": "Dataset", "shot": "Level",
                          "d_train_mean_pts": "Training AUC change (mean)",
                          "d_test_mean_pts": "Test AUC change (mean)",
                          "d_test_median_pts": "Test AUC change (median)",
                          "draw_sd_pts": "Draw SD", "frac_phenotype_changed": "Rule changed",
                          "holm_p": "Holm p", "significant_05": "Significant"})
    L.append(t[["Dataset", "Level", "Training AUC change (mean)", "Test AUC change (mean)",
                "Test AUC change (median)", "Draw SD", "Rule changed", "Holm p", "Significant"]]
             .to_markdown(index=False))
    L.append("\n### Pooled over the hard tier, per method\n")
    pt = pooled.pivot(index="control", columns="shot", values="d_test_mean_pts")[SHOTS].round(2)
    pt.index = [names[c] for c in pt.index]; pt.index.name = "Method"
    L.append("Mean change in test AUC (points):\n")
    L.append(pt.loc[[names[c] for c in CONTROLS]].to_markdown())
    pt = pooled.pivot(index="control", columns="shot", values="d_train_mean_pts")[SHOTS].round(2)
    pt.index = [names[c] for c in pt.index]; pt.index.name = "Method"
    L.append("\nMean change in training fitness (points):\n")
    L.append(pt.loc[[names[c] for c in CONTROLS]].to_markdown())
    L.append("\n## Confirmatory contrasts on the hard tier under both budgets\n")
    L.append("Blocked on dataset (n = 7), two-sided Wilcoxon, Holm across the four contrasts "
             "within a level and budget. With seven blocks the smallest attainable two-sided "
             "p is 0.016, so the differences rather than the significance decisions are the "
             "quantity of interest. Positive favours the LLM prior.\n")
    c = contrasts.copy()
    c["median_diff_pts"] = c.median_diff_pts.round(1)
    c["cliffs_delta"] = c.cliffs_delta.round(2)
    c["holm_p"] = c.holm_p.map(lambda v: f"{v:.3f}")
    c["contrast"] = c.contrast.map(lambda s: "LLM prior vs " + names[s.split(" vs ")[1]].lower())
    wide = c.pivot(index=["shot", "contrast"], columns="generations",
                   values=["median_diff_pts", "holm_p", "cliffs_delta"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()[["shot", "contrast", "median_diff_pts_40", "median_diff_pts_80",
                               "cliffs_delta_40", "cliffs_delta_80", "holm_p_40", "holm_p_80"]]
    wide["shot"] = pd.Categorical(wide.shot, SHOTS)
    wide = wide.rename(columns={"shot": "Level", "contrast": "Contrast",
                                "median_diff_pts_40": "Median diff., 40 gen.",
                                "median_diff_pts_80": "Median diff., 80 gen.",
                                "cliffs_delta_40": "Cliff's delta, 40 gen.",
                                "cliffs_delta_80": "Cliff's delta, 80 gen.",
                                "holm_p_40": "Holm p, 40 gen.", "holm_p_80": "Holm p, 80 gen."})
    L.append(wide.sort_values(["Level", "Contrast"]).to_markdown(index=False))
    L.append("\n## Omnibus over the five selection methods on the hard tier\n")
    o = omni.copy()
    for col in [f"rank_{c}" for c in PRIMARY] + ["friedman_chi2", "nemenyi_cd",
                                                  "rank_spearman_40_vs_80"]:
        o[col] = o[col].round(2)
    o["friedman_p"] = o.friedman_p.map(lambda v: f"{v:.3f}")
    o = o.rename(columns={"shot": "Level", "generations": "Generations",
                          "friedman_chi2": "Friedman chi-squared", "friedman_p": "p",
                          "nemenyi_cd": "CD", "rank_spearman_40_vs_80": "Spearman rho (40 vs 80)",
                          **{f"rank_{c}": names[c] for c in PRIMARY}})
    L.append(o[["Level", "Generations", "Friedman chi-squared", "p"] +
               [names[c] for c in PRIMARY] + ["CD", "Spearman rho (40 vs 80)"]]
             .to_markdown(index=False))
    L.append("\n## Summary\n")
    worst = effect.loc[effect.d_test_mean_pts.abs().idxmax()]
    flips = contrasts.pivot(index=["shot", "contrast"], columns="generations",
                            values="significant_05")
    n_flip = int((flips[40] != flips[80]).sum())
    n_pos = int((effect.d_train_mean_pts > 0).sum())
    L.append(f"Doubling the generation budget raised the mean training fitness in {n_pos} of "
             f"{len(effect)} (dataset, level) cells (mean {p.d_train.mean() * 100:+.2f} points over "
             f"all paired cells) and changed the evolved rule in {p.changed.mean():.0%} of cells, "
             f"so the budget was binding for the search. The effect on test AUC was small: mean "
             f"{p.d_test.mean() * 100:+.2f} points over all paired cells, with the largest mean "
             f"change in any (dataset, level) being {labels[worst.dataset]} at "
             f"{'k = ' + worst.shot if worst.shot != 'full' else 'the full level'} "
             f"({worst.d_test_mean_pts:+.2f} points against a draw SD of {worst.draw_sd_pts:.1f}). "
             f"The confirmatory contrasts reached the same decision under both budgets in "
             f"{len(flips) - n_flip} of {len(flips)} cases.")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  {OUT / 'report.md'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = load()
    print(f"paired cells: {len(p)}")
    effect, pooled = budget_effect(p)
    contrasts = contrasts_both_budgets(p)
    omni = omnibus_both_budgets(p)
    p.drop(columns=["phenotype_40", "phenotype_80"]).to_csv(OUT / "paired_cells.csv", index=False)
    report(p, effect, pooled, contrasts, omni)


if __name__ == "__main__":
    main()
