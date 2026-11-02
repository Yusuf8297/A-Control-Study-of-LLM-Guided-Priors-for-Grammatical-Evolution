"""The two designs side by side -> results/comparison/.

Reads the analysis tables of both campaigns (results/analysis and
results/matched/analysis) and writes, per family, one table with the two
designs as column groups: pooled medians, the confirmatory contrasts, the
semantic floor and the Friedman mean ranks; plus a one-page markdown
summary. Nothing is recomputed; the numbers are the ones each campaign's
analyse_selectors.py wrote.

Usage:  python compare_designs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config  # noqa: E402

ROOT = config.ROOT
DESIGNS = {"full_information": ROOT / "results" / "analysis",
           "matched": ROOT / "results" / "matched" / "analysis"}
OUT = ROOT / "results" / "comparison"
CONTROLS = list(config.FEWSHOT_CONTROLS)
PRIMARY = list(config.FEWSHOT_PRIMARY_CONTROLS)
SHOTS = list(config.ALL_SHOTS)
NAMES = {"none": "All features", "random": "Random subset", "corr": "Correlation",
         "mi": "Mutual information", "pca": "PCA loadings", "rf": "RF importance", "llm": "LLM prior"}


def read(design, family, name):
    p = DESIGNS[design] / family / name
    return pd.read_csv(p, dtype={"shot": str}) if p.exists() else None


def pooled(family):
    frames = []
    for d in DESIGNS:
        t = read(d, family, "pooled_summary.csv")
        if t is None:
            continue
        w = t.pivot(index="control", columns="shot", values="pooled_median").reindex(CONTROLS)
        w = (w[[s for s in SHOTS if s in w.columns]] * 100).round(1)
        w.columns = pd.MultiIndex.from_product([[d], w.columns])
        frames.append(w)
    out = pd.concat(frames, axis=1)
    out.index = out.index.map(NAMES)
    return out


def contrasts(family, name):
    frames = []
    for d in DESIGNS:
        t = read(d, family, name)
        if t is None:
            continue
        t = t[["shot", "contrast", "n_datasets", "median_diff_pts", "holm_p", "cliffs_delta", "significant_05"]]
        t = t.rename(columns={c: f"{d}:{c}" for c in t.columns if c not in ("shot", "contrast")})
        frames.append(t.set_index(["shot", "contrast"]))
    return pd.concat(frames, axis=1).reset_index()


def ranks(family):
    frames = []
    for d in DESIGNS:
        t = read(d, family, "friedman_nemenyi_primary.csv")
        if t is None:
            continue
        t = t[["shot", "n_datasets", "friedman_p"] + [f"rank_{a}" for a in PRIMARY]].round(3)
        t = t.rename(columns={c: f"{d}:{c}" for c in t.columns if c != "shot"})
        frames.append(t.set_index("shot"))
    return pd.concat(frames, axis=1).reset_index()


def stability(family):
    frames = []
    for d in DESIGNS:
        t = read(d, family, "stability_pooled.csv")
        if t is None:
            continue
        w = t.pivot(index="control", columns="shot", values="stability").reindex(CONTROLS)
        w = w[[s for s in SHOTS if s in w.columns]].round(2)
        w.columns = pd.MultiIndex.from_product([[d], w.columns])
        frames.append(w)
    out = pd.concat(frames, axis=1)
    out.index = out.index.map(NAMES)
    return out


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# The two designs side by side", "",
             "Full-information: the data-driven selectors score the whole training partition of the "
             "split. Matched: they score only the k labelled rows the learner sees. The LLM prior, the "
             "random subset and the all-features baseline are identical in both (verified cell for cell "
             "by check_reference.py). Numbers are read from each campaign's analysis tables.", ""]
    for family in config.FAMILIES:
        if not all((DESIGNS[d] / family / "family.csv").exists() for d in DESIGNS):
            continue
        info = pd.read_csv(DESIGNS["full_information"] / family / "family.csv").iloc[0]
        metric = config.METRIC_LABEL[info.task]
        p = pooled(family)
        p.to_csv(OUT / f"{family}_pooled.csv")
        c = contrasts(family, "confirmatory_wilcoxon.csv")
        c.to_csv(OUT / f"{family}_confirmatory.csv", index=False)
        f = contrasts(family, "semantic_floor_wilcoxon.csv")
        f.to_csv(OUT / f"{family}_semantic_floor.csv", index=False)
        r = ranks(family)
        r.to_csv(OUT / f"{family}_friedman_ranks.csv", index=False)
        st = stability(family)
        st.to_csv(OUT / f"{family}_stability.csv")
        stf = st.copy()
        stf.columns = [f"{d[:4]} k={s}" if s != "full" else f"{d[:4]} full" for d, s in stf.columns]
        flat = p.copy()
        flat.columns = [f"{d[:4]} k={s}" if s != "full" else f"{d[:4]} full" for d, s in flat.columns]
        lines += [f"## {family} ({int(info.n_datasets)} datasets, test {metric} x 100, role: {info.role})", "",
                  "Pooled medians (full = full-information, matc = matched):", "",
                  flat.reset_index().rename(columns={"control": "Method"}).to_markdown(index=False, floatfmt=".1f"), "",
                  "Confirmatory contrasts, LLM prior against each selector (median difference, Holm p, Cliff's delta, significant):", "",
                  md(c), "", "Semantic floor, LLM prior against the random subset:", "", md(f), "",
                  "Friedman mean ranks over the five selection methods (1 = best):", "", md(r), "",
                  "Subset stability, mean pairwise Jaccard across draws:", "",
                  stf.reset_index().rename(columns={"control": "Method"}).to_markdown(index=False, floatfmt=".2f"), ""]
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
