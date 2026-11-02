"""Compare a local campaign against the bundled reference run, and the two
campaigns against each other on the design-invariant methods.

The reference run (2 October 2026) records the matched design on the eleven
FeatLLM datasets: selectors restricted to the k labelled rows. The
all-features baseline, the LLM prior and the random subset do not depend on
that choice, so under the full-information design those methods must still
reproduce the reference cell for cell, which is the protocol check, while
corr, mi, pca and rf differ from the reference by design and are left out.
Under "shots" every reported method is compared. The thirteen benchmark
datasets added in October 2026 have no bundled reference; for them the
check is the invariance of none/llm/random between the two campaigns, which
is reported whenever the other campaign's file exists. Writes the agreement
summary that report_selectors.py quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, fewshot  # noqa: E402

KEYS = ["dataset", "shot", "control", "seed"]
INVARIANT = tuple(a for a in ("none", "random", "llm") if a in config.FEWSHOT_CONTROLS)
REFERENCE = config.REFERENCE_RUN
LOCAL = config.RESULTS_DIR / "fewshot.csv"
OTHER = config.OTHER_RESULTS_DIR / "fewshot.csv"


def compare(new, ref, label):
    m = new.merge(ref, on=KEYS, suffixes=("_new", "_ref"))
    if m.empty:
        print(f"{label}: no shared cells")
        return None
    diff = (m.test_score_new - m.test_score_ref).abs()
    pheno = (m.phenotype_new == m.phenotype_ref)
    print(f"{label}:")
    print(f"  cells compared:       {len(m)} (local {len(new)}, other {len(ref)})")
    print(f"  identical score:      {(diff < 1e-12).sum()} / {len(m)} (max |diff| {diff.max():.2e})")
    print(f"  identical phenotypes: {pheno.sum()} / {len(m)}")
    if not pheno.all():
        print(m[~pheno][KEYS].head(20).to_string(index=False))
    return dict(shared_cells=len(m), identical_score=int((diff < 1e-12).sum()),
                max_abs_diff=float(diff.max()), identical_phenotypes=int(pheno.sum()))


def main():
    if not LOCAL.exists():
        raise SystemExit(f"no local campaign at {LOCAL.relative_to(config.ROOT)}; run the campaign first")
    if config.SMOKE:
        print("smoke run: reduced search, so agreement with the reference is not expected; "
              "the comparison below only exercises the code path")
    new = fewshot._load(LOCAL).drop_duplicates(subset=KEYS)
    ref = fewshot._load(REFERENCE).drop_duplicates(subset=KEYS)
    rows = []
    if config.FEWSHOT_SELECTOR_ROWS == "train":
        compared = "|".join(INVARIANT)
        n, r = new[new.control.isin(INVARIANT)], ref[ref.control.isin(INVARIANT)]
        print("full-information design: comparing the design-invariant methods "
              f"({', '.join(INVARIANT)}) with the matched-design reference; corr/mi/pca/rf "
              "differ from it by design and are not compared")
    else:
        compared = "all"
        n = new[new.control.isin(config.FEWSHOT_CONTROLS)]
        r = ref[ref.control.isin(config.FEWSHOT_CONTROLS)]
        shared = n.merge(r, on=KEYS, suffixes=("_new", "_ref"))
        if len(shared) == len(n) and (shared.phenotype_new == shared.phenotype_ref).all():
            print(f"{LOCAL.relative_to(config.ROOT)} holds exactly the reference run's cells with "
                  "identical phenotypes, i.e. the seeded copy, not a local run of those datasets.")
    res = compare(n, r, "against the bundled reference (FeatLLM eleven)")
    if res:
        rows.append(dict(comparison="reference", controls_compared=compared, **res))
    if OTHER.exists():
        other = fewshot._load(OTHER).drop_duplicates(subset=KEYS)
        res = compare(new[new.control.isin(INVARIANT)], other[other.control.isin(INVARIANT)],
                      f"against the other campaign ({'|'.join(INVARIANT)}, all datasets)")
        if res:
            rows.append(dict(comparison="other_campaign", controls_compared="|".join(INVARIANT), **res))
    out = config.RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "reference_agreement.csv", index=False)


if __name__ == "__main__":
    main()
