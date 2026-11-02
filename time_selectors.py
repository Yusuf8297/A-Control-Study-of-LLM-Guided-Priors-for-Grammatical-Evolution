"""Selector fitting cost -> results/analysis/selector_timing.csv.

The campaign log times whole GE cells (selection + search). Under the
full-information design a data-driven selector is fitted once per draw seed
and cached across shot levels, so its fitting cost lands in the first level
visited (k = 4) only and is invisible elsewhere. This script measures that
cost on its own: for each dataset and each of the ten draw seeds, the wall
time to score the standardised training partition and rank the features,
through the campaign's own code paths (fewshot.selector_rows,
fewshot._column_scores, fewshot._ranked_subset). The LLM prior is timed as
the campaign ran it: a cached lookup, i.e. its cost is the one-off query
that produced the cache and is not a per-run cost. The all-features baseline
has no selection step and is timed as such.

For scale, the same machine also times a handful of GE cells (the
all-features baseline at k = 16, seeds 0-2, on the three datasets named in
REFERENCE_CELLS) so the selector cost can be read as a fraction of one GE
search on identical hardware. Absolute seconds belong to this machine;
ratios travel.

Usage:  python time_selectors.py            # -> results/analysis/selector_timing.csv
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, fewshot, priors, tabular  # noqa: E402

OUT = config.RESULTS_DIR / "analysis"
SELECTORS = ["corr", "mi", "pca", "rf"]
REFERENCE_CELLS = [("heart", "16"), ("adult", "16"), ("car", "16")]
REFERENCE_SEEDS = (0, 1, 2)


def time_selector(enc, seed, control):
    t0 = time.perf_counter()
    if control == "none":
        _ = list(enc.features)
    elif control == "llm":
        _ = priors.LLM.select_names(enc.key, list(enc.features), enc.budget, seed,
                                    enc.task, enc.descriptions)
    else:
        X, y = fewshot.selector_rows(enc, seed)
        _ = fewshot._ranked_subset(enc, fewshot._column_scores(control, X, y, enc.n_classes))
    return time.perf_counter() - t0


def time_reference_cells():
    rows = []
    for key, shot in REFERENCE_CELLS:
        enc = tabular.load(key)
        for seed in REFERENCE_SEEDS:
            cell = fewshot.Cell(enc, shot, seed)
            cols = enc.group_columns(list(enc.features))
            t0 = time.perf_counter()
            fewshot.ge_cell(cell, cols, tag=f"timing_{key}")
            rows.append(dict(dataset=key, shot=shot, seed=seed,
                             ge_cell_seconds=time.perf_counter() - t0))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in tabular.DATASETS:
        enc = tabular.load(key)
        n_train = len(tabular.featllm_split(enc.strata, 0)[0])
        for seed in range(config.FEWSHOT_DRAWS):
            for control in config.FEWSHOT_CONTROLS:
                rows.append(dict(dataset=key, seed=seed, control=control,
                                 n_train_rows=n_train, n_encoded_cols=enc.X.shape[1],
                                 n_features=len(enc.features), budget=enc.budget,
                                 seconds=time_selector(enc, seed, control)))
        print(f"  {key}: timed", flush=True)
    timing = pd.DataFrame(rows)
    timing["machine"] = f"{platform.machine()} {platform.system()} py{platform.python_version()}"
    timing.to_csv(OUT / "selector_timing.csv", index=False)
    print("timing reference GE cells on this machine ...", flush=True)
    ref = time_reference_cells()
    ref.to_csv(OUT / "selector_timing_reference_cells.csv", index=False)
    per = timing.groupby(["dataset", "control"]).seconds.median().unstack()[list(config.FEWSHOT_CONTROLS)]
    print(per.round(3).to_string())
    print(f"median GE cell on this machine: {ref.ge_cell_seconds.median():.1f} s "
          f"(cells: {', '.join(f'{d} k={s}' for d, s in REFERENCE_CELLS)})")
    print(f"-> {OUT / 'selector_timing.csv'}")


if __name__ == "__main__":
    main()
