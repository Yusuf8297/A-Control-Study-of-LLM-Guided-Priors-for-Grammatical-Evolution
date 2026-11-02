"""Per-generation convergence harvest for the selector comparison.

The campaign keeps only final results (results/fewshot.csv); the GRAPE
logbook is discarded inside engine.search_fewshot. This re-runs chosen
cells with the campaign's exact Cell construction, subsets, grammar and
search seeds, and records the logbook's min fitness per generation so
best-so-far curves can be plotted. Multiclass datasets average the
best-so-far curves of the one-versus-rest rules, matching the macro
averaging of the reported AUC.

Resumable: a harvested (control, seed) cell is skipped on re-run.

Usage:
    python harvest_convergence.py heart --shot 64
    python harvest_convergence.py heart --shot 64 --controls corr mi pca rf llm

Writes results/convergence/<dataset>__shot<shot>.csv, one row per
(control, seed, gen).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geprior import config, engine, fewshot, grammar, tabular  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "convergence"


def out_path(dataset: str, shot: str) -> Path:
    return OUT_DIR / f"{dataset}__shot{shot}.csv"


def harvest_cell(enc, shot: str, seed: int, control: str) -> list[dict]:
    """One campaign cell, campaign seeds; returns best-so-far rows per gen."""
    cell = fewshot.Cell(enc, shot, seed)
    features = fewshot.control_features(cell, control)
    cols = enc.group_columns(features)
    bnf = grammar.build_grammar(cols, tag=f"fs_{enc.key}_{control}")
    search_seed = config.SEARCH_SEED_OFFSET + seed
    targets = [1] if enc.n_classes == 2 else list(range(enc.n_classes))
    curves = []
    for c in targets:
        y_c = (cell.y_fit == c).astype(int)
        _, _, logbook = engine.evolve(bnf, cell.X_fit, y_c, seed=search_seed + 10 * c)
        gens = list(logbook.select("gen"))
        mins = np.asarray(logbook.select("min"), dtype=float)
        curves.append(np.minimum.accumulate(mins))
    best = np.mean(curves, axis=0)
    return [dict(dataset=enc.key, shot=shot, control=control, seed=seed,
                 gen=int(g), best_so_far=float(b)) for g, b in zip(gens, best)]


def _run_control(dataset: str, shot: str, control: str, seeds: list[int]) -> pd.DataFrame:
    """Worker: all draws of one method, sequential (one grammar file each)."""
    enc = tabular.load(dataset)
    rows = []
    for seed in seeds:
        t0 = time.time()
        rows.extend(harvest_cell(enc, shot, seed, control))
        print(f"  {dataset} shot={shot} {control} seed={seed} ({time.time() - t0:.0f}s)",
              flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=tabular.keys(task_type="classification"))
    ap.add_argument("--shot", default="64", choices=[*map(str, config.FEWSHOT_SHOTS), "full"])
    ap.add_argument("--controls", nargs="*", default=list(config.FEWSHOT_CONTROLS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path(args.dataset, args.shot)
    done = pd.read_csv(path, dtype={"shot": str}) if path.exists() else pd.DataFrame()
    done_cells = (set(map(tuple, done[["control", "seed"]].astype(str).values))
                  if not done.empty else set())

    seeds = list(range(config.FEWSHOT_DRAWS))
    jobs = {c: [s for s in seeds if (c, str(s)) not in done_cells]
            for c in args.controls}
    jobs = {c: s for c, s in jobs.items() if s}
    if not jobs:
        print(f"nothing to do; {path.relative_to(config.ROOT)} is complete")
        return

    frames = [done] if not done.empty else []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_control, args.dataset, args.shot, c, s): c
                   for c, s in jobs.items()}
        for future in futures:
            frames.append(future.result())
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"wrote {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
