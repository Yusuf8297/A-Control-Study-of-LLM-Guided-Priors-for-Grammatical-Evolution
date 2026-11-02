"""Driver for the campaign, in resumable time-bounded slices.

Usage:  python run_study.py fewshot [time_budget_seconds] [dataset,list]
        python run_study.py merge
        python run_study.py status
        python run_study.py share-invariant

Safe to re-run; completed cells are skipped. The optional dataset list lets
several processes share a campaign, each writing its own part file under
results/parts/; `merge` folds the parts into results/fewshot.csv.
run_fewshot_parallel.py does the split, launch and merge in one go.

`share-invariant` copies the cells of the design-invariant methods (all
features, LLM prior, random subset) from the other campaign's results file
into this one where they are missing. Those methods do not depend on which
rows the selectors score, so the cells are identical by construction and
the copy saves re-evolving three sevenths of a campaign; check_reference.py
verifies the invariance on the cells both campaigns have evolved.

GEPRIOR_SMOKE=1 (set by run_fewshot_parallel.py --smoke) shrinks everything
to a quick end-to-end check written under results/_smoke.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geprior import config, fewshot   # noqa: E402

PARTS = config.RESULTS_DIR / "parts"
INVARIANT = ("none", "llm", "random")


def _checkpoint(name, keys):
    if keys is None:
        return config.RESULTS_DIR / name
    PARTS.mkdir(exist_ok=True)
    return PARTS / f"{'-'.join(keys)}__{name}"


def merge():
    """Fold per-dataset part files into the canonical checkpoint."""
    import pandas as pd
    for part in sorted(PARTS.glob("*.csv")):
        name = part.name.split("__", 1)[1]
        target = config.RESULTS_DIR / name
        frames = [fewshot._load(target)] if target.exists() else []
        frames.append(fewshot._load(part))
        merged = pd.concat(frames, ignore_index=True)
        keys = [c for c in ("study", "dataset", "shot", "control", "seed", "learner")
                if c in merged.columns]
        merged = merged.drop_duplicates(subset=keys, keep="last")
        merged.to_csv(target, index=False)
        print(f"merged {part.name} -> {target.name} ({len(merged)} rows)")


def expected_cells(enc=None, key=None):
    from geprior import tabular
    enc = enc or tabular.load(key)
    return len(enc.available_shots()) * len(config.FEWSHOT_CONTROLS) * config.FEWSHOT_DRAWS


def status():
    """Completed GE cells per dataset against the campaign size."""
    import pandas as pd
    from geprior import tabular
    files = [config.RESULTS_DIR / "fewshot.csv", *PARTS.glob("*__fewshot.csv")]
    frames = [fewshot._load(f) for f in files if f.exists()]
    done = (pd.concat(frames).drop_duplicates(subset=["dataset", "shot", "control", "seed"])
            if frames else pd.DataFrame(columns=["dataset", "shot", "control", "seed", "seconds"]))
    done = done[done.control.isin(config.FEWSHOT_CONTROLS)]   # reported methods only
    counts = done.groupby("dataset").size() if len(done) else pd.Series(dtype=int)
    total = 0
    for key, spec in tabular.DATASETS.items():
        per = expected_cells(key=key)
        total += per
        n = int(counts.get(key, 0))
        print(f"  {key:<20} {spec.task_type[:3]} {n:5d} / {per:<5d} {'done' if n >= per else ''}")
    print(f"  {'total':<24} {len(done):5d} / {total}")
    if len(done) and len(done) < total:
        rate = done["seconds"].mean()
        print(f"  mean {rate:.1f} s per GE cell so far; roughly "
              f"{(total - len(done)) * rate / 3600:.1f} CPU-hours remain at that rate")


def share_invariant():
    """Copy design-invariant cells from the other campaign where missing."""
    import pandas as pd
    other = config.OTHER_RESULTS_DIR / "fewshot.csv"
    here = config.RESULTS_DIR / "fewshot.csv"
    if not other.exists():
        print(f"no results in the other campaign ({other.relative_to(config.ROOT)}); nothing to share")
        return
    src = fewshot._load(other)
    src = src[src.control.isin(INVARIANT)]
    have = fewshot._completed(fewshot._load(here))
    keys = fewshot.CELL_KEYS
    new = src[[tuple(r) not in have for r in src[keys].astype(str).values]]
    if new.empty:
        print("nothing to share: every invariant cell of the other campaign is already here")
        return
    fewshot._append(here, new.to_dict("records"))
    print(f"shared {len(new)} invariant cells ({', '.join(INVARIANT)}) from "
          f"{other.relative_to(config.ROOT)} into {here.relative_to(config.ROOT)}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "fewshot"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 and float(sys.argv[2]) > 0 else None
    keys = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    t0 = time.time()
    if stage == "merge":
        merge()
        return
    if stage == "status":
        status()
        return
    if stage == "share-invariant":
        share_invariant()
        return
    if stage != "fewshot":
        raise SystemExit(f"unknown stage: {stage} (fewshot | merge | status | share-invariant)")
    fewshot.run_fewshot_benchmark(dataset_keys=keys, time_budget=budget,
                                  checkpoint=_checkpoint("fewshot.csv", keys),
                                  anchor_path=_checkpoint("fewshot_anchors.csv", keys))
    print(f"{stage}: {time.time()-t0:.0f}s | {config.summary()}")


if __name__ == "__main__":
    main()
