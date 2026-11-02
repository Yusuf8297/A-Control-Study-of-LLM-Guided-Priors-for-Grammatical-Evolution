"""Search-budget sensitivity: the hard tier re-run at the doubled generation
budget, levels 32, 64 and full, all methods, all draws.

Usage:  python run_budget_sensitivity.py [n_processes]            # launch + merge
        python run_budget_sensitivity.py worker dataset,list     # one worker
        python run_budget_sensitivity.py status

Tiers and budgets are in geprior/config.py (SEARCH_TIER, GE_BY_TIER). Seeds,
draws, subsets and grammars are the campaign's; only n_generations differs,
so generations 0-40 of every cell here are exactly the campaign's. Writes
results/sensitivity/fewshot.csv through per-worker part files; resumable.
Never merged into results/fewshot.csv.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from geprior import config, fewshot   # noqa: E402

OUT = config.SENSITIVITY_DIR
PARTS = OUT / "parts"
CANONICAL = OUT / "fewshot.csv"
HARD = [k for k in ("communities", "adult", "bank", "myocardial", "credit_g", "diabetes",
                    "cultivars") if config.SEARCH_TIER[k] == "hard"]


def worker(keys: list[str]) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    fewshot.run_fewshot_benchmark(dataset_keys=keys, shots=config.SENSITIVITY_SHOTS,
                                  checkpoint=PARTS / f"{'-'.join(keys)}__fewshot.csv",
                                  anchor_path=PARTS / f"{'-'.join(keys)}__anchors.csv",
                                  tiered=True, canonical=CANONICAL)


def merge() -> None:
    frames = [pd.read_csv(CANONICAL, dtype={"shot": str})] if CANONICAL.exists() else []
    frames += [pd.read_csv(p, dtype={"shot": str}) for p in sorted(PARTS.glob("*__fewshot.csv"))]
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=fewshot.CELL_KEYS,
                                                                  keep="last")
    merged.to_csv(CANONICAL, index=False)
    print(f"merged -> {CANONICAL} ({len(merged)} rows)")


def status() -> None:
    expected = len(config.SENSITIVITY_SHOTS) * config.FEWSHOT_DRAWS * len(config.FEWSHOT_CONTROLS)
    paths = ([CANONICAL] if CANONICAL.exists() else []) + sorted(PARTS.glob("*__fewshot.csv"))
    done = (pd.concat([pd.read_csv(p, dtype={"shot": str}) for p in paths])
              .drop_duplicates(subset=fewshot.CELL_KEYS) if paths else pd.DataFrame())
    if len(done):
        done = done[done.control.isin(config.FEWSHOT_CONTROLS)]   # reported methods only
    for key in HARD:
        n = int((done.dataset == key).sum()) if len(done) else 0
        print(f"  {key:12s} {n:4d}/{expected}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "worker":
        worker(args[1].split(","))
        return
    if args and args[0] == "status":
        status()
        return
    n_proc = int(args[0]) if args else max(1, (os.cpu_count() or 2) - 1)
    groups = [HARD[i::n_proc] for i in range(n_proc)]
    t0 = time.time()
    procs = []
    for group in groups:
        if group:
            cmd = [sys.executable, str(ROOT / "run_budget_sensitivity.py"), "worker", ",".join(group)]
            print("launch:", " ".join(cmd), flush=True)
            procs.append(subprocess.Popen(cmd, cwd=ROOT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: exit codes {codes}")
    merge()
    print(f"\nsensitivity campaign finished in {(time.time() - t0) / 60:.1f} min over {n_proc} processes")
    status()
    print("next: python analyse_budget_sensitivity.py")


if __name__ == "__main__":
    main()
