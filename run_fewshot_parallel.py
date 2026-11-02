"""Run the campaign with one process per dataset group, then merge.

Usage:  python run_fewshot_parallel.py [n_processes] [time_budget_seconds] [--smoke] [--matched]
                                       [--datasets key,key,...]

Defaults to all cores but one and no time budget. --smoke is a short
end-to-end check into <results>/_smoke with a reduced search, two draws and
two shot levels. --matched runs the matched design (selectors see only the
k labelled rows) into results/matched/; without it the full-information
design runs into results/. --datasets restricts the run to the listed keys
(for instance the thirteen benchmark datasets added in October 2026, see
README). Safe to re-run: each process resumes its own part file under
<results>/parts/, and the merge folds the parts into <results>/fewshot.csv.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if "--matched" in sys.argv:
    os.environ["GEPRIOR_SELECTOR_ROWS"] = "shots"   # before geprior.config is imported
if "--smoke" in sys.argv:
    os.environ["GEPRIOR_SMOKE"] = "1"

from geprior import config, tabular   # noqa: E402

# heaviest first so the groups balance: the multiclass datasets evolve one
# rule per class and dominate the runtime, then the large or wide ones
ORDER = ["optdigits", "mice_protein", "dermatology", "car", "communities", "student_dropout",
         "spambase", "ames_housing", "adult", "bank", "myocardial", "credit_g",
         "wine_quality", "geo_music", "parkinsons", "student_performance", "sonar",
         "ionosphere", "breast_cancer", "heart", "diabetes", "cultivars", "nhanes", "blood"]


def main():
    argv = list(sys.argv[1:])
    only = None
    if "--datasets" in argv:
        i = argv.index("--datasets")
        only = argv[i + 1].split(",")
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    smoke = "--smoke" in argv
    n_proc = int(args[0]) if args else max(1, (os.cpu_count() or 2) - 1)
    budget = args[1] if len(args) > 1 else "0"
    env = dict(os.environ)
    keys = [k for k in ORDER if k in tabular.DATASETS]
    missing = [k for k in tabular.DATASETS if k not in keys]
    keys += missing
    if only:
        unknown = [k for k in only if k not in tabular.DATASETS]
        if unknown:
            raise SystemExit(f"unknown dataset keys: {unknown}")
        keys = [k for k in keys if k in only]
    groups = [keys[i::n_proc] for i in range(n_proc)]
    t0 = time.time()
    procs = []
    for group in groups:
        if not group:
            continue
        cmd = [sys.executable, str(ROOT / "run_study.py"), "fewshot", budget, ",".join(group)]
        print("launch:", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: exit codes {codes}")
    subprocess.check_call([sys.executable, str(ROOT / "run_study.py"), "merge"], cwd=ROOT, env=env)
    where = config.RESULTS_DIR.relative_to(ROOT)
    kind = "SMOKE CHECK (reduced search, 2 draws, 2 shot levels; not the campaign)" if smoke \
        else f"FULL CAMPAIGN ({config.CAMPAIGN} design)"
    print(f"\n{kind} finished in {(time.time() - t0) / 60:.1f} min wall-clock over {n_proc} processes")
    print(f"checkpoint: {where}/fewshot.csv")
    subprocess.call([sys.executable, str(ROOT / "run_study.py"), "status"], cwd=ROOT, env=env)
    if not smoke:
        print("next: python check_reference.py, then python analyse_selectors.py")


if __name__ == "__main__":
    main()
