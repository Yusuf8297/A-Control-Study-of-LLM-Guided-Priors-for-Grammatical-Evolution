"""The matched design (selectors see only the k labelled rows) from the
command line: campaign, checks, tables, report and figures, all written
under results/matched/ and figures/matched/.

Usage:  python matched_campaign.py            # complete the campaign, then analyse
        python matched_campaign.py --rerun    # evolve the FeatLLM eleven again as well
        python matched_campaign.py --analyse  # tables, report and figures only

Without --rerun, results/matched/fewshot.csv is seeded from the bundled
reference run of 2 October 2026 (the FeatLLM eleven) if it is absent, the
design-invariant cells already evolved by the full-information campaign are
shared in, and the runner evolves whatever is still missing, which on a
fresh checkout is the thirteen benchmark datasets. With --rerun the
reference's cells are evolved again on this machine and compared cell for
cell. The full-information campaign under results/ is never touched.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, GEPRIOR_SELECTOR_ROWS="shots")
RESULTS = ROOT / "results" / "matched"
REFERENCE = ROOT / "results" / "reference_run" / "fewshot.csv"


def run(*script):
    cmd = [sys.executable, *script]
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=ENV)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    local = RESULTS / "fewshot.csv"
    if "--analyse" not in sys.argv:
        if "--rerun" in sys.argv:
            if local.exists():
                local.unlink()   # a seeded copy would make the runner skip every cell
        elif not local.exists():
            shutil.copy(REFERENCE, local)
            print(f"seeded {local.relative_to(ROOT)} from the bundled reference run")
        run(ROOT / "run_study.py", "share-invariant")
        run(ROOT / "run_fewshot_parallel.py", "--matched")
    run(ROOT / "check_reference.py")
    run(ROOT / "analyse_selectors.py")
    run(ROOT / "report_selectors.py")
    run(ROOT / "selector_figures.py")
    print(f"\nmatched campaign: tables in {RESULTS.relative_to(ROOT)}/analysis, "
          f"figures in figures/matched")


if __name__ == "__main__":
    main()
