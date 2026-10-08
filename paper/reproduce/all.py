"""Reproduce every RQ result and appendix, then rebuild the figures.

Runs each paper/reproduce/rq*.py and appendix.py in turn (each recomputes its numbers from
data/gt + data/paper and checks them against paper/expected/), then
paper/figures/make_*.py. Exit status 1 if any check fails.

    .venvs/core/bin/python paper/reproduce/all.py [--bootstrap] [--no-figures]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
SCRIPTS = ["rq1.py", "rq2.py", "rq3.py", "appendix.py"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", action="store_true",
                    help="rq3: recompute the scenario bootstrap (minutes) instead of reading it")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    failed = []
    for s in SCRIPTS:
        print(f"== {s}", flush=True)
        cmd = [sys.executable, str(HERE / s)]
        if s == "rq3.py" and args.bootstrap:
            cmd.append("--bootstrap")
        if subprocess.run(cmd).returncode:
            failed.append(s)
    if not args.no_figures:
        for f in sorted(FIGURES.glob("*/make_*.py")):
            print(f"== {f.relative_to(FIGURES.parent)}", flush=True)
            if subprocess.run([sys.executable, str(f)]).returncode:
                failed.append(str(f.relative_to(FIGURES.parent)))
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nall results reproduce")
    return 0


if __name__ == "__main__":
    sys.exit(main())
