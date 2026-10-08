"""Reproduce the appendices' numbers, tables and figure inputs.

Runs paper/reproduce/app{c,d,e,f,h,i}.py in turn; each recomputes its
appendix from data/gt + data/paper and checks it against paper/expected/ and
the figures' cached inputs. Appendix G (the RQ2 metric table) is part of
rq2.py; appendix I's shared-value-space table is part of rq1.py. Exit status
1 if any check fails.

    .venvs/core/bin/python paper/reproduce/appendix.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPENDICES = ["appc", "appd", "appe", "appf", "apph", "appi"]


def main() -> int:
    failed = []
    for a in APPENDICES:
        print(f"== {a}.py", flush=True)
        if subprocess.run([sys.executable, str(HERE / f"{a}.py")]).returncode:
            failed.append(a)
    if failed:
        print(f"\nappendix: FAILED {', '.join(failed)}")
        return 1
    print("\nappendix: all appendices reproduce")
    return 0


if __name__ == "__main__":
    sys.exit(main())
