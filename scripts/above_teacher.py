"""Touchdown audit of every student cell above its teacher (logic: ila.eval.above_teacher).

Simulation only. Run after ``scripts/score.py``: reads ``results/e08/above_teacher.csv``, the
committed rows and P8's committed teacher rows, and writes
``results/e08/above_teacher/<cell>.json`` plus ``audit.md`` for ``results-skeptic``.

    python scripts/above_teacher.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.eval import above_teacher, score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=score.OUT_ROOT, help="rows root")
    parser.add_argument("--out", type=Path, default=None, help="scored dir (default: --root)")
    args = parser.parse_args()
    audits = above_teacher.run_audits(score.default_inputs(args.root), args.out)
    print(f"{len(audits)} above-teacher cells audited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
