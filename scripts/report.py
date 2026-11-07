"""Render ``results/results.md`` from committed files only (logic in :mod:`ila.eval.report`).

Simulation only. Reads ``results/e08/{run_info,score_info,hypotheses}.json`` and
``results/e08/{summary,baselines,aggregate,contrasts,above_teacher,missing}.csv`` (plus
``above_teacher/reviewed.json`` if present); recomputes nothing. ``make report`` runs this and
then ``git diff --exit-code results/results.md`` to prove the committed file re-renders
byte-identically.

    python scripts/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.eval import report, score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=score.OUT_ROOT, help="scored dir")
    parser.add_argument("--out", type=Path, default=report.RESULTS_MD, help="markdown file")
    args = parser.parse_args()
    text = report.write_report(args.root, args.out)
    print(f"{args.out}: {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
