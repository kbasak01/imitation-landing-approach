"""Score the Phase 8 matrix from committed rows (logic in :mod:`ila.eval.score`).

Simulation only. Reads ``results/e08/rows/`` (finished files: sidecar present, SHA-256 checked),
P8's committed teacher and baseline rows, the committed Phase 4/5 run manifests (H1's labelled
frames) and selection records (H4's best end-to-end student). Writes, deterministically:
``summary.csv``, ``baselines.csv``, ``aggregate.csv``, ``contrasts.csv``, ``hypotheses.csv`` /
``.json``, ``above_teacher.csv``, ``missing.csv``, ``h2_touchdown_times.csv`` and
``score_info.json`` under ``--out`` (default ``results/e08``). Runs on a partial matrix: every
missing cell is listed in ``missing.csv`` and marked in every table.

    python scripts/score.py                 # the committed tree
    python scripts/score.py --require-complete   # exit 3 if any cell is missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.eval import score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=score.OUT_ROOT, help="rows root")
    parser.add_argument("--out", type=Path, default=None, help="output dir (default: --root)")
    parser.add_argument("--require-complete", action="store_true", help="exit 3 if partial")
    args = parser.parse_args()
    info = score.run_score(score.default_inputs(args.root), args.out)
    print(
        f"cells: {info['cells_present']} present, {info['cells_missing']} missing "
        f"of {info['cells_expected']}"
    )
    if args.require_complete and not info["complete"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
