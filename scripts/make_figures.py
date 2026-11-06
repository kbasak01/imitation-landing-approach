"""Draw the Phase 8 figures from committed CSVs only (logic in :mod:`ila.viz.figures`).

Simulation only. Reads ``results/e08/{aggregate,baselines,contrasts,h2_touchdown_times,
above_teacher}.csv`` (+ JSON beside them) and writes ``results/e08/figures/<name>.png`` with
the ``<name>.csv`` each is drawn from: success vs sea state per list (teacher and P8 baselines
beside), H4's gap bars, H3's Ta curve with and without ensembling, H2's touchdown-time
distributions at ``id`` SS4, and the K sweep.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.eval import score
from ila.viz import figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=score.OUT_ROOT, help="scored dir")
    args = parser.parse_args()
    for name, path in figures.make_figures(args.root).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
