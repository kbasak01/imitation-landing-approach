"""Draw the README figures from committed CSV/JSON only (logic in :mod:`ila.viz.readme_figures`).

Simulation only. Reads ``results/e08/{aggregate,baselines,hypotheses}.csv``,
``results/e08/hypotheses.json``, ``results/e08/figures/{h3_ta_curve,h4_gaps}.csv`` and
``results/latency/{latency,parity,closed_loop,h5a}.csv`` + ``delay.json``; writes
``docs/figures/<name>.png`` (light), ``<name>_dark.png`` (dark) and ``<name>.csv`` (the values
plotted, each traced to its source file, row and column). Same inputs, same bytes.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.viz import readme_figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=p8_adapter.REPO_ROOT, help="repository root (inputs)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output dir (default <root>/{readme_figures.OUT_DIR})",
    )
    args = parser.parse_args()
    for name, paths in readme_figures.make_readme_figures(args.root, args.out).items():
        print(f"{name}: " + " ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
