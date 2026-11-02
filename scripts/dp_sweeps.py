"""``dp`` inference-time sweeps on ``select`` (Phase 6, P6-D1). Background job; reported only.

For each named run, flies the checkpoint selected at ``Ta = 8``, ``K = 10`` (its
``selection.json``) on the 150 ``select`` episodes at ``Ta in {1, 2, 4, 8, 12}`` x ensembling
``{off, on}`` (K = 10) and ``K in {1, 2, 5, 10, 25, 100}`` (Ta = 8, off), CPU fp32
single-sample path, EMA weights. The ``Ta = 8, K = 10, off`` cell is flown first and must
reproduce the selection rows bit for bit. Rows go to ``results/phase6/sweeps/<arm>/s<n>/``,
the summary to ``results/dp_sweeps.json``. Resumable per cell (re-run the same command).
Nothing is re-selected; no frozen list is read. ``make dp-sweeps`` wraps it::

    python scripts/dp_sweeps.py --arm dp --seeds 0 1 2 3 4
    python scripts/dp_sweeps.py artifacts/runs/dp/s0 --workers 24

About 15-25 min per seed at 24 workers if ``dp`` episodes are teacher-length (61.5 steps), up
to about 6 times that if they time out (``ila.train.dp_sweeps`` module docstring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.train import manifest as mf
from ila.train.dp_sweeps import run_sweeps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    args = parser.parse_args()
    runs = list(args.runs)
    if args.arm is not None:
        runs += [mf.run_dir(args.arm, s, root=args.run_root) for s in (args.seeds or [])]
    if not runs:
        parser.error("nothing to sweep: give run directories or --arm with --seeds")
    for run in runs:
        entry = run_sweeps(run, workers=args.workers)
        print(
            f"{run}: {len(entry['cells'])} cells; reproduces selection rows: "
            f"{entry['reproduces_selection_rows']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
