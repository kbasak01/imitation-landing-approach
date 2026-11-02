"""``dp`` diagnostics of each run's selected checkpoint (Phase 6, P6-D1). Background job.

Per run: the per-state spread of 16 samples and the fraction of sampled elements at the
``[-1, 1]`` bound, on P4-D5's held-out states and on ``select`` replanning-step states; the
chunk-boundary vs within-chunk first and second differences of the executed actions, open loop
on the held-out episodes and closed loop on ``select`` (re-flown with an action recorder; its
rows must equal the stored selection rows). EMA weights, CPU fp32, Ta = 8, K = 10. Writes
``results/phase6/diagnostics/<arm>/s<n>.json`` (definitions and units:
``ila.train.dp_diagnostics``). ``make dp-diagnostics`` wraps it::

    python scripts/dp_diagnostics.py --arm dp --seeds 0 1 2 3 4
    python scripts/dp_diagnostics.py artifacts/runs/dp/s0 --no-closed-loop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila.train import manifest as mf
from ila.train.dp_diagnostics import run_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--no-closed-loop", action="store_true", help="held-out states only")
    args = parser.parse_args()
    runs = list(args.runs)
    if args.arm is not None:
        runs += [mf.run_dir(args.arm, s, root=args.run_root) for s in (args.seeds or [])]
    if not runs:
        parser.error("nothing to measure: give run directories or --arm with --seeds")
    for run in runs:
        rec = run_diagnostics(
            run,
            data_root=args.data_root,
            workers=args.workers,
            closed_loop=not args.no_closed_loop,
        )
        held = rec["heldout"]
        print(
            json.dumps(
                {
                    "run": str(run),
                    "heldout_spread_executed_median": held["spread_executed"]["median"],
                    "heldout_clip_executed": held["clip_executed"],
                    "heldout_d1_ratio": held["open_loop_jerk"]["d1_ratio_boundary_over_within"],
                    "select_d1_ratio": rec.get("select", {})
                    .get("closed_loop_jerk", {})
                    .get("d1_ratio_boundary_over_within"),
                    "wall_clock_s": rec["wall_clock_s"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
