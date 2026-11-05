"""Closed-loop clock intervention on ``select`` (Phase 7 diagnostic). A background job.

Re-flies each ``plus_clock`` run's selected EMA checkpoint (``selection.json``, SHA-256
re-checked) on the 150 ``select`` episodes on the CPU fp32 single-sample path, and P8's
``T_ppo``, with only the ``time_fraction`` input changed: ``none`` (must reproduce the
reference rows; raises otherwise), ``shift_s`` (+1 s model scale), ``const_mean`` (the
training normaliser mean) and ``clamp`` (clock stops at 1.6 s model scale)
(:mod:`ila.train.clock_diagnostics`). Writes
``results/phase7/diagnostics/clock_intervention.{csv,json}`` and
``clock_intervention_episodes.csv.gz``::

    python scripts/clock_intervention.py --run-root artifacts/runs --seeds 0 1 2 --workers 4

A diagnostic on ``select``: it chooses nothing and is not scored. Model scale, lambda = 1/25.
No frozen list is read; nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.policies.registry import REPO_ROOT
from ila.train.clock_diagnostics import MODES, run_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=REPO_ROOT / "artifacts" / "runs", help="runs root"
    )
    parser.add_argument("--arm", default="bc_plus_clock", help="a plus_clock arm")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--no-teacher", action="store_true", help="skip T_ppo")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--teacher-const-mean",
        type=float,
        default=None,
        help="T_ppo's const_mean time_fraction (default: the students' training mean)",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "phase7" / "diagnostics"
    )
    parser.add_argument("--limit", type=int, default=None, help="tests only: first N episodes")
    args = parser.parse_args()
    runs = [args.run_root / args.arm / f"s{s}" for s in args.seeds]
    record = run_study(
        runs,
        out_dir=args.out,
        teacher=not args.no_teacher,
        modes=args.modes,
        workers=args.workers,
        limit=args.limit,
        teacher_const_mean=args.teacher_const_mean,
        log=lambda m: print(m, flush=True),
    )
    print(
        f"wrote {args.out}/clock_intervention.{{csv,json}} and _episodes.csv.gz: "
        f"{len(record['summary'])} rows, {record['wall_s']:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
