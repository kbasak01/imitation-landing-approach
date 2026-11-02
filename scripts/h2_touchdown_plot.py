"""H2 touchdown-time distributions on ``select``: plotted only, never scored (plan Phase 7 item 3).

Logic in :mod:`ila.eval.h2_select`. Simulation only; times in seconds, model scale
(lambda = 1/25; x5 for full scale). No P8 frozen list is read. Three steps::

    # 1. Fly the T_mix reference (P8 ppo s0 + gated on their assigned select episodes; CPU,
    #    render off; 150 teacher episodes) -> results/phase7/h2/tmix_select_rows.csv (+ run info)
    python scripts/h2_touchdown_plot.py fly-reference --workers 8

    # 2. Read the committed reference and every bc_mix / dp_mix seed's selected select rows
    #    (verified against selection.json; missing runs are reported, never skipped), write the
    #    episode table and sources, then render. --require-complete exits 3 if any is missing.
    python scripts/h2_touchdown_plot.py build --require-complete

    # 3. Re-render the summary and figure from the committed episode table and sources only.
    python scripts/h2_touchdown_plot.py render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.eval import h2_select
from ila.train.manifest import RUNS_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    fly = sub.add_parser("fly-reference", help="fly the T_mix reference on select")
    fly.add_argument("--workers", type=int, default=8)
    fly.add_argument("--chunk", type=int, default=10)
    fly.add_argument("--limit-per-cell", type=int, default=None, help="tests only")
    fly.add_argument(
        "--render",
        choices=["off", "full"],
        default="off",
        help="off: the committed reference; full: render-invariance check only",
    )
    fly.add_argument("--out", type=Path, default=h2_select.OUT_DIR / h2_select.REFERENCE_FILE)
    b = sub.add_parser("build", help="collect reference + students, then render")
    b.add_argument("--run-root", type=Path, default=RUNS_ROOT)
    b.add_argument("--reference", type=Path, default=h2_select.OUT_DIR / h2_select.REFERENCE_FILE)
    b.add_argument("--out-dir", type=Path, default=h2_select.OUT_DIR)
    b.add_argument("--require-complete", action="store_true", help="exit 3 if any run missing")
    r = sub.add_parser("render", help="re-render from the committed episode table and sources")
    r.add_argument("--out-dir", type=Path, default=h2_select.OUT_DIR)
    args = parser.parse_args()

    if args.cmd == "fly-reference":
        rows, assigned, info = h2_select.fly_reference(
            workers=args.workers,
            chunk=args.chunk,
            render=args.render,
            limit_per_cell=args.limit_per_cell,
        )
        h2_select.write_reference(args.out, rows, assigned, info)
        print(f"wrote {args.out}: {len(rows)} rows, {info['wall_s']} s", flush=True)
        return 0
    if args.cmd == "build":
        record = h2_select.build(
            reference_csv=args.reference, run_root=args.run_root, out_dir=args.out_dir
        )
    else:
        record = h2_select.render(args.out_dir)
    for line in record["missing"]:
        print(f"MISSING {line}", flush=True)
    print(f"wrote {args.out_dir}: partial={record['partial']}", flush=True)
    if args.cmd == "build" and args.require_complete and record["partial"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
