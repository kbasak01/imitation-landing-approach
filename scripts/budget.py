"""P3-D1 §5: project every planned training run from the timing runs against 120 GPU-hours.

Reads the timing-run manifests (``artifacts/runs/bc_timing/s0/manifest.json`` and
``artifacts/runs/dp_timing/s0/manifest.json``; ``bc_state`` from its timing manifest if one
exists, else ``results/phase4/smoke_bc_state.json``), writes the projection CSV (default
``results/phase4/budget_projection.csv``) and prints the total and P3-D1 §5's cut factor (1.0 when
the projection fits). It applies nothing: a cut is a dated protocol entry (P4-D1), decided
before any ``select`` number exists. Example::

    python scripts/budget.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila.policies.registry import REPO_ROOT
from ila.train import budget
from ila.train import manifest as mf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc", type=Path, default=mf.RUNS_ROOT / "bc_timing" / "s0" / "manifest.json"
    )
    parser.add_argument(
        "--dp", type=Path, default=mf.RUNS_ROOT / "dp_timing" / "s0" / "manifest.json"
    )
    parser.add_argument(
        "--bc-state", type=Path, default=mf.RUNS_ROOT / "bc_state_timing" / "s0" / "manifest.json"
    )
    parser.add_argument(
        "--bc-state-smoke",
        type=Path,
        default=REPO_ROOT / "results" / "phase4" / "smoke_bc_state.json",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "phase4" / "budget_projection.csv"
    )
    parser.add_argument("--budget-hours", type=float, default=budget.BUDGET_HOURS)
    args = parser.parse_args()
    rates = {
        "bc": budget.rate_from_manifest(args.bc, "bc"),
        "dp": budget.rate_from_manifest(args.dp, "dp"),
        "state": (
            budget.rate_from_manifest(args.bc_state, "state")
            if args.bc_state.is_file()
            else budget.rate_from_smoke(args.bc_state_smoke, "state")
        ),
    }
    runs = budget.planned_runs()
    rows = budget.project(runs, rates)
    total = sum(float(r["gpu_hours"]) for r in rows)
    factor = budget.cut_factor(runs, rates, args.budget_hours)
    cut_rows = budget.project(runs, rates, factor) if factor < 1.0 else rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(budget.projection_csv(rows), encoding="utf-8")
    if factor < 1.0:
        args.out.with_name(args.out.stem + "_cut.csv").write_text(
            budget.projection_csv(cut_rows), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "total_gpu_hours": total,
                "budget_hours": args.budget_hours,
                "fits": factor == 1.0,
                "cut_factor": factor,
                "total_after_cut": sum(float(r["gpu_hours"]) for r in cut_rows),
                "rates": {k: vars(v) for k, v in rates.items()},
                "csv": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
