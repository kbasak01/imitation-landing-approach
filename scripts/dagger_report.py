"""Per-iteration ``bc_dagger`` report (plan Phase 5 item 3). A background job: it flies ``select``.

``make dagger-report`` wraps it. For each seed and iterations 0..5 it writes one row of
``results/phase5/dagger_iterations.csv`` (and ``.json`` with provenance): rollout success on
``train`` (``policy_flown`` says whether the beta-mixture or the student flew), labelled episodes
and frames, teacher agreement on the student's own proposals, the copycat diagonal, and the
``select`` success of the EMA at each iteration's last step (CPU fp32 single-sample path; rows
already flown for the same checkpoint are reused). Descriptive only: ``bc_dagger`` is selected on
iteration 5's checkpoints (P3-D1 §3). No frozen list is read. Examples::

    python scripts/dagger_report.py --seeds 0 1 2 3 4
    python scripts/dagger_report.py --seeds 0 --no-fly        # reuse rows only, no flight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ila import p8_adapter
from ila.policies.registry import REPO_ROOT
from ila.train import manifest as mf
from ila.train.dagger_report import build_report, write_report

OUT = REPO_ROOT / "results" / "phase5" / "dagger_iterations.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=24, help="select rollout processes")
    parser.add_argument("--threads", type=int, default=8, help="torch threads for the Jacobians")
    parser.add_argument("--no-fly", action="store_true", help="never fly; reuse rows only")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data_root = p8_adapter.load_paths()["data_root"] if args.data_root is None else args.data_root
    report = build_report(
        args.seeds,
        run_root=args.run_root,
        data_root=Path(data_root),
        workers=args.workers,
        fly=not args.no_fly,
        log=lambda m: print(m, flush=True),
    )
    out_json = write_report(report, args.out)
    print(f"wrote {args.out} and {out_json}; missing: {report['missing'] or 'none'}", flush=True)
    return 0 if not report["missing"] else 2


if __name__ == "__main__":
    sys.exit(main())
