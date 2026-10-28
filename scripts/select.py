"""Closed-loop checkpoint selection on ``splits/select.parquet`` (Phase 4). Background job.

Flies every candidate checkpoint (EMA weights) of a finished run on the 150 ``select``
episodes through the runner's CPU fp32 single-sample path and writes ``selection.json``
(success, outcome classes, pooled p95 closing speed per candidate; the rule's choice) plus
per-candidate rows under ``<run>/select/``. ``make select-bg`` wraps it. Examples::

    python scripts/select.py --arm bc --seeds 0 1 2 3 4
    python scripts/select.py artifacts/runs/bc/s0
    python scripts/select.py --dart-trials        # the three sigma trials, then sigma*

``--dart-trials`` selects each ``bc_dart`` sigma trial that has no ``selection.json`` yet, then
applies P3-D1 §4's sigma* rule (ties: the smaller sigma) and writes
``artifacts/runs/bc_dart_tune/sigma_star.json``, which ``scripts/train.py`` requires before any
core ``bc_dart`` run. Timing, smoke and test runs are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila.train import manifest as mf
from ila.train.config import DART_SIGMAS
from ila.train.select import choose_sigma, select_run


def _one(run_dir: Path, workers: int, redo: bool) -> dict[str, object]:
    out = run_dir / "selection.json"
    if out.is_file() and not redo:
        print(f"{run_dir}: selection.json exists (use --redo to fly again)", flush=True)
        return mf.read_json(out)
    rec = select_run(run_dir, workers=workers)
    sel = rec["selected"]
    assert isinstance(sel, dict)
    print(
        f"{run_dir}: selected {Path(sel['candidate']['file']).name} "
        f"success {sel['success']}/{sel['n']} outcomes {sel['outcomes']} "
        f"p95 {sel['p95_closing_speed_m_s']}",
        flush=True,
    )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dart-trials", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    runs = list(args.runs)
    if args.arm is not None:
        runs += [mf.run_dir(args.arm, s, root=args.run_root) for s in (args.seeds or [])]
    if args.dart_trials:
        records = {}
        for sigma in DART_SIGMAS:
            run = mf.run_dir("bc_dart", 0, root=args.run_root, sigma=sigma, tuning=True)
            records[sigma] = _one(run, args.workers, args.redo)
        star = choose_sigma(records)
        path = args.run_root / "bc_dart_tune" / "sigma_star.json"
        mf.write_json_atomic(path, star)
        print(json.dumps({"sigma_star": star["sigma_star"], "file": str(path)}), flush=True)
    for run in runs:
        _one(run, args.workers, args.redo)
    if not runs and not args.dart_trials:
        parser.error("nothing to select")
    return 0


if __name__ == "__main__":
    sys.exit(main())
