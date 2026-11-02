"""``modular`` per-field estimation error (Phase 7 item 1; P7-D2). A background job.

For each run directory, the selected checkpoint (``selection.json``, SHA-256 re-checked) is
evaluated open loop on the held-out ``ppo_2000`` demos 1950-1999 (``train`` pool) and closed
loop by re-flying the 150 ``select`` episodes on the CPU fp32 single-sample path, logging the
estimate and the true privileged fields at every step (the truth is logged, never an input).
Per field, component and norm: bias, RMSE and p95 of ``|error|``, model scale, with full-scale
columns and the ratio to P8's sigma_p = 0.04 m / sigma_v = 0.2 m/s
(:mod:`ila.train.modular_eval`). Writes ``results/phase7/modular_estimation.{csv,json}``::

    python scripts/modular_estimation.py artifacts/runs/modular/s0 ... artifacts/runs/modular/s4

``--checkpoint`` evaluates one named checkpoint instead of the selected one (diagnostics and
tests only; recorded as ``selected: false``). ``--select-limit`` flies only the first N
``select`` episodes (tests only; recorded). No frozen list is read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.policies.registry import REPO_ROOT
from ila.train.config import sha256_file
from ila.train.modular_eval import selected_checkpoint, write_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="modular run directories")
    parser.add_argument("--checkpoint", type=Path, default=None, help="override (one run only)")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPO_ROOT / "results" / "phase7" / "modular_estimation",
    )
    parser.add_argument("--select-limit", type=int, default=None, help="tests only")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["open_loop", "closed_loop"],
        choices=["open_loop", "closed_loop"],
    )
    args = parser.parse_args()
    if args.checkpoint is not None and len(args.runs) != 1:
        parser.error("--checkpoint names one checkpoint of one run")
    runs = []
    for run in args.runs:
        if args.checkpoint is not None:
            runs.append((run, args.checkpoint, sha256_file(args.checkpoint), False))
        else:
            path, sha = selected_checkpoint(run)
            runs.append((run, path, sha, True))
    data_root = p8_adapter.load_paths()["data_root"] if args.data_root is None else args.data_root
    record = write_study(
        runs,
        data_root=data_root,
        out_prefix=args.out_prefix,
        workers=args.workers,
        select_limit=args.select_limit,
        modes=args.modes,
    )
    print(f"wrote {args.out_prefix}.csv/.json: {len(record['rows'])} rows, "
          f"{record['wall_s']:.0f} s", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
