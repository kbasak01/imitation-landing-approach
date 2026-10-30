"""R1 ONNX parity on ORT CPU against the torch reference (Phase 6; ``.venv`` only).

Thin wrapper over :func:`ila.deploy.r1_parity.run_parity`: for each export directory
(``<run>/onnx/step<N>/`` from ``scripts/r1/export_onnx.py``) it writes ``parity.json`` beside
the graphs and prints the verdict. Exit status 1 if any export fails. Examples::

    .venv/bin/python scripts/r1/parity.py artifacts/runs/R1_lerobot_dp/s0/onnx/step20000
    .venv/bin/python scripts/r1/parity.py --run artifacts/runs/R1_lerobot_dp/s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.deploy import r1_parity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("step_dirs", nargs="*", type=Path, default=[], help="<run>/onnx/step<N>")
    src.add_argument("--run", type=Path, help="check every <run>/onnx/step<N>/")
    parser.add_argument("--threads", type=int, default=1, help="ORT intra-op threads")
    args = parser.parse_args()
    dirs = (
        sorted(
            (p for p in (args.run / "onnx").glob("step*") if p.is_dir()),
            key=lambda p: int(p.name.removeprefix("step")),
        )
        if args.run is not None
        else list(args.step_dirs)
    )
    if not dirs:
        sys.exit("no export directories")
    failed = 0
    for d in dirs:
        r = r1_parity.run_parity(d, threads=args.threads)
        failed += not r["pass"]
        print(
            f"{d}: {'PASS' if r['pass'] else 'FAIL'}  eps/step max-abs "
            f"{max(r['eps_per_step_max_abs']):.2e} (<= {r1_parity.EPS_TOL:g})  probe "
            f"{max(r['eps_probe_max_abs'].values()):.2e}  global_cond "
            f"{r['global_cond_max_abs']:.2e}  chunk(norm) {r['chunk_norm_max_abs']:.2e} "
            f"[reported]  vs LeRobot select_action "
            f"{r['chunk_executed_vs_lerobot_select_action_max_abs']:.2e}  "
            f"failed: {[k for k, v in r['judged'].items() if not v]}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
