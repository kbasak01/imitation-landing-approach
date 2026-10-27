"""Phase 2 throughput: render, teacher, GPU inference, lockstep end-to-end; then the budget.

Full run (background): ``make throughput``. Pre-flight: ``python scripts/throughput.py --smoke``
(writes only under ``artifacts/runs/throughput/smoke/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.eval import throughput

_RUN_DIR = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "throughput"
_OUT_DIR = p8_adapter.REPO_ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or (_RUN_DIR / "smoke" if args.smoke else _RUN_DIR)
    out_dir = args.out_dir or (run_dir if args.smoke else _OUT_DIR)
    info = throughput.run(out_dir=out_dir, run_dir=run_dir, smoke=args.smoke)
    print(f"wrote {out_dir / 'throughput.csv'} and the projection; wall {info['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
