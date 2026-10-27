"""Phase 2 determinism contract: dummy dp (K 10, Ta 8) on select, CPU w1/w8 vs GPU B 1/8/24.

Full run (background): ``make determinism``. Pre-flight: ``python scripts/determinism.py
--smoke`` (3 episodes; writes only under ``artifacts/runs/determinism/smoke/``). Exit 0 iff the
CPU single-sample rows are bit-identical across worker counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.eval import determinism

_RUN_DIR = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "determinism"
_OUT_DIR = p8_adapter.REPO_ROOT / "results" / "determinism"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or (_RUN_DIR / "smoke" if args.smoke else _RUN_DIR)
    out_dir = args.out_dir or (run_dir if args.smoke else _OUT_DIR)
    record = determinism.run(out_dir=out_dir, run_dir=run_dir, smoke=args.smoke)
    print(
        f"CPU single-sample bit-identical across workers: "
        f"{record['cpu_single_sample_bit_identical_across_workers']}"
    )
    for name, c in record["comparisons_vs_reference"].items():
        print(
            f"{name}: text differ {c['n_text_differ']}/{c['n_rows']}, outcome flips "
            f"{c['n_outcome_flips']} {c['flips_by_class']}"
        )
    print(f"outcomes: {record['outcomes']}")
    return 0 if record["cpu_single_sample_bit_identical_across_workers"] else 1


if __name__ == "__main__":
    sys.exit(main())
