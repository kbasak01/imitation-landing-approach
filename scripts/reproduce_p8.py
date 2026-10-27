"""Phase 2 reproduction gate: P8's pid_feedforward and ppo rows, re-flown with render on.

Full run (background; ~10 min at 24 workers): ``make reproduce-p8``. Pre-flight in the
foreground: ``python scripts/reproduce_p8.py --smoke`` (first 2 episodes per cell; writes only
under ``artifacts/runs/reproduce_p8/smoke/``). Exit code 0 only if every comparison passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila import p8_adapter
from ila.eval import reproduce

_RUN_DIR = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "reproduce_p8"
_OUT_DIR = p8_adapter.REPO_ROOT / "results" / "reproduce_p8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="first N episodes per cell only")
    parser.add_argument("--smoke-n", type=int, default=2, help="episodes per cell in --smoke")
    parser.add_argument("--workers", type=int, default=p8_adapter.DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=p8_adapter.DEFAULT_CHUNK)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    smoke_n = args.smoke_n if args.smoke else None
    run_dir = args.run_dir or (_RUN_DIR / "smoke" if args.smoke else _RUN_DIR)
    out_dir = args.out_dir or (run_dir if args.smoke else _OUT_DIR)
    record = reproduce.run(
        out_dir=out_dir, run_dir=run_dir, workers=args.workers, chunk=args.chunk, smoke_n=smoke_n
    )
    for method, result in record["methods"].items():
        stats = result["compare_to_reference"]["methods"].get(method, {})
        print(
            f"{method}: expected {result['expected']}, compared {stats.get('n_compared')}, "
            f"identical {stats.get('n_identical')}, differ {stats.get('n_differ')}, "
            f"missing {stats.get('n_missing_in_reference')}, first mismatch "
            f"{stats.get('first_mismatch')} -> {'PASS' if result['passed'] else 'FAIL'}"
        )
    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    for method, ev in info["render_evidence"].items():
        print(
            f"{method}: render {ev['render']}, frames {ev['frames_rendered']} = steps "
            f"{ev['steps']} + episodes {ev['episodes']}: "
            f"{ev['checks']['frames_equal_steps_plus_episodes']}; timed checks {ev['timed_checks']}"
        )
    print(f"wrote {out_dir / 'comparison.json'}; passed={record['passed']}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
