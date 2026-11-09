"""Write or check ``results/readme/model_specs.json`` (controller and network specs; Phase 11).

Every student arm with a config under ``configs/train/`` is built on the CPU from that config
(no checkpoint) and its architecture, horizons, training settings and parameter counts are
recorded; P8's PPO teacher and classical controllers are described from P8's files through
``ila.p8_adapter``. Logic: :mod:`ila.policies.specs`. Examples::

    python scripts/model_specs.py                   # re-derive and write (workstation)
    python scripts/model_specs.py --check           # re-derive and compare; exit 1 on a diff
    python scripts/model_specs.py --check --no-checkpoints   # skip the checkpoint cross-checks

Where a pinned seed-0 checkpoint (or P8's PPO ``final/``) is absent -- always on CI -- its
cross-check cannot run; the committed cross-check record is carried over (after checking that
it describes the same config and parameter total) and the output says so, per arm. Sizes are
px, horizons control steps of 1/30 s model scale (lambda = 1/25), training lengths optimiser
steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.policies import specs


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        0 on success; 1 if ``--check`` finds a difference.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and compare with the committed file; exit 1 on any difference",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="do not run the checkpoint cross-checks (carry the committed records over)",
    )
    parser.add_argument("--out", type=Path, default=specs.OUTPUT, help="the JSON file")
    args = parser.parse_args(argv)
    run = specs.check if args.check else specs.write
    result = run(args.out, checkpoints=not args.no_checkpoints)
    for line in result.notes:
        print(line)
    if args.check:
        print("model_specs --check:", "OK" if result.ok else "FAILED")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
