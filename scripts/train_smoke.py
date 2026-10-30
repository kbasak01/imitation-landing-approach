"""Smoke pre-flight for one arm: overfit 8 train episodes, land >= 7/8 closed loop (Phase 4).

``make train-smoke ARM=bc`` (or ``bc_state``). Foreground, a few minutes on the A4000. Writes
``results/phase4/smoke_<arm>.json`` and the smoke run under ``artifacts/runs/smoke/<arm>/``
(replaced each time; purpose ``smoke``, never selectable). Exit code 0 iff both criteria pass
(action MSE < 1e-4 on the 8 episodes, EMA weights, CPU fp32; >= 7/8 ``success`` on the CPU fp32
single-sample path). Example::

    python scripts/train_smoke.py bc --steps 4000 --workers 8

``make train-smoke ARM=bc_dagger`` (Phase 5, about 30 min: CPU rollouts, two 1 000-demo
aggregates, an 8 000-step overfit) runs :func:`ila.train.smoke.run_dagger_smoke` instead: two
DAgger iterations of 4 rollouts from ``bc`` s0's last step plus a resume, then an overfit of those
8 rollout episodes flown closed loop. It writes ``results/phase5/smoke_bc_dagger.json``.
``--steps`` is then the steps per DAgger iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila.policies.registry import REGISTRY
from ila.train.smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(a for a, e in REGISTRY.items() if e.kind != "dp"))
    parser.add_argument("--steps", type=int, default=None, help="smoke steps (default per arm)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    record = run_smoke(
        args.arm, steps=args.steps, device=args.device, workers=args.workers, out_json=args.out
    )
    cl = record["closed_loop"]
    mse = record["mse"] if "mse" in record else record["overfit"]["mse"]
    print(
        json.dumps(
            {
                "arm": record["arm"],
                "checks": record.get("checks"),
                "mse": mse["value"],
                "landed": f"{cl['success']}/{cl['n']}",
                "outcomes": cl["outcomes"],
                "train_steps_per_s": record.get("train", {}).get("steps_per_s"),
                "train_wall_s": record.get("train", record.get("loop", {})).get("wall_s"),
                "fly_wall_s": cl["wall_s"],
                "passed": record["passed"],
            },
            indent=2,
        )
    )
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
