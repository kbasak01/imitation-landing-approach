"""Audit demo stores and write the demo summary (plan Phase 3 item 3; demo-data rules).

Per store: split audit (train pool only, disjoint by realization key from ``tune`` and the
five frozen lists, no SS6 / 90 deg, identities equal the committed split), re-fly of the
episodes of 20 random stored steps (byte-identical frames and arrays), re-label of 1 000 random
steps with the teacher shadowed in order, counts by sea state / heading / speed, outcomes, kept
fraction after success filtering, bytes on disk. Writes one JSON (default
``results/demos_summary.json``). Example::

    python scripts/audit_demos.py /home/kbasak/data/ila/demos/*.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ila import p8_adapter
from ila.data import audit
from ila.eval.jobs import now_utc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path, nargs="+")
    parser.add_argument(
        "--out", type=Path, default=p8_adapter.REPO_ROOT / "results" / "demos_summary.json"
    )
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--n-labels", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    report: dict[str, Any] = {"written_utc": now_utc(), "collections": {}}
    passed = True
    for path in args.stores:
        manifest_path = path.with_suffix(".manifest.json")
        entry: dict[str, Any] = {
            "counts": audit.counts(path),
            "split_audit": audit.split_audit(path),
            "refly_audit": audit.refly_audit(path, args.n_states, args.seed),
            "relabel_audit": audit.relabel_audit(path, args.n_labels, args.seed),
            "manifest": (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else None
            ),
        }
        ok = all(entry[k]["passed"] for k in ("split_audit", "refly_audit", "relabel_audit"))
        entry["passed"] = ok
        passed &= ok
        report["collections"][entry["counts"]["collection"]] = entry
        print(
            f"{path.name}: split {entry['split_audit']['passed']}, refly "
            f"{entry['refly_audit']['frames_identical']}/{entry['refly_audit']['n_states']}, "
            f"relabel {entry['relabel_audit']['identical']}/{entry['relabel_audit']['n_steps']}, "
            f"kept {entry['counts']['kept_fraction_success_filter']:.3f}"
        )
    report["passed"] = passed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
