"""Verify P8's pin, frozen files and PPO seed-0 checkpoint -> results/p8_provenance.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/p8_provenance.json"))
    parser.add_argument("--paths", type=Path, default=None, help="paths.local.yaml override")
    args = parser.parse_args()
    paths = p8_adapter.load_paths() if args.paths is None else p8_adapter.load_paths(args.paths)
    record = provenance.collect(paths)
    provenance.write(record, args.out)
    for check in record["checks"]:
        flag = "OK  " if check["match"] else "FAIL"
        print(f"{flag} {check['name']}: {check['actual']}")
    print(f"wrote {args.out}; all_match={record['all_match']}")
    return 0 if record["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
