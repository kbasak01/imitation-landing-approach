"""GIF gallery hand-picked by the stated rule (logic in :mod:`ila.viz.gifs`).

Simulation only (PyBullet; frames rendered by PyBullet TinyRenderer). The rule is
``results/e08/gifs/RULE.md`` (= ``ila.viz.gifs.RULE``), committed before any GIF. Needs the
committed core ``id`` rows of ``bc``, ``bc_dagger`` and ``dp`` seed 0; exits 2 without writing
anything if they are not flown yet. Re-flies 4 x 4 panels on the CPU single-sample path
(``--workers`` <= 4) and asserts every row equals its committed row before writing a GIF.
Background job (a few minutes to ~1 h on a loaded machine):

    mkdir -p artifacts/runs/gifs && nohup make gifs > artifacts/runs/gifs/run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.eval import score
from ila.viz import gifs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=score.OUT_ROOT, help="committed rows root")
    parser.add_argument("--out", type=Path, default=score.OUT_ROOT / "gifs")
    parser.add_argument(
        "--release-dir", type=Path, default=p8_adapter.REPO_ROOT / "artifacts" / "release"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        rec = gifs.run_gallery(
            args.root, args.out, release_dir=args.release_dir, workers=min(4, args.workers)
        )
    except gifs.RowsNotReadyError as exc:
        print(f"not ready: {exc}")
        return 2
    print(f"{len(rec['gifs'])} GIFs, {rec['total_bytes']} bytes, committed={rec['gifs_committed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
