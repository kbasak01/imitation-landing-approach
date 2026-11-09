r"""README hero GIFs: the gallery's picks from outside, with the onboard frame (``ila.viz.hero``).

Simulation only (PyBullet physics; every image rendered by PyBullet TinyRenderer; model scale,
lambda = 1/25). Episodes are the four picks already in ``results/e08/gifs/manifest.json``. There
is no new selection. Re-flies 4 x 4 panels on the CPU single-sample path (``--workers`` <= 4),
asserts every row equals its committed row before any GIF is written, and writes
``docs/media/hero_*.gif`` plus ``docs/media/manifest.json``. The full run refuses a dirty tree.
Background job:

    mkdir -p artifacts/runs/hero && nohup make hero-gifs > artifacts/runs/hero/run.log 2>&1 &

Smoke run (one category, may be dirty, never writes to docs/media):

    .venv/bin/python scripts/make_hero_gifs.py --only a_all_four_succeed --panels teacher \
        --out artifacts/hero_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import p8_adapter
from ila.eval.jobs import JobStatus
from ila.eval.matrix import DirtyTreeError
from ila.viz import gifs, hero

SMOKE_OUT = p8_adapter.REPO_ROOT / "artifacts" / "hero_smoke"
RUN_DIR = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "hero"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None, help="output dir (default docs/media)")
    parser.add_argument("--workers", type=int, default=4, help="re-flight processes, <= 4")
    parser.add_argument("--only", choices=gifs.CATEGORIES, default=None, help="smoke: one category")
    parser.add_argument(
        "--panels", nargs="+", choices=gifs.PANELS, default=list(gifs.PANELS), help="smoke only"
    )
    parser.add_argument("--stride", type=int, default=hero.FRAME_STRIDE, help="steps per frame")
    args = parser.parse_args()
    smoke = args.only is not None or tuple(args.panels) != gifs.PANELS
    out = args.out if args.out is not None else (SMOKE_OUT if smoke else hero.OUT_DIR)
    if smoke and out.resolve() == hero.OUT_DIR.resolve():
        parser.error("a smoke run (--only / --panels) never writes to docs/media; pass --out")
    if args.workers > 4:
        parser.error("--workers must be <= 4")
    status = JobStatus(RUN_DIR / "smoke" if smoke else RUN_DIR, "hero_gifs")
    try:
        rec = hero.run_hero(
            out_dir=out,
            workers=args.workers,
            only=args.only,
            panels=args.panels,
            stride=args.stride,
            require_clean=not smoke,
            status=status,
        )
    except gifs.RowsNotReadyError as exc:
        status.update("failed", error=str(exc))
        print(f"not ready: {exc}")
        return 2
    except DirtyTreeError as exc:
        status.update("failed", error=str(exc))
        print(f"refused: {exc}")
        return 3
    except Exception as exc:
        status.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    for g in rec["gifs"]:
        print(
            f"{g['file']}: {g['bytes']} B, {g['frames']} frames, stride {g['frame_stride_steps']}"
        )
    print(f"total {rec['total_bytes']} bytes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
