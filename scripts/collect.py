"""Collect one demo set into the HDF5 store (Phase 3; plan §1.5). Background job.

Reads ``configs/data/demos.yaml`` (collections ``ppo_2000``, ``pid_1000``, ``mix_1000``,
``dart_0.05``, ``dart_0.10``, ``dart_0.20``), flies the collection's split prefix through the
runner with render ``full`` and writes ``<data_root>/demos/<name>.h5`` plus
``<name>.manifest.json`` (``data_root`` from ``configs/paths.local.yaml``). Run bookkeeping goes
to ``artifacts/runs/collect/<name>/``: ``status.json`` (state, episodes done/total, ETA, PID),
``run.log``, ``manifest.json`` and ``episodes.csv`` (the runner rows). Examples::

    python scripts/collect.py ppo_2000                          # full collection, 24 workers
    python scripts/collect.py pid_1000 --limit 8 --data-root /tmp/pilot --run-root /tmp/pilot/runs

A full collection refuses a dirty working tree (the store records the ``ila`` commit), unless
``--allow-dirty``; a ``--limit`` pilot does not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

from ila import p8_adapter
from ila.data import collect
from ila.eval.jobs import JobStatus, rel


def main() -> int:
    collections = collect.load_collections()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(collections), help="collection name")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=collect.DEFAULT_COLLECT_CHUNK)
    parser.add_argument("--limit", type=int, default=None, help="first N episodes only (pilot)")
    parser.add_argument("--data-root", type=Path, default=None, help="default: paths.local.yaml")
    parser.add_argument(
        "--run-root", type=Path, default=p8_adapter.REPO_ROOT / "artifacts" / "runs" / "collect"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing store")
    parser.add_argument("--allow-dirty", action="store_true", help="full run on a dirty tree")
    args = parser.parse_args()

    data_root = args.data_root or p8_adapter.load_paths()["data_root"]
    out = Path(data_root) / "demos" / f"{args.name}.h5"
    run_dir = Path(args.run_root) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    status = JobStatus(run_dir, f"collect:{args.name}")
    log_path = run_dir / "run.log"

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=p8_adapter.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty and args.limit is None and not args.allow_dirty:
        status.update("failed", error="dirty working tree; commit first or --allow-dirty")
        log(f"refusing a full collection on a dirty tree:\n{dirty}")
        return 2

    def progress(done: int, total: int, elapsed: float) -> None:
        eta = elapsed / done * (total - done) if done else None
        status.update(
            "running",
            progress={"episodes_done": done, "episodes_total": total},
            elapsed_s=round(elapsed, 1),
            eta_s=None if eta is None else round(eta, 1),
        )

    status.update("running", output=str(out), limit=args.limit, workers=args.workers)
    try:
        manifest = collect.run_collection(
            collections[args.name],
            out,
            run_dir=run_dir,
            workers=args.workers,
            chunk=args.chunk,
            limit=args.limit,
            overwrite=args.overwrite,
            on_progress=progress,
            log=log,
        )
    except Exception:
        log(traceback.format_exc())
        status.update("failed", error=traceback.format_exc().splitlines()[-1])
        return 1
    status.update(
        "done",
        manifest=rel(run_dir / "manifest.json"),
        file_sha256=manifest["file_sha256"],
        outcomes=manifest["outcomes"],
        wall_s=round(manifest["wall_s"], 1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
