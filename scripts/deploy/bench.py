"""Phase 9 bench: export, inputs, parity, closed-loop, latency, score, check, d0-check.

The stages live in ila.deploy.pipeline.

Examples (see the Makefile ``bench-*`` targets; long stages run in the background):

    .venv/bin/python scripts/deploy/bench.py export
    .venv/bin/python scripts/deploy/bench.py inputs
    .venv/bin/python scripts/deploy/bench.py parity
    .venv/bin/python scripts/deploy/bench.py closed-loop --arms bc_dagger dp
    .venv/bin/python scripts/deploy/bench.py latency --batches 1
    .venv/bin/python scripts/deploy/bench.py score
    .venv/bin/python scripts/deploy/bench.py check
    .venv/bin/python scripts/deploy/bench.py recheck-reference --arms bc_dagger dp
    .venv/bin/python scripts/deploy/bench.py d0-check --arms bc_dagger dp --workers 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ila.deploy import pipeline
from ila.deploy.settings import BENCH_YAML, load_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--settings", type=Path, default=BENCH_YAML)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("export", "inputs", "parity", "closed-loop", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--allow-dirty", action="store_true", help="smoke runs only")
        if name == "inputs":
            p.add_argument("--workers", type=int, default=24)
        if name in ("parity", "closed-loop", "latency"):
            p.add_argument("--providers", nargs="*", default=None)
        if name == "closed-loop":
            p.add_argument("--arms", nargs="*", default=["bc_dagger", "dp"])
            p.add_argument("--include-failed", action="store_true")
        if name in ("closed-loop", "latency"):
            p.add_argument("--force", action="store_true")
        if name == "latency":
            p.add_argument("--rows", nargs="*", default=None)
            p.add_argument("--batches", nargs="*", type=int, default=None)
    sub.add_parser("score")
    sub.add_parser("check")
    p = sub.add_parser("recheck-reference")
    p.add_argument("--arms", nargs="*", default=["bc_dagger", "dp"])
    p.add_argument("--allow-dirty", action="store_true", help="smoke runs only")
    p = sub.add_parser("d0-check")
    p.add_argument("--arms", nargs="*", default=["bc_dagger", "dp"])
    p.add_argument("--workers", type=int, default=None, help="default: bench.yaml workers_cpu")
    p.add_argument("--allow-dirty", action="store_true", help="smoke runs only")
    a = ap.parse_args(argv)
    s = load_settings(a.settings)

    def log(msg: str) -> None:
        print(msg, flush=True)

    if a.stage == "export":
        rec = pipeline.stage_export(s, allow_dirty=a.allow_dirty)
        for name, g in sorted(rec["graphs"].items()):
            log(f"{name}: {g['sha256']}")
    elif a.stage == "inputs":
        rec = pipeline.stage_inputs(s, workers=a.workers, allow_dirty=a.allow_dirty)
        log(json.dumps({k: rec[k] for k in ("sha256", "n", "windows_total", "teacher_outcomes")}))
    elif a.stage == "parity":
        pipeline.stage_parity(s, providers=a.providers, allow_dirty=a.allow_dirty, log=log)
        log((s.results_dir / "parity.csv").read_text("utf-8"))
    elif a.stage == "closed-loop":
        pipeline.stage_closed_loop(
            s,
            arms=a.arms,
            providers=a.providers,
            include_failed=a.include_failed,
            force=a.force,
            allow_dirty=a.allow_dirty,
            log=log,
        )
        log((s.results_dir / "closed_loop.csv").read_text("utf-8"))
    elif a.stage == "latency":
        pipeline.stage_latency(
            s,
            providers=a.providers,
            rows=a.rows,
            batches=a.batches,
            force=a.force,
            allow_dirty=a.allow_dirty,
            log=log,
        )
    elif a.stage == "recheck-reference":
        pipeline.stage_recheck_reference(s, arms=a.arms, allow_dirty=a.allow_dirty, log=log)
        log((s.results_dir / "closed_loop.csv").read_text("utf-8"))
    elif a.stage == "d0-check":
        rec = pipeline.stage_d0_check(
            s, arms=a.arms, workers=a.workers, allow_dirty=a.allow_dirty, log=log
        )
        log(json.dumps({arm: rec["arms"][arm]["identical"] for arm in a.arms}))
        if not all(rec["arms"][arm]["identical"] for arm in a.arms):
            log("d = 0 rows differ from the selection rows (results/latency/d0_check.json)")
            return 1
    elif a.stage == "score":
        log(json.dumps(pipeline.stage_score(s), indent=2))
    else:
        bad = pipeline.check(s)
        d0_bad = pipeline.check_d0(s)
        if bad:
            log(f"differ from their records: {bad}")
        if d0_bad:
            log(f"d0_check.json disagrees with its files: {d0_bad}")
        if bad or d0_bad:
            return 1
        log("results/latency re-renders byte-identically")
        if (s.results_dir / "d0_check.json").is_file():
            log("d0_check.json agrees with its rows files (d = 0 equals selection, bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
