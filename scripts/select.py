"""Closed-loop checkpoint selection on ``splits/select.parquet`` (Phase 4). Background job.

Flies every candidate checkpoint (EMA weights) of a finished run on the 150 ``select``
episodes through the runner's CPU fp32 single-sample path and writes ``selection.json``
(success, outcome classes, pooled p95 closing speed per candidate; the rule's choice) plus
per-candidate rows under ``<run>/select/``. ``make select-bg`` wraps it. Examples::

    python scripts/select.py --arm bc --seeds 0 1 2 3 4
    python scripts/select.py artifacts/runs/bc/s0
    python scripts/select.py --dart-trials        # the three sigma trials, then sigma*

``--resummarise`` flies nothing: it rebuilds each named run's ``selection.json`` from its stored
``select/*_rows.csv`` (adding ``n_tunnelled`` and ``n_detectors_disagree``) and refuses, writing
nothing, if any candidate's success, outcomes, p95 or the selected checkpoint would change.
``--update-summary CSV`` then adds those two columns (last, existing columns and their bytes kept)
to a runs summary whose ``run`` column names run directories under ``--run-root``, after checking
that its selection columns agree with the rebuilt records::

    python scripts/select.py --resummarise --all --update-summary results/phase4/runs_summary.csv

``--dart-trials`` selects each ``bc_dart`` sigma trial that has no ``selection.json`` yet, then
applies P3-D1 §4's sigma* rule (ties: the smaller sigma) and writes
``artifacts/runs/bc_dart_tune/sigma_star.json``, which ``scripts/train.py`` requires before any
core ``bc_dart`` run. Timing, smoke and test runs are refused.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from ila.train import manifest as mf
from ila.train.config import DART_SIGMAS
from ila.train.select import choose_sigma, resummarise_run, select_run

#: Runs-summary columns added by ``--update-summary``, appended in this order.
FLAG_COLUMNS = ("n_tunnelled", "n_detectors_disagree")


def _all_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("**/selection.json"))


def _update_summary(path: Path, root: Path, records: dict[Path, dict[str, object]]) -> None:
    """Append FLAG_COLUMNS to a runs summary after cross-checking its selection columns."""
    with path.open(newline="", encoding="utf-8") as f:
        table = list(csv.reader(f))
    header, body = table[0], table[1:]
    base = [c for c in header if c not in FLAG_COLUMNS]
    if header[: len(base)] != base:
        raise SystemExit(f"{path}: flag columns must be last")
    col = {c: i for i, c in enumerate(header)}
    out = [base + list(FLAG_COLUMNS)]
    seen = set()
    for row in body:
        run = (root / row[col["run"]]).resolve()
        rec = records.get(run)
        if rec is None:
            raise SystemExit(f"{path}: run {row[col['run']]} was not re-summarised")
        seen.add(run)
        sel = rec["selected"]
        assert isinstance(sel, dict)
        checks = {
            "selected_checkpoint": Path(sel["candidate"]["file"]).name,
            "selected_global_step": str(sel["candidate"]["global_step"]),
            "select_success": str(sel["success"]),
            "select_n": str(sel["n"]),
            "selected_ckpt_sha256": sel["candidate"]["sha256"],
        }
        for name, want in checks.items():
            if name in col and row[col[name]] != want:
                raise SystemExit(f"{row[col['run']]}: {name} {row[col[name]]} != {want}")
        if "p95_closing_speed_m_s" in col:
            p95 = sel["p95_closing_speed_m_s"]
            have = row[col["p95_closing_speed_m_s"]]
            same = have in ("", "None", "inf") if p95 is None else float(have) == float(p95)
            if not same:
                raise SystemExit(f"{row[col['run']]}: p95 {have} != {p95}")
        out.append([row[col[c]] for c in base] + [str(sel[c]) for c in FLAG_COLUMNS])
    if seen != set(records):
        raise SystemExit(f"{path}: rows {len(seen)} != re-summarised runs {len(records)}")
    buf = io.StringIO()
    csv.writer(buf).writerows(out)  # csv's default \r\n terminator, as the file was written
    path.write_text(buf.getvalue(), encoding="utf-8", newline="")


def _one(run_dir: Path, workers: int, redo: bool) -> dict[str, object]:
    out = run_dir / "selection.json"
    if out.is_file() and not redo:
        print(f"{run_dir}: selection.json exists (use --redo to fly again)", flush=True)
        return mf.read_json(out)
    rec = select_run(run_dir, workers=workers)
    sel = rec["selected"]
    assert isinstance(sel, dict)
    print(
        f"{run_dir}: selected {Path(sel['candidate']['file']).name} "
        f"success {sel['success']}/{sel['n']} outcomes {sel['outcomes']} "
        f"p95 {sel['p95_closing_speed_m_s']}",
        flush=True,
    )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dart-trials", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--resummarise", action="store_true", help="rebuild from rows, no flight")
    parser.add_argument("--all", action="store_true", help="with --resummarise: every run")
    parser.add_argument("--update-summary", type=Path, default=None)
    args = parser.parse_args()
    if args.resummarise:
        targets = list(args.runs) + (_all_runs(args.run_root) if args.all else [])
        if not targets:
            parser.error("--resummarise needs run directories or --all")
        checked = {}
        for run in targets:  # check every run before writing any
            checked[run.resolve()] = resummarise_run(run, write=False)
        for run in targets:
            rec = resummarise_run(run)
            sel = rec["selected"]
            print(
                f"{run}: {Path(sel['candidate']['file']).name} success {sel['success']}/"
                f"{sel['n']} n_tunnelled {sel['n_tunnelled']} n_detectors_disagree "
                f"{sel['n_detectors_disagree']}",
                flush=True,
            )
        if args.update_summary is not None:
            _update_summary(args.update_summary, args.run_root, checked)
        return 0
    runs = list(args.runs)
    if args.arm is not None:
        runs += [mf.run_dir(args.arm, s, root=args.run_root) for s in (args.seeds or [])]
    if args.dart_trials:
        records = {}
        for sigma in DART_SIGMAS:
            run = mf.run_dir("bc_dart", 0, root=args.run_root, sigma=sigma, tuning=True)
            records[sigma] = _one(run, args.workers, args.redo)
        star = choose_sigma(records)
        path = args.run_root / "bc_dart_tune" / "sigma_star.json"
        mf.write_json_atomic(path, star)
        print(json.dumps({"sigma_star": star["sigma_star"], "file": str(path)}), flush=True)
    for run in runs:
        _one(run, args.workers, args.redo)
    if not runs and not args.dart_trials:
        parser.error("nothing to select")
    return 0


if __name__ == "__main__":
    sys.exit(main())
