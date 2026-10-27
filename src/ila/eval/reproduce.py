"""The Phase 2 reproduction gate: P8's committed rows, flown again here with rendering on.

Two arms, both with render ``full`` (ground recolour, P8's marker hidden, the visual twin, a
96 px TinyRenderer frame after every reset and every ``env.step``), workers 24, chunk 25,
one thread per process (P8's run conditions), through :func:`ila.eval.runner.run_arms_vision`:

* ``pid_feedforward`` on all 2 800 listed episodes (the five frozen lists, MANIFEST order)
  against P8's ``results/e01/episodes.csv``;
* ``ppo`` seed 0 (P8's SB3 path, ``ila.p8_adapter.teacher("ppo", 0)``) on ``id`` SS5 + SS6
  (400 episodes) against ``results/e07/matrix/episodes.csv(.gz)``.

The comparison is P8's own ``compare_to_reference`` (text lines keyed by method, run seed,
pad, regime, sea state and index) on the P8-column rows, pad ``aft``. The gate additionally
requires ``n_compared`` to be the number listed and ``n_differ == n_missing == 0``, because
``all_identical`` alone would pass a short run. Nothing here loosens P8's rule.

Outputs: rows (all columns and P8 columns) under ``artifacts/runs/reproduce_p8/`` (gitignored)
with a live ``status.json``; ``comparison.json`` and ``run_info.json`` under
``results/reproduce_p8/`` (the smoke writes both to its run directory instead). Times are
wall-clock seconds; episodes are model scale (lambda = 1/25).
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.eval.runner import (
    RowTags,
    VisionArm,
    p8_rows,
    run_arms_vision,
    set_single_thread_env,
    vision_rows_csv,
)

__all__ = [
    "PPO_CELLS",
    "ReproArm",
    "arms",
    "render_evidence",
    "run",
]

#: The ``id`` cells the PPO teacher is re-flown on.
PPO_CELLS: Final[tuple[str, ...]] = ("SS5", "SS6")

_E01: Final[str] = "results/e01/episodes.csv"
_E07: Final[str] = "results/e07/matrix/episodes.csv"  # read through storage: the .csv.gz


@dataclass(frozen=True)
class ReproArm:
    """One arm of the gate.

    Attributes:
        arm: What to fly.
        reference: P8's committed logical ``episodes.csv`` (relative to P8's root).
        expected: Rows the comparison must cover (= episodes flown).
    """

    arm: VisionArm
    reference: str
    expected: int


def _rel(path: Path | str) -> str:
    """Return ``path`` relative to this repository's root (POSIX), else unchanged."""
    p = Path(path)
    try:
        return p.resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first_per_cell(
    episodes: Sequence[p8_adapter.ListedEpisode], n: int
) -> list[p8_adapter.ListedEpisode]:
    """Return the first ``n`` listed episodes of every (regime, ss) cell, list order kept."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for e in episodes:
        k = (e.regime, e.ss)
        if seen.get(k, 0) < n:
            out.append(e)
            seen[k] = seen.get(k, 0) + 1
    return out


def arms(smoke_n: int | None = None) -> list[ReproArm]:
    """Return the gate's two arms (or their first ``smoke_n`` episodes per cell).

    Args:
        smoke_n: ``None`` for the full gate; else episodes per (regime, ss) cell.

    Returns:
        ``[pid_feedforward, ppo]``.
    """
    lists = {name: p8_adapter.read_list(path) for name, path in p8_adapter.list_paths().items()}
    every = [e for name in p8_adapter.list_names() for e in lists[name]]
    ppo_eps = [e for e in lists["id"] if e.ss in PPO_CELLS]
    if smoke_n is not None:
        every = _first_per_cell(every, smoke_n)
        ppo_eps = _first_per_cell(ppo_eps, smoke_n)
    run_dir = p8_adapter.load_paths()["p8_ppo_seed0_run"]
    ppo_sha = p8_adapter.sha256_file(run_dir / "final" / "model.zip")
    pid = p8_adapter.controller_spec("pid_feedforward")
    ppo = p8_adapter.callable_spec(
        "ppo", functools.partial(p8_adapter.teacher, "ppo", 0), run_seed=0
    )
    return [
        ReproArm(VisionArm(pid, every, RowTags(arm="pid_feedforward"), "full"), _E01, len(every)),
        ReproArm(
            VisionArm(ppo, ppo_eps, RowTags(arm="ppo", ckpt_sha256=ppo_sha), "full"),
            _E07,
            len(ppo_eps),
        ),
    ]


def render_evidence(arm: VisionArm, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return what the flown rows show about rendering, and whether it is what the gate needs.

    Everything is read from the arm that was flown and from the rows' own render-evidence
    columns (``frames_rendered``, ``timed_checks``, ``render_cfg_sha256``), never from constants.

    Args:
        arm: The arm as handed to the runner.
        rows: Its rows.

    Returns:
        ``render`` (the arm's mode), the distinct ``render_cfg_sha256`` values of the rows, and
        ``episodes``, ``steps``, ``frames_rendered``, ``timed_checks`` (sums), the per-episode
        minimum of timed checks, and the checks ``render_is_full``,
        ``render_cfg_sha256_single_nonempty`` and ``frames_equal_steps_plus_episodes``.
    """
    shas = sorted({str(r["render_cfg_sha256"]) for r in rows})
    episodes = len(rows)
    steps = sum(int(r["steps"]) for r in rows)
    frames = sum(int(r["frames_rendered"]) for r in rows)
    timed = [int(r["timed_checks"]) for r in rows]
    return {
        "render": arm.render,
        "render_cfg_sha256": shas,
        "episodes": episodes,
        "steps": steps,
        "frames_rendered": frames,
        "timed_checks": sum(timed),
        "timed_checks_min_per_episode": min(timed) if timed else 0,
        "checks": {
            "render_is_full": arm.render == "full",
            "render_cfg_sha256_single_nonempty": len(shas) == 1 and shas[0] != "",
            "frames_equal_steps_plus_episodes": episodes > 0 and frames == steps + episodes,
        },
    }


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=p8_adapter.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(
    *,
    out_dir: Path,
    run_dir: Path,
    workers: int = p8_adapter.DEFAULT_WORKERS,
    chunk: int = p8_adapter.DEFAULT_CHUNK,
    smoke_n: int | None = None,
) -> dict[str, Any]:
    """Fly the gate, compare, and write rows, status, comparison and run info.

    Args:
        out_dir: Where ``comparison.json`` and ``run_info.json`` go.
        run_dir: Where rows, ``status.json`` go (gitignored).
        workers: Worker processes (P8: 24).
        chunk: Episodes per task (P8: 25).
        smoke_n: ``None`` for the full gate; else episodes per cell (pre-flight).

    Returns:
        The comparison record; ``record["passed"]`` is the gate verdict.
    """
    threads = set_single_thread_env()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    started = _now()
    status: dict[str, Any] = {
        "job": "reproduce_p8" + ("_smoke" if smoke_n is not None else ""),
        "state": "starting",
        "pid": os.getpid(),
        "started_utc": started,
        "updated_utc": started,
        "progress": None,
        "log": _rel(run_dir / "run.log"),
    }
    _write_json(status_path, status)
    try:
        gate = arms(smoke_n)
        total = sum(a.expected for a in gate)
        t_start = time.perf_counter()

        def progress(done: int, n_chunks: int, episodes: int) -> None:
            elapsed = time.perf_counter() - t_start
            status.update(
                state="running",
                updated_utc=_now(),
                progress={
                    "chunks_done": done,
                    "chunks_total": n_chunks,
                    "episodes_done": episodes,
                    "episodes_total": total,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(elapsed / episodes * (total - episodes), 1)
                    if episodes
                    else None,
                },
            )
            _write_json(status_path, status)

        status["state"] = "running"
        _write_json(status_path, status)
        rows = run_arms_vision(
            [a.arm for a in gate],
            p8_adapter.load_eval_configs(),
            workers=workers,
            chunk=chunk,
            on_chunk=progress,
        )
        wall_s = time.perf_counter() - t_start
        comparisons: dict[str, Any] = {}
        row_sha: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        start = 0
        passed = True
        for a in gate:
            method = a.arm.spec.method
            mine = rows[start : start + a.expected]
            start += a.expected
            if [r["method"] for r in mine] != [method] * a.expected:
                raise RuntimeError(f"{method}: rows out of order")
            full_text = vision_rows_csv(mine)
            p8_text = p8_adapter.episode_rows_csv(p8_rows(mine))
            (run_dir / f"episodes_{method}.csv").write_text(full_text, encoding="utf-8")
            (run_dir / f"episodes_{method}_p8cols.csv").write_text(p8_text, encoding="utf-8")
            row_sha[method] = {
                "episodes_csv_sha256": _sha(full_text),
                "p8_columns_csv_sha256": _sha(p8_text),
                "n_rows": len(mine),
            }
            result = p8_adapter.compare_to_reference(
                p8_rows(mine), p8_adapter.P8_ROOT / a.reference, pad="aft"
            )
            result["reference"] = _rel(result["reference"])  # repo-relative, not host paths
            stats = result["methods"].get(method)
            evidence[method] = render_evidence(a.arm, mine)
            checks = {
                "all_identical": bool(result["all_identical"]),
                "header_identical": bool(result["header_identical"]),
                "n_compared_is_expected": stats is not None and stats["n_compared"] == a.expected,
                "n_differ_is_zero": stats is not None and stats["n_differ"] == 0,
                "n_missing_is_zero": stats is not None and stats["n_missing_in_reference"] == 0,
                **evidence[method]["checks"],
            }
            ok = all(checks.values())
            passed = passed and ok
            comparisons[method] = {
                "expected": a.expected,
                "checks": checks,
                "passed": ok,
                "compare_to_reference": result,
            }
        record = {
            "gate": "Phase 2 reproduction (plan Phase 2 item 2)",
            "smoke": smoke_n is not None,
            "smoke_per_cell": smoke_n,
            "passed": passed,
            "methods": comparisons,
        }
        info = {
            "started_utc": started,
            "finished_utc": _now(),
            "wall_s": round(wall_s, 1),
            "episodes_flown": len(rows),
            "p8_commit": p8_adapter.p8_commit(),
            "ila_commit": _git("rev-parse", "HEAD"),
            "ila_dirty_paths": [ln for ln in _git("status", "--porcelain").splitlines() if ln][:50],
            "host": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "threads": {**threads, "torch_num_threads": 1},
            "workers": workers,
            "chunk": chunk,
            "start_method": "spawn",
            "render": {m: e["render"] for m, e in evidence.items()},
            "render_cfg_sha256": {m: e["render_cfg_sha256"] for m, e in evidence.items()},
            "render_evidence": evidence,
            "manifest_sha256": p8_adapter.manifest_sha256(),
            "success_yaml_sha256": p8_adapter.success_yaml_sha256(),
            "ppo_model_sha256": gate[1].arm.tags.ckpt_sha256,
            "rows": row_sha,
            "rows_dir": _rel(run_dir),
        }
        _write_json(out_dir / "comparison.json", record)
        _write_json(out_dir / "run_info.json", info)
        status.update(state="done" if passed else "failed", updated_utc=_now(), passed=passed)
        _write_json(status_path, status)
        return record
    except BaseException as exc:
        status.update(state="failed", updated_utc=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_json(status_path, status)
        raise
