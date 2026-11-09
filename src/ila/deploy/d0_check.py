"""The latency arm at ``d = 0`` on the real pinned checkpoints equals selection, byte for byte.

P3-D1 §7: "d = 0: the arm equals the synchronous arm row for row. This is tested." The unit
tests (``tests/test_deploy_latency_arm.py``) fly dummy students; this check flies the arms the
latency arm actually flies.

* **Arms**: ``bc_dagger`` and ``dp`` (``Ta = 8``, ``K = 10``, no temporal ensembling), seed 0,
  the pinned, byte-verified checkpoints of ``results/e08/selection_pins.json``
  (:func:`ila.deploy.export.load_arm`).
* **Policy**: the latency arm's own build at ``d = 0``,
  ``functools.partial(delayed_checkpoint_policy, path, 0, n_action_steps=8,
  num_inference_steps=10)`` (``bc_dagger``: no ``Ta``/``K``), exactly what
  :func:`ila.eval.matrix.latency_flight` builds for ``d > 0``: the EMA student on the CPU,
  fp32, wrapped by :func:`ila.deploy.latency_arm.wrap_delayed`.
* **Flight**: the runner call of the closed-loop reference flight
  (:func:`ila.deploy.closed_loop.fly_provider` -> :func:`ila.train.select.fly_checkpoint`):
  :func:`ila.eval.runner.run_list_vision` on the 150 ``select`` episodes, render ``full``, the
  checkpoint's contract variant, one torch thread per worker, ``chunk`` episodes per task.
  Rows do not depend on the worker count (P2-D3 item 1).
* **Tags**: selection's (``arm``, ``teacher``, ``student_seed``, ``ckpt_sha256``;
  ``latency_steps = 0``; ``Ta``/``K``/``ensemble`` empty), so the two files can be compared
  byte for byte. The matrix's latency rows also fill ``Ta``/``K``/``ensemble`` for ``dp``;
  those are labels, not behaviour.
* **Comparison**: the rows file (written as the reference flight writes it:
  :func:`ila.deploy.closed_loop.rows_text`, CRLF row endings) against the committed selection
  rows of the same checkpoint (located from its ``selection.json`` as the reference flight
  locates them), by :func:`ila.deploy.closed_loop.rows_files_identical` (bytes, as ``cmp``).

Outputs: ``results/latency/d0_check/<arm>_rows.csv`` and ``results/latency/d0_check.json``
(both SHA-256s, ``n``, ``identical``, outcome flips and differing rows beside, the ``ila``
commit and the dirty flag). :func:`verify_record` re-decides the record from the files on disk
without flying (``make bench-check``). Simulation only; control steps are 1/30 s model scale.
"""

from __future__ import annotations

import csv
import datetime as dt
import functools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.deploy import closed_loop as cl
from ila.deploy import latency_arm as la
from ila.deploy.export import load_arm, sha256_file
from ila.deploy.settings import DEPLOY_ARMS, BenchSettings
from ila.eval import runner

__all__ = [
    "D0_DIR",
    "D0_FILE",
    "D0Flight",
    "compare_rows_files",
    "d0_spec",
    "fly_d0",
    "run_d0_check",
    "verify_record",
]

#: The record, under ``results/latency/``.
D0_FILE: Final[str] = "d0_check.json"

#: The rows files, under ``results/latency/``.
D0_DIR: Final[str] = "d0_check"

#: The comparison method, as recorded.
METHOD: Final[str] = "bytes (Path.read_bytes equality, as cmp; closed_loop.rows_files_identical)"


@dataclass(frozen=True)
class D0Flight:
    """One arm flown at ``d = 0``, and what it is compared with.

    Attributes:
        rows: Runner rows, in ``select`` order.
        checkpoint_sha256: The pinned checkpoint flown.
        selection_rows: The committed selection rows file of the same checkpoint.
        build: How the policy was built (recorded).
    """

    rows: list[dict[str, Any]]
    checkpoint_sha256: str
    selection_rows: Path
    build: Mapping[str, Any]


def d0_spec(
    arm: str,
    ckpt_path: Path,
    *,
    run_seed: int,
    sha256: str,
    teacher: str,
    settings: BenchSettings,
) -> tuple[p8_adapter.PolicySpec, runner.RowTags, dict[str, Any]]:
    """The latency arm's policy spec at ``d = 0`` and selection's row tags (module docstring).

    Args:
        arm: ``bc_dagger`` or ``dp``.
        ckpt_path: The pinned checkpoint.
        run_seed: Its run seed (``dp``: first element of every noise key).
        sha256: Its SHA-256 (row column).
        teacher: Its training teacher (row column).
        settings: Bench settings (``dp``: ``Ta``, ``K``).

    Returns:
        ``(spec, tags, build record)``.

    Raises:
        ValueError: For an arm the latency arm does not fly.
    """
    if arm not in DEPLOY_ARMS:
        raise ValueError(f"the latency arm flies {DEPLOY_ARMS}, not {arm!r}")
    kw: dict[str, int] = (
        {"n_action_steps": settings.dp_ta, "num_inference_steps": settings.dp_k}
        if arm == "dp"
        else {}
    )
    build = (
        functools.partial(
            la.delayed_checkpoint_policy,
            str(ckpt_path),
            0,
            n_action_steps=settings.dp_ta,
            num_inference_steps=settings.dp_k,
        )
        if arm == "dp"
        else functools.partial(la.delayed_checkpoint_policy, str(ckpt_path), 0)
    )
    spec = p8_adapter.callable_spec(arm, build, run_seed=run_seed)
    tags = runner.RowTags(arm=arm, teacher=teacher, student_seed=run_seed, ckpt_sha256=sha256)
    record = {
        "policy": "ila.deploy.latency_arm.delayed_checkpoint_policy",
        "delay": 0,
        **kw,
        **({"temporal_ensembling": False} if arm == "dp" else {}),
        "weights": "ema",
        "device": "cpu",
        "tags": "selection's (Ta, K, ensemble empty; latency_steps 0)",
    }
    return spec, tags, record


def fly_d0(
    arm: str,
    settings: BenchSettings,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    workers: int,
) -> D0Flight:
    """Fly one arm's pinned seed checkpoint at ``d = 0`` (module docstring)."""
    loaded = load_arm(arm, settings.arm_seeds[arm], settings.selection_pins)
    ck = loaded.ckpt
    spec, tags, build = d0_spec(
        arm,
        ck.path,
        run_seed=loaded.seed,
        sha256=ck.sha256,
        teacher=ck.teacher,
        settings=settings,
    )
    rows = runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="full",
        workers=workers,
        chunk=settings.chunk,
        contract_variant=loaded.obs_variant,
    )
    sel = cl._selection_rows_path(p8_adapter.REPO_ROOT / ck.selection_json)
    return D0Flight(rows, ck.sha256, sel, build)


def _n_rows(path: Path) -> int:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def compare_rows_files(rows_path: Path, selection_path: Path) -> dict[str, Any]:
    """Both files' SHA-256, row counts and byte identity; flips and differing rows beside.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    identical = cl.rows_files_identical(rows_path, selection_path)
    rows, sel = _read_rows(rows_path), _read_rows(selection_path)
    try:
        fl = cl.count_flips(rows, sel)
        flips: int | None = int(fl["flips"])
        differ: int | None = int(fl["rows_text_differ"])
    except ValueError:  # not the same episodes in the same order
        flips = differ = None
    return {
        "identical": identical,
        "n": len(rows),
        "selection_n": len(sel),
        "rows_file": _rel(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "selection_rows_file": _rel(selection_path),
        "selection_rows_sha256": sha256_file(selection_path),
        "outcome_flips": flips,
        "rows_text_differ": differ,
    }


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


type Flyer = Callable[[str, BenchSettings, Sequence[p8_adapter.ListedEpisode], int], D0Flight]


def _default_fly(
    arm: str,
    settings: BenchSettings,
    episodes: Sequence[p8_adapter.ListedEpisode],
    workers: int,
) -> D0Flight:
    return fly_d0(arm, settings, episodes, workers=workers)


def run_d0_check(
    settings: BenchSettings,
    *,
    git: Mapping[str, Any],
    arms: Sequence[str] = DEPLOY_ARMS,
    workers: int | None = None,
    episodes: Sequence[p8_adapter.ListedEpisode] | None = None,
    fly: Flyer = _default_fly,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly each arm at ``d = 0``, write its rows, compare, and write ``d0_check.json``.

    An arm not flown this time keeps its entry from an existing record (with its own commit).

    Args:
        settings: Bench settings (``results_dir``, ``arm_seeds``, ``selection_pins``, ``chunk``).
        git: ``ila.deploy.export.git_state()`` (plus ``dirty_outside_results``) of this run.
        arms: Arms to fly.
        workers: Processes (default ``settings.workers_cpu``; rows do not depend on it).
        episodes: The listed episodes (default: the 150 ``select`` episodes).
        fly: The flight (tests inject one).
        log: Progress sink.

    Returns:
        The record written.

    Raises:
        ValueError: For an arm the latency arm does not fly.
    """
    for arm in arms:
        if arm not in DEPLOY_ARMS:
            raise ValueError(f"the latency arm flies {DEPLOY_ARMS}, not {arm!r}")
    eps = list(data_episodes.read_split("select") if episodes is None else episodes)
    n_workers = settings.workers_cpu if workers is None else int(workers)
    out_dir = settings.results_dir / D0_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = settings.results_dir / D0_FILE
    old: dict[str, Any] = (
        json.loads(rec_path.read_text("utf-8")) if rec_path.is_file() else {"arms": {}}
    )
    git_rec = {
        "ila_commit": str(git["ila_commit"]),
        "dirty": bool(git["dirty"]),
        "dirty_outside_results": bool(git.get("dirty_outside_results", git["dirty"])),
    }
    entries: dict[str, Any] = dict(old.get("arms", {}))
    for arm in arms:
        t0 = time.perf_counter()
        flight = fly(arm, settings, eps, n_workers)
        rows_path = out_dir / f"{arm}_rows.csv"
        rows_path.write_text(cl.rows_text(flight.rows), "utf-8")  # as the reference flight
        cmp = compare_rows_files(rows_path, flight.selection_rows)
        entries[arm] = {
            **cmp,
            "checkpoint_sha256": flight.checkpoint_sha256,
            "build": dict(flight.build),
            "workers": n_workers,
            "wall_s": time.perf_counter() - t0,
            "time_utc": _now(),
            **git_rec,
        }
        log(
            f"{arm} d=0: {cmp['n']} rows; identical to selection rows: {cmp['identical']} "
            f"({cmp['rows_sha256'][:12]} vs {cmp['selection_rows_sha256'][:12]})"
        )
    entries = {a: entries[a] for a in DEPLOY_ARMS if a in entries}
    rec = {
        "claim": "P3-D1 §7: d = 0: the latency arm equals the synchronous arm row for row",
        "what": (
            "the latency arm's delayed build at d = 0 on the seed-0 pinned checkpoints, flown "
            "on the 150 select episodes through the closed-loop reference flight's runner call "
            "(CPU fp32 single-sample path, render full), compared with the committed selection "
            "rows of the same checkpoint"
        ),
        "method": METHOD,
        "list": "select",
        "arms": entries,
        "identical": bool(entries)
        and set(entries) == set(DEPLOY_ARMS)
        and all(bool(e["identical"]) for e in entries.values()),
        **git_rec,
        "time_utc": _now(),
    }
    rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")
    return rec


def verify_record(settings: BenchSettings) -> list[str]:
    """Re-decide ``d0_check.json`` from the files on disk (no flight); the problems found.

    Nothing to verify (an empty list) when the record does not exist. Otherwise every arm of
    :data:`ila.deploy.settings.DEPLOY_ARMS` must be present and identical, its rows file must
    hash to the recorded SHA-256, the two recorded SHA-256s must be equal, and the selection
    rows file, when present on this machine (``bc_dagger``'s lives in its run directory under
    ``artifacts/``), must hash to the recorded SHA-256 and equal the rows file byte for byte.
    """
    path = settings.results_dir / D0_FILE
    if not path.is_file():
        return []
    rec: dict[str, Any] = json.loads(path.read_text("utf-8"))
    bad: list[str] = []
    arms: Mapping[str, Any] = rec.get("arms", {})
    for arm in DEPLOY_ARMS:
        e = arms.get(arm)
        if e is None:
            bad.append(f"{D0_FILE}: {arm} not flown")
            continue
        if e.get("identical") is not True:
            bad.append(f"{D0_FILE}: {arm} rows differ from its selection rows")
        rows_path = settings.results_dir / D0_DIR / f"{arm}_rows.csv"
        if not rows_path.is_file():
            bad.append(f"{D0_FILE}: {arm} rows file missing")
            continue
        if sha256_file(rows_path) != e.get("rows_sha256"):
            bad.append(f"{D0_FILE}: {arm} rows file differs from the recorded SHA-256")
        if _n_rows(rows_path) != e.get("n"):
            bad.append(f"{D0_FILE}: {arm} rows file has not the recorded n")
        if bool(e.get("identical")) != (e.get("rows_sha256") == e.get("selection_rows_sha256")):
            bad.append(f"{D0_FILE}: {arm} identical disagrees with the recorded SHA-256s")
        sel = Path(str(e.get("selection_rows_file", "")))
        sel = sel if sel.is_absolute() else p8_adapter.REPO_ROOT / sel
        if sel.is_file():
            if sha256_file(sel) != e.get("selection_rows_sha256"):
                bad.append(f"{D0_FILE}: {arm} selection rows differ from the recorded SHA-256")
            if cl.rows_files_identical(rows_path, sel) != bool(e.get("identical")):
                bad.append(f"{D0_FILE}: {arm} identical disagrees with the files' bytes")
    want = (
        bool(arms)
        and set(arms) == set(DEPLOY_ARMS)
        and all(bool(e.get("identical")) for e in arms.values())
    )
    if bool(rec.get("identical")) != want:
        bad.append(f"{D0_FILE}: top-level identical disagrees with its arms")
    return bad
