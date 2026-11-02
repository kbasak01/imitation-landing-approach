"""H2 touchdown-time distributions on ``select``: plotted only, never scored (plan Phase 7 item 3).

Everything here is simulation only (PyBullet physics; no frame is involved, see below). Times
are seconds, model scale (lambda = 1/25); full scale is x5.

What this is, and what it is not
--------------------------------
* **Not H2.** H2 is scored in Phase 8 on ``id`` SS4 (P3-D1 §6), with a reference built from
  P8's committed rows and a W1 bootstrap. Here W1 is a **descriptive, unscored** distance on
  ``select``, with no interval, and nothing is chosen from it.
* **The list.** ``splits/select.parquet`` only (150 ``tune`` episodes, 50 each of SS3, SS4,
  SS5; SHA-256 checked by :func:`ila.data.episodes.read_split`). No P8 frozen list is read
  (``tests/test_eval_h2_select.py`` scans this module and its script for P8's list accessors).

The ``T_mix`` reference on ``select``
-------------------------------------
:func:`tmix_assignment` applies :func:`ila.data.mix.assign_rows` with the ``T_mix`` seed of
``configs/data/demos.yaml`` (20261023, the seed ``mix_1000`` was collected with) to each sea
state's ``select`` rows, one cell per call, as P3-D1 §2 defines the draw. With 50 episodes per
cell there is no odd surplus, so this equals one call on the whole list (asserted): 25 ``ppo``
and 25 ``gated`` per sea state. :func:`fly_reference` then flies P8's ``ppo`` seed 0 (the SB3
model with P8's frozen VecNormalize, ``training=False``) and P8's ``gated`` controller on their
assigned episodes through :func:`ila.eval.runner.run_arms_vision`, the runner of the demos and
of the Phase 2 reproduction gate, render ``off`` (teachers discard frames; Gate 2 shows
rendering does not change physics), each policy reset on every episode. Rows do not depend on
the worker count.

Students
--------
``bc_mix`` and ``dp_mix`` (3 seeds each, P3-D1 §3 ablations, distilled from ``T_mix``:
P8 ``ppo`` s0 + P8 ``gated``; ``dp_mix`` at the frozen ``Ta`` = 8, K = 10) are **not flown**.
Their rows are the ``select`` rows of each run's selected checkpoint, written by
:func:`ila.train.select.select_run` under ``<run_root>/<arm>/s<seed>/select/`` and named by
``selection.json``'s ``selected.rows_file``. :func:`load_student` verifies them
(:func:`verify_selection`) and reports a run without ``selection.json`` as **missing**, never
skipping it silently. Those rows are **in-sample**: the checkpoint was chosen for its success
on these same 150 episodes.

Outputs (:func:`build`, then :func:`render`)
--------------------------------------------
* ``h2_touchdown_select_episodes.csv``: one row per (arm, seed, episode), the figure's and the
  summary's only data input;
* ``h2_touchdown_select_sources.json``: provenance (reference file SHA-256, every student
  run's status and verified SHA-256s, the missing list);
* ``h2_touchdown_select.csv`` / ``.json``: per group (``T_mix``, ``T_mix:ppo``,
  ``T_mix:gated``, ``bc_mix``, ``dp_mix``; students pooled over their seeds, and per seed) and
  per cell (SS3, SS4, SS5, all): episodes, touchdowns, the fraction without a touchdown, the
  six outcome classes, p10 / p50 / p90 of touched-down ``time_to_touchdown_s`` (P8's
  ``quantile_linear``) and the descriptive W1 to the ``T_mix`` reference of the same cell
  (:func:`ila.eval.metrics.wasserstein_1`);
* ``h2_touchdown_select.png``: ECDF and histogram panels per cell.

:func:`render` reads only the two committed files above, so ``render`` re-creates the summary
and the figure byte for byte (PNG written without a software/date stamp).
"""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import textwrap
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.data import mix
from ila.eval import runner
from ila.eval.metrics import wasserstein_1
from ila.train import manifest as mf
from ila.train import select as train_select

__all__ = [
    "ALL_CELL",
    "CAPTION",
    "CELLS",
    "EPISODE_COLUMNS",
    "EPISODES_FILE",
    "FIGURE_FILE",
    "GROUPS",
    "OUTCOME_ORDER",
    "QUANTILES",
    "REFERENCE_FILE",
    "REFERENCE_GROUP",
    "SOURCES_FILE",
    "STUDENT_ARMS",
    "STUDENT_SEEDS",
    "SUMMARY_COLUMNS",
    "SUMMARY_CSV",
    "SUMMARY_JSON",
    "TMIX_SEED",
    "EpisodeRecord",
    "H2SourceError",
    "StudentStatus",
    "build",
    "fly_reference",
    "load_reference",
    "load_student",
    "read_episodes",
    "render",
    "summarise",
    "tmix_assignment",
    "verify_selection",
    "write_reference",
]

#: The ``T_mix`` assignment seed (P3-D1 §2; ``configs/data/demos.yaml`` ``teachers.T_mix.seed``,
#: checked against the file by :func:`tmix_seed_from_config`).
TMIX_SEED: Final[int] = 20261023

#: The reference's group label and its two modes (``ila.data.mix.MIX_POLICIES`` order).
REFERENCE_GROUP: Final[str] = "T_mix"
MODE_GROUPS: Final[tuple[str, str]] = ("T_mix:ppo", "T_mix:gated")

#: Students distilled from ``T_mix`` (P3-D1 §3 ablations) and their seeds.
STUDENT_ARMS: Final[tuple[str, str]] = ("bc_mix", "dp_mix")
STUDENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

#: Every group, in table and legend order.
GROUPS: Final[tuple[str, ...]] = (REFERENCE_GROUP, *MODE_GROUPS, *STUDENT_ARMS)

#: Sea-state cells of ``select`` and the pooled cell.
CELLS: Final[tuple[str, ...]] = ("SS3", "SS4", "SS5")
ALL_CELL: Final[str] = "all"

#: Touchdown-time quantiles reported (P8's ``quantile_linear``).
QUANTILES: Final[tuple[float, ...]] = (0.10, 0.50, 0.90)

#: P8's outcome classes, in P8's order (P3-D1 §1).
OUTCOME_ORDER: Final[tuple[str, ...]] = train_select.OUTCOME_ORDER

#: The frozen ``dp`` inference setting selection flies (P3-D1 §3/§4).
DP_INFERENCE: Final[Mapping[str, Any]] = {"Ta": 8, "K": 10, "temporal_ensembling": False}

#: Output file names (in ``results/phase7/h2/``).
REFERENCE_FILE: Final[str] = "tmix_select_rows.csv"
REFERENCE_INFO_FILE: Final[str] = "tmix_select_run_info.json"
EPISODES_FILE: Final[str] = "h2_touchdown_select_episodes.csv"
SOURCES_FILE: Final[str] = "h2_touchdown_select_sources.json"
SUMMARY_CSV: Final[str] = "h2_touchdown_select.csv"
SUMMARY_JSON: Final[str] = "h2_touchdown_select.json"
FIGURE_FILE: Final[str] = "h2_touchdown_select.png"

#: Default output directory.
OUT_DIR: Final[Path] = p8_adapter.REPO_ROOT / "results" / "phase7" / "h2"

#: Columns of the reference rows file: list position and assigned teacher, then the runner's.
REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "select_position",
    "tmix_assigned",
    *runner.VISION_EPISODE_COLUMNS,
)

#: Columns of the per-episode table (:data:`EPISODES_FILE`).
EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "seed",
    "tmix_assigned",
    "select_position",
    "ss",
    "index",
    "episode_seed",
    "outcome",
    "time_to_touchdown_s",
    "ckpt_sha256",
)

#: Columns of the summary table (:data:`SUMMARY_CSV`); times in seconds, model scale.
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "level",
    "seeds",
    "cell",
    "status",
    "n_episodes",
    "n_touchdown",
    "n_without_touchdown",
    "frac_without_touchdown",
    *(f"n_{o}" for o in OUTCOME_ORDER),
    *(f"p{round(q * 100)}_model_s" for q in QUANTILES),
    "w1_to_tmix_ref_model_s",
)

#: Labels that travel with every output (CSV header comment lines are avoided: the JSON and
#: the figure carry them).
CAPTION: Final[tuple[str, ...]] = (
    "Simulated: PyBullet physics; P8 success criteria and outcome classes. Not a frozen list: "
    "splits/select.parquet (150 P8 tune-pool episodes, 50 per sea state).",
    "Model scale, lambda = 1/25: times x5 for full scale.",
    "bc_mix and dp_mix are distilled from T_mix (P8 ppo s0 + gated, balanced per sea state); "
    "dp_mix at Ta = 8, K = 10. Students are pooled over seeds 0-2; their rows are the select "
    "rows of the checkpoint selected on these same episodes (in-sample).",
    "W1 is descriptive and unscored, on select, without an interval. The scored H2 is id SS4 "
    "in Phase 8.",
    "dp's failures at Ta = 8 are lateral (off_pad, Phase 6) and do not depend on sea state; do "
    "not read them as multimodality effects without checking.",
    "P8's teacher saw the tune realizations in P8's own tuning; its select rows may read high.",
)


class H2SourceError(ValueError):
    """A student run's ``select`` rows do not match its ``selection.json`` (or the list)."""


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode of one group.

    Attributes:
        arm: ``T_mix`` for the reference, else the student arm.
        seed: Student training seed, or ``None`` for the reference.
        tmix_assigned: ``ppo`` / ``gated`` for the reference, ``""`` for a student.
        select_position: Position in ``select`` (0-149).
        ss: Sea state (``SS3``-``SS5``).
        index: The list row's ``index``.
        episode_seed: The listed episode seed.
        outcome: P8 outcome class.
        time_to_touchdown_s: P8's contact touchdown time, s model scale; NaN without touchdown.
        ckpt_sha256: The flown checkpoint's SHA-256 (``ppo``'s ``model.zip`` for the
            ``ppo``-assigned reference; ``""`` for ``gated``).
    """

    arm: str
    seed: int | None
    tmix_assigned: str
    select_position: int
    ss: str
    index: int
    episode_seed: int
    outcome: str
    time_to_touchdown_s: float
    ckpt_sha256: str

    @property
    def touched_down(self) -> bool:
        """Whether P8 recorded a contact touchdown time."""
        return math.isfinite(self.time_to_touchdown_s)


@dataclass(frozen=True)
class StudentStatus:
    """One student run's status for this plot.

    Attributes:
        arm: ``bc_mix`` or ``dp_mix``.
        seed: Training seed.
        status: ``present`` (verified rows) or ``missing``.
        reason: Why it is missing, or ``""``.
        run_dir: The run directory looked at, relative to the run root.
        selection_json_sha256: SHA-256 of ``selection.json`` (present only).
        rows_file: ``select/<selected.rows_file>`` (present only).
        rows_sha256: SHA-256 of that rows file (present only).
        ckpt_sha256: The selected checkpoint's SHA-256 (present only).
        selected_global_step: The selected checkpoint's step (present only).
        select_success: ``selected.success`` of 150 (present only).
        inference: ``selection.json``'s ``inference`` (``dp_mix``: Ta, K, ensembling).
    """

    arm: str
    seed: int
    status: str
    reason: str = ""
    run_dir: str = ""
    selection_json_sha256: str = ""
    rows_file: str = ""
    rows_sha256: str = ""
    ckpt_sha256: str = ""
    selected_global_step: int | None = None
    select_success: int | None = None
    inference: Mapping[str, Any] | None = field(default=None)


# --------------------------------------------------------------------------- the reference


def tmix_seed_from_config() -> int:
    """Return ``mix_1000``'s ``T_mix`` seed from ``configs/data/demos.yaml``.

    Raises:
        ValueError: If it differs from :data:`TMIX_SEED` (P3-D1 §2).
    """
    from ila.data.collect import load_collections

    seed = load_collections()["mix_1000"].mix_seed
    if seed != TMIX_SEED:
        raise ValueError(f"demos.yaml T_mix seed {seed} != P3-D1's {TMIX_SEED}")
    return seed


def tmix_assignment(rows: Sequence[p8_adapter.ListedEpisode], seed: int = TMIX_SEED) -> list[str]:
    """Assign ``ppo`` / ``gated`` to ``select`` rows, one ``assign_rows`` call per sea state.

    Args:
        rows: The ``select`` rows, in list order.
        seed: The ``T_mix`` seed.

    Returns:
        The policy of each row, in ``rows`` order.

    Raises:
        AssertionError: If the per-cell calls differ from one whole-list call (only possible
            with an odd cell) or a cell is not split in exact halves.
    """
    out = [""] * len(rows)
    for cell in dict.fromkeys(r.ss for r in rows):
        idx = [i for i, r in enumerate(rows) if r.ss == cell]
        for i, policy in zip(idx, mix.assign_rows(seed, [rows[i] for i in idx]), strict=True):
            out[i] = policy
        counts = Counter(out[i] for i in idx)
        assert abs(counts["ppo"] - counts["gated"]) <= 1, (cell, counts)
    assert out == mix.assign_rows(seed, rows), "per-cell and whole-list T_mix draws differ"
    return out


def _ppo_sha256() -> str:
    run_dir = p8_adapter.load_paths()["p8_ppo_seed0_run"]
    return p8_adapter.sha256_file(run_dir / "final" / "model.zip")


def fly_reference(
    *,
    workers: int = 8,
    chunk: int = 10,
    render: runner.RenderMode = "off",
    limit_per_cell: int | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Fly the ``T_mix`` reference on ``select`` (module docstring).

    Args:
        workers: Processes (rows do not depend on it).
        chunk: Episodes per task (rows do not depend on it).
        render: ``off`` (the committed reference) or ``full`` (the render-invariance check:
            P8's columns must come out byte-identical).
        limit_per_cell: Tests only: fly the first N episodes of each (cell, teacher).

    Returns:
        ``(rows, assigned, info)``: runner rows in ``select`` order, each row's assigned
        teacher, and run information (no wall clock in anything :func:`render` reads).
    """
    seed = tmix_seed_from_config()
    episodes = data_episodes.read_split(data_episodes.SELECT_NAME)
    assigned = tmix_assignment(episodes, seed)
    positions = {
        name: [i for i, p in enumerate(assigned) if p == name] for name in mix.MIX_POLICIES
    }
    if limit_per_cell is not None:
        for name, pos in positions.items():
            kept: list[int] = []
            per: Counter[str] = Counter()
            for i in pos:
                if per[episodes[i].ss] < limit_per_cell:
                    kept.append(i)
                    per[episodes[i].ss] += 1
            positions[name] = kept
    ppo_sha = _ppo_sha256()
    specs = {
        "ppo": p8_adapter.callable_spec(
            "ppo", functools.partial(p8_adapter.teacher, "ppo", 0), run_seed=0
        ),
        "gated": p8_adapter.controller_spec("gated"),
    }
    tags = {
        "ppo": runner.RowTags(arm="ppo", ckpt_sha256=ppo_sha),
        "gated": runner.RowTags(arm="gated"),
    }
    arms = [
        runner.VisionArm(
            spec=specs[name],
            episodes=[episodes[i] for i in positions[name]],
            tags=tags[name],
            render=render,
        )
        for name in mix.MIX_POLICIES
    ]
    t0 = time.perf_counter()
    flown = runner.run_arms_vision(
        arms, p8_adapter.load_eval_configs(), workers=workers, chunk=chunk
    )
    wall_s = time.perf_counter() - t0
    by_pos: dict[int, dict[str, Any]] = {}
    start = 0
    for name in mix.MIX_POLICIES:
        mine = flown[start : start + len(positions[name])]
        start += len(positions[name])
        for i, row in zip(positions[name], mine, strict=True):
            if row["method"] != name or int(row["episode_seed"]) != episodes[i].episode_seed:
                raise RuntimeError(f"{name}: rows out of order at select position {i}")
            by_pos[i] = row
    order = sorted(by_pos)
    split = data_episodes.SPLITS_DIR / f"{data_episodes.SELECT_NAME}.parquet"
    info = {
        "what": "T_mix reference flown on select for the H2 touchdown-time plot (plan Phase 7 "
        "item 3); plotted only, never scored",
        "list": {
            "name": data_episodes.SELECT_NAME,
            "file": split.name,
            "file_sha256": data_episodes.file_sha256(split),
            "episodes": len(episodes),
        },
        "assignment": {
            "function": "ila.data.mix.assign_rows, one call per sea-state cell",
            "seed": seed,
            "per_cell": {
                cell: {
                    p: sum(
                        1
                        for a, e in zip(assigned, episodes, strict=True)
                        if e.ss == cell and a == p
                    )
                    for p in mix.MIX_POLICIES
                }  # fmt: skip
                for cell in dict.fromkeys(e.ss for e in episodes)
            },
        },
        "teachers": {
            "ppo": {
                "source": "P8 ppo seed 0, final/ (SB3, VecNormalize training=False)",
                "model_zip_sha256": ppo_sha,
            },
            "gated": {"source": "P8 controller registry 'gated' (run_seed 0)"},
        },  # fmt: skip
        "runner": "ila.eval.runner.run_arms_vision (CPU)",
        "render": render,
        "workers": workers,
        "chunk": chunk,
        "limit_per_cell": limit_per_cell,
        "episodes_flown": len(order),
        "p8_commit": p8_adapter.p8_commit(),
        **_ila_provenance(),
        "wall_s": round(wall_s, 1),
    }
    return [by_pos[i] for i in order], [assigned[i] for i in order], info


def _ila_provenance() -> dict[str, Any]:
    """The checkout's commit, dirty flag and dirty paths (``ila.train.manifest.git_state``)."""
    state = mf.git_state()
    return {k: state[k] for k in ("ila_commit", "ila_dirty", "ila_dirty_paths")}


def write_reference(
    path: Path, rows: Sequence[Mapping[str, Any]], assigned: Sequence[str], info: Mapping[str, Any]
) -> None:
    """Write the reference rows (:data:`REFERENCE_COLUMNS`) and their run information.

    The runner columns are serialised by :func:`ila.eval.runner.vision_rows_csv` (P8's text for
    P8's columns); ``select_position`` and ``tmix_assigned`` are prepended.

    Args:
        path: The rows CSV (its directory gets :data:`REFERENCE_INFO_FILE`).
        rows: Runner rows in ``select`` order.
        assigned: Each row's assigned teacher.
        info: :func:`fly_reference`'s run information.
    """
    episodes = data_episodes.read_split(data_episodes.SELECT_NAME)
    pos_of = {(e.ss, e.index, e.episode_seed): i for i, e in enumerate(episodes)}
    body = list(csv.reader(io.StringIO(runner.vision_rows_csv(rows))))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(REFERENCE_COLUMNS)
    for row, cells, teacher in zip(rows, body[1:], assigned, strict=True):
        pos = pos_of[(str(row["ss"]), int(row["index"]), int(row["episode_seed"]))]
        writer.writerow([pos, teacher, *cells])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    (path.parent / REFERENCE_INFO_FILE).write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )


def _time(cell: str) -> float:
    """A ``time_to_touchdown_s`` cell -> float; empty or ``nan`` -> NaN."""
    return float(cell) if cell not in ("", "nan", "None") else float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_reference(path: Path, *, require_full: bool = True) -> list[EpisodeRecord]:
    """Read and check the committed reference rows.

    Checks: the rows are ``select`` episodes in list order (all 150 unless
    ``require_full=False``), each row's ``tmix_assigned`` equals :func:`tmix_assignment` and its
    ``method``, and every outcome is one of P8's classes.

    Args:
        path: :data:`REFERENCE_FILE`.
        require_full: Require all 150 episodes.

    Returns:
        One record per episode (``arm = "T_mix"``).

    Raises:
        H2SourceError: If any check fails.
    """
    episodes = data_episodes.read_split(data_episodes.SELECT_NAME)
    assigned = tmix_assignment(episodes, TMIX_SEED)
    rows = _read_csv(path)
    if require_full and len(rows) != len(episodes):
        raise H2SourceError(f"{path.name}: {len(rows)} rows, expected {len(episodes)}")
    out = []
    last = -1
    for r in rows:
        pos = int(r["select_position"])
        e = episodes[pos]
        if pos <= last:
            raise H2SourceError(f"{path.name}: not in select order at position {pos}")
        last = pos
        if (r["ss"], int(r["index"]), int(r["episode_seed"])) != (e.ss, e.index, e.episode_seed):
            raise H2SourceError(f"{path.name}: position {pos} is not that select episode")
        if not r["tmix_assigned"] == r["method"] == assigned[pos]:
            raise H2SourceError(f"{path.name}: position {pos} flew {r['method']!r}, "
                                f"assigned {assigned[pos]!r}")  # fmt: skip
        if r["outcome"] not in OUTCOME_ORDER:
            raise H2SourceError(f"{path.name}: unknown outcome {r['outcome']!r}")
        out.append(
            EpisodeRecord(
                arm=REFERENCE_GROUP,
                seed=None,
                tmix_assigned=r["tmix_assigned"],
                select_position=pos,
                ss=e.ss,
                index=e.index,
                episode_seed=e.episode_seed,
                outcome=r["outcome"],
                time_to_touchdown_s=_time(r["time_to_touchdown_s"]),
                ckpt_sha256=r["ckpt_sha256"],
            )
        )
    return out


# --------------------------------------------------------------------------- students


def verify_selection(run: Path, arm: str, seed: int) -> tuple[StudentStatus, list[EpisodeRecord]]:
    """Verify one run's selected ``select`` rows against its ``selection.json``; read them.

    Checks, each raising :class:`H2SourceError`:

    * ``manifest.json``: ``arm``, ``run_seed``, ``state == "done"``, every config file frozen
      and ``configs/train/ablations/<arm>.yaml`` at P3-D1's SHA-256;
    * ``selection.json``: ``arm``, ``run_seed``, ``teacher == "T_mix"``, ``tag == "ablation"``,
      ``obs_variant == "default"``, EMA weights on the CPU fp32 single-sample path, the list is
      ``select`` with the current file's SHA-256 and 150 episodes; ``dp_mix``: ``inference``
      is ``Ta`` 8, K 10, no ensembling;
    * the whole selection re-summarises from its stored rows without change
      (:func:`ila.train.select.resummarise_run`, ``write=False``);
    * the selected rows file (``selected.rows_file``) holds the ``select`` episodes in order,
      every row's ``ckpt_sha256`` is ``selected.candidate.sha256``, ``arm`` and
      ``student_seed`` are the run's, and its summary equals ``selected``'s.

    Args:
        run: ``<run_root>/<arm>/s<seed>``.
        arm: ``bc_mix`` or ``dp_mix``.
        seed: Training seed.

    Returns:
        ``(status, records)``.
    """
    sel_path = run / "selection.json"
    man = mf.read_json(run / "manifest.json")
    rec = mf.read_json(sel_path)

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise H2SourceError(f"{run}: {what}")

    need(man.get("arm") == arm and int(man.get("run_seed", -1)) == seed, "manifest arm/seed")
    need(man.get("state") == "done", f"manifest state {man.get('state')!r}")
    cfg = man.get("config", {})
    files = {f["path"]: f["sha256"] for f in cfg.get("files", [])}
    frozen = mf.FROZEN_CONFIG_SHA256[f"configs/train/ablations/{arm}.yaml"]
    need(cfg.get("all_frozen") is True, "manifest config not all frozen")
    need(files.get(f"configs/train/ablations/{arm}.yaml") == frozen, "ablation config SHA-256")
    need(rec.get("arm") == arm and int(rec.get("run_seed", -1)) == seed, "selection arm/seed")
    need(rec.get("teacher") == "T_mix", f"selection teacher {rec.get('teacher')!r}")
    need(rec.get("tag") == "ablation", f"selection tag {rec.get('tag')!r}")
    need(rec.get("obs_variant", "default") == "default", "obs_variant")
    need(rec.get("weights") == "ema", "weights")
    need(rec.get("inference_path") == "cpu_fp32_single_sample", "inference path")
    split = data_episodes.SPLITS_DIR / f"{data_episodes.SELECT_NAME}.parquet"
    lst = rec.get("list", {})
    need(lst.get("name") == data_episodes.SELECT_NAME, "list name")
    need(lst.get("file_sha256") == data_episodes.file_sha256(split), "select.parquet SHA-256")
    episodes = data_episodes.read_split(data_episodes.SELECT_NAME)
    need(int(lst.get("episodes", -1)) == len(episodes), "list episode count")
    if arm.startswith("dp"):
        need(rec.get("inference") == dict(DP_INFERENCE), f"inference {rec.get('inference')}")
    try:
        train_select.resummarise_run(run, write=False)
    except ValueError as exc:  # ResummaryMismatchError and missing-rows errors
        raise H2SourceError(f"{run}: selection does not re-summarise: {exc}") from exc
    selected = rec["selected"]
    rows_path = run / "select" / str(selected["rows_file"])
    need(rows_path.is_file(), f"rows file {rows_path.name} missing")
    rows = _read_csv(rows_path)
    keys = [(r["ss"], int(r["index"]), int(r["episode_seed"])) for r in rows]
    need(keys == [(e.ss, e.index, e.episode_seed) for e in episodes], "rows not select in order")
    ckpt = str(selected["candidate"]["sha256"])
    need(all(r["ckpt_sha256"] == ckpt for r in rows), "rows of another checkpoint")
    need(all(r["arm"] == arm and r["student_seed"] == str(seed) for r in rows), "rows arm/seed")
    summary = train_select.summarise_rows(rows)
    for k in ("n", "success", "outcomes", "n_touchdown", "mean_steps"):
        need(summary[k] == selected[k], f"rows {k} != selection.json")
    status = StudentStatus(
        arm=arm,
        seed=seed,
        status="present",
        run_dir=f"{arm}/s{seed}",
        selection_json_sha256=p8_adapter.sha256_file(sel_path),
        rows_file=f"select/{rows_path.name}",
        rows_sha256=p8_adapter.sha256_file(rows_path),
        ckpt_sha256=ckpt,
        selected_global_step=int(selected["candidate"]["global_step"]),
        select_success=int(selected["success"]),
        inference=rec.get("inference"),
    )
    records = [
        EpisodeRecord(
            arm=arm,
            seed=seed,
            tmix_assigned="",
            select_position=i,
            ss=e.ss,
            index=e.index,
            episode_seed=e.episode_seed,
            outcome=r["outcome"],
            time_to_touchdown_s=_time(r["time_to_touchdown_s"]),
            ckpt_sha256=ckpt,
        )
        for i, (e, r) in enumerate(zip(episodes, rows, strict=True))
    ]
    return status, records


def load_student(run_root: Path, arm: str, seed: int) -> tuple[StudentStatus, list[EpisodeRecord]]:
    """Return a student run's verified rows, or its ``missing`` status with the reason.

    A run is missing when its directory or its ``selection.json`` does not exist (the reason
    names the training state when a manifest exists). A ``selection.json`` that exists but
    does not verify raises (:func:`verify_selection`): that is an error to report, not an
    absence.

    Args:
        run_root: The runs root (``<run_root>/<arm>/s<seed>``).
        arm: ``bc_mix`` or ``dp_mix``.
        seed: Training seed.

    Returns:
        ``(status, records)``; ``records`` is empty when missing.
    """
    run = run_root / arm / f"s{seed}"
    rel = f"{arm}/s{seed}"
    if not run.is_dir():
        return StudentStatus(arm, seed, "missing", "run directory absent", rel), []
    if not (run / "selection.json").is_file():
        state = "no manifest"
        if (run / "manifest.json").is_file():
            state = f"manifest state {mf.read_json(run / 'manifest.json').get('state')!r}"
        return StudentStatus(arm, seed, "missing", f"no selection.json ({state})", rel), []
    return verify_selection(run, arm, seed)


# --------------------------------------------------------------------------- tables


def _fmt(value: float | None) -> str:
    """Serialise a float deterministically (``repr``); ``None``/NaN -> ``nan``."""
    if value is None or not math.isfinite(value):
        return "nan"
    return repr(float(value))


def write_episodes(path: Path, records: Sequence[EpisodeRecord]) -> None:
    """Write the per-episode table (:data:`EPISODE_COLUMNS`)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(EPISODE_COLUMNS)
    for r in records:
        writer.writerow(
            [
                r.arm,
                "" if r.seed is None else r.seed,
                r.tmix_assigned,
                r.select_position,
                r.ss,
                r.index,
                r.episode_seed,
                r.outcome,
                _fmt(r.time_to_touchdown_s),
                r.ckpt_sha256,
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def read_episodes(path: Path) -> list[EpisodeRecord]:
    """Read :data:`EPISODES_FILE` back."""
    return [
        EpisodeRecord(
            arm=r["arm"],
            seed=None if r["seed"] == "" else int(r["seed"]),
            tmix_assigned=r["tmix_assigned"],
            select_position=int(r["select_position"]),
            ss=r["ss"],
            index=int(r["index"]),
            episode_seed=int(r["episode_seed"]),
            outcome=r["outcome"],
            time_to_touchdown_s=_time(r["time_to_touchdown_s"]),
            ckpt_sha256=r["ckpt_sha256"],
        )
        for r in _read_csv(path)
    ]


def group_records(
    records: Sequence[EpisodeRecord], group: str, cell: str, seed: int | None = None
) -> list[EpisodeRecord]:
    """The records of one group in one cell (``all`` pools the sea states).

    ``T_mix`` is every reference episode; ``T_mix:ppo`` / ``T_mix:gated`` its assigned halves;
    a student arm pools its seeds unless ``seed`` is given.
    """
    if group == REFERENCE_GROUP:
        mine = [r for r in records if r.arm == REFERENCE_GROUP]
    elif group in MODE_GROUPS:
        teacher = group.split(":", 1)[1]
        mine = [r for r in records if r.arm == REFERENCE_GROUP and r.tmix_assigned == teacher]
    else:
        mine = [r for r in records if r.arm == group and (seed is None or r.seed == seed)]
    return [r for r in mine if cell == ALL_CELL or r.ss == cell]


def _touched(records: Sequence[EpisodeRecord]) -> list[float]:
    return [r.time_to_touchdown_s for r in records if r.touched_down]


def _quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(p8_adapter.stats.quantile_linear(arr, q)) if arr.size else float("nan")


def cell_stats(
    records: Sequence[EpisodeRecord], reference: Sequence[EpisodeRecord] | None
) -> dict[str, Any]:
    """Counts, outcome breakdown, quantiles and W1 for one group in one cell.

    Args:
        records: The group's records in the cell.
        reference: The ``T_mix`` reference's records in the same cell (``None``: no W1).

    Returns:
        ``n_episodes``, ``n_touchdown``, ``n_without_touchdown``, ``frac_without_touchdown``,
        ``outcomes`` (P8's order, zeros included), ``p10/p50/p90_model_s`` and
        ``w1_to_tmix_ref_model_s`` (NaN if either side has no touchdown or ``reference`` is
        ``None``). Times in seconds, model scale.
    """
    times = _touched(records)
    n = len(records)
    counts = Counter(r.outcome for r in records)
    unknown = set(counts) - set(OUTCOME_ORDER)
    if unknown:
        raise ValueError(f"unknown outcomes {sorted(unknown)}")
    w1 = float("nan") if reference is None else wasserstein_1(times, _touched(reference))
    return {
        "n_episodes": n,
        "n_touchdown": len(times),
        "n_without_touchdown": n - len(times),
        "frac_without_touchdown": (n - len(times)) / n if n else float("nan"),
        "outcomes": {o: counts.get(o, 0) for o in OUTCOME_ORDER},
        **{f"p{round(q * 100)}_model_s": _quantile(times, q) for q in QUANTILES},
        "w1_to_tmix_ref_model_s": w1,
    }


def _seeds_text(seeds: Sequence[int]) -> str:
    return " ".join(str(s) for s in seeds)


def summarise(
    records: Sequence[EpisodeRecord], statuses: Sequence[StudentStatus]
) -> list[dict[str, Any]]:
    """The summary rows (:data:`SUMMARY_COLUMNS` keys, ``outcomes`` flattened to ``n_<class>``).

    For each group and cell: one ``pooled`` row (students pooled over their present seeds),
    then, for students, one ``seed`` row per present seed. A student arm with no present seed
    gets one ``pooled`` row per cell with ``status`` ``missing`` and zero episodes; a partly
    present arm's status names the missing seeds.

    Args:
        records: Every record (reference and students).
        statuses: Every student run's status (present and missing).

    Returns:
        Rows in :data:`GROUPS` x cell order.
    """
    cells = (*CELLS, ALL_CELL)
    out: list[dict[str, Any]] = []
    for group in GROUPS:
        if group in STUDENT_ARMS:
            mine = [s for s in statuses if s.arm == group]
            present = sorted(s.seed for s in mine if s.status == "present")
            missing = sorted(s.seed for s in mine if s.status != "present")
            if not present:
                status = f"missing (seeds {_seeds_text(missing)})"
            elif missing:
                status = f"partial (seeds {_seeds_text(missing)} missing)"
            else:
                status = "present"
        else:
            present, status = [], "present"
        for cell in cells:
            ref = group_records(records, REFERENCE_GROUP, cell)
            levels: list[tuple[str, str, int | None]] = [("pooled", _seeds_text(present), None)]
            if group in STUDENT_ARMS:
                levels += [("seed", str(s), s) for s in present]
            for level, seeds, seed in levels:
                mine_r = group_records(records, group, cell, seed)
                stats = cell_stats(mine_r, None if group == REFERENCE_GROUP else ref)
                outcomes = stats.pop("outcomes")
                out.append(
                    {
                        "group": group,
                        "level": level,
                        "seeds": seeds,
                        "cell": cell,
                        "status": status,
                        **stats,
                        **{f"n_{o}": outcomes[o] for o in OUTCOME_ORDER},
                    }
                )
    return out


def _summary_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for r in rows:
        writer.writerow([_fmt(r[c]) if isinstance(r[c], float) else r[c] for c in SUMMARY_COLUMNS])
    return buf.getvalue()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _dump(record: Any) -> str:
    return json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- build / render


def build(
    *,
    reference_csv: Path,
    run_root: Path,
    out_dir: Path,
    arms: Sequence[str] = STUDENT_ARMS,
    seeds: Sequence[int] = STUDENT_SEEDS,
) -> dict[str, Any]:
    """Collect the reference and every student run, write the episode table and sources, render.

    Args:
        reference_csv: The committed :data:`REFERENCE_FILE`.
        run_root: The runs root (``<run_root>/<arm>/s<seed>``).
        out_dir: Output directory.
        arms: Student arms.
        seeds: Student seeds.

    Returns:
        The summary record (:func:`render`'s).
    """
    records = load_reference(reference_csv)
    statuses: list[StudentStatus] = []
    for arm in arms:
        for seed in seeds:
            status, recs = load_student(run_root, arm, seed)
            statuses.append(status)
            records.extend(recs)
    write_episodes(out_dir / EPISODES_FILE, records)
    sources = {
        "reference": {
            "file": reference_csv.name,
            "sha256": p8_adapter.sha256_file(reference_csv),
            "tmix_seed": TMIX_SEED,
        },
        "run_root": str(run_root),
        "students": [asdict(s) for s in statuses],
        "missing": [f"{s.arm} s{s.seed}: {s.reason}" for s in statuses if s.status != "present"],
        "select": {
            "file": f"{data_episodes.SELECT_NAME}.parquet",
            "file_sha256": data_episodes.file_sha256(
                data_episodes.SPLITS_DIR / f"{data_episodes.SELECT_NAME}.parquet"
            ),
        },
    }
    (out_dir / SOURCES_FILE).write_text(_dump(sources), encoding="utf-8")
    return render(out_dir)


def render(out_dir: Path) -> dict[str, Any]:
    """Write the summary CSV/JSON and the figure from the committed episode table and sources.

    Reads only :data:`EPISODES_FILE` and :data:`SOURCES_FILE` in ``out_dir``; the outputs are a
    pure function of those two files (and the plotting library's version).

    Args:
        out_dir: The directory holding both inputs; outputs are written beside them.

    Returns:
        The summary record written to :data:`SUMMARY_JSON`.
    """
    records = read_episodes(out_dir / EPISODES_FILE)
    sources = json.loads((out_dir / SOURCES_FILE).read_text(encoding="utf-8"))
    statuses = [
        StudentStatus(**{**s, "inference": s.get("inference")}) for s in sources["students"]
    ]
    rows = summarise(records, statuses)
    partial = any(s.status != "present" for s in statuses)
    ref = [r for r in records if r.arm == REFERENCE_GROUP]
    modes = {
        cell: wasserstein_1(
            _touched(group_records(ref, MODE_GROUPS[0], cell)),
            _touched(group_records(ref, MODE_GROUPS[1], cell)),
        )
        for cell in (*CELLS, ALL_CELL)
    }
    record = {
        "what": "H2 touchdown-time distributions on select (plan Phase 7 item 3): plotted "
        "only, never scored",
        "scored": False,
        "partial": partial,
        "missing": sources["missing"],
        "caption": list(CAPTION),
        "units": "time_to_touchdown_s: seconds, model scale (lambda = 1/25); x5 for full scale",
        "statistics": "quantiles: P8 quantile_linear (type 7) over touched-down episodes; W1: "
        "exact 1-D empirical Wasserstein-1 (ila.eval.metrics.wasserstein_1) between a group's "
        "pooled touched-down times and the T_mix reference's in the same cell; descriptive, "
        "no interval, no test",
        "reference": {
            **sources["reference"],
            "w1_ppo_vs_gated_model_s": modes,
        },
        "select": sources["select"],
        "students": sources["students"],
        "summary": rows,
    }
    (out_dir / SUMMARY_CSV).write_text(_summary_csv(rows), encoding="utf-8")
    (out_dir / SUMMARY_JSON).write_text(_dump(record), encoding="utf-8")
    plot(records, rows, statuses, out_dir / FIGURE_FILE, partial=partial)
    return record


# --------------------------------------------------------------------------- figure

#: Okabe-Ito colours (colour-blind safe): reference pooled, its two modes, the two students.
COLOURS: Final[Mapping[str, str]] = {
    REFERENCE_GROUP: "#000000",
    "T_mix:ppo": "#56B4E9",
    "T_mix:gated": "#E69F00",
    "bc_mix": "#009E73",
    "dp_mix": "#CC79A7",
}

#: Legend labels.
LABELS: Final[Mapping[str, str]] = {
    REFERENCE_GROUP: "T_mix reference (pooled)",
    "T_mix:ppo": "  ppo-assigned half",
    "T_mix:gated": "  gated-assigned half",
    "bc_mix": "bc_mix (seeds pooled)",
    "dp_mix": "dp_mix, Ta = 8, K = 10 (seeds pooled)",
}

#: Histogram bin width, s model scale (three control steps at 30 Hz).
BIN_S: Final[float] = 0.1


def plot(
    records: Sequence[EpisodeRecord],
    rows: Sequence[Mapping[str, Any]],
    statuses: Sequence[StudentStatus],
    path: Path,
    *,
    partial: bool,
) -> None:
    """Draw ECDF (top) and histogram (bottom) panels per cell and save a stamp-free PNG.

    Top: ECDF of touched-down times (fraction of each group's touched-down episodes, the
    distribution W1 compares). Bottom: per 0.1 s bin, the fraction of each group's episodes
    (so the missing mass is the fraction without a touchdown); the reference is drawn as
    stacked bars of its two halves, so both modes stay visible.

    Args:
        records: Every record.
        rows: :func:`summarise`'s rows.
        statuses: Student statuses (for the missing notes).
        path: The PNG.
        partial: Whether any student run is missing (marks the title).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = (*CELLS, ALL_CELL)
    finite = [r.time_to_touchdown_s for r in records if r.touched_down]
    x_max = max(math.ceil(max(finite) + BIN_S), 1.0) if finite else 1.0
    bins = np.arange(0.0, x_max + BIN_S / 2, BIN_S)
    by = {(r["group"], r["cell"]): r for r in rows if r["level"] == "pooled"}
    present_groups = [g for g in GROUPS if any(by[(g, c)]["n_episodes"] for c in cells)]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "svg.hashsalt": "h2"}):
        fig, axes = plt.subplots(
            2, len(cells), figsize=(14.0, 8.2), sharex=True, sharey="row", squeeze=False
        )
        for j, cell in enumerate(cells):
            top, bottom = axes[0, j], axes[1, j]
            for g in present_groups:
                times = np.sort(_touched(group_records(records, g, cell)))
                if not times.size:
                    continue
                y = np.arange(1, times.size + 1) / times.size
                style = {"ls": "--", "lw": 1.1} if g in MODE_GROUPS else {"ls": "-", "lw": 1.6}
                top.step(
                    np.concatenate([[0.0], times]),
                    np.concatenate([[0.0], y]),
                    where="post",
                    color=COLOURS[g],
                    label=LABELS[g],
                    **style,
                )
            notes = []
            for g in present_groups:
                r = by[(g, cell)]
                w1 = r["w1_to_tmix_ref_model_s"]
                w1_txt = (
                    ""
                    if g == REFERENCE_GROUP
                    else f", W1 {w1:.2f} s"
                    if math.isfinite(w1)
                    else ", W1 n/a"
                )
                notes.append(f"{g}: td {r['n_touchdown']}/{r['n_episodes']}{w1_txt}")
            for arm in STUDENT_ARMS:
                if arm not in present_groups:
                    notes.append(f"{arm}: missing")
            top.text(
                0.98, 0.04, "\n".join(notes), transform=top.transAxes, ha="right", va="bottom",
                fontsize=6.5, family="DejaVu Sans Mono",
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
            )  # fmt: skip
            top.set_title(cell if cell != ALL_CELL else "SS3-SS5 pooled (descriptive)", pad=26)
            top.grid(True, lw=0.3, alpha=0.5)
            sec = top.secondary_xaxis("top", functions=(lambda t: t * 5.0, lambda t: t / 5.0))
            sec.set_xlabel("full scale, s (x5)", fontsize=7)
            # Bottom: reference halves stacked, students as outlines.
            ref_all = group_records(records, REFERENCE_GROUP, cell)
            n_ref = len(ref_all)
            base = np.zeros(bins.size - 1)
            for g in MODE_GROUPS:
                if n_ref == 0:
                    break
                h, _ = np.histogram(_touched(group_records(records, g, cell)), bins=bins)
                frac = h / n_ref
                bottom.bar(
                    bins[:-1], frac, width=BIN_S, bottom=base, align="edge",
                    color=COLOURS[g], alpha=0.75, lw=0, label=f"T_mix: {g.split(':')[1]}-assigned",
                )  # fmt: skip
                base = base + frac
            for g in STUDENT_ARMS:
                if g not in present_groups:
                    continue
                mine = group_records(records, g, cell)
                if not mine:
                    continue
                h, _ = np.histogram(_touched(mine), bins=bins)
                bottom.stairs(h / len(mine), bins, color=COLOURS[g], lw=1.4, label=LABELS[g])
            bottom.grid(True, lw=0.3, alpha=0.5)
            bottom.set_xlabel("time to touchdown, s (model scale)")
        axes[0, 0].set_ylabel("ECDF over touched-down episodes")
        axes[1, 0].set_ylabel(f"fraction of all episodes per {BIN_S:g} s bin")
        axes[0, 0].set_xlim(0.0, x_max)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        h2, l2 = axes[1, 0].get_legend_handles_labels()
        fig.legend(
            handles + h2, labels + l2, loc="upper center", ncol=4, fontsize=7,
            bbox_to_anchor=(0.5, 0.965), frameon=False,
        )  # fmt: skip
        missing = [f"{s.arm} s{s.seed}" for s in statuses if s.status != "present"]
        title = (
            "H2 touchdown-time distributions on select, simulated (PyBullet): plotted only, "
            "not scored"
        )
        if partial:
            title = f"PARTIAL ({len(missing)} student runs missing): " + title
        fig.suptitle(title, fontsize=10, y=0.995)
        caption = "\n".join(textwrap.fill(c, 200) for c in CAPTION)
        if missing:
            caption += "\n" + textwrap.fill("Missing: " + ", ".join(missing), 200)
        fig.text(0.01, 0.005, caption, fontsize=6.3, va="bottom", ha="left")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.19, wspace=0.06, hspace=0.12)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, metadata={"Software": None})
        plt.close(fig)
