"""Closed-loop clock intervention on ``select``: a diagnostic, not a score (Phase 7).

Simulation only (PyBullet physics, TinyRenderer frames). Times are seconds **model scale**
(lambda = 1/25; full scale x5); ``time_fraction`` is P8's dimensionless
``t_episode / episode_len`` clipped into ``[0, 1]``, ``episode_len`` = 12 s model scale (60 s
full scale).

What it asks
------------
The ``+clock`` ablation (``bc_plus_clock``; P3-D1 §3, declared clock exception 1) gives the
student P8's ``time_fraction`` as a 14th proprio column. How much does its closed-loop
behaviour depend on that column? Each subject is re-flown on the 150 ``select`` episodes with
**only the clock value changed**, at the input of the network, and compared with the same
subject under the true clock:

* ``none``: the true clock, through the same wrapper. It must reproduce the reference rows
  cell for cell, and the module raises if it does not (students: the selection's own rows of
  the selected checkpoint; ``T_ppo``: the bare teacher flown through the same runner, plus the
  committed ``T_mix`` ``ppo`` rows on ``select`` where they overlap).
* ``shift_s``: ``time_fraction + 1.0 s / episode_len`` (+1/12), clipped into ``[0, 1]`` as P8
  clips it: the subject is told it is 1 s later than it is.
* ``const_mean``: ``time_fraction`` held at the student's training normaliser mean (the
  ``proprio_mean`` buffer of the selected EMA checkpoint, clock column); ``T_ppo`` gets the
  same value, so the two subjects see the same constant.
* ``clamp``: ``min(time_fraction, 1.6 s / episode_len)``: the clock stops at 1.6 s.

Subjects
--------
* every ``plus_clock`` run given (``<run_root>/<arm>/s<seed>``): the **selected** checkpoint of
  ``selection.json`` (SHA-256 re-checked; EMA weights), on the **CPU fp32 single-sample path**
  of selection (:func:`ila.train.select.checkpoint_policy`, one torch thread per worker,
  render ``full``, the checkpoint's ``plus_clock`` contract). A checkpoint of any other
  contract variant is refused: it has no clock column.
* ``T_ppo``: P8's ``ppo`` seed 0 (SB3, P8's frozen VecNormalize, ``training=False``), on P8's
  CPU path, render ``off`` (a teacher discards frames; Gate 2 shows rendering does not change
  physics). Its intervention changes P8's 25-d observation's ``time_fraction`` field only.

How the clock is changed
------------------------
In a wrapper around the subject's **input**, never in the env or the physics. The runner
builds the student input exactly as in selection (``ObsContract.student_input``: images and
the 14-d proprio at steps ``t-1, t``); :class:`ClockInterventionStudent` then replaces the
``time_fraction`` column of both rows through :meth:`ila.obs.split.ObsContract.
with_proprio_clock` (selected by field name, all other columns and the images bit-identical)
and calls the student's ``act_vision``. :class:`ClockInterventionTeacher` does the same to the
teacher's observation through :meth:`~ila.obs.split.ObsContract.with_obs_clock`. Nothing else
reaches either network: no field is added, and the env's own observation is not modified.

Outputs (:func:`run_study`)
---------------------------
``clock_intervention.csv`` (one summary row per subject and mode), ``clock_intervention.json``
(the same plus provenance, definitions and the reproduction checks) and
``clock_intervention_episodes.csv.gz`` (per-episode rows: P8 outcome class,
``time_to_touchdown_s``, ``closing_speed_normal_m_s``; gzip with a zero mtime, so the bytes
depend on the rows only). Summary: success, P8's six outcome classes, touchdown time p50 and
sample SD over touched-down episodes and their shift against ``none`` (p50 difference, and the
paired mean over episodes touched down in both), pooled p95 closing speed (P8's
``quantile_linear``, touched-down episodes).

Scope: a diagnostic on ``select``. It **chooses nothing** (no checkpoint, no hyper-parameter, no
arm) and **is not scored**; ``select`` rows of a selected checkpoint are in-sample. It reads no
frozen list (``tests/test_render_access.py``), writes no demo store and trains nothing.
"""

from __future__ import annotations

import csv
import functools
import gzip
import io
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import runner
from ila.obs.split import CLOCK_FIELD, ObsContract, load_contract
from ila.policies.registry import read_checkpoint
from ila.train import manifest as mf
from ila.train.config import sha256_file
from ila.train.select import SELECT_SPLIT, checkpoint_policy, summarise_rows

__all__ = [
    "CLAMP_S",
    "CLOCK_VARIANT",
    "EPISODE_COLUMNS",
    "MODES",
    "SHIFT_S",
    "SUMMARY_COLUMNS",
    "TEACHER",
    "ClockIntervention",
    "ClockInterventionStudent",
    "ClockInterventionTeacher",
    "ReproductionError",
    "SelectedRun",
    "build_student",
    "build_teacher",
    "fly_student",
    "fly_teacher",
    "make_intervention",
    "rows_mismatch",
    "run_study",
    "selected_run",
    "summarise_mode",
    "training_clock_mean",
]

#: Interventions, in report order.
MODES: Final[tuple[str, ...]] = ("none", "shift_s", "const_mean", "clamp")

#: ``shift_s``: seconds model scale added to the episode clock (5 s full scale).
SHIFT_S: Final[float] = 1.0

#: ``clamp``: seconds model scale at which the clock stops (8 s full scale).
CLAMP_S: Final[float] = 1.6

#: The only contract variant with a clock column in the student proprio.
CLOCK_VARIANT: Final[str] = "plus_clock"

#: The teacher subject's label.
TEACHER: Final[str] = "T_ppo"

#: The scope label written into every output.
SCOPE: Final[str] = (
    "diagnostic on select; chooses nothing; not scored; simulation only; model scale "
    "lambda = 1/25 (times x5 full scale)"
)

#: Per-episode output columns (subject and intervention, then P8's row cells).
EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "subject",
    "run_seed",
    "mode",
    "clock_value",
    "select_position",
    "regime",
    "ss",
    "index",
    "episode_seed",
    "outcome",
    "termination_reason",
    "steps",
    "td_t_episode_s",
    "time_to_touchdown_s",
    "closing_speed_normal_m_s",
    "lateral_offset_m",
    "tunnelled",
    "detectors_disagree",
    "ckpt_sha256",
)

#: P8's outcome classes in P8's order (``ila.train.select.OUTCOME_ORDER``).
_OUTCOMES: Final[tuple[str, ...]] = (
    "crash",
    "off_pad",
    "hard_landing",
    "bounce",
    "success",
    "timeout",
)

#: Summary columns (``clock_intervention.csv``).
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "subject",
    "run_seed",
    "checkpoint_sha256",
    "mode",
    "clock_value",
    "n",
    "success",
    "success_minus_none",
    *_OUTCOMES,
    "n_outcome_changed_vs_none",
    "n_touchdown",
    "td_p50_s",
    "td_sd_s",
    "td_p50_shift_vs_none_s",
    "td_paired_mean_shift_vs_none_s",
    "n_td_paired",
    "p95_closing_speed_m_s",
    "mean_steps",
    "n_tunnelled",
    "n_detectors_disagree",
)

#: The committed ``T_ppo`` rows on ``select`` (the ``T_mix`` reference's ``ppo`` half).
_TMIX_ROWS: Final[Path] = (
    p8_adapter.REPO_ROOT / "results" / "phase7" / "h2" / "tmix_select_rows.csv"
)


class ReproductionError(RuntimeError):
    """``none`` did not reproduce its reference rows: the wrapper is not transparent."""


# --------------------------------------------------------------------------- interventions


@dataclass(frozen=True)
class ClockIntervention:
    """One clock intervention (picklable; applied to ``time_fraction`` values only).

    Attributes:
        mode: One of :data:`MODES`.
        value: ``shift_s``: the added ``time_fraction`` (``SHIFT_S / episode_len``);
            ``const_mean``: the held ``time_fraction``; ``clamp``: the ceiling
            (``CLAMP_S / episode_len``); ``none``: 0. Dimensionless.
    """

    mode: str
    value: float = 0.0

    def __post_init__(self) -> None:
        """Validate the mode and value.

        Raises:
            ValueError: On an unknown mode or a value outside ``[0, 1]``.
        """
        if self.mode not in MODES:
            raise ValueError(f"unknown clock intervention {self.mode!r}; expected {MODES}")
        if not (math.isfinite(self.value) and 0.0 <= self.value <= 1.0):
            raise ValueError(f"intervention value must be in [0, 1], got {self.value}")
        if self.mode == "none" and self.value != 0.0:
            raise ValueError("'none' takes no value")

    def apply(self, clock: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the intervened ``time_fraction`` values.

        Args:
            clock: ``float32[..., 1]``, true ``time_fraction`` (dimensionless, ``[0, 1]``).

        Returns:
            Float32 of the same shape. ``none`` returns the values unchanged (bit for bit);
            ``clamp`` returns them unchanged below the ceiling.
        """
        c = np.asarray(clock, dtype=np.float32)
        if self.mode == "none":
            return c.copy()
        if self.mode == "shift_s":  # P8 clips the clock into [0, 1] (rld landing_env)
            return np.clip(c.astype(np.float64) + self.value, 0.0, 1.0).astype(np.float32)
        if self.mode == "const_mean":
            return np.full_like(c, np.float32(self.value))
        return np.minimum(c.astype(np.float64), self.value).astype(np.float32)  # clamp

    def describe(self, episode_len_s: float) -> str:
        """Human-readable definition (seconds model scale)."""
        if self.mode == "none":
            return "true clock"
        if self.mode == "shift_s":
            return f"clip(time_fraction + {self.value * episode_len_s:g} s / {episode_len_s:g} s)"
        if self.mode == "const_mean":
            return f"time_fraction = {self.value:.8g} (= {self.value * episode_len_s:.4g} s)"
        return f"min(time_fraction, {self.value * episode_len_s:g} s / {episode_len_s:g} s)"


def make_intervention(mode: str, *, episode_len_s: float, const_mean: float) -> ClockIntervention:
    """Build a mode's intervention from P8's episode length and the training clock mean.

    Args:
        mode: One of :data:`MODES`.
        episode_len_s: P8's ``episode_len_s`` (12 s model scale).
        const_mean: The ``time_fraction`` held by ``const_mean`` (dimensionless).

    Returns:
        The intervention.
    """
    if episode_len_s <= 0.0:
        raise ValueError(f"episode_len_s must be positive, got {episode_len_s}")
    value = {
        "none": 0.0,
        "shift_s": SHIFT_S / episode_len_s,
        "const_mean": float(const_mean),
        "clamp": CLAMP_S / episode_len_s,
    }
    if mode not in value:
        raise ValueError(f"unknown clock intervention {mode!r}; expected {MODES}")
    return ClockIntervention(mode, value[mode])


# --------------------------------------------------------------------------- wrappers


class ClockInterventionStudent:
    """A ``plus_clock`` student whose input clock column is replaced; nothing else changes.

    P8's ``EvalPolicy`` shape plus ``act_vision`` (the student contract). Never privileged.

    Attributes:
        inner: The student (``act_vision``; not a ``modular`` student).
        contract: The ``plus_clock`` observation contract.
        intervention: What happens to the clock column.
    """

    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(self, inner: Any, contract: ObsContract, intervention: ClockIntervention) -> None:
        """Hold the student.

        Raises:
            ValueError: If the contract is not ``plus_clock``.
            TypeError: If ``inner`` has no ``act_vision`` or is a ``modular`` student.
        """
        if contract.variant != CLOCK_VARIANT:
            raise ValueError(
                f"the clock intervention needs a {CLOCK_VARIANT!r} student, "
                f"got {contract.variant!r}"
            )
        if not runner.is_student(inner) or runner.is_modular(inner):
            raise TypeError("the clock intervention wraps an act_vision student (not modular)")
        self.inner = inner
        self.contract = contract
        self.intervention = intervention

    def reset(
        self, seed: int, context: object | None = None, motion_feed: object | None = None
    ) -> None:
        """Reset the student (a student never receives a context or a motion feed).

        Raises:
            ValueError: If a motion feed is passed (the context is refused by the student).
        """
        if motion_feed is not None:
            raise ValueError("no subject of the clock diagnostic consumes a motion feed")
        self.inner.reset(seed, context)

    def act(self, obs: object) -> NDArray[np.float32]:
        """Act on a ``StudentInput``; anything else (a raw 25-d observation) raises.

        Raises:
            TypeError: If ``obs`` is not a ``StudentInput``.
        """
        from ila.obs.split import StudentInput

        if not isinstance(obs, StudentInput):
            raise TypeError(f"a vision student takes a StudentInput, got {type(obs).__name__}")
        return self.act_vision(obs.images, obs.proprio)

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Replace the clock column of ``proprio`` (by name) and call the student.

        Args:
            image_stack: ``uint8[To, 96, 96, 3]``, passed through untouched.
            proprio: ``float32[To, 14]``, the ``plus_clock`` proprio (P8 units).

        Returns:
            The student's ``float32[3]`` normalised world-frame velocity setpoint.
        """
        new = self.contract.with_proprio_clock(proprio, self.intervention.apply)
        return np.asarray(self.inner.act_vision(image_stack, new), dtype=np.float32)


class ClockInterventionTeacher:
    """A P8 teacher whose observation's ``time_fraction`` field is replaced; nothing else.

    Attributes:
        inner: A P8 ``EvalPolicy`` (``reset``, ``act`` on the 25-d observation); never a
            student or a ``modular`` policy.
        contract: Any contract (only its P8 layout is used, by field name).
        intervention: What happens to ``time_fraction``.
    """

    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(self, inner: Any, contract: ObsContract, intervention: ClockIntervention) -> None:
        """Hold the teacher.

        Raises:
            TypeError: If ``inner`` is a vision student (``act_vision``) or a ``modular``
                policy (``act_modular``): this wrapper feeds a 25-d P8 observation, which only a
                P8 teacher may receive (non-negotiable 2).
        """
        if runner.is_student(inner) or runner.is_modular(inner):
            raise TypeError(
                "the teacher clock intervention wraps a P8 teacher, not a student or modular "
                f"policy (got {type(inner).__name__})"
            )
        self.inner = inner
        self.contract = contract
        self.intervention = intervention

    def reset(
        self, seed: int, context: object | None = None, motion_feed: object | None = None
    ) -> None:
        """Reset the teacher (per episode, as P8's runner does).

        Raises:
            ValueError: If a motion feed is passed.
        """
        if motion_feed is not None:
            raise ValueError("no subject of the clock diagnostic consumes a motion feed")
        self.inner.reset(seed, context)

    def act(self, obs: ArrayLike) -> NDArray[np.float64]:
        """Replace ``time_fraction`` in a copy of ``obs`` and return the teacher's action.

        Args:
            obs: P8's ``float32[25]`` observation (not modified).

        Returns:
            The teacher's ``(3,)`` normalised world-frame velocity setpoint.
        """
        new = self.contract.with_obs_clock(obs, self.intervention.apply)
        return np.asarray(self.inner.act(new), dtype=np.float64)


def build_student(path: str, intervention: ClockIntervention, cfgs: p8_adapter.EvalConfigs) -> Any:
    """Picklable ``PolicySpec.build``: selection's CPU fp32 EMA student, clock-wrapped.

    Returns:
        A :class:`ClockInterventionStudent`, typed ``Any`` as P8's ``PolicySpec.build``
        result (as :func:`ila.train.select.checkpoint_policy`).

    Raises:
        ValueError: If the checkpoint is not a ``plus_clock`` student.
    """
    variant = str(read_checkpoint(Path(path))["obs_variant"])
    if variant != CLOCK_VARIANT:
        raise ValueError(f"{path}: contract {variant!r}, not {CLOCK_VARIANT!r}; no clock column")
    inner = checkpoint_policy(path, cfgs)
    return ClockInterventionStudent(inner, load_contract(variant=CLOCK_VARIANT), intervention)


def build_teacher(intervention: ClockIntervention, cfgs: p8_adapter.EvalConfigs) -> Any:
    """Picklable ``PolicySpec.build``: P8's ``ppo`` seed 0, its observation clock wrapped.

    Returns:
        A :class:`ClockInterventionTeacher` (typed ``Any``, as above).
    """
    return ClockInterventionTeacher(
        p8_adapter.teacher("ppo", 0, cfgs), load_contract(), intervention
    )


# --------------------------------------------------------------------------- runs


@dataclass(frozen=True)
class SelectedRun:
    """A ``plus_clock`` run's selected checkpoint (``selection.json``).

    Attributes:
        run_dir: The run directory.
        arm: The arm (``bc_plus_clock``).
        run_seed: Its seed.
        teacher: The rows' ``teacher`` column (the training teacher).
        path: The selected checkpoint.
        sha256: Its SHA-256 (re-checked against the file).
        global_step: Its step.
        rows_csv: The selection's ``select`` rows of this checkpoint.
    """

    run_dir: Path
    arm: str
    run_seed: int
    teacher: str
    path: Path
    sha256: str
    global_step: int
    rows_csv: Path


def selected_run(run_dir: Path) -> SelectedRun:
    """Read and verify a run's selection.

    Raises:
        ValueError: If the run was not selected on the current ``select`` list, is not a
            ``plus_clock`` run, its checkpoint's bytes changed, or it is a diffusion run
            selected at non-default inference settings.
    """
    sel = mf.read_json(run_dir / "selection.json")
    if sel.get("obs_variant") != CLOCK_VARIANT:
        raise ValueError(
            f"{run_dir}: obs_variant {sel.get('obs_variant')!r}, not {CLOCK_VARIANT!r}"
        )
    lst = sel["list"]
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    if lst["name"] != SELECT_SPLIT or lst["file_sha256"] != data_episodes.file_sha256(split):
        raise ValueError(f"{run_dir}: selected on another list than the current {SELECT_SPLIT}")
    if sel.get("weights") != "ema" or sel.get("inference_path") != "cpu_fp32_single_sample":
        raise ValueError(f"{run_dir}: not selected on the CPU fp32 path with EMA weights")
    if sel.get("inference") is not None:
        raise ValueError(f"{run_dir}: selected at inference settings {sel['inference']}")
    chosen = sel["selected"]
    cand = chosen["candidate"]
    path = Path(cand["file"])
    if not path.is_absolute() or not path.is_file():
        path = run_dir / "checkpoints" / path.name
    if sha256_file(path) != cand["sha256"]:
        raise ValueError(f"{path}: SHA-256 differs from selection.json")
    ckpt = read_checkpoint(path)
    if str(ckpt["obs_variant"]) != CLOCK_VARIANT:
        raise ValueError(f"{path}: contract {ckpt['obs_variant']!r}, not {CLOCK_VARIANT!r}")
    return SelectedRun(
        run_dir=run_dir,
        arm=str(sel["arm"]),
        run_seed=int(sel["run_seed"]),
        teacher=str(sel["teacher"]),
        path=path,
        sha256=str(cand["sha256"]),
        global_step=int(cand["global_step"]),
        rows_csv=run_dir / "select" / str(chosen["rows_file"]),
    )


def training_clock_mean(path: Path) -> float:
    """The selected EMA checkpoint's proprio normaliser mean, ``time_fraction`` column.

    The mean of the training frames' clock (dimensionless), selected by field name.
    """
    ckpt = read_checkpoint(path)
    mean = ckpt["ema"]["encoder.proprio_mean"].detach().cpu().numpy()
    clock = load_contract(variant=CLOCK_VARIANT).proprio_clock(mean)
    return float(clock.reshape(-1)[0])


def fly_student(
    run: SelectedRun,
    intervention: ClockIntervention,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    workers: int,
    chunk: int = 5,
) -> list[dict[str, Any]]:
    """Fly a selected ``plus_clock`` checkpoint with a clock intervention (selection's path).

    The spec's method and the row tags are selection's (arm, training teacher, seed, SHA), so
    ``none`` rows are comparable with the selection's rows cell for cell.

    Returns:
        Runner rows, in episode order.
    """
    build = functools.partial(build_student, str(run.path), intervention)
    spec = p8_adapter.callable_spec(run.arm, build, run_seed=run.run_seed)
    tags = runner.RowTags(
        arm=run.arm, teacher=run.teacher, student_seed=run.run_seed, ckpt_sha256=run.sha256
    )
    return runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="full",
        workers=workers,
        chunk=chunk,
        contract_variant=CLOCK_VARIANT,
    )


def teacher_sha256() -> str:
    """SHA-256 of P8's ``ppo`` seed-0 ``final/model.zip``."""
    run_dir = p8_adapter.load_paths()["p8_ppo_seed0_run"]
    return p8_adapter.sha256_file(run_dir / "final" / "model.zip")


def fly_teacher(
    intervention: ClockIntervention | None,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    workers: int,
    chunk: int = 10,
) -> list[dict[str, Any]]:
    """Fly P8's ``ppo`` seed 0 on P8's CPU path (render ``off``), clock-wrapped or bare.

    Args:
        intervention: The intervention, or ``None`` for the bare teacher (the reference).
        episodes: Listed episodes.
        workers: Processes (rows do not depend on it).
        chunk: Episodes per task.

    Returns:
        Runner rows (method ``ppo``, as in the committed ``T_mix`` reference).
    """
    build: Callable[[p8_adapter.EvalConfigs], Any] = (
        functools.partial(p8_adapter.teacher, "ppo", 0)
        if intervention is None
        else functools.partial(build_teacher, intervention)
    )
    spec = p8_adapter.callable_spec("ppo", build, run_seed=0)
    tags = runner.RowTags(arm=TEACHER, ckpt_sha256=teacher_sha256())
    return runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="off",
        workers=workers,
        chunk=chunk,
    )


# --------------------------------------------------------------------------- checks


def _same(a: Any, b: Any) -> bool:
    """Cell equality across live rows and CSV text: floats by value (NaN == NaN)."""
    sa = "" if a is None else str(a)
    sb = "" if b is None else str(b)
    if sa == sb:
        return True
    try:
        fa, fb = float(sa), float(sb)
    except ValueError:
        return False
    return fa == fb or (math.isnan(fa) and math.isnan(fb))


def rows_mismatch(
    rows: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> list[str]:
    """Cells where ``rows`` differ from ``reference`` (same length and order required).

    Returns:
        ``"<i>:<column>: <row> != <reference>"`` per differing cell (empty when identical).
    """
    if len(rows) != len(reference):
        return [f"{len(rows)} rows vs {len(reference)} reference rows"]
    return [
        f"{i}:{c}: {r.get(c)!r} != {ref.get(c)!r}"
        for i, (r, ref) in enumerate(zip(rows, reference, strict=True))
        for c in columns
        if not _same(r.get(c), ref.get(c))
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tmix_ppo_check(rows: Sequence[Mapping[str, Any]], limit: int | None) -> dict[str, Any]:
    """Compare teacher ``none`` rows with the committed ``T_mix`` ``ppo`` rows (P8 columns)."""
    if not _TMIX_ROWS.is_file():
        return {"file": None, "compared": 0}
    ref = [r for r in _read_csv(_TMIX_ROWS) if r["tmix_assigned"] == "ppo"]
    ref = [r for r in ref if limit is None or int(r["select_position"]) < limit]
    mine = [rows[int(r["select_position"])] for r in ref]
    bad = rows_mismatch(mine, ref, p8_adapter.EPISODE_COLUMNS)
    if bad:
        raise ReproductionError(f"T_ppo 'none' differs from {_TMIX_ROWS.name}: {bad[:5]}")
    return {
        "file": _TMIX_ROWS.relative_to(p8_adapter.REPO_ROOT).as_posix(),
        "file_sha256": sha256_file(_TMIX_ROWS),
        "compared": len(ref),
        "columns": "P8 EPISODE_COLUMNS",
        "identical": True,
    }


# --------------------------------------------------------------------------- summary


def _touchdown(rows: Sequence[Mapping[str, Any]]) -> NDArray[np.float64]:
    """``time_to_touchdown_s`` per row (NaN without a touchdown), seconds model scale."""
    return np.asarray([float(r["time_to_touchdown_s"]) for r in rows], dtype=np.float64)


def summarise_mode(
    rows: Sequence[Mapping[str, Any]], none_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One subject's summary under one mode, with its shift against ``none``.

    Args:
        rows: The mode's rows, in ``select`` order.
        none_rows: The same subject's ``none`` rows, same order.

    Returns:
        :data:`SUMMARY_COLUMNS` cells from ``n`` on. Touchdown times are seconds model scale
        over touched-down episodes (finite ``time_to_touchdown_s``); ``td_sd_s`` is the sample
        SD (ddof 1); ``td_paired_mean_shift_vs_none_s`` averages ``mode - none`` over episodes
        touched down in both.
    """
    if [r["episode_seed"] for r in rows] != [r["episode_seed"] for r in none_rows]:
        raise ValueError("mode and none rows are not the same episodes in the same order")
    s = summarise_rows(rows)
    s0 = summarise_rows(none_rows)
    td, td0 = _touchdown(rows), _touchdown(none_rows)
    q = p8_adapter.stats.quantile_linear
    fin, fin0 = td[np.isfinite(td)], td0[np.isfinite(td0)]
    p50 = float(q(fin, 0.5)) if fin.size else math.nan
    p50_0 = float(q(fin0, 0.5)) if fin0.size else math.nan
    both = np.isfinite(td) & np.isfinite(td0)
    return {
        "n": s["n"],
        "success": s["success"],
        "success_minus_none": s["success"] - s0["success"],
        **{o: s["outcomes"][o] for o in _OUTCOMES},
        "n_outcome_changed_vs_none": sum(
            str(a["outcome"]) != str(b["outcome"]) for a, b in zip(rows, none_rows, strict=True)
        ),
        "n_touchdown": s["n_touchdown"],
        "td_p50_s": p50,
        "td_sd_s": float(np.std(fin, ddof=1)) if fin.size > 1 else math.nan,
        "td_p50_shift_vs_none_s": p50 - p50_0,
        "td_paired_mean_shift_vs_none_s": float(np.mean(td[both] - td0[both]))
        if both.any()
        else math.nan,
        "n_td_paired": int(both.sum()),
        "p95_closing_speed_m_s": s["p95_closing_speed_m_s"],
        "mean_steps": s["mean_steps"],
        "n_tunnelled": s["n_tunnelled"],
        "n_detectors_disagree": s["n_detectors_disagree"],
    }


def _episode_rows(
    subject: str, seed: int, mode: str, clock: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for pos, r in enumerate(rows):
        cells = {k: r[k] for k in EPISODE_COLUMNS[5:]}
        out.append(
            {
                "subject": subject,
                "run_seed": seed,
                "mode": mode,
                "clock_value": clock,
                "select_position": pos,
                **cells,
            }
        )
    return out


def _cell(value: Any) -> Any:
    """CSV cell: floats to 6 significant digits (repr for exactness is in the gz rows)."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return "inf" if math.isinf(value) else f"{value:.6g}"
    return value


def _json(value: Any) -> Any:
    """Strict JSON value: NaN and inf as ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- study


def run_study(
    run_dirs: Sequence[Path],
    *,
    out_dir: Path,
    teacher: bool = True,
    modes: Sequence[str] = MODES,
    workers: int = 4,
    limit: int | None = None,
    teacher_const_mean: float | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly every subject under every mode on ``select`` and write the outputs.

    Args:
        run_dirs: ``plus_clock`` run directories (``<run_root>/<arm>/s<seed>``).
        out_dir: ``results/phase7/diagnostics``.
        teacher: Also fly ``T_ppo``.
        modes: Interventions; must include ``none`` (the reproduction check).
        workers: Processes (rows do not depend on it).
        limit: Tests only: the first ``limit`` ``select`` episodes (recorded).
        teacher_const_mean: ``T_ppo``'s ``const_mean`` value; by default the students'
            (which must agree). Required when ``teacher`` is set without students.
        log: Progress sink.

    Returns:
        The JSON record (also written).

    Raises:
        ReproductionError: If any ``none`` differs from its reference rows.
        ValueError: On a non-``plus_clock`` run, a changed checkpoint, or students whose
            training clock means differ while ``teacher_const_mean`` is unset.
    """
    if "none" not in modes or any(m not in MODES for m in modes):
        raise ValueError(f"modes must include 'none' and be among {MODES}, got {modes}")
    modes = [m for m in MODES if m in modes]  # report order; none first
    t0 = time.perf_counter()
    git = mf.git_state()
    cfgs = p8_adapter.load_eval_configs()
    episode_len_s = float(cfgs.landing.episode_len_s)
    select = data_episodes.read_split(SELECT_SPLIT)
    if limit is not None:
        select = select[:limit]
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    runs = [selected_run(Path(d)) for d in run_dirs]
    means = {r.run_seed: training_clock_mean(r.path) for r in runs}
    if teacher_const_mean is None and teacher:
        if not means:
            raise ValueError("T_ppo's const_mean needs teacher_const_mean or a student run")
        if len(set(means.values())) != 1:
            raise ValueError(f"students' training clock means differ: {means}")
        teacher_const_mean = next(iter(means.values()))

    summary: list[dict[str, Any]] = []
    episodes_out: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []

    def one_subject(
        label: str,
        seed: int,
        sha: str,
        fly: Callable[[ClockIntervention], list[dict[str, Any]]],
        const_mean: float,
        reference: Callable[[list[dict[str, Any]]], dict[str, Any]],
        info: dict[str, Any],
    ) -> None:
        by_mode: dict[str, list[dict[str, Any]]] = {}
        defs: dict[str, Any] = {}
        walls: dict[str, float] = {}
        for mode in modes:
            iv = make_intervention(mode, episode_len_s=episode_len_s, const_mean=const_mean)
            t1 = time.perf_counter()
            rows = fly(iv)
            walls[mode] = time.perf_counter() - t1
            by_mode[mode] = rows
            defs[mode] = {"value_time_fraction": iv.value, "definition": iv.describe(episode_len_s)}
            if mode == "none":
                info["none_reproduces"] = reference(rows)
            s = summarise_rows(rows)
            log(
                f"{label} s{seed} {mode}: success {s['success']}/{s['n']} "
                f"{ {k: v for k, v in s['outcomes'].items() if v} } ({walls[mode]:.0f} s)"
            )
        for mode in modes:
            desc = defs[mode]["definition"]
            summary.append(
                {
                    "subject": label,
                    "run_seed": seed,
                    "checkpoint_sha256": sha,
                    "mode": mode,
                    "clock_value": desc,
                    **summarise_mode(by_mode[mode], by_mode["none"]),
                }
            )
            episodes_out.extend(_episode_rows(label, seed, mode, desc, by_mode[mode]))
        subjects.append({**info, "interventions": defs, "wall_s": walls})

    for run in runs:

        def fly_s(iv: ClockIntervention, run: SelectedRun = run) -> list[dict[str, Any]]:
            return fly_student(run, iv, select, workers=workers)

        def ref_s(rows: list[dict[str, Any]], run: SelectedRun = run) -> dict[str, Any]:
            stored = _read_csv(run.rows_csv)
            stored = stored if limit is None else stored[:limit]
            bad = rows_mismatch(rows, stored, runner.VISION_EPISODE_COLUMNS)
            if bad:
                raise ReproductionError(f"{run.run_dir}: 'none' differs from selection: {bad[:5]}")
            return {
                "reference": f"selection rows {run.rows_csv.name}",
                "reference_sha256": sha256_file(run.rows_csv),
                "episodes": len(rows),
                "columns": "all runner columns (VISION_EPISODE_COLUMNS)",
                "identical": True,
            }

        one_subject(
            run.arm,
            run.run_seed,
            run.sha256,
            fly_s,
            means[run.run_seed],
            ref_s,
            {
                "subject": run.arm,
                "run_seed": run.run_seed,
                "run_dir": str(run.run_dir),
                "checkpoint": run.path.name,
                "checkpoint_sha256": run.sha256,
                "checkpoint_sha256_rechecked": True,
                "global_step": run.global_step,
                "weights": "ema",
                "contract_variant": CLOCK_VARIANT,
                "path": "cpu_fp32_single_sample (selection's checkpoint_policy), render full",
                "training_clock_mean": means[run.run_seed],
            },
        )

    if teacher:
        assert teacher_const_mean is not None
        t_sha = teacher_sha256()

        def fly_t(iv: ClockIntervention) -> list[dict[str, Any]]:
            return fly_teacher(iv, select, workers=workers)

        def ref_t(rows: list[dict[str, Any]]) -> dict[str, Any]:
            bare = fly_teacher(None, select, workers=workers)
            bad = rows_mismatch(rows, bare, runner.VISION_EPISODE_COLUMNS)
            if bad:
                raise ReproductionError(f"T_ppo 'none' differs from the bare teacher: {bad[:5]}")
            return {
                "reference": "bare T_ppo flown through the same runner (render off)",
                "episodes": len(rows),
                "columns": "all runner columns (VISION_EPISODE_COLUMNS)",
                "identical": True,
                "committed_tmix_ppo_rows": _tmix_ppo_check(rows, limit),
            }

        one_subject(
            TEACHER,
            0,
            t_sha,
            fly_t,
            teacher_const_mean,
            ref_t,
            {
                "subject": TEACHER,
                "run_seed": 0,
                "model_zip_sha256": t_sha,
                "path": "P8 CPU path (SB3, frozen VecNormalize, training=False), render off",
                "const_mean_from": "the students' training clock mean"
                if means
                else "teacher_const_mean argument",
            },
        )

    record: dict[str, Any] = {
        "what": "closed-loop clock intervention: only the time_fraction input value changes",
        "scope": SCOPE,
        "list": {
            "name": SELECT_SPLIT,
            "file_sha256": data_episodes.file_sha256(split),
            "episodes": len(select),
            "limited_for_test": limit,
        },
        "clock": {
            "field": CLOCK_FIELD,
            "definition": "P8 time_fraction = clip(t_episode / episode_len_s, 0, 1)",
            "episode_len_s": episode_len_s,
            "shift_s": SHIFT_S,
            "clamp_s": CLAMP_S,
            "where": "student: the plus_clock proprio column (both To rows), by field name via "
            "ObsContract.with_proprio_clock; T_ppo: its 25-d observation's time_fraction via "
            "ObsContract.with_obs_clock. Never the env, the physics or the env's observation.",
        },
        "units": {
            "td_*": "s model scale (x5 full scale), time_to_touchdown_s over touched-down episodes",
            "td_sd_s": "sample SD, ddof 1",
            "p95_closing_speed_m_s": "m/s model scale (x5 full scale), P8 quantile_linear 0.95, "
            "touched-down episodes",
        },
        "subjects": subjects,
        "summary": [{k: _json(v) for k, v in r.items()} for r in summary],
        "git": git,
        "workers": workers,
        "wall_s": time.perf_counter() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows([{k: _cell(r[k]) for k in SUMMARY_COLUMNS} for r in summary])
    (out_dir / "clock_intervention.csv").write_text(buf.getvalue(), encoding="utf-8")
    ebuf = io.StringIO()
    ew = csv.DictWriter(ebuf, fieldnames=list(EPISODE_COLUMNS), lineterminator="\n")
    ew.writeheader()
    ew.writerows(episodes_out)
    gz_path = out_dir / "clock_intervention_episodes.csv.gz"
    # mtime 0 and no file name: the bytes depend on the rows only
    gz_path.write_bytes(gzip.compress(ebuf.getvalue().encode("utf-8"), mtime=0))
    mf.write_json_atomic(out_dir / "clock_intervention.json", record)
    return record
