"""Scoring of H1-H5 exactly as frozen in P3-D1 §6 (H1-H4 Phase 8; H5(a), H5(b) Phase 9).

Everything here is simulation only (PyBullet physics, PyBullet TinyRenderer frames). Success is
dimensionless in ``[0, 1]``; "points" are hundredths of it. Times are seconds, model scale
(lambda = 1/25; x5 for full scale); one 30 Hz control step is 33.3 ms model (167 ms full).

Inputs
------
Episode rows (CSV-read mappings, string or typed cells) become :class:`ArmMatrix` objects via
:func:`student_matrix` (this project's rows: ``arm``, ``student_seed``, ``Ta``, ``K``,
``ensemble``, ``latency_steps``, ``render_cfg_sha256``, ``ckpt_sha256``) or :func:`p8_matrix`
(P8's committed rows: ``method``, ``run_seed``). Both select one cell (``regime``, ``ss``),
refuse a missing or extra seed (no seed is dropped, P3-D1 §4), refuse rows of a non-committed
render (appearance shift, ``camera``), and require every seed to cover the same episodes exactly
once. :func:`student_matrix` keeps only rows of one ``latency_steps`` (0, the synchronous arm,
unless the latency arm's ``d`` is asked for), so latency-arm rows never enter a synchronous
matrix. Contrasts then require identical episodes on both sides.

Scorers and their rules (P3-D1 §6, verbatim in substance)
----------------------------------------------------------
* :func:`score_h1`: ``id`` SS5 ``bc_dagger - bc_matched`` (primary), ``id`` SS6 (secondary),
  ``bc_dart - bc`` at SS5 and SS6 (DART secondaries); P8 ``paired_bootstrap``; +5 points.
  Supported: estimate >= +5 points and the CI excludes 0. Inconclusive: the CI excludes 0 and
  the estimate is positive but below +5. Not supported: every other case; a CI wholly below 0
  is "an effect in the opposite direction". When the reference arm (``bc_matched`` / ``bc``)
  has a point estimate above 95 % in that cell the ceiling applies: the rule's verdict is kept
  (a null with a CI that includes 0 is printed "not supported (ceiling-limited: the largest
  attainable gain is below +5 points)"), a caveat states the largest attainable gain computed
  from the reference arm's rate, and the SS6 secondary is printed beside it (P8-D1 item 3).
  ``paired_iqm_difference`` is printed beside as "post hoc, not scored"; labelled-frame counts
  and the success difference with P8-tunnelled successes removed (descriptive) are printed
  beside.
* :func:`score_h2`: ``id`` SS4, ``bc_mix`` and ``dp_mix``, 3 seeds each; D = W1(bc_mix, ref) -
  W1(dp_mix, ref) (:func:`ila.eval.stats.w1_bootstrap`); supported: D >= 0.033 s and the CI
  excludes 0; inconclusive: the CI excludes 0 and 0 < D < 0.033 s; not supported otherwise.
  The reference comes from :func:`build_h2_reference` (P8's committed rows, nothing flown).
* :func:`score_h3`: ``id`` SS6, ``dp`` ``Ta = 1`` minus ``Ta = 8`` on the same checkpoints
  (:func:`ila.eval.stats.joint_seed_contrast`); H1's rule at +5 points; ceiling when ``dp`` at
  ``Ta = 8`` is above 95 %. P6-D5's deck-motion caveat is stated with it, with the same-checkpoint
  gain at every ``id`` sea state given and ``dp``'s static-deck rate at ``Ta = 8``.
* :func:`score_h4`: ``id`` SS6; D = P - I (:func:`ila.eval.stats.h4_three_arm_bootstrap`);
  supported: D > 0 and the CI excludes 0; not supported otherwise, a CI wholly below 0 printed
  as "imitation gap larger".
* :func:`score_h5b`: ``id`` SS5, latency-arm ``dp`` minus synchronous ``dp`` (``Ta = 8``,
  ``K = 10``, no ensembling) on the same checkpoints (:func:`ila.eval.stats.joint_seed_contrast`).
  Supported (non-inferior): the CI lower bound >= -3 points (within :data:`MAGNITUDE_TOL`).
  Not supported: otherwise. No two-sided reversal (P3-D1 §6): a CI wholly above 0 is reported
  as non-inferior, never as an effect in the other direction (:func:`noninferiority_rule`).
* :func:`read_h5a`: H5(a) is scored by ``ila.deploy.latency.score`` (``make bench-score``);
  its verdict is read from ``results/latency/delay.json`` and ``h5a.csv`` as written, checked
  for consistency, never re-derived here.
* :func:`read_latency_facts`: every Phase 9 number quoted in a sentence (why H5(a) failed,
  why ``d`` is what it is, the bc_dagger delay reading, this ``dp``'s measured closed-loop flip
  rate printed beside every ``dp`` contrast) is read from ``results/latency`` when the scorer
  runs; nothing is retyped. Delays are d control steps of simulated time (d x 33.3 ms model
  scale, d x 166.7 ms full scale); measured wall-clock latency is never scaled.
* Unscored, two-sided (:class:`UnscoredContrast`): :func:`h4_p_prime`, :func:`modular_vs_best`
  (the best end-to-end student from committed ``select`` scores, :func:`best_end_to_end`),
  :func:`dart_vs_matched` and :func:`latency_context` (latency arm vs synchronous, same
  checkpoints, at the cells H5(b) does not score).

"Separates" means the 95 % CI excludes 0; magnitudes are judged on the point estimate; there is
no multiplicity correction (P3-D1 §1). Every SS6 contrast carries :data:`FLIP_FLOOR` beside it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from ila import p8_adapter
from ila.eval import h2_select, runner
from ila.eval import stats as st
from ila.eval.determinism import P8_FLOOR

__all__ = [
    "CEILING",
    "CONTROL_STEP_MS",
    "H5A_THRESHOLD_MS",
    "H5B_MARGIN",
    "CORE_SEEDS",
    "ABLATION_SEEDS",
    "END_TO_END_ARMS",
    "FLIP_FLOOR",
    "H2_MAGNITUDE_S",
    "H3_DECK_MOTION",
    "H3_SELECTED_AT",
    "MAGNITUDE_TOL",
    "NO_MULTIPLICITY",
    "SUCCESS_MAGNITUDE",
    "ArmMatrix",
    "DP_FLIP_LABEL",
    "H2Reference",
    "HypothesisResult",
    "LatencyFacts",
    "ProviderFlips",
    "RowError",
    "SelectScore",
    "UnscoredContrast",
    "Verdict",
    "best_end_to_end",
    "build_h2_reference",
    "dart_vs_matched",
    "h4_p_prime",
    "latency_context",
    "load_select_scores",
    "magnitude_rule",
    "modular_vs_best",
    "noninferiority_rule",
    "outcome_counts",
    "read_h5a",
    "read_latency_facts",
    "p8_committed_rows",
    "p8_matrix",
    "pooled_select_p95",
    "score_h1",
    "score_h2",
    "score_h3",
    "score_h4",
    "score_h5b",
    "student_matrix",
]

type Verdict = Literal["supported", "not supported", "inconclusive"]
type FloatArray = npt.NDArray[np.float64]
type Row = Mapping[str, Any]
type Cell = tuple[str, str]

#: H1 and H3 magnitude: +5 points of success (P3-D1 §6).
SUCCESS_MAGNITUDE: Final[float] = 0.05

#: H2 magnitude: 0.033 s model scale (one control step, as P3-D1 §6 writes it).
H2_MAGNITUDE_S: Final[float] = 0.033

#: Ceiling: a reference arm whose point estimate exceeds 100 % - magnitude (P3-D1 §6).
CEILING: Final[float] = 1.0 - SUCCESS_MAGNITUDE

#: H5(b)'s non-inferiority margin: the 95 % CI lower bound must be >= -3 points (P3-D1 §6).
H5B_MARGIN: Final[float] = -0.03

#: H5(a)'s p99 bound as P3-D1 §6 writes it, ms (model-scale control step; P9-D1 item 9).
H5A_THRESHOLD_MS: Final[float] = 33.3

#: One 30 Hz control step, ms model scale (P3-D1 §7's 33.333 ms; 166.7 ms full scale).
CONTROL_STEP_MS: Final[float] = 33.333

#: Model-to-full-scale factor for simulated times (Froude, lambda = 1/25: sqrt(25) = 5). It
#: applies to sim time only; measured wall-clock compute latency is never scaled.
FULL_SCALE_TIME: Final[float] = 5.0

#: Tolerance on "estimate >= magnitude" for float rounding of a difference of means (far
#: below one episode's 1/1 000 resolution); a recorded implementation choice.
MAGNITUDE_TOL: Final[float] = 1e-9

#: Seeds of core arms and of ablations (P3-D1 §3; CLAUDE.md non-negotiable 6).
CORE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ABLATION_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

#: Candidates for "the best end-to-end vision student", in P3-D1 §6's tie order.
END_TO_END_ARMS: Final[tuple[str, ...]] = ("bc", "bc_matched", "bc_dart", "bc_dagger", "dp")

#: P8's one-ulp flip floor, as P3-D1 §1 says it must be described beside every SS6 contrast.
#: The rate comes from ``ila.eval.determinism.P8_FLOOR`` (53/800 -> 6.6 %); the single-draw
#: median (1.9 %) is P3-D1 §1's text and is not in that constant.
FLIP_FLOOR: Final[str] = (
    f"{100.0 * float(P8_FLOOR['rate']):.1f} % of `id` SS6 policy-episodes flip under at least "
    "one of 20 one-ulp draws (single-draw median 1.9 %), measured on P8's MLPs, not on these "
    "students"
)

#: Stated beside every contrast (P3-D1 §1).
NO_MULTIPLICITY: Final[str] = "No multiplicity correction is applied (P3-D1 §1)."

_CEILING_NOTE: Final[str] = (
    "not supported (ceiling-limited: the largest attainable gain is below +5 points)"
)
_OPPOSITE: Final[str] = (
    "not supported (effect in the opposite direction: the CI lies wholly below 0)"
)
_UNITS_SUCCESS: Final[str] = "success difference, dimensionless (x100 = points)"
_UNITS_S: Final[str] = "seconds, model scale (lambda = 1/25; x5 full scale)"

#: P8's committed result files per method (P3-D1 §3 "always printed"; §6 H2's reference).
P8_ROW_FILES: Final[Mapping[str, str]] = {
    "ppo": "results/e07/matrix/episodes.csv",
    "gated": "results/e01/episodes.csv",
    "pid_feedforward": "results/e01/episodes.csv",
    "pid_track_descend": "results/e01/episodes.csv",
    "oracle_gated": "results/e01/episodes.csv",
}


class RowError(ValueError):
    """Rows do not form the cell, arm, seeds or episodes a scorer needs."""


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class HypothesisResult:
    """One scored hypothesis or secondary (P3-D1 §6).

    Attributes:
        hypothesis: ``H1``, ``H1-SS6``, ``H1-DART-SS5``, ``H1-DART-SS6``, ``H2``, ``H3``, ``H4``.
        cell: ``"<list> <sea state>"``, e.g. ``"id SS5"``.
        statistic: The statistic's name and implementation.
        contrast: What is subtracted from what.
        estimate: Point estimate, in ``units``.
        ci_lo: Lower 95 % bound.
        ci_hi: Upper 95 % bound.
        units: Units of ``estimate`` and the bounds.
        n_seeds: ``(arm, seeds)`` per arm.
        n_episodes: Paired episodes in the cell.
        magnitude: The frozen magnitude, in ``units`` (``None`` for H4).
        verdict: ``supported`` / ``not supported`` / ``inconclusive``.
        verdict_note: The verdict as printed (with its reason).
        caveats: Sentences P3-D1 requires stated with the result.
        beside: ``(label, text)`` printed beside the result (unscored numbers, counts, the flip
            floor).
        ceiling_limited: P3-D1 §6's *reference-arm* ceiling only: whether the reference arm's
            point estimate in the scored cell exceeds 100 % - magnitude (:data:`CEILING`), so
            that "supported" was unattainable (H1 family, H3). Hypotheses without a magnitude
            (H2 in seconds, H4) never set it. In particular H4's imitation gap I is bounded
            below by the teacher's own rate (I >= S(teacher) - 1); that is a different ceiling,
            stated in ``caveats`` and ``notes``, and leaves this field False.
        notes: Short reading aids printed with the result, not part of the rule (e.g. H4's
            I-floor and why ``ceiling_limited`` is False there).
    """

    hypothesis: str
    cell: str
    statistic: str
    contrast: str
    estimate: float
    ci_lo: float
    ci_hi: float
    units: str
    n_seeds: tuple[tuple[str, int], ...]
    n_episodes: int
    magnitude: float | None
    verdict: Verdict
    verdict_note: str
    caveats: tuple[str, ...]
    beside: tuple[tuple[str, str], ...]
    ceiling_limited: bool = False
    notes: tuple[str, ...] = ()

    @property
    def separates(self) -> bool:
        """Whether the 95 % CI excludes 0."""
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


@dataclass(frozen=True)
class UnscoredContrast:
    """A contrast P3-D1 reports two-sided and does not score.

    Attributes:
        name: What it is (e.g. ``H4 P'``).
        cell: ``"<list> <sea state>"``.
        statistic: The statistic's name.
        contrast: What is subtracted from what.
        estimate: Point estimate, in ``units``.
        ci_lo: Lower 95 % bound.
        ci_hi: Upper 95 % bound.
        units: Units.
        n_seeds: ``(arm, seeds)`` per arm.
        n_episodes: Paired episodes.
        reading: Two-sided reading: which side the CI lies on, or "does not separate".
        caveats: Sentences stated with it.
        beside: ``(label, text)`` printed beside it.
    """

    name: str
    cell: str
    statistic: str
    contrast: str
    estimate: float
    ci_lo: float
    ci_hi: float
    units: str
    n_seeds: tuple[tuple[str, int], ...]
    n_episodes: int
    reading: str
    caveats: tuple[str, ...]
    beside: tuple[tuple[str, str], ...]


# --------------------------------------------------------------------------- rows -> matrices


@dataclass(frozen=True, eq=False)
class ArmMatrix:
    """One arm's rows in one cell, as seeds x episodes arrays.

    Attributes:
        arm: Arm (``student_matrix``) or P8 method (``p8_matrix``).
        cell: ``(regime, ss)``.
        seeds: Training seeds (rows of the arrays), ascending.
        episodes: ``(index, episode_seed)`` per column, ascending ``index``.
        success: ``(n_seeds, n_episodes)`` 1.0 for ``success``, else 0.0.
        touchdown_s: ``(n_seeds, n_episodes)`` P8's ``time_to_touchdown_s``, s model scale; NaN
            without a touchdown.
        outcomes: Outcome class per seed and episode.
        ckpt_sha256: Checkpoint per seed (``""`` for P8 rows).
        Ta: ``Ta`` of the rows (``None`` when not a chunked policy).
        K: DDIM steps (``None`` when not applicable).
        ensemble: Temporal ensembling (``None`` when not applicable).
        tunnelled: ``(n_seeds, n_episodes)`` 1.0 where P8's ``tunnelled`` flag is set, else
            0.0; ``None`` when the rows carry no ``tunnelled`` column.
        latency_steps: The rows' ``latency_steps`` (the latency arm's ``d``, control steps;
            0 for every synchronous arm and for P8 rows).
    """

    arm: str
    cell: Cell
    seeds: tuple[int, ...]
    episodes: tuple[tuple[int, int], ...]
    success: FloatArray
    touchdown_s: FloatArray
    outcomes: tuple[tuple[str, ...], ...]
    ckpt_sha256: tuple[str, ...]
    Ta: int | None = None
    K: int | None = None
    ensemble: bool | None = None
    tunnelled: FloatArray | None = None
    latency_steps: int = 0

    @property
    def rate(self) -> float:
        """Mean success over seeds x episodes (the point estimate the contrasts use)."""
        return float(self.success.mean())

    @property
    def cell_label(self) -> str:
        """``"<regime> <ss>"``."""
        return f"{self.cell[0]} {self.cell[1]}"


def _blank(v: Any) -> bool:
    return (
        v is None
        or (isinstance(v, str) and v.strip() == "")
        or (isinstance(v, float) and math.isnan(v))
    )


def _opt_int(v: Any) -> int | None:
    return None if _blank(v) else int(float(v))


def _opt_bool(v: Any) -> bool | None:
    if _blank(v):
        return None
    if isinstance(v, bool | np.bool_):
        return bool(v)
    text = str(v).strip()
    if text in ("True", "true", "1"):
        return True
    if text in ("False", "false", "0"):
        return False
    raise RowError(f"not a boolean cell: {v!r}")


def _time(v: Any) -> float:
    if _blank(v):
        return math.nan
    t = float(v)
    return t if math.isfinite(t) else math.nan


_OUTCOMES: Final[frozenset[str]] = frozenset(
    ("crash", "off_pad", "hard_landing", "bounce", "success", "timeout")
)


def _assemble(
    arm: str,
    cell: Cell,
    by_seed: Mapping[int, list[Row]],
    expected_seeds: Sequence[int],
    *,
    ckpt_column: str | None,
    tags: tuple[int | None, int | None, bool | None] = (None, None, None),
    latency_steps: int = 0,
) -> ArmMatrix:
    if sorted(by_seed) != sorted(expected_seeds):
        raise RowError(
            f"{arm} {cell}: seeds {sorted(by_seed)} != expected {sorted(expected_seeds)} "
            "(no seed is dropped or added, P3-D1 §4)"
        )
    seeds = tuple(sorted(by_seed))
    episodes: tuple[tuple[int, int], ...] | None = None
    succ, td, outs, ckpts = [], [], [], []
    tun: list[list[float]] | None = []
    for s in seeds:
        rows = sorted(by_seed[s], key=lambda r: int(r["index"]))
        keys = tuple((int(r["index"]), int(r["episode_seed"])) for r in rows)
        if len(set(keys)) != len(keys):
            raise RowError(f"{arm} {cell} seed {s}: an episode appears twice")
        if episodes is None:
            episodes = keys
        elif keys != episodes:
            raise RowError(f"{arm} {cell} seed {s}: episodes differ from seed {seeds[0]}")
        o = tuple(str(r["outcome"]) for r in rows)
        if not set(o) <= _OUTCOMES:
            raise RowError(f"{arm} {cell} seed {s}: unknown outcome {sorted(set(o) - _OUTCOMES)}")
        outs.append(o)
        succ.append([1.0 if x == "success" else 0.0 for x in o])
        td.append([_time(r["time_to_touchdown_s"]) for r in rows])
        if tun is not None and all("tunnelled" in r for r in rows):
            tun.append([1.0 if _opt_bool(r["tunnelled"]) else 0.0 for r in rows])
        else:
            tun = None
        if ckpt_column is None:
            ckpts.append("")
        else:
            c = {str(r[ckpt_column]) for r in rows}
            if len(c) != 1:
                raise RowError(f"{arm} {cell} seed {s}: rows of {len(c)} checkpoints")
            ckpts.append(c.pop())
    assert episodes is not None
    return ArmMatrix(
        arm=arm,
        cell=cell,
        seeds=seeds,
        episodes=episodes,
        success=np.asarray(succ, dtype=np.float64),
        touchdown_s=np.asarray(td, dtype=np.float64),
        outcomes=tuple(outs),
        ckpt_sha256=tuple(ckpts),
        Ta=tags[0],
        K=tags[1],
        ensemble=tags[2],
        tunnelled=None if tun is None else np.asarray(tun, dtype=np.float64),
        latency_steps=latency_steps,
    )


def _core_render_shas() -> frozenset[str]:
    """``render_cfg_sha256`` of the committed look and of ``off`` (never a variant)."""
    return frozenset((runner.render_cfg_sha256("full", None), runner.render_cfg_sha256("off")))


def student_matrix(
    rows: Iterable[Row],
    *,
    arm: str,
    cell: Cell,
    seeds: Sequence[int] = CORE_SEEDS,
    ta: int | None = None,
    k: int | None = None,
    ensemble: bool | None = None,
    latency_steps: int = 0,
) -> ArmMatrix:
    """Select one student arm's rows in one cell and stack them as seeds x episodes.

    Rows are selected on ``arm``, ``regime``/``ss``, ``Ta``, ``K`` and ``ensemble`` (``None``
    means the cell must be empty) and ``latency_steps`` (0: the synchronous arm; the latency
    arm's ``d`` otherwise). Selected rows must carry the committed render
    (``render_cfg_sha256`` of ``full`` or ``off``): an appearance-shift or ``camera`` row is
    refused, never mixed into a core section.

    Args:
        rows: This project's episode rows (P8 columns, then plan §1.7's columns).
        arm: The arm label.
        cell: ``(regime, ss)``, e.g. ``("id", "SS5")``.
        seeds: The seeds that must all be present (5 core, 3 ablation).
        ta: ``Ta``, executed actions per chunk, or ``None``.
        k: ``K``, DDIM steps, or ``None``.
        ensemble: Temporal ensembling flag, or ``None``.
        latency_steps: The rows' delay ``d`` in control steps (0: synchronous).

    Returns:
        The :class:`ArmMatrix`.

    Raises:
        RowError: On a missing/extra seed, unequal episodes, a duplicate, a non-core render or
            a mixed-checkpoint seed.
    """
    allowed = _core_render_shas()
    by_seed: dict[int, list[Row]] = {}
    for r in rows:
        if r["arm"] != arm or (r["regime"], r["ss"]) != cell:
            continue
        if (_opt_int(r["Ta"]), _opt_int(r["K"]), _opt_bool(r["ensemble"])) != (ta, k, ensemble):
            continue
        if (_opt_int(r["latency_steps"]) or 0) != latency_steps:
            continue
        if str(r["render_cfg_sha256"]) not in allowed:
            raise RowError(f"{arm} {cell}: non-core render {r['render_cfg_sha256']!r}")
        seed = _opt_int(r["student_seed"])
        if seed is None:
            raise RowError(f"{arm} {cell}: a student row without student_seed")
        by_seed.setdefault(seed, []).append(r)
    return _assemble(
        arm,
        cell,
        by_seed,
        seeds,
        ckpt_column="ckpt_sha256",
        tags=(ta, k, ensemble),
        latency_steps=latency_steps,
    )


def p8_matrix(rows: Iterable[Row], *, method: str, cell: Cell, run_seed: int = 0) -> ArmMatrix:
    """Select one P8 method's committed rows (one run seed) in one cell.

    Args:
        rows: P8 episode rows (``method``, ``run_seed``, ``regime``, ``ss`` ...).
        method: P8 method, e.g. ``ppo`` (the teacher, seed 0), ``gated``.
        cell: ``(regime, ss)``.
        run_seed: P8 run seed (the teacher is seed 0, P3-D1 §1).

    Returns:
        A one-row :class:`ArmMatrix`.
    """
    by_seed: dict[int, list[Row]] = {}
    for r in rows:
        if r["method"] != method or (r["regime"], r["ss"]) != cell:
            continue
        if int(float(r["run_seed"])) != run_seed:
            continue
        by_seed.setdefault(run_seed, []).append(r)
    return _assemble(method, cell, by_seed, (run_seed,), ckpt_column=None)


def p8_committed_rows(method: str, cell: Cell, run_seed: int = 0) -> list[dict[str, str]]:
    """Read P8's committed rows of one method, run seed and cell (read-only, nothing flown).

    ``ppo`` comes from ``results/e07/matrix/episodes.csv(.gz)``; ``gated``, ``pid_feedforward``,
    ``pid_track_descend`` and ``oracle_gated`` from ``results/e01/episodes.csv`` (P3-D1 §3, §6).

    Returns:
        The rows as string mappings, in file order.
    """
    text = p8_adapter.read_text(p8_adapter.P8_ROOT / P8_ROW_FILES[method])
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        same_cell = (r["regime"], r["ss"]) == cell
        if r["method"] == method and same_cell and int(r["run_seed"]) == run_seed:
            out.append(r)
    return out


def _same_episodes(*mats: ArmMatrix) -> None:
    first = mats[0]
    for m in mats[1:]:
        if m.cell != first.cell or m.episodes != first.episodes:
            raise RowError(
                f"{m.arm} and {first.arm} do not cover identical episodes of one cell "
                f"({m.cell_label} vs {first.cell_label})"
            )


# --------------------------------------------------------------------------- rules


def magnitude_rule(
    estimate: float, lo: float, hi: float, magnitude: float, *, unit: str
) -> tuple[Verdict, str]:
    """H1/H2/H3's rule (P3-D1 §6), with the two-sided reading.

    Supported: ``estimate >= magnitude`` (within :data:`MAGNITUDE_TOL`) and the CI excludes 0.
    Inconclusive: the CI excludes 0 and ``0 < estimate < magnitude``. Not supported: every
    other case; a CI wholly below 0 is reported as an effect in the opposite direction.

    Args:
        estimate: Point estimate.
        lo: Lower 95 % bound.
        hi: Upper 95 % bound.
        magnitude: Frozen magnitude, same units.
        unit: How to print the magnitude (e.g. ``"+5 points"``, ``"0.033 s"``).

    Returns:
        ``(verdict, verdict_note)``.
    """
    if not all(math.isfinite(x) for x in (estimate, lo, hi)):
        return "not supported", "not supported (statistic or CI undefined)"
    if hi < 0.0:
        return "not supported", _OPPOSITE
    if lo > 0.0 and estimate >= magnitude - MAGNITUDE_TOL:
        return "supported", "supported"
    if lo > 0.0 and estimate > 0.0:
        return "inconclusive", f"inconclusive (CI excludes 0; estimate positive but below {unit})"
    if lo > 0.0:
        return "not supported", "not supported (CI excludes 0 but the estimate is not positive)"
    return "not supported", "not supported (CI includes 0)"


def _two_sided(lo: float, hi: float, a: str, b: str) -> str:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "undefined"
    if lo > 0.0:
        return f"separates: {a} above {b} (CI wholly above 0)"
    if hi < 0.0:
        return f"separates: {a} below {b} (CI wholly below 0)"
    return "does not separate (CI includes 0)"


def _pts(x: float) -> str:
    return f"{100.0 * x:+.1f}"


def _ci_pts(est: float, lo: float, hi: float) -> str:
    return f"{_pts(est)} points [{_pts(lo)}, {_pts(hi)}] (1 decimal)"


def _is_ss6(m: ArmMatrix) -> bool:
    return m.cell[1] == "SS6"


def _flip(m: ArmMatrix) -> tuple[tuple[str, str], ...]:
    return (("SS6 flip floor", FLIP_FLOOR),) if _is_ss6(m) else ()


#: The label of Phase 9's measured ``dp`` flip rate wherever it is printed beside a contrast.
DP_FLIP_LABEL: Final[str] = "dp closed-loop flip rate (Phase 9)"


def _dp_flip_beside(text: str | None) -> tuple[tuple[str, str], ...]:
    return ((DP_FLIP_LABEL, text),) if text else ()


def _require_seeds(m: ArmMatrix, seeds: Sequence[int]) -> None:
    if m.seeds != tuple(sorted(seeds)):
        raise RowError(f"{m.arm} {m.cell_label}: seeds {m.seeds} != {tuple(sorted(seeds))}")


# --------------------------------------------------------------------------- H1


_H1_CAVEAT: Final[str] = (
    "P8's paired_bootstrap treats seeds as unpaired, which is conservative here, because "
    "bc_dagger and bc_matched share iteration 0 (P3-D1 §6 H1)."
)
_H1_MATCHING: Final[str] = (
    "bc_dagger and bc_matched share iteration 0, the loop, the step count, the candidate set "
    "and the 2 000 starting conditions; they differ only in who chose the actions on the "
    "second 1 000 (P3-D1 §6 H1)."
)
_DART_CAVEAT: Final[str] = (
    "bc_dart and bc are both at N = 1 000 labelled episodes on the same 1 000 starting "
    "conditions; bc_dart - bc_matched (1 000 vs 2 000) is reported descriptively, unscored "
    "(P3-D1 §6 H1)."
)


def _frames_text(frames: Mapping[str, Sequence[int]]) -> str:
    parts = []
    for arm, per_seed in frames.items():
        seeds = ", ".join(f"{int(n):,}" for n in per_seed)
        parts.append(f"{arm}: {sum(int(n) for n in per_seed):,} (per seed {seeds})")
    return "; ".join(parts)


def _ceiling_caveat(ref: ArmMatrix, what: str) -> str:
    """The ceiling caveat with the largest attainable gain computed from ``ref``'s rows.

    The gain is ``1 - rate(ref)`` (mean over seeds x episodes), in points.
    """
    k, n = int(ref.success.sum()), ref.success.size
    return (
        f"Ceiling-limited: {what} is at {100.0 * ref.rate:.1f} % ({k}/{n}) in {ref.cell_label}, "
        f"so the largest attainable gain is {100.0 * (1.0 - ref.rate):.1f} points, below +5; "
        "'supported' was unattainable. The rule's verdict is kept (P3-D1 §6 ceilings; P8-D1 "
        "item 3)."
    )


def _tunnel_removed(a: ArmMatrix, b: ArmMatrix) -> tuple[str, str] | None:
    """Success difference with P8-tunnelled successes removed, descriptive (no CI).

    Returns:
        ``(label, text)`` to print beside the contrast, or ``None`` when a side's rows carry no
        ``tunnelled`` column.
    """
    if a.tunnelled is None or b.tunnelled is None:
        return None
    parts, rates = [], []
    for m in (a, b):
        assert m.tunnelled is not None
        tun = int((m.success * m.tunnelled).sum())
        k = int(m.success.sum()) - tun
        rates.append(k / m.success.size)
        parts.append(f"{m.arm} {k}/{m.success.size} (tunnelled successes removed: {tun})")
    return (
        "descriptive, not scored: success minus tunnelled successes",
        f"{'; '.join(parts)}; difference {_pts(rates[0] - rates[1])} points (point estimate "
        "only, no CI; P8's tunnelled flag; the verdict uses success as P8 scores it)",
    )


def _success_contrast(
    hypothesis: str,
    a: ArmMatrix,
    b: ArmMatrix,
    *,
    caveats: tuple[str, ...],
    labelled_frames: Mapping[str, Sequence[int]] | None,
) -> HypothesisResult:
    _same_episodes(a, b)
    _require_seeds(a, CORE_SEEDS)
    _require_seeds(b, CORE_SEEDS)
    c = st.paired_contrast(a.success, b.success)
    post_hoc = st.paired_iqm_contrast(a.success, b.success)
    verdict, note = magnitude_rule(c.diff, c.lo, c.hi, SUCCESS_MAGNITUDE, unit="+5 points")
    extra: tuple[str, ...] = ()
    ceiling = b.rate > CEILING
    if ceiling:
        if verdict == "not supported" and note != _OPPOSITE:
            note = _CEILING_NOTE
        extra = (_ceiling_caveat(b, b.arm),)
    beside: list[tuple[str, str]] = [
        (
            "post hoc, not scored: paired_iqm_difference",
            _ci_pts(post_hoc.diff, post_hoc.lo, post_hoc.hi),
        ),
        (
            "reference arm success (mean over seeds x episodes)",
            f"{b.arm} {100.0 * b.rate:.1f} % ({int(b.success.sum())}/{b.success.size})",
        ),
    ]
    if labelled_frames is not None:
        beside.append(("labelled frames", _frames_text(labelled_frames)))
    tunnel = _tunnel_removed(a, b)
    if tunnel is not None:
        beside.append(tunnel)
    beside.extend(_flip(a))
    return HypothesisResult(
        hypothesis=hypothesis,
        cell=a.cell_label,
        statistic="P8 paired_bootstrap (seeds per arm, shared episodes; 10 000 reps, "
        "seed 20261023)",
        contrast=f"success({a.arm}) - success({b.arm})",
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds=((a.arm, len(a.seeds)), (b.arm, len(b.seeds))),
        n_episodes=c.n_episodes,
        magnitude=SUCCESS_MAGNITUDE,
        verdict=verdict,
        verdict_note=note,
        caveats=(*caveats, *extra, NO_MULTIPLICITY),
        beside=tuple(beside),
        ceiling_limited=ceiling,
    )


def _with_ss6_beside(primary: HypothesisResult, ss6: HypothesisResult) -> HypothesisResult:
    """Attach the SS6 secondary beside ``primary`` whenever the ceiling applies to it."""
    if not primary.ceiling_limited:
        return primary
    text = f"{_ci_pts(ss6.estimate, ss6.ci_lo, ss6.ci_hi)}: {ss6.verdict_note}"
    return replace(primary, beside=(*primary.beside, (f"{ss6.hypothesis} beside it", text)))


def score_h1(
    *,
    dagger_ss5: ArmMatrix,
    matched_ss5: ArmMatrix,
    dagger_ss6: ArmMatrix,
    matched_ss6: ArmMatrix,
    dart_ss5: ArmMatrix,
    bc_ss5: ArmMatrix,
    dart_ss6: ArmMatrix,
    bc_ss6: ArmMatrix,
    labelled_frames: Mapping[str, Sequence[int]],
) -> tuple[HypothesisResult, HypothesisResult, HypothesisResult, HypothesisResult]:
    """Score H1 and its three secondaries (P3-D1 §6 H1).

    Args:
        dagger_ss5: ``bc_dagger`` at ``id`` SS5 (5 seeds).
        matched_ss5: ``bc_matched`` at ``id`` SS5.
        dagger_ss6: ``bc_dagger`` at ``id`` SS6.
        matched_ss6: ``bc_matched`` at ``id`` SS6.
        dart_ss5: ``bc_dart`` at ``id`` SS5.
        bc_ss5: ``bc`` at ``id`` SS5.
        dart_ss6: ``bc_dart`` at ``id`` SS6.
        bc_ss6: ``bc`` at ``id`` SS6.
        labelled_frames: Labelled training frames per arm, per seed in seed order (at least
            ``bc_dagger`` and ``bc_matched``), printed beside H1 and its SS6 secondary.

    Returns:
        ``(H1, H1-SS6, H1-DART-SS5, H1-DART-SS6)``.

    Raises:
        RowError: If an arm or cell is not the one H1 names, or episodes/seeds differ.
    """
    expect = [
        (dagger_ss5, "bc_dagger", "SS5"),
        (matched_ss5, "bc_matched", "SS5"),
        (dagger_ss6, "bc_dagger", "SS6"),
        (matched_ss6, "bc_matched", "SS6"),
        (dart_ss5, "bc_dart", "SS5"),
        (bc_ss5, "bc", "SS5"),
        (dart_ss6, "bc_dart", "SS6"),
        (bc_ss6, "bc", "SS6"),
    ]
    for m, arm, ss in expect:
        if (m.arm, m.cell) != (arm, ("id", ss)):
            raise RowError(f"H1 needs {arm} at id {ss}, got {m.arm} at {m.cell_label}")
    if not {"bc_dagger", "bc_matched"} <= set(labelled_frames):
        raise RowError("H1 prints labelled frames of bc_dagger and bc_matched beside it")
    frames = {k: labelled_frames[k] for k in ("bc_dagger", "bc_matched")}
    dart_frames = {k: v for k, v in labelled_frames.items() if k in ("bc_dart", "bc")} or None
    h1_cav = (_H1_CAVEAT, _H1_MATCHING)
    ss6 = _success_contrast(
        "H1-SS6", dagger_ss6, matched_ss6, caveats=h1_cav, labelled_frames=frames
    )
    primary = _success_contrast(
        "H1", dagger_ss5, matched_ss5, caveats=h1_cav, labelled_frames=frames
    )
    d6 = _success_contrast(
        "H1-DART-SS6", dart_ss6, bc_ss6, caveats=(_DART_CAVEAT,), labelled_frames=dart_frames
    )
    d5 = _success_contrast(
        "H1-DART-SS5", dart_ss5, bc_ss5, caveats=(_DART_CAVEAT,), labelled_frames=dart_frames
    )
    return _with_ss6_beside(primary, ss6), ss6, _with_ss6_beside(d5, d6), d6


def dart_vs_matched(dart: ArmMatrix, matched: ArmMatrix) -> UnscoredContrast:
    """``bc_dart - bc_matched`` (1 000 vs 2 000 labelled episodes), descriptive, unscored."""
    if (dart.arm, matched.arm) != ("bc_dart", "bc_matched"):
        raise RowError(f"need bc_dart and bc_matched, got {dart.arm}, {matched.arm}")
    return _unscored_paired(
        "H1 DART descriptive (unscored)",
        dart,
        matched,
        caveats=(
            "Unequal data: bc_dart has 1 000 labelled episodes, bc_matched 2 000 (P3-D1 §6 H1).",
        ),
    )


# --------------------------------------------------------------------------- H2


@dataclass(frozen=True, eq=False)
class H2Reference:
    """H2's ``T_mix`` reference on ``id`` SS4, from P8's committed rows (P3-D1 §6 H2).

    Attributes:
        cell: ``("id", "SS4")``.
        episodes: ``(index, episode_seed)`` per episode, ascending ``index``.
        assigned: ``ppo`` / ``gated`` per episode (``ila.data.mix.assign_rows``, seed 20261023).
        outcomes: The assigned policy's committed outcome per episode.
        touchdown_s: ``(n_episodes,)`` its ``time_to_touchdown_s``, s model scale; NaN without
            a touchdown.
    """

    cell: Cell
    episodes: tuple[tuple[int, int], ...]
    assigned: tuple[str, ...]
    outcomes: tuple[str, ...]
    touchdown_s: FloatArray


def _index_rows(rows: Iterable[Row], method: str, cell: Cell) -> dict[int, Row]:
    out: dict[int, Row] = {}
    for r in rows:
        if r["method"] != method or (r["regime"], r["ss"]) != cell or int(float(r["run_seed"])):
            continue
        i = int(r["index"])
        if i in out:
            raise RowError(f"{method} {cell}: index {i} appears twice")
        out[i] = r
    return out


def build_h2_reference(
    listed: Sequence[p8_adapter.ListedEpisode],
    ppo_rows: Iterable[Row],
    gated_rows: Iterable[Row],
    *,
    seed: int = h2_select.TMIX_SEED,
) -> H2Reference:
    """Build H2's reference from P8's committed ``ppo`` s0 (e07) and ``gated`` (e01) rows.

    ``ila.data.mix.assign_rows(20261023, id SS4 rows)`` (through
    :func:`ila.eval.h2_select.tmix_assignment`, which asserts one call per cell equals one
    call on the list and that each cell splits in halves) gives each listed episode to
    ``ppo`` or ``gated``; each takes P8's committed row for that episode. Nothing is flown.

    Args:
        listed: P8's frozen ``id`` list rows of SS4, in list order.
        ppo_rows: P8 ``ppo`` rows (``run_seed`` 0 is used) covering those episodes.
        gated_rows: P8 ``gated`` rows covering them.
        seed: The ``T_mix`` seed (20261023).

    Returns:
        The :class:`H2Reference`.

    Raises:
        RowError: If ``listed`` is not one ``id`` SS4 cell, or a committed row is missing or
            disagrees with the listed episode seed.
    """
    cells = {(e.regime, e.ss) for e in listed}
    if cells != {("id", "SS4")}:
        raise RowError(f"H2's reference is id SS4 only, got {sorted(cells)}")
    cell: Cell = ("id", "SS4")
    assigned = h2_select.tmix_assignment(listed, seed)
    by_policy = {
        "ppo": _index_rows(ppo_rows, "ppo", cell),
        "gated": _index_rows(gated_rows, "gated", cell),
    }
    order = sorted(range(len(listed)), key=lambda k: listed[k].index)
    eps, pol, outs, times = [], [], [], []
    for k in order:
        e, policy = listed[k], assigned[k]
        r = by_policy[policy].get(e.index)
        if r is None:
            raise RowError(f"no committed {policy} row for id SS4 index {e.index}")
        if int(r["episode_seed"]) != e.episode_seed:
            raise RowError(f"{policy} row index {e.index}: episode seed differs from the list")
        eps.append((e.index, e.episode_seed))
        pol.append(policy)
        outs.append(str(r["outcome"]))
        times.append(_time(r["time_to_touchdown_s"]))
    return H2Reference(
        cell=cell,
        episodes=tuple(eps),
        assigned=tuple(pol),
        outcomes=tuple(outs),
        touchdown_s=np.asarray(times, dtype=np.float64),
    )


_H2_CAVEATS: Final[tuple[str, ...]] = (
    "Scored on 3 seeds per arm, the ablation count (P3-D1 §6 H2).",
    "The training mixture is 500 : 478 episodes after filtering, but about 1 : 2 by frames "
    "(gated episodes are longer) (P3-D1 §6 H2).",
    "All 22 filtered episodes are gated, mostly its longest waits (P3-D1 §6 H2).",
    "last_action in the proprio can reveal the mode after the first step (P3-D1 §6 H2).",
    "W1's bias grows as touchdowns get fewer, which works against an arm that times out "
    "(P3-D1 §6 H2).",
    "dp_mix's gain over dp is not shown to come from multimodality (Phase 8 before-you-start).",
    "select W1 (results/phase7/h2) is not H2 evidence (Phase 8 before-you-start).",
)


def score_h2(bc_mix: ArmMatrix, dp_mix: ArmMatrix, reference: H2Reference) -> HypothesisResult:
    """Score H2 (P3-D1 §6 H2) at ``id`` SS4.

    Args:
        bc_mix: ``bc_mix`` at ``id`` SS4, seeds 0-2.
        dp_mix: ``dp_mix`` at ``id`` SS4 (``Ta = 8``, K = 10), seeds 0-2.
        reference: :func:`build_h2_reference`.

    Returns:
        The H2 :class:`HypothesisResult`, D in seconds model scale.
    """
    if (bc_mix.arm, dp_mix.arm) != ("bc_mix", "dp_mix"):
        raise RowError(f"H2 needs bc_mix and dp_mix, got {bc_mix.arm}, {dp_mix.arm}")
    if bc_mix.cell != ("id", "SS4") or reference.cell != ("id", "SS4"):
        raise RowError("H2 is scored at id SS4")
    _same_episodes(bc_mix, dp_mix)
    if reference.episodes != bc_mix.episodes:
        raise RowError("the reference does not cover the students' episodes")
    _require_seeds(bc_mix, ABLATION_SEEDS)
    _require_seeds(dp_mix, ABLATION_SEEDS)
    w = st.w1_bootstrap(bc_mix.touchdown_s, dp_mix.touchdown_s, reference.touchdown_s)
    verdict, note = magnitude_rule(w.d, w.lo, w.hi, H2_MAGNITUDE_S, unit="0.033 s")
    if not math.isfinite(w.d):
        note = "not supported (D undefined: an arm or the reference has no touchdown in the cell)"
    n_ppo = sum(p == "ppo" for p in reference.assigned)
    beside = (
        (
            "fraction without touchdown",
            f"bc_mix {w.frac_without_touchdown_bc:.3f}; dp_mix {w.frac_without_touchdown_dp:.3f}; "
            f"reference {w.frac_without_touchdown_ref:.3f}",
        ),
        ("W1 to reference, s model scale", f"bc_mix {w.w1_bc:.4f}; dp_mix {w.w1_dp:.4f}"),
        (
            "undefined replicates (empty pool, excluded)",
            f"{w.n_reps_undefined} of {w.reps}",
        ),
        (
            "reference",
            f"assign_rows(20261023, id SS4): {n_ppo} ppo (P8 e07, seed 0) / "
            f"{len(reference.assigned) - n_ppo} gated (P8 e01); "
            f"{w.n_touchdowns_ref} touched down",
        ),
    )
    return HypothesisResult(
        hypothesis="H2",
        cell="id SS4",
        statistic="W1 bootstrap (ila.eval.stats.w1_bootstrap; 10 000 reps, seed 20261023)",
        contrast="W1(bc_mix, ref) - W1(dp_mix, ref), touched-down time_to_touchdown_s",
        estimate=w.d,
        ci_lo=w.lo,
        ci_hi=w.hi,
        units=_UNITS_S,
        n_seeds=(("bc_mix", w.n_seeds_bc), ("dp_mix", w.n_seeds_dp)),
        n_episodes=w.n_episodes,
        magnitude=H2_MAGNITUDE_S,
        verdict=verdict,
        verdict_note=note,
        caveats=(*_H2_CAVEATS, NO_MULTIPLICITY),
        beside=beside,
    )


# --------------------------------------------------------------------------- H3


#: H3's selection caveat (P3-D1 §6 H3); also the first line of ``h3_ta_curve.png``'s caption.
H3_SELECTED_AT: Final[str] = "Checkpoints were selected at Ta = 8 (P3-D1 §6 H3)."

_H3_CAVEATS: Final[tuple[str, ...]] = (
    H3_SELECTED_AT,
    "Ta = 1 without ensembling can switch modes between replans; so a null cannot separate "
    "'chunking is harmless' from 'replanning causes mode switching' (P3-D1 §6 H3).",
    "dp_minus_last_action (P7-D3 item 7) is a competing explanation for dp recovering at "
    "Ta = 1: an input-freshness effect, stale last_action, alongside chunk length; any sentence "
    "that says dp fails 'because of chunking' must name it (P7-D3, Phase 8 before-you-start).",
)

#: P6-D5's binding H3 caveat (docs/protocol.md P6-D5 item 5, "Reading (revised after review)").
H3_DECK_MOTION: Final[str] = "P6-D5: a Ta contrast cannot attribute the loss to deck motion."


def _check_dp(m: ArmMatrix, cell: Cell, ta: int, ckpts: tuple[str, ...]) -> None:
    if (m.arm, m.cell, m.Ta, m.K, m.ensemble) != ("dp", cell, ta, 10, False):
        raise RowError(
            f"H3 needs dp at {cell[0]} {cell[1]}, Ta={ta}, K=10, no ensembling; got {m.arm} "
            f"{m.cell_label} Ta={m.Ta} K={m.K} ensemble={m.ensemble}"
        )
    _require_seeds(m, CORE_SEEDS)
    if m.ckpt_sha256 != ckpts:
        raise RowError(f"H3: {m.cell_label} Ta={ta} is not on the same checkpoints")


def _h3_deck_motion(
    pairs: Sequence[tuple[ArmMatrix, ArmMatrix]], static_ta8: ArmMatrix | None
) -> str:
    """P6-D5's caveat with the same-checkpoint gains per sea state and the static-deck rate."""
    text = H3_DECK_MOTION
    if pairs:
        gains = " / ".join(_pts(t1.rate - t8.rate) for t1, t8 in pairs)
        cells = " / ".join(t1.cell[1] for t1, _ in pairs)
        text += (
            f" On the frozen lists the gain is {gains} points at {cells} (Ta = 1 minus Ta = 8, "
            "mean over seeds x episodes, same checkpoints; comparing the gains across sea "
            "states is an unpaired reading)"
        )
        if static_ta8 is None:
            text += "."
    if static_ta8 is not None:
        k, n = int(static_ta8.success.sum()), static_ta8.success.size
        text += (", and" if pairs else "") + (
            f" `dp` at Ta = 8 succeeds on {100.0 * static_ta8.rate:.1f} % ({k}/{n}) of the "
            "static-deck episodes."
        )
    return text


def score_h3(
    ta1: ArmMatrix,
    ta8: ArmMatrix,
    *,
    other_ss: Sequence[tuple[ArmMatrix, ArmMatrix]] = (),
    static_ta8: ArmMatrix | None = None,
    dp_flip: str | None = None,
) -> HypothesisResult:
    """Score H3 (P3-D1 §6 H3): ``dp`` success at ``Ta = 1`` minus ``Ta = 8``, ``id`` SS6.

    Same-checkpoint contrast (:func:`ila.eval.stats.joint_seed_contrast`); H1's rule at +5
    points; ceiling-limited when ``dp`` at ``Ta = 8`` is above 95 %. P6-D5's caveat (a ``Ta``
    contrast cannot attribute the loss to deck motion) is always stated; ``other_ss`` and
    ``static_ta8`` supply its numbers and do not enter the statistic.

    Args:
        ta1: ``dp`` at ``Ta = 1``, K = 10, no ensembling, ``id`` SS6, seeds 0-4.
        ta8: ``dp`` at ``Ta = 8``, K = 10, no ensembling, same cell and seeds.
        other_ss: ``(Ta = 1, Ta = 8)`` pairs of the same checkpoints at other ``id`` sea states
            (descriptive gains beside the SS6 one).
        static_ta8: ``dp`` at ``Ta = 8`` on the ``static`` list, same checkpoints.
        dp_flip: Phase 9's measured closed-loop flip rate of this ``dp``
            (:meth:`LatencyFacts.dp_flip_text`), printed beside the P8 flip floor; ``None``
            when Phase 9's results are absent.

    Returns:
        The H3 :class:`HypothesisResult`.

    Raises:
        RowError: If the sides are not the same ``dp`` checkpoints at the stated settings.
    """
    ckpts = ta1.ckpt_sha256
    _check_dp(ta1, ("id", "SS6"), 1, ckpts)
    _check_dp(ta8, ("id", "SS6"), 8, ckpts)
    _same_episodes(ta1, ta8)
    for t1, t8 in other_ss:
        if t1.cell[0] != "id" or t1.cell[1] == "SS6":
            raise RowError(f"H3's other sea states are id cells other than SS6, got {t1.cell}")
        _check_dp(t1, t1.cell, 1, ckpts)
        _check_dp(t8, t1.cell, 8, ckpts)
        _same_episodes(t1, t8)
    if static_ta8 is not None:
        _check_dp(static_ta8, ("static", "static"), 8, ckpts)
    pairs = sorted((*other_ss, (ta1, ta8)), key=lambda p: p[0].cell[1])
    c = st.joint_seed_contrast(ta1.success, ta8.success)
    verdict, note = magnitude_rule(c.diff, c.lo, c.hi, SUCCESS_MAGNITUDE, unit="+5 points")
    extra: tuple[str, ...] = ()
    ceiling = ta8.rate > CEILING
    if ceiling:
        if verdict == "not supported" and note != _OPPOSITE:
            note = _CEILING_NOTE
        extra = (_ceiling_caveat(ta8, "dp at Ta = 8"),)
    return HypothesisResult(
        hypothesis="H3",
        cell="id SS6",
        statistic="joint-seed contrast (ila.eval.stats.joint_seed_contrast; 10 000 reps, "
        "seed 20261023)",
        contrast="success(dp, Ta = 1) - success(dp, Ta = 8), same checkpoints, K = 10",
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds=(("dp", c.n_seeds_a),),
        n_episodes=c.n_episodes,
        magnitude=SUCCESS_MAGNITUDE,
        verdict=verdict,
        verdict_note=note,
        caveats=(
            *_H3_CAVEATS,
            _h3_deck_motion(pairs, static_ta8),
            *extra,
            NO_MULTIPLICITY,
        ),
        beside=(
            ("dp at Ta = 8 (mean over seeds x episodes)", f"{100.0 * ta8.rate:.1f} %"),
            *_flip(ta1),
            *_dp_flip_beside(dp_flip),
        ),
        ceiling_limited=ceiling,
    )


# --------------------------------------------------------------------------- H4


_H4_CAVEATS: Final[tuple[str, ...]] = (
    "P includes the effect of withholding time_fraction, which bc_state sees and bc does not; "
    "P' = S(bc_state) - S(bc_plus_clock) is reported beside it, unscored (P3-D1 §6 H4).",
    "P' uses a bc_plus_clock that relies on the clock to a seed-dependent degree, and "
    "bc_state's clock reliance was not measured (Phase 8 before-you-start).",
    "P also includes bc_state's fp32 training against bc's bf16 (P3-D1 §3, §6 H4).",
    "H4 is conditional on teacher seed 0 (P8 ppo s0) (P3-D1 §1, §6 H4).",
)


def _h4_ceiling(s_teacher: float) -> str:
    """H4's ceiling caveat: I cannot fall below S(teacher) - 1 (stated, not a scoring change)."""
    return (
        f"I is ceiling-limited: the teacher is at {100.0 * s_teacher:.1f} % at id SS6, so "
        f"I = S(teacher) - S(bc_state) >= S(teacher) - 1 = {s_teacher - 1.0:+.3f}; an imitation "
        "gap in bc_state's favour larger than that cannot be observed, which bounds how much "
        "of D can come from a negative I."
    )


def _h4_ceiling_note(s_teacher: float) -> str:
    """Why H4's ``ceiling_limited`` is False although I is ceiling-limited (a reading aid)."""
    return (
        f"I is ceiling-limited (teacher at {100.0 * s_teacher:.1f} % at id SS6, so I >= "
        f"{_pts(s_teacher - 1.0)} points). That is not P3-D1 §6's reference-arm ceiling, which "
        "needs a magnitude H4 does not have, so ceiling_limited is False; the verdict is unchanged."
    )


def score_h4(bc_state: ArmMatrix, bc: ArmMatrix, teacher: ArmMatrix) -> HypothesisResult:
    """Score H4 (P3-D1 §6 H4) at ``id`` SS6: D = P - I.

    Supported: D > 0 and the CI excludes 0. Not supported otherwise; a CI wholly below 0 is
    printed "imitation gap larger".

    Args:
        bc_state: ``bc_state`` at ``id`` SS6, seeds 0-4.
        bc: ``bc`` at ``id`` SS6, seeds 0-4.
        teacher: P8 ``ppo`` seed 0's committed rows at ``id`` SS6 (:func:`p8_matrix`).

    Returns:
        The H4 :class:`HypothesisResult` (P and I beside it).
    """
    if (bc_state.arm, bc.arm, teacher.arm) != ("bc_state", "bc", "ppo"):
        raise RowError(f"H4 needs bc_state, bc, ppo; got {bc_state.arm}, {bc.arm}, {teacher.arm}")
    if bc_state.cell != ("id", "SS6"):
        raise RowError("H4 is scored at id SS6")
    _same_episodes(bc_state, bc, teacher)
    _require_seeds(bc_state, CORE_SEEDS)
    _require_seeds(bc, CORE_SEEDS)
    _require_seeds(teacher, (0,))
    g = st.h4_three_arm_bootstrap(bc_state.success, bc.success, teacher.success)
    verdict: Verdict
    if not all(math.isfinite(x) for x in (g.d, g.d_lo, g.d_hi)):
        verdict, note = "not supported", "not supported (statistic or CI undefined)"
    elif g.d_hi < 0.0:
        verdict, note = "not supported", "not supported (imitation gap larger: CI wholly below 0)"
    elif g.d > 0.0 and g.d_lo > 0.0:
        verdict, note = "supported", "supported"
    elif g.d_lo > 0.0:
        verdict, note = "not supported", "not supported (CI excludes 0 but D is not positive)"
    else:
        verdict, note = "not supported", "not supported (CI includes 0)"
    return HypothesisResult(
        hypothesis="H4",
        cell="id SS6",
        statistic="three-arm bootstrap (ila.eval.stats.h4_three_arm_bootstrap; 10 000 reps, "
        "seed 20261023)",
        contrast="D = P - I; P = S(bc_state) - S(bc), I = S(teacher) - S(bc_state)",
        estimate=g.d,
        ci_lo=g.d_lo,
        ci_hi=g.d_hi,
        units=_UNITS_SUCCESS,
        n_seeds=(("bc_state", g.n_seeds_bc_state), ("bc", g.n_seeds_bc), ("teacher ppo s0", 1)),
        n_episodes=g.n_episodes,
        magnitude=None,
        verdict=verdict,
        verdict_note=note,
        caveats=(*_H4_CAVEATS, _h4_ceiling(g.s_teacher), NO_MULTIPLICITY),
        beside=(
            ("perception gap P", _ci_pts(g.p, g.p_lo, g.p_hi)),
            (
                "imitation gap I",
                f"{_ci_pts(g.i, g.i_lo, g.i_hi)}; ceiling-limited: I >= "
                f"{_pts(g.s_teacher - 1.0)} points (teacher at {100.0 * g.s_teacher:.1f} %)",
            ),
            (
                "success (mean over seeds x episodes)",
                f"teacher {100.0 * g.s_teacher:.1f} %; bc_state {100.0 * g.s_bc_state:.1f} %; "
                f"bc {100.0 * g.s_bc:.1f} %",
            ),
            *_flip(bc_state),
        ),
        ceiling_limited=False,
        notes=(_h4_ceiling_note(g.s_teacher),),
    )


def _unscored_paired(
    name: str, a: ArmMatrix, b: ArmMatrix, *, caveats: tuple[str, ...]
) -> UnscoredContrast:
    _same_episodes(a, b)
    c = st.paired_contrast(a.success, b.success)
    return UnscoredContrast(
        name=name,
        cell=a.cell_label,
        statistic="P8 paired_bootstrap (seeds per arm, shared episodes; 10 000 reps, "
        "seed 20261023)",
        contrast=f"success({a.arm}) - success({b.arm})",
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds=((a.arm, c.n_seeds_a), (b.arm, c.n_seeds_b)),
        n_episodes=c.n_episodes,
        reading=_two_sided(c.lo, c.hi, a.arm, b.arm),
        caveats=(*caveats, "Unscored; no direction predicted.", NO_MULTIPLICITY),
        beside=_flip(a),
    )


def h4_p_prime(bc_state: ArmMatrix, bc_plus_clock: ArmMatrix) -> UnscoredContrast:
    """H4's P' = S(``bc_state``) - S(``bc_plus_clock``) at ``id`` SS6, unscored.

    ``bc_plus_clock`` is a 3-seed ablation (tag ``+clock``); P8 ``paired_bootstrap``.
    """
    if (bc_state.arm, bc_plus_clock.arm) != ("bc_state", "bc_plus_clock"):
        raise RowError(f"P' needs bc_state, bc_plus_clock; got {bc_state.arm}, {bc_plus_clock.arm}")
    if bc_state.cell != ("id", "SS6"):
        raise RowError("P' is reported at id SS6")
    _require_seeds(bc_state, CORE_SEEDS)
    _require_seeds(bc_plus_clock, ABLATION_SEEDS)
    return _unscored_paired(
        "H4 P' (unscored, tag +clock)",
        bc_state,
        bc_plus_clock,
        caveats=(
            "bc_plus_clock relies on the clock to a seed-dependent degree; bc_state's clock "
            "reliance was not measured (Phase 8 before-you-start).",
            "bc_plus_clock has 3 seeds (ablation count), bc_state 5.",
        ),
    )


def modular_vs_best(modular: ArmMatrix, best: ArmMatrix, reason: str) -> UnscoredContrast:
    """``modular`` minus the best end-to-end vision student at ``id`` SS6, two-sided, unscored.

    Args:
        modular: ``modular`` at ``id`` SS6, seeds 0-4.
        best: The arm :func:`best_end_to_end` names, same cell and seeds.
        reason: :func:`best_end_to_end`'s reason, printed beside.
    """
    if modular.arm != "modular" or best.arm not in END_TO_END_ARMS:
        raise RowError(f"need modular vs one of {END_TO_END_ARMS}; got {modular.arm}, {best.arm}")
    if modular.cell != ("id", "SS6"):
        raise RowError("modular vs best is reported at id SS6")
    _require_seeds(modular, CORE_SEEDS)
    _require_seeds(best, CORE_SEEDS)
    out = _unscored_paired(
        "H4 modular vs best end-to-end (unscored)",
        modular,
        best,
        caveats=("modular's frozen teacher receives the true time_fraction (P3-D1 §3).",),
    )
    return replace(out, beside=(("best end-to-end", f"{best.arm}: {reason}"), *out.beside))


# --------------------------------------------------------------------------- H5


def noninferiority_rule(
    estimate: float, lo: float, hi: float, margin: float = H5B_MARGIN
) -> tuple[Verdict, str]:
    """H5(b)'s rule (P3-D1 §6): supported (non-inferior) iff the CI lower bound >= ``margin``.

    ``lo >= margin`` is compared within :data:`MAGNITUDE_TOL` (the P8-D1 item 3 precedent).
    There is no two-sided reversal: a CI wholly above 0 is non-inferior and printed as such; a
    CI wholly below 0 whose lower bound clears the margin is still non-inferior by the rule,
    and the note says the delay's cost is below the margin.

    Args:
        estimate: Point estimate (latency minus synchronous success, dimensionless).
        lo: Lower 95 % bound.
        hi: Upper 95 % bound.
        margin: The non-inferiority margin (-0.03, i.e. -3 points).

    Returns:
        ``(verdict, verdict_note)``.
    """
    pts = f"{100.0 * margin:+.0f} points"
    if not all(math.isfinite(x) for x in (estimate, lo, hi)):
        return "not supported", "not supported (statistic or CI undefined)"
    if lo < margin - MAGNITUDE_TOL:
        return "not supported", f"not supported (CI lower bound below {pts})"
    if lo > 0.0:
        return "supported", (
            f"supported (non-inferior: CI lower bound >= {pts}; the CI lies wholly above 0, "
            "which P3-D1 §6 reports as non-inferior, not as a gain)"
        )
    if hi < 0.0:
        return "supported", (
            f"supported (non-inferior: CI lower bound >= {pts}; the CI lies wholly below 0, so "
            f"the delay costs success, by less than the {pts} margin)"
        )
    return "supported", f"supported (non-inferior: CI lower bound >= {pts})"


def outcome_counts(m: ArmMatrix) -> dict[str, int]:
    """Outcome class -> count over seeds x episodes, in P8's matching order."""
    flat = [o for row in m.outcomes for o in row]
    order = ("crash", "off_pad", "hard_landing", "bounce", "success", "timeout")
    return {o: flat.count(o) for o in order}


def _outcomes_text(m: ArmMatrix) -> str:
    counts = outcome_counts(m)
    return ", ".join(f"{o} {n}" for o, n in counts.items()) + f" (of {m.success.size})"


def _side_label(m: ArmMatrix) -> str:
    return f"{m.arm} d={m.latency_steps}" if m.latency_steps else f"{m.arm} synchronous"


def _check_latency_pair(latency: ArmMatrix, sync: ArmMatrix, delay_steps: int) -> None:
    """Same arm, cell, inference and checkpoints; the left side delayed by ``delay_steps``."""
    if latency.arm != sync.arm or latency.cell != sync.cell:
        raise RowError(
            f"a latency contrast needs one arm and cell, got {latency.arm} {latency.cell_label} "
            f"and {sync.arm} {sync.cell_label}"
        )
    if (latency.Ta, latency.K, latency.ensemble) != (sync.Ta, sync.K, sync.ensemble):
        raise RowError(f"{latency.arm} {latency.cell_label}: the sides differ in inference")
    if latency.latency_steps != delay_steps or sync.latency_steps != 0:
        raise RowError(
            f"{latency.arm} {latency.cell_label}: latency_steps {latency.latency_steps} / "
            f"{sync.latency_steps}, expected {delay_steps} / 0"
        )
    _require_seeds(latency, CORE_SEEDS)
    _require_seeds(sync, CORE_SEEDS)
    if latency.ckpt_sha256 != sync.ckpt_sha256:
        raise RowError(
            f"{latency.arm} {latency.cell_label}: the sides are not the same checkpoints"
        )
    _same_episodes(latency, sync)


def _sim_time_text(d: int) -> str:
    """``d`` control steps as simulated time, model and full scale (never wall clock)."""
    ms = d * CONTROL_STEP_MS
    full_step = FULL_SCALE_TIME * CONTROL_STEP_MS
    return (
        f"{d} × {CONTROL_STEP_MS:.1f} ms = {ms:.1f} ms of model-scale sim time ({d} × "
        f"{full_step:.1f} ms = {FULL_SCALE_TIME * ms:.1f} ms at full scale, as sim time)"
    )


#: Wall-clock latency is measured, not simulated: it does not Froude-scale (P9 review M5).
WALL_CLOCK_NOT_SCALED: Final[str] = (
    "the measured wall-clock latency itself is on this machine (RTX A4000, WSL2) and is not scaled"
)


def _period_text() -> str:
    """H5(a)'s bound described as what it is (P3-D1 §6; P9 review M5)."""
    return (
        f"{H5A_THRESHOLD_MS:.1f} ms, the model-scale 30 Hz control period "
        f"({FULL_SCALE_TIME * CONTROL_STEP_MS:.1f} ms full scale)"
    )


def _delay_text(delay_steps: int, record: Mapping[str, Any] | None) -> str:
    text = f"d = {delay_steps} control steps = {_sim_time_text(delay_steps)}"
    if record:
        text += (
            f"; from results/latency/delay.json: provider {record.get('provider')}, row "
            f"{record.get('row')}, batch-1 p99 {record.get('p99_ms')} ms wall clock, fallback "
            f"{record.get('fallback')}"
        )
    return f"{text}; {WALL_CLOCK_NOT_SCALED}"


_COUNT_WORDS: Final[Mapping[int, str]] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def _ms_short(x: float) -> str:
    return f"{x:.0f}" if x >= 1.0 else f"{x:.2f}"


def _upper_2sig(x: float) -> tuple[str, int]:
    """``x`` rounded up to two significant figures, and its decimal exponent.

    Rounding up keeps a printed "<= x" bound true.
    """
    e = math.floor(math.log10(x))
    unit = 10.0 ** (e - 1)
    up = math.ceil(round(x / unit, 9)) * unit
    return f"{up:.1e}", e


@dataclass(frozen=True)
class ProviderFlips:
    """One flown provider's closed-loop parity flight against the reference (Phase 9).

    Attributes:
        provider: Provider id (``ila.deploy.settings``).
        n: Episodes flown (P8's ``tune`` pool's ``select`` episodes).
        flips: Episodes whose outcome class differs from the reference's.
        max_flips: The parity threshold (``flips <= max_flips`` passes).
        success: Successes of the provider's flight.
        lost: Flips from reference success to a provider non-success.
        gained: Flips from a reference non-success to provider success.
        chunk_gap: Largest open-loop chunk max-abs against the reference (normalised action
            units, dimensionless; ``parity.csv`` ``chunk_unrolled_max_abs`` /
            ``chunk_loop_max_abs``), or NaN for ``bc_dagger``.
    """

    provider: str
    n: int
    flips: int
    max_flips: int
    success: int
    lost: int
    gained: int
    chunk_gap: float


@dataclass(frozen=True)
class LatencyFacts:
    """Phase 9's committed latency results, read once for every sentence that quotes them.

    Every number in the H5 texts, the dp flip rate and the bc_dagger delay reading comes from
    these files (``results/latency``), read when the scorer runs; nothing is retyped.

    Attributes:
        control_period_ms: ``delay.json``'s ``control_period_ms`` (33.333 ms model scale).
        delays: ``delay.json``'s ``arms`` (``d``, provider, row, ``p99_ms``, fallback).
        h5a_rows: ``h5a.csv`` rows (dp_e2e, batch 1).
        threads: ``(row, provider) -> threads`` of the batch-1 rows in ``latency.csv``.
        dp_reference_provider: The reference flight's provider (``torch_cpu_fp32``).
        dp_reference_success: Successes of the reference flight.
        dp_flips: The flown, scored, fp32 ``dp`` providers, in ``closed_loop.csv`` order.
        sources: ``results/latency/<file>`` -> SHA-256 of every file read.
    """

    control_period_ms: float
    delays: Mapping[str, Mapping[str, Any]]
    h5a_rows: tuple[Mapping[str, str], ...]
    threads: Mapping[tuple[str, str], str]
    dp_reference_provider: str
    dp_reference_success: int
    dp_flips: tuple[ProviderFlips, ...]
    sources: Mapping[str, str]

    # ------------------------------------------------------------------ helpers

    def _scored(self) -> list[Mapping[str, str]]:
        return [r for r in self.h5a_rows if r["scored"] == "True"]

    def _timed(self, rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
        return [r for r in rows if r["latency_status"] == "timed" and r["p99_ms"]]

    def _d_of(self, p99_ms: float) -> int:
        """P3-D1 §7's ``d = ceil(p99 / control period)``."""
        return math.ceil(p99_ms / self.control_period_ms)

    def _delay(self, arm: str, d: int) -> Mapping[str, Any] | None:
        rec = self.delays.get(arm)
        if rec is None:
            return None
        if rec.get("d") != d:
            raise RowError(f"{arm}: delay.json d {rec.get('d')} != flown d {d}")
        p99 = float(rec["p99_ms"])
        if self._d_of(p99) != d:
            raise RowError(
                f"{arm}: delay.json d {d} != ceil({p99} / {self.control_period_ms}) = "
                f"{self._d_of(p99)}"
            )
        return rec

    # ------------------------------------------------------------------ texts

    def dp_flip_text(self) -> str | None:
        """Phase 9's measured closed-loop flip rate of this ``dp`` (``None``: none flown).

        The range of ``flips / n`` over the flown, scored, fp32 providers (``closed_loop.csv``),
        with the open-loop chunk gap that perturbation amounts to (``parity.csv``).
        """
        fl = self.dp_flips
        if not fl:
            return None
        n = fl[0].n
        lo, hi = min(f.flips for f in fl), max(f.flips for f in fl)
        rng = f"{lo}/{n}" if lo == hi else f"{lo}/{n}–{hi}/{n}"
        pct = (
            f"{100.0 * lo / n:.1f} %"
            if lo == hi
            else f"{100.0 * lo / n:.1f}–{100.0 * hi / n:.1f} %"
        )
        gap, e = _upper_2sig(max(f.chunk_gap for f in fl))
        per = ", ".join(f"{f.provider} {f.flips}/{n}" for f in fl)
        return (
            f"{rng} episodes ({pct}) change outcome class against the "
            f"{self.dp_reference_provider} reference under numeric perturbations of order "
            f"1e{e} (open-loop chunk max-abs <= {gap}), the range over the flown scored fp32 "
            f"providers ({per}); measured on this dp's {n} select episodes in Phase 9 "
            "(closed-loop parity), not on a frozen list (seed 0's selected checkpoint; "
            "results/latency/closed_loop.csv, parity.csv)"
        )

    def h5a_note(self, verdict: str) -> str:
        """Sentences appended to H5(a)'s verdict note when it is not supported (S3)."""
        if verdict != "not supported":
            return ""
        parts: list[str] = []
        scored = self._scored()
        fl = self.dp_flips
        cl_failed = [
            r for r in scored if r["open_loop_pass"] == "True" and r["closed_loop_pass"] == "False"
        ]
        if cl_failed and fl:
            n, ref = fl[0].n, self.dp_reference_success
            gap, _ = _upper_2sig(max(f.chunk_gap for f in fl))
            tc = next((f for f in fl if f.provider == "torch_cuda_fp32"), None)
            both = all(f.lost > 0 and f.gained > 0 for f in fl)
            lost_gained = "lost / gained: " + ", ".join(
                f"{f.provider} {f.lost} / {f.gained}" for f in fl
            )
            counts = ", ".join(f"{f.provider} {f.success}" for f in fl)
            spread = max(abs(f.success - ref) for f in fl)
            evidence = (
                f"torch_cuda_fp32 runs the identical Python sampler as the "
                f"{self.dp_reference_provider} reference (torch eager, the DPStudent path) and "
                f"still flips ({tc.flips}/{n} > {tc.max_flips}); "
                if tc is not None and tc.flips > tc.max_flips
                else ""
            )
            lead = (
                "dp's closed-loop parity failure reflects outcome sensitivity of this dp, not a "
                "provider error: "
                if evidence and both
                else "dp's closed-loop parity failed on every flown scored fp32 provider: "
            )
            parts.append(
                f"{lead}{evidence}open-loop chunk gaps are <= {gap} "
                "(results/latency/parity.csv); "
                + ("the flips go both ways " if both else "")
                + "(successes "
                + f"{lost_gained}), and the provider success counts lie near the reference "
                f"({counts} of {n} against {ref}, within ±{spread}; "
                "results/latency/closed_loop.csv)."
            )
        timed = self._timed(scored)
        untimed = [r for r in scored if r not in timed]
        if timed:
            thr = H5A_THRESHOLD_MS
            p99s = ", ".join(f"{r['provider']} {r['p99_ms']} ms" for r in timed)
            failing = [r for r in timed if float(r["p99_ms"]) >= thr]
            if len(failing) == len(timed):
                rel = ">" if all(float(r["p99_ms"]) > thr for r in timed) else ">="
                text = (
                    "The latency bound also fails independently on every timed scored provider: "
                    f"dp_e2e batch-1 p99 {p99s} (results/latency/h5a.csv), all {rel} "
                    f"{_period_text()}; {WALL_CLOCK_NOT_SCALED}"
                )
            else:
                below = ", ".join(r["provider"] for r in timed if r not in failing)
                text = (
                    f"The latency bound alone (p99 < {_period_text()}) is met by {below}: "
                    f"dp_e2e batch-1 p99 {p99s} (results/latency/h5a.csv); "
                    f"{WALL_CLOCK_NOT_SCALED}"
                )
            if untimed:
                text += "; not timed: " + ", ".join(
                    f"{r['provider']}: {r['latency_status']}" for r in untimed
                )
            parts.append(text + ".")
        flip = self.dp_flip_text()
        if flip:
            parts.append(f"Phase 9 measured this dp's own closed-loop flip rate: {flip}.")
        return " ".join(parts)

    def h5b_beside(self, d: int) -> tuple[tuple[str, str], ...]:
        """Why ``d`` is what it is, and the zero-action start it includes (S4)."""
        rec = self._delay("dp", d)
        out: list[tuple[str, str]] = []
        if rec is not None:
            p99 = float(rec["p99_ms"])
            prov, row = str(rec.get("provider")), str(rec.get("row"))
            if rec.get("fallback"):
                th = self.threads.get((row, prov))
                th_txt = f" ({th} thread{'' if th == '1' else 's'})" if th else ""
                passed = [r for r in self._scored() if r["parity_pass"] == "True"]
                why = (
                    "because no scored provider passed open- and closed-loop parity"
                    if not passed
                    else "as delay.json records a fallback"
                )
                text = (
                    f"d = {d} is the {prov}{th_txt} fallback delay from "
                    f"results/latency/delay.json ({row} batch-1 p99 {p99:.1f} ms wall clock; "
                    f"ceil({p99:.1f} / {self.control_period_ms}) = {d}), {why} (P3-D1 §7)"
                )
                cand = sorted(
                    (r for r in self._timed(self._scored()) if r["open_loop_pass"] == "True"),
                    key=lambda r: float(r["p99_ms"]),
                )
                if cand:
                    f0 = cand[0]
                    p0 = float(f0["p99_ms"])
                    text += (
                        ". Under §7's rule, the fastest open-loop-passing scored provider, "
                        f"{f0['provider']} (dp_e2e batch-1 p99 {f0['p99_ms']} ms, "
                        f"results/latency/h5a.csv), would have given d = ceil({f0['p99_ms']} / "
                        f"{self.control_period_ms}) = {self._d_of(p0)}"
                    )
                    if len(cand) > 1:
                        text += (
                            " (every open-loop-passing scored provider: "
                            + ", ".join(
                                f"{r['provider']} -> d = {self._d_of(float(r['p99_ms']))}"
                                for r in cand
                            )
                            + ")"
                        )
                    text += "; those delays were not flown"
                text += f". The verdict is unchanged; {WALL_CLOCK_NOT_SCALED}"
            else:
                text = (
                    f"d = {d} from {prov} ({row} batch-1 p99 {p99:.1f} ms wall clock, the "
                    "fastest scored provider passing open- and closed-loop parity; "
                    f"results/latency/delay.json); {WALL_CLOCK_NOT_SCALED}"
                )
            out.append(("why d", text))
        out.append(
            (
                "zero-action start",
                f"the contrast includes the d = {d} zero-action steps at the start of every "
                f"episode ({_sim_time_text(d)}; P3-D1 §7), which cannot be separated out of "
                "the estimate. The verdict is unchanged",
            )
        )
        return tuple(out)

    def bc_context_caveats(self, d: int) -> tuple[str, ...]:
        """The bc_dagger latency-context reading (M4): a whole control step, not the compute."""
        rec = self._delay("bc_dagger", d)
        if rec is None:
            return ()
        p99 = float(rec["p99_ms"])
        word = _COUNT_WORDS.get(d, str(d))
        steps = "step" if d == 1 else "steps"
        return (
            f"This contrast is the cost of a {word}-control-step delay (ceil of a {p99:.2f} ms "
            f"p99 to {word} {CONTROL_STEP_MS:.1f} ms model-scale {steps}), not of "
            f"{_ms_short(p99)} ms of compute (results/latency/delay.json: {rec.get('provider')}, "
            f"{rec.get('row')} batch-1 p99; {WALL_CLOCK_NOT_SCALED}).",
            "last_action in proprio carries the applied (delayed) action, a feedback channel the "
            "student never saw in training; no teacher-under-delay reference exists to "
            "attribute the loss.",
        )


#: ``closed_loop.csv`` / ``parity.csv`` columns the facts read (checked on read).
_CLOSED_LOOP_NEEDED: Final[tuple[str, ...]] = (
    "arm",
    "provider",
    "scored",
    "status",
    "n",
    "flips",
    "max_flips",
    "success",
    "reference_equals_selection_rows",
)
_PARITY_NEEDED: Final[tuple[str, ...]] = (
    "arm",
    "provider",
    "precision",
    "chunk_unrolled_max_abs",
    "chunk_loop_max_abs",
)
_LATENCY_NEEDED: Final[tuple[str, ...]] = ("row", "provider", "batch_size", "threads")


def _read_rows(path: Path, needed: Sequence[str]) -> tuple[list[dict[str, str]], str]:
    text = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    missing = [c for c in needed if rows and c not in rows[0]]
    if missing:
        raise RowError(f"{path}: missing columns {missing}")
    return rows, hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_latency_facts(latency_dir: Path) -> LatencyFacts | None:
    """Read Phase 9's committed latency results (``None`` before ``make bench-score``).

    Reads ``h5a.csv``, ``delay.json``, ``closed_loop.csv``, ``parity.csv``, ``latency.csv``
    and ``closed_loop/dp/<provider>.json`` (the flipped episodes) under ``latency_dir``.

    Raises:
        RowError: If ``h5a.csv`` and ``delay.json`` exist but another file is missing, lacks a
            column, or disagrees with itself (flip lists against flip counts, success counts
            against the reference and the flips, the reference against the selection rows).
    """
    h5a_path, delay_path = latency_dir / "h5a.csv", latency_dir / "delay.json"
    if not (h5a_path.is_file() and delay_path.is_file()):
        return None
    need = ("closed_loop.csv", "parity.csv", "latency.csv")
    absent = [f for f in need if not (latency_dir / f).is_file()]
    if absent:
        raise RowError(f"{latency_dir}: h5a.csv is written but {absent} are not")
    h5a, h5a_sha = _read_rows(h5a_path, _H5A_NEEDED)
    cl, cl_sha = _read_rows(latency_dir / "closed_loop.csv", _CLOSED_LOOP_NEEDED)
    par, par_sha = _read_rows(latency_dir / "parity.csv", _PARITY_NEEDED)
    lat, lat_sha = _read_rows(latency_dir / "latency.csv", _LATENCY_NEEDED)
    delay = json.loads(delay_path.read_text(encoding="utf-8"))
    sources = {
        "results/latency/h5a.csv": h5a_sha,
        "results/latency/delay.json": hashlib.sha256(delay_path.read_bytes()).hexdigest(),
        "results/latency/closed_loop.csv": cl_sha,
        "results/latency/parity.csv": par_sha,
        "results/latency/latency.csv": lat_sha,
    }
    threads = {(r["row"], r["provider"]): r["threads"] for r in lat if r["batch_size"] == "1"}
    precision = {(r["arm"], r["provider"]): r["precision"] for r in par}
    gaps = {
        r["provider"]: max(float(r["chunk_unrolled_max_abs"]), float(r["chunk_loop_max_abs"]))
        for r in par
        if r["arm"] == "dp" and r["chunk_unrolled_max_abs"] and r["chunk_loop_max_abs"]
    }
    refs = [r for r in cl if r["arm"] == "dp" and r["status"] == "reference"]
    flips: list[ProviderFlips] = []
    ref_provider, ref_success = "", 0
    if refs:
        if len(refs) != 1 or refs[0]["reference_equals_selection_rows"] != "True":
            raise RowError(
                "closed_loop.csv: dp needs one reference flight equal to its selection rows"
            )
        ref_provider, ref_success = refs[0]["provider"], int(refs[0]["success"])
        ref_n = int(refs[0]["n"])
        for r in cl:
            prov = r["provider"]
            if not (
                r["arm"] == "dp"
                and r["scored"] == "True"
                and r["status"] == "flown"
                and precision.get(("dp", prov)) == "fp32"
            ):
                continue
            rel = f"closed_loop/dp/{prov}.json"
            path = latency_dir / rel
            if not path.is_file():
                raise RowError(f"{latency_dir}: {rel} is missing (dp {prov} was flown)")
            raw = path.read_bytes()
            flipped = json.loads(raw).get("flipped", [])
            n, k, succ = int(r["n"]), int(r["flips"]), int(r["success"])
            if len(flipped) != k or n != ref_n:
                raise RowError(f"{rel}: {len(flipped)} flipped episodes / n {n}; closed_loop.csv "
                               f"says {k} flips / reference n {ref_n}")  # fmt: skip
            lost = sum(f["reference"] == "success" and f["provider"] != "success" for f in flipped)
            gained = sum(
                f["provider"] == "success" and f["reference"] != "success" for f in flipped
            )
            if succ != ref_success - lost + gained:
                raise RowError(
                    f"{rel}: success {succ} != reference {ref_success} - {lost} + {gained}"
                )
            if prov not in gaps:
                raise RowError(f"parity.csv: no dp chunk gap for {prov}")
            sources[f"results/latency/{rel}"] = hashlib.sha256(raw).hexdigest()
            flips.append(
                ProviderFlips(
                    provider=prov,
                    n=n,
                    flips=k,
                    max_flips=int(r["max_flips"]),
                    success=succ,
                    lost=lost,
                    gained=gained,
                    chunk_gap=gaps[prov],
                )
            )
    return LatencyFacts(
        control_period_ms=float(delay.get("control_period_ms", CONTROL_STEP_MS)),
        delays=delay.get("arms", {}),
        h5a_rows=tuple(h5a),
        threads=threads,
        dp_reference_provider=ref_provider,
        dp_reference_success=ref_success,
        dp_flips=tuple(flips),
        sources=sources,
    )


_H5B_CAVEATS: Final[tuple[str, ...]] = (
    "No two-sided reading for H5(b): a CI wholly above 0 is non-inferior and reported as such "
    "(P3-D1 §6).",
    "Same checkpoints on both sides (the core dp selections); each replicate resamples seed "
    "indices jointly for both sides (P3-D1 §1). The parity verdicts behind d were measured on "
    "seed 0's selected checkpoint and apply to all five seeds (P3-D1 §3).",
    "The delay is simulated: PyBullet does not advance while the network computes; each "
    "request's result is computed when it is issued, on the CPU fp32 single-sample path, and "
    "held back d control steps (P3-D1 §7; P9-D1 item 10). Both sides fly that same path "
    "(P3-D1 §3).",
    "One request in flight at a time; the returned chunk executes from index (To - 1) + d, and "
    "past Tp - 1 its last action is held (from d > 7 at To = 2, Tp = 16, Ta = 8); the first "
    "action of an episode is zero until the first result lands (P3-D1 §7).",
    "d comes from a p99 measured on this machine (RTX A4000, i9-10980XE, WSL2); no embedded "
    "target is measured (P3-D1 §6 H5).",
)


def _dp_baseline_caveat(
    sync: ArmMatrix, sync_ss6: ArmMatrix | None, sync_static: ArmMatrix | None
) -> str:
    """Phase 9 before-you-start: the low synchronous dp baseline, from the rows."""

    def at(m: ArmMatrix) -> str:
        where = "static" if m.cell[0] == "static" else m.cell_label
        return f"{100.0 * m.rate:.1f} % ({int(m.success.sum())}/{m.success.size}) at {where}"

    parts = [at(m) for m in (sync, sync_ss6, sync_static) if m is not None]
    return (
        "Synchronous core dp at Ta = 8 succeeds on " + "; ".join(parts) + " (mean over seeds x "
        "episodes). H5(b) compares this dp, distilled from P8 ppo s0, with itself, so a low "
        "baseline does not invalidate the contrast; non-inferiority is relative to that baseline "
        "(Phase 9 before-you-start)."
    )


def score_h5b(
    latency: ArmMatrix,
    sync: ArmMatrix,
    *,
    delay_steps: int,
    delay_record: Mapping[str, Any] | None = None,
    sync_ss6: ArmMatrix | None = None,
    sync_static: ArmMatrix | None = None,
    facts: LatencyFacts | None = None,
) -> HypothesisResult:
    """Score H5(b) (P3-D1 §6): latency-arm ``dp`` minus synchronous ``dp`` at ``id`` SS5.

    Same-checkpoint contrast (:func:`ila.eval.stats.joint_seed_contrast`) and
    :func:`noninferiority_rule` (CI lower bound >= -3 points). ``sync_ss6`` and
    ``sync_static`` only supply the low-baseline caveat's numbers.

    Args:
        latency: ``dp`` at ``id`` SS5, ``Ta = 8``, ``K = 10``, no ensembling, rows with
            ``latency_steps == delay_steps``, seeds 0-4.
        sync: The synchronous core ``dp`` rows of the same checkpoints and cell.
        delay_steps: ``d`` (control steps) the matrix flew.
        delay_record: ``delay.json``'s ``arms.dp`` record (provider, p99), printed beside.
        sync_ss6: Synchronous core ``dp`` at ``id`` SS6 (caveat numbers only).
        sync_static: Synchronous core ``dp`` on ``static`` (caveat numbers only).
        facts: Phase 9's latency results (:func:`read_latency_facts`): why ``d`` is what it
            is, the zero-action start and this ``dp``'s measured flip rate, printed beside.

    Returns:
        The H5(b) :class:`HypothesisResult`.

    Raises:
        RowError: If the sides are not the same ``dp`` checkpoints at id SS5, Ta 8, K 10, no
            ensembling, with the stated delays.
    """
    want = ("dp", ("id", "SS5"), 8, 10, False)
    for m in (latency, sync):
        if (m.arm, m.cell, m.Ta, m.K, m.ensemble) != want:
            raise RowError(
                f"H5(b) needs dp at id SS5, Ta=8, K=10, no ensembling; got {m.arm} "
                f"{m.cell_label} Ta={m.Ta} K={m.K} ensemble={m.ensemble}"
            )
    _check_latency_pair(latency, sync, delay_steps)
    if delay_steps < 0:
        raise RowError(f"H5(b): d = {delay_steps} is negative")
    if delay_record is not None and delay_record.get("d") != delay_steps:
        raise RowError(f"H5(b): delay.json d {delay_record.get('d')} != flown d {delay_steps}")
    c = st.joint_seed_contrast(latency.success, sync.success)
    verdict, note = noninferiority_rule(c.diff, c.lo, c.hi)
    beside = (
        ("delay", _delay_text(delay_steps, delay_record)),
        *(facts.h5b_beside(delay_steps) if facts is not None else ()),
        (
            "success (mean over seeds x episodes)",
            f"latency arm {100.0 * latency.rate:.1f} % ({int(latency.success.sum())}/"
            f"{latency.success.size}); synchronous {100.0 * sync.rate:.1f} % "
            f"({int(sync.success.sum())}/{sync.success.size})",
        ),
        (
            "outcome classes, seed-episodes",
            f"latency arm: {_outcomes_text(latency)}; synchronous: {_outcomes_text(sync)}",
        ),
        *_dp_flip_beside(facts.dp_flip_text() if facts is not None else None),
    )
    return HypothesisResult(
        hypothesis="H5(b)",
        cell="id SS5",
        statistic="joint-seed contrast (ila.eval.stats.joint_seed_contrast; 10 000 reps, "
        "seed 20261023)",
        contrast=f"success(dp, latency arm, d = {delay_steps}) - success(dp, synchronous), same "
        "checkpoints, Ta = 8, K = 10, no ensembling",
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds=(("dp", c.n_seeds_a),),
        n_episodes=c.n_episodes,
        magnitude=H5B_MARGIN,
        verdict=verdict,
        verdict_note=note,
        caveats=(
            *_H5B_CAVEATS,
            _dp_baseline_caveat(sync, sync_ss6, sync_static),
            NO_MULTIPLICITY,
        ),
        beside=beside,
    )


def latency_context(
    latency: ArmMatrix,
    sync: ArmMatrix,
    *,
    delay_steps: int,
    facts: LatencyFacts | None = None,
) -> UnscoredContrast:
    """Latency arm minus synchronous arm in one cell, same checkpoints, two-sided, unscored.

    P3-D1 §7 flies the latency arm on ``id`` SS4-SS6 and scores H5(b) at SS5 only; this is the
    rest (``bc_dagger`` at SS4-SS6, ``dp`` at SS4 and SS6) as context. Statistic:
    :func:`ila.eval.stats.joint_seed_contrast` (the same checkpoints on both sides). With
    ``facts`` (Phase 9's results), ``bc_dagger`` cells state that the contrast is the cost of
    whole control steps and that ``last_action`` carries the delayed action, and ``dp`` cells
    carry this ``dp``'s measured closed-loop flip rate beside the P8 flip floor.

    Raises:
        RowError: If the sides are not one arm's same checkpoints in one cell.
    """
    if latency.arm not in ("bc_dagger", "dp") or latency.cell[0] != "id":
        raise RowError(f"latency context is bc_dagger / dp on id, got {latency.arm}")
    if latency.arm == "dp" and latency.cell[1] == "SS5":
        raise RowError("dp at id SS5 is H5(b), scored, not context")
    _check_latency_pair(latency, sync, delay_steps)
    c = st.joint_seed_contrast(latency.success, sync.success)
    a, b = _side_label(latency), _side_label(sync)
    record = None if facts is None else facts.delays.get(latency.arm)
    bc_reading: tuple[str, ...] = ()
    dp_flip: str | None = None
    if facts is not None and latency.arm == "bc_dagger":
        bc_reading = facts.bc_context_caveats(delay_steps)
    elif facts is not None:
        dp_flip = facts.dp_flip_text()
    return UnscoredContrast(
        name=f"H5(b) context: {latency.arm} latency arm - synchronous ({latency.cell[1]}, "
        "unscored)",
        cell=latency.cell_label,
        statistic="joint-seed contrast (ila.eval.stats.joint_seed_contrast; 10 000 reps, "
        "seed 20261023)",
        contrast=f"success({a}) - success({b}), same checkpoints",
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds=((a, c.n_seeds_a), (b, c.n_seeds_b)),
        n_episodes=c.n_episodes,
        reading=_two_sided(c.lo, c.hi, a, b),
        caveats=(
            "Unscored context: P3-D1 §7 scores H5(b) at id SS5 only (dp).",
            *bc_reading,
            f"Outcome classes, seed-episodes: {a}: {_outcomes_text(latency)}; {b}: "
            f"{_outcomes_text(sync)}.",
            NO_MULTIPLICITY,
        ),
        beside=(
            ("delay", _delay_text(delay_steps, record)),
            *_flip(latency),
            *_dp_flip_beside(dp_flip),
        ),
    )


#: ``h5a.csv``'s columns (``ila.deploy.latency.H5A_COLUMNS``; checked on read).
_H5A_NEEDED: Final[tuple[str, ...]] = (
    "hypothesis",
    "row",
    "provider",
    "scored",
    "batch_size",
    "latency_status",
    "p50_ms",
    "p99_ms",
    "max_ms",
    "open_loop_pass",
    "closed_loop_pass",
    "parity_pass",
    "threshold_ms",
    "p99_below_threshold",
    "verdict",
)

_H5A_CAVEATS: Final[tuple[str, ...]] = (
    "Wall-clock latency of one full replanning call at batch 1 on this machine (RTX A4000, "
    "i9-10980XE, WSL2); no embedded target is measured, and nothing here transfers to one "
    "(P3-D1 §6 H5).",
    "Parity verdicts are measured on seed 0's selected checkpoint and apply to all five seeds "
    "(P3-D1 §3).",
    f"The bound is {_period_text()}, as P3-D1 §6 writes it; d uses {CONTROL_STEP_MS} ms "
    f"(P3-D1 §7; P9-D1 item 9); {WALL_CLOCK_NOT_SCALED}.",
    "Scored by ila.deploy.latency.score (make bench-score) and read here as written; not "
    "re-derived (results/latency/h5a.csv, delay.json).",
)


def read_h5a(latency_dir: Path) -> dict[str, Any]:
    """H5(a)'s record from the deploy scorer's committed output (never re-derived).

    Reads ``h5a.csv`` and ``delay.json`` under ``latency_dir`` (and, through
    :func:`read_latency_facts`, the closed-loop, parity and latency tables the verdict note
    quotes). The verdict is ``delay.json``'s ``h5a_verdict``; every scored ``h5a.csv`` row must
    carry the same verdict. The printed estimate is the lowest ``p99_ms`` among scored rows with
    ``parity_pass`` True and a timed latency (the rows H5(a) counts), ms wall clock on this
    machine. A "not supported" note adds :meth:`LatencyFacts.h5a_note` (why the parity failure
    is this dp's outcome sensitivity, the latency bound on its own, this dp's flip rate).

    Args:
        latency_dir: ``results/latency``.

    Returns:
        The record (``hypotheses.json`` shape plus ``providers`` and ``delays``), with
        ``status`` ``scored`` for supported / not supported, else the scorer's ``not scored``
        text, or ``not scored: ...`` when the files are absent.

    Raises:
        RowError: If the files disagree, or ``h5a.csv`` lacks a column.
    """
    csv_path, delay_path = latency_dir / "h5a.csv", latency_dir / "delay.json"
    base: dict[str, Any] = {
        "hypothesis": "H5(a)",
        "cell": "exported dp (K = 10), dp_e2e, batch 1",
        "kind": "latency",
    }
    if not (csv_path.is_file() and delay_path.is_file()):
        return {
            **base,
            "status": "not scored: results/latency/h5a.csv and delay.json are not written yet "
            "(make bench-score after the closed-loop parity and latency stages)",
        }
    facts = read_latency_facts(latency_dir)
    assert facts is not None  # both files exist
    text = csv_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if rows and tuple(rows[0]) != _H5A_NEEDED:
        raise RowError(f"{csv_path}: columns {tuple(rows[0])} != {_H5A_NEEDED}")
    delay = json.loads(delay_path.read_text(encoding="utf-8"))
    verdict = str(delay.get("h5a_verdict", ""))
    scored = [r for r in rows if r["scored"] == "True"]
    if {r["verdict"] for r in scored} - {verdict}:
        raise RowError(
            f"{csv_path}: scored rows' verdicts {sorted({r['verdict'] for r in scored})} != "
            f"delay.json h5a_verdict {verdict!r}"
        )
    counted = [
        float(r["p99_ms"])
        for r in scored
        if r["parity_pass"] == "True" and r["latency_status"] == "timed" and r["p99_ms"]
    ]
    providers = [dict(r) for r in rows]
    beside = [
        (
            f"{r['provider']} ({'scored' if r['scored'] == 'True' else 'unscored'})",
            f"latency {r['latency_status']}; p50 {r['p50_ms'] or 'n/a'} ms, p99 "
            f"{r['p99_ms'] or 'n/a'} ms, max {r['max_ms'] or 'n/a'} ms; open-loop parity "
            f"{r['open_loop_pass']}, closed-loop {r['closed_loop_pass'] or 'not flown'}",
        )
        for r in rows
    ]
    for arm, rec in sorted(delay.get("arms", {}).items()):
        beside.append(
            (
                f"delay d, {arm}",
                f"d = {rec.get('d')} control steps = {_sim_time_text(int(rec['d']))}; provider "
                f"{rec.get('provider')}, p99 {rec.get('p99_ms')} ms wall clock (not scaled), "
                f"fallback {rec.get('fallback')}"
                if rec.get("d") is not None
                else str(rec.get("status")),
            )
        )
    sources = dict(facts.sources)
    if verdict in ("supported", "not supported"):
        status, note = "scored", verdict
        if verdict == "not supported" and not counted:
            note = "not supported (no scored provider passed both parities and was timed)"
        extra = facts.h5a_note(verdict)
        if extra:
            note = f"{note}. {extra}"
    else:
        status, note = verdict or "not scored: delay.json has no h5a_verdict", ""
    return {
        **base,
        "status": status,
        "verdict": verdict if status == "scored" else None,
        "verdict_note": note,
        "estimate": min(counted) if counted else math.nan,
        "ci_lo": None,
        "ci_hi": None,
        "units": "ms wall clock per call, batch 1, on this machine (RTX A4000, i9-10980XE, "
        "WSL2); lowest p99 among scored providers passing open- and closed-loop parity",
        "magnitude": H5A_THRESHOLD_MS,
        "statistic": "p99 of >= 2 000 timed batch-1 calls after >= 200 warm-up calls (P8's "
        "latency harness through ila.p8_adapter; P9-D1 item 8)",
        "contrast": f"p99(dp_e2e, batch 1) < {_period_text()}, on at least one scored "
        "provider that passes open- and closed-loop parity (P3-D1 §6 H5(a)); the p99 is wall "
        "clock on this machine and is not scaled",
        "n_seeds": [{"arm": "dp", "seeds": 1}],
        "inputs": sorted(sources),
        "sources": sources,
        "providers": providers,
        "delays": delay.get("arms", {}),
        "beside": [{"label": k, "text": v} for k, v in beside],
        "caveats": list(_H5A_CAVEATS),
        "ceiling_limited": False,
        "notes": [],
    }


# --------------------------------------------------------------------------- best end-to-end


_RESULT_PHASES: Final[tuple[str, ...]] = ("phase4", "phase5", "phase6")


@dataclass(frozen=True)
class SelectScore:
    """One arm's selected checkpoints' ``select`` scores (committed ``selection.json``).

    Attributes:
        arm: Arm.
        seeds: Seeds 0-4.
        success: Selected checkpoint's ``select`` successes per seed.
        n: ``select`` episodes per seed (150).
        rows_files: Candidate paths of each seed's selected-checkpoint ``select`` rows: the
            committed ``results/.../select/`` copy, then the run directory's.
    """

    arm: str
    seeds: tuple[int, ...]
    success: tuple[int, ...]
    n: tuple[int, ...]
    rows_files: tuple[tuple[Path, ...], ...]

    @property
    def mean_success(self) -> Fraction:
        """Mean over seeds of each seed's success rate, exact."""
        return sum(
            (Fraction(k, n) for k, n in zip(self.success, self.n, strict=True)), Fraction(0)
        ) / len(self.seeds)


def load_select_scores(
    arms: Sequence[str] = END_TO_END_ARMS,
    *,
    results_root: Path = p8_adapter.REPO_ROOT / "results",
) -> dict[str, SelectScore]:
    """Read every arm's selected ``select`` scores from committed ``selection.json`` files.

    Looks in ``results/phase{4,5,6}/runs/<arm>/s<seed>/selection.json``; the arm must live in
    exactly one phase and have seeds 0-4.

    Returns:
        ``arm -> SelectScore``.
    """
    out: dict[str, SelectScore] = {}
    for arm in arms:
        homes = [results_root / p / "runs" / arm for p in _RESULT_PHASES]
        homes = [h for h in homes if h.is_dir()]
        if len(homes) != 1:
            raise RowError(f"{arm}: selection results in {len(homes)} phases")
        succ, ns, files = [], [], []
        for s in CORE_SEEDS:
            path = homes[0] / f"s{s}" / "selection.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["arm"] != arm or int(rec["run_seed"]) != s:
                raise RowError(f"{path}: arm/seed mismatch")
            sel = rec["selected"]
            succ.append(int(sel["success"]))
            ns.append(int(sel["n"]))
            name = str(sel["rows_file"])
            files.append((path.parent / "select" / name, Path(rec["run_dir"]) / "select" / name))
        out[arm] = SelectScore(arm, CORE_SEEDS, tuple(succ), tuple(ns), tuple(files))
    return out


def pooled_select_p95(score: SelectScore) -> float:
    """Pooled ``select`` p95 closing speed over all seeds' touched-down episodes.

    P8's ``quantile_linear`` at 0.95 over ``closing_speed_normal_m_s`` (m/s model scale, x5
    full scale) of every seed's selected-checkpoint rows; NaN entries (no touchdown) excluded.

    Raises:
        FileNotFoundError: If a seed's rows are in neither location.
    """
    speeds: list[float] = []
    for candidates in score.rows_files:
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"select rows not found: {[str(p) for p in candidates]}")
        with path.open(encoding="utf-8", newline="") as f:
            speeds.extend(_time(r["closing_speed_normal_m_s"]) for r in csv.DictReader(f))
    return st.quantile(np.asarray(speeds, dtype=np.float64), 0.95)


def best_end_to_end(
    scores: Mapping[str, SelectScore],
    *,
    p95: Any = pooled_select_p95,
) -> tuple[str, str]:
    """The best end-to-end vision student (P3-D1 §6 H4), from ``select`` scores only.

    Highest mean ``select`` success of the selected checkpoints among
    :data:`END_TO_END_ARMS`; ties go to the lower pooled ``select`` p95 closing speed (computed
    only when needed), then to :data:`END_TO_END_ARMS` order.

    Args:
        scores: :func:`load_select_scores`.
        p95: ``SelectScore -> float`` pooled p95 (m/s model scale); injectable for tests.

    Returns:
        ``(arm, reason)``.
    """
    arms = [a for a in END_TO_END_ARMS if a in scores]
    if arms != list(END_TO_END_ARMS):
        raise RowError(f"need select scores for all of {END_TO_END_ARMS}")
    top = max(scores[a].mean_success for a in arms)
    tied = [a for a in arms if scores[a].mean_success == top]
    pct = f"{100.0 * float(top):.2f} %"
    if len(tied) == 1:
        return tied[0], f"highest mean select success ({pct})"
    speeds = {a: float(p95(scores[a])) for a in tied}
    low = min(speeds.values())
    tied2 = [a for a in tied if speeds[a] == low]
    if len(tied2) == 1:
        return tied2[0], (
            f"tied at {pct} mean select success with "
            f"{', '.join(a for a in tied if a != tied2[0])}; "
            f"lower pooled select p95 closing speed ({low:.4f} m/s model scale)"
        )
    return tied2[0], (
        f"tied at {pct} mean select success and pooled p95 with "
        f"{', '.join(tied2[1:])}; first in P3-D1 §6 order"
    )
