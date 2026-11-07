"""Scoring of H1-H4 exactly as frozen in P3-D1 §6 (Phase 8); H5 is Phase 9's.

Everything here is simulation only (PyBullet physics, PyBullet TinyRenderer frames). Success is
dimensionless in ``[0, 1]``; "points" are hundredths of it. Times are seconds, model scale
(lambda = 1/25; x5 for full scale); one 30 Hz control step is 33.3 ms model (167 ms full).

Inputs
------
Episode rows (CSV-read mappings, string or typed cells) become :class:`ArmMatrix` objects via
:func:`student_matrix` (this project's rows: ``arm``, ``student_seed``, ``Ta``, ``K``,
``ensemble``, ``latency_steps``, ``render_cfg_sha256``, ``ckpt_sha256``) or :func:`p8_matrix`
(P8's committed rows: ``method``, ``run_seed``). Both select one cell (``regime``, ``ss``),
refuse a missing or extra seed (no seed is dropped, P3-D1 §4), refuse rows of the latency arm
or of a non-committed render (appearance shift, ``camera``), and require every seed to cover
the same episodes exactly once. Contrasts then require identical episodes on both sides.

Scorers and their rules (P3-D1 §6, verbatim in substance)
----------------------------------------------------------
* :func:`score_h1`: ``id`` SS5 ``bc_dagger - bc_matched`` (primary), ``id`` SS6 (secondary),
  ``bc_dart - bc`` at SS5 and SS6 (DART secondaries); P8 ``paired_bootstrap``; +5 points.
  Supported: estimate >= +5 points and the CI excludes 0. Inconclusive: the CI excludes 0 and
  the estimate is positive but below +5. Not supported: every other case; a CI wholly below 0
  is "an effect in the opposite direction". A null whose reference arm (``bc_matched`` /
  ``bc``) has a point estimate above 95 % in that cell is printed "not supported
  (ceiling-limited: the largest attainable gain is below +5 points)", with the SS6 secondary
  beside it. ``paired_iqm_difference`` is printed beside as "post hoc, not scored";
  labelled-frame counts are printed beside.
* :func:`score_h2`: ``id`` SS4, ``bc_mix`` and ``dp_mix``, 3 seeds each; D = W1(bc_mix, ref) -
  W1(dp_mix, ref) (:func:`ila.eval.stats.w1_bootstrap`); supported: D >= 0.033 s and the CI
  excludes 0; inconclusive: the CI excludes 0 and 0 < D < 0.033 s; not supported otherwise.
  The reference comes from :func:`build_h2_reference` (P8's committed rows, nothing flown).
* :func:`score_h3`: ``id`` SS6, ``dp`` ``Ta = 1`` minus ``Ta = 8`` on the same checkpoints
  (:func:`ila.eval.stats.joint_seed_contrast`); H1's rule at +5 points; ceiling when ``dp`` at
  ``Ta = 8`` is above 95 %.
* :func:`score_h4`: ``id`` SS6; D = P - I (:func:`ila.eval.stats.h4_three_arm_bootstrap`);
  supported: D > 0 and the CI excludes 0; not supported otherwise, a CI wholly below 0 printed
  as "imitation gap larger".
* Unscored, two-sided (:class:`UnscoredContrast`): :func:`h4_p_prime`, :func:`modular_vs_best`
  (the best end-to-end student from committed ``select`` scores, :func:`best_end_to_end`) and
  :func:`dart_vs_matched`.

"Separates" means the 95 % CI excludes 0; magnitudes are judged on the point estimate; there is
no multiplicity correction (P3-D1 §1). Every SS6 contrast carries :data:`FLIP_FLOOR` beside it.
"""

from __future__ import annotations

import csv
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
    "CORE_SEEDS",
    "ABLATION_SEEDS",
    "END_TO_END_ARMS",
    "FLIP_FLOOR",
    "H2_MAGNITUDE_S",
    "MAGNITUDE_TOL",
    "NO_MULTIPLICITY",
    "SUCCESS_MAGNITUDE",
    "ArmMatrix",
    "H2Reference",
    "HypothesisResult",
    "RowError",
    "SelectScore",
    "UnscoredContrast",
    "Verdict",
    "best_end_to_end",
    "build_h2_reference",
    "dart_vs_matched",
    "h4_p_prime",
    "load_select_scores",
    "magnitude_rule",
    "modular_vs_best",
    "p8_committed_rows",
    "p8_matrix",
    "pooled_select_p95",
    "score_h1",
    "score_h2",
    "score_h3",
    "score_h4",
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
) -> ArmMatrix:
    if sorted(by_seed) != sorted(expected_seeds):
        raise RowError(
            f"{arm} {cell}: seeds {sorted(by_seed)} != expected {sorted(expected_seeds)} "
            "(no seed is dropped or added, P3-D1 §4)"
        )
    seeds = tuple(sorted(by_seed))
    episodes: tuple[tuple[int, int], ...] | None = None
    succ, td, outs, ckpts = [], [], [], []
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
) -> ArmMatrix:
    """Select one student arm's rows in one cell and stack them as seeds x episodes.

    Rows are selected on ``arm``, ``regime``/``ss``, ``Ta``, ``K`` and ``ensemble`` (``None``
    means the cell must be empty) and ``latency_steps == 0``. Selected rows must carry the
    committed render (``render_cfg_sha256`` of ``full`` or ``off``): an appearance-shift or
    ``camera`` row is refused, never mixed into a core section.

    Args:
        rows: This project's episode rows (P8 columns, then plan §1.7's columns).
        arm: The arm label.
        cell: ``(regime, ss)``, e.g. ``("id", "SS5")``.
        seeds: The seeds that must all be present (5 core, 3 ablation).
        ta: ``Ta``, executed actions per chunk, or ``None``.
        k: ``K``, DDIM steps, or ``None``.
        ensemble: Temporal ensembling flag, or ``None``.

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
        if (_opt_int(r["latency_steps"]) or 0) != 0:
            continue
        if str(r["render_cfg_sha256"]) not in allowed:
            raise RowError(f"{arm} {cell}: non-core render {r['render_cfg_sha256']!r}")
        seed = _opt_int(r["student_seed"])
        if seed is None:
            raise RowError(f"{arm} {cell}: a student row without student_seed")
        by_seed.setdefault(seed, []).append(r)
    return _assemble(arm, cell, by_seed, seeds, ckpt_column="ckpt_sha256", tags=(ta, k, ensemble))


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
    if ceiling and verdict == "not supported":
        if note == _OPPOSITE:
            extra = (
                f"{b.arm} is above 95 % in {b.cell_label}: ceiling-limited (the largest "
                "attainable gain is below +5 points).",
            )
        else:
            note = _CEILING_NOTE
    elif ceiling:
        extra = (f"{b.arm} is above 95 % in {b.cell_label} (the ceiling of P3-D1 §6).",)
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
    )


def _with_ss6_beside(primary: HypothesisResult, ss6: HypothesisResult) -> HypothesisResult:
    if primary.verdict_note != _CEILING_NOTE:
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


_H3_CAVEATS: Final[tuple[str, ...]] = (
    "Checkpoints were selected at Ta = 8 (P3-D1 §6 H3).",
    "Ta = 1 without ensembling can switch modes between replans; so a null cannot separate "
    "'chunking is harmless' from 'replanning causes mode switching' (P3-D1 §6 H3).",
    "dp_minus_last_action (P7-D3 item 7) is a competing explanation for dp recovering at "
    "Ta = 1: an input-freshness effect, stale last_action, alongside chunk length; any sentence "
    "that says dp fails 'because of chunking' must name it (P7-D3, Phase 8 before-you-start).",
)


def score_h3(ta1: ArmMatrix, ta8: ArmMatrix) -> HypothesisResult:
    """Score H3 (P3-D1 §6 H3): ``dp`` success at ``Ta = 1`` minus ``Ta = 8``, ``id`` SS6.

    Same-checkpoint contrast (:func:`ila.eval.stats.joint_seed_contrast`); H1's rule at +5
    points; ceiling-limited when ``dp`` at ``Ta = 8`` is above 95 %.

    Args:
        ta1: ``dp`` at ``Ta = 1``, K = 10, no ensembling, ``id`` SS6, seeds 0-4.
        ta8: ``dp`` at ``Ta = 8``, K = 10, no ensembling, same cell and seeds.

    Returns:
        The H3 :class:`HypothesisResult`.

    Raises:
        RowError: If the sides are not the same ``dp`` checkpoints at the stated settings.
    """
    for m, ta in ((ta1, 1), (ta8, 8)):
        if (m.arm, m.cell, m.Ta, m.K, m.ensemble) != ("dp", ("id", "SS6"), ta, 10, False):
            raise RowError(
                f"H3 needs dp at id SS6, Ta={ta}, K=10, no ensembling; got {m.arm} "
                f"{m.cell_label} Ta={m.Ta} K={m.K} ensemble={m.ensemble}"
            )
        _require_seeds(m, CORE_SEEDS)
    _same_episodes(ta1, ta8)
    if ta1.ckpt_sha256 != ta8.ckpt_sha256:
        raise RowError("H3 compares the same checkpoints; ckpt_sha256 differs per seed")
    c = st.joint_seed_contrast(ta1.success, ta8.success)
    verdict, note = magnitude_rule(c.diff, c.lo, c.hi, SUCCESS_MAGNITUDE, unit="+5 points")
    extra: tuple[str, ...] = ()
    if ta8.rate > CEILING and verdict == "not supported":
        if note == _OPPOSITE:
            extra = (
                "dp at Ta = 8 is above 95 % at id SS6: ceiling-limited (the largest attainable "
                "gain is below +5 points).",
            )
        else:
            note = _CEILING_NOTE
    elif ta8.rate > CEILING:
        extra = ("dp at Ta = 8 is above 95 % at id SS6 (the ceiling of P3-D1 §6).",)
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
        caveats=(*_H3_CAVEATS, *extra, NO_MULTIPLICITY),
        beside=(
            ("dp at Ta = 8 (mean over seeds x episodes)", f"{100.0 * ta8.rate:.1f} %"),
            *_flip(ta1),
        ),
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
