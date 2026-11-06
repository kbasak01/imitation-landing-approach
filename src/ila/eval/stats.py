"""Phase 8 statistics: P8's functions with the frozen seed and replicates, and three that P8 lacks.

Everything here is simulation only (PyBullet physics). Success rates are dimensionless in
``[0, 1]`` ("points" are hundredths of that scale); touchdown times are seconds, model scale
(lambda = 1/25; x5 for full scale); closing speeds m/s model scale (x5 for full scale).

P8's statistics (P3-D1 §1)
--------------------------
:func:`wilson`, :func:`iqm_ci`, :func:`aggregate`, :func:`paired_contrast`,
:func:`paired_iqm_contrast`, :func:`relative_p95` and :func:`quantile` call P8's own objects
through :data:`ila.p8_adapter.stats` (``rld.eval.stats``), nothing re-implemented. Each passes
the bootstrap seed and the replicate count **explicitly**, read from the SHA-frozen
``configs/eval/matrix.yaml`` (:func:`matrix_statistics`): seed 20261023, 2 000 replicates for
IQM intervals, 10 000 for paired contrasts. P8's default seed (20260926) is never relied on.

The three statistics that are not P8's (P3-D1 §1, the only exceptions to "P8 functions only")
---------------------------------------------------------------------------------------------
* :func:`joint_seed_contrast` -- the same-checkpoint contrast (H3: ``Ta = 1`` vs ``Ta = 8``;
  later H5(b): latency arm vs synchronous). P8's :func:`paired_bootstrap` design with one
  change: seed indices are drawn **once** per replicate and used for both sides, because both
  sides are the same trained runs.
* :func:`w1_bootstrap` -- H2's ``D = W1(bc_mix, ref) - W1(dp_mix, ref)`` over touched-down
  ``time_to_touchdown_s``, with the exact 1-D W1 of :func:`ila.eval.metrics.wasserstein_1`.
* :func:`h4_three_arm_bootstrap` -- H4's perception gap, imitation gap and their difference,
  with ``bc_state``'s seeds drawn once per replicate and used in both gaps.

Each has a known-answer unit test in ``tests/test_eval_stats.py``. Percentile intervals follow
P8: ``np.percentile`` (linear) at 2.5 / 97.5 % when every replicate is defined (as P8's
``paired_bootstrap``), and P8's ``quantile_linear`` over the defined replicates, NaNs excluded
and counted, when some can be undefined (as P8's ``paired_relative_p95``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import yaml

from ila import p8_adapter
from ila.eval.metrics import wasserstein_1

__all__ = [
    "CONFIDENCE",
    "FROZEN_BOOTSTRAP_SEED",
    "MATRIX_SHA256",
    "MATRIX_YAML",
    "H4Gaps",
    "MatrixStatistics",
    "W1Contrast",
    "aggregate",
    "h4_replicates",
    "h4_three_arm_bootstrap",
    "iqm_ci",
    "joint_seed_contrast",
    "matrix_statistics",
    "paired_contrast",
    "paired_iqm_contrast",
    "quantile",
    "relative_p95",
    "w1_bootstrap",
    "wilson",
]

type FloatArray = npt.NDArray[np.float64]

#: The evaluation matrix (P3-D1 §1) and its frozen SHA-256.
MATRIX_YAML: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "eval" / "matrix.yaml"
MATRIX_SHA256: Final[str] = "4d8a78fdfc59ddfb5cc51add8bba121af8da43ee0e35787ffd0e2611d7e1e96a"

#: P3-D1 §1's bootstrap seed; ``matrix.yaml`` must agree.
FROZEN_BOOTSTRAP_SEED: Final[int] = 20261023

#: Two-sided coverage of every interval (P3-D1 §1: 95 %).
CONFIDENCE: Final[float] = 0.95

#: Replicates per vectorised batch (memory bound only; results do not depend on it).
_BATCH: Final[int] = 250


@dataclass(frozen=True)
class MatrixStatistics:
    """The statistics block of ``configs/eval/matrix.yaml``.

    Attributes:
        bootstrap_seed: Seed of every bootstrap generator (20261023).
        iqm_reps: Stratified-bootstrap replicates for IQM intervals (2 000).
        paired_reps: Replicates for every contrast (10 000).
    """

    bootstrap_seed: int
    iqm_reps: int
    paired_reps: int


@functools.cache
def matrix_statistics(path: Path = MATRIX_YAML) -> MatrixStatistics:
    """Read the bootstrap seed and replicate counts from the frozen evaluation matrix.

    Args:
        path: ``configs/eval/matrix.yaml``.

    Returns:
        The :class:`MatrixStatistics`.

    Raises:
        ValueError: If the file's SHA-256 is not P3-D1's or its seed is not 20261023.
    """
    sha = p8_adapter.sha256_file(path)
    if path == MATRIX_YAML and sha != MATRIX_SHA256:
        raise ValueError(f"{path}: SHA-256 {sha} != P3-D1's {MATRIX_SHA256}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = MatrixStatistics(
        bootstrap_seed=int(raw["bootstrap_seed"]),
        iqm_reps=int(raw["replicates"]["iqm_ci"]),
        paired_reps=int(raw["replicates"]["paired"]),
    )
    if out.bootstrap_seed != FROZEN_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap seed {out.bootstrap_seed} != {FROZEN_BOOTSTRAP_SEED}")
    return out


def _seed(seed: int | None) -> int:
    return matrix_statistics().bootstrap_seed if seed is None else int(seed)


def _paired_reps(reps: int | None) -> int:
    return matrix_statistics().paired_reps if reps is None else int(reps)


# --------------------------------------------------------------------------- P8's statistics


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95 % interval of ``k`` successes in ``n`` episodes (P8 ``wilson_interval``).

    Returns:
        ``(lo, hi)``, dimensionless; ``(nan, nan)`` for ``n == 0``.
    """
    lo, hi = p8_adapter.stats.wilson_interval(int(k), int(n), p8_adapter.stats.WILSON_Z_95)
    return float(lo), float(hi)


def iqm_ci(scores: FloatArray) -> Any:
    """IQM over seeds x cells with its stratified-bootstrap 95 % CI (P8).

    Args:
        scores: ``(n_seeds, n_cells)`` success rates, dimensionless.

    Returns:
        P8's ``IntervalEstimate`` (2 000 replicates, seed 20261023 from ``matrix.yaml``).
    """
    cfg = matrix_statistics()
    return p8_adapter.stats.stratified_bootstrap_ci(
        scores,
        p8_adapter.stats.iqm,
        reps=cfg.iqm_reps,
        confidence=CONFIDENCE,
        seed=cfg.bootstrap_seed,
    )


def aggregate(scores: FloatArray) -> dict[str, Any]:
    """IQM and optimality gap, each with its stratified-bootstrap 95 % CI (P8).

    Args:
        scores: ``(n_seeds, n_cells)`` success rates, dimensionless.

    Returns:
        P8's ``{"iqm": IntervalEstimate, "optimality_gap": IntervalEstimate}``.
    """
    cfg = matrix_statistics()
    out: dict[str, Any] = p8_adapter.stats.aggregate_estimates(
        scores, reps=cfg.iqm_reps, confidence=CONFIDENCE, seed=cfg.bootstrap_seed
    )
    return out


def paired_contrast(a: FloatArray, b: FloatArray) -> Any:
    """``mean(a) - mean(b)`` over identical episodes, seeds resampled per arm (P8).

    P8's ``paired_bootstrap``: 10 000 replicates, seed 20261023, one shared set of episode
    indices per replicate. Used for H1 and its secondaries (P3-D1 §1).

    Args:
        a: ``(n_seeds_a, n_episodes)`` 0/1 success (or a continuous per-episode metric).
        b: ``(n_seeds_b, n_episodes)``, the same episodes in the same column order.

    Returns:
        P8's ``PairedContrast``; ``diff`` in the outcome's units.
    """
    cfg = matrix_statistics()
    return p8_adapter.stats.paired_bootstrap(
        a, b, reps=cfg.paired_reps, confidence=CONFIDENCE, seed=cfg.bootstrap_seed
    )


def paired_iqm_contrast(a: FloatArray, b: FloatArray) -> Any:
    """IQM-over-seeds success of ``a`` minus that of ``b`` (P8 ``paired_iqm_difference``).

    Printed beside H1 as "post hoc, not scored" (P3-D1 §1).

    Args:
        a: ``(n_seeds_a, n_episodes)`` 0/1 success.
        b: ``(n_seeds_b, n_episodes)``, same episodes.

    Returns:
        P8's ``PairedContrast``.
    """
    cfg = matrix_statistics()
    return p8_adapter.stats.paired_iqm_difference(
        a, b, reps=cfg.paired_reps, confidence=CONFIDENCE, seed=cfg.bootstrap_seed
    )


def relative_p95(
    a_speeds: FloatArray,
    b_speeds: FloatArray,
    *,
    a_timeouts: FloatArray | None = None,
    b_timeouts: FloatArray | None = None,
    timeouts_as_worst: bool = False,
) -> Any:
    """Paired relative p95 of closing speed, touched-down episodes only (P8).

    Args:
        a_speeds: ``(n_seeds_a, n_episodes)`` closing speeds along the deck normal, m/s model
            scale; NaN where there was no touchdown (never imputed).
        b_speeds: ``(n_seeds_b, n_episodes)``, same episodes.
        a_timeouts: ``a``'s timeout mask (sensitivity variant only).
        b_timeouts: ``b``'s timeout mask.
        timeouts_as_worst: P8's sensitivity variant.

    Returns:
        P8's ``RelativeP95``.
    """
    cfg = matrix_statistics()
    return p8_adapter.stats.paired_relative_p95(
        a_speeds,
        b_speeds,
        a_timeouts=a_timeouts,
        b_timeouts=b_timeouts,
        timeouts_as_worst=timeouts_as_worst,
        reps=cfg.paired_reps,
        confidence=CONFIDENCE,
        seed=cfg.bootstrap_seed,
    )


def quantile(values: FloatArray, q: float) -> float:
    """P8's ``quantile_linear`` (type 7) of a 1-D array over its non-NaN entries.

    Returns:
        The quantile in the values' units; NaN if no entry is valid.
    """
    return float(p8_adapter.stats.quantile_linear(np.asarray(values, dtype=np.float64), q))


# --------------------------------------------------------------------------- helpers


def _matrix(name: str, values: Any, *, finite: bool) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be (n_seeds >= 1, n_episodes >= 1), got {arr.shape}")
    if finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite: a missing seed or episode is not a zero")
    if not finite and np.any(np.isinf(arr)):
        raise ValueError(f"{name}: a touchdown time is never infinite (NaN = no touchdown)")
    return arr


def _check_reps(reps: int) -> None:
    if reps < 1:
        raise ValueError(f"need reps >= 1, got {reps}")


def _percentile(values: FloatArray) -> tuple[float, float]:
    """P8 ``paired_bootstrap``'s interval: ``np.percentile`` at 2.5 / 97.5 % (all defined)."""
    tail = 100.0 * (1.0 - CONFIDENCE) / 2.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def _percentile_nan(values: FloatArray) -> tuple[float, float, int]:
    """P8 ``_percentile_interval``: ``quantile_linear`` over defined replicates; NaNs counted."""
    tail = (1.0 - CONFIDENCE) / 2.0
    q = p8_adapter.stats.quantile_linear
    return float(q(values, tail)), float(q(values, 1.0 - tail)), int(np.sum(np.isnan(values)))


# --------------------------------------------------------------------------- joint-seed contrast


def joint_seed_contrast(
    a: FloatArray,
    b: FloatArray,
    *,
    reps: int | None = None,
    seed: int | None = None,
) -> Any:
    """Same-checkpoint contrast ``mean(a) - mean(b)``, seeds resampled jointly (P3-D1 §1).

    Mirrors P8's ``paired_bootstrap`` (mean over seeds x episodes; one set of episode indices
    per replicate, shared by both sides; percentile interval; point estimate on the
    unresampled data) except that row ``i`` of ``a`` and of ``b`` are the **same trained
    run**, so each replicate draws one set of ``n_seeds`` seed indices and uses it for both
    sides. Draw order per batch of 250 replicates, from one generator seeded with ``seed``:
    seeds ``(batch, n_seeds)``, then episodes ``(batch, n_episodes)``.

    Args:
        a: ``(n_seeds, n_episodes)`` per-episode outcomes of side ``a`` (0/1 success, or a
            continuous metric), e.g. ``dp`` at ``Ta = 1``.
        b: ``(n_seeds, n_episodes)`` of side ``b`` (e.g. ``Ta = 8``), the same runs in the same
            row order and the same episodes in the same column order.
        reps: Replicates; ``None`` reads ``matrix.yaml`` (10 000).
        seed: Generator seed; ``None`` reads ``matrix.yaml`` (20261023).

    Returns:
        P8's ``PairedContrast`` with ``n_seeds_a == n_seeds_b == n_seeds``; ``diff`` in the
        outcome's units (a difference of success rates, dimensionless, for 0/1 outcomes).

    Raises:
        ValueError: If shapes differ, a value is not finite or ``reps < 1``.
    """
    arr_a = _matrix("a", a, finite=True)
    arr_b = _matrix("b", b, finite=True)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"same runs on the same episodes needed, got {arr_a.shape}, {arr_b.shape}")
    n_reps = _paired_reps(reps)
    _check_reps(n_reps)
    n_s, n_e = arr_a.shape
    per_run = arr_a - arr_b  # (n_s, n_e): the paired difference of each run on each episode
    rng = np.random.default_rng(_seed(seed))
    values = np.empty(n_reps, dtype=np.float64)
    for start in range(0, n_reps, _BATCH):
        size = min(_BATCH, n_reps - start)
        seeds = rng.integers(0, n_s, size=(size, n_s))
        episodes = rng.integers(0, n_e, size=(size, n_e))
        per_episode = per_run[seeds].mean(axis=1)  # (size, n_e)
        values[start : start + size] = np.take_along_axis(per_episode, episodes, axis=1).mean(
            axis=1
        )
    lo, hi = _percentile(values)
    return p8_adapter.stats.PairedContrast(
        diff=float(arr_a.mean() - arr_b.mean()),
        lo=lo,
        hi=hi,
        reps=n_reps,
        confidence=CONFIDENCE,
        n_episodes=n_e,
        n_seeds_a=n_s,
        n_seeds_b=n_s,
    )


# --------------------------------------------------------------------------- H2: W1 bootstrap


@dataclass(frozen=True)
class W1Contrast:
    """H2's ``D = W1(bc_mix, ref) - W1(dp_mix, ref)`` with its bootstrap interval.

    All times are touched-down ``time_to_touchdown_s``, seconds model scale (lambda = 1/25;
    x5 for full scale). Positive ``d`` means ``dp_mix`` is closer to the reference.

    Attributes:
        d: Point estimate on the unresampled pools, s; NaN if a pool is empty.
        w1_bc: ``W1(bc_mix pooled, ref)``, s, unresampled.
        w1_dp: ``W1(dp_mix pooled, ref)``, s, unresampled.
        lo: Lower 95 % percentile bound over defined replicates, s.
        hi: Upper bound, s.
        reps: Replicates drawn (defined + undefined).
        confidence: Two-sided coverage.
        n_episodes: Paired episodes (the list's cell).
        n_seeds_bc: ``bc_mix`` seeds.
        n_seeds_dp: ``dp_mix`` seeds.
        n_touchdowns_bc: Touched-down (seed, episode) pairs of ``bc_mix``.
        n_touchdowns_dp: The same for ``dp_mix``.
        n_touchdowns_ref: Touched-down reference episodes.
        frac_without_touchdown_bc: ``1 - n_touchdowns_bc / (n_seeds_bc * n_episodes)``.
        frac_without_touchdown_dp: The same for ``dp_mix``.
        frac_without_touchdown_ref: The same for the reference.
        n_reps_undefined: Replicates with an empty pool (NaN, excluded from the bounds).
    """

    d: float
    w1_bc: float
    w1_dp: float
    lo: float
    hi: float
    reps: int
    confidence: float
    n_episodes: int
    n_seeds_bc: int
    n_seeds_dp: int
    n_touchdowns_bc: int
    n_touchdowns_dp: int
    n_touchdowns_ref: int
    frac_without_touchdown_bc: float
    frac_without_touchdown_dp: float
    frac_without_touchdown_ref: float
    n_reps_undefined: int

    @property
    def separates(self) -> bool:
        """Whether the 95 % CI excludes 0 (False when the CI is undefined)."""
        return self.lo > 0.0 or self.hi < 0.0


def _pool(arr: FloatArray) -> FloatArray:
    flat = arr.reshape(-1)
    return flat[np.isfinite(flat)]


def _w1_or_nan(x: FloatArray, ref: FloatArray) -> float:
    return wasserstein_1(x, ref) if x.size and ref.size else float("nan")


def w1_bootstrap(
    bc_times: FloatArray,
    dp_times: FloatArray,
    ref_times: FloatArray,
    *,
    reps: int | None = None,
    seed: int | None = None,
) -> W1Contrast:
    """H2's W1 difference with its bootstrap interval (P3-D1 §1, §6 H2).

    ``D = W1(bc pool, ref pool) - W1(dp pool, ref pool)``, where each pool is the pooled
    touched-down ``time_to_touchdown_s`` of all seeds x episodes (the reference: its single
    row set) and W1 is the exact 1-D empirical distance
    (:func:`ila.eval.metrics.wasserstein_1`). Per replicate, from one generator seeded with
    ``seed``: ``n_seeds_bc`` ``bc`` seed indices, then ``n_seeds_dp`` ``dp`` seed indices
    (independently: different trained runs), then **one** set of ``n_episodes`` episode
    indices shared by both arms and the reference. A replicate with an empty pool is NaN,
    excluded from the interval and counted. The point estimate is on the unresampled pools.

    Args:
        bc_times: ``(n_seeds_bc, n_episodes)`` touchdown times, s model scale; NaN where the
            episode has no touchdown.
        dp_times: ``(n_seeds_dp, n_episodes)``, same episodes in the same column order.
        ref_times: ``(n_episodes,)`` (or ``(1, n_episodes)``) the reference's touchdown times,
            same episodes; NaN where it has no touchdown.
        reps: Replicates; ``None`` reads ``matrix.yaml``'s paired count (10 000).
        seed: Generator seed; ``None`` reads ``matrix.yaml`` (20261023).

    Returns:
        The :class:`W1Contrast`, seconds model scale.

    Raises:
        ValueError: On malformed input (shape, infinite values) or ``reps < 1``.
    """
    bc = _matrix("bc_times", bc_times, finite=False)
    dp = _matrix("dp_times", dp_times, finite=False)
    ref = _matrix("ref_times", ref_times, finite=False)
    if ref.shape[0] != 1:
        raise ValueError(f"the reference is one row set, got {ref.shape[0]} rows")
    n_e = bc.shape[1]
    if dp.shape[1] != n_e or ref.shape[1] != n_e:
        raise ValueError(f"same episodes needed, got {bc.shape[1]}, {dp.shape[1]}, {ref.shape[1]}")
    n_reps = _paired_reps(reps)
    _check_reps(n_reps)
    n_bc, n_dp = bc.shape[0], dp.shape[0]
    ref_row = ref[0]
    pool_bc, pool_dp, pool_ref = _pool(bc), _pool(dp), _pool(ref)
    w1_bc = _w1_or_nan(pool_bc, pool_ref)
    w1_dp = _w1_or_nan(pool_dp, pool_ref)
    rng = np.random.default_rng(_seed(seed))
    values = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        seeds_bc = np.asarray(rng.integers(0, n_bc, size=n_bc), dtype=np.intp)
        seeds_dp = np.asarray(rng.integers(0, n_dp, size=n_dp), dtype=np.intp)
        episodes = np.asarray(rng.integers(0, n_e, size=n_e), dtype=np.intp)
        r_ref = _pool(ref_row[episodes])
        r_bc = _pool(bc[seeds_bc[:, None], episodes[None, :]])
        r_dp = _pool(dp[seeds_dp[:, None], episodes[None, :]])
        if r_ref.size == 0 or r_bc.size == 0 or r_dp.size == 0:
            values[rep] = math.nan
        else:
            values[rep] = wasserstein_1(r_bc, r_ref) - wasserstein_1(r_dp, r_ref)
    lo, hi, undefined = _percentile_nan(values)
    return W1Contrast(
        d=w1_bc - w1_dp,
        w1_bc=w1_bc,
        w1_dp=w1_dp,
        lo=lo,
        hi=hi,
        reps=n_reps,
        confidence=CONFIDENCE,
        n_episodes=n_e,
        n_seeds_bc=n_bc,
        n_seeds_dp=n_dp,
        n_touchdowns_bc=int(pool_bc.size),
        n_touchdowns_dp=int(pool_dp.size),
        n_touchdowns_ref=int(pool_ref.size),
        frac_without_touchdown_bc=1.0 - pool_bc.size / bc.size,
        frac_without_touchdown_dp=1.0 - pool_dp.size / dp.size,
        frac_without_touchdown_ref=1.0 - pool_ref.size / ref.size,
        n_reps_undefined=undefined,
    )


# --------------------------------------------------------------------------- H4: three arms


@dataclass(frozen=True)
class H4Gaps:
    """H4's gaps with their 95 % bootstrap intervals (all dimensionless success differences).

    ``p = S(bc_state) - S(bc)`` (perception gap), ``i = S(teacher) - S(bc_state)``
    (imitation gap), ``d = p - i``. Points are on the unresampled data.

    Attributes:
        p: Perception gap point estimate.
        p_lo: Its lower bound.
        p_hi: Its upper bound.
        i: Imitation gap point estimate.
        i_lo: Its lower bound.
        i_hi: Its upper bound.
        d: ``p - i`` point estimate.
        d_lo: Its lower bound.
        d_hi: Its upper bound.
        s_bc_state: Mean success of ``bc_state`` over seeds x episodes.
        s_bc: Mean success of ``bc``.
        s_teacher: Mean success of the teacher's single row set.
        reps: Replicates.
        confidence: Two-sided coverage.
        n_episodes: Paired episodes.
        n_seeds_bc_state: ``bc_state`` seeds.
        n_seeds_bc: ``bc`` seeds.
    """

    p: float
    p_lo: float
    p_hi: float
    i: float
    i_lo: float
    i_hi: float
    d: float
    d_lo: float
    d_hi: float
    s_bc_state: float
    s_bc: float
    s_teacher: float
    reps: int
    confidence: float
    n_episodes: int
    n_seeds_bc_state: int
    n_seeds_bc: int


def h4_replicates(
    bc_state: FloatArray,
    bc: FloatArray,
    teacher: FloatArray,
    *,
    reps: int,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The bootstrap replicates of H4's ``(P, I, D)`` (see :func:`h4_three_arm_bootstrap`).

    Returns:
        Three ``(reps,)`` arrays: P, I and D = P - I, dimensionless.
    """
    bs = _matrix("bc_state", bc_state, finite=True)
    b = _matrix("bc", bc, finite=True)
    t = _matrix("teacher", teacher, finite=True)
    if t.shape[0] != 1:
        raise ValueError(f"the teacher is one row set (seed 0), got {t.shape[0]} rows")
    n_e = bs.shape[1]
    if b.shape[1] != n_e or t.shape[1] != n_e:
        raise ValueError(f"same episodes needed, got {bs.shape[1]}, {b.shape[1]}, {t.shape[1]}")
    _check_reps(reps)
    n_bs, n_b = bs.shape[0], b.shape[0]
    rng = np.random.default_rng(seed)
    p = np.empty(reps, dtype=np.float64)
    i = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, _BATCH):
        size = min(_BATCH, reps - start)
        seeds_bs = rng.integers(0, n_bs, size=(size, n_bs))
        seeds_b = rng.integers(0, n_b, size=(size, n_b))
        episodes = rng.integers(0, n_e, size=(size, n_e))
        s_bs = np.take_along_axis(bs[seeds_bs].mean(axis=1), episodes, axis=1).mean(axis=1)
        s_b = np.take_along_axis(b[seeds_b].mean(axis=1), episodes, axis=1).mean(axis=1)
        s_t = t[0][episodes].mean(axis=1)
        p[start : start + size] = s_bs - s_b
        i[start : start + size] = s_t - s_bs
    return p, i, p - i


def h4_three_arm_bootstrap(
    bc_state: FloatArray,
    bc: FloatArray,
    teacher: FloatArray,
    *,
    reps: int | None = None,
    seed: int | None = None,
) -> H4Gaps:
    """H4's three-arm bootstrap (P3-D1 §1, §6 H4).

    Per replicate, from one generator seeded with ``seed`` (draw order per batch of 250):
    ``bc_state``'s seed indices, drawn **once** and used in both P and I; ``bc``'s seed
    indices, independently; then one set of episode indices shared by all three arms. The
    teacher's single row set is held fixed (only its episodes are resampled). S is mean
    success over the resampled seeds x episodes; P = S(bc_state) - S(bc),
    I = S(teacher) - S(bc_state), D = P - I. Percentile intervals as P8's
    ``paired_bootstrap``.

    Args:
        bc_state: ``(n_seeds_bc_state, n_episodes)`` 0/1 success.
        bc: ``(n_seeds_bc, n_episodes)``, same episodes in the same column order.
        teacher: ``(n_episodes,)`` or ``(1, n_episodes)``: P8 ``ppo`` s0's committed rows.
        reps: Replicates; ``None`` reads ``matrix.yaml``'s paired count (10 000).
        seed: Generator seed; ``None`` reads ``matrix.yaml`` (20261023).

    Returns:
        The :class:`H4Gaps`.
    """
    n_reps = _paired_reps(reps)
    p, i, d = h4_replicates(bc_state, bc, teacher, reps=n_reps, seed=_seed(seed))
    bs = _matrix("bc_state", bc_state, finite=True)
    b = _matrix("bc", bc, finite=True)
    t = _matrix("teacher", teacher, finite=True)
    s_bs, s_b, s_t = float(bs.mean()), float(b.mean()), float(t.mean())
    p_lo, p_hi = _percentile(p)
    i_lo, i_hi = _percentile(i)
    d_lo, d_hi = _percentile(d)
    p_pt, i_pt = s_bs - s_b, s_t - s_bs
    return H4Gaps(
        p=p_pt,
        p_lo=p_lo,
        p_hi=p_hi,
        i=i_pt,
        i_lo=i_lo,
        i_hi=i_hi,
        d=p_pt - i_pt,
        d_lo=d_lo,
        d_hi=d_hi,
        s_bc_state=s_bs,
        s_bc=s_b,
        s_teacher=s_t,
        reps=n_reps,
        confidence=CONFIDENCE,
        n_episodes=bs.shape[1],
        n_seeds_bc_state=bs.shape[0],
        n_seeds_bc=b.shape[0],
    )
