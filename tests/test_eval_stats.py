"""Phase 8 statistics (``ila.eval.stats``): P8 wrappers and the three non-P8 statistics.

P3-D1 §1 requires a known-answer unit test for each statistic that is not P8's: the
joint-seed contrast, H2's W1 bootstrap and H4's three-arm bootstrap. Every array here is
synthetic; nothing is flown and no frozen list or student row is read. Times are seconds,
model scale (lambda = 1/25).
"""

from __future__ import annotations

import numpy as np
import pytest

from ila import p8_adapter
from ila.eval import stats
from ila.eval.metrics import wasserstein_1

SEED = 20261023


# --------------------------------------------------------------------------- P8 wrappers


def test_matrix_statistics_frozen() -> None:
    cfg = stats.matrix_statistics()
    assert cfg == stats.MatrixStatistics(bootstrap_seed=SEED, iqm_reps=2000, paired_reps=10000)
    assert p8_adapter.sha256_file(stats.MATRIX_YAML) == stats.MATRIX_SHA256


def test_wrappers_pass_seed_and_reps_explicitly() -> None:
    rng = np.random.default_rng(0)
    a = (rng.random((5, 40)) < 0.7).astype(float)
    b = (rng.random((5, 40)) < 0.5).astype(float)
    s = p8_adapter.stats
    assert stats.paired_contrast(a, b) == s.paired_bootstrap(a, b, reps=10000, seed=SEED)
    assert stats.paired_contrast(a, b) != s.paired_bootstrap(a, b)  # P8's default seed differs
    assert stats.paired_iqm_contrast(a, b) == s.paired_iqm_difference(a, b, reps=10000, seed=SEED)
    rates = a.mean(axis=1).reshape(-1, 1)
    assert stats.iqm_ci(rates) == s.stratified_bootstrap_ci(rates, s.iqm, reps=2000, seed=SEED)
    assert stats.aggregate(rates) == s.aggregate_estimates(rates, reps=2000, seed=SEED)
    assert stats.wilson(197, 200) == s.wilson_interval(197, 200)
    speeds_a = np.where(a > 0, 0.3, np.nan)
    speeds_b = np.where(b > 0, 0.4, np.nan)
    assert stats.relative_p95(speeds_a, speeds_b) == s.paired_relative_p95(
        speeds_a, speeds_b, reps=10000, seed=SEED
    )
    assert stats.quantile(np.array([1.0, 2.0, np.nan, 3.0]), 0.5) == 2.0


# --------------------------------------------------------------------------- joint-seed contrast


def test_joint_constant_difference_is_degenerate() -> None:
    # Per-episode difference exactly 0.25 on every run and episode (dyadic values: exact).
    rng = np.random.default_rng(1)
    b = rng.integers(0, 3, size=(5, 50)) * 0.25
    a = b + 0.25
    c = stats.joint_seed_contrast(a, b)
    assert (c.diff, c.lo, c.hi) == (0.25, 0.25, 0.25)
    assert (c.n_seeds_a, c.n_seeds_b, c.n_episodes, c.reps) == (5, 5, 50, 10000)


def test_joint_differs_from_independent_where_it_should() -> None:
    # Seeds differ wildly (run 0 never succeeds, run 1 always does), but within a run side a
    # leads side b by exactly 0.125 on every episode. Joint resampling sees a constant; P8's
    # independent resampling mixes runs across sides and spreads over about +-1.
    base = np.repeat(np.array([[0.0], [0.75]]), 30, axis=1)
    a, b = base + 0.125, base
    joint = stats.joint_seed_contrast(a, b)
    indep = stats.paired_contrast(a, b)
    assert joint.diff == indep.diff == 0.125
    assert (joint.lo, joint.hi) == (0.125, 0.125)
    assert indep.lo < -0.5 and indep.hi > 0.75


def test_joint_matches_unvectorised_reference() -> None:
    rng = np.random.default_rng(2)
    a = (rng.random((3, 17)) < 0.6).astype(float)
    b = (rng.random((3, 17)) < 0.5).astype(float)
    reps = 600  # spans three batches of 250
    got = stats.joint_seed_contrast(a, b, reps=reps, seed=7)
    gen = np.random.default_rng(7)
    vals = []
    for start in range(0, reps, 250):
        size = min(250, reps - start)
        seeds = gen.integers(0, 3, size=(size, 3))
        eps = gen.integers(0, 17, size=(size, 17))
        for r in range(size):
            vals.append(float(np.mean(a[np.ix_(seeds[r], eps[r])] - b[np.ix_(seeds[r], eps[r])])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    assert got.lo == pytest.approx(lo, abs=1e-12) and got.hi == pytest.approx(hi, abs=1e-12)
    assert got.diff == pytest.approx(a.mean() - b.mean(), abs=0)


def test_joint_deterministic_and_refuses_unpaired_shapes() -> None:
    rng = np.random.default_rng(3)
    a = (rng.random((5, 20)) < 0.6).astype(float)
    b = (rng.random((5, 20)) < 0.4).astype(float)
    assert stats.joint_seed_contrast(a, b) == stats.joint_seed_contrast(a, b)
    assert stats.joint_seed_contrast(a, b) != stats.joint_seed_contrast(a, b, seed=SEED + 1)
    with pytest.raises(ValueError):
        stats.joint_seed_contrast(a, b[:4])
    with pytest.raises(ValueError):
        stats.joint_seed_contrast(a, np.where(b > 0, np.nan, b))


# --------------------------------------------------------------------------- H2: W1 bootstrap


def test_w1_of_shifted_point_masses() -> None:
    n = 200
    ref = np.full(n, 1.5)
    bc = np.full((3, n), 2.0)  # W1 = 0.5 s
    dp = np.full((3, n), 1.75)  # W1 = 0.25 s
    w = stats.w1_bootstrap(bc, dp, ref)
    assert (w.w1_bc, w.w1_dp, w.d) == (0.5, 0.25, 0.25)
    assert (w.lo, w.hi, w.n_reps_undefined) == (0.25, 0.25, 0)
    assert w.separates
    assert (w.frac_without_touchdown_bc, w.frac_without_touchdown_ref) == (0.0, 0.0)


def test_w1_point_is_pooled_and_ignores_missing_touchdowns() -> None:
    rng = np.random.default_rng(4)
    ref = rng.normal(2.0, 0.5, 200)
    bc = rng.normal(2.5, 0.3, (3, 200))
    dp = rng.normal(2.1, 0.5, (3, 200))
    bc[0, :20] = np.nan
    ref[:10] = np.nan
    w = stats.w1_bootstrap(bc, dp, ref, reps=200)
    pool_ref = ref[np.isfinite(ref)]
    assert w.w1_bc == wasserstein_1(bc[np.isfinite(bc)], pool_ref)
    assert w.w1_dp == wasserstein_1(dp.ravel(), pool_ref)
    assert w.d == w.w1_bc - w.w1_dp
    assert (w.n_touchdowns_bc, w.n_touchdowns_dp, w.n_touchdowns_ref) == (580, 600, 190)
    assert w.frac_without_touchdown_bc == pytest.approx(20 / 600)
    assert w.frac_without_touchdown_ref == pytest.approx(10 / 200)


def test_w1_empty_pool_replicates_are_nan_and_counted() -> None:
    # bc touches down on episode 0 only; any replicate that never draws episode 0 has an
    # empty bc pool. With 3 episodes that is (2/3)^3 of replicates in expectation.
    ref = np.array([1.0, 2.0, 3.0])
    bc = np.array([[1.5, np.nan, np.nan]])
    dp = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.25]])
    reps = 4000
    w = stats.w1_bootstrap(bc, dp, ref, reps=reps, seed=11)
    gen = np.random.default_rng(11)
    expected = 0
    for _ in range(reps):
        gen.integers(0, 1, size=1)
        gen.integers(0, 2, size=2)
        eps = gen.integers(0, 3, size=3)
        expected += int(0 not in eps)
    assert w.n_reps_undefined == expected
    assert abs(expected / reps - (2 / 3) ** 3) < 0.03
    assert np.isfinite(w.lo) and np.isfinite(w.hi) and np.isfinite(w.d)


def test_w1_arm_without_any_touchdown_is_undefined() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    bc = np.full((2, 3), np.nan)
    dp = np.array([[1.0, 2.0, 3.0]])
    w = stats.w1_bootstrap(bc, dp, ref, reps=50)
    assert np.isnan(w.d) and np.isnan(w.lo) and np.isnan(w.hi)
    assert w.n_reps_undefined == 50 and not w.separates
    assert w.frac_without_touchdown_bc == 1.0


def test_w1_deterministic() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(2.0, 0.5, 40)
    bc = rng.normal(2.5, 0.3, (3, 40))
    dp = rng.normal(2.1, 0.5, (3, 40))
    first = stats.w1_bootstrap(bc, dp, ref, reps=300)
    assert first == stats.w1_bootstrap(bc, dp, ref, reps=300)
    assert first != stats.w1_bootstrap(bc, dp, ref, reps=300, seed=SEED + 1)


# --------------------------------------------------------------------------- H4: three arms


def test_h4_algebra_on_constructed_matrices() -> None:
    n = 200
    bc_state = np.zeros((5, n))
    bc_state[:, :160] = 1.0  # 0.80 on every seed
    bc = np.zeros((5, n))
    bc[:, :100] = 1.0  # 0.50
    teacher = np.zeros(n)
    teacher[:190] = 1.0  # 0.95
    g = stats.h4_three_arm_bootstrap(bc_state, bc, teacher, reps=500)
    assert (g.s_bc_state, g.s_bc, g.s_teacher) == (0.8, 0.5, 0.95)
    assert g.p == pytest.approx(0.30, abs=1e-15)
    assert g.i == pytest.approx(0.15, abs=1e-15)
    assert g.d == g.p - g.i
    assert g.p_lo <= g.p <= g.p_hi and g.i_lo <= g.i <= g.i_hi and g.d_lo <= g.d <= g.d_hi


def test_h4_constant_arms_give_degenerate_intervals() -> None:
    n = 50
    g = stats.h4_three_arm_bootstrap(np.ones((5, n)), np.zeros((5, n)), np.ones(n), reps=300)
    assert (g.p, g.p_lo, g.p_hi) == (1.0, 1.0, 1.0)
    assert (g.i, g.i_lo, g.i_hi) == (0.0, 0.0, 0.0)
    assert (g.d, g.d_lo, g.d_hi) == (1.0, 1.0, 1.0)


def test_h4_bc_state_draw_is_shared_by_both_gaps() -> None:
    # bc and the teacher are constant across episodes; bc_state's runs differ wildly. If one
    # bc_state draw enters both gaps, P + I = S(teacher) - S(bc) exactly in every replicate.
    n = 30
    bc_state = np.vstack([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n), np.ones(n)])
    bc = np.full((5, n), 0.0)
    teacher = np.ones(n)
    p, i, d = stats.h4_replicates(bc_state, bc, teacher, reps=1000, seed=SEED)
    assert np.all(p + i == 1.0)
    assert np.ptp(p) > 0.5  # the bc_state draw does vary
    assert np.array_equal(d, p - i)


def test_h4_deterministic_and_teacher_is_one_row_set() -> None:
    rng = np.random.default_rng(6)
    bs = (rng.random((5, 40)) < 0.8).astype(float)
    b = (rng.random((5, 40)) < 0.6).astype(float)
    t = (rng.random(40) < 0.95).astype(float)
    assert stats.h4_three_arm_bootstrap(bs, b, t) == stats.h4_three_arm_bootstrap(bs, b, t)
    assert stats.h4_three_arm_bootstrap(bs, b, t).reps == 10000
    with pytest.raises(ValueError):
        stats.h4_three_arm_bootstrap(bs, b, np.vstack([t, t]))
