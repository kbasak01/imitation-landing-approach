"""Table tests for ``ila.policies.dp.chunking`` (the only place To/Tp/Ta indices live).

Hand-written tables, not re-derivations of the formulas: the point is to pin the off-by-one
(executed actions start at chunk index ``To - 1``, not 0) and the ``Ta <= Tp - To + 1`` bound.
One control step = 1/30 s model scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from ila.policies.dp import chunking as ch

# (To, Tp, Ta) -> executed chunk indices.
EXECUTED_TABLE = [
    (1, 16, 1, (0,)),
    (1, 16, 8, (0, 1, 2, 3, 4, 5, 6, 7)),
    (1, 16, 16, tuple(range(16))),
    (2, 16, 1, (1,)),
    (2, 16, 8, (1, 2, 3, 4, 5, 6, 7, 8)),
    (2, 16, 12, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
    (2, 16, 15, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
    (3, 8, 6, (2, 3, 4, 5, 6, 7)),
]


@pytest.mark.parametrize(("to", "tp", "ta", "expected"), EXECUTED_TABLE)
def test_executed_indices_table(to: int, tp: int, ta: int, expected: tuple[int, ...]) -> None:
    assert ch.executed_indices(to, tp, ta) == expected
    s = ch.executed_slice(to, tp, ta)
    assert tuple(range(tp))[s] == expected


def test_executed_starts_at_to_minus_one_not_zero() -> None:
    # The pitfall: with To = 2 the first executed index is 1, which maps to step t itself.
    idx = ch.executed_indices(2, 16, 8)
    assert idx[0] == 1
    steps = ch.chunk_steps(10, 2, 16)
    assert [steps[j] for j in idx] == list(range(10, 18))


@pytest.mark.parametrize(
    ("to", "tp", "bound"), [(1, 16, 16), (2, 16, 15), (2, 2, 1), (3, 8, 6), (16, 16, 1)]
)
def test_max_action_steps(to: int, tp: int, bound: int) -> None:
    assert ch.max_action_steps(to, tp) == bound
    ch.ChunkSpec(to, tp, bound)  # the bound itself is admissible
    with pytest.raises(ValueError, match="Ta must satisfy"):
        ch.ChunkSpec(to, tp, bound + 1)


@pytest.mark.parametrize("ta", [0, -1, 16, 17])
def test_ta_out_of_range_refused(ta: int) -> None:
    with pytest.raises(ValueError):
        ch.ChunkSpec(2, 16, ta)
    with pytest.raises(ValueError):
        ch.executed_slice(2, 16, ta)


def test_invalid_to_tp_refused() -> None:
    with pytest.raises(ValueError):
        ch.max_action_steps(0, 16)
    with pytest.raises(ValueError):
        ch.max_action_steps(4, 3)


def test_chunkspec_with_action_steps() -> None:
    spec = ch.ChunkSpec(2, 16, 8)
    assert spec.with_action_steps(15).executed == slice(1, 16)
    with pytest.raises(ValueError):
        spec.with_action_steps(16)


# t -> obs window steps for To = 2 (first-step padding repeats step 0).
OBS_TABLE_TO2 = [(0, (0, 0)), (1, (0, 1)), (2, (1, 2)), (17, (16, 17))]
OBS_TABLE_TO1 = [(0, (0,)), (1, (1,)), (9, (9,))]
OBS_TABLE_TO3 = [(0, (0, 0, 0)), (1, (0, 0, 1)), (2, (0, 1, 2)), (5, (3, 4, 5))]


@pytest.mark.parametrize(("t", "expected"), OBS_TABLE_TO2)
def test_obs_window_to2(t: int, expected: tuple[int, ...]) -> None:
    assert ch.obs_window_steps(t, 2) == expected


@pytest.mark.parametrize(("t", "expected"), OBS_TABLE_TO1)
def test_obs_window_to1(t: int, expected: tuple[int, ...]) -> None:
    assert ch.obs_window_steps(t, 1) == expected


@pytest.mark.parametrize(("t", "expected"), OBS_TABLE_TO3)
def test_obs_window_to3(t: int, expected: tuple[int, ...]) -> None:
    assert ch.obs_window_steps(t, 3) == expected


def test_negative_step_refused() -> None:
    with pytest.raises(ValueError):
        ch.obs_window_steps(-1, 2)
    with pytest.raises(ValueError):
        ch.chunk_steps(-1, 2, 16)
    with pytest.raises(ValueError):
        ch.is_replan_step(-1, 8)


def test_chunk_steps_table() -> None:
    assert ch.chunk_steps(0, 2, 16) == tuple(range(-1, 15))
    assert ch.chunk_steps(8, 2, 16) == tuple(range(7, 23))
    assert ch.chunk_steps(5, 1, 4) == (5, 6, 7, 8)


# (t, Ta) -> (is replan, replan step)
REPLAN_TABLE = [
    (0, 8, True, 0),
    (1, 8, False, 0),
    (7, 8, False, 0),
    (8, 8, True, 8),
    (15, 8, False, 8),
    (16, 8, True, 16),
    (3, 1, True, 3),
    (14, 15, False, 0),
    (15, 15, True, 15),
]


@pytest.mark.parametrize(("t", "ta", "replan", "anchor"), REPLAN_TABLE)
def test_replan_table(t: int, ta: int, replan: bool, anchor: int) -> None:
    assert ch.is_replan_step(t, ta) is replan
    assert ch.replan_step(t, ta) == anchor


@pytest.mark.parametrize("ta", [1, 8, 15])
def test_replan_executes_every_step_exactly_once(ta: int) -> None:
    # Walk 40 steps: the action at t must be the executed index To-1 + (t - anchor) of the
    # chunk planned at anchor, and that index must map back to step t.
    to, tp = 2, 16
    idx = ch.executed_indices(to, tp, ta)
    for t in range(40):
        anchor = ch.replan_step(t, ta)
        j = idx[t - anchor]
        assert ch.chunk_steps(anchor, to, tp)[j] == t


# (t, T) -> target steps, To = 2, Tp = 4 (small Tp keeps the table readable).
TARGET_TABLE = [
    (0, 10, (0, 0, 1, 2)),
    (1, 10, (0, 1, 2, 3)),
    (5, 10, (4, 5, 6, 7)),
    (8, 10, (7, 8, 9, 9)),
    (9, 10, (8, 9, 9, 9)),
    (0, 1, (0, 0, 0, 0)),
]


@pytest.mark.parametrize(("t", "big_t", "expected"), TARGET_TABLE)
def test_target_steps_table(t: int, big_t: int, expected: tuple[int, ...]) -> None:
    assert ch.target_steps(t, 2, 4, big_t) == expected


def test_target_steps_default_horizon_padding() -> None:
    steps = ch.target_steps(0, 2, 16, 375)
    assert steps[:3] == (0, 0, 1) and steps[-1] == 14
    steps = ch.target_steps(374, 2, 16, 375)
    assert steps[:2] == (373, 374) and set(steps[2:]) == {374}


def test_target_anchor_out_of_range() -> None:
    with pytest.raises(ValueError):
        ch.target_steps(10, 2, 16, 10)
    with pytest.raises(ValueError):
        ch.target_steps(-1, 2, 16, 10)


def test_ensemble_weights() -> None:
    w = ch.ensemble_weights(1)
    np.testing.assert_array_equal(w, [1.0])
    w = ch.ensemble_weights(3, m=0.01)
    raw = np.exp(-0.01 * np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(w, raw / raw.sum(), rtol=0, atol=1e-15)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]  # i = 0 is the oldest and weighs most
    with pytest.raises(ValueError):
        ch.ensemble_weights(0)
