"""Table tests for ``ila.policies.dp.chunking`` (the only place To/Tp/Ta indices live).

Hand-written tables, not re-derivations of the formulas: the point is to pin the off-by-one
(executed actions start at chunk index ``To - 1``, not 0) and the ``Ta <= Tp - To + 1`` bound.
One control step = 1/30 s model scale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ila.policies.dp import chunking as ch

# (To, Tp, Ta) -> executed chunk indices.
EXECUTED_TABLE = [
    (1, 16, 1, (0,)),
    (1, 16, 8, (0, 1, 2, 3, 4, 5, 6, 7)),
    (1, 16, 15, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)),
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


# (t, T) -> target steps, To = 1, Tp = 4: the chunk starts at t itself, so there is no
# first-step padding; only the end repeats label[T-1].
TARGET_TABLE_TO1 = [
    (0, 10, (0, 1, 2, 3)),
    (1, 10, (1, 2, 3, 4)),
    (7, 10, (7, 8, 9, 9)),
    (9, 10, (9, 9, 9, 9)),
    (0, 1, (0, 0, 0, 0)),
]


@pytest.mark.parametrize(("t", "big_t", "expected"), TARGET_TABLE_TO1)
def test_target_steps_table_to1(t: int, big_t: int, expected: tuple[int, ...]) -> None:
    assert ch.target_steps(t, 1, 4, big_t) == expected


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


# --------------------------------------------------------------------------- ensembling sources
# (To, Ta, t) -> ((replan step s, chunk index j), ...), oldest first, Tp = 16. Step t averages
# the replans s in [max(0, t - Ta + 1), t], each at chunk index To - 1 + (t - s). Rows with
# t < Ta - 1 are the start-of-episode truncation (fewer than Ta predictions exist).
ENSEMBLE_TABLE = [
    # To = 1
    (1, 1, 0, ((0, 0),)),
    (1, 1, 5, ((5, 0),)),
    (1, 8, 0, ((0, 0),)),
    (1, 8, 3, ((0, 3), (1, 2), (2, 1), (3, 0))),
    (1, 8, 7, ((0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0))),
    (1, 8, 20, ((13, 7), (14, 6), (15, 5), (16, 4), (17, 3), (18, 2), (19, 1), (20, 0))),
    (1, 15, 0, ((0, 0),)),
    (1, 15, 2, ((0, 2), (1, 1), (2, 0))),
    (1, 15, 14, ((0, 14), (1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8), (7, 7), (8, 6),
                 (9, 5), (10, 4), (11, 3), (12, 2), (13, 1), (14, 0))),
    (1, 15, 30, ((16, 14), (17, 13), (18, 12), (19, 11), (20, 10), (21, 9), (22, 8), (23, 7),
                 (24, 6), (25, 5), (26, 4), (27, 3), (28, 2), (29, 1), (30, 0))),
    # To = 2
    (2, 1, 0, ((0, 1),)),
    (2, 1, 9, ((9, 1),)),
    (2, 8, 0, ((0, 1),)),
    (2, 8, 1, ((0, 2), (1, 1))),
    (2, 8, 3, ((0, 4), (1, 3), (2, 2), (3, 1))),
    (2, 8, 7, ((0, 8), (1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1))),
    (2, 8, 20, ((13, 8), (14, 7), (15, 6), (16, 5), (17, 4), (18, 3), (19, 2), (20, 1))),
    (2, 15, 0, ((0, 1),)),
    (2, 15, 1, ((0, 2), (1, 1))),
    (2, 15, 14, ((0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7),
                 (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1))),
    (2, 15, 40, ((26, 15), (27, 14), (28, 13), (29, 12), (30, 11), (31, 10), (32, 9), (33, 8),
                 (34, 7), (35, 6), (36, 5), (37, 4), (38, 3), (39, 2), (40, 1))),
]  # fmt: skip


@pytest.mark.parametrize(("to", "ta", "t", "expected"), ENSEMBLE_TABLE)
def test_ensemble_sources_table(
    to: int, ta: int, t: int, expected: tuple[tuple[int, int], ...]
) -> None:
    sources = ch.ensemble_sources(t, to, 16, ta)
    assert sources == expected
    w = ch.ensemble_weights(len(sources))
    assert len(w) == len(sources) and math.fsum(w) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(("to", "ta"), [(1, 1), (1, 8), (1, 15), (2, 1), (2, 8), (2, 15)])
def test_ensemble_sources_point_at_step_t_inside_the_executed_window(to: int, ta: int) -> None:
    idx = ch.executed_indices(to, 16, ta)
    for t in range(40):
        sources = ch.ensemble_sources(t, to, 16, ta)
        assert len(sources) == min(t + 1, ta)
        assert [s for s, _ in sources] == sorted(s for s, _ in sources)  # oldest first
        for s, j in sources:
            assert ch.chunk_steps(s, to, 16)[j] == t  # the prediction made for step t
            assert j in idx  # ... and one that chunk would have executed without ensembling


@pytest.mark.parametrize("to", [1, 2])
def test_ensembling_with_ta_1_is_no_ensembling(to: int) -> None:
    # Ta = 1: one source, the replan at t itself, read at the executed index To - 1, weight 1.
    first = ch.executed_indices(to, 16, 1)[0]
    for t in range(20):
        assert ch.ensemble_sources(t, to, 16, 1) == ((ch.replan_step(t, 1), first),)
    np.testing.assert_array_equal(ch.ensemble_weights(1), [1.0])


def test_ensemble_sources_refuse_inadmissible_ta() -> None:
    with pytest.raises(ValueError, match="Ta must satisfy"):
        ch.ensemble_sources(0, 2, 16, 16)  # Ta > Tp - To + 1 = 15
    with pytest.raises(ValueError, match="Ta must satisfy"):
        ch.ensemble_sources(0, 1, 16, 17)
    with pytest.raises(ValueError):
        ch.ensemble_sources(0, 2, 16, 0)
    with pytest.raises(ValueError):
        ch.ensemble_sources(-1, 2, 16, 8)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 12, 15])
def test_ensemble_weights_sum_to_one_oldest_heaviest(n: int) -> None:
    w = ch.ensemble_weights(n)
    assert math.fsum(w) == pytest.approx(1.0, abs=1e-15)
    assert np.all(np.diff(w) < 0) or n == 1
    # w_i / w_0 = exp(-m i), m = 0.01 (the config's m).
    np.testing.assert_allclose(w / w[0], np.exp(-0.01 * np.arange(n)), rtol=1e-15)
    assert ch.ENSEMBLE_DECAY == 0.01


# Latency arm (P3-D1 §7): (start, Ta, d, t) -> request due. Hand-written, Ta = 8.
REQUEST_DUE_TABLE = [
    (None, 8, 0, 0, True),  # no chunk yet: always due
    (None, 8, 5, 3, True),
    (0, 8, 0, 0, False),  # d = 0: due only once the chunk has run out (t = start + Ta)
    (0, 8, 0, 7, False),
    (0, 8, 0, 8, True),
    (2, 8, 2, 7, False),  # 3 steps left > d = 2
    (2, 8, 2, 8, True),  # 2 steps left = d
    (2, 8, 2, 9, True),  # 1 left
    (9, 8, 9, 9, True),  # d > Ta: due on landing
    (9, 8, 9, 20, True),  # run out
    (0, 1, 0, 0, False),
    (0, 1, 0, 1, True),
]


@pytest.mark.parametrize(("start", "ta", "d", "t", "due"), REQUEST_DUE_TABLE)
def test_delayed_request_due_table(start: int | None, ta: int, d: int, t: int, due: bool) -> None:
    assert ch.delayed_request_due(start, ta, d, t) is due


# (start, Ta, t) -> offset into the executing chunk, None when none executes.
CHUNK_OFFSET_TABLE = [
    (None, 8, 0, None),
    (None, 8, 11, None),
    (0, 8, 0, 0),
    (0, 8, 7, 7),
    (0, 8, 8, None),  # ran out
    (9, 8, 9, 0),
    (9, 8, 16, 7),
    (9, 8, 17, None),
    (3, 1, 3, 0),
    (3, 1, 4, None),
]


@pytest.mark.parametrize(("start", "ta", "t", "offset"), CHUNK_OFFSET_TABLE)
def test_delayed_chunk_offset_table(start: int | None, ta: int, t: int, offset: int | None) -> None:
    assert ch.delayed_chunk_offset(start, ta, t) == offset


@pytest.mark.parametrize(
    ("start", "ta", "d", "t"),
    [
        (0, 8, -1, 0),  # d < 0
        (0, 0, 0, 0),  # Ta < 1
        (5, 8, 0, 4),  # landed after t
        (-1, 8, 0, 0),  # landed before step 0
        (None, 8, 0, -1),  # t < 0
    ],
)
def test_delayed_schedule_refuses_bad_arguments(start: int | None, ta: int, d: int, t: int) -> None:
    with pytest.raises(ValueError):
        ch.delayed_request_due(start, ta, d, t)
    if d >= 0:
        with pytest.raises(ValueError):
            ch.delayed_chunk_offset(start, ta, t)
