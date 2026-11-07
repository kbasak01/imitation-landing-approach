"""Phase 9 latency arm (P3-D1 §7): exact timelines, d = 0 equals the synchronous arm, holds.

Fake students (no network) pin the request/landing/hold timeline step by step; the runner tests
fly dummy ``bc`` and ``dp`` students on two ``select`` episodes (CPU path, PyBullet).
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ila.deploy import latency_arm as la
from ila.policies.dp import chunking
from ila.policies.dp.policy import ChunkedDiffusionStudent, SeedTuple
from ila.policies.encoder import EncoderConfig, VisionStudentBase

TO, TP = 2, 16


class FakeDP(ChunkedDiffusionStudent):
    """chunk[j] = (request step, j, episode seed); records each request's key."""

    def __init__(self, ta: int = 8) -> None:
        """Executor at ``Ta = ta``, run seed 7."""
        super().__init__(
            7, chunking.ChunkSpec(TO, TP, 8), 100, n_action_steps=ta, num_inference_steps=10
        )
        self.cfg = EncoderConfig()
        self.keys: list[SeedTuple] = []

    def sample_batch(
        self,
        images: Any,
        proprio: Any,
        seeds: Sequence[SeedTuple],
        num_inference_steps: int | None = None,
    ) -> NDArray[np.float32]:
        """The chunk encodes its key (class docstring)."""
        del images, proprio, num_inference_steps
        self.keys.extend(seeds)
        out = np.zeros((len(seeds), TP, 3), np.float32)
        for i, (_, ep, t) in enumerate(seeds):
            out[i, :, 0] = t
            out[i, :, 1] = np.arange(TP)
            out[i, :, 2] = ep
        return out


class FakeBC(VisionStudentBase):
    """action = (call number, 0, 0), from 1; records the calls."""

    def __init__(self) -> None:
        """No network."""
        self.cfg = EncoderConfig()
        self._t = 0
        self.calls: list[int] = []

    def _reset_episode(self, seed: int) -> None:
        del seed
        self._t = 0

    def act_vision(self, image_stack: Any, proprio: Any) -> NDArray[np.float32]:
        """The call number as the action."""
        del image_stack, proprio
        self.calls.append(self._t)
        a = np.array([self._t + 1, 0, 0], np.float32)
        self._t += 1
        return a


def _window() -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    return np.zeros((TO, 96, 96, 3), np.uint8), np.zeros((TO, 13), np.float32)


def _fly(student: Any, steps: int, seed: int = 3) -> list[NDArray[np.float32]]:
    student.reset(seed)
    img, prop = _window()
    return [student.act_vision(img, prop) for _ in range(steps)]


def test_delayed_index_table() -> None:
    assert [chunking.delayed_chunk_index(2, 16, 8, 0, o) for o in range(8)] == list(
        chunking.executed_indices(2, 16, 8)
    )
    assert [chunking.delayed_chunk_index(2, 16, 8, 3, o) for o in range(8)] == list(range(4, 12))
    assert [chunking.delayed_chunk_index(2, 16, 8, 7, o) for o in range(8)] == list(range(8, 16))
    assert [chunking.delayed_chunk_index(2, 16, 8, 8, o) for o in range(8)] == [*range(9, 16), 15]
    with pytest.raises(ValueError):
        chunking.delayed_chunk_index(2, 16, 8, -1, 0)
    with pytest.raises(ValueError):
        chunking.delayed_chunk_index(2, 16, 8, 0, 8)


@pytest.mark.parametrize("ta", [1, 4, 8, 12])
def test_dp_d0_equals_the_synchronous_student(ta: int) -> None:
    sync = FakeDP(ta)
    want = _fly(sync, 40)
    inner = FakeDP(ta)
    got = _fly(la.DelayedDiffusionStudent(inner, 0), 40)
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
    assert inner.keys == sync.keys


def test_dp_d2_timeline() -> None:
    inner = FakeDP()
    out = _fly(la.DelayedDiffusionStudent(inner, 2), 20)
    assert [k[2] for k in inner.keys] == [0, 8, 16]  # issued when 2 steps are left
    assert all(np.array_equal(out[t], [0, 0, 0]) for t in (0, 1))  # zero until the first lands
    # Landed at 2: chunk planned at 0, index (To-1)+d+offset = 3 ... 10, then the next at 10.
    assert [(int(a[0]), int(a[1])) for a in out[2:10]] == [(0, j) for j in range(3, 11)]
    assert [(int(a[0]), int(a[1])) for a in out[10:18]] == [(8, j) for j in range(3, 11)]
    assert all(int(a[2]) == 3 for a in out[2:])  # the episode seed in the noise key


def test_dp_d9_holds_the_last_chunk_action_then_the_last_action() -> None:
    inner = FakeDP()
    out = _fly(la.DelayedDiffusionStudent(inner, 9), 30)
    assert [k[2] for k in inner.keys] == [
        0,
        9,
        18,
        27,
    ]  # d > Ta: issued on landing, never overlapping
    assert all(np.array_equal(a, [0, 0, 0]) for a in out[:9])
    idx = [(int(a[0]), int(a[1])) for a in out[9:18]]
    # Index (To-1)+9 = 10 ... 15, then index 15 held (past Tp-1), then the chunk runs out at 17.
    assert idx == [(0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 15), (0, 15), (0, 15)]
    assert [(int(a[0]), int(a[1])) for a in out[18:20]] == [(9, 10), (9, 11)]


def test_dp_requests_never_overlap() -> None:
    for d in range(0, 20):
        inner = FakeDP()
        _fly(la.DelayedDiffusionStudent(inner, d), 60)
        issued = [k[2] for k in inner.keys]
        assert all(b - a >= max(d, 1) for a, b in zip(issued, issued[1:], strict=False)), (
            d,
            issued,
        )


def test_dp_refuses_ensembling() -> None:
    inner = FakeDP()
    inner.temporal_ensembling = True
    with pytest.raises(ValueError):
        la.DelayedDiffusionStudent(inner, 1)


def test_bc_d0_equals_inner_and_d3_timeline() -> None:
    inner0 = FakeBC()
    assert [int(a[0]) for a in _fly(la.DelayedBCStudent(inner0, 0), 10)] == list(range(1, 11))
    inner = FakeBC()
    out = [int(a[0]) for a in _fly(la.DelayedBCStudent(inner, 3), 12)]
    # Requests at 0, 3, 6, 9 (inner sees only those observations); each lands 3 steps later.
    assert len(inner.calls) == 4
    assert out == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]  # zero, then each result held 3 steps


class _StubTeacher:
    """P8 ``EvalPolicy`` shape; never asked to act here."""

    def reset(self, seed: int, context: Any = None) -> None:  # noqa: D102
        del seed, context

    def act(self, obs: Any) -> NDArray[np.float32]:  # noqa: D102
        del obs
        raise AssertionError("not flown")


def test_wrap_delayed_accepts_only_bc_and_chunked_dp() -> None:
    from ila.policies.bc import BCStudent
    from ila.policies.modular import ModularEstimator, ModularStudent
    from ila.policies.registry import build_dummy, build_net

    bc = build_dummy("bc", 0)
    assert isinstance(bc, BCStudent)
    assert isinstance(la.wrap_delayed(bc, 2), la.DelayedBCStudent)
    assert isinstance(la.wrap_delayed(FakeDP(), 2), la.DelayedDiffusionStudent)
    net = build_net("modular", 0)
    assert isinstance(net, ModularEstimator)
    modular = ModularStudent(net, "cpu", teacher=_StubTeacher())
    assert isinstance(modular, VisionStudentBase) and not hasattr(modular, "chunk_spec")
    for other in (modular, FakeBC(), object()):
        with pytest.raises(TypeError, match="no latency-arm wrapper"):
            la.wrap_delayed(other, 2)


def test_delayed_policy_build_pickles() -> None:
    build = functools.partial(
        la.delayed_checkpoint_policy, "/x.pt", 4, n_action_steps=8, num_inference_steps=10
    )
    assert pickle.loads(pickle.dumps(build)).args == ("/x.pt", 4)


def test_read_delays(tmp_path: Any) -> None:
    p = tmp_path / "delay.json"
    p.write_text('{"arms": {"dp": {"d": 7}, "bc_dagger": {"d": 1}}}')
    assert la.read_delays(p) == {"dp": 7, "bc_dagger": 1}
    p.write_text('{"arms": {"dp": {"d": null, "status": "x"}}}')
    with pytest.raises(ValueError):
        la.read_delays(p)


# --------------------------------------------------------------------------- runner flights


def _select(n: int) -> list[Any]:
    from ila.data import episodes as data_episodes

    return list(data_episodes.read_split("select"))[:n]


def _dummy(arm: str, delay: int | None, cfgs: Any) -> Any:
    from ila.policies.registry import build_dummy

    s = (
        build_dummy(arm, 0, n_action_steps=8, num_inference_steps=2)
        if arm == "dp"
        else build_dummy(arm, 0)
    )
    return s if delay is None else la.wrap_delayed(s, delay)


def _rows(arm: str, delay: int | None) -> list[dict[str, Any]]:
    from ila import p8_adapter
    from ila.eval import runner

    spec = p8_adapter.callable_spec(arm, functools.partial(_dummy, arm, delay), run_seed=0)
    return runner.run_list_vision(
        _select(2), spec, p8_adapter.load_eval_configs(), runner.RowTags(arm=arm), workers=1
    )


@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_d0_flies_row_for_row_like_the_synchronous_arm(arm: str) -> None:
    from ila.deploy.closed_loop import rows_text

    # Compared as the CSV text selection writes (NaN cells compare equal as text).
    assert rows_text(_rows(arm, 0)) == rows_text(_rows(arm, None))


def test_d9_hold_path_flies() -> None:
    rows = _rows("dp", 9)
    assert len(rows) == 2 and all(int(r["steps"]) >= 1 for r in rows)
