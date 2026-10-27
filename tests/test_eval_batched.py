"""Phase 2: lockstep rendering workers + one batching server (CPU path).

On the CPU single-sample path the rows do not depend on the worker count, equal the in-worker
runner's rows, and every request reaches the policy with its ``(run_seed, episode_seed, step)``
key. The policy here is a trivial deterministic function of the pixels and proprio
(``runner_support``), never a network. Flown on P8's ``tiny_list`` fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.eval.episodes import ListedEpisode

from ila import p8_adapter
from ila.eval import batched, runner
from runner_support import build_pixel_student, build_tiny_list, pixel_action, replay_inputs

pytestmark = [pytest.mark.pybullet, pytest.mark.slow]


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    return load_eval_configs()


@pytest.fixture(scope="module")
def tiny_list(cfgs: EvalConfigs) -> list[ListedEpisode]:
    return build_tiny_list(cfgs)


@pytest.fixture(scope="module")
def spec() -> p8_adapter.PolicySpec:
    return p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=7)


@pytest.fixture(scope="module")
def in_worker_text(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs, spec: p8_adapter.PolicySpec
) -> str:
    rows = runner.run_list_vision(tiny_list, spec, cfgs, runner.RowTags("pixel"), workers=1)
    return runner.vision_rows_csv(rows)


def _arm(
    spec: p8_adapter.PolicySpec, episodes: Sequence[ListedEpisode], **kw: Any
) -> batched.BatchedArm:
    return batched.BatchedArm(spec=spec, episodes=episodes, tags=runner.RowTags("pixel"), **kw)


@pytest.mark.parametrize(("workers", "chunk"), [(1, 25), (3, 1), (2, 2)])
def test_cpu_single_sample_rows_independent_of_workers(
    workers: int,
    chunk: int,
    tiny_list: list[ListedEpisode],
    cfgs: EvalConfigs,
    spec: p8_adapter.PolicySpec,
    in_worker_text: str,
) -> None:
    rows = batched.run_batched(_arm(spec, tiny_list), cfgs, workers=workers, chunk=chunk)
    assert runner.vision_rows_csv(rows) == in_worker_text
    assert [r["index"] for r in rows] == [e.index for e in tiny_list]


class _KeyRecorder:
    """A native BatchPolicy recording resets, keys, inputs and batch sizes (in the server)."""

    def __init__(self, run_seed: int) -> None:
        self.run_seed = run_seed
        self.resets: list[tuple[int, int]] = []
        self.keys: dict[int, list[batched.NoiseKey]] = {}
        self.inputs: dict[int, list[tuple[bytes, bytes, tuple[int, ...], tuple[int, ...]]]] = {}
        self.batch_sizes: list[int] = []

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        self.resets.append((slot, episode_seed))

    def act_batch(self, requests: Sequence[batched.SlotRequest]) -> np.ndarray:
        self.batch_sizes.append(len(requests))
        for req in requests:
            assert not req.images.flags.writeable and not req.proprio.flags.writeable
            self.keys.setdefault(req.key.episode_seed, []).append(req.key)
            assert req.images.dtype == np.uint8 and req.proprio.dtype == np.float32
            self.inputs.setdefault(req.key.episode_seed, []).append(
                (req.images.tobytes(), req.proprio.tobytes(), req.images.shape, req.proprio.shape)
            )
        return np.stack([pixel_action(r.images, r.proprio) for r in requests])


def _constant(policy: _KeyRecorder) -> Any:
    def build(cfgs: EvalConfigs, slots: int) -> _KeyRecorder:
        del cfgs, slots
        return policy

    return build


@pytest.mark.parametrize("cap", batched.BATCH_CAPS)
def test_noise_keys_and_batch_cap(
    cap: int,
    tiny_list: list[ListedEpisode],
    cfgs: EvalConfigs,
    spec: p8_adapter.PolicySpec,
    in_worker_text: str,
) -> None:
    recorder = _KeyRecorder(spec.run_seed)
    arm = _arm(spec, tiny_list, server_build=_constant(recorder), batch_cap=cap)
    rows = batched.run_batched(arm, cfgs, workers=3, chunk=1)
    # A per-sample policy gives the same rows at any cap (not claimed for a real network).
    assert runner.vision_rows_csv(rows) == in_worker_text
    assert sorted(seed for _, seed in recorder.resets) == sorted(e.episode_seed for e in tiny_list)
    for row in rows:
        keys = recorder.keys[int(row["episode_seed"])]
        assert keys == [
            batched.NoiseKey(spec.run_seed, int(row["episode_seed"]), t)
            for t in range(int(row["steps"]))
        ]
    # P1-D2 obligation 2 on the batched path: what reached the server is, byte for byte, the
    # contract-built StudentInput of an independent hand replay of each episode.
    by_seed = {e.episode_seed: e for e in tiny_list}
    for row in rows:
        seed = int(row["episode_seed"])
        expected = replay_inputs(by_seed[seed], cfgs, int(row["steps"]))
        got = recorder.inputs[seed]
        assert len(got) == len(expected) == int(row["steps"])
        for (img_b, prop_b, img_shape, prop_shape), (exp_img, exp_prop) in zip(
            got, expected, strict=True
        ):
            assert img_shape == exp_img.shape and prop_shape == exp_prop.shape
            assert img_b == exp_img.tobytes() and prop_b == exp_prop.tobytes()
    assert max(recorder.batch_sizes) <= min(cap, 3)
    if cap >= 3:
        assert max(recorder.batch_sizes) == 3  # all live slots in one batch


def test_refuses_privileged_and_feed(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs, spec: p8_adapter.PolicySpec
) -> None:
    priv = p8_adapter.PolicySpec("pixel", True, build_pixel_student)
    with pytest.raises(ValueError, match="privileged"):
        batched.run_batched(_arm(priv, tiny_list[:1]), cfgs, workers=1)
    feed = p8_adapter.PolicySpec("pixel", False, build_pixel_student, needs_motion_feed=True)
    with pytest.raises(ValueError, match="privileged or fed"):
        batched.run_batched(_arm(feed, tiny_list[:1]), cfgs, workers=1)
    bad_seed = _arm(spec, tiny_list[:1], server_build=_constant(_KeyRecorder(spec.run_seed + 1)))
    with pytest.raises(ValueError, match="run seed"):
        batched.run_batched(bad_seed, cfgs, workers=1)


def test_worker_error_surfaces(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    from dataclasses import replace

    spec = p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=0)
    bad = [replace(tiny_list[1], t0_model_s=tiny_list[1].t0_model_s + 1e-12)]
    with pytest.raises(batched.WorkerError, match="EpisodeListMismatchError"):
        batched.run_batched(_arm(spec, bad), cfgs, workers=1)
