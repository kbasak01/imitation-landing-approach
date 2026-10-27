"""Phase 1: the P8 adapter re-exports P8's objects by reference and exposes scene handles."""

from __future__ import annotations

import inspect

import numpy as np
import pybullet as pyb
import pytest
import rld.eval.stats as p8_stats
from rld.envs.observation import obs_fields as p8_obs_fields

from ila import p8_adapter


def _env_after_reset(
    list_name: str = "id", ss: str = "SS6"
) -> tuple[p8_adapter.DeckLandingAviary, np.ndarray]:
    cfgs = p8_adapter.load_eval_configs()
    listed = next(r for r in p8_adapter.read_list(p8_adapter.list_path(list_name)) if r.ss == ss)
    env = p8_adapter.make_env(cfgs, listed)
    obs, _ = env.reset(seed=listed.episode_seed)
    return env, obs


# --------------------------------------------------------------------------- stats


def test_stats_namespace_covers_p8_all_exactly() -> None:
    exported = {n for n in dir(p8_adapter.stats) if not n.startswith("_")}
    assert exported == set(p8_stats.__all__)


@pytest.mark.parametrize("name", sorted(p8_stats.__all__))
def test_stats_reexport_is_identity(name: str) -> None:
    assert getattr(p8_adapter.stats, name) is getattr(p8_stats, name)
    assert getattr(p8_adapter.P8Stats, name) is getattr(p8_stats, name)


def test_stats_namespace_is_read_only() -> None:
    with pytest.raises(AttributeError):
        p8_adapter.stats.iqm = np.mean
    with pytest.raises(AttributeError):
        p8_adapter.P8Stats.iqm = np.mean


def test_stats_call_through() -> None:
    assert p8_adapter.stats.wilson_interval(5, 10) == p8_stats.wilson_interval(5, 10)


# --------------------------------------------------------------------------- obs layout


def test_obs_fields_is_p8_layout_under_eval_config() -> None:
    cfgs = p8_adapter.load_eval_configs()
    assert p8_adapter.obs_fields() == p8_obs_fields(cfgs.observation)
    assert sum(f.size for f in p8_adapter.obs_fields()) == 25


def test_list_paths_are_p8_lists() -> None:
    paths = p8_adapter.list_paths()
    assert list(paths) == ["id", "unseen_seastate", "unseen_heading", "unseen_vessel", "static"]
    assert all(p.is_file() for p in paths.values())


# --------------------------------------------------------------------------- teachers


def _assert_eval_policy(policy: p8_adapter.EvalPolicy) -> None:
    for method in ("reset", "act"):
        assert callable(getattr(policy, method, None)), method
    params = list(inspect.signature(policy.reset).parameters)
    assert params[:2] == ["seed", "context"]
    assert getattr(policy, "privileged", False) is False


def _assert_acts(policy: p8_adapter.EvalPolicy) -> None:
    env, obs = _env_after_reset()
    try:
        policy.reset(0, None)
        action = np.asarray(policy.act(obs))
    finally:
        env.close()
    assert action.shape == (3,)
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0)


@pytest.mark.pybullet
@pytest.mark.parametrize("name", ["pid_feedforward", "gated"])
def test_controller_teacher_is_eval_policy(name: str) -> None:
    policy = p8_adapter.teacher(name, 0)
    _assert_eval_policy(policy)
    _assert_acts(policy)


@pytest.mark.checkpoints
@pytest.mark.pybullet
def test_ppo_teacher_is_eval_policy() -> None:
    policy = p8_adapter.teacher("ppo", 0)
    _assert_eval_policy(policy)
    _assert_acts(policy)


@pytest.mark.parametrize(
    ("name", "seed"), [("ppo", 1), ("pid_feedforward", 3), ("oracle_gated", 0), ("bc", 0)]
)
def test_teacher_rejects_unavailable(name: str, seed: int) -> None:
    with pytest.raises(ValueError, match="teacher"):
        p8_adapter.teacher(name, seed)


# --------------------------------------------------------------------------- scene


@pytest.mark.pybullet
def test_scene_handles_find_p8_bodies() -> None:
    env, _ = _env_after_reset()
    try:
        h = p8_adapter.scene_handles(env)
        assert len({h.plane, h.drone, h.plate, h.pad_marker}) == 4
        marker = pyb.getVisualShapeData(h.pad_marker, physicsClientId=h.client)
        assert len(marker) == 1 and marker[0][2] == pyb.GEOM_CYLINDER
        assert tuple(marker[0][7]) == pytest.approx(p8_adapter.PAD_MARKER_RGBA)
        assert pyb.getCollisionShapeData(h.pad_marker, -1, physicsClientId=h.client) == ()
        plate = pyb.getVisualShapeData(h.plate, physicsClientId=h.client)
        assert plate[0][2] == pyb.GEOM_BOX
        # URDF_IGNORE_VISUAL_SHAPES drops cf2.dae, but PyBullet then renders the collision
        # cylinder (length 0.025 m, radius 0.06 m, white) as the drone's visual (P1-D1).
        drone = pyb.getVisualShapeData(h.drone, physicsClientId=h.client)
        assert [(v[1], v[2]) for v in drone] == [(-1, pyb.GEOM_CYLINDER)]
        assert tuple(drone[0][3][:2]) == pytest.approx((0.025, 0.06))
    finally:
        env.close()


@pytest.mark.pybullet
def test_physics_step_count() -> None:
    env, _ = _env_after_reset()
    try:
        assert p8_adapter.physics_step_count(env) == 0
        env.step(np.zeros(3))
        assert p8_adapter.physics_step_count(env) == 8
        env.step(np.zeros(3))
        assert p8_adapter.physics_step_count(env) == 16
    finally:
        env.close()
