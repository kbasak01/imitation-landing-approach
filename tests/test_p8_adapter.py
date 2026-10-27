"""Phase 1: the P8 adapter re-exports P8's objects by reference and exposes scene handles."""

from __future__ import annotations

import inspect

import numpy as np
import pybullet as pyb
import pytest
import rld.eval.stats as p8_stats
from rld.envs.observation import obs_fields as p8_obs_fields

from ila import p8_adapter
from ila.data.episodes import read_split


def _env_after_reset(ss: str = "SS5") -> tuple[p8_adapter.DeckLandingAviary, np.ndarray]:
    """Reset on a ``train``-pool row of the committed ``demos`` split (P1-D2 obligation 1)."""
    cfgs = p8_adapter.load_eval_configs()
    listed = next(r for r in read_split("demos_0-1999") if r.ss == ss)
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


# ------------------------------------------------------------ runner contract (Phase 2)


def _p8_runner_objects() -> dict[str, object]:
    """Every Phase 2 re-export, keyed by adapter name, as P8 defines it (pinned commit)."""
    import dmf.data.splits
    import rld.control.base
    import rld.deck.splits
    import rld.deploy.onnx_policy
    import rld.deploy.parity
    import rld.envs.landing_env
    import rld.eval.envs
    import rld.eval.episodes
    import rld.eval.reproduce
    import rld.eval.runner
    import rld.eval.storage
    import rld.eval.truth

    runner = rld.eval.runner
    return {
        "EPISODE_COLUMNS": runner.EPISODE_COLUMNS,
        "LIST_ROW_COLUMNS": runner.LIST_ROW_COLUMNS,
        "RECORD_COLUMNS": runner.RECORD_COLUMNS,
        "DEFAULT_CHUNK": runner.DEFAULT_CHUNK,
        "DEFAULT_WORKERS": runner.DEFAULT_WORKERS,
        "PolicySpec": runner.PolicySpec,
        "controller_spec": runner.controller_spec,
        "callable_spec": runner.callable_spec,
        "episode_rows_csv": runner.episode_rows_csv,
        "feed_skip_reason": runner.feed_skip_reason,
        "split_runnable": runner.split_runnable,
        "EpisodeListMismatchError": runner.EpisodeListMismatchError,
        "EvalPolicy": runner.EvalPolicy,
        "motion_for": rld.eval.envs.motion_for,
        "pad_offset_for": rld.eval.envs.pad_offset_for,
        "PrivilegedContext": rld.control.base.PrivilegedContext,
        "td_in_quiescent_window": rld.eval.truth.td_in_quiescent_window,
        "touchdown_rule": rld.eval.truth.touchdown_rule,
        "ACTION_DIM": rld.envs.landing_env.ACTION_DIM,
        "compare_to_reference": rld.eval.reproduce.compare_to_reference,
        "read_text": rld.eval.storage.read_text,
        "dev_pool": rld.deck.splits.dev_pool,
        "EpisodeDraw": rld.eval.episodes.EpisodeDraw,
        "resolve_draws": rld.eval.episodes.resolve_draws,
        "label_entropy": rld.eval.episodes.label_entropy,
        "cell_order": rld.eval.episodes.cell_order,
        "canonical_csv": rld.eval.episodes.canonical_csv,
        "content_sha256": rld.eval.episodes.content_sha256,
        "DEFAULT_PAD": rld.eval.episodes.DEFAULT_PAD,
        "reference_actions": rld.deploy.parity.reference_actions,
        "open_session": rld.deploy.parity.open_session,
        "check_ort": rld.deploy.parity.check_ort,
        "PARITY_TOLERANCE": rld.deploy.parity.PARITY_TOLERANCE,
        "cpu_session": rld.deploy.onnx_policy.cpu_session,
        "attach_onnx": rld.deploy.onnx_policy.attach_onnx,
        # P8's runner imports dmf's function; the adapter re-exports that same object.
        "realization_key": dmf.data.splits.realization_key,
    }


@pytest.mark.parametrize("name", sorted(_p8_runner_objects()))
def test_runner_reexport_is_identity(name: str) -> None:
    assert getattr(p8_adapter, name) is _p8_runner_objects()[name]
    assert name in p8_adapter.__all__


def test_realization_key_is_the_one_p8_runner_uses() -> None:
    import rld.eval.runner

    assert p8_adapter.realization_key is rld.eval.runner.realization_key


def test_resolve_draws_signature_is_p8s() -> None:
    import inspect as _inspect

    params = _inspect.signature(p8_adapter.resolve_draws).parameters
    assert list(params)[:2] == ["draws", "cfgs"]
    assert {"workers", "chunk"} <= set(params)
    assert params["workers"].kind is _inspect.Parameter.KEYWORD_ONLY
    assert p8_adapter.DEFAULT_PAD == "aft"


def test_runner_constants_are_p8_values() -> None:
    # Values as read in P8's source (rld/eval/runner.py, rld/envs/landing_env.py).
    assert p8_adapter.DEFAULT_CHUNK == 25
    assert p8_adapter.DEFAULT_WORKERS == 24
    assert p8_adapter.ACTION_DIM == 3
    assert len(p8_adapter.EPISODE_COLUMNS) == 13 + 3 + 23 + 6
    assert p8_adapter.EPISODE_COLUMNS[: len(p8_adapter.LIST_ROW_COLUMNS)] == (
        p8_adapter.LIST_ROW_COLUMNS
    )


@pytest.mark.pybullet
def test_set_motion_matches_p8_run_chunk() -> None:
    """``set_motion`` re-points a reused env the way P8's ``run_chunk`` does (hull switch)."""
    cfgs = p8_adapter.load_eval_configs()
    first = p8_adapter.read_list(p8_adapter.list_path("static"))[0]
    moving = p8_adapter.read_list(p8_adapter.list_path("unseen_vessel"))[0]
    assert moving.vessel != first.vessel
    env = p8_adapter.make_env(cfgs, first)
    try:
        for listed in (first, moving, first):
            pad, offset = p8_adapter.set_motion(env, cfgs, listed)
            assert pad == listed.pad
            assert offset == p8_adapter.pad_offset_for(cfgs, listed.vessel, listed.pad)
            assert env.pad == pad and tuple(env.pad_offset_m) == offset
            _, info = env.reset(seed=listed.episode_seed)
            key = p8_adapter.realization_key(
                listed.ss,
                listed.heading_deg,
                listed.speed_kn,
                listed.vessel,
                listed.realization_seed,
            )
            assert info["realization_key"] == key
            assert float(env.record.t0_model_s) == listed.t0_model_s
    finally:
        env.close()
