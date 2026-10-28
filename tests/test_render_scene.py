"""Phase 1 render scene: no inertial reference, no world-fixed visual, timing, determinism."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pybullet as pyb
import pytest

import ila
from ila import p8_adapter
from ila.obs.split import load_contract
from ila.render import camera, scene
from ila.render.config import load_render_config
from render_support import airborne_actions, descent_actions, fly, listed, new_env, train_listed

RENDER_SRC = Path(ila.__file__).resolve().parent / "render"


# --------------------------------------------------------------------------- static guards


def test_render_never_steps_or_changes_dynamics() -> None:
    forbidden = {"stepSimulation", "changeDynamics", "resetBaseVelocity", "applyExternalForce"}
    offenders = []
    for path in sorted(RENDER_SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append(f"{path.name}:{node.lineno}:{node.attr}")
    assert offenders == []


# --------------------------------------------------------------------------- scene bodies


@pytest.mark.pybullet
def test_attach_bodies() -> None:
    row = train_listed("SS5")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        sc = scene.attach(env)
        h, c = sc.handles, sc.handles.client
        assert (h.plane, h.drone, h.plate, h.pad_marker, sc.twin) == (0, 1, 2, 3, 4)
        assert pyb.getCollisionShapeData(sc.twin, -1, physicsClientId=c) == ()
        assert pyb.getDynamicsInfo(sc.twin, -1, physicsClientId=c)[0] == 0.0
        assert pyb.getNumJoints(sc.twin, physicsClientId=c) == 0
        marker = pyb.getVisualShapeData(h.pad_marker, physicsClientId=c)[0]
        assert marker[7][3] == 0.0
        # Bodies with a drawable visual (alpha > 0): ground, drone (never seen by its own
        # camera, P1-D1), plate, twin. Nothing else.
        visible = {
            b
            for b in range(pyb.getNumBodies(physicsClientId=c))
            for v in pyb.getVisualShapeData(b, physicsClientId=c)
            if v[7][3] > 0.0
        }
        assert visible == {h.plane, h.drone, h.plate, sc.twin}
    finally:
        env.close()


# --------------------------------------------------------------------------- background


def _ground_pixels_after_translation(
    attach: bool,
    delta: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple:
    row = listed("static")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        h = p8_adapter.scene_handles(env)
        cfg = load_render_config().camera
        if attach:
            sc = scene.attach(env)
            sc.pre_render()
        pos, _ = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
        # Non-level attitudes too (test-only env): set the drone's orientation, radians.
        quat = pyb.getQuaternionFromEuler(list(rpy))
        pyb.resetBasePositionAndOrientation(h.drone, pos, quat, physicsClientId=h.client)
        pos, quat = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
        fa, a = camera.render_pose(h.client, pos, quat, cfg, segmentation=True)
        # Translate the drone itself (test-only env), then render through the same path.
        moved = np.add(pos, delta)
        pyb.resetBasePositionAndOrientation(h.drone, moved, quat, physicsClientId=h.client)
        pos2, quat2 = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
        fb, b = camera.render_pose(h.client, pos2, quat2, cfg, segmentation=True)
        assert a.segmentation is not None and b.segmentation is not None
        ground = (a.segmentation == h.plane) & (b.segmentation == h.plane)
        return fa.rgb, fb.rgb, ground
    finally:
        env.close()


@pytest.mark.pybullet
@pytest.mark.parametrize("delta", [(0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.05)])
@pytest.mark.parametrize(
    "rpy", [(0.0, 0.0, 0.0), (0.3, -0.3, 1.2), (-0.25, 0.2, -2.0), (0.1, 0.35, 0.6)]
)
def test_background_identical_under_translation(
    delta: tuple[float, float, float], rpy: tuple[float, float, float]
) -> None:
    a, b, ground = _ground_pixels_after_translation(True, delta, rpy)
    assert ground.sum() > 0.5 * ground.size
    np.testing.assert_array_equal(a[ground], b[ground])
    # Uniform: one colour across every ground pixel.
    assert len(np.unique(a[ground], axis=0)) == 1


@pytest.mark.pybullet
def test_background_test_has_power_on_p8_checkerboard() -> None:
    a, b, ground = _ground_pixels_after_translation(False, (0.05, 0.0, 0.0))
    assert (a[ground] != b[ground]).any()


# --------------------------------------------------------------------------- no world-fixed visual


@pytest.mark.pybullet
@pytest.mark.slow
def test_no_world_fixed_visual_on_moving_deck() -> None:
    row = train_listed("SS5")
    env = new_env(row)
    seen: set[int] = set()
    twin_frames = 0

    def check(t: int, out: tuple[scene.Frame, scene.RenderMeta]) -> None:
        nonlocal twin_frames
        sc = holder["scene"]
        h, c = sc.handles, sc.handles.client
        frame = out[1]
        assert frame.segmentation is not None
        ids = set(np.unique(frame.segmentation).tolist())
        seen.update(ids)
        assert ids <= {h.plane, h.plate, sc.twin}, (t, ids)
        twin_frames += sc.twin in ids
        # Twin pose = plate physics pose at render time: position exact, quaternion to a few ulp
        # (PyBullet renormalises the quaternion it is given).
        twin_pos, twin_q = pyb.getBasePositionAndOrientation(sc.twin, physicsClientId=c)
        plate_pos, plate_q = pyb.getBasePositionAndOrientation(h.plate, physicsClientId=c)
        assert twin_pos == plate_pos
        np.testing.assert_allclose(twin_q, plate_q, rtol=0, atol=1e-12)

    holder: dict[str, scene.Scene] = {}
    try:
        env.reset(seed=row.episode_seed)
        holder["scene"] = scene.attach(env)
        check(0, holder["scene"].render_with_meta(segmentation=True))
        for t, action in enumerate(airborne_actions(200), start=1):
            result = env.step(action)
            check(t, holder["scene"].render_with_meta(segmentation=True))
            if result[2] or result[3]:
                break
    finally:
        env.close()
    assert twin_frames > 150
    assert 1 not in seen and 3 not in seen  # drone and P8's marker never appear


# --------------------------------------------------------------------------- timing


@pytest.mark.pybullet
def test_frame_and_obs_share_physics_step_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame t is rendered after exactly 8 t stepSimulation calls, counted independently.

    The count in RenderMeta and ``p8_adapter.physics_step_count`` both read P8's substep
    counter, so their equality alone is tautological. The independent guards are (1) a wrapper
    around ``pybullet.stepSimulation`` (P8 calls it as ``pyb.stepSimulation``, looked up at call
    time) counting real calls since reset, and (2) the frame's view matrix rebuilt from
    ``obs[t]``'s attitude: the camera pose belongs to the same physics state as the observation.
    """
    calls = {"n": 0}
    real_step = pyb.stepSimulation

    def counting_step(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(pyb, "stepSimulation", counting_step)
    row = train_listed("SS5")
    env = new_env(row)
    attitude = load_contract().layout["attitude"]
    cfg = load_render_config().camera
    try:
        obs, _ = env.reset(seed=row.episode_seed)
        calls["n"] = 0  # reset itself must not step; checked by frame 0 below
        sc = scene.attach(env)
        for t, action in enumerate([None, *descent_actions(30)]):
            if action is not None:
                obs, *_ = env.step(action)
            before = calls["n"]
            frame, meta = sc.render_with_meta()
            assert calls["n"] == before, "rendering called stepSimulation"
            assert calls["n"] == 8 * t
            assert meta.physics_step_count == p8_adapter.physics_step_count(env) == calls["n"]
            assert frame.rgb.shape == (96, 96, 3)
            # The frame's camera is the obs' attitude: rebuild the view from obs[attitude].
            rpy = np.asarray(obs[attitude[0] : attitude[0] + attitude[1]], dtype=np.float64)
            pos, _ = pyb.getBasePositionAndOrientation(sc.handles.drone, physicsClientId=0)
            quat = pyb.getQuaternionFromEuler(rpy.tolist())
            expected = camera.view_matrix(camera.camera_pose(pos, quat, cfg))
            np.testing.assert_allclose(
                meta.view, np.asarray(expected).reshape(4, 4).T, rtol=0, atol=1e-5
            )
    finally:
        env.close()


# --------------------------------------------------------------------------- student-facing type


def test_student_facing_frame_is_pixels_only() -> None:
    """``Frame`` (what Scene.render returns) holds one array, uint8 [H, W, 3], and nothing else."""
    import dataclasses
    import typing

    fields = dataclasses.fields(camera.Frame)
    assert [f.name for f in fields] == ["rgb"]
    hints = typing.get_type_hints(camera.Frame)
    assert set(hints) == {"rgb"}
    for name in ("view", "projection", "segmentation", "physics_step_count", "meta"):
        assert not hasattr(camera.Frame, name)
    assert typing.get_type_hints(scene.Scene.render)["return"] is camera.Frame
    assert scene.Frame is camera.Frame
    frame = camera.Frame(rgb=np.zeros((96, 96, 3), np.uint8))
    assert not frame.rgb.flags.writeable
    for bad in (np.zeros((96, 96, 3), np.float32), np.zeros((96, 96, 4), np.uint8)):
        with pytest.raises(ValueError):
            camera.Frame(rgb=bad)
    with pytest.raises(TypeError):
        camera.Frame(rgb=np.zeros((96, 96, 3), np.uint8), view=np.eye(4))  # type: ignore[call-arg]


@pytest.mark.pybullet
def test_scene_render_returns_pixels_only() -> None:
    row = train_listed("SS5")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        out = scene.attach(env).render()
    finally:
        env.close()
    assert type(out) is camera.Frame
    assert out.rgb.dtype == np.uint8 and out.rgb.shape == (96, 96, 3)
    assert not hasattr(scene.Scene, "plate_pose")


# --------------------------------------------------------------------------- determinism


@pytest.mark.pybullet
@pytest.mark.slow
def test_same_episode_renders_byte_identical() -> None:
    row = train_listed("SS5", 1)
    actions = descent_actions(80)
    runs = []
    for _ in range(2):
        frames: list[bytes] = []
        env = new_env(row)
        try:
            out = fly(
                env,
                row,
                actions,
                "full",
                on_frame=lambda _t, f, fr=frames: fr.append(f.rgb.tobytes()),
            )
        finally:
            env.close()
        runs.append((frames, out["digests"]))
    assert runs[0] == runs[1]
    assert len(runs[0][0]) > 40


@pytest.mark.pybullet
def test_segmentation_flag_does_not_change_rgb() -> None:
    row = train_listed("SS5")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        sc = scene.attach(env)
        env.step(np.zeros(3))
        plain = sc.render()
        frame_a, meta_a = sc.render_with_meta()
        frame_b, meta_b = sc.render_with_meta(segmentation=True)
    finally:
        env.close()
    np.testing.assert_array_equal(plain.rgb, frame_a.rgb)
    np.testing.assert_array_equal(plain.rgb, frame_b.rgb)
    assert meta_a.segmentation is None and meta_b.segmentation is not None
