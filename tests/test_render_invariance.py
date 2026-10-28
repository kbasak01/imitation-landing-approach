"""Phase 1: rendering, the visual twin and the ground recolour change no physics, bit for bit.

Three runs of the same listed moving-deck episode with the same fixed action sequence:
``off`` (no render), ``camera`` (TinyRenderer frame after reset and after every step, P8's scene
untouched) and ``full`` (``scene.attach``: ground recolour + P8 marker hidden + visual twin,
twin re-posed before every frame). Per step, the drone and plate pose and velocity, the
observation, the reward, the termination flags and the info must be byte-identical. On a miss
the assertion names the first differing step (index 0 = after reset), which is where the
bisection off -> camera -> full starts.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ila.render import scene
from render_support import (
    airborne_actions,
    descent_actions,
    first_difference,
    fly,
    new_env,
    train_listed,
)

pytestmark = pytest.mark.pybullet

MODES = ("off", "camera", "full")


def _three_runs(row, actions):  # type: ignore[no-untyped-def]
    runs = {}
    for mode in MODES:
        env = new_env(row)
        try:
            start = time.perf_counter()
            runs[mode] = fly(env, row, actions, mode)
            runs[mode]["seconds"] = time.perf_counter() - start
        finally:
            env.close()
    return runs


def _assert_identical(runs):  # type: ignore[no-untyped-def]
    ref = runs["off"]["digests"]
    for mode in ("camera", "full"):
        diff = first_difference(ref, runs[mode]["digests"])
        assert diff is None, f"render mode {mode!r}: first differing step {diff}"


@pytest.mark.slow
def test_airborne_300_steps_identical() -> None:
    row = train_listed("SS5")
    runs = _three_runs(row, airborne_actions(300))
    for mode in MODES:
        assert runs[mode]["steps"] == 300, (mode, runs[mode]["info"])
        assert runs[mode]["first_contact"] is None
    _assert_identical(runs)
    print({m: round(r["seconds"], 2) for m, r in runs.items()})


@pytest.mark.slow
def test_descent_to_contact_identical() -> None:
    row = train_listed("SS5", 1)  # row 0 strikes the plate edge before in_contact; row 1 at 49
    runs = _three_runs(row, descent_actions(150))
    assert runs["off"]["first_contact"] is not None, "the scripted descent never touched the deck"
    for mode in MODES:
        assert runs[mode]["first_contact"] == runs["off"]["first_contact"]
        assert runs[mode]["steps"] == runs["off"]["steps"]
    _assert_identical(runs)


@pytest.mark.slow
def test_reused_env_reattach_identical() -> None:
    """P8's runner reuses one env across episodes: a second reset must re-attach cleanly."""
    row = train_listed("SS5", 1)
    actions = descent_actions(120)
    results = {}
    for mode in ("off", "full"):
        env = new_env(row)
        try:
            fly(env, row, actions[:40], mode)  # a first, partial episode in the same env
            results[mode] = fly(env, row, actions, mode)
        finally:
            env.close()
    diff = first_difference(results["off"]["digests"], results["full"]["digests"])
    assert diff is None, f"first differing step {diff}"


def test_stale_scene_raises_after_reset() -> None:
    row = train_listed("SS5")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        sc = scene.attach(env)
        sc.render()
        env.step(np.zeros(3))
        with pytest.raises(RuntimeError, match="reset"):
            scene.attach(env)  # attach only right after a reset
        env.reset(seed=row.episode_seed)
        with pytest.raises(scene.StaleSceneError):
            sc.render()
    finally:
        env.close()


@pytest.mark.slow
def test_negative_control_collision_twin_is_detected() -> None:
    """The digest comparison has power: a twin WITH a collision shape changes the contact."""
    import pybullet as pyb

    from ila import p8_adapter

    row = train_listed("SS5")
    actions = descent_actions(150)
    env = new_env(row)
    try:
        ref = fly(env, row, actions, "off")
    finally:
        env.close()
    env = new_env(row)
    bad: dict[str, int] = {}

    def repose(t: int, frame: object) -> None:
        del frame
        h = p8_adapter.scene_handles(env)
        pos, quat = pyb.getBasePositionAndOrientation(h.plate, physicsClientId=h.client)
        if t == 0:
            col = pyb.createCollisionShape(
                pyb.GEOM_BOX, halfExtents=[0.4, 0.26, 0.0005], physicsClientId=h.client
            )
            bad["id"] = pyb.createMultiBody(0.0, col, -1, pos, quat, physicsClientId=h.client)
        pyb.resetBasePositionAndOrientation(
            bad["id"], [pos[0], pos[1], pos[2] + 0.004], quat, physicsClientId=h.client
        )

    try:
        perturbed = fly(env, row, actions, "camera", on_frame=repose)
    finally:
        env.close()
    assert first_difference(ref["digests"], perturbed["digests"]) is not None
