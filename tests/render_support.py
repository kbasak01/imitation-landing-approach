"""Shared helpers for the render tests: listed episodes, scripted actions, per-step digests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pybullet as pyb

from ila import p8_adapter
from ila.obs.split import load_contract
from ila.render import camera, scene
from ila.render.config import load_render_config

IN_CONTACT = load_contract().layout["in_contact"][0]


#: The committed train-pool list the Phase 1 fixtures fly (P1-D2 obligation 1): ``demos`` stream.
TRAIN_FIXTURE_SPLIT = "demos_0-1999"


def train_listed(ss: str = "SS5", index: int = 0) -> p8_adapter.ListedEpisode:
    """Return the ``index``-th row at sea state ``ss`` of the committed ``demos`` split.

    A ``train``-pool realization (P1-D2 obligation 1): scripted-action tests fly it instead of
    a frozen-list row. SS5 is the roughest sea state in the train pool (SS6 is held out).
    """
    from ila.data.episodes import read_split

    rows = [r for r in read_split(TRAIN_FIXTURE_SPLIT) if r.ss == ss]
    return rows[index]


def listed(list_name: str, ss: str | None = None, index: int = 0) -> p8_adapter.ListedEpisode:
    """Return a frozen-list row. Only for decks the train pool lacks (``static``)."""
    rows = p8_adapter.read_list(p8_adapter.list_path(list_name))
    if ss is not None:
        rows = [r for r in rows if r.ss == ss]
    return rows[index]


def new_env(row: p8_adapter.ListedEpisode) -> p8_adapter.DeckLandingAviary:
    return p8_adapter.make_env(p8_adapter.load_eval_configs(), row)


def airborne_actions(n: int = 300) -> np.ndarray:
    """Gentle lateral weave at zero vertical command: stays airborne for 300 steps."""
    t = np.arange(n, dtype=np.float64)
    return np.stack(
        [0.15 * np.sin(2 * np.pi * t / 90.0), 0.15 * np.cos(2 * np.pi * t / 70.0), 0.0 * t], 1
    )


def descent_actions(n: int = 150) -> np.ndarray:
    """Noisy descent at ~0.5 m/s model scale: reaches deck contact (train SS5 fixture)."""
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.0, 0.2, (n, 3)) + np.array([0.0, 0.0, -0.35]), -1.0, 1.0)


def state_bytes(h: p8_adapter.SceneHandles) -> bytes:
    """Drone and plate pose and velocity, float64 bytes (read only)."""
    out = []
    for body in (h.drone, h.plate):
        pos, quat = pyb.getBasePositionAndOrientation(body, physicsClientId=h.client)
        lin, ang = pyb.getBaseVelocity(body, physicsClientId=h.client)
        out.append(np.asarray([*pos, *quat, *lin, *ang], dtype=np.float64).tobytes())
    return b"".join(out)


def step_digest(h: p8_adapter.SceneHandles, step: tuple[Any, ...]) -> str:
    obs, reward, terminated, truncated, info = step
    d = hashlib.sha256(state_bytes(h))
    d.update(np.asarray(obs).tobytes())
    d.update(np.float64(reward).tobytes())
    d.update(bytes([bool(terminated), bool(truncated)]))
    d.update(repr(sorted((k, repr(v)) for k, v in info.items())).encode())
    return d.hexdigest()


RenderMode = str  # "off" | "camera" | "full"


def fly(
    env: p8_adapter.DeckLandingAviary,
    row: p8_adapter.ListedEpisode,
    actions: Sequence[np.ndarray],
    mode: RenderMode,
    on_frame: Callable[[int, Any], None] | None = None,
) -> dict[str, Any]:
    """Reset, (maybe) attach, and fly ``actions``, rendering after reset and after each step.

    Returns per-step digests (index 0 = after reset), the number of steps flown, the first step
    with ``in_contact`` and the final info.
    """
    obs, info = env.reset(seed=row.episode_seed)
    h = p8_adapter.scene_handles(env)
    sc = scene.attach(env) if mode == "full" else None
    cam_cfg = load_render_config().camera

    def render(t: int) -> None:
        if mode == "off":
            return
        if sc is not None:
            frame: Any = sc.render()
        else:
            pos, quat = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
            frame, _ = camera.render_pose(h.client, pos, quat, cam_cfg)
        if on_frame is not None:
            on_frame(t, frame)

    digests = [step_digest(h, (obs, 0.0, False, False, info))]
    render(0)
    first_contact = None
    steps = 0
    for t, action in enumerate(actions, start=1):
        result = env.step(np.asarray(action))
        steps = t
        digests.append(step_digest(h, result))
        render(t)
        if first_contact is None and result[0][IN_CONTACT] > 0.5:
            first_contact = t
        info = result[4]
        if result[2] or result[3]:
            break
    return {"digests": digests, "steps": steps, "first_contact": first_contact, "info": info}


def first_difference(a: Sequence[str], b: Sequence[str]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
