"""Render cost per frame (Phase 2 throughput): camera-only vs the full scene, at several sizes.

One environment, one listed episode, a fixed controller flying it (so the deck, the drone and
the view change as in a real episode). After ``env.reset()`` and after every ``env.step()``
one frame is rendered and timed:

* ``camera``: ``render_pose`` from the drone's physics pose over P8's untouched scene;
* ``full``: :func:`ila.render.scene.attach` once per reset, then ``Scene.render()`` (re-pose the
  twin + render).

Only the render call is timed (``time.perf_counter``, wall seconds). Sizes are square
(``width_px = height_px``); the texture and every other render setting are the committed ones.
Nothing here changes physics (P1-D1). Lengths metres model scale (lambda = 1/25).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pybullet as pyb
from numpy.typing import NDArray

from ila import p8_adapter
from ila.render import scene
from ila.render.camera import render_pose
from ila.render.config import RenderConfig, load_render_config

__all__ = ["render_frame_times", "sized_config"]


def sized_config(size_px: int, base: RenderConfig | None = None) -> RenderConfig:
    """Return the committed render config with a square ``size_px`` image."""
    cfg = load_render_config() if base is None else base
    return replace(cfg, camera=replace(cfg.camera, width_px=size_px, height_px=size_px))


def render_frame_times(
    cfgs: p8_adapter.EvalConfigs,
    listed: p8_adapter.ListedEpisode,
    policy: Any,
    *,
    mode: str,
    size_px: int,
    n_frames: int,
) -> NDArray[np.float64]:
    """Time ``n_frames`` renders along controller-flown episodes (re-reset when one ends).

    Args:
        cfgs: P8's configs.
        listed: The episode (re-flown from its reset whenever it ends).
        policy: A P8 ``EvalPolicy`` (``reset(seed, None)`` per episode, ``act(obs)``).
        mode: ``"camera"`` or ``"full"``.
        size_px: Square image size, px.
        n_frames: Frames to time.

    Returns:
        ``float64[n_frames]`` seconds per frame (wall clock).

    Raises:
        ValueError: On an unknown mode.
    """
    if mode not in ("camera", "full"):
        raise ValueError(f"mode must be 'camera' or 'full', got {mode!r}")
    cfg = sized_config(size_px)
    env = p8_adapter.make_env(cfgs, listed)
    times: list[float] = []
    try:
        while len(times) < n_frames:
            p8_adapter.set_motion(env, cfgs, listed)
            obs, _ = env.reset(seed=listed.episode_seed)
            policy.reset(listed.episode_seed, None)
            sc = scene.attach(env, cfg) if mode == "full" else None
            h = p8_adapter.scene_handles(env)
            done = False
            while not done and len(times) < n_frames:
                start = time.perf_counter()
                if sc is not None:
                    sc.render()
                else:
                    pos, quat = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
                    render_pose(h.client, pos, quat, cfg.camera)
                times.append(time.perf_counter() - start)
                obs, _, terminated, truncated, _ = env.step(
                    np.asarray(policy.act(obs), dtype=np.float64)
                )
                done = bool(terminated or truncated)
    finally:
        env.close()
    return np.asarray(times, dtype=np.float64)
