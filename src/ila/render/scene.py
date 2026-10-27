"""Visual-only scene edits and the per-step render (P1-D1).

:func:`attach` runs after **every** ``env.reset()`` (``resetSimulation`` destroys every body,
texture binding and visual change) and before the first render of the episode:

1. **Ground:** P8's checkerboard texture is removed and the plane recoloured to a uniform
   ``ground_rgba`` (no inertial reference). Its collision shape is P8's and untouched.
2. **P8's pad marker** (a static, world-fixed disc) is hidden with alpha 0; TinyRenderer then
   draws it neither in RGB nor in the segmentation mask.
3. **Visual twin:** one new body, mass 0, **no collision shape**, whose visual is a
   ``GEOM_MESH`` quad (two triangles, explicit UVs) covering the plate's top face
   1.0 mm above it, textured with the procedural helipad (:mod:`ila.render.texture`). It is
   created after P8's bodies, so P8's body ids are unchanged.

:meth:`Scene.render` re-poses the twin to the plate's **physics** pose (``pre_render``) and
renders the body-fixed camera from the drone's physics pose. Call it only after
``env.reset()``/``attach`` or after ``env.step()`` has returned: the frame then shares its
``stepSimulation`` count with the observation the env just returned.

Output split (privileged-leak audit, P1-D1): :meth:`Scene.render` returns a
:class:`~ila.render.camera.Frame`, pixels only, which is the one render output a student may
see. :meth:`Scene.render_with_meta` also returns a :class:`~ila.render.camera.RenderMeta`
(view and projection matrices, the step count, the optional segmentation mask). That carries
the drone's world position, the clock and a perfect deck mask, so it is never a student input:
runners use it for timing checks, and tests and audits for geometry.

Nothing in this module calls ``stepSimulation`` or ``changeDynamics``; every PyBullet body id
comes from :func:`ila.p8_adapter.scene_handles`. Lengths metres model scale (lambda = 1/25),
world frame unless stated.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import pybullet as pyb
from numpy.typing import NDArray

from ila import p8_adapter
from ila.render.camera import Frame, RenderMeta, render_pose
from ila.render.config import MarkingConfig, RenderConfig, load_render_config
from ila.render.texture import helipad_texture, texture_png

__all__ = [
    "TEXTURE_CACHE_DIR",
    "TWIN_LIFT_M",
    "Frame",
    "RenderMeta",
    "Scene",
    "StaleSceneError",
    "attach",
]

#: Where the helipad PNG is written for ``loadTexture`` (gitignored).
TEXTURE_CACHE_DIR: Final[Path] = p8_adapter.REPO_ROOT / "artifacts" / "render"

#: Height of the twin quad above the plate's top surface, plate frame, m model.
TWIN_LIFT_M: Final[float] = 1.0e-3


class StaleSceneError(RuntimeError):
    """The environment was reset since :func:`attach`; call :func:`attach` again."""


@functools.cache
def _texture(marking: MarkingConfig, half_extents_m: tuple[float, float]) -> NDArray[np.uint8]:
    """Return the (cached, read-only) helipad texture for a plate face."""
    tex = helipad_texture(marking, half_extents_m)
    tex.setflags(write=False)
    return tex


class Scene:
    """The render-side scene of one episode; build it with :func:`attach`.

    Attributes:
        env: The P8 environment.
        cfg: The render config.
        handles: P8's body ids for this episode.
        twin: The visual twin's body id.
        texture_id: PyBullet texture id of the helipad.
    """

    def __init__(
        self,
        env: p8_adapter.DeckLandingAviary,
        cfg: RenderConfig,
        handles: p8_adapter.SceneHandles,
        twin: int,
        texture_id: int,
    ) -> None:
        """Hold the ids; use :func:`attach` rather than calling this directly."""
        self.env = env
        self.cfg = cfg
        self.handles = handles
        self.twin = twin
        self.texture_id = texture_id
        self._n_bodies = int(pyb.getNumBodies(physicsClientId=handles.client))

    def _check_fresh(self) -> None:
        """Raise if the env was reset after :func:`attach` (the twin no longer exists)."""
        if pyb.getNumBodies(physicsClientId=self.handles.client) != self._n_bodies:
            raise StaleSceneError("the environment was reset; call attach(env) again")

    def _plate_pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the plate's physics pose: position (m model, world), quaternion (x, y, z, w)."""
        pos, quat = pyb.getBasePositionAndOrientation(
            self.handles.plate, physicsClientId=self.handles.client
        )
        return tuple(pos), tuple(quat)

    def pre_render(self) -> None:
        """Re-pose the twin to the plate's current physics pose (visual only).

        Raises:
            StaleSceneError: If the env was reset since :func:`attach`.
        """
        self._check_fresh()
        pos, quat = self._plate_pose()
        pyb.resetBasePositionAndOrientation(
            self.twin, list(pos), list(quat), physicsClientId=self.handles.client
        )

    def render(self) -> Frame:
        """Render the student-facing frame (pixels only) from the current physics state.

        Call only after ``env.reset()`` + :func:`attach` or after ``env.step()`` returned.

        Returns:
            The :class:`~ila.render.camera.Frame`.

        Raises:
            StaleSceneError: If the env was reset since :func:`attach`.
        """
        frame, _ = self.render_with_meta()
        return frame

    def render_with_meta(self, *, segmentation: bool = False) -> tuple[Frame, RenderMeta]:
        """Render the frame and its never-student metadata (timing checks, tests, audits).

        The pixels are the same bytes :meth:`render` returns.

        Args:
            segmentation: Also compute the body-id mask.

        Returns:
            ``(frame, meta)``; ``meta.physics_step_count`` is the env's ``stepSimulation``
            count since reset at render time.

        Raises:
            StaleSceneError: If the env was reset since :func:`attach`.
        """
        self.pre_render()
        pos, quat = pyb.getBasePositionAndOrientation(
            self.handles.drone, physicsClientId=self.handles.client
        )
        frame, meta = render_pose(
            self.handles.client, pos, quat, self.cfg.camera, segmentation=segmentation
        )
        return frame, replace(meta, physics_step_count=p8_adapter.physics_step_count(self.env))


def attach(
    env: p8_adapter.DeckLandingAviary,
    cfg: RenderConfig | None = None,
    *,
    cache_dir: Path = TEXTURE_CACHE_DIR,
) -> Scene:
    """Apply the visual-only scene edits to a freshly reset environment.

    Args:
        env: A P8 environment, reset and not yet stepped.
        cfg: Render config; the committed default look when ``None``.
        cache_dir: Where the helipad PNG is written for ``loadTexture``.

    Returns:
        The episode's :class:`Scene`.

    Raises:
        RuntimeError: If the env has been stepped since its last reset.
    """
    if p8_adapter.physics_step_count(env) != 0:
        raise RuntimeError("attach() must run right after env.reset(), before any env.step()")
    cfg = load_render_config() if cfg is None else cfg
    h = p8_adapter.scene_handles(env)
    c = h.client

    # 1. Ground: texture removed, uniform colour.
    pyb.changeVisualShape(
        h.plane,
        -1,
        textureUniqueId=-1,
        rgbaColor=list(cfg.background.ground_rgba),
        physicsClientId=c,
    )
    # 2. P8's static pad marker: hidden.
    pyb.changeVisualShape(
        h.pad_marker, -1, rgbaColor=[*p8_adapter.PAD_MARKER_RGBA[:3], 0.0], physicsClientId=c
    )
    # 3. Visual twin: collision-less, mass-0 textured quad on the plate's top face.
    hx, hy, _ = h.plate_half_extents_m
    tex = _texture(cfg.marking, (hx, hy))
    texture_id = int(pyb.loadTexture(str(texture_png(tex, cache_dir)), physicsClientId=c))
    z = TWIN_LIFT_M
    visual = int(
        pyb.createVisualShape(
            pyb.GEOM_MESH,
            vertices=[[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]],
            indices=[0, 1, 2, 0, 2, 3],
            uvs=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            normals=[[0.0, 0.0, 1.0]] * 4,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=c,
        )
    )
    pos, quat = pyb.getBasePositionAndOrientation(h.plate, physicsClientId=c)
    twin = int(
        pyb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=list(pos),
            baseOrientation=list(quat),
            physicsClientId=c,
        )
    )
    pyb.changeVisualShape(twin, -1, textureUniqueId=texture_id, physicsClientId=c)
    if twin <= max(h.plane, h.drone, h.plate, h.pad_marker):
        raise RuntimeError(f"twin id {twin} is not after P8's bodies")
    return Scene(env, cfg, h, twin, texture_id)
