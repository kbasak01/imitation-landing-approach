"""Body-fixed downward camera, rendered with PyBullet's TinyRenderer.

The camera is rigidly mounted on the drone. With the drone's physics pose ``(p, R)`` (``R``
body-to-world), the camera offset ``o``, look axis ``l`` and up axis ``u`` in the body frame
(after the mount tilt ``M``):

* ``eye = p + R o``; ``target = eye + R M l``; ``up = R M u``.

Defaults: ``o = (0, 0, -0.02)`` m, ``l = (0, 0, -1)`` (body -z), ``u = (1, 0, 0)`` (image up
= body +x, forward). Image right is therefore body -y (starboard). Vertical FOV 90°, aspect
1, near 0.01 m, far 20 m (P1-D1), 96 x 96 RGB, ``ER_TINY_RENDERER``, shadows off, fixed light
direction (world frame). Lengths metres model scale (lambda = 1/25), world frame.

A frame must be rendered strictly after ``env.reset()`` or ``env.step()`` has returned, never
inside a physics substep; the caller (``ila.render.scene.Scene.render``) records the physics
step count with it. Nothing here calls ``stepSimulation`` or changes dynamics.

Pixel convention (:func:`project_points`), **measured** for TinyRenderer in PyBullet 3.2.7
(``tests/test_render_geometry.py``): ``(col, row)`` in index space with

* ``col = (ndc_x + 1) / 2 * W`` and ``row = (1 - ndc_y) / 2 * H - 1``.

TinyRenderer samples at integer viewport coordinates and then flips the rows. That puts the
optical axis at pixel ``(col, row) = (W/2, H/2 - 1)``, i.e. (48, 47) at 96 px, half a pixel off
the OpenGL pixel-centre convention (47.5, 47.5) in each axis. Row 0 is the top of the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as pyb
from numpy.typing import ArrayLike, NDArray

from ila.render.config import CameraConfig

__all__ = [
    "CameraPose",
    "Frame",
    "RenderMeta",
    "camera_pose",
    "project_points",
    "projection_matrix",
    "render_pose",
    "view_matrix",
]


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is and where it looks, world frame.

    Attributes:
        eye: Camera centre, m model, world.
        target: A point one metre along the optical axis, m model, world.
        up: Image-up direction, unit, world.
    """

    eye: NDArray[np.float64]
    target: NDArray[np.float64]
    up: NDArray[np.float64]


@dataclass(frozen=True)
class Frame:
    """The student-facing camera frame: pixels and nothing else.

    This is the only render output that may enter a student input, a demo store's student
    columns or a training batch (il-conventions). It deliberately has no matrices, no step
    count and no mask: those live in :class:`RenderMeta`.

    Attributes:
        rgb: ``uint8[H, W, 3]`` RGB, row 0 at the top (read-only).
    """

    rgb: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Validate and freeze the pixels.

        Raises:
            ValueError: If ``rgb`` is not ``uint8[H, W, 3]``.
        """
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Frame.rgb must be uint8[H, W, 3], got {rgb.dtype} {rgb.shape}")
        rgb = np.ascontiguousarray(rgb)
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)


@dataclass(frozen=True)
class RenderMeta:
    """Everything about a frame that is not pixels; NEVER a student input.

    The view matrix gives the camera's world position (``eye = -R^T t``), which together
    with the plate pose is the privileged relative position and height; the step count is
    the episode clock; the segmentation mask is a perfect deck mask. Runners use this for
    timing checks, tests and audits only, and nothing here may reach a student, a demo
    store's student columns or a batch.

    Attributes:
        view: 4 x 4 view matrix (world -> camera), row-major (``view @ [x, y, z, 1]``).
        projection: 4 x 4 OpenGL projection matrix, row-major.
        segmentation: ``int32[H, W]`` body unique id per pixel (-1 where nothing is drawn),
            or ``None`` when not requested.
        physics_step_count: ``stepSimulation`` calls since reset when the frame was taken
            (``8 t`` at control step ``t``, uncapped), or ``None`` from :func:`render_pose`,
            which knows no environment.
    """

    view: NDArray[np.float64]
    projection: NDArray[np.float64]
    segmentation: NDArray[np.int32] | None
    physics_step_count: int | None = None


def _mount_rotation(tilt_deg: float) -> NDArray[np.float64]:
    """Return the mount rotation about body y; positive tilt pitches body -z toward +x.

    Args:
        tilt_deg: Mount tilt, degrees.

    Returns:
        3 x 3 rotation (body -> tilted body), dimensionless.
    """
    t = np.radians(tilt_deg)
    c, s = float(np.cos(t)), float(np.sin(t))
    # R_y(-t): maps (0, 0, -1) to (sin t, 0, -cos t), i.e. toward +x for t > 0.
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def camera_pose(position_m: ArrayLike, quaternion: ArrayLike, cfg: CameraConfig) -> CameraPose:
    """Return the camera pose for a drone pose.

    Args:
        position_m: Drone body-origin position, m model, world.
        quaternion: Drone orientation ``(x, y, z, w)``, body -> world.
        cfg: Camera config.

    Returns:
        The :class:`CameraPose`.
    """
    p = np.asarray(position_m, dtype=np.float64).reshape(3)
    q = [float(v) for v in np.asarray(quaternion, dtype=np.float64).reshape(4)]
    rot = np.asarray(pyb.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3, 3)
    mount = _mount_rotation(cfg.mount_tilt_deg)
    eye = p + rot @ np.asarray(cfg.offset_body_m, dtype=np.float64)
    look = rot @ mount @ np.asarray(cfg.look_axis_body, dtype=np.float64)
    up = rot @ mount @ np.asarray(cfg.up_axis_body, dtype=np.float64)
    return CameraPose(eye=eye, target=eye + look, up=up)


def _as_matrix(flat: Any) -> NDArray[np.float64]:
    """Convert PyBullet's column-major 16-tuple into a row-major 4 x 4 array."""
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def view_matrix(pose: CameraPose) -> tuple[float, ...]:
    """Return PyBullet's (column-major) view matrix for a camera pose."""
    return tuple(
        pyb.computeViewMatrix(
            [float(v) for v in pose.eye],
            [float(v) for v in pose.target],
            [float(v) for v in pose.up],
        )
    )


def projection_matrix(cfg: CameraConfig) -> tuple[float, ...]:
    """Return PyBullet's (column-major) projection matrix for the camera intrinsics."""
    return tuple(
        pyb.computeProjectionMatrixFOV(cfg.fov_vertical_deg, cfg.aspect, cfg.near_m, cfg.far_m)
    )


def render_pose(
    client: int,
    position_m: ArrayLike,
    quaternion: ArrayLike,
    cfg: CameraConfig,
    *,
    segmentation: bool = False,
) -> tuple[Frame, RenderMeta]:
    """Render one frame from the camera on a drone at a given pose.

    Reads no physics state and changes none: ``getCameraImage`` only rasterises the visual
    shapes at their current poses.

    Args:
        client: PyBullet physics client id (DIRECT).
        position_m: Drone body-origin position, m model, world.
        quaternion: Drone orientation ``(x, y, z, w)``.
        cfg: Camera config.
        segmentation: Also return the body-id mask (tests, audits).

    Returns:
        ``(frame, meta)``: the pixels, and the never-student :class:`RenderMeta`.
    """
    pose = camera_pose(position_m, quaternion, cfg)
    view = view_matrix(pose)
    proj = projection_matrix(cfg)
    flags = 0 if segmentation else pyb.ER_NO_SEGMENTATION_MASK
    _, _, rgba, _, seg = pyb.getCameraImage(
        cfg.width_px,
        cfg.height_px,
        viewMatrix=list(view),
        projectionMatrix=list(proj),
        shadow=int(cfg.shadow),
        lightDirection=list(cfg.light_direction),
        renderer=pyb.ER_TINY_RENDERER,
        flags=flags,
        physicsClientId=client,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(cfg.height_px, cfg.width_px, 4)[..., :3]
    mask = (
        np.asarray(seg, dtype=np.int32).reshape(cfg.height_px, cfg.width_px)
        if segmentation
        else None
    )
    return Frame(rgb=rgb), RenderMeta(
        view=_as_matrix(view), projection=_as_matrix(proj), segmentation=mask
    )


def project_points(
    points_world: ArrayLike,
    view: NDArray[np.float64],
    projection: NDArray[np.float64],
    width_px: int,
    height_px: int,
) -> NDArray[np.float64]:
    """Project world points through the pinhole camera to pixel index coordinates.

    Args:
        points_world: ``(N, 3)`` or ``(3,)`` points, m model, world.
        view: Row-major 4 x 4 view matrix (:attr:`RenderMeta.view`).
        projection: Row-major 4 x 4 projection matrix (:attr:`RenderMeta.projection`).
        width_px: Image width, px.
        height_px: Image height, px.

    Returns:
        ``(N, 2)`` ``(col, row)`` in pixel index space, TinyRenderer's convention (module
        docstring): the centroid of a rendered blob's pixel indices lands here. Points
        behind the camera come out as NaN.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    clip = homo @ (projection @ view).T
    w = clip[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :2] / w[:, None]
    # TinyRenderer's measured sampling convention (module docstring).
    col = (ndc[:, 0] + 1.0) * 0.5 * width_px
    row = (1.0 - ndc[:, 1]) * 0.5 * height_px - 1.0
    out = np.stack([col, row], axis=1)
    out[w <= 0.0] = np.nan
    return out
