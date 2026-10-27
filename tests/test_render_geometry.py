"""Phase 1: camera geometry -- pinhole projection, attitude and deck-tilt effects, mount tilt.

Sign conventions (body frame x = forward, y = port/left, z = up; camera looks along body -z with
image up = body +x, so image right = body -y):

* roll +phi (right-handed about body x; PyBullet Euler ``[phi, 0, 0]``) lifts the port side and
  swings the optical axis toward port, so a deck straight below moves to image RIGHT
  (column increases by about f * tan(phi), f = 48 px at 96 px / 90 deg FOV);
* pitch +theta (about body y) swings the optical axis toward aft, so the deck moves to image UP
  (row decreases);
* mount tilt +2 deg pitches the optical axis toward body +x, so the deck moves to image DOWN.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pybullet as pyb
import pytest

from ila import p8_adapter
from ila.render import camera, scene
from ila.render.config import load_render_config
from render_support import listed, new_env

pytestmark = pytest.mark.pybullet

PLATE_TOP = (0.0, 0.0, 1.0)  # the static fixture's plate origin (top surface), m model, world
F_PX = 48.0  # focal length at 96 px, 90 deg vertical FOV


@pytest.fixture
def static_scene() -> Iterator[scene.Scene]:
    row = listed("static")
    env = new_env(row)
    try:
        env.reset(seed=row.episode_seed)
        sc = scene.attach(env)
        sc.pre_render()
        assert _plate_pose(sc)[0] == pytest.approx(PLATE_TOP)
        yield sc
    finally:
        env.close()


def _plate_pose(sc: scene.Scene) -> tuple:
    return pyb.getBasePositionAndOrientation(sc.handles.plate, physicsClientId=sc.handles.client)


def _centroid(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean(), rows.mean()])


def _quat(roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> tuple:
    return pyb.getQuaternionFromEuler(np.radians([roll_deg, pitch_deg, yaw_deg]).tolist())


def _twin_mask(sc: scene.Scene, pos, quat, cfg=None) -> np.ndarray:  # type: ignore[no-untyped-def]
    cfg = load_render_config().camera if cfg is None else cfg
    _, img = camera.render_pose(sc.handles.client, pos, quat, cfg, segmentation=True)
    assert img.segmentation is not None
    return img.segmentation == sc.twin


# --------------------------------------------------------------------------- pinhole


@pytest.mark.parametrize(
    ("cam_pos", "rpy_deg", "disc_xy"),
    [
        ((2.0, 2.0, 0.8), (0, 0, 0), (2.0, 2.0)),  # image centre
        ((2.0, 2.0, 0.8), (0, 0, 0), (2.35, 1.70)),  # up and right of centre
        ((2.0, 2.0, 0.8), (8, -6, 25), (1.80, 2.25)),  # tilted, yawed camera
        ((-2.0, 1.0, 1.1), (-5, 10, -40), (-1.6, 0.8)),
    ],
)
def test_pinhole_projection_within_half_pixel(
    static_scene: scene.Scene, cam_pos: tuple, rpy_deg: tuple, disc_xy: tuple
) -> None:
    c = static_scene.handles.client
    disc_centre = (disc_xy[0], disc_xy[1], 0.002)
    vis = pyb.createVisualShape(
        pyb.GEOM_CYLINDER, radius=0.15, length=0.001, rgbaColor=[1, 0, 0, 1], physicsClientId=c
    )
    disc = pyb.createMultiBody(0.0, -1, vis, disc_centre, [0, 0, 0, 1], physicsClientId=c)
    cfg = load_render_config().camera
    _, img = camera.render_pose(c, cam_pos, _quat(*rpy_deg), cfg, segmentation=True)
    assert img.segmentation is not None
    mask = img.segmentation == disc
    assert mask.sum() > 100
    predicted = camera.project_points(disc_centre, img.view, img.projection, 96, 96)[0]
    error = np.linalg.norm(_centroid(mask) - predicted)
    print(f"pinhole residual {error:.3f} px")
    assert error < 0.5, (error, _centroid(mask), predicted)


def test_project_points_behind_camera_is_nan(static_scene: scene.Scene) -> None:
    cfg = load_render_config().camera
    _, img = camera.render_pose(static_scene.handles.client, (0, 0, 2.0), _quat(), cfg)
    out = camera.project_points([[0, 0, 3.0], [0, 0, 1.0]], img.view, img.projection, 96, 96)
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()


# --------------------------------------------------------------------------- attitude


def _quad_centroid(sc: scene.Scene, img: camera.RenderMeta) -> np.ndarray:
    """Area centroid of the twin quad projected through the frame's matrices (shoelace)."""
    hx, hy, _ = sc.handles.plate_half_extents_m
    pos, quat = _plate_pose(sc)
    rot = np.asarray(pyb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    corners = [np.add(pos, rot @ [x, y, scene.TWIN_LIFT_M]) for x, y in local]
    xy = camera.project_points(corners, img.view, img.projection, 96, 96)
    x, y = xy[:, 0], xy[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    return np.array(
        [((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]
    ) / (6.0 * area)


def _render_twin(sc: scene.Scene, pos, quat):  # type: ignore[no-untyped-def]
    _, img = camera.render_pose(
        sc.handles.client, pos, quat, load_render_config().camera, segmentation=True
    )
    assert img.segmentation is not None
    mask = img.segmentation == sc.twin
    centroid = _centroid(mask)
    assert np.linalg.norm(centroid - _quad_centroid(sc, img)) < 0.5
    return centroid


def test_roll_and_pitch_move_the_deck_the_right_way(static_scene: scene.Scene) -> None:
    above = (0.0, 0.0, 1.7)
    level = _render_twin(static_scene, above, _quat())
    assert level == pytest.approx([48.0, 47.0], abs=0.5)  # optical axis pixel (camera docstring)
    small = 0.7 * F_PX * np.tan(np.radians(10.0))  # first-order shift, less a margin
    d_col, d_row = _render_twin(static_scene, above, _quat(roll_deg=10.0)) - level
    assert d_col > small and abs(d_row) < 1.0, (d_col, d_row)  # roll +10 deg: deck moves right
    d_col, d_row = _render_twin(static_scene, above, _quat(roll_deg=-10.0)) - level
    assert d_col < -small and abs(d_row) < 1.0, (d_col, d_row)
    d_col, d_row = _render_twin(static_scene, above, _quat(pitch_deg=10.0)) - level
    assert d_row < -small and abs(d_col) < 1.0, (d_col, d_row)  # pitch +10 deg: deck moves up


def test_mount_tilt_moves_the_deck_down(static_scene: scene.Scene) -> None:
    above = (0.0, 0.0, 1.7)
    base = _centroid(_twin_mask(static_scene, above, _quat()))
    for name, sign in (("mount_tilt_p2", 1.0), ("mount_tilt_m2", -1.0)):
        cfg = load_render_config(appearance_shift=name).camera
        d_row = _centroid(_twin_mask(static_scene, above, _quat(), cfg))[1] - base[1]
        expected = sign * F_PX * np.tan(np.radians(2.0))
        assert d_row == pytest.approx(expected, abs=0.6), (name, d_row)


# --------------------------------------------------------------------------- deck tilt


def _edge_spans(mask: np.ndarray) -> tuple[float, float]:
    """Vertical extent (rows) of the mask in its 3 leftmost and 3 rightmost columns."""
    cols = np.nonzero(mask.any(axis=0))[0]
    span = mask.sum(axis=0).astype(float)
    return float(span[cols[:3]].mean()), float(span[cols[-3:]].mean())


def test_deck_tilt_changes_the_projected_shape(static_scene: scene.Scene) -> None:
    above = (0.0, 0.0, 1.7)
    left, right = _edge_spans(_twin_mask(static_scene, above, _quat()))
    assert abs(left - right) <= 1.0
    # Tilt the plate (test-only env) +10 deg about world x: its +y edge (image left) rises.
    h = static_scene.handles
    pyb.resetBasePositionAndOrientation(h.plate, PLATE_TOP, _quat(10.0), physicsClientId=h.client)
    static_scene.pre_render()
    left_t, right_t = _edge_spans(_twin_mask(static_scene, above, _quat()))
    assert left_t - right_t >= 3.0, (left_t, right_t)


def test_ring_is_circular_top_down(static_scene: scene.Scene) -> None:
    """Texture layout on the twin: ring round (texels square, axes not swapped)."""
    cfg = replace(load_render_config().camera, width_px=256, height_px=256)
    c = static_scene.handles.client
    frame, img = camera.render_pose(c, (0.0, 0.0, 1.6), _quat(), cfg, segmentation=True)
    ring = load_render_config().marking.ring
    target = np.rint(np.asarray(ring) * 255)
    close = np.abs(frame.rgb.astype(float) - target).sum(-1) < 90  # lit ring colour
    rows, cols = np.nonzero(close)
    assert rows.size > 200
    height, width = rows.max() - rows.min(), cols.max() - cols.min()
    assert abs(height - width) <= 2, (height, width)
    # Ring outer diameter = 0.30 m at 0.6 m distance: 0.30 / 0.6 * 128 px.
    assert width == pytest.approx(0.30 / 0.6 * 128, abs=3)


def test_handles_come_from_adapter(static_scene: scene.Scene) -> None:
    assert static_scene.handles == p8_adapter.scene_handles(static_scene.env)
