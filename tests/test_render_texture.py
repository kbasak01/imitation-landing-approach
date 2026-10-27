"""Phase 1: the procedural helipad and the render config (no physics client needed)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ila.render import texture
from ila.render.config import APPEARANCE_SHIFTS, load_render_config

FACE = (0.40, 0.26)  # P8's plate top-face half extents, m model
#: SHA-256 of the default helipad's pixel bytes (uint8[333, 512, 3], C order).
PINNED_SHA256 = "143346e7c1b2a7974a9a232a5575e8a31ad951e3594a99b9768633193abd7773"
#: Same, for the evaluation-only `low_contrast` variant.
PINNED_LOW_CONTRAST_SHA256 = "31b041b829c35a5a731e8f8c115bddfb01f4949b7936d7c70481d5da70466c20"


def _rgb8(colour: tuple[float, float, float]) -> list[int]:
    return [int(np.rint(c * 255.0)) for c in colour]


def _texel(x: float, y: float, shape: tuple[int, int]) -> tuple[int, int]:
    """(row, col) of the texel containing plate point (x, y), m."""
    rows, cols = shape
    col = int((x + FACE[0]) / (2 * FACE[0]) * cols)
    row = int((FACE[1] - y) / (2 * FACE[1]) * rows)
    return row, col


@pytest.fixture(scope="module")
def tex() -> np.ndarray:
    return texture.helipad_texture(load_render_config().marking, FACE)


def test_shape_and_pinned_bytes(tex: np.ndarray) -> None:
    assert tex.shape == (333, 512, 3) and tex.dtype == np.uint8
    assert hashlib.sha256(tex.tobytes()).hexdigest() == PINNED_SHA256


def test_deterministic(tex: np.ndarray) -> None:
    again = texture.helipad_texture(load_render_config().marking, FACE)
    np.testing.assert_array_equal(tex, again)


def test_texels_square(tex: np.ndarray) -> None:
    rows, cols = tex.shape[:2]
    assert abs((2 * FACE[0] / cols) / (2 * FACE[1] / rows) - 1.0) < 2e-3


def test_layout(tex: np.ndarray) -> None:
    m = load_render_config().marking
    shape = tex.shape[:2]
    r = m.ring_radius_m - texture.RING_WIDTH_M / 2  # mid-ring radius
    assert tex[_texel(-0.399, 0.259, shape)].tolist() == _rgb8(m.band)  # corner: edge band
    assert tex[_texel(0.0, 0.0, shape)].tolist() == _rgb8(m.letter)  # crossbar centre
    assert tex[_texel(0.05, 0.035, shape)].tolist() == _rgb8(m.letter)  # an H leg (runs along x)
    assert tex[_texel(0.0, 0.035, shape)].tolist() == _rgb8(m.letter)
    assert tex[_texel(0.05, 0.0, shape)].tolist() == _rgb8(m.base)  # between the legs
    for x, y in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        assert tex[_texel(x, y, shape)].tolist() == _rgb8(m.ring), (x, y)
    assert tex[_texel(0.0, 0.10, shape)].tolist() == _rgb8(m.base)
    assert tex[_texel(0.30, 0.0, shape)].tolist() == _rgb8(m.base)


def test_symmetric_under_half_turn(tex: np.ndarray) -> None:
    # Ring, H and band are symmetric about both plate axes (to within 1 count of rounding).
    diff = np.abs(tex.astype(int) - tex[::-1, ::-1].astype(int))
    assert diff.max() <= 1


def test_low_contrast_variant(tex: np.ndarray) -> None:
    low = texture.helipad_texture(load_render_config(appearance_shift="low_contrast").marking, FACE)
    assert hashlib.sha256(low.tobytes()).hexdigest() == PINNED_LOW_CONTRAST_SHA256
    ratio = low.reshape(-1, 3).astype(float).std(0) / tex.reshape(-1, 3).astype(float).std(0)
    np.testing.assert_allclose(ratio, 0.7, atol=0.01)
    np.testing.assert_allclose(low.reshape(-1, 3).mean(0), tex.reshape(-1, 3).mean(0), atol=0.5)


def test_png_round_trip_and_cache(tex: np.ndarray, tmp_path: Path) -> None:
    data = texture.encode_png(tex)
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    np.testing.assert_array_equal(decoded, tex)
    path = texture.texture_png(tex, tmp_path)
    assert path.read_bytes() == data
    assert texture.texture_png(tex, tmp_path) == path
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


# --------------------------------------------------------------------------- config


def test_default_camera_numbers() -> None:
    cam = load_render_config().camera
    assert cam.offset_body_m == (0.0, 0.0, -0.02)
    assert cam.look_axis_body == (0.0, 0.0, -1.0) and cam.up_axis_body == (1.0, 0.0, 0.0)
    assert (cam.fov_vertical_deg, cam.near_m, cam.far_m) == (90.0, 0.01, 20.0)
    assert (cam.width_px, cam.height_px, cam.shadow, cam.mount_tilt_deg) == (96, 96, False, 0.0)
    assert load_render_config().appearance_shift is None


def test_appearance_shifts_change_one_thing_each() -> None:
    base = load_render_config()
    changed = {}
    for name in APPEARANCE_SHIFTS:
        cfg = load_render_config(appearance_shift=name)
        assert cfg.appearance_shift == name
        diffs = [
            f
            for f in ("camera", "marking", "background", "random_crop_px")
            if getattr(cfg, f) != getattr(base, f)
        ]
        assert len(diffs) == 1, (name, diffs)
        changed[name] = diffs[0]
    assert changed == {
        "light_rot_60": "camera",
        "bg_hue": "background",
        "low_contrast": "marking",
        "mount_tilt_p2": "camera",
        "mount_tilt_m2": "camera",
    }
    assert load_render_config(appearance_shift="mount_tilt_m2").camera.mount_tilt_deg == -2.0


def test_light_rot_60_is_a_60_degree_azimuth_rotation() -> None:
    a = np.asarray(load_render_config().camera.light_direction)
    b = np.asarray(load_render_config(appearance_shift="light_rot_60").camera.light_direction)
    azimuth = np.degrees(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))
    assert azimuth == pytest.approx(60.0, abs=0.01)
    assert b[2] == a[2] and np.hypot(*b[:2]) == pytest.approx(np.hypot(*a[:2]), abs=1e-4)


def test_unknown_shift_raises() -> None:
    with pytest.raises(ValueError, match="appearance shift"):
        load_render_config(appearance_shift="train_on_this")
