"""``configs/render/camera.yaml`` as frozen dataclasses, with the appearance-shift variants.

Render-only configuration: nothing here can change physics. Lengths are metres model scale
(lambda = 1/25), angles degrees (config boundary), colours RGB(A) in [0, 1]. The appearance-shift
variants are **evaluation only** and never trained on (plan §1.3); a config built for one
carries its name in :attr:`RenderConfig.appearance_shift` so a data writer can refuse it.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import yaml

from ila import p8_adapter

__all__ = [
    "APPEARANCE_SHIFTS",
    "RENDER_CONFIG_PATH",
    "BackgroundConfig",
    "CameraConfig",
    "MarkingConfig",
    "RenderConfig",
    "load_render_config",
]

#: The committed render config.
RENDER_CONFIG_PATH: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "render" / "camera.yaml"

#: Evaluation-only appearance-shift variants defined in the committed config.
APPEARANCE_SHIFTS: Final[tuple[str, ...]] = (
    "light_rot_60",
    "bg_hue",
    "low_contrast",
    "mount_tilt_p2",
    "mount_tilt_m2",
)


def _vec(values: Sequence[Any], n: int, what: str) -> tuple[float, ...]:
    """Return ``values`` as a float tuple of length ``n``.

    Raises:
        ValueError: If the length is wrong.
    """
    out = tuple(float(v) for v in values)
    if len(out) != n:
        raise ValueError(f"{what} must have {n} entries, got {len(out)}")
    return out


@dataclass(frozen=True)
class CameraConfig:
    """The body-fixed downward camera.

    Attributes:
        look_axis_body: Optical axis in the drone body frame (unit), before mount tilt.
        up_axis_body: Image "up" in the body frame (unit), before mount tilt.
        offset_body_m: Camera position relative to the drone body origin, body frame, m model.
        fov_vertical_deg: Vertical field of view, degrees.
        near_m: Near clip distance, m model.
        far_m: Far clip distance, m model (20 m, P1-D1; 5 m clipped the ground).
        width_px: Image width, px.
        height_px: Image height, px.
        shadow: Whether TinyRenderer draws shadows (off).
        light_direction: TinyRenderer light direction, world frame (fixed, not normalised).
        mount_tilt_deg: Mount pitch about body y, degrees; positive pitches the optical axis
            toward body +x (forward). 0 except in the ``mount_tilt_*`` variants.
    """

    look_axis_body: tuple[float, float, float]
    up_axis_body: tuple[float, float, float]
    offset_body_m: tuple[float, float, float]
    fov_vertical_deg: float
    near_m: float
    far_m: float
    width_px: int
    height_px: int
    shadow: bool
    light_direction: tuple[float, float, float]
    mount_tilt_deg: float = 0.0

    @property
    def aspect(self) -> float:
        """Width / height, dimensionless."""
        return self.width_px / self.height_px


@dataclass(frozen=True)
class MarkingConfig:
    """The procedural helipad drawn on the deck's visual twin.

    Attributes:
        texture_px: Texture size along the plate's long side (x), px; the short side follows
            the plate's aspect ratio so texels are square.
        ring_radius_m: Outer radius of the ring, m model (P8's painted-pad radius).
        base: Deck paint colour, RGB in [0, 1].
        ring: Ring colour, RGB in [0, 1].
        letter: "H" colour, RGB in [0, 1].
        band: Edge-band colour, RGB in [0, 1].
        contrast_scale: Every texel is pulled toward the texture's mean colour by this factor
            (1 = as designed; 0.7 in the ``low_contrast`` variant).
        strategy: How the marking is shown: ``"visual_twin"`` (P1-D1).
    """

    texture_px: int
    ring_radius_m: float
    base: tuple[float, float, float]
    ring: tuple[float, float, float]
    letter: tuple[float, float, float]
    band: tuple[float, float, float]
    contrast_scale: float = 1.0
    strategy: str = "visual_twin"


@dataclass(frozen=True)
class BackgroundConfig:
    """The ground plane's render-only appearance.

    Attributes:
        ground_rgba: Uniform ground colour, RGBA in [0, 1].
        remove_texture: Remove P8's checkerboard texture (always true; an inertial reference
            otherwise).
    """

    ground_rgba: tuple[float, float, float, float]
    remove_texture: bool = True


@dataclass(frozen=True)
class RenderConfig:
    """Camera, marking and background, for the default look or one appearance shift.

    Attributes:
        camera: The camera.
        marking: The deck marking.
        background: The ground.
        random_crop_px: Training-time random crop size, px (centre crop at evaluation).
        appearance_shift: ``None`` for the default look, else the evaluation-only variant.
    """

    camera: CameraConfig
    marking: MarkingConfig
    background: BackgroundConfig
    random_crop_px: int
    appearance_shift: str | None = None


_SHIFT_KEYS: Final[frozenset[str]] = frozenset(
    {"light_direction", "ground_rgba", "marking_contrast_scale", "mount_tilt_deg"}
)


def _apply_shift(cfg: RenderConfig, name: str, spec: Mapping[str, Any]) -> RenderConfig:
    """Return ``cfg`` with one appearance-shift override applied.

    Raises:
        ValueError: On an override key this module does not know.
    """
    unknown = set(spec) - _SHIFT_KEYS
    if unknown:
        raise ValueError(f"appearance shift {name!r}: unknown keys {sorted(unknown)}")
    camera, marking, background = cfg.camera, cfg.marking, cfg.background
    if "light_direction" in spec:
        light = _vec(spec["light_direction"], 3, "light_direction")
        camera = replace(camera, light_direction=(light[0], light[1], light[2]))
    if "mount_tilt_deg" in spec:
        camera = replace(camera, mount_tilt_deg=float(spec["mount_tilt_deg"]))
    if "ground_rgba" in spec:
        g = _vec(spec["ground_rgba"], 4, "ground_rgba")
        background = replace(background, ground_rgba=(g[0], g[1], g[2], g[3]))
    if "marking_contrast_scale" in spec:
        marking = replace(marking, contrast_scale=float(spec["marking_contrast_scale"]))
    return replace(
        cfg, camera=camera, marking=marking, background=background, appearance_shift=name
    )


@functools.cache
def load_render_config(
    path: Path = RENDER_CONFIG_PATH, appearance_shift: str | None = None
) -> RenderConfig:
    """Load the render config (cached per ``(path, appearance_shift)``).

    Args:
        path: ``configs/render/camera.yaml``.
        appearance_shift: ``None`` for the default look, or one of :data:`APPEARANCE_SHIFTS`
            (evaluation only).

    Returns:
        The frozen :class:`RenderConfig`.

    Raises:
        ValueError: On an unknown variant, a camera that is not body-fixed TinyRenderer, or
            a marking strategy other than ``visual_twin``.
    """
    raw: Mapping[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cam = raw["camera"]
    if cam["mount"] != "body_fixed" or cam["renderer"] != "ER_TINY_RENDERER":
        raise ValueError("only a body-fixed ER_TINY_RENDERER camera is supported")
    look = _vec(cam["look_axis_body"], 3, "look_axis_body")
    up = _vec(cam["up_axis_body"], 3, "up_axis_body")
    offset = _vec(cam["offset_body_m"], 3, "offset_body_m")
    light = _vec(cam["light_direction"], 3, "light_direction")
    camera = CameraConfig(
        look_axis_body=(look[0], look[1], look[2]),
        up_axis_body=(up[0], up[1], up[2]),
        offset_body_m=(offset[0], offset[1], offset[2]),
        fov_vertical_deg=float(cam["fov_vertical_deg"]),
        near_m=float(cam["near_m"]),
        far_m=float(cam["far_m"]),
        width_px=int(cam["width_px"]),
        height_px=int(cam["height_px"]),
        shadow=bool(cam["shadow"]),
        light_direction=(light[0], light[1], light[2]),
    )
    mk = raw["marking"]
    if mk["strategy"] != "visual_twin" or mk["kind"] != "ring_h_band":
        raise ValueError("P1-D1 fixed the marking to a ring_h_band visual twin")
    colours = {k: _vec(v, 3, f"marking colour {k}") for k, v in mk["colours"].items()}
    marking = MarkingConfig(
        texture_px=int(mk["texture_px"]),
        ring_radius_m=float(mk["ring_radius_m"]),
        base=(colours["base"][0], colours["base"][1], colours["base"][2]),
        ring=(colours["ring"][0], colours["ring"][1], colours["ring"][2]),
        letter=(colours["letter"][0], colours["letter"][1], colours["letter"][2]),
        band=(colours["band"][0], colours["band"][1], colours["band"][2]),
        strategy=str(mk["strategy"]),
    )
    bg = raw["background"]
    if not bool(bg["remove_texture"]):
        raise ValueError("the ground texture must be removed (inertial reference)")
    g = _vec(bg["ground_rgba"], 4, "ground_rgba")
    cfg = RenderConfig(
        camera=camera,
        marking=marking,
        background=BackgroundConfig(ground_rgba=(g[0], g[1], g[2], g[3])),
        random_crop_px=int(raw["train_augmentation"]["random_crop_px"]),
    )
    if appearance_shift is None:
        return cfg
    shifts: Mapping[str, Mapping[str, Any]] = raw["appearance_shift"]
    if appearance_shift not in shifts:
        raise ValueError(f"unknown appearance shift {appearance_shift!r}; have {sorted(shifts)}")
    return _apply_shift(cfg, appearance_shift, shifts[appearance_shift])
