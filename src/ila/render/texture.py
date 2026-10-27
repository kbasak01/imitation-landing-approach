"""Procedural helipad texture (ring + "H" + edge band) for the deck's visual twin.

An original design drawn with numpy from the config's dimensions and colours (not a copied
marking). The texture covers the plate's whole top face at the face's aspect ratio, so texels
are square (to 0.1 %) and the ring is circular on the deck.

Texel layout (pinned by ``tests/test_render_texture.py`` and the top-down render test):
column ``j`` spans plate x from ``-hx`` (j = 0) to ``+hx``; row ``i`` spans plate y from
``+hy`` (i = 0) to ``-hy``. The pad centre (plate body origin) is the texture centre.
Lengths are metres model scale (lambda = 1/25), plate frame.

Drawing, in paint order over the deck ``base`` colour:

* edge band, :data:`BAND_WIDTH_M` wide, along all four edges;
* ring, outer radius ``ring_radius_m`` (P8's pad radius), :data:`RING_WIDTH_M` wide;
* "H", :data:`H_LENGTH_M` along plate x by :data:`H_WIDTH_M` along y, strokes
  :data:`H_STROKE_M`; its legs run fore-aft (plate x).

Each texel is the mean of a 4 x 4 supersample (anti-aliasing), then
``contrast_scale`` pulls every texel toward the texture's mean colour, then values are
rounded to uint8. Everything is deterministic: same config, same bytes.
"""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ila.render.config import MarkingConfig

__all__ = [
    "BAND_WIDTH_M",
    "H_LENGTH_M",
    "H_STROKE_M",
    "H_WIDTH_M",
    "RING_WIDTH_M",
    "SUPERSAMPLE",
    "encode_png",
    "helipad_texture",
    "texture_png",
    "texture_shape",
]

#: Edge-band width, m model.
BAND_WIDTH_M: Final[float] = 0.025
#: Ring width (radial), m model.
RING_WIDTH_M: Final[float] = 0.02
#: "H" extent along plate x (leg length), m model.
H_LENGTH_M: Final[float] = 0.14
#: "H" extent along plate y, m model.
H_WIDTH_M: Final[float] = 0.09
#: "H" stroke width, m model.
H_STROKE_M: Final[float] = 0.022
#: Supersamples per texel along each axis.
SUPERSAMPLE: Final[int] = 4


def texture_shape(marking: MarkingConfig, half_extents_m: tuple[float, float]) -> tuple[int, int]:
    """Return ``(rows, cols)`` of the texture for a plate face.

    Args:
        marking: The marking config (``texture_px`` = columns, along plate x).
        half_extents_m: Plate top-face half extents ``(hx, hy)``, m model.

    Returns:
        ``(round(texture_px * hy / hx), texture_px)``.
    """
    hx, hy = half_extents_m
    return int(round(marking.texture_px * hy / hx)), int(marking.texture_px)


def helipad_texture(
    marking: MarkingConfig, half_extents_m: tuple[float, float]
) -> NDArray[np.uint8]:
    """Draw the helipad for one plate face.

    Args:
        marking: Dimensions and colours.
        half_extents_m: Plate top-face half extents ``(hx, hy)``, m model.

    Returns:
        ``uint8[rows, cols, 3]`` RGB, laid out as in the module docstring.
    """
    hx, hy = half_extents_m
    rows, cols = texture_shape(marking, half_extents_m)
    s = SUPERSAMPLE
    # Supersample centres in plate coordinates, metres.
    x = -hx + (np.arange(cols * s, dtype=np.float64) + 0.5) * (2.0 * hx / (cols * s))
    y = hy - (np.arange(rows * s, dtype=np.float64) + 0.5) * (2.0 * hy / (rows * s))
    xx, yy = np.meshgrid(x, y)  # shape (rows*s, cols*s)

    img = np.empty((rows * s, cols * s, 3), dtype=np.float64)
    img[...] = marking.base
    band = (np.abs(xx) > hx - BAND_WIDTH_M) | (np.abs(yy) > hy - BAND_WIDTH_M)
    img[band] = marking.band
    r = np.hypot(xx, yy)
    ring = (r <= marking.ring_radius_m) & (r >= marking.ring_radius_m - RING_WIDTH_M)
    img[ring] = marking.ring
    half_l, half_w = H_LENGTH_M / 2.0, H_WIDTH_M / 2.0
    in_box = (np.abs(xx) <= half_l) & (np.abs(yy) <= half_w)
    legs = in_box & (np.abs(yy) >= half_w - H_STROKE_M)
    bar = in_box & (np.abs(xx) <= H_STROKE_M / 2.0)
    img[legs | bar] = marking.letter

    texels = img.reshape(rows, s, cols, s, 3).mean(axis=(1, 3))
    if marking.contrast_scale != 1.0:
        mean = texels.reshape(-1, 3).mean(axis=0)
        texels = mean + marking.contrast_scale * (texels - mean)
    out: NDArray[np.uint8] = np.rint(np.clip(texels, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def encode_png(rgb: NDArray[np.uint8]) -> bytes:
    """Encode an RGB image as a PNG (8-bit, no filter, zlib level 9, no metadata).

    Args:
        rgb: ``uint8[rows, cols, 3]``.

    Returns:
        The PNG file's bytes; identical for identical input on one zlib build.

    Raises:
        ValueError: If ``rgb`` is not ``uint8[rows, cols, 3]``.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected uint8[rows, cols, 3], got {rgb.dtype} {rgb.shape}")
    rows, cols = int(rgb.shape[0]), int(rgb.shape[1])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(rows))
    header = struct.pack(">IIBBBBB", cols, rows, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def texture_png(rgb: NDArray[np.uint8], cache_dir: Path) -> Path:
    """Write the texture as a PNG named by its pixel hash, once; return its path.

    PyBullet's ``loadTexture`` needs a file. The name carries the SHA-256 of the pixel bytes,
    so concurrent workers write identical content and the atomic rename makes a half-written
    file impossible.

    Args:
        rgb: The texture, ``uint8[rows, cols, 3]``.
        cache_dir: Directory for the file (created if missing; gitignored ``artifacts/``).

    Returns:
        The PNG path.
    """
    digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()[:16]
    path = cache_dir / f"helipad_{digest}.png"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f".{path.name}.{os.getpid()}.tmp"
        tmp.write_bytes(encode_png(rgb))
        tmp.replace(path)
    return path
