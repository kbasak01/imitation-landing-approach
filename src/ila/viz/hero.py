"""README hero GIFs: the Phase 8 gallery's picks, seen from outside, with the onboard frame.

Simulation only: PyBullet physics; every image is rendered by PyBullet's TinyRenderer. Model
scale, lambda = 1/25 (1 model s = 5 full-scale s; 1 model m = 25 full-scale m). These GIFs
illustrate. They are not evidence, and no rate is read from them.

**No new selection.** The episodes are exactly the four picks in
``results/e08/gifs/manifest.json``, which ``ila.viz.gifs.RULE`` chose before any GIF existed.
:func:`load_picks` copies them, and :func:`check_picks` confirms them against the committed rows
and P8's ``id`` list. This module has no selection logic of its own. Panels: the teacher (P8
``ppo`` seed 0), ``bc`` s0, ``bc_dagger`` s0 and ``dp`` s0 (Ta = 8, K = 10, no ensembling), as in
the gallery.

**Re-flight.** Every panel is re-flown through the gallery's code path:
:func:`ila.viz.gifs._task` gives the same :class:`~ila.eval.runner.VisionTask`, flown by
:func:`ila.eval.runner.fly_chunk` on the CPU fp32 single-sample path. The actor is the gallery's
frame recorder, wrapped in :class:`_OutsideRecorder`. Before any GIF is written,
:func:`ila.viz.gifs.check_panel` asserts each re-flown row equals its committed row. For students
the whole CSV line is compared; for the teacher, P8's ``compare_to_reference`` is used. Any
difference raises. Nothing is ever relabelled.

**Outside view.** P8's ``rld.viz.gifs.Camera`` (``p8_adapter.OutsideCamera``) is a fixed
world-frame camera. It renders through ``getCameraImage`` (TinyRenderer) on the env's own DIRECT
client and never steps the simulation. The fly loop calls the recorder's ``act(k, ...)`` only
after ``env.reset()`` (k = 0) or after the k-th ``env.step()`` has returned and the onboard
frame k has been rendered. :func:`ila.p8_adapter.grab_outside_view` asserts that the physics
step count is ``min(8 k, cap)`` before the grab and is unchanged after it, so frame k shares its
``stepSimulation`` count with ``obs[k]`` and the onboard frame k. The grab makes one
visual-only change and undoes it: the drone's collision-cylinder visual is tinted orange for the
grab, then restored to its own colour before ``act`` returns. Nothing is created, removed or
re-posed. The scene is ila's P1-D1 scene, unchanged: the uniform sea-blue ground, P8's world-fixed
pad marker hidden, and the helipad twin, already re-posed to the plate's physics pose by the
onboard render of the same step. The env is built without P8's ``visual_shapes``, so the drone
is drawn as its collision cylinder (0.12 m diameter, 0.025 m tall), which is also the shape the
physics sees.

**Reaching the env.** The runner hands its actor factory only ``(cfgs, contract)`` and its actor
only pixels. While one panel flies, :func:`_env_tap` wraps ``p8_adapter.make_env`` inside the
worker process. The wrapper calls the original and keeps a reference to the env it returns, the
same object, unmodified. The original function is restored on exit. ``runner.py`` and the render
and physics code are not changed.

**Frames.** Frames are captured at control steps 0, s, 2s, ... (s = frame stride), up to the last
step a policy acted on (n - 1). The terminal state after the n-th ``env.step`` is not drawn,
because the runner calls no actor after it. That is at most 33.3 ms model scale before the end
(the onboard gallery has the same property). A finished panel holds its last frame and shows its
committed outcome. Frame durations follow model-scale real time, rounded to the GIF's 10 ms
ticks with no cumulative drift (x 5 slower at full scale), and the final composite is held for
:data:`HOLD_MS`.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import multiprocessing as mp
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import runner
from ila.obs.split import ObsContract
from ila.viz import gifs

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from PIL import ImageFont

    from ila.eval.jobs import JobStatus

__all__ = [
    "CAMERA",
    "DRONE_RGBA",
    "FRAME_STRIDE",
    "GIF_LIMIT_BYTES",
    "HOLD_MS",
    "OUT_DIR",
    "SOURCE_MANIFEST",
    "TITLES",
    "TOTAL_LIMIT_BYTES",
    "Column",
    "OUTCOME_CLASSES",
    "HeroResult",
    "check_picks",
    "compose",
    "durations_ms",
    "encode_gif",
    "fly_hero_panel",
    "gif_name",
    "header_boxes",
    "header_text",
    "load_picks",
    "n_composite_frames",
    "run_hero",
]

REPO: Final[Path] = p8_adapter.REPO_ROOT
#: The gallery manifest whose picks are reused verbatim.
SOURCE_MANIFEST: Final[Path] = REPO / "results" / "e08" / "gifs" / "manifest.json"
#: Committed rows root (``results/e08``).
ROWS_ROOT: Final[Path] = REPO / "results" / "e08"
OUT_DIR: Final[Path] = REPO / "docs" / "media"

#: Outside camera, metres model scale, world frame (x bow, y port, z up), degrees. The plate
#: (0.80 x 0.52 m) oscillates about (0, 0, 1.0) m; drones start 0.6-1.0 m above the mean deck
#: within +-0.3 m laterally. This framing keeps a drone at start height and the whole plate in
#: view with the plate about 2/3 of the panel width (P8 used target (0, 0, 1.40), 1.45 m,
#: yaw 38, pitch -7 for its mesh drone and painted disc).
CAMERA: Final = p8_adapter.OutsideCamera(
    target_m=(0.0, 0.0, 1.40),
    distance_m=1.45,
    yaw_deg=35.0,
    pitch_deg=-20.0,
    fov_deg=50.0,
    width_px=300,
    height_px=330,
    supersample=2,
)
#: The drone's tint in the outside view only (its own visual is white; restored after each grab).
DRONE_RGBA: Final[tuple[float, float, float, float]] = (0.95, 0.42, 0.05, 1.0)
#: Control rate, Hz (P8's ``ctrl_freq_hz``; checked at run time): one step = 33.3 ms model scale.
CTRL_HZ: Final[int] = 30
#: Control steps per GIF frame (1: 33.3 ms model scale per frame, real time at model scale).
FRAME_STRIDE: Final[int] = 1
#: Display time added to the final composite, ms.
HOLD_MS: Final[int] = 1500
GIF_LIMIT_BYTES: Final[int] = 2_500_000
TOTAL_LIMIT_BYTES: Final[int] = 10_000_000
#: Onboard 96 px frame enlargement (nearest neighbour).
ONBOARD_ZOOM: Final[int] = 2
#: GIF palette size (one index stays free for Pillow's transparent delta frames).
PALETTE_COLOURS: Final[int] = 128

TITLES: Final[Mapping[str, str]] = {
    "a_all_four_succeed": "(a) all four panels succeed",
    "b_students_split": "(b) the students split",
    "c_dp_off_pad": "(c) dp lands off the pad",
    "d_teacher_ss6_failure": "(d) the teacher fails in SS6",
}
ARM_LABELS: Final[Mapping[str, str]] = {
    "teacher": "teacher (ppo s0)",
    "bc": "bc s0",
    "bc_dagger": "bc_dagger s0",
    "dp": "dp s0 · Ta 8, K 10",
}
SUBTITLE: Final[str] = "hand-picked by stated rule · simulation (PyBullet, TinyRenderer)"
FOOTER: Final[str] = (
    "Simulation only · model scale, λ = 1/25 (1 s model = 5 s full scale) · plays in real time "
    "at model scale · drone drawn as its collision cylinder (orange) · outcomes from committed rows"
)

SUCCESS_RGB: Final = (22, 120, 60)
FAILURE_RGB: Final = (176, 40, 38)
INK: Final = (20, 20, 20)
INK_2: Final = (82, 81, 78)
SURFACE: Final = (252, 252, 251)
HEADER_H: Final[int] = 62
#: Column header: the arm label on row 1, the outcome badge (once ended) on row 2 beneath it,
#: so the two never share a row and cannot collide for any label or outcome word.
LABEL_H: Final[int] = 62
#: P8's outcome classes (the badge words), matched in P8's order.
OUTCOME_CLASSES: Final[tuple[str, ...]] = (
    "crash",
    "off_pad",
    "hard_landing",
    "bounce",
    "success",
    "timeout",
)
ONBOARD_H: Final[int] = 96 * ONBOARD_ZOOM + 18
FOOTER_H: Final[int] = 24

type Image = npt.NDArray[np.uint8]


# --------------------------------------------------------------------------- picks


def load_picks(path: Path = SOURCE_MANIFEST) -> tuple[dict[str, gifs.Pick], str, dict[str, Any]]:
    """Read the gallery's picks verbatim from its committed manifest.

    Args:
        path: ``results/e08/gifs/manifest.json``.

    Returns:
        ``(category -> Pick, the manifest's SHA-256, the parsed manifest)``. A category the
        gallery left empty (``null``) is omitted. No selection happens here.

    Raises:
        ValueError: If a category is not one of :data:`ila.viz.gifs.CATEGORIES`.
    """
    data = path.read_bytes()
    record: dict[str, Any] = json.loads(data)
    picks: dict[str, gifs.Pick] = {}
    for cat, p in record["picks"].items():
        if cat not in gifs.CATEGORIES:
            raise ValueError(f"unknown category {cat!r} in {path}")
        if p is None:
            continue
        picks[cat] = gifs.Pick(
            category=cat,
            ss=str(p["ss"]),
            index=int(p["index"]),
            episode_seed=int(p["episode_seed"]),
            position=int(p["list_position"]),
            outcomes={k: str(v) for k, v in p["outcomes"].items()},
        )
    return picks, hashlib.sha256(data).hexdigest(), record


def check_picks(
    picks: Mapping[str, gifs.Pick],
    record: Mapping[str, Any],
    listed: Sequence[p8_adapter.ListedEpisode],
    rows: Mapping[str, Sequence[Mapping[str, str]]],
    sources: Mapping[str, str],
) -> None:
    """Confirm the copied picks against what they were chosen from (no re-selection).

    Args:
        picks: From :func:`load_picks`.
        record: The gallery manifest.
        listed: P8's ``id`` list in list order.
        rows: Panel -> committed ``id`` rows (:func:`ila.viz.gifs.committed_panel_rows`).
        sources: Their files' SHA-256 (same function).

    Raises:
        AssertionError: If the rule, the committed row files, a listed episode or a committed
            outcome differs from what the gallery recorded.
    """
    rule_sha = hashlib.sha256(gifs.RULE.encode("utf-8")).hexdigest()
    if record["rule_sha256"] != rule_sha:
        raise AssertionError("gallery manifest's rule SHA-256 is not ila.viz.gifs.RULE's")
    if dict(record["committed_row_sources"]) != dict(sources):
        raise AssertionError(
            f"committed rows changed since the gallery: {record['committed_row_sources']} "
            f"!= {dict(sources)}"
        )
    for cat, p in picks.items():
        e = listed[p.position]
        if (e.regime, e.ss, int(e.index), int(e.episode_seed)) != (
            gifs.LIST_NAME,
            p.ss,
            p.index,
            p.episode_seed,
        ):
            raise AssertionError(f"{cat}: list position {p.position} is not {p}")
        for panel, outcome in p.outcomes.items():
            hit = [
                r
                for r in rows[panel]
                if r["regime"] == gifs.LIST_NAME and r["ss"] == p.ss and int(r["index"]) == p.index
            ]
            if len(hit) != 1 or hit[0]["outcome"] != outcome:
                raise AssertionError(f"{cat}/{panel}: committed outcome is not {outcome!r}")


# --------------------------------------------------------------------------- re-flight


@contextlib.contextmanager
def _env_tap() -> Iterator[list[p8_adapter.DeckLandingAviary]]:
    """Within the block, keep a reference to each env ``p8_adapter.make_env`` builds.

    The wrapper calls the original function with the same arguments and returns its result
    unchanged; the original is restored on exit. Use only inside one worker's re-flight.
    """
    built: list[p8_adapter.DeckLandingAviary] = []
    original = p8_adapter.make_env

    def make_env(
        cfgs: p8_adapter.EvalConfigs, listed: p8_adapter.ListedEpisode
    ) -> p8_adapter.DeckLandingAviary:
        env = original(cfgs, listed)
        built.append(env)
        return env

    p8_adapter.make_env = make_env
    try:
        yield built
    finally:
        p8_adapter.make_env = original


class _OutsideRecorder:
    """Wraps the gallery's recorder; grabs the outside view at every ``stride``-th step.

    Read-only towards physics. It changes nothing but the drone's visual tint, and only for
    the duration of one grab.
    """

    def __init__(
        self,
        inner: Any,
        envs: list[p8_adapter.DeckLandingAviary],
        camera: p8_adapter.OutsideCamera,
        stride: int,
    ) -> None:
        self.inner = inner
        self.needs_frames = inner.needs_frames
        self.envs = envs
        self.camera = camera
        self.stride = stride
        self.steps: list[int] = []
        self.outside: list[Image] = []
        self._env: p8_adapter.DeckLandingAviary | None = None

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        if len(self.envs) != 1:
            raise RuntimeError(f"expected the one env of this chunk, tapped {len(self.envs)}")
        self._env = self.envs[0]
        self.inner.begin(listed, context)

    def act(self, step: int, obs: Any, frame: Any) -> npt.NDArray[np.float64]:
        if step % self.stride == 0:
            self.outside.append(self._grab(step))
            self.steps.append(step)
        out: npt.NDArray[np.float64] = self.inner.act(step, obs, frame)
        return out

    def _grab(self, step: int) -> Image:
        if self._env is None:
            raise RuntimeError("act before begin")
        try:
            img = p8_adapter.grab_outside_view(self._env, self.camera, step, DRONE_RGBA)
        except p8_adapter.OutsideViewTimingError as exc:
            raise runner.RenderTimingError(str(exc)) from exc
        return np.ascontiguousarray(img)


@dataclass
class HeroResult:
    """One re-flown panel.

    Attributes:
        result: The gallery's :class:`~ila.viz.gifs.PanelResult` (row; onboard frame of every
            step the policy acted on, 96 x 96 x 3 uint8).
        steps: Control steps of the captured outside frames (0, s, 2s, ...).
        outside: Outside frames at ``steps`` (H x W x 3 uint8).
    """

    result: gifs.PanelResult
    steps: list[int] = field(default_factory=list)
    outside: list[Image] = field(default_factory=list)


def fly_hero_panel(
    panel: gifs.Panel,
    *,
    stride: int = FRAME_STRIDE,
    camera: p8_adapter.OutsideCamera = CAMERA,
) -> HeroResult:
    """Re-fly one panel in this process (one torch thread), recording both views.

    Args:
        panel: The gallery panel (:func:`ila.viz.gifs.student_panel` or the teacher).
        stride: Control steps between outside frames.
        camera: The outside camera.

    Returns:
        The :class:`HeroResult` (not yet checked: :func:`ila.viz.gifs.check_panel`).
    """
    runner.set_single_thread_env()
    runner.init_worker_threads()
    task = gifs._task(panel)
    onboard: list[Image] = []
    holder: list[_OutsideRecorder] = []
    with _env_tap() as envs:
        base = gifs._recording_factory(task.spec, onboard)

        def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Any, object]:
            inner, policy = base(cfgs, contract)
            rec = _OutsideRecorder(inner, envs, camera, stride)
            holder.append(rec)
            return rec, policy

        rows = runner.fly_chunk(task, make)
    (rec,) = holder
    return HeroResult(gifs.PanelResult(panel, rows[0], onboard), rec.steps, rec.outside)


# --------------------------------------------------------------------------- drawing


@dataclass(frozen=True)
class Column:
    """One panel as drawn.

    Attributes:
        panel: ``teacher`` or a student arm.
        outcome: Its committed outcome class.
        n_steps: Control steps its episode took (committed ``steps``).
        steps: Control steps of its frames (0, s, 2s, ...; all < ``n_steps``).
        outside: Outside frames (H x W x 3 uint8), one per step in ``steps``.
        onboard: Onboard frames (96 x 96 x 3 uint8), one per step in ``steps``.
    """

    panel: str
    outcome: str
    n_steps: int
    steps: Sequence[int]
    outside: Sequence[Image]
    onboard: Sequence[Image]


def column_of(hero: HeroResult, stride: int) -> Column:
    """The :class:`Column` of a checked panel, subsampled to ``stride`` (a multiple of its own).

    Raises:
        ValueError: If the frames do not cover steps 0 ... n - 1 at the recorded stride.
    """
    row = hero.result.row
    n = int(row["steps"])
    frames = hero.result.frames
    if len(frames) != n or not hero.steps or hero.steps[-1] >= n:
        raise ValueError(f"{hero.result.panel.panel}: {len(frames)} onboard frames, {n} steps")
    keep = [i for i, s in enumerate(hero.steps) if s % stride == 0]
    steps = [hero.steps[i] for i in keep]
    if steps != list(range(0, n, stride)):
        raise ValueError(f"{hero.result.panel.panel}: outside frames at {hero.steps[:5]} ...")
    return Column(
        panel=hero.result.panel.panel,
        outcome=str(row["outcome"]),
        n_steps=n,
        steps=steps,
        outside=[hero.outside[i] for i in keep],
        onboard=[frames[s] for s in steps],
    )


def n_composite_frames(columns: Sequence[Column]) -> int:
    """Composite frames: up to the longest panel's last frame, plus one with every badge shown."""
    return max(len(c.steps) for c in columns) + 1


def ended(column: Column, j: int, stride: int) -> bool:
    """Whether composite frame ``j`` (time ``j * stride`` control steps) is past the episode."""
    return j * stride >= column.n_steps


def durations_ms(n_frames: int, stride: int, hold_ms: int = HOLD_MS) -> list[int]:
    """Frame durations, ms, multiples of 10 (GIF ticks), tracking model-scale real time.

    Frame j covers control steps ``[j s, (j + 1) s)``; its end is rounded to 10 ms, so the
    sum never drifts from ``n s / 30`` s by more than 5 ms. The last frame also gets
    ``hold_ms``.
    """

    def tick(j: int) -> int:
        return 10 * round(j * stride * 1000.0 / CTRL_HZ / 10.0)

    out = [tick(j + 1) - tick(j) for j in range(n_frames)]
    out[-1] += hold_ms
    return out


def header_text(pick: gifs.Pick) -> str:
    """The header line: category, episode identity, provenance."""
    return f"{TITLES[pick.category]} · id {pick.ss} episode {pick.index} · {SUBTITLE}"


def gif_name(pick: gifs.Pick) -> str:
    """``hero_<category>__id_<SS>_ep<index>.gif``."""
    return f"hero_{pick.category}__id_{pick.ss}_ep{pick.index}.gif"


@functools.cache
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Matplotlib's bundled DejaVu Sans (as P8's ``_font``; it has the λ glyph), ``size`` px."""
    import matplotlib
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


type Box = tuple[float, float, float, float]


def header_boxes(panel: str, outcome: str, column_width: int) -> tuple[Box, Box, Box]:
    """Column-header geometry relative to the column's top-left corner, px.

    Args:
        panel: ``teacher`` or a student arm (its :data:`ARM_LABELS` text).
        outcome: The outcome word on the badge.
        column_width: Column width, px.

    Returns:
        ``(label text box, badge box, badge text box)``, each ``(x0, y0, x1, y1)``. The badge
        sits on its own row under the label, left-aligned with it.

    Raises:
        ValueError: If the label or the badge does not fit inside the column and the header.
    """
    label_font, badge_font = _font(18, True), _font(16, True)
    label = label_font.getbbox(ARM_LABELS[panel])
    label_box = (10.0 + label[0], 6.0 + label[1], 10.0 + label[2], 6.0 + label[3])
    text = badge_font.getbbox(outcome)
    top = 33.0
    text_box = (19.0 + text[0], top + 3 + text[1], 19.0 + text[2], top + 3 + text[3])
    badge = (10.0, top, text_box[2] + 9.0, LABEL_H - 5.0)
    for box in (label_box, badge):
        if box[2] > column_width - 6 or box[3] > LABEL_H:
            raise ValueError(f"{panel}/{outcome}: {box} does not fit {column_width} x {LABEL_H}")
    return label_box, badge, text_box


def _layout(columns: Sequence[Column]) -> tuple[int, int, int, int]:
    """``(column width, outside height, canvas width, canvas height)``, px."""
    oh, ow = columns[0].outside[0].shape[:2]
    return ow, oh, ow * len(columns), HEADER_H + LABEL_H + oh + ONBOARD_H + FOOTER_H


def compose(pick: gifs.Pick, columns: Sequence[Column], stride: int) -> list[PILImage.Image]:
    """Composite RGB frames of one GIF.

    Args:
        pick: The episode (header).
        columns: Panels, left to right; all frames of one kind share a shape.
        stride: Control steps per composite frame (the columns' stride).

    Returns:
        :func:`n_composite_frames` RGB images; frame j shows control step ``j * stride`` of
        every panel (a finished panel holds its last frame and shows its outcome badge).
    """
    from PIL import Image, ImageDraw

    cw, oh, width, height = _layout(columns)
    zoom = 96 * ONBOARD_ZOOM
    f_title, f_sub, f_label = _font(20, True), _font(16), _font(18, True)
    f_badge, f_tag, f_foot = _font(16, True), _font(14), _font(13)
    geometry = {c.panel: header_boxes(c.panel, c.outcome, cw) for c in columns}
    title = f"{TITLES[pick.category]} · id {pick.ss} episode {pick.index}"
    out: list[PILImage.Image] = []
    for j in range(n_composite_frames(columns)):
        canvas = Image.new("RGB", (width, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill=INK, font=f_title)
        draw.text((12, 36), SUBTITLE, fill=INK_2, font=f_sub)
        clock = f"t = {j * stride / CTRL_HZ:.1f} s (model scale)"
        draw.text(
            (width - 12 - draw.textlength(clock, font=f_title), 8), clock, fill=INK, font=f_title
        )
        for p, col in enumerate(columns):
            i = min(j, len(col.steps) - 1)
            x0, y = p * cw, HEADER_H
            _, badge, _ = geometry[col.panel]
            draw.text((x0 + 10, y + 6), ARM_LABELS[col.panel], fill=INK, font=f_label)
            if ended(col, j, stride):
                colour = SUCCESS_RGB if col.outcome == "success" else FAILURE_RGB
                draw.rounded_rectangle(
                    (x0 + badge[0], y + badge[1], x0 + badge[2], y + badge[3]), 5, fill=colour
                )
                draw.text((x0 + 19, y + badge[1] + 3), col.outcome, fill=SURFACE, font=f_badge)
            y += LABEL_H
            canvas.paste(Image.fromarray(np.asarray(col.outside[i])), (x0, y))
            y += oh
            board = Image.fromarray(np.asarray(col.onboard[i])).resize(
                (zoom, zoom), Image.Resampling.NEAREST
            )
            bx, by = x0 + 10, y + 9
            draw.rectangle((bx - 1, by - 1, bx + zoom, by + zoom), outline=INK_2)
            canvas.paste(board, (bx, by))
            tag = ["onboard", "camera", "(rendered)", f"96 px, ×{ONBOARD_ZOOM}"]
            if col.panel == "teacher":
                tag += ["", "not a", "teacher", "input"]
            for k, line in enumerate(tag):
                draw.text((bx + zoom + 9, by + 2 + 18 * k), line, fill=INK_2, font=f_tag)
            if p:
                draw.line((x0, HEADER_H, x0, height - FOOTER_H), fill=SURFACE, width=3)
        draw.text((12, height - FOOTER_H + 5), FOOTER, fill=INK_2, font=f_foot)
        out.append(canvas)
    return out


def encode_gif(frames: Sequence[PILImage.Image], durations: Sequence[int]) -> bytes:
    """One looping GIF: one shared adaptive palette, no dithering, Pillow's delta frames.

    The palette is taken from the first, middle and last frames, so a pixel that does not
    change keeps its index, and unchanged regions cost (almost) nothing.
    """
    from PIL import Image

    w, h = frames[0].size
    n = len(frames)
    probe = Image.new("RGB", (w, h * 3))
    for k, idx in enumerate((0, n // 2, n - 1)):
        probe.paste(frames[idx], (0, k * h))
    palette = probe.quantize(colors=PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    buf = io.BytesIO()
    quant[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=quant[1:],
        duration=list(durations),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return buf.getvalue()


# --------------------------------------------------------------------------- the job


def _fly_all(panels: Sequence[gifs.Panel], workers: int, stride: int) -> Iterator[HeroResult]:
    """Fly panels, at most ``workers`` (<= 4) processes, yielding in input order."""
    fly: Callable[[gifs.Panel], HeroResult] = functools.partial(fly_hero_panel, stride=stride)
    workers = max(1, min(4, workers))
    if workers == 1 or len(panels) <= 1:
        for p in panels:
            yield fly(p)
        return
    with mp.get_context("spawn").Pool(processes=min(workers, len(panels))) as pool:
        yield from pool.imap(fly, panels, chunksize=1)


def run_hero(
    *,
    out_dir: Path = OUT_DIR,
    source_manifest: Path = SOURCE_MANIFEST,
    rows_root: Path = ROWS_ROOT,
    workers: int = 4,
    only: str | None = None,
    panels: Sequence[str] = gifs.PANELS,
    stride: int = FRAME_STRIDE,
    require_clean: bool = True,
    status: JobStatus | None = None,
) -> dict[str, Any]:
    """Re-fly the gallery's picks with the outside view, assert, compose, write.

    Args:
        out_dir: Where the GIFs and ``manifest.json`` go (``docs/media`` for the full run).
        source_manifest: The gallery manifest whose picks are reused.
        rows_root: ``results/e08`` (committed rows).
        workers: Re-flight processes (<= 4).
        only: One category (smoke run).
        panels: Panels to fly (smoke runs may fly fewer than :data:`ila.viz.gifs.PANELS`).
        stride: Control steps per GIF frame.
        require_clean: Refuse a dirty tree (outside ``out_dir``).
        status: Optional ``status.json`` to update after each panel.

    Returns:
        The manifest record (also written to ``out_dir / "manifest.json"``).

    Raises:
        ila.eval.matrix.DirtyTreeError: On a dirty tree with ``require_clean``.
        AssertionError: If a pick no longer matches its source or a re-flown row differs from
            its committed row (raised before any GIF is written).
        ValueError: If a GIF exceeds :data:`GIF_LIMIT_BYTES` even at twice the stride, or all
            exceed :data:`TOTAL_LIMIT_BYTES` (nothing is written).
    """
    dirty = ev_matrix.dirty_paths(exclude=out_dir)
    if dirty and require_clean:
        raise ev_matrix.DirtyTreeError(f"uncommitted changes (commit before the hero job): {dirty}")
    unknown = set(panels) - set(gifs.PANELS)
    if unknown or not panels:
        raise ValueError(f"panels must be a non-empty subset of {gifs.PANELS}, got {panels}")
    order = [p for p in gifs.PANELS if p in panels]
    cfgs = p8_adapter.load_eval_configs()
    if int(cfgs.landing.ctrl_freq_hz) != CTRL_HZ:
        raise ValueError(f"control rate {cfgs.landing.ctrl_freq_hz} Hz != {CTRL_HZ}")
    picks, source_sha, source = load_picks(source_manifest)
    if only is not None:
        if only not in picks:
            raise ValueError(f"--only {only!r}: not a picked category {sorted(picks)}")
        picks = {only: picks[only]}
    rows, sources = gifs.committed_panel_rows(rows_root)
    listed = ev_arms.list_episodes(gifs.LIST_NAME)
    check_picks(picks, source, listed, rows, sources)

    flights: list[gifs.Panel] = []
    for pick in picks.values():
        e = listed[pick.position]
        for name in order:
            flights.append(
                gifs.Panel("teacher", e) if name == "teacher" else gifs.student_panel(name, e)
            )
    for f in flights:
        if f.panel == "dp" and f.inference != (8, 10, False):
            raise AssertionError(f"dp core inference is {f.inference}, not (Ta 8, K 10, no ens)")
    jobs = {arm: gifs.student_job(arm) for arm in gifs.STUDENTS if arm in order}
    lines = {arm: gifs.committed_lines(rows_root / jobs[arm].rel_path) for arm in jobs}
    if status is not None:
        status.update("running", progress={"panels_done": 0, "panels": len(flights)})

    encoded: list[tuple[gifs.Pick, bytes, int, int, list[dict[str, Any]]]] = []
    group: list[HeroResult] = []
    cats = list(picks.values())
    done = 0
    for hero in _fly_all(flights, workers, stride):
        r = hero.result
        p = r.panel
        if p.panel == "teacher":
            gifs.check_panel(r, "", None)
        else:
            gifs.check_panel(r, lines[p.panel][(p.episode.ss, int(p.episode.index))], jobs[p.panel])
        done += 1
        if status is not None:
            status.update(progress={"panels_done": done, "panels": len(flights)})
        group.append(hero)
        if len(group) < len(order):
            continue
        pick = cats[len(encoded)]
        reflight = [
            {
                "panel": h.result.panel.panel,
                "ckpt_sha256": h.result.panel.sha256,
                "steps": int(h.result.row["steps"]),
                "outcome": str(h.result.row["outcome"]),
                "committed_outcome": pick.outcomes[h.result.panel.panel],
                "check": _check_text(h),
                "equal": True,
            }
            for h in group
        ]
        for rec in reflight:
            if rec["outcome"] != rec["committed_outcome"]:
                raise AssertionError(f"{pick.category}: {rec}")
        data, used, n = b"", stride, 0
        for factor in (1, 2):
            used = stride * factor
            columns = [column_of(h, used) for h in group]
            frames = compose(pick, columns, used)
            n = len(frames)
            data = encode_gif(frames, durations_ms(n, used))
            if len(data) <= GIF_LIMIT_BYTES:
                break
        if len(data) > GIF_LIMIT_BYTES:
            raise ValueError(f"{gif_name(pick)}: {len(data)} bytes > {GIF_LIMIT_BYTES}")
        encoded.append((pick, data, used, n, reflight))
        group = []

    total = sum(len(d) for _, d, _, _, _ in encoded)
    if total > TOTAL_LIMIT_BYTES:
        raise ValueError(f"hero GIFs total {total} bytes > {TOTAL_LIMIT_BYTES}; nothing written")
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_records = []
    for pick, data, used, n, reflight in encoded:
        (out_dir / gif_name(pick)).write_bytes(data)
        gif_records.append(
            {
                "category": pick.category,
                "file": gif_name(pick),
                "header": header_text(pick),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "frames": n,
                "frame_stride_steps": used,
                "frame_ms_model_scale": round(used * 1000.0 / CTRL_HZ, 3),
                "reflight": reflight,
            }
        )
    record = {
        "simulation_only": "PyBullet physics; every image rendered by PyBullet TinyRenderer; "
        "model scale lambda = 1/25 (1 model s = 5 full-scale s)",
        "illustrative": "hand-picked by the stated rule (results/e08/gifs/RULE.md); not evidence",
        "source_manifest": {
            "path": _rel(source_manifest),
            "sha256": source_sha,
        },
        "rule_sha256": source["rule_sha256"],
        "committed_row_sources": dict(sources),
        "picks": {cat: source["picks"][cat] for cat in picks},
        "panels": order,
        "arm_labels": {p: ARM_LABELS[p] for p in order},
        "camera": {
            "class": "rld.viz.gifs.Camera (via ila.p8_adapter.OutsideCamera)",
            **asdict(CAMERA),
            "renderer": "ER_TINY_RENDERER, shadow 1, light direction [1.0, -0.6, 2.0] "
            "(fixed in P8's Camera.grab)",
            "units": "metres model scale, world frame; degrees",
            "drone_tint_rgba_outside_view_only": list(DRONE_RGBA),
        },
        "onboard": {"source": "the frame each policy was given", "zoom": ONBOARD_ZOOM},
        "frames": {
            "requested_stride_steps": stride,
            "control_step_ms_model_scale": round(1000.0 / CTRL_HZ, 3),
            "durations": "frame end times rounded to 10 ms GIF ticks (real time at model "
            "scale; x5 slower at full scale)",
            "hold_final_ms": HOLD_MS,
            "terminal_state": "not drawn: frames are the states each policy acted on "
            "(steps 0 ... n-1); the state after the last env.step is <= 33.3 ms later",
            "palette_colours": PALETTE_COLOURS,
        },
        "reflight": "every panel re-flown on the CPU fp32 single-sample path with the outside "
        "view; every row asserted equal to its committed row before any GIF was written",
        "gifs": gif_records,
        "total_bytes": total,
        "limits_bytes": {"per_gif": GIF_LIMIT_BYTES, "total": TOTAL_LIMIT_BYTES},
        "ila_commit": _ila_commit(),
        "dirty": bool(dirty),
        "p8_commit": p8_adapter.p8_commit(),
        "smoke": only is not None or tuple(order) != gifs.PANELS,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if status is not None:
        status.update("done", total_bytes=total, gifs=[g["file"] for g in gif_records])
    return record


def _check_text(h: HeroResult) -> str:
    return (
        "P8 compare_to_reference on P8's columns: identical"
        if h.result.panel.panel == "teacher"
        else "whole committed CSV line: identical"
    )


def _ila_commit() -> str:
    """This checkout's HEAD commit."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)
