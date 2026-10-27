"""Vision-aware twin of P8's evaluation runner: P8's contract, plus a camera.

Contract (P8's, ``rld.eval.runner`` at the pinned commit, kept verbatim)
-----------------------------------------------------------------------
* **The list is the only source of episodes.** Every reset uses the listed episode seed on
  the listed realization, and on **every** reset :func:`check_start` asserts that ``t0``,
  the drone's initial position, the realization key and the pad (with its lever arm) equal
  the list's, or raises P8's ``EpisodeListMismatchError``.
* **Privilege comes from the spec.** A ``PrivilegedContext`` is built and passed to ``reset``
  only under a privileged spec, and a policy whose declared flags differ from its spec's is
  refused (:func:`check_privilege`). A vision student is never privileged.
* **No ship-motion feed.** No teacher or student of this project consumes one, so a
  ``needs_motion_feed`` spec is refused outright (:class:`MotionFeedRefusedError`); P8's
  ``_make_feed`` / ``_advance_feed`` are therefore not re-implemented.
* **Rows do not depend on ``workers`` or ``chunk``.** The list is cut into contiguous chunks
  of ``chunk`` episodes whatever the worker count; each chunk builds one environment and one
  policy, re-points the environment with ``set_motion`` between episodes, and rows are
  reassembled in list order.

The loop (:func:`fly_chunk`, mirroring P8's ``run_chunk`` line for line)
-----------------------------------------------------------------------
One env per chunk (``p8_adapter.make_env`` on the chunk's first episode), one policy. Per
episode: ``set_motion`` -> ``env.reset(seed)`` -> start check -> ``scene.attach(env)`` ->
render frame 0 -> context -> ``policy.reset``; per control step k: act -> record the clipped
action -> ``env.step(action)`` -> render frame k+1; after the episode: the ground truth
(``td_in_quiescent_window`` on ``PrivilegedContext.from_env``) -> the row. Frame k is taken
strictly after ``env.reset()`` / the k-th ``env.step()`` has returned, so it shares its
``stepSimulation`` count (``8 k``, model 30 Hz control / 240 Hz physics) with ``obs[k]``.

Render modes (:data:`RENDER_MODES`): ``off`` (none), ``camera`` (TinyRenderer frame from the
body-fixed camera, P8's scene untouched; a diagnostic for the bisection only) and ``full``
(``scene.attach``: ground recolour, P8's marker hidden, the visual twin; the reproduction gate
and every student run with it). Teachers get frames and discard them.

Students
--------
A policy with ``act_vision`` is a student. At step t it receives
``act_vision(si.images, si.proprio)`` where ``si = contract.student_input(frames, obs_window)``
(:meth:`ila.obs.split.ObsContract.student_input`): ``Frame.rgb`` pixels of steps
``t-To+1 ... t`` (indices < 0 repeat index 0) and the allowlisted proprio of the same steps. It
never sees the 25-d observation, a ``RenderMeta`` or a step count; its ``act`` is never called.
Students require render mode ``full`` (``camera`` frames show P8's world-fixed checkerboard).

Timing assertion
----------------
On sampled steps (k = 0, every ``timing_every``-th step and the terminal step) the frame is
taken with ``Scene.render_with_meta()`` and its ``physics_step_count`` must equal the env's
counter, which must be ``min(8 k, max_substeps)``; :class:`RenderTimingError` otherwise. The
meta is a local of :func:`_timed_frame` and is never returned, yielded or stored (P1-D2).

Rows
----
:data:`VISION_EPISODE_COLUMNS`: P8's ``EPISODE_COLUMNS`` verbatim and in P8's order, then
:data:`P9_COLUMNS` (plan §1.7), then :data:`RENDER_EVIDENCE_COLUMNS` (frames rendered and timed
checks per episode; with render on, :class:`FrameCountError` unless frames == steps + 1).
:func:`p8_rows` strips everything after P8's columns for ``compare_to_reference``. Units are P8's:
times seconds model scale, lengths metres model scale, speeds m/s model scale, angles degrees,
actions normalised (x v_max = 1.5 m/s model). Model scale lambda = 1/25 (1 model s = 5 full-scale s;
1 model m = 25 full-scale m).
"""

from __future__ import annotations

import csv
import hashlib
import io
import multiprocessing as mp
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import pybullet as pyb
from numpy.typing import NDArray

from ila import p8_adapter
from ila.obs.split import ObsContract, load_contract
from ila.policies.dp import chunking
from ila.render import scene as render_scene
from ila.render.camera import Frame, render_pose
from ila.render.config import RENDER_CONFIG_PATH, RenderConfig, load_render_config

__all__ = [
    "DEFAULT_TIMING_EVERY",
    "P9_COLUMNS",
    "RENDER_EVIDENCE_COLUMNS",
    "RENDER_MODES",
    "THREAD_ENV_VARS",
    "VISION_EPISODE_COLUMNS",
    "Actor",
    "FrameCountError",
    "MotionFeedRefusedError",
    "RenderMode",
    "RenderTimingError",
    "RowTags",
    "StudentActor",
    "StudentWindow",
    "TeacherActor",
    "VisionArm",
    "VisionTask",
    "check_privilege",
    "check_start",
    "chunk_tasks",
    "episode_row",
    "fly_chunk",
    "init_worker_threads",
    "is_student",
    "none_to_nan",
    "p8_rows",
    "render_cfg_sha256",
    "run_arms_vision",
    "run_chunk_vision",
    "run_list_vision",
    "set_single_thread_env",
    "vision_rows_csv",
]

#: How a chunk renders (module docstring).
type RenderMode = Literal["off", "camera", "full"]

#: Every :data:`RenderMode`.
RENDER_MODES: Final[tuple[str, ...]] = ("off", "camera", "full")

#: This project's columns after P8's (plan §1.7), in order.
P9_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "teacher",
    "student_seed",
    "Ta",
    "K",
    "ensemble",
    "latency_steps",
    "render_cfg_sha256",
    "ckpt_sha256",
)

#: Render evidence per episode, after the P9 columns (never P8 columns: ``p8_rows`` drops them).
#: ``frames_rendered`` counts the frames ``fly_chunk`` actually received from the renderer
#: (``steps + 1`` whenever render is not ``off``, 0 for ``off``); ``timed_checks`` counts the
#: frames that passed the step-count assertion (step 0, every ``timing_every``-th step, the
#: terminal step).
RENDER_EVIDENCE_COLUMNS: Final[tuple[str, ...]] = ("frames_rendered", "timed_checks")

#: Column order of every per-episode table this runner writes.
VISION_EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    *p8_adapter.EPISODE_COLUMNS,
    *P9_COLUMNS,
    *RENDER_EVIDENCE_COLUMNS,
)

#: Thread-count variables set to 1 before any worker pool is spawned (P8 run conditions:
#: ``OMP_NUM_THREADS=1``, and ``MKL_NUM_THREADS=1`` from ``rld.eval.learned.main``).
THREAD_ENV_VARS: Final[tuple[str, ...]] = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")

#: Default stride of the timing assertion, control steps (1 s model at 30 Hz).
DEFAULT_TIMING_EVERY: Final[int] = 30


class RenderTimingError(AssertionError):
    """A frame and its observation do not share one ``stepSimulation`` count."""


class FrameCountError(AssertionError):
    """Render was on, but an episode did not get exactly one frame per reset and per step."""


class MotionFeedRefusedError(ValueError):
    """A ``needs_motion_feed`` spec was handed to this runner (no arm here consumes a feed)."""


# --------------------------------------------------------------------------- tags and arms


@dataclass(frozen=True)
class RowTags:
    """This project's per-row columns (plan §1.7); ``None`` is written as an empty cell.

    Attributes:
        arm: Arm label, e.g. ``"bc"``, ``"dp"``; for a re-flown P8 method, its method name.
        teacher: The teacher an arm was distilled from (``""`` for a teacher itself).
        student_seed: Training seed of a student, or ``None``.
        Ta: Executed actions per chunk (Diffusion Policy), or ``None``.
        K: DDIM denoising steps (Diffusion Policy), or ``None``.
        ensemble: Temporal ensembling on, or ``None`` when not applicable.
        latency_steps: Control steps of injected decision latency (latency arm; 0 otherwise),
            30 Hz control steps, model scale.
        ckpt_sha256: SHA-256 of the checkpoint flown, or ``""`` for a controller.
    """

    arm: str
    teacher: str = ""
    student_seed: int | None = None
    Ta: int | None = None
    K: int | None = None
    ensemble: bool | None = None
    latency_steps: int = 0
    ckpt_sha256: str = ""


@dataclass(frozen=True)
class VisionArm:
    """One policy on one ordered list of listed episodes, rendered one way.

    Attributes:
        spec: P8's ``PolicySpec`` (method, privileged, picklable build, run seed). Its
            ``needs_motion_feed`` must be False.
        episodes: Listed episodes, in committed order.
        tags: The P9 columns.
        render: ``off``, ``camera`` or ``full``.
        appearance_shift: ``None`` for the committed look, else an evaluation-only variant
            (``ila.render.config.APPEARANCE_SHIFTS``).
        contract_variant: The observation contract variant a student is fed
            (``ila.obs.split.VARIANTS``).
        timing_every: Stride of the timing assertion, control steps.
    """

    spec: p8_adapter.PolicySpec
    episodes: Sequence[p8_adapter.ListedEpisode]
    tags: RowTags
    render: RenderMode = "full"
    appearance_shift: str | None = None
    contract_variant: str = "default"
    timing_every: int = DEFAULT_TIMING_EVERY


@dataclass(frozen=True)
class VisionTask:
    """One picklable unit of work: a contiguous chunk of one arm.

    Attributes:
        episodes: The chunk, in committed order.
        spec: What to fly.
        cfgs: P8's configs.
        tags: The P9 columns.
        render: Render mode.
        appearance_shift: Render variant, or ``None``.
        contract_variant: Student contract variant.
        timing_every: Stride of the timing assertion, control steps.
    """

    episodes: Sequence[p8_adapter.ListedEpisode]
    spec: p8_adapter.PolicySpec
    cfgs: p8_adapter.EvalConfigs
    tags: RowTags
    render: RenderMode = "full"
    appearance_shift: str | None = None
    contract_variant: str = "default"
    timing_every: int = DEFAULT_TIMING_EVERY


def render_cfg_sha256(render: RenderMode, appearance_shift: str | None = None) -> str:
    """Return the ``render_cfg_sha256`` cell for a render mode.

    Args:
        render: The render mode.
        appearance_shift: The variant, or ``None``.

    Returns:
        ``""`` for ``off``; the SHA-256 of ``configs/render/camera.yaml`` for the committed
        look (P1-D1's number); for a variant, the SHA-256 of ``"<file sha>:<variant>"``; for
        ``camera`` (P8's scene untouched, diagnostics only), ``"camera:<file sha>"``.
    """
    if render == "off":
        return ""
    file_sha = p8_adapter.sha256_file(RENDER_CONFIG_PATH)
    if render == "camera":
        return f"camera:{file_sha}"
    if appearance_shift is None:
        return file_sha
    return hashlib.sha256(f"{file_sha}:{appearance_shift}".encode()).hexdigest()


# --------------------------------------------------------------------------- P8 helpers
# Re-implementations of P8's private helpers in rld/eval/runner.py (pinned commit). Each is
# tested equal to P8's on P8's own fixture in tests/test_eval_runner.py.


def check_privilege(spec: p8_adapter.PolicySpec, policy: object) -> None:
    """Refuse a policy whose declared flags differ from its spec's (P8 ``_check_privilege``).

    Args:
        spec: The spec.
        policy: The built policy.

    Raises:
        ValueError: If the policy declares a ``privileged`` or ``needs_motion_feed`` flag
            that differs from the spec's.
    """
    declared = getattr(policy, "privileged", None)
    if declared is not None and bool(declared) != spec.privileged:
        raise ValueError(
            f"{spec.method}: policy declares privileged={declared}, spec says {spec.privileged}"
        )
    feed = getattr(policy, "needs_motion_feed", None)
    if feed is not None and bool(feed) != spec.needs_motion_feed:
        raise ValueError(
            f"{spec.method}: policy declares needs_motion_feed={feed}, spec says "
            f"{spec.needs_motion_feed}"
        )


def check_start(
    env: p8_adapter.DeckLandingAviary,
    info: Mapping[str, Any],
    row: p8_adapter.ListedEpisode,
    pad: str,
    pad_offset_m: tuple[float, float, float],
    *,
    check_t0: bool = True,
) -> None:
    """Assert the reset reproduced the listed episode on the effective pad (P8 ``_check_start``).

    Args:
        env: The environment, immediately after ``reset``.
        info: The reset's info dict.
        row: The listed episode.
        pad: The effective pad.
        pad_offset_m: That pad's standing lever arm, metres model scale.
        check_t0: Compare ``t0`` (seconds model scale) with the list's.

    Raises:
        EpisodeListMismatchError: On any difference in ``t0`` (when checked), initial
            position (metres model scale, world), realization key, pad or lever arm.
    """
    mismatch = p8_adapter.EpisodeListMismatchError
    t0 = float(env.record.t0_model_s)
    position = tuple(float(v) for v in np.asarray(env.pos[0], dtype=np.float64).reshape(3))
    key = p8_adapter.realization_key(
        row.ss, row.heading_deg, row.speed_kn, row.vessel, row.realization_seed
    )
    if check_t0 and t0 != row.t0_model_s:
        raise mismatch(f"{row.regime}/{row.ss}/{row.index}: t0 {t0!r} != listed")
    if position != row.init_xyz_m:
        raise mismatch(
            f"{row.regime}/{row.ss}/{row.index}: init {position} != listed {row.init_xyz_m}"
        )
    if info["realization_key"] != key:
        raise mismatch(f"{row.regime}/{row.ss}/{row.index}: key {info['realization_key']} != {key}")
    if env.pad != pad or tuple(env.pad_offset_m) != tuple(pad_offset_m):
        raise mismatch(
            f"{row.regime}/{row.ss}/{row.index}: env pad {env.pad!r} {env.pad_offset_m} != "
            f"effective {pad!r} {pad_offset_m}"
        )


def none_to_nan(value: Any) -> Any:
    """Return NaN for ``None`` so every numeric column is numeric (P8 ``_none_to_nan``)."""
    return float("nan") if value is None else value


def episode_row(
    row: p8_adapter.ListedEpisode,
    pad: str,
    spec: p8_adapter.PolicySpec,
    env: p8_adapter.DeckLandingAviary,
    actions: Sequence[NDArray[np.float64]],
    td_quiescent: bool | float,
) -> dict[str, Any]:
    """Flatten one finished episode into P8's committed row (P8 ``_episode_row``).

    Args:
        row: The listed episode.
        pad: The effective pad.
        spec: The evaluated policy's spec.
        env: The environment, immediately after the episode ended.
        actions: The actions sent to ``env.step``, clipped to ``[-1, 1]^3`` as the env clips
            them, normalised units.
        td_quiescent: ``td_in_quiescent_window``: a bool, or NaN without a contact touchdown.

    Returns:
        A dict with keys P8's ``EPISODE_COLUMNS``, in order.

    Raises:
        RuntimeError: If the episode ended unclassified or the record's columns changed.
    """
    record = env.record.as_row()
    if tuple(record) != p8_adapter.RECORD_COLUMNS:
        raise RuntimeError(f"EpisodeRecord.as_row() columns changed: {list(record)}")
    if record["outcome"] is None or record["termination_reason"] is None:
        raise RuntimeError(f"{row.regime}/{row.ss}/{row.index}: episode ended unclassified")
    record = {name: none_to_nan(value) for name, value in record.items()}
    contact = env.record.contact_record
    if contact is None:
        record["n_contacts"] = 0
    stack = np.asarray(actions, dtype=np.float64).reshape(-1, p8_adapter.ACTION_DIM)
    effort = float(np.mean(np.sum(stack * stack, axis=1)))
    jerk = (
        float(np.mean(np.linalg.norm(np.diff(stack, axis=0), axis=1)))
        if stack.shape[0] > 1
        else float("nan")
    )
    out: dict[str, Any] = {name: getattr(row, name) for name in p8_adapter.LIST_ROW_COLUMNS}
    out["pad"] = pad
    out.update(method=spec.method, privileged=spec.privileged, run_seed=spec.run_seed)
    out.update(record)
    out.update(
        detectors_disagree=env.record.detectors_disagree(
            env.success.detector_disagreement_window_s
        ),
        closing_speed_world_z_m_s=(
            float("nan") if contact is None else float(contact.closing_speed_world_z_m_s)
        ),
        time_to_touchdown_s=float("nan") if contact is None else float(contact.t_episode_s),
        effort_mean_sq=effort,
        action_jerk_mean=jerk,
        td_in_quiescent_window=td_quiescent,
    )
    if tuple(out) != p8_adapter.EPISODE_COLUMNS:
        raise RuntimeError(f"row columns {list(out)} != {list(p8_adapter.EPISODE_COLUMNS)}")
    return out


# --------------------------------------------------------------------------- actors


@runtime_checkable
class _VisionPolicy(Protocol):
    """A student: P8's ``EvalPolicy`` reset plus ``act_vision``."""

    def reset(self, seed: int, context: Any = None) -> None:
        """Clear per-episode state (``context`` is always ``None`` for a student)."""
        ...

    def act_vision(self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]) -> Any:
        """Return a ``(3,)`` action in normalised units from ``uint8[To, H, W, 3]`` and proprio."""
        ...


def is_student(policy: object) -> bool:
    """Return whether a built policy is a vision student (it has ``act_vision``)."""
    return callable(getattr(policy, "act_vision", None))


class Actor(Protocol):
    """What :func:`fly_chunk` asks for an action: a teacher, a student or a remote slot."""

    #: Whether the actor consumes frames (students); teachers' frames are discarded.
    needs_frames: bool

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Start an episode (calls the policy's ``reset``)."""
        ...

    def act(self, step: int, obs: NDArray[Any], frame: Frame | None) -> NDArray[np.float64]:
        """Return the raw action for control step ``step`` (before clipping), normalised."""
        ...


class TeacherActor:
    """P8's path: ``policy.reset(seed, context)``, then ``policy.act(obs)``; frames dropped.

    Attributes:
        policy: A P8 ``EvalPolicy`` (registry controller or SB3 policy).
        needs_frames: False.
    """

    needs_frames: bool = False

    def __init__(self, policy: p8_adapter.EvalPolicy) -> None:
        """Hold the policy."""
        self.policy = policy

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Reset the policy with the listed episode seed and the (spec-gated) context."""
        self.policy.reset(listed.episode_seed, context)

    def act(self, step: int, obs: NDArray[Any], frame: Frame | None) -> NDArray[np.float64]:
        """Return ``policy.act(obs)`` as float64; the frame is discarded."""
        del step, frame
        return np.asarray(self.policy.act(obs), dtype=np.float64)


class StudentWindow:
    """The frame and observation history of one episode (il-conventions rule 2).

    At step t the window is :func:`ila.policies.dp.chunking.obs_window_steps` ``(t, To)``
    (steps ``t-To+1 ... t``, indices < 0 repeat step 0), with ``To`` from the contract
    (``configs/obs/student.yaml`` ``frames.n_obs_steps``); ``chunking`` is the only place
    those indices are computed. The window is built into a
    :class:`~ila.obs.split.StudentInput` only through :meth:`ObsContract.student_input`, so the
    privileged and teacher-only fields are dropped before anything leaves.

    Attributes:
        contract: The observation contract.
    """

    def __init__(self, contract: ObsContract) -> None:
        """Hold the contract; the history starts empty."""
        self.contract = contract
        self._frames: dict[int, NDArray[np.uint8]] = {}
        self._obs: dict[int, NDArray[Any]] = {}
        self._step = -1

    def clear(self) -> None:
        """Forget the previous episode."""
        self._frames.clear()
        self._obs.clear()
        self._step = -1

    def push(self, step: int, obs: NDArray[Any], frame: Frame | None) -> Any:
        """Add step ``step`` and return the contract-built ``StudentInput`` for it.

        Args:
            step: Control step, 0 after reset; must follow the previous one.
            obs: P8's 25-d observation at the step.
            frame: The frame rendered from the same physics state.

        Returns:
            ``ObsContract.student_input(images[To], obs[To])`` over
            ``obs_window_steps(step, To)``.

        Raises:
            ValueError: If no frame is given or the steps are not consecutive.
        """
        if frame is None:
            raise ValueError("a student needs frames: render mode must be 'full'")
        if step != self._step + 1:
            raise ValueError(f"student window at step {self._step}, pushed {step}")
        self._step = step
        self._frames[step] = frame.rgb
        self._obs[step] = np.array(obs, copy=True)
        window = chunking.obs_window_steps(step, self.contract.n_obs_steps)
        for old in [s for s in self._frames if s < window[0]]:
            del self._frames[old], self._obs[old]
        return self.contract.student_input(
            np.stack([self._frames[s] for s in window]), np.stack([self._obs[s] for s in window])
        )


class StudentActor:
    """The in-worker student path: ``act_vision(si.images, si.proprio)``, nothing else.

    Attributes:
        policy: The student (has ``act_vision``; never privileged).
        window: Its frame/observation history.
        needs_frames: True.
    """

    needs_frames: bool = True

    def __init__(self, policy: _VisionPolicy, contract: ObsContract) -> None:
        """Hold the student and a fresh window."""
        self.policy = policy
        self.window = StudentWindow(contract)

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Reset the student with the listed episode seed; it is never given a context.

        Raises:
            ValueError: If a context is passed (a student is never privileged).
        """
        if context is not None:
            raise ValueError("a vision student is never privileged")
        self.window.clear()
        self.policy.reset(listed.episode_seed, None)

    def act(self, step: int, obs: NDArray[Any], frame: Frame | None) -> NDArray[np.float64]:
        """Return ``act_vision`` on the contract-built input for this step, float64."""
        si = self.window.push(step, obs, frame)
        return np.asarray(self.policy.act_vision(si.images, si.proprio), dtype=np.float64)


# --------------------------------------------------------------------------- rendering


class _Renderer:
    """One chunk's renderer: ``off``, ``camera`` (P8's scene) or ``full`` (the scene edits)."""

    def __init__(self, mode: RenderMode, cfg: RenderConfig) -> None:
        if mode not in RENDER_MODES:
            raise ValueError(f"unknown render mode {mode!r}; expected {RENDER_MODES}")
        self.mode = mode
        self.cfg = cfg
        self._scene: render_scene.Scene | None = None
        self._env: p8_adapter.DeckLandingAviary | None = None
        self.timed_checks = 0

    def attach(self, env: p8_adapter.DeckLandingAviary) -> None:
        """Apply the visual-only edits after a reset (``full``), or note the env."""
        self._env = env
        self._scene = render_scene.attach(env, self.cfg) if self.mode == "full" else None

    def frame(self, step: int, timed: bool) -> Frame | None:
        """Render the frame for control step ``step`` (after reset / the step-th step)."""
        if self.mode == "off":
            return None
        env = self._env
        if env is None:
            raise RuntimeError("render before attach")
        if timed:
            frame = _timed_frame(env, self._scene, self.cfg, step)
            self.timed_checks += 1
            return frame
        if self._scene is not None:
            return self._scene.render()
        return _camera_frame(env, self.cfg)


def _camera_frame(env: p8_adapter.DeckLandingAviary, cfg: RenderConfig) -> Frame:
    """Render the body-fixed camera over P8's untouched scene (``camera`` mode)."""
    h = p8_adapter.scene_handles(env)
    pos, quat = pyb.getBasePositionAndOrientation(h.drone, physicsClientId=h.client)
    frame, _ = render_pose(h.client, pos, quat, cfg.camera)
    return frame


def _timed_frame(
    env: p8_adapter.DeckLandingAviary,
    sc: render_scene.Scene | None,
    cfg: RenderConfig,
    step: int,
) -> Frame:
    """Render one frame and assert it shares the observation's ``stepSimulation`` count.

    The ``RenderMeta`` is a local here and is dropped on return (P1-D2): only the
    :class:`Frame` leaves.

    Args:
        env: The environment, after ``reset`` / the ``step``-th ``env.step`` returned.
        sc: The episode's scene (``full``), or ``None`` (``camera``).
        cfg: The render config.
        step: Control step k.

    Returns:
        The frame (pixels only).

    Raises:
        RenderTimingError: If the frame's count, the env's count and ``min(8 k, cap)``
            disagree, or rendering moved the counter.
    """
    before = p8_adapter.physics_step_count(env)
    if sc is not None:
        frame, meta = sc.render_with_meta()
        rendered_at = meta.physics_step_count
        del meta
    else:
        frame = _camera_frame(env, cfg)
        rendered_at = p8_adapter.physics_step_count(env)
    after = p8_adapter.physics_step_count(env)
    stride = int(env.PYB_STEPS_PER_CTRL)
    expected = min(stride * step, int(env.cfg.max_substeps))
    if not (rendered_at == before == after == expected) or env.record.steps != step:
        raise RenderTimingError(
            f"control step {step} (env says {env.record.steps}): frame at substep "
            f"{rendered_at}, env {before} -> {after}, expected {expected}"
        )
    return frame


# --------------------------------------------------------------------------- the loop


def _refuse_feed(spec: p8_adapter.PolicySpec) -> None:
    """Raise :class:`MotionFeedRefusedError` for a ``needs_motion_feed`` spec."""
    if spec.needs_motion_feed:
        raise MotionFeedRefusedError(
            f"{spec.method}: needs_motion_feed specs are not supported by the vision runner "
            "(no teacher or student of this project consumes a ship-motion feed)"
        )


def _tag_row(
    row: dict[str, Any], tags: RowTags, render_sha: str, frames: int, timed_checks: int
) -> dict[str, Any]:
    """Append the P9 and render-evidence columns to a P8 row, in column order."""
    out = dict(row)
    out.update(
        arm=tags.arm,
        teacher=tags.teacher,
        student_seed=tags.student_seed,
        Ta=tags.Ta,
        K=tags.K,
        ensemble=tags.ensemble,
        latency_steps=tags.latency_steps,
        render_cfg_sha256=render_sha,
        ckpt_sha256=tags.ckpt_sha256,
        frames_rendered=frames,
        timed_checks=timed_checks,
    )
    if tuple(out) != VISION_EPISODE_COLUMNS:
        raise RuntimeError(f"row columns {list(out)} != {list(VISION_EPISODE_COLUMNS)}")
    return out


def fly_chunk(
    task: VisionTask,
    make_actor: Callable[[p8_adapter.EvalConfigs, ObsContract], tuple[Actor, object]],
) -> list[dict[str, Any]]:
    """Fly one chunk in one env with one actor: P8's ``run_chunk`` loop, rendered.

    Args:
        task: The chunk and how to fly it.
        make_actor: ``(cfgs, contract) -> (actor, built policy)``; the policy (or a stand-in
            carrying its flags) is checked against the spec by :func:`check_privilege`.

    Returns:
        One row per episode (:data:`VISION_EPISODE_COLUMNS`), in the order given.

    Raises:
        EpisodeListMismatchError: If any reset does not reproduce its listed episode.
        MotionFeedRefusedError: For a ``needs_motion_feed`` spec.
        RenderTimingError: If a frame and its observation do not share a step count.
        ValueError: If the policy's flags disagree with the spec, it returns a non-finite or
            wrongly shaped action, or a student is flown privileged or without ``full``.
    """
    spec, cfgs = task.spec, task.cfgs
    _refuse_feed(spec)
    if task.timing_every < 1:
        raise ValueError(f"timing_every must be positive, got {task.timing_every}")
    if not task.episodes:
        return []
    contract = load_contract(variant=task.contract_variant)
    renderer = _Renderer(task.render, load_render_config(appearance_shift=task.appearance_shift))
    render_sha = render_cfg_sha256(task.render, task.appearance_shift)
    action_dim = p8_adapter.ACTION_DIM
    env = p8_adapter.make_env(cfgs, task.episodes[0])
    try:
        actor, policy = make_actor(cfgs, contract)
        check_privilege(spec, policy)
        if actor.needs_frames:
            if spec.privileged:
                raise ValueError(f"{spec.method}: a vision student is never privileged")
            if task.render != "full":
                raise ValueError(f"{spec.method}: a student needs render 'full', not {task.render}")
        max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
        rule = p8_adapter.touchdown_rule(cfgs)
        stride = int(cfgs.landing.pyb_steps_per_ctrl)
        rows: list[dict[str, Any]] = []
        for listed in task.episodes:
            pad, offset = p8_adapter.set_motion(env, cfgs, listed)
            obs, info = env.reset(seed=listed.episode_seed)
            check_start(env, info, listed, pad, offset, check_t0=True)
            renderer.attach(env)
            timed_before = renderer.timed_checks
            frame = renderer.frame(0, timed=True)
            frames = 0 if frame is None else 1
            context = p8_adapter.PrivilegedContext.from_env(env) if spec.privileged else None
            actor.begin(listed, context)
            actions: list[NDArray[np.float64]] = []
            for k in range(max_steps):
                action = actor.act(k, obs, frame)
                if action.shape != (action_dim,) or not np.all(np.isfinite(action)):
                    raise ValueError(f"{spec.method}: bad action {action!r}")
                actions.append(np.clip(action, -1.0, 1.0))
                obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                timed = done or (k + 1) % task.timing_every == 0
                frame = renderer.frame(k + 1, timed=timed)
                frames += 0 if frame is None else 1
                if done:
                    break
            n_steps = len(actions)
            expected_frames = 0 if task.render == "off" else n_steps + 1
            if frames != expected_frames or int(env.record.steps) != n_steps:
                raise FrameCountError(
                    f"{listed.regime}/{listed.ss}/{listed.index}: render {task.render!r}, "
                    f"{frames} frames for {n_steps} steps (env {env.record.steps}); "
                    f"expected {expected_frames}"
                )
            # Evaluation ground truth, after the episode, for every method alike.
            contact = env.record.contact_record
            truth = p8_adapter.td_in_quiescent_window(
                p8_adapter.PrivilegedContext.from_env(env),
                rule,
                None if contact is None else float(contact.t_episode_s),
                stride,
            )
            row = episode_row(listed, pad, spec, env, actions, truth)
            n_timed = renderer.timed_checks - timed_before
            rows.append(_tag_row(row, task.tags, render_sha, frames, n_timed))
    finally:
        env.close()
    return rows


def _local_actor(
    spec: p8_adapter.PolicySpec,
) -> Callable[[p8_adapter.EvalConfigs, ObsContract], tuple[Actor, object]]:
    """Return the in-worker actor factory for a spec: student if it has ``act_vision``."""

    def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Actor, object]:
        policy = spec.build(cfgs)
        if is_student(policy):
            if not isinstance(policy, _VisionPolicy):
                raise TypeError(f"{spec.method}: student lacks reset/act_vision")
            return StudentActor(policy, contract), policy
        return TeacherActor(policy), policy

    return make


def run_chunk_vision(task: VisionTask) -> list[dict[str, Any]]:
    """Fly one chunk with an in-worker policy (picklable entry point for the pool).

    Args:
        task: The chunk.

    Returns:
        One row per episode, in order (:data:`VISION_EPISODE_COLUMNS`).
    """
    return fly_chunk(task, _local_actor(task.spec))


# --------------------------------------------------------------------------- parallelism


def set_single_thread_env() -> dict[str, str]:
    """Set ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` to 1 (spawned workers inherit them).

    Returns:
        The values now set, for ``run_info.json``.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = "1"
    return {var: os.environ[var] for var in THREAD_ENV_VARS}


def init_worker_threads() -> None:
    """Pin torch to one intra-op thread in this process (each worker; P8 ran threads 1)."""
    import torch

    torch.set_num_threads(1)


def chunk_tasks(
    arms: Sequence[VisionArm], cfgs: p8_adapter.EvalConfigs, chunk: int
) -> list[VisionTask]:
    """Cut every arm into contiguous chunks of ``chunk`` episodes (P8's ``run_arms`` cut)."""
    return [
        VisionTask(
            episodes=arm.episodes[i : i + chunk],
            spec=arm.spec,
            cfgs=cfgs,
            tags=arm.tags,
            render=arm.render,
            appearance_shift=arm.appearance_shift,
            contract_variant=arm.contract_variant,
            timing_every=arm.timing_every,
        )
        for arm in arms
        for i in range(0, len(arm.episodes), chunk)
    ]


def run_arms_vision(
    arms: Sequence[VisionArm],
    cfgs: p8_adapter.EvalConfigs,
    *,
    workers: int = p8_adapter.DEFAULT_WORKERS,
    chunk: int = p8_adapter.DEFAULT_CHUNK,
    on_chunk: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Fly several arms through one spawn pool; rows independent of ``workers`` and ``chunk``.

    ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` are set to 1 before the pool is spawned and
    every worker (and this process, on the in-process path) calls
    :func:`init_worker_threads`.

    Args:
        arms: What to fly, in output order.
        cfgs: P8's configs.
        workers: Processes; rows do not depend on it.
        chunk: Episodes per task; rows do not depend on it.
        on_chunk: Progress callback ``(chunks_done, chunks_total, episodes_done)``, called in
            this process in list order.

    Returns:
        Every arm's rows, arm after arm, each in its episodes' order.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
        MotionFeedRefusedError: If any arm's spec needs a motion feed.
    """
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    for arm in arms:
        _refuse_feed(arm.spec)
        if arm.render not in RENDER_MODES:
            raise ValueError(f"unknown render mode {arm.render!r}")
    tasks = chunk_tasks(arms, cfgs, chunk)
    set_single_thread_env()
    results: list[list[dict[str, Any]]] = []
    done_episodes = 0

    def note(rows: list[dict[str, Any]]) -> None:
        nonlocal done_episodes
        results.append(rows)
        done_episodes += len(rows)
        if on_chunk is not None:
            on_chunk(len(results), len(tasks), done_episodes)

    if workers == 1 or len(tasks) <= 1:
        init_worker_threads()
        for task in tasks:
            note(run_chunk_vision(task))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(tasks)), initializer=init_worker_threads) as pool:
            for rows in pool.imap(run_chunk_vision, tasks, chunksize=1):
                note(rows)
    return [row for rows in results for row in rows]


def run_list_vision(
    episodes: Sequence[p8_adapter.ListedEpisode],
    spec: p8_adapter.PolicySpec,
    cfgs: p8_adapter.EvalConfigs,
    tags: RowTags | None = None,
    *,
    render: RenderMode = "full",
    workers: int = p8_adapter.DEFAULT_WORKERS,
    chunk: int = p8_adapter.DEFAULT_CHUNK,
    appearance_shift: str | None = None,
    contract_variant: str = "default",
    timing_every: int = DEFAULT_TIMING_EVERY,
) -> list[dict[str, Any]]:
    """Fly one policy on a list of listed episodes (one arm of :func:`run_arms_vision`).

    Args:
        episodes: Listed episodes, in committed order.
        spec: What to fly.
        cfgs: P8's configs.
        tags: P9 columns; ``RowTags(arm=spec.method)`` when ``None``.
        render: Render mode.
        workers: Processes; rows do not depend on it.
        chunk: Episodes per task; rows do not depend on it.
        appearance_shift: Render variant, or ``None``.
        contract_variant: Student contract variant.
        timing_every: Stride of the timing assertion, control steps.

    Returns:
        One row per episode (:data:`VISION_EPISODE_COLUMNS`), in list order.
    """
    arm = VisionArm(
        spec=spec,
        episodes=episodes,
        tags=RowTags(arm=spec.method) if tags is None else tags,
        render=render,
        appearance_shift=appearance_shift,
        contract_variant=contract_variant,
        timing_every=timing_every,
    )
    return run_arms_vision([arm], cfgs, workers=workers, chunk=chunk)


# --------------------------------------------------------------------------- rows out


def p8_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows restricted to P8's ``EPISODE_COLUMNS``, in P8's order.

    This is what ``compare_to_reference`` is given.
    """
    return [{name: row[name] for name in p8_adapter.EPISODE_COLUMNS} for row in rows]


def _cell(value: Any) -> Any:
    """Serialise one cell as P8's ``episode_rows_csv`` does; ``None`` -> empty."""
    if value is None:
        return ""
    return repr(float(value)) if isinstance(value, float) else value


def vision_rows_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialise rows with every column (:data:`VISION_EPISODE_COLUMNS`) deterministically.

    The P8 columns are written exactly as P8's ``episode_rows_csv`` writes them (floats as
    ``repr``, booleans ``True``/``False``, newline endings), so the first
    ``len(EPISODE_COLUMNS)`` fields of each line are P8's text. ``None`` tags are empty.

    Args:
        rows: Runner rows.

    Returns:
        The CSV text, header included.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(VISION_EPISODE_COLUMNS)
    for row in rows:
        writer.writerow([_cell(row[c]) for c in VISION_EPISODE_COLUMNS])
    return buffer.getvalue()
