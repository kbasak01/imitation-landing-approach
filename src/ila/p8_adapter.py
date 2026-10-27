"""The only module in ``ila`` that imports P8 (``rld``).

P8 (``external/rl-land-on-moving-deck``) is a pinned, read-only submodule. Everything this
project takes from it -- configs, frozen episode lists, the observation layout, the
environment, teachers and statistics -- passes through here, unchanged. Every access to a
private ``rld`` attribute (``env._platform``, ``env._substep``) is confined to this module,
so a P8 refactor breaks one file. Lengths are metres and times seconds, model scale (Froude
lambda = 1/25: 1 model s = 5 full-scale s, 1 model m = 25 full-scale m), unless stated.

Names taken from P8's source (pinned commit, P0-D1), not from memory:

* observation layout: ``rld.envs.observation.obs_fields(cfg) -> tuple[ObsField, ...]``. P8
  has **no** ``OBS_FIELDS`` constant; the layout depends on the observation config, whose
  optional ``last_action`` and ``in_contact`` blocks are both on in the eval config (25-d).
* statistics: every name in ``rld.eval.stats.__all__``, held by reference in :data:`stats`.
* teachers: ``rld.rl.train.build_policy`` (PPO) and ``rld.eval.runner.controller_spec``
  (registry controllers), exactly as P8's evaluation builds them.
* the runner contract (Phase 2): ``rld.eval.runner``'s public constants, ``PolicySpec``,
  ``controller_spec``, ``callable_spec``, ``episode_rows_csv``, ``feed_skip_reason``,
  ``split_runnable`` and ``EpisodeListMismatchError``; ``rld.eval.envs.motion_for`` /
  ``pad_offset_for``; ``rld.control.base.PrivilegedContext``; ``rld.eval.truth``'s
  ``td_in_quiescent_window`` / ``touchdown_rule``; ``rld.envs.landing_env.ACTION_DIM``;
  ``rld.eval.reproduce.compare_to_reference``; ``rld.eval.storage.read_text``;
  ``rld.deck.splits.dev_pool``; for the splits (Phase 3), ``rld.eval.episodes``'s
  ``EpisodeDraw``, ``resolve_draws``, ``label_entropy``, ``cell_order``, ``canonical_csv``,
  ``content_sha256`` and ``DEFAULT_PAD``; for teacher label parity, ``rld.deploy.parity``'s
  ``reference_actions``, ``open_session``, ``check_ort``, ``PARITY_TOLERANCE`` and
  ``rld.deploy.onnx_policy``'s ``cpu_session`` / ``attach_onnx``. ``realization_key`` is dmf's
  (``dmf.data.splits``), the very object P8's runner imports. All are held by reference
  (``tests/test_p8_adapter.py`` asserts ``is``-identity). :func:`set_motion` is the one
  helper added here, a transcription of the ``set_motion`` call in P8's ``run_chunk``.
"""

from __future__ import annotations

import functools
import hashlib
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pybullet as pyb
import yaml
from dmf.data.splits import realization_key
from rld.config import REPO_ROOT as P8_ROOT
from rld.control.base import PrivilegedContext
from rld.deck.splits import dev_pool
from rld.deploy.onnx_policy import attach_onnx, cpu_session
from rld.deploy.parity import PARITY_TOLERANCE, check_ort, open_session, reference_actions
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.envs.observation import OBS_DTYPE, ObsField
from rld.envs.observation import obs_fields as _p8_obs_fields
from rld.envs.platform import PAD_MARKER_HEIGHT_M, PAD_MARKER_RGBA, PLATE_RGBA
from rld.eval import stats as _p8_stats
from rld.eval import storage as _p8_storage
from rld.eval.envs import EvalConfigs, MotionKind, load_eval_configs, motion_for, pad_offset_for
from rld.eval.envs import make_env as _p8_make_env
from rld.eval.episodes import (
    DEFAULT_PAD,
    EpisodeDraw,
    ListedEpisode,
    canonical_csv,
    cell_order,
    content_sha256,
    label_entropy,
    list_names,
    read_list,
    resolve_draws,
)
from rld.eval.reproduce import compare_to_reference
from rld.eval.runner import (
    DEFAULT_CHUNK,
    DEFAULT_WORKERS,
    EPISODE_COLUMNS,
    LIST_ROW_COLUMNS,
    RECORD_COLUMNS,
    EpisodeListMismatchError,
    EvalPolicy,
    PolicySpec,
    callable_spec,
    controller_spec,
    episode_rows_csv,
    feed_skip_reason,
    split_runnable,
)
from rld.eval.truth import td_in_quiescent_window, touchdown_rule
from rld.rl.train import build_policy

__all__ = [
    "ACTION_DIM",
    "DEFAULT_CHUNK",
    "DEFAULT_PAD",
    "DEFAULT_WORKERS",
    "EPISODE_COLUMNS",
    "LIST_ROW_COLUMNS",
    "OBS_DTYPE",
    "P8_ROOT",
    "PARITY_TOLERANCE",
    "PAD_MARKER_HEIGHT_M",
    "PAD_MARKER_RGBA",
    "PLATE_RGBA",
    "RECORD_COLUMNS",
    "REPO_ROOT",
    "TEACHERS",
    "DeckLandingAviary",
    "EpisodeDraw",
    "EpisodeListMismatchError",
    "EvalConfigs",
    "EvalPolicy",
    "ListedEpisode",
    "MotionKind",
    "ObsField",
    "P8Stats",
    "PolicySpec",
    "PrivilegedContext",
    "SceneHandles",
    "attach_onnx",
    "callable_spec",
    "canonical_csv",
    "check_ort",
    "cell_order",
    "compare_to_reference",
    "content_sha256",
    "controller_spec",
    "cpu_session",
    "dev_pool",
    "episode_rows_csv",
    "feed_skip_reason",
    "label_entropy",
    "list_names",
    "list_path",
    "list_paths",
    "load_eval_configs",
    "load_paths",
    "make_env",
    "manifest_path",
    "manifest_sha256",
    "motion_for",
    "obs_fields",
    "open_session",
    "p8_commit",
    "p8_submodule_commits",
    "pad_offset_for",
    "physics_step_count",
    "read_list",
    "read_text",
    "realization_key",
    "reference_actions",
    "resolve_draws",
    "scene_handles",
    "set_motion",
    "sha256_file",
    "split_runnable",
    "stats",
    "success_yaml_path",
    "success_yaml_sha256",
    "td_in_quiescent_window",
    "teacher",
    "touchdown_rule",
]

#: P8's ``rld.eval.storage.read_text``: the uncompressed text of a logical CSV, reading its
#: ``.csv.gz`` sibling when present (P8 P7-D1a §12). Held by reference.
read_text: Final = _p8_storage.read_text

#: This repository's root, resolved from this file so nothing depends on the working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"
_VAR = re.compile(r"\$\{(\w+)\}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paths(path: Path = _PATHS_FILE) -> dict[str, Path]:
    """Load ``configs/paths.local.yaml``, expanding ``${key}`` references to earlier keys.

    Args:
        path: The local paths file (gitignored; copy of ``paths.local.example.yaml``).

    Returns:
        Key -> absolute path; relative entries resolve against this repository's root.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a ``${key}`` reference names an undefined key.
    """
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = _VAR.sub(lambda m: resolved[m.group(1)], str(value))
    return {k: (REPO_ROOT / v).resolve() for k, v in resolved.items()}


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout
    return out.strip()


def p8_commit() -> str:
    """Return the P8 submodule's checked-out commit SHA (40 hex characters)."""
    return _git("rev-parse", "HEAD", cwd=P8_ROOT)


def p8_submodule_commits() -> dict[str, str]:
    """Return P8's nested submodules (``third_party/*``) as path -> checked-out commit SHA."""
    commits: dict[str, str] = {}
    for line in _git("submodule", "status", cwd=P8_ROOT).splitlines():
        # Each line is "<status char><sha> <path> [(<describe>)]"; the status char is one of
        # " +-U", and the leading space of the first line is lost to strip().
        sha, sub_path = line.split()[:2]
        commits[sub_path] = sha.lstrip("+-U")
    return commits


def success_yaml_path() -> Path:
    """Return the path of P8's frozen ``configs/env/success.yaml``."""
    return P8_ROOT / "configs" / "env" / "success.yaml"


def success_yaml_sha256() -> str:
    """Return the SHA-256 of P8's frozen success criteria file."""
    return sha256_file(success_yaml_path())


def manifest_path() -> Path:
    """Return the path of P8's frozen episode-list ``results/episodes/MANIFEST.csv``."""
    return P8_ROOT / "results" / "episodes" / "MANIFEST.csv"


def manifest_sha256() -> str:
    """Return the SHA-256 of P8's frozen episode-list manifest."""
    return sha256_file(manifest_path())


def list_path(name: str) -> Path:
    """Return the path of one of P8's frozen episode lists, e.g. ``"id"`` or ``"static"``."""
    return P8_ROOT / "results" / "episodes" / f"{name}.parquet"


def make_env(cfgs: EvalConfigs, listed: ListedEpisode) -> DeckLandingAviary:
    """Build P8's landing environment for one listed episode, exactly as P8's runner does.

    Args:
        cfgs: P8's committed configs (:func:`load_eval_configs`).
        listed: One row of a frozen list (:func:`read_list`).

    Returns:
        P8's ``DeckLandingAviary`` (240 Hz physics, 30 Hz control). The caller resets it with
        ``seed=listed.episode_seed`` and closes it.
    """
    motion = motion_for(
        cfgs,
        listed.vessel,
        listed.ss,
        listed.heading_deg,
        listed.speed_kn,
        listed.realization_seed,
        episode_seed=listed.episode_seed,
    )
    return _p8_make_env(cfgs, motion, listed.vessel, listed.pad, listed.episode_seed)


def set_motion(
    env: DeckLandingAviary,
    cfgs: EvalConfigs,
    listed: ListedEpisode,
    pad: str | None = None,
    *,
    kind: MotionKind = "jonswap",
) -> tuple[str, tuple[float, float, float]]:
    """Re-point a reused environment at one listed episode, exactly as P8's ``run_chunk`` does.

    A transcription of the per-episode ``env.set_motion(motion_for(...), pad, offset)`` call in
    ``rld.eval.runner.run_chunk`` (pinned commit): the motion source of the listed realization
    (``motion_for`` with the listed ``episode_seed``), the effective pad and that pad's standing
    lever arm (``pad_offset_for``). The new source takes effect at the next ``env.reset()``.

    Args:
        env: A P8 environment (built by :func:`make_env`), between episodes.
        cfgs: P8's committed configs.
        listed: The listed episode about to be flown.
        pad: The effective pad, or ``None`` for the listed pad (the frozen aft pad).
        kind: P8's deck-motion model; ``"jonswap"`` (the frozen lists as committed).

    Returns:
        ``(pad, pad_offset_m)``: the effective pad and its lever arm from the ship's CG,
        metres model scale (lambda = 1/25), which P8's start check compares with the env's.
    """
    effective = listed.pad if pad is None else pad
    offset = pad_offset_for(cfgs, listed.vessel, effective)
    env.set_motion(
        motion_for(
            cfgs,
            listed.vessel,
            listed.ss,
            listed.heading_deg,
            listed.speed_kn,
            listed.realization_seed,
            kind=kind,
            episode_seed=listed.episode_seed,
        ),
        effective,
        offset,
    )
    return effective, offset


def list_paths() -> dict[str, Path]:
    """Return every frozen episode list P8 committed, in P8's manifest order.

    Returns:
        List name (``"id"``, ``"unseen_seastate"``, ``"unseen_heading"``,
        ``"unseen_vessel"``, ``"static"``) -> path of its parquet file. Evaluation only
        (non-negotiable 4): no list here is ever trained, tuned or selected on.
    """
    return {name: list_path(name) for name in list_names()}


@functools.cache
def obs_fields() -> tuple[ObsField, ...]:
    """Return P8's observation layout under P8's committed evaluation config.

    This is ``rld.envs.observation.obs_fields(load_eval_configs().observation)``: the 25-d
    layout with both optional blocks (``last_action``, ``in_contact``) on, in vector order.
    Offsets are the running sums of ``ObsField.size``. Units and frames are P8's, per field
    (``ObsField.unit``, ``ObsField.frame``); lengths metres and speeds m/s, model scale.

    Returns:
        The fields, in the order they appear in every observation vector.
    """
    return _p8_obs_fields(load_eval_configs().observation)


# --------------------------------------------------------------------------- statistics


class _ReadOnly(type):
    """Metaclass that refuses attribute assignment on the class itself."""

    def __setattr__(cls, name: str, value: object) -> None:
        raise AttributeError(f"{cls.__name__} is read-only; cannot set {name!r}")

    def __delattr__(cls, name: str) -> None:
        raise AttributeError(f"{cls.__name__} is read-only; cannot delete {name!r}")


class P8Stats(metaclass=_ReadOnly):
    """P8's statistics (``rld.eval.stats.__all__``), held by reference and read-only.

    Every attribute *is* P8's object (``stats.iqm is rld.eval.stats.iqm``;
    ``tests/test_p8_adapter.py`` asserts identity for every name in P8's ``__all__``), so no
    statistic in this project can drift from P8's. Use the module-level instance
    :data:`stats`. Neither the class nor the instance accepts attribute assignment.

    Note: P8's ``DEFAULT_BOOTSTRAP_SEED`` is P8's own; this project's bootstrap seed is a
    protocol decision (P3-D1) passed explicitly by callers.
    """

    __slots__ = ()

    # Constants.
    DEFAULT_BOOTSTRAP_SEED: Final = _p8_stats.DEFAULT_BOOTSTRAP_SEED
    PAIRED_REPS: Final = _p8_stats.PAIRED_REPS
    STRATIFIED_REPS: Final = _p8_stats.STRATIFIED_REPS
    WILSON_Z_95: Final = _p8_stats.WILSON_Z_95
    # Result types.
    DropContrast: Final = _p8_stats.DropContrast
    DropMode: Final = _p8_stats.DropMode
    IntervalEstimate: Final = _p8_stats.IntervalEstimate
    PairedContrast: Final = _p8_stats.PairedContrast
    RelativeP95: Final = _p8_stats.RelativeP95
    # Functions (staticmethod, so attribute access returns P8's function itself).
    aggregate_estimates: Final = staticmethod(_p8_stats.aggregate_estimates)
    drop_difference_bootstrap: Final = staticmethod(_p8_stats.drop_difference_bootstrap)
    iqm: Final = staticmethod(_p8_stats.iqm)
    optimality_gap: Final = staticmethod(_p8_stats.optimality_gap)
    paired_bootstrap: Final = staticmethod(_p8_stats.paired_bootstrap)
    paired_iqm_difference: Final = staticmethod(_p8_stats.paired_iqm_difference)
    paired_relative_p95: Final = staticmethod(_p8_stats.paired_relative_p95)
    quantile_linear: Final = staticmethod(_p8_stats.quantile_linear)
    stratified_bootstrap_ci: Final = staticmethod(_p8_stats.stratified_bootstrap_ci)
    wilson_interval: Final = staticmethod(_p8_stats.wilson_interval)


#: The read-only namespace of P8's statistics; ``stats.wilson_interval(k, n)`` etc.
stats: Final = P8Stats()


# --------------------------------------------------------------------------- teachers

#: Teachers this project distils from (plan §1.4), mapped to the seeds that exist.
#: ``ppo`` is P8's seed-0 run, checkpoint ``final/`` (P0-D1); the registry controllers are
#: deterministic and carry P8's ``run_seed`` 0.
TEACHERS: Final[Mapping[str, tuple[int, ...]]] = {
    "ppo": (0,),
    "pid_feedforward": (0,),
    "gated": (0,),
}

_PPO_RUN_KEYS: Final[Mapping[int, str]] = {0: "p8_ppo_seed0_run"}


def teacher(name: str, seed: int = 0, cfgs: EvalConfigs | None = None) -> EvalPolicy:
    """Build one of P8's teachers exactly as P8's evaluation builds it.

    * ``ppo``: ``rld.rl.train.build_policy(cfgs, run_dir=<p8_ppo_seed0_run>,
      ckpt="final")``, the SB3 model on CPU with P8's frozen ``VecNormalize`` statistics
      (``training=False``). The run directory comes from ``configs/paths.local.yaml``.
    * ``pid_feedforward``, ``gated``: ``rld.eval.runner.controller_spec(name).build(cfgs)``;
      both are non-privileged registry controllers with per-episode state, so the caller
      must ``reset`` them on every episode and call ``act`` once per step, in order.

    Every teacher consumes P8's full 25-d observation (privileged). It returns a ``(3,)``
    normalised world-frame velocity setpoint in ``[-1, 1]``; the env scales it by
    v_max = 1.5 m/s model scale with a norm cap.

    Args:
        name: ``"ppo"``, ``"pid_feedforward"`` or ``"gated"``.
        seed: Training seed. Only ``0`` exists for ``ppo`` here (P0-D1); the deterministic
            controllers accept only P8's ``run_seed`` 0.
        cfgs: P8's configs; the committed ones (:func:`load_eval_configs`) when ``None``.

    Returns:
        A policy satisfying P8's ``EvalPolicy`` (``reset(seed, context=None)``,
        ``act(obs)``). It is never privileged (``privileged is False``).

    Raises:
        ValueError: If ``name`` is not a teacher or ``seed`` is not available for it.
    """
    if name not in TEACHERS:
        raise ValueError(f"unknown teacher {name!r}; expected one of {sorted(TEACHERS)}")
    if seed not in TEACHERS[name]:
        raise ValueError(
            f"teacher {name!r} seed {seed} is not available; this project uses seeds "
            f"{TEACHERS[name]} only (ppo: P8's seed-0 run, P0-D1; controllers: run_seed 0)"
        )
    cfgs = load_eval_configs() if cfgs is None else cfgs
    if name == "ppo":
        run_dir = load_paths()[_PPO_RUN_KEYS[seed]]
        policy: EvalPolicy = build_policy(cfgs, run_dir=run_dir, ckpt="final")
        return policy
    spec = controller_spec(name)
    if spec.privileged or spec.needs_motion_feed:
        raise ValueError(f"{name!r} is privileged or needs a motion feed; not a teacher here")
    return spec.build(cfgs)


# --------------------------------------------------------------------------- scene handles


@dataclass(frozen=True)
class SceneHandles:
    """PyBullet ids of P8's scene, valid until the next ``env.reset()``.

    ``env.reset()`` calls ``resetSimulation``, which destroys every body, so these must be
    re-read after every reset. Body ids are assigned in creation order (plane, drone, plate,
    pad marker), and anything this project adds gets a later id.

    Attributes:
        client: PyBullet physics client id (DIRECT).
        drone: Drone body id. Loaded with ``URDF_IGNORE_VISUAL_SHAPES``, so no mesh; PyBullet
            nevertheless reports and TinyRenderer draws its collision cylinder (length
            0.025 m, radius 0.06 m, white) as a visual (P1-D1).
        plate: Deck plate body id: ``GEOM_BOX`` collision + uniform-colour visual
            (``PLATE_RGBA``), body origin on the top surface at the pad centre, driven
            kinematically every physics substep.
        plane: Ground plane body id (``plane.urdf``, checkerboard-textured visual).
        pad_marker: P8's painted-pad disc: a separate mass-0, collision-less
            ``GEOM_CYLINDER`` body (radius = pad radius, length ``PAD_MARKER_HEIGHT_M``)
            created once at the reset pose and never re-posed by P8, so a world-fixed visual
            once the deck moves.
        pad_radius_m: Painted-pad radius, metres model scale (``configs/deck/pad.yaml``).
        plate_half_extents_m: Plate half extents ``(hx, hy, hz)``, metres model scale.
    """

    client: int
    drone: int
    plate: int
    plane: int
    pad_marker: int
    pad_radius_m: float
    plate_half_extents_m: tuple[float, float, float]


def _is_pad_marker(client: int, body: int, pad_radius_m: float) -> bool:
    """Return whether one body is P8's painted-pad disc (matched on geometry, not colour).

    Args:
        client: PyBullet client id.
        body: Body id.
        pad_radius_m: Painted-pad radius, metres model scale.

    Returns:
        True for a collision-less body whose only visual is a ``GEOM_CYLINDER`` of radius
        ``pad_radius_m`` and length ``PAD_MARKER_HEIGHT_M``. Colour is not matched, so a
        marker this project has recoloured or hidden is still found.
    """
    if pyb.getCollisionShapeData(body, -1, physicsClientId=client):
        return False
    visuals = pyb.getVisualShapeData(body, physicsClientId=client)
    if len(visuals) != 1:
        return False
    geometry, dims = int(visuals[0][2]), visuals[0][3]
    # Cylinder dimensions are reported as (length, radius, 0) in PyBullet 3.2.7.
    return (
        geometry == pyb.GEOM_CYLINDER
        and abs(float(dims[0]) - PAD_MARKER_HEIGHT_M) < 1e-9
        and abs(float(dims[1]) - pad_radius_m) < 1e-9
    )


def scene_handles(env: DeckLandingAviary) -> SceneHandles:
    """Return the ids of P8's scene bodies for the current episode (call after reset).

    Args:
        env: A P8 environment that has been reset.

    Returns:
        The :class:`SceneHandles`.

    Raises:
        RuntimeError: If the plate is not spawned or not exactly one pad marker exists.
    """
    client = int(env.CLIENT)
    platform = env._platform  # private P8 attribute; read here only
    plate = int(platform.body_id)
    if plate < 0:
        raise RuntimeError("scene_handles() before env.reset(): the plate is not spawned")
    drone = int(env.DRONE_IDS[0])
    plane = int(env.PLANE_ID)
    radius = float(platform.pad_radius_m)
    known = {drone, plate, plane}
    markers = [
        body
        for body in (
            pyb.getBodyUniqueId(i, physicsClientId=client)
            for i in range(pyb.getNumBodies(physicsClientId=client))
        )
        if body not in known and _is_pad_marker(client, body, radius)
    ]
    if len(markers) != 1:
        raise RuntimeError(f"expected exactly one P8 pad marker, found {markers}")
    hx, hy, hz = (float(v) for v in platform.cfg.half_extents_m)
    return SceneHandles(
        client=client,
        drone=drone,
        plate=plate,
        plane=plane,
        pad_marker=int(markers[0]),
        pad_radius_m=radius,
        plate_half_extents_m=(hx, hy, hz),
    )


def physics_step_count(env: DeckLandingAviary) -> int:
    """Return the number of ``stepSimulation`` calls since the last ``env.reset()``.

    This is P8's substep counter (240 Hz physics): 0 after reset and 8 per completed
    ``env.step`` (fewer only if the episode's hard substep cap cut a step short). Unlike
    gym-pybullet-drones' ``step_counter``, it counts substeps actually taken. An image and
    an observation paired at control step t must report the same value (``8 t`` in the
    uncapped case).

    Args:
        env: A P8 environment.

    Returns:
        Physics substeps since reset, dimensionless.
    """
    return int(env._substep)  # private P8 attribute; read here only
