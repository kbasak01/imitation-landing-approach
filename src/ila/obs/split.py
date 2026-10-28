"""The only place observation fields are selected.

Splits P8's 25-d observation into the student proprio allowlist, the privileged block and
the teacher-only extras, by field **name** (``configs/obs/student.yaml``). Offsets and
sizes come from P8's layout function ``rld.envs.observation.obs_fields`` (through
:func:`ila.p8_adapter.obs_fields`) at load time; P8 has no ``OBS_FIELDS`` constant. Nothing
else in ``ila`` indexes an observation vector with literal indices
(``tests/test_obs_split.py`` scans for it).

Units and frames are P8's (``rld/envs/observation.py``): angles rad (world), body rates
rad/s (P8's table labels them "body", but its builder passes PyBullet's world-axis angular
velocity through unrotated), velocities m/s **model scale** in the drone-yaw frame (world
rotated by -yaw about z), lengths m model scale (lambda = 1/25), ``last_action``
dimensionless in [-1, 1] (world), ``in_contact`` in {0, 1}, ``time_fraction`` dimensionless
in [0, 1].

Default split (13 / 11 / 1):

* student proprio: ``attitude 3, body_rates 3, velocity 3, last_action 3, in_contact 1``;
* privileged (estimator target only): ``rel_position 3, rel_velocity 3, deck_normal 3,
  rel_tilt 1, height 1``;
* teacher-only: ``time_fraction 1``.

Two labelled ablations each move exactly one field between the student proprio and the
teacher-only block: ``plus_clock`` appends ``time_fraction`` (14-d), ``minus_last_action``
removes ``last_action`` (10-d). The five privileged names are refused in a student input
whatever the YAML says, and ``time_fraction`` is refused outside ``plus_clock``.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from ila import p8_adapter

__all__ = [
    "CLOCK_FIELD",
    "CONTRACT_PATH",
    "NEVER_STUDENT",
    "VARIANTS",
    "ObsContract",
    "StudentInput",
    "load_contract",
    "split",
]

#: The committed observation contract.
CONTRACT_PATH: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "obs" / "student.yaml"

#: P8 fields that never reach a student input, under any variant (non-negotiable 2). Held
#: here as well as in the YAML so that editing the YAML alone cannot leak one.
NEVER_STUDENT: Final[frozenset[str]] = frozenset(
    {"rel_position", "rel_velocity", "deck_normal", "rel_tilt", "height"}
)

#: The episode clock: a student input only in the labelled ``plus_clock`` ablation.
CLOCK_FIELD: Final[str] = "time_fraction"

#: Contract variants: the default and the two labelled ablations in the YAML.
VARIANTS: Final[tuple[str, ...]] = ("default", "plus_clock", "minus_last_action")

_CLOCK_VARIANT: Final[str] = "plus_clock"


def _index(layout: Mapping[str, tuple[int, int]], names: Sequence[str]) -> NDArray[np.intp]:
    """Return the vector indices of ``names``, concatenated in the given order.

    Args:
        layout: Field name -> ``(offset, size)`` in P8's observation vector.
        names: Field names.

    Returns:
        A read-only ``intp`` index array (possibly empty).
    """
    parts = [np.arange(layout[n][0], layout[n][0] + layout[n][1], dtype=np.intp) for n in names]
    out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.intp)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ObsContract:
    """Which P8 observation fields a student, an estimator and only the teacher see.

    Attributes:
        layout: Field name -> ``(offset, size)`` in P8's observation vector, in vector
            order, from P8's ``obs_fields`` (read-only mapping).
        student_proprio: Student proprio fields, in the order they are concatenated.
        privileged: Privileged fields (the ``modular`` estimator's target), in order.
        teacher_only: Fields neither a student input nor an estimator target, in order.
        n_obs_steps: ``To``, observation steps per student input (images and proprio at
            ``t-To+1 ... t``; indices < 0 repeat index 0).
        variant: ``"default"``, ``"plus_clock"`` or ``"minus_last_action"``.
    """

    layout: Mapping[str, tuple[int, int]]
    student_proprio: tuple[str, ...]
    privileged: tuple[str, ...]
    teacher_only: tuple[str, ...]
    n_obs_steps: int
    variant: str = "default"
    _proprio_idx: NDArray[np.intp] = field(init=False, repr=False, compare=False)
    _priv_idx: NDArray[np.intp] = field(init=False, repr=False, compare=False)
    _extras_idx: NDArray[np.intp] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the classification and precompute the index arrays.

        Raises:
            ValueError: If a field is unclassified, classified twice or unknown to P8; a
                privileged field is in the student proprio; the clock is in it outside
                ``plus_clock``; or ``n_obs_steps`` or ``variant`` is invalid.
        """
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown contract variant {self.variant!r}; expected {VARIANTS}")
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}")
        classified = [*self.student_proprio, *self.privileged, *self.teacher_only]
        duplicates = sorted({n for n in classified if classified.count(n) > 1})
        if duplicates:
            raise ValueError(f"fields classified more than once: {duplicates}")
        missing = sorted(set(self.layout) - set(classified))
        unknown = sorted(set(classified) - set(self.layout))
        if missing or unknown:
            raise ValueError(f"fields not classified: {missing}; not P8 fields: {unknown}")
        leaked = sorted(NEVER_STUDENT & set(self.student_proprio))
        if leaked:
            raise ValueError(f"privileged fields in the student proprio: {leaked}")
        if CLOCK_FIELD in self.student_proprio and self.variant != _CLOCK_VARIANT:
            raise ValueError(
                f"{CLOCK_FIELD!r} is a student input only in the {_CLOCK_VARIANT!r} ablation"
            )
        object.__setattr__(self, "_proprio_idx", _index(self.layout, self.student_proprio))
        object.__setattr__(self, "_priv_idx", _index(self.layout, self.privileged))
        object.__setattr__(self, "_extras_idx", _index(self.layout, self.teacher_only))

    @property
    def obs_dim(self) -> int:
        """Width of P8's full observation (25)."""
        return sum(size for _, size in self.layout.values())

    @property
    def proprio_dim(self) -> int:
        """Width of the student proprio vector (13 default, 14 ``+clock``, 10 ``-last_action``)."""
        return int(self._proprio_idx.size)

    @property
    def priv_dim(self) -> int:
        """Width of the privileged block (11)."""
        return int(self._priv_idx.size)

    @property
    def extras_dim(self) -> int:
        """Width of the teacher-only block (1 default)."""
        return int(self._extras_idx.size)

    def _full(self, obs: ArrayLike) -> NDArray[np.float32]:
        """Return ``obs`` as float32, checking its last axis is P8's full width.

        Args:
            obs: P8 observation(s), shape ``(..., 25)``.

        Returns:
            The float32 array (a view when already float32).

        Raises:
            ValueError: If the last axis is not ``obs_dim`` wide.
        """
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim < 1 or arr.shape[-1] != self.obs_dim:
            raise ValueError(f"expected a P8 observation of width {self.obs_dim}, got {arr.shape}")
        return arr

    def split(
        self, obs: ArrayLike
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Split P8 observation(s) into ``(proprio, priv, extras)`` by field name.

        Args:
            obs: P8's observation(s), shape ``(..., 25)``, units and frames per field (module
                docstring).

        Returns:
            Three new float32 arrays of shapes ``(..., proprio_dim)``, ``(..., priv_dim)``
            and ``(..., extras_dim)``, each concatenating its fields in contract order.

        Raises:
            ValueError: If the last axis is not 25 wide.
        """
        arr = self._full(obs)
        return (
            np.take(arr, self._proprio_idx, axis=-1),
            np.take(arr, self._priv_idx, axis=-1),
            np.take(arr, self._extras_idx, axis=-1),
        )

    def proprio(self, obs: ArrayLike) -> NDArray[np.float32]:
        """Return only the student proprio part of P8 observation(s).

        Args:
            obs: Shape ``(..., 25)``.

        Returns:
            Float32, shape ``(..., proprio_dim)``.
        """
        return np.take(self._full(obs), self._proprio_idx, axis=-1)

    def audit_field(self, obs: ArrayLike, name: str) -> NDArray[np.float32]:
        """Return one named P8 field of observation(s), for checks and audits.

        Not a student path: the field may be privileged. Data code uses it to verify
        alignment (``last_action[t] == action_exec[t-1]``) and to poison stored columns in
        tests, without ever indexing an observation by position. Forbidden in ``ila/train``,
        ``ila/policies``, ``ila/deploy`` and ``ila/eval`` except by a reviewed allowlist entry
        (``tests/test_render_access.py``).

        Args:
            obs: P8 observation(s), shape ``(..., 25)``.
            name: A P8 field name (``layout`` key).

        Returns:
            Float32, shape ``(..., size)`` of the field, a new array.

        Raises:
            KeyError: If ``name`` is not a P8 field.
            ValueError: If the last axis is not 25 wide.
        """
        offset, size = self.layout[name]
        return np.take(self._full(obs), np.arange(offset, offset + size, dtype=np.intp), axis=-1)

    def student_input(self, images: ArrayLike, obs_window: ArrayLike) -> StudentInput:
        """Build a :class:`StudentInput` from images and the matching full observations.

        Args:
            images: ``uint8[To, H, W, 3]``, frame ``i`` rendered from the same physics state
                (``stepSimulation`` count) as ``obs_window[i]``.
            obs_window: P8 observations at the same ``To`` steps, ``float32[To, 25]``.

        Returns:
            The student input; the privileged and teacher-only fields are dropped here.

        Raises:
            ValueError: If ``obs_window`` is not ``(n_obs_steps, 25)`` or the images do not
                validate (:class:`StudentInput`).
        """
        window = self._full(obs_window)
        if window.shape != (self.n_obs_steps, self.obs_dim):
            raise ValueError(
                f"expected an observation window ({self.n_obs_steps}, {self.obs_dim}), "
                f"got {window.shape}"
            )
        return StudentInput(
            images=np.asarray(images),
            proprio=self.proprio(window),
            proprio_fields=self.student_proprio,
            variant=self.variant,
        )


@dataclass(frozen=True)
class StudentInput:
    """Everything an end-to-end student receives at one control step.

    There is deliberately no field that could carry the privileged block, the teacher-only
    clock (outside ``plus_clock``) or a full observation. The proprio must name its columns
    (``proprio_fields``), and both the names and the width are checked against the variant's
    committed contract, so a 25-d vector raises, and so does a hand-built array of the right
    width that does not carry the contract's names. Build it with
    :meth:`ObsContract.student_input`.

    Attributes:
        images: ``uint8[To, H, W, 3]`` RGB frames from the body-fixed camera at steps
            ``t-To+1 ... t``, oldest first (read-only copy). Pixels only
            (``ila.render.camera.Frame.rgb``), never render metadata.
        proprio: ``float32[To, D]`` proprio at the same steps, ``D`` = the variant's
            ``proprio_dim`` (13 default), fields in contract order (read-only copy).
        proprio_fields: The P8 field names of ``proprio``'s column blocks, in order; must
            equal the variant's ``ObsContract.student_proprio``.
        variant: The contract variant the proprio was selected under.
    """

    images: NDArray[np.uint8]
    proprio: NDArray[np.float32]
    proprio_fields: tuple[str, ...]
    variant: str = "default"

    def __post_init__(self) -> None:
        """Validate shapes and dtypes and store read-only copies.

        Raises:
            ValueError: On an unknown variant, field names other than the variant's contract
                (or any privileged name), a wrong dtype, a wrong rank, a proprio width
                other than the variant's, a non-RGB image, or ``To`` mismatched between
                images and proprio.
        """
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown contract variant {self.variant!r}; expected {VARIANTS}")
        names = tuple(self.proprio_fields)
        leaked = sorted(NEVER_STUDENT & set(names))
        if leaked:
            raise ValueError(f"privileged fields named in a student input: {leaked}")
        expected = load_contract(CONTRACT_PATH, self.variant).student_proprio
        if names != expected:
            raise ValueError(
                f"proprio_fields {names} differ from the {self.variant!r} contract {expected}"
            )
        object.__setattr__(self, "proprio_fields", names)
        images = np.asarray(self.images)
        proprio = np.asarray(self.proprio)
        width = _proprio_dim(self.variant)
        if proprio.dtype != np.float32:
            raise ValueError(f"proprio must be float32, got {proprio.dtype}")
        if proprio.ndim != 2 or proprio.shape[1] != width:
            raise ValueError(
                f"proprio must be [To, {width}] for variant {self.variant!r}, got {proprio.shape}"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"images must be uint8, got {images.dtype}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"images must be [To, H, W, 3], got {images.shape}")
        if images.shape[0] != proprio.shape[0] or proprio.shape[0] < 1:
            raise ValueError(f"To differs: images {images.shape[0]}, proprio {proprio.shape[0]}")
        images = images.copy()
        proprio = proprio.copy()
        images.setflags(write=False)
        proprio.setflags(write=False)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "proprio", proprio)

    @property
    def n_obs_steps(self) -> int:
        """``To``, the number of stacked steps."""
        return int(self.proprio.shape[0])


def _apply_ablation(
    variant: str,
    spec: Mapping[str, Any],
    proprio: list[str],
    privileged: list[str],
    teacher_only: list[str],
) -> None:
    """Apply one YAML ablation in place: ``remove`` -> teacher-only, ``add`` <- teacher-only.

    Args:
        variant: The ablation's name, for error messages.
        spec: ``{"remove": [...]}`` and/or ``{"add": [...]}``.
        proprio: Student proprio names (mutated).
        privileged: Privileged names (read only; an ``add`` from here raises).
        teacher_only: Teacher-only names (mutated).

    Raises:
        ValueError: If a removed field is not a student field, or an added field is not
            teacher-only (in particular, if it is privileged).
    """
    for name in spec.get("remove", []):
        if name not in proprio:
            raise ValueError(f"{variant}: cannot remove {name!r}, not a student proprio field")
        proprio.remove(name)
        teacher_only.append(name)
    for name in spec.get("add", []):
        if name in privileged or name not in teacher_only:
            raise ValueError(f"{variant}: cannot add {name!r}, not a teacher-only field")
        teacher_only.remove(name)
        proprio.append(name)


@functools.cache
def load_contract(path: Path = CONTRACT_PATH, variant: str = "default") -> ObsContract:
    """Load the observation contract (cached per ``(path, variant)``).

    Args:
        path: The contract YAML (``configs/obs/student.yaml``).
        variant: ``"default"`` or the name of an ablation under ``ablations:``.

    Returns:
        The frozen :class:`ObsContract`, with offsets and sizes from P8's ``obs_fields``.

    Raises:
        ValueError: If ``variant`` is unknown or the contract does not validate.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown contract variant {variant!r}; expected {VARIANTS}")
    raw: Mapping[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    proprio = [str(n) for n in raw["student_proprio"]]
    privileged = [str(n) for n in raw["privileged"]]
    teacher_only = [str(n) for n in raw["teacher_only"]]
    if variant != "default":
        ablations: Mapping[str, Mapping[str, Any]] = raw.get("ablations", {})
        if variant not in ablations:
            raise ValueError(f"variant {variant!r} is not defined in {path}")
        _apply_ablation(variant, ablations[variant], proprio, privileged, teacher_only)
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for f in p8_adapter.obs_fields():
        layout[f.name] = (offset, int(f.size))
        offset += int(f.size)
    return ObsContract(
        layout=MappingProxyType(layout),
        student_proprio=tuple(proprio),
        privileged=tuple(privileged),
        teacher_only=tuple(teacher_only),
        n_obs_steps=int(raw["frames"]["n_obs_steps"]),
        variant=variant,
    )


@functools.cache
def _proprio_dim(variant: str) -> int:
    """Return the committed contract's proprio width for one variant."""
    return load_contract(CONTRACT_PATH, variant).proprio_dim


def split(
    obs: ArrayLike, contract: ObsContract | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Split P8 observation(s) into ``(proprio, priv, extras)`` under a contract.

    Args:
        obs: Shape ``(..., 25)``.
        contract: The contract; the committed default when ``None``.

    Returns:
        See :meth:`ObsContract.split`.
    """
    return (load_contract() if contract is None else contract).split(obs)
