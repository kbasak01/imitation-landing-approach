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

The ``modular`` arm (P3-D1 §3, declared clock exception 2) is the one other path for
``time_fraction``: :meth:`ObsContract.modular_input` wraps the default-variant
:class:`StudentInput` and the true ``time_fraction`` of the same ``To`` steps in a
:class:`ModularInput`. The clock goes to the frozen teacher's observation assembly only
(:func:`assemble_teacher_obs`, which writes each block at its fields' P8 offsets), never to the
estimator.

The closed-loop clock diagnostic (``ila.train.clock_diagnostics``) changes the *value* of
``time_fraction`` where a network already receives it, and nothing else:
:meth:`ObsContract.with_proprio_clock` (the ``plus_clock`` proprio column; refused under any
other variant) and :meth:`ObsContract.with_obs_clock` (a teacher's 25-d observation) copy the
input and replace that one field, by name. Neither adds the clock to any input.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
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
    "ModularInput",
    "ObsContract",
    "StudentInput",
    "assemble_teacher_obs",
    "load_contract",
    "split_privileged",
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

#: The only contract variant a ``ModularInput`` is built under (the arm has no ablation).
_MODULAR_VARIANT: Final[str] = "default"


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

    def split_privileged(
        self, obs: ArrayLike
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Split P8 observation(s) into ``(proprio, priv, extras)`` by field name.

        Not a student path: ``priv`` and ``extras`` are privileged and teacher-only. Data code
        (the demo store) uses it to write its columns. Like :meth:`audit_field`, it is an audit
        name (``tests/test_render_access.py``): forbidden in ``ila/train``, ``ila/policies``,
        ``ila/deploy`` and ``ila/eval`` except by a reviewed allowlist entry. The distinctive
        name lets the static scan find it (a bare ``split`` would also match ``str.split``).
        A student path takes :meth:`proprio` or :meth:`student_input`.

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

    # ------------------------------------------------------------------ clock intervention
    # The closed-loop clock diagnostic (``ila.train.clock_diagnostics``) changes the value of
    # the ``time_fraction`` field and nothing else. These three methods are the only way it
    # reaches that column: by field name, here, never by position elsewhere. They never add
    # a field to anything: a student proprio stays the contract's (``plus_clock`` only) and a
    # teacher observation stays P8's 25-d vector.

    def _proprio_clock_idx(self) -> NDArray[np.intp]:
        """The ``time_fraction`` column(s) of this contract's student proprio, by name.

        Raises:
            ValueError: Outside the ``plus_clock`` variant (no clock column exists).
        """
        if self.variant != _CLOCK_VARIANT or CLOCK_FIELD not in self.student_proprio:
            raise ValueError(
                f"the student proprio has a {CLOCK_FIELD!r} column only under the "
                f"{_CLOCK_VARIANT!r} contract, not {self.variant!r}"
            )
        start = 0
        for name in self.student_proprio:
            size = self.layout[name][1]
            if name == CLOCK_FIELD:
                return np.arange(start, start + size, dtype=np.intp)
            start += size
        raise AssertionError("unreachable: the clock is in the proprio")  # pragma: no cover

    def proprio_clock(self, proprio: ArrayLike) -> NDArray[np.float32]:
        """Return the ``time_fraction`` column of a ``plus_clock`` student proprio.

        Args:
            proprio: ``float32[..., proprio_dim]`` (14 under ``plus_clock``), contract order.
                A proprio-shaped vector such as a normaliser mean is accepted too.

        Returns:
            Float32, shape ``(..., 1)``, dimensionless (``t_episode / episode_len``,
            ``episode_len`` = 12 s model scale), a new array.

        Raises:
            ValueError: Outside ``plus_clock`` or on a wrong width.
        """
        idx = self._proprio_clock_idx()
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.ndim < 1 or arr.shape[-1] != self.proprio_dim:
            raise ValueError(f"expected a proprio of width {self.proprio_dim}, got {arr.shape}")
        return np.take(arr, idx, axis=-1)

    def with_proprio_clock(
        self,
        proprio: ArrayLike,
        clock_fn: Callable[[NDArray[np.float32]], ArrayLike],
    ) -> NDArray[np.float32]:
        """Return a copy of a ``plus_clock`` proprio with only its clock column replaced.

        Every other column is copied bit for bit; the clock column becomes
        ``clock_fn(old_clock)``, cast to float32.

        Args:
            proprio: ``float32[..., 14]``, the ``plus_clock`` student proprio, contract order.
            clock_fn: ``float32[..., 1] -> [..., 1]``, the new ``time_fraction`` values,
                dimensionless, finite, in ``[0, 1]`` (P8's range).

        Returns:
            A new writable float32 array of the input's shape.

        Raises:
            ValueError: Outside ``plus_clock``, on a wrong width, or if ``clock_fn`` returns a
                wrong shape or a value that is not finite in ``[0, 1]``.
        """
        idx = self._proprio_clock_idx()
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.ndim < 1 or arr.shape[-1] != self.proprio_dim:
            raise ValueError(f"expected a proprio of width {self.proprio_dim}, got {arr.shape}")
        return _replace_columns(arr, idx, clock_fn)

    def with_obs_clock(
        self,
        obs: ArrayLike,
        clock_fn: Callable[[NDArray[np.float32]], ArrayLike],
    ) -> NDArray[np.float32]:
        """Return a copy of P8 observation(s) with only ``time_fraction`` replaced, by name.

        For the teacher side of the clock diagnostic: the observation stays P8's 25-d vector
        (it is never a student input); every other field is copied bit for bit.

        Args:
            obs: ``float32[..., 25]``, P8 observation(s).
            clock_fn: ``float32[..., 1] -> [..., 1]``, the new ``time_fraction`` values,
                dimensionless, finite, in ``[0, 1]``.

        Returns:
            A new writable float32 array of the input's shape.

        Raises:
            ValueError: On a wrong width, or if ``clock_fn`` returns a wrong shape or a value
                that is not finite in ``[0, 1]``.
        """
        offset, size = self.layout[CLOCK_FIELD]
        idx = np.arange(offset, offset + size, dtype=np.intp)
        return _replace_columns(self._full(obs), idx, clock_fn)

    def modular_input(self, images: ArrayLike, obs_window: ArrayLike) -> ModularInput:
        """Build the ``modular`` arm's :class:`ModularInput` from one observation window.

        The student part is exactly :meth:`student_input` on the same arguments; the clock
        is P8's ``time_fraction`` field, selected by name from the same window, so both share
        one step alignment and one start-of-episode padding. The window itself (steps
        ``t-To+1 ... t``, indices < 0 repeat step 0) is assembled by the caller from
        :func:`ila.policies.dp.chunking.obs_window_steps`, as for :meth:`student_input`
        (``ila.eval.runner.StudentWindow``).

        Args:
            images: ``uint8[To, H, W, 3]``, frame ``i`` rendered from the same physics state
                (``stepSimulation`` count) as ``obs_window[i]``.
            obs_window: P8 observations at the same ``To`` steps, ``float32[To, 25]``,
                oldest first.

        Returns:
            ``ModularInput(student_input, time_fraction)``; the privileged fields are
            dropped here and never reach the result.

        Raises:
            ValueError: If this contract is not the ``default`` variant, the window is not
                ``(n_obs_steps, 25)``, the images do not validate, or ``time_fraction`` is
                not finite in ``[0, 1]``.
        """
        if self.variant != _MODULAR_VARIANT:
            raise ValueError(
                f"a ModularInput is built under the {_MODULAR_VARIANT!r} contract only, "
                f"not {self.variant!r}"
            )
        student = self.student_input(images, obs_window)
        offset, size = self.layout[CLOCK_FIELD]
        clock = np.take(
            self._full(obs_window), np.arange(offset, offset + size, dtype=np.intp), axis=-1
        )
        return ModularInput(student_input=student, time_fraction=clock.reshape(-1))

    def assemble_teacher_obs(
        self, proprio_t: ArrayLike, priv_estimate_t: ArrayLike, time_fraction_t: ArrayLike
    ) -> NDArray[np.float32]:
        """Assemble one 25-d P8 teacher observation from its three blocks, by field name.

        See :func:`assemble_teacher_obs` (the module-level entry point).
        """
        if self.variant != _MODULAR_VARIANT:
            raise ValueError(
                f"a teacher observation is assembled under the {_MODULAR_VARIANT!r} contract "
                f"only, not {self.variant!r}"
            )
        if self.teacher_only != (CLOCK_FIELD,):
            raise ValueError(f"teacher-only block {self.teacher_only} is not ({CLOCK_FIELD!r},)")
        blocks = (
            ("proprio_t", proprio_t, self._proprio_idx),
            ("priv_estimate_t", priv_estimate_t, self._priv_idx),
            ("time_fraction_t", time_fraction_t, self._extras_idx),
        )
        out = np.full(self.obs_dim, np.nan, dtype=np.float32)
        written = np.zeros(self.obs_dim, dtype=bool)
        for name, value, idx in blocks:
            arr = np.asarray(value)
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(f"{name} must be a float array, got {arr.dtype}")
            if arr.ndim == 0:  # a scalar clock
                arr = arr.reshape(1)
            if arr.shape != (idx.size,):
                raise ValueError(f"{name} must have shape ({idx.size},), got {np.shape(value)}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} has non-finite entries: {arr}")
            out[idx] = arr.astype(np.float32, copy=False)
            written[idx] = True
        if not written.all():  # every P8 field is classified once, so this cannot happen
            raise RuntimeError("assembled teacher observation has unwritten fields")
        clock = out[self._extras_idx]
        if np.any(clock < 0.0) or np.any(clock > 1.0):
            raise ValueError(f"time_fraction_t must be in [0, 1], got {clock}")
        return out


def _replace_columns(
    arr: NDArray[np.float32],
    idx: NDArray[np.intp],
    fn: Callable[[NDArray[np.float32]], ArrayLike],
) -> NDArray[np.float32]:
    """Copy ``arr`` and set its last-axis columns ``idx`` to ``fn(arr[..., idx])``.

    Args:
        arr: Float32 array, columns on the last axis.
        idx: Column indices (computed by name by the caller).
        fn: Old columns ``float32[..., len(idx)]`` -> new ones, finite in ``[0, 1]``.

    Returns:
        A new writable float32 array; columns outside ``idx`` are bit-identical to ``arr``'s.

    Raises:
        ValueError: If ``fn`` returns a wrong shape or a value not finite in ``[0, 1]``.
    """
    out = np.array(arr, dtype=np.float32, copy=True)
    old = np.take(arr, idx, axis=-1)
    new = np.asarray(fn(old.copy()), dtype=np.float32)
    if new.shape != old.shape:
        raise ValueError(f"replacement has shape {new.shape}, expected {old.shape}")
    if not (np.all(np.isfinite(new)) and np.all(new >= 0.0) and np.all(new <= 1.0)):
        raise ValueError(f"{CLOCK_FIELD} must be finite in [0, 1], got {new}")
    out[..., idx] = new
    return out


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


@dataclass(frozen=True)
class ModularInput:
    """What the ``modular`` arm receives at one control step (P3-D1 §3, clock exception 2).

    ``modular`` = vision estimator -> estimated privileged block -> frozen teacher. The
    estimator sees ``student_input`` only. The teacher MLP additionally needs P8's
    ``time_fraction`` observation field, which is neither a student input (outside the
    labelled ``+clock`` ablation) nor an estimator target; this dataclass is the only path
    that carries it (leak RISK 2). ``time_fraction`` is for the teacher-observation assembly
    alone: it is never passed to the estimator's ``forward``, a feature, a normaliser or a
    training batch. There is no field for the privileged block: the teacher gets the
    estimator's *estimate* of it, never the true values.

    Build it with :meth:`ObsContract.modular_input`, which selects the clock by field name
    from the same observation window as the student input.

    Attributes:
        student_input: The ``default``-variant :class:`StudentInput` (images and 13-d
            proprio at steps ``t-To+1 ... t``), bit-identical to
            :meth:`ObsContract.student_input` on the same window.
        time_fraction: ``float32[To]``, P8's ``time_fraction`` at the same steps, oldest
            first (last entry = step ``t``; indices < 0 repeat step 0, as in the student
            input). Dimensionless: ``t_episode / episode_len`` clipped into ``[0, 1]`` by P8,
            ``episode_len`` = 12 s model scale (60 s full scale, lambda = 1/25). Read-only copy.
    """

    student_input: StudentInput
    time_fraction: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate the student part and the clock, and store a read-only clock copy.

        Raises:
            TypeError: If ``student_input`` is not a :class:`StudentInput`.
            ValueError: If the student input is not the ``default`` variant, or
                ``time_fraction`` is not float32 of shape ``(To,)`` with finite values in
                ``[0, 1]``.
        """
        if not isinstance(self.student_input, StudentInput):
            raise TypeError(f"student_input must be a StudentInput, got {type(self.student_input)}")
        if self.student_input.variant != _MODULAR_VARIANT:
            raise ValueError(
                f"modular uses the {_MODULAR_VARIANT!r} student input, "
                f"got {self.student_input.variant!r}"
            )
        clock = np.asarray(self.time_fraction)
        if clock.dtype != np.float32:
            raise ValueError(f"time_fraction must be float32, got {clock.dtype}")
        n_obs = self.student_input.n_obs_steps
        if clock.shape != (n_obs,):
            raise ValueError(f"time_fraction must be [{n_obs}] (To), got {clock.shape}")
        if not (np.all(np.isfinite(clock)) and np.all(clock >= 0.0) and np.all(clock <= 1.0)):
            raise ValueError(f"time_fraction must be finite in [0, 1], got {clock}")
        clock = clock.copy()
        clock.setflags(write=False)
        object.__setattr__(self, "time_fraction", clock)

    @property
    def n_obs_steps(self) -> int:
        """``To``, the number of stacked steps."""
        return self.student_input.n_obs_steps


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


def split_privileged(
    obs: ArrayLike, contract: ObsContract | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Split P8 observation(s) into ``(proprio, priv, extras)`` under a contract.

    Args:
        obs: Shape ``(..., 25)``.
        contract: The contract; the committed default when ``None``.

    Returns:
        See :meth:`ObsContract.split_privileged`.
    """
    return (load_contract() if contract is None else contract).split_privileged(obs)


def assemble_teacher_obs(
    contract: ObsContract,
    proprio_t: ArrayLike,
    priv_estimate_t: ArrayLike,
    time_fraction_t: ArrayLike,
) -> NDArray[np.float32]:
    """Assemble the ``modular`` arm's 25-d teacher observation at one step, by field name.

    The inverse of :meth:`ObsContract.split_privileged` for the ``default`` contract: each block is
    written at the positions P8's layout gives its fields (``contract.layout``, from P8's
    ``obs_fields``), so the result is in P8's field order whatever the block order. Fed the
    true blocks of an env observation it reproduces that observation bit for bit; in the
    ``modular`` arm ``priv_estimate_t`` is the estimator's output, and ``time_fraction_t``
    comes only from :attr:`ModularInput.time_fraction` (its last entry, step ``t``).

    Args:
        contract: The ``default`` observation contract (P8 layout and field classes).
        proprio_t: ``float[13]``, the student proprio at step t in contract order
            (``attitude`` rad world, ``body_rates`` rad/s world axes, ``velocity`` m/s model
            scale drone-yaw frame, ``last_action`` normalised world, ``in_contact`` {0, 1}),
            e.g. ``ModularInput.student_input.proprio[-1]``.
        priv_estimate_t: ``float[11]``, the privileged block in contract order
            (``rel_position`` m and ``rel_velocity`` m/s model scale drone-yaw frame,
            ``deck_normal`` unit drone-yaw, ``rel_tilt`` rad, ``height`` m model scale along the
            deck normal); the estimator's estimate in the ``modular`` arm.
        time_fraction_t: Scalar or ``float[1]``, dimensionless in ``[0, 1]``
            (``t_episode / episode_len``, ``episode_len`` = 12 s model scale).

    Returns:
        ``float32[25]``, a new array in P8's observation layout. Float inputs are cast to
        float32 (P8's observation dtype).

    Raises:
        ValueError: If the contract is not ``default``, a block has the wrong shape, a
            non-float dtype or a non-finite entry, or the clock is outside ``[0, 1]``.
    """
    return contract.assemble_teacher_obs(proprio_t, priv_estimate_t, time_fraction_t)
