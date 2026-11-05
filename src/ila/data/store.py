"""HDF5 demo store via h5py, lzf compression, one group per episode (plan §1.5, Phase 3).

Schema (exactly plan §1.5; :data:`STORE_KEYS`), per episode group ``ep_<demo index:05d>``:

* ``images``: uint8 ``(T, 96, 96, 3)``, the frame paired with step t's observation (``Frame.rgb``);
* ``proprio``: float32 ``(T, 13)``, the student allowlist in ``student.yaml`` order, by name;
* ``priv``: float32 ``(T, 11)``, the five privileged fields (estimator target only);
* ``teacher_obs``: float32 ``(T, 25)``, P8's full observation;
* ``action_exec``: float32 ``(T, 3)``, the action executed at step t, clipped to [-1, 1];
* ``label``: float32 ``(T, 3)``, ``teacher(teacher_obs[t])``, clean;
* ``t_model_s``: float64 ``(T,)``, ``t0 + t / 30``, seconds MODEL scale (lambda = 1/25).

Alignment (il-conventions): ``images[t]`` and ``teacher_obs[t]`` come from the state after the
t-th ``env.step`` (t = 0: after reset), same ``stepSimulation`` count; ``action_exec[t]`` is
chosen after seeing them; ``last_action`` inside ``teacher_obs[t]`` equals ``action_exec[t-1]``
(zero at t = 0). The terminal observation (after the last step) has no action and is not
stored, so every key has the same ``T`` = steps flown. Actions are normalised world-frame
velocity setpoints (x v_max = 1.5 m/s model scale, norm-capped by the env after this clip).

Attributes: :class:`EpisodeMeta` (realization key, episode seed, teacher, beta, DART sigma,
outcome class, P8 commit, camera-config SHA-256, ``ila`` commit, the per-frame clear-colour
fraction and its statistics, ...) on every group; :data:`FILE_ATTR_KEYS` on the file.

Read paths. :meth:`DemoStore.student_inputs` is THE student read path: it returns a
:class:`StudentEpisode` holding ``images`` and ``proprio`` and nothing else, and it reads no
other dataset. Training targets come from :meth:`DemoStore.labels`; the estimator target from
:meth:`DemoStore.estimator_targets`. ``teacher_obs`` and ``t_model_s`` are reachable only
through the explicitly privileged audit accessors, never through the student path, and the
store holds no render metadata at all (P1-D1, P1-D2 items 3, 4 and 6).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray

from ila.obs.split import CLOCK_FIELD, NEVER_STUDENT, VARIANTS, ObsContract, load_contract

__all__ = [
    "ACTION_DIM",
    "CLEAR_RGB",
    "CTRL_HZ",
    "FILE_ATTR_KEYS",
    "FORMAT",
    "IMAGE_SHAPE",
    "STORE_KEYS",
    "STUDENT_INPUT_KEYS",
    "DemoStore",
    "DemoWriter",
    "EpisodeMeta",
    "EpisodeRecord",
    "StoreSchemaError",
    "StudentEpisode",
    "VARIANT_DELTA",
    "clear_fraction",
    "contract_field_attrs",
    "episode_group",
    "schema",
    "total_frames",
    "validate",
    "variant_contract",
]

#: Store format tag, a file attribute.
FORMAT: Final[str] = "ila-demos/1"

#: Every dataset of an episode group, in plan §1.5 order. No other dataset may exist.
STORE_KEYS: Final[tuple[str, ...]] = (
    "images",
    "proprio",
    "priv",
    "teacher_obs",
    "action_exec",
    "label",
    "t_model_s",
)

#: The only datasets a student input is ever built from.
STUDENT_INPUT_KEYS: Final[tuple[str, ...]] = ("images", "proprio")

#: Frame shape ``(H, W, 3)``, px (``configs/render/camera.yaml``, 96 px; P2-D2).
IMAGE_SHAPE: Final[tuple[int, int, int]] = (96, 96, 3)

#: Control rate, Hz (P8: 30 Hz control, 240 Hz physics); ``t_model_s[t] = t0 + t / CTRL_HZ``.
CTRL_HZ: Final[float] = 30.0

#: Action width (P8's ``ACTION_DIM``), normalised units.
ACTION_DIM: Final[int] = 3

#: TinyRenderer's clear colour (white), measured: where nothing is drawn every channel is 255
#: (segmentation -1), and the brightest scene texel at the committed light is 242.
CLEAR_RGB: Final[int] = 255

#: File-level attributes every store carries.
FILE_ATTR_KEYS: Final[tuple[str, ...]] = (
    "format",
    "collection",
    "teacher",
    "dart_sigma",
    "beta",
    "split_file",
    "split_file_sha256",
    "generator_seed",
    "p8_commit",
    "ila_commit",
    "camera_cfg_sha256",
    "proprio_fields",
    "priv_fields",
    "obs_fields",
)


class StoreSchemaError(ValueError):
    """An episode or a store does not follow the plan §1.5 schema or its alignment rule."""


def schema(contract: ObsContract | None = None) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    """Return ``key -> (dtype, per-step shape)`` for :data:`STORE_KEYS`.

    Args:
        contract: The observation contract; the committed default (13/11/25) when ``None``.
            Only the default contract is stored (the ablations select from it later).

    Returns:
        The schema, in :data:`STORE_KEYS` order.
    """
    c = load_contract() if contract is None else contract
    return {
        "images": (np.dtype(np.uint8), IMAGE_SHAPE),
        "proprio": (np.dtype(np.float32), (c.proprio_dim,)),
        "priv": (np.dtype(np.float32), (c.priv_dim,)),
        "teacher_obs": (np.dtype(np.float32), (c.obs_dim,)),
        "action_exec": (np.dtype(np.float32), (ACTION_DIM,)),
        "label": (np.dtype(np.float32), (ACTION_DIM,)),
        "t_model_s": (np.dtype(np.float64), ()),
    }


#: Each ablation variant differs from the default student proprio by exactly one named field:
#: ``(how, field)``. ``plus_clock`` appends the clock; ``minus_last_action`` drops last_action.
VARIANT_DELTA: Final[Mapping[str, tuple[str, str]]] = {
    "minus_last_action": ("remove", "last_action"),
    "plus_clock": ("add", CLOCK_FIELD),
}


def variant_contract(variant: str) -> ObsContract:
    """Return a variant's contract after checking it is the default +- exactly one field.

    Args:
        variant: An ``ila.obs.split.VARIANTS`` name.

    Returns:
        The variant's :class:`ObsContract`.

    Raises:
        ValueError: On an unknown variant.
        StoreSchemaError: If the variant's student fields are not the default fields with
            exactly :data:`VARIANT_DELTA`'s field removed/appended, or name a privileged field.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown contract variant {variant!r}; expected {VARIANTS}")
    contract = load_contract(variant=variant)
    default = load_contract().student_proprio
    if variant == "default":
        expected = default
    else:
        how, name = VARIANT_DELTA[variant]
        expected = (*default, name) if how == "add" else tuple(f for f in default if f != name)
        if expected == default:
            raise StoreSchemaError(f"{variant}: {name!r} not removable from {default}")
    if contract.student_proprio != expected or NEVER_STUDENT & set(contract.student_proprio):
        raise StoreSchemaError(
            f"{variant}: student fields {contract.student_proprio} != expected {expected}"
        )
    return contract


def clear_fraction(rgb: NDArray[np.uint8]) -> float:
    """Return the fraction of pixels showing TinyRenderer's clear colour (P1-D1).

    Args:
        rgb: ``uint8[H, W, 3]``.

    Returns:
        Fraction of pixels with every channel equal to 255, in [0, 1].
    """
    return float(np.mean(np.all(np.asarray(rgb) == CLEAR_RGB, axis=-1)))


def episode_group(demo_index: int) -> str:
    """Return an episode's group name from its global index in its stream."""
    return f"ep_{int(demo_index):05d}"


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class EpisodeMeta:
    """Per-episode attributes (HDF5 attrs of the group). Never a student input.

    Attributes:
        collection: Collection name, e.g. ``"ppo_2000"``.
        stream: The ``train`` stream the episode was drawn from (``"demos"``).
        demo_index: Global index of the episode in that stream.
        realization_key: dmf's key ``(ss, heading_deg, speed_kn, vessel, seed)`` as JSON.
        regime: The row's list label (= ``stream``).
        ss: Sea state.
        index: Index in its (stream, sea state) cell.
        pad: Pad name.
        vessel: dmf vessel stem.
        heading_deg: Encounter heading, degrees.
        speed_kn: Ship speed, knots full scale.
        realization_seed: dmf seed ordinal.
        episode_seed: The env's episode seed.
        t0_model_s: Start offset into the realization, s model scale.
        teacher: Collection teacher ID (``T_ppo``, ``T_pid``, ``T_mix``).
        episode_teacher: The P8 policy flown and labelling this episode (``ppo``,
            ``pid_feedforward``, ``gated``); for ``T_mix`` the balanced assignment's choice.
        beta: Probability per step that the teacher's (possibly noised) action is executed;
            1.0 for teacher demos (DAgger sets it per iteration).
        dart_sigma: DART noise std on the executed action, normalised units; 0.0 for none.
        noise_seed: The DART seed, or -1 without noise; the episode's generator entropy is
            ``[noise_seed, H("dart"), demo_index, episode_seed]`` (``ila.data.dart``).
        dagger_iteration: 0 for demos; the DAgger iteration otherwise.
        student_ckpt_sha256: ``""`` for demos; the rolled-out student's checkpoint SHA.
        outcome: P8 outcome class (crash, off_pad, hard_landing, bounce, success, timeout).
        termination_reason: P8's free-text reason.
        success: ``outcome == "success"``.
        steps: ``T``, control steps flown (30 Hz).
        p8_commit: P8 submodule commit.
        ila_commit: This repository's commit (``+dirty`` if the tree was dirty).
        camera_cfg_sha256: ``render_cfg_sha256("full", None)``, the committed look.
        clear_frac: Per-frame clear-colour fraction, float32 ``(T,)``.
        clear_frac_max: Its maximum.
        clear_frac_mean: Its mean.
        clear_frames: Frames with any clear-colour pixel.
    """

    collection: str
    stream: str
    demo_index: int
    realization_key: str
    regime: str
    ss: str
    index: int
    pad: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int
    t0_model_s: float
    teacher: str
    episode_teacher: str
    beta: float
    dart_sigma: float
    noise_seed: int
    dagger_iteration: int
    student_ckpt_sha256: str
    outcome: str
    termination_reason: str
    success: bool
    steps: int
    p8_commit: str
    ila_commit: str
    camera_cfg_sha256: str
    clear_frac: NDArray[np.float32]
    clear_frac_max: float
    clear_frac_mean: float
    clear_frames: int

    def as_attrs(self) -> dict[str, Any]:
        """Return HDF5-attribute values (plain Python scalars, one float32 array)."""
        out = asdict(self)
        out["clear_frac"] = np.asarray(self.clear_frac, dtype=np.float32)
        out["success"] = bool(self.success)
        return out

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> EpisodeMeta:
        """Rebuild from HDF5 attributes (types restored per field).

        Raises:
            StoreSchemaError: If an attribute is missing or extra.
        """
        names = [f.name for f in fields(cls)]
        if set(attrs) != set(names):
            raise StoreSchemaError(
                f"episode attrs {sorted(attrs)} != {sorted(names)} (missing/extra attributes)"
            )
        kw: dict[str, Any] = {}
        for f in fields(cls):
            value = attrs[f.name]
            if f.name == "clear_frac":
                kw[f.name] = np.asarray(value, dtype=np.float32)
            elif f.type in ("int", int):
                kw[f.name] = int(value)
            elif f.type in ("float", float):
                kw[f.name] = float(value)
            elif f.type in ("bool", bool):
                kw[f.name] = bool(value)
            else:
                kw[f.name] = str(value)
        return cls(**kw)


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode in memory, ready to be written (arrays per :data:`STORE_KEYS`).

    Build with :meth:`build`, which derives ``proprio`` and ``priv`` from ``teacher_obs`` by
    field name through ``ila.obs.split`` and validates the alignment rule.

    Attributes:
        arrays: Key -> array, exactly :data:`STORE_KEYS`.
        meta: The attributes.
    """

    arrays: Mapping[str, NDArray[Any]]
    meta: EpisodeMeta

    @classmethod
    def build(
        cls,
        *,
        images: NDArray[np.uint8],
        teacher_obs: NDArray[np.float32],
        action_exec: NDArray[np.float32],
        label: NDArray[np.float32],
        t_model_s: NDArray[np.float64],
        meta: EpisodeMeta,
        contract: ObsContract | None = None,
    ) -> EpisodeRecord:
        """Assemble and validate one episode.

        Args:
            images: ``uint8[T, 96, 96, 3]``.
            teacher_obs: ``float32[T, 25]``.
            action_exec: ``float32[T, 3]``, clipped to [-1, 1].
            label: ``float32[T, 3]``.
            t_model_s: ``float64[T]``, s model scale.
            meta: The attributes.
            contract: The default observation contract when ``None``.

        Returns:
            The validated record.

        Raises:
            StoreSchemaError: On any schema or alignment violation (:func:`validate`).
        """
        c = load_contract() if contract is None else contract
        obs = np.ascontiguousarray(teacher_obs, dtype=np.float32)
        proprio, priv, _ = c.split_privileged(obs)
        record = cls(
            arrays={
                "images": np.ascontiguousarray(images, dtype=np.uint8),
                "proprio": proprio,
                "priv": priv,
                "teacher_obs": obs,
                "action_exec": np.ascontiguousarray(action_exec, dtype=np.float32),
                "label": np.ascontiguousarray(label, dtype=np.float32),
                "t_model_s": np.ascontiguousarray(t_model_s, dtype=np.float64),
            },
            meta=meta,
        )
        validate(record.arrays, meta, c)
        return record


def validate(
    arrays: Mapping[str, NDArray[Any]], meta: EpisodeMeta, contract: ObsContract | None = None
) -> None:
    """Check one episode against the schema and the alignment rule.

    Checks: exactly :data:`STORE_KEYS`; dtypes and per-step shapes; one ``T`` = ``meta.steps``
    >= 1; finite floats; ``proprio``/``priv`` equal the by-name selection of ``teacher_obs``;
    ``last_action`` in ``teacher_obs[0]`` is zero and in ``teacher_obs[t]`` equals
    ``action_exec[t-1]``; ``action_exec`` in [-1, 1]; for a noise-free, beta = 1 episode
    ``action_exec == clip(label)``; ``t_model_s`` steps by 1/30 s model scale from ``t0``.

    Raises:
        StoreSchemaError: On the first violation.
    """
    c = load_contract() if contract is None else contract
    if tuple(arrays) != STORE_KEYS:
        raise StoreSchemaError(f"datasets {list(arrays)} != {list(STORE_KEYS)}")
    t_len = int(meta.steps)
    if t_len < 1:
        raise StoreSchemaError(f"episode with {t_len} steps")
    for key, (dtype, shape) in schema(c).items():
        arr = arrays[key]
        if arr.dtype != dtype or arr.shape != (t_len, *shape):
            raise StoreSchemaError(f"{key}: {arr.dtype} {arr.shape} != {dtype} {(t_len, *shape)}")
        if key != "images" and not np.all(np.isfinite(arr)):
            raise StoreSchemaError(f"{key}: non-finite values")
    obs = arrays["teacher_obs"]
    proprio, priv, _ = c.split_privileged(obs)
    if not (np.array_equal(arrays["proprio"], proprio) and np.array_equal(arrays["priv"], priv)):
        raise StoreSchemaError("proprio/priv are not the by-name selection of teacher_obs")
    last = c.audit_field(obs, "last_action")
    executed = arrays["action_exec"]
    if np.any(last[0] != 0.0) or not np.array_equal(last[1:], executed[:-1]):
        raise StoreSchemaError("last_action[t] != action_exec[t-1] (or last_action[0] != 0)")
    if np.any(np.abs(executed) > 1.0):
        raise StoreSchemaError("action_exec outside [-1, 1]")
    noise_free = meta.dart_sigma == 0.0 and meta.beta == 1.0 and meta.dagger_iteration == 0
    if noise_free and not np.array_equal(executed, np.clip(arrays["label"], -1.0, 1.0)):
        raise StoreSchemaError("noise-free teacher episode with action_exec != clip(label)")
    times = arrays["t_model_s"]
    expected = meta.t0_model_s + np.arange(t_len, dtype=np.float64) / CTRL_HZ
    if np.max(np.abs(times - expected)) > 1e-9:
        raise StoreSchemaError("t_model_s is not t0 + t / 30 s")
    if meta.clear_frac.shape != (t_len,):
        raise StoreSchemaError(f"clear_frac shape {meta.clear_frac.shape} != ({t_len},)")


# --------------------------------------------------------------------------- writer


class DemoWriter:
    """Write episodes into a new HDF5 store, in the order given (deterministic bytes).

    Every dataset uses lzf compression; ``images`` is chunked one frame per chunk (27 kB) for
    random frame access, the small arrays one episode per chunk. Timestamps are off, so the
    same episodes in the same order give the same file bytes.

    Attributes:
        path: The store file.
        n_episodes: Episodes written so far.
    """

    def __init__(
        self,
        path: Path,
        file_attrs: Mapping[str, Any],
        *,
        expected_camera_sha256: str,
        overwrite: bool = False,
    ) -> None:
        """Create the file and write its attributes.

        Args:
            path: Output file (``.h5``); its directory is created.
            file_attrs: Exactly :data:`FILE_ATTR_KEYS` (``format`` is filled in).
            expected_camera_sha256: The committed look's ``render_cfg_sha256``. Episodes and
                the file must carry it: an appearance-shift render (evaluation only, plan
                §1.3) can never be written (P1-D2 obligation 2).
            overwrite: Replace an existing file.

        Raises:
            FileExistsError: If the file exists and ``overwrite`` is False.
            StoreSchemaError: If the file attributes are not :data:`FILE_ATTR_KEYS` or the
                camera SHA is not the committed one.
        """
        attrs = {"format": FORMAT, **file_attrs}
        if set(attrs) != set(FILE_ATTR_KEYS):
            raise StoreSchemaError(f"file attrs {sorted(attrs)} != {sorted(FILE_ATTR_KEYS)}")
        if attrs["camera_cfg_sha256"] != expected_camera_sha256:
            raise StoreSchemaError("store camera config is not the committed look")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; refusing to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._camera = expected_camera_sha256
        self._h5 = h5py.File(path, "w", track_order=True)
        for key in FILE_ATTR_KEYS:
            self._h5.attrs[key] = attrs[key]
        self.n_episodes = 0

    def write(self, record: EpisodeRecord) -> None:
        """Write one validated episode as group ``ep_<demo index>``.

        Raises:
            StoreSchemaError: If the episode's camera SHA is not the committed one, or it
                does not validate.
            ValueError: If the group already exists.
        """
        if record.meta.camera_cfg_sha256 != self._camera:
            raise StoreSchemaError("episode rendered with a non-committed camera config")
        validate(record.arrays, record.meta)
        name = episode_group(record.meta.demo_index)
        if name in self._h5:
            raise ValueError(f"{name} already written")
        group = self._h5.create_group(name, track_order=True)
        for key in STORE_KEYS:
            arr = record.arrays[key]
            chunks = (1, *arr.shape[1:]) if key == "images" else arr.shape
            group.create_dataset(key, data=arr, compression="lzf", chunks=chunks, track_times=False)
        for key, value in record.meta.as_attrs().items():
            group.attrs[key] = value
        self.n_episodes += 1

    def close(self) -> None:
        """Flush and close the file."""
        self._h5.close()

    def __enter__(self) -> DemoWriter:
        """Return self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()


# --------------------------------------------------------------------------- reader


@dataclass(frozen=True)
class StudentEpisode:
    """What a student may be built from, for one episode: pixels and allowlisted proprio.

    There is deliberately no other field (``tests/test_data_store.py`` pins the field set).

    Attributes:
        images: ``uint8[T, 96, 96, 3]``.
        proprio: ``float32[T, 13]``, fields in ``student.yaml`` order (``proprio_fields`` of
            the file).
    """

    images: NDArray[np.uint8]
    proprio: NDArray[np.float32]


class DemoStore:
    """Read a demo store.

    Attributes:
        path: The store file.
        file_attrs: The file attributes (:data:`FILE_ATTR_KEYS`).
    """

    def __init__(self, path: Path) -> None:
        """Open the file read-only and check its format and contract field names.

        Raises:
            StoreSchemaError: If the format tag, the file attributes or the stored proprio
                field names differ from the committed contract.
        """
        self.path = path
        self._h5 = h5py.File(path, "r")
        self.file_attrs: dict[str, Any] = {k: _plain(v) for k, v in self._h5.attrs.items()}
        if set(self.file_attrs) != set(FILE_ATTR_KEYS) or self.file_attrs["format"] != FORMAT:
            self._h5.close()
            raise StoreSchemaError(f"{path}: not an {FORMAT} store")
        c = load_contract()
        if json.loads(self.file_attrs["proprio_fields"]) != list(c.student_proprio):
            self._h5.close()
            raise StoreSchemaError(f"{path}: proprio fields differ from student.yaml")

    def episodes(self) -> list[str]:
        """Return the episode group names, in write order."""
        return list(self._h5.keys())

    def __len__(self) -> int:
        """Return the number of episodes."""
        return len(self._h5)

    def __iter__(self) -> Iterator[str]:
        """Iterate over episode group names, in write order."""
        return iter(self.episodes())

    def _group(self, name: str) -> h5py.Group:
        group = self._h5[name]
        if not isinstance(group, h5py.Group) or tuple(group.keys()) != STORE_KEYS:
            keys = list(group.keys()) if isinstance(group, h5py.Group) else None
            raise StoreSchemaError(f"{name}: datasets {keys} != {list(STORE_KEYS)}")
        return group

    def meta(self, name: str) -> EpisodeMeta:
        """Return one episode's attributes (for filtering and reports; never a model input)."""
        return EpisodeMeta.from_attrs({k: _plain(v) for k, v in self._group(name).attrs.items()})

    def student_inputs(self, name: str, variant: str = "default") -> StudentEpisode:
        """THE student read path: ``images`` and the variant's ``proprio``, nothing else.

        ``default`` reads exactly the datasets in :data:`STUDENT_INPUT_KEYS` and returns the
        stored ``proprio`` unchanged. An ablation variant (``minus_last_action``,
        ``plus_clock``) builds its proprio here, inside ``ila/data``, from ``teacher_obs`` by
        field name through the variant's ``ObsContract`` (:func:`variant_contract`: exactly the
        default fields minus ``last_action`` / plus ``time_fraction``); ``teacher_obs`` itself
        never leaves this method. This is the only way a dataset gets variant proprio.

        Args:
            name: Episode group name.
            variant: An ``ila.obs.split.VARIANTS`` name.

        Returns:
            The :class:`StudentEpisode`: ``images`` uint8 ``(T, 96, 96, 3)``, ``proprio``
            float32 ``(T, D)`` with ``D`` 13 (default), 10 (``minus_last_action``) or 14
            (``plus_clock``).

        Raises:
            ValueError: On an unknown variant.
            StoreSchemaError: If the variant's contract is not default +- one named field.
        """
        contract = variant_contract(variant)
        group = self._group(name)
        images = np.asarray(group["images"][()], dtype=np.uint8)
        if variant == "default":
            proprio = np.asarray(group["proprio"][()], dtype=np.float32)
        else:
            proprio = contract.proprio(np.asarray(group["teacher_obs"][()], dtype=np.float32))
        return StudentEpisode(images=images, proprio=proprio)

    def labels(self, name: str) -> NDArray[np.float32]:
        """Return the training targets ``label`` ``float32[T, 3]`` (normalised actions)."""
        return np.asarray(self._group(name)["label"][()], dtype=np.float32)

    def executed_actions(self, name: str) -> NDArray[np.float32]:
        """Return ``action_exec`` ``float32[T, 3]`` (diagnostics: copycat, DART audit)."""
        return np.asarray(self._group(name)["action_exec"][()], dtype=np.float32)

    def estimator_targets(self, name: str) -> NDArray[np.float32]:
        """Return ``priv`` ``float32[T, 11]``: the ``modular`` estimator's TARGET only."""
        return np.asarray(self._group(name)["priv"][()], dtype=np.float32)

    def privileged_observations(self, name: str) -> NDArray[np.float32]:
        """Return ``teacher_obs`` ``float32[T, 25]``: audits and ``bc_state`` only.

        Never a vision-student input (non-negotiable 2).
        """
        return np.asarray(self._group(name)["teacher_obs"][()], dtype=np.float32)

    def model_times(self, name: str) -> NDArray[np.float64]:
        """Return ``t_model_s`` ``float64[T]`` (s model scale): audits only, never a batch."""
        return np.asarray(self._group(name)["t_model_s"][()], dtype=np.float64)

    def audit_arrays(self, name: str) -> dict[str, NDArray[Any]]:
        """Return every dataset of one episode: byte-level audits ONLY, never a batch."""
        group = self._group(name)
        return {k: np.asarray(group[k][()]) for k in STORE_KEYS}

    def validate_episode(self, name: str) -> None:
        """Re-validate one stored episode (:func:`validate`)."""
        validate(self.audit_arrays(name), self.meta(name))

    def close(self) -> None:
        """Close the file."""
        self._h5.close()

    def __enter__(self) -> DemoStore:
        """Return self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()


def _plain(value: Any) -> Any:
    """Return an HDF5 attribute as a plain Python value (arrays kept)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def contract_field_attrs(contract: ObsContract | None = None) -> dict[str, str]:
    """Return the file attributes naming the stored fields (JSON lists), by contract."""
    c = load_contract() if contract is None else contract
    return {
        "proprio_fields": json.dumps(list(c.student_proprio)),
        "priv_fields": json.dumps(list(c.privileged)),
        "obs_fields": json.dumps(list(c.layout)),
    }


def total_frames(store: DemoStore, names: Sequence[str] | None = None) -> int:
    """Return the number of stored frames (sum of ``T``) over episodes."""
    return sum(store.meta(n).steps for n in (store.episodes() if names is None else names))
