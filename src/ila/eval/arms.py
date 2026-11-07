"""Phase 8 evaluation matrix: ``matrix.yaml`` -> jobs, and arm -> its selected checkpoint.

Everything here is simulation only (PyBullet physics, PyBullet TinyRenderer frames). Episode
counts are per P8 frozen-list cell (200 episodes per cell, P3-D1 §1).

The matrix (P3-D1 §1, §3, §6; ``configs/eval/matrix.yaml`` SHA-256 ``4d8a78fd…e96a``)
------------------------------------------------------------------------------------
:func:`load_matrix` reads the YAML through :func:`ila.eval.stats.read_matrix_yaml` (which refuses
the committed file unless its SHA-256 is P3-D1's) and :func:`expand` turns it into
:class:`Job` s. A job is one output file: (tag, arm, seed, list, sea states of that list,
inference ``(Ta, K, ensemble)``, appearance shift). Every top-level key of the YAML is either
expanded or listed in :data:`SKIPPED_BLOCKS` with its reason; an unknown key raises, so nothing
is dropped silently.

* ``core`` -- 7 arms x seeds 0-4 x every list (``lists_all``: 2 800 episodes per seed). The
  diffusion arm ``dp`` flies at its core inference ``Ta = 8``, ``K = 10``, no ensembling.
* ``dp_Ta`` -- ``dp`` seeds 0-4, ``Ta in {1, 2, 4, 8, 12}`` x ensembling ``{off, on}`` at the
  core ``K``, on ``id`` SS4-SS6. Two of the ten settings are **shared cells**, not flown:

  - ``(Ta = 8, off)`` is the core ``dp`` setting; its rows are the ``id`` SS4-SS6 rows of the
    core ``dp`` job of the same seed (same checkpoint, same episodes, same inference; the
    flight would be bit-identical, P2-D3 item 1). P3-D1 §6 H3 scores ``Ta = 1`` against
    exactly these rows.
  - ``(Ta = 1, on)`` is the same policy as ``(Ta = 1, off)``: with ensembling the window is
    ``Ta`` and a window of 1 is one source of weight 1.0
    (:func:`ila.policies.dp.chunking.ensemble_sources`; ``ChunkedDiffusionStudent``: "the
    action is bit-identical to ``Ta = 1`` without ensembling", tested in
    ``tests/test_policies.py``), with the same noise keys ``(run_seed, episode_seed, t)``. As
    on ``select`` (P6-D1 item 3: "``Ta`` = 1 on is recorded as an alias of ``Ta`` = 1 off"),
    it is recorded as an alias of the ``(Ta = 1, off)`` job. ``alias_ta1_ensemble=False``
    flies it instead.

  For ``Ta > 1``, ensembling is a different policy at every ``Ta``: it replans every step and
  averages the ``Ta`` most recent predictions of the step (``Ta`` is the ensemble window), so
  ``(Ta, on)`` for ``Ta in {2, 4, 8, 12}`` are four distinct flown cells, each costing one
  inference call per control step.
* ``dp_K`` -- ``dp`` seeds 0-4, ``K in {1, 2, 5, 10, 25, 100}`` at the core ``Ta``, no
  ensembling, on ``id`` SS5. ``K = 10`` is the core setting: a shared cell (core ``dp`` rows,
  ``id`` SS5).
* ``appearance_shift`` -- ``bc_dagger``, ``dp``, ``modular`` x every variant of
  :data:`ila.render.config.APPEARANCE_SHIFTS` x the arm's seeds (0-4) x ``id`` SS5-SS6; ``dp`` at
  its core inference.
* ``ablation`` -- the 11 ablations x seeds 0-2 (:data:`ABLATION_SEEDS`, P3-D1 §3) x ``id``
  SS4-SS6 (:data:`ABLATION_LISTS`, P3-D1 §3); the ``dp_*`` ablations at the core ``dp``
  inference.
* ``reference_external`` -- ``R1_lerobot_dp`` seeds 0-2, ``id`` SS3-SS6, at the core ``dp``
  inference (P3-D1 §4: R1 is selected by the same rule at ``Ta = 8``, ``K = 10``; its manifest's
  ``selection.at`` is checked).
* ``latency_arm`` -- Phase 9 (P3-D1 §7): ``bc_dagger`` and ``dp`` x their core seeds (0-4) x
  ``id`` SS4-SS6, ``dp`` at its core inference (``Ta = 8``, ``K = 10``, no ensembling), each
  arm delayed by its one ``d`` control steps (30 Hz; 33.333 ms model scale each, 166.7 ms full
  scale). ``d`` comes only from the committed ``results/latency/delay.json``
  (:func:`ila.deploy.latency_arm.read_delays`, written by ``make bench-score``); without it
  expansion raises :class:`LatencyDelayError` (never a guessed ``d``). The arm specs are the
  core arms' (same checkpoints, same selection pins); only the tag, the delay and the output
  name differ (``<stem>__latency-d<d>``). Rows carry ``latency_steps = d``. The seeds come from
  the core arm (``matrix.yaml``'s block names neither seeds nor ``Ta``/``K``; P3-D1 §3 / §7 do,
  as :data:`ABLATION_SEEDS` precedent). ``latency_arm=False`` (:func:`load_matrix`) records the
  block in ``skipped`` instead, for callers that only need core arm specs (the GIF gallery);
  the driver and the scorer never pass it.

Arms -> checkpoints (:func:`resolve_checkpoint`)
------------------------------------------------
Each (arm, seed) resolves to exactly one committed
``results/phase{4,5,6,7}/runs/<arm>/s<seed>/selection.json``; its ``selected.candidate``
(``file``, ``sha256``) is the checkpoint flown. The record and the committed ``manifest.json``
beside it are cross-checked against the arm's expected teacher (``T_ppo``; ``T_pid`` for
``*_pid``; ``T_mix`` for ``*_mix``), tag, contract variant, inference setting and candidate
kind. :func:`verify_checkpoint_bytes` re-hashes the file (an ONNX export directory: its bundle
hash, :func:`ila.train.select.candidate_sha256`) before it is flown. The contract variant a
checkpoint flies with is read from the checkpoint itself by
:func:`ila.train.select.fly_checkpoint`; nothing here overrides it.

Shadow teachers (``teacher_agreement``)
--------------------------------------
:data:`SHADOW_TEACHER`: ``T_ppo`` arms are shadowed by P8's ``ppo`` seed 0, ``T_pid`` arms by
``pid_feedforward``. ``T_mix`` arms are shadowed per episode: ``ila.data.mix.assign_rows`` (the
H2 reference's construction, via :func:`ila.eval.h2_select.tmix_assignment`) splits each cell's
episodes into a ``ppo`` half and a ``gated`` half (:mod:`ila.eval.matrix`).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.deploy import latency_arm as deploy_latency_arm
from ila.eval.stats import MATRIX_YAML, read_matrix_yaml
from ila.policies.registry import read_checkpoint
from ila.render.config import APPEARANCE_SHIFTS
from ila.train import manifest as mf
from ila.train import select as train_select

__all__ = [
    "ABLATION_LISTS",
    "LATENCY_TAG",
    "ABLATION_SEEDS",
    "DEFAULT_ORDER",
    "EPISODES_PER_CELL",
    "R1_ARM",
    "RESULTS_ROOT",
    "SELECTION_PHASES",
    "SHADOW_TEACHER",
    "SKIPPED_BLOCKS",
    "TAGS",
    "ArmResolutionError",
    "ArmSpec",
    "Checkpoint",
    "Inference",
    "Job",
    "LatencyDelayError",
    "Matrix",
    "SharedCell",
    "expand",
    "expected_obs_variant",
    "expected_teacher",
    "frozen_list_cells",
    "is_diffusion",
    "list_episodes",
    "read_delay_file",
    "load_matrix",
    "resolve_checkpoint",
    "selection_path",
    "verify_checkpoint_bytes",
]

#: The latency-in-the-loop arm's tag (P3-D1 §7; Phase 9).
LATENCY_TAG: Final[str] = "latency_arm"

#: Every job tag, in the default flying order (core first, the external reference, then the
#: Phase 9 latency arm last).
TAGS: Final[tuple[str, ...]] = (
    "core",
    "ablation",
    "dp_Ta",
    "dp_K",
    "appearance_shift",
    "reference_external",
    LATENCY_TAG,
)

#: Default order in which ``scripts/eval_matrix.py`` flies the tags.
DEFAULT_ORDER: Final[tuple[str, ...]] = TAGS

#: Blocks of ``matrix.yaml`` never expanded, with the reason (never dropped silently). Empty
#: since Phase 9 enabled ``latency_arm`` (it was the only one; P8-D1 item 1).
SKIPPED_BLOCKS: Final[Mapping[str, str]] = {}

#: ``matrix.yaml``'s ``latency_arm`` rule, provider and fallback, as ``ila.deploy.latency.score``
#: implements them (P3-D1 §6 H5(a), §7; P9-D1 item 9). Any other text is refused.
_LATENCY_BLOCK: Final[Mapping[str, Any]] = {
    "rule": "d = ceil(p99_ms / 33.333)",
    "provider": "fastest_parity_passing",
    "fallback": "torch_cpu_fp32_single_sample",
}

#: The latency arm's arms and cells (P3-D1 §7).
_LATENCY_ARMS: Final[tuple[str, ...]] = ("bc_dagger", "dp")
_LATENCY_LISTS: Final[Mapping[str, tuple[str, ...]]] = {"id": ("SS4", "SS5", "SS6")}

#: Why the block is recorded in ``skipped`` when a caller passes ``latency_arm=False``.
_LATENCY_NOT_EXPANDED: Final[str] = (
    "not expanded by this caller (core arm specs only); the driver and the scorer always "
    "expand it, with d from results/latency/delay.json (P3-D1 §7)"
)

#: ``matrix.yaml`` keys that hold no arm (read by :mod:`ila.eval.stats` and the report).
_NON_ARM_KEYS: Final[frozenset[str]] = frozenset(
    {"bootstrap_seed", "replicates", "always_printed", "lists_all"}
)

#: ``matrix.yaml`` keys this module expands.
_ARM_KEYS: Final[frozenset[str]] = frozenset(
    {"core_arms", "sweeps", "appearance_shift", "ablations", "reference"}
)

#: Ablation seeds (P3-D1 §3: "3 seeds each"; ``matrix.yaml`` states it in a comment only).
ABLATION_SEEDS: Final[int] = 3

#: Ablation cells (P3-D1 §3: "flown on ``id`` SS4-SS6"; ``matrix.yaml`` comment).
ABLATION_LISTS: Final[Mapping[str, tuple[str, ...]]] = {"id": ("SS4", "SS5", "SS6")}

#: Episodes in every P8 frozen-list cell (P3-D1 §1).
EPISODES_PER_CELL: Final[int] = 200

#: The external reference arm (P3-D1 §3, tag ``reference_external``).
R1_ARM: Final[str] = "R1_lerobot_dp"

#: Committed results root.
RESULTS_ROOT: Final[Path] = p8_adapter.REPO_ROOT / "results"

#: Phases whose ``results/phase<N>/runs/<arm>/s<seed>/selection.json`` hold the selections.
SELECTION_PHASES: Final[tuple[str, ...]] = ("phase4", "phase5", "phase6", "phase7")

#: Teacher label -> the P8 teacher flown in lockstep for ``teacher_agreement`` (``None``:
#: per episode, by the ``T_mix`` assignment; :mod:`ila.eval.matrix`).
SHADOW_TEACHER: Final[Mapping[str, str | None]] = {
    "T_ppo": "ppo",
    "T_pid": "pid_feedforward",
    "T_mix": None,
}

#: Selection inference of every diffusion arm (P3-D1 §4; ``selection.json`` ``inference``).
_SELECT_INFERENCE: Final[Mapping[str, Any]] = {"Ta": 8, "K": 10, "temporal_ensembling": False}


class ArmResolutionError(ValueError):
    """An (arm, seed) does not resolve to exactly one consistent selected checkpoint."""


class LatencyDelayError(ValueError):
    """The ``latency_arm`` block cannot be expanded: no committed ``d`` for an arm.

    ``d`` is read only from ``results/latency/delay.json`` (``make bench-score``); it is never
    guessed (P3-D1 §7).
    """


# --------------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class Inference:
    """Diffusion inference setting of a job.

    Attributes:
        Ta: Executed actions per chunk; with ensembling, the ensemble window (control steps,
            1/30 s model scale each).
        K: DDIM denoising steps.
        ensemble: ACT temporal ensembling (m = 0.01; replans every control step).
    """

    Ta: int
    K: int
    ensemble: bool

    def as_tuple(self) -> tuple[int, int, bool]:
        """``(Ta, K, ensemble)``, :func:`ila.train.select.fly_checkpoint`'s ``inference``."""
        return (self.Ta, self.K, self.ensemble)

    @property
    def suffix(self) -> str:
        """File-name part, e.g. ``Ta8_K10_ens0``."""
        return f"Ta{self.Ta}_K{self.K}_ens{int(self.ensemble)}"


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the matrix.

    Attributes:
        arm: Arm name (``matrix.yaml``).
        tag: The arm's own tag: ``core``, ``ablation`` or ``reference_external``.
        seeds: Training seeds evaluated.
        teacher: Teacher label of its training data (``T_ppo``, ``T_pid``, ``T_mix``).
        obs_variant: Observation-contract variant it was trained on (checked against the
            selection record; the flight reads it from the checkpoint).
        inference: Diffusion arms: the core inference; ``None`` otherwise.
    """

    arm: str
    tag: str
    seeds: tuple[int, ...]
    teacher: str
    obs_variant: str
    inference: Inference | None


@dataclass(frozen=True)
class Job:
    """One flown cell group: one output file.

    Attributes:
        tag: One of :data:`TAGS`.
        arm: Arm name.
        seed: Training seed of the run flown.
        list_name: P8 frozen list (``select`` only in a smoke run).
        sea_states: Cells of that list flown, in the list's order.
        full_list: Whether ``sea_states`` is every cell of the list.
        inference: Diffusion inference, or ``None`` (non-diffusion arms).
        appearance_shift: Evaluation-only render variant, or ``None``.
        latency_steps: The latency arm's delay ``d``, control steps (30 Hz; 33.333 ms model
            scale each), or ``None`` for every other job (synchronous, rows carry 0). ``None``
            leaves :attr:`stem` and :meth:`record` exactly as in Phase 8, so the committed
            sidecars still match (:func:`ila.eval.matrix.job_done`).
    """

    tag: str
    arm: str
    seed: int
    list_name: str
    sea_states: tuple[str, ...]
    full_list: bool
    inference: Inference | None = None
    appearance_shift: str | None = None
    latency_steps: int | None = None

    @property
    def stem(self) -> str:
        """Deterministic file stem.

        ``<list>[-<SS>...][__shift-<v>][__Ta<k>_K<k>_ens<0|1>][__latency-d<d>]``: the list name
        alone when every cell of the list is flown; otherwise the list name and the flown cells
        joined by ``-`` (e.g. ``id-SS4-SS5-SS6``).
        """
        parts = [self.list_name if self.full_list else "-".join((self.list_name, *self.sea_states))]
        if self.appearance_shift is not None:
            parts.append(f"shift-{self.appearance_shift}")
        if self.inference is not None:
            parts.append(self.inference.suffix)
        if self.latency_steps is not None:
            parts.append(f"latency-d{self.latency_steps}")
        return "__".join(parts)

    @property
    def rel_path(self) -> Path:
        """Rows file relative to the output root: ``rows/<arm>/s<seed>/<stem>.csv.gz``."""
        return Path("rows") / self.arm / f"s{self.seed}" / f"{self.stem}.csv.gz"

    @property
    def n_episodes(self) -> int:
        """Episodes flown on a frozen list (200 per cell)."""
        return EPISODES_PER_CELL * len(self.sea_states)

    def record(self) -> dict[str, Any]:
        """JSON-serialisable spec (the resumability key in each sidecar)."""
        inf = self.inference
        return {
            "tag": self.tag,
            "arm": self.arm,
            "seed": self.seed,
            "list": self.list_name,
            "sea_states": list(self.sea_states),
            "full_list": self.full_list,
            "Ta": None if inf is None else inf.Ta,
            "K": None if inf is None else inf.K,
            "ensemble": None if inf is None else inf.ensemble,
            "appearance_shift": self.appearance_shift,
            **({} if self.latency_steps is None else {"latency_steps": self.latency_steps}),
        }


@dataclass(frozen=True)
class SharedCell:
    """A matrix cell not flown because another job's rows are its rows.

    Attributes:
        tag: The cell's tag (``dp_Ta`` or ``dp_K``).
        arm: Arm.
        seed: Seed.
        list_name: List.
        sea_states: Its cells (a subset of ``source.sea_states``).
        inference: Its inference setting.
        source: The flown job whose rows (restricted to ``sea_states``) are this cell's.
        reason: Why the rows are the same.
    """

    tag: str
    arm: str
    seed: int
    list_name: str
    sea_states: tuple[str, ...]
    inference: Inference
    source: Job
    reason: str

    def record(self) -> dict[str, Any]:
        """JSON-serialisable record (``run_info.json``)."""
        return {
            "tag": self.tag,
            "arm": self.arm,
            "seed": self.seed,
            "list": self.list_name,
            "sea_states": list(self.sea_states),
            "Ta": self.inference.Ta,
            "K": self.inference.K,
            "ensemble": self.inference.ensemble,
            "rows_from": self.source.rel_path.as_posix(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Matrix:
    """The expanded matrix.

    Attributes:
        arms: Every arm with its own tag, seeds, teacher and core inference.
        jobs: Flown jobs, in expansion order (tag blocks in :data:`TAGS` order).
        shared: Cells whose rows come from another job.
        skipped: Blocks not expanded -> reason (empty for the driver and the scorer).
        lists_all: Every frozen list, ``matrix.yaml`` order.
        always_printed: Baselines printed beside every student (P8's committed rows).
    """

    arms: tuple[ArmSpec, ...]
    jobs: tuple[Job, ...]
    shared: tuple[SharedCell, ...]
    skipped: Mapping[str, str]
    lists_all: tuple[str, ...]
    always_printed: tuple[str, ...]

    def arm(self, name: str) -> ArmSpec:
        """Return the :class:`ArmSpec` of ``name``.

        Raises:
            KeyError: If the matrix has no such arm.
        """
        for a in self.arms:
            if a.arm == name:
                return a
        raise KeyError(name)


# --------------------------------------------------------------------------- arm properties


def is_diffusion(arm: str) -> bool:
    """Whether an arm samples chunks by diffusion (``dp``, ``dp_*``, ``R1_lerobot_dp``)."""
    return arm == "dp" or arm.startswith("dp_") or arm == R1_ARM


def expected_teacher(arm: str) -> str:
    """Teacher label of an arm's training data (P3-D1 §3).

    ``T_pid`` for ``*_pid``, ``T_mix`` for ``*_mix``, else ``T_ppo`` (core arms, R1, ``*_n*``,
    ``*_minus_last_action``, ``bc_plus_clock``, ``modular``).
    """
    if arm.endswith("_pid"):
        return "T_pid"
    if arm.endswith("_mix"):
        return "T_mix"
    return "T_ppo"


def expected_obs_variant(arm: str) -> str:
    """Observation-contract variant an arm was trained on (P3-D1 §3 ablations)."""
    if arm.endswith("_minus_last_action"):
        return "minus_last_action"
    if arm.endswith("_plus_clock"):
        return "plus_clock"
    return "default"


# --------------------------------------------------------------------------- lists


@functools.cache
def list_episodes(name: str) -> tuple[p8_adapter.ListedEpisode, ...]:
    """Return a P8 frozen list's episodes, in committed order (read once per process).

    Evaluation only (non-negotiable 4); called by the Phase 8 driver after every run of every
    arm has been selected (P3-D1 §4).
    """
    return tuple(p8_adapter.read_list(p8_adapter.list_path(name)))


def frozen_list_cells() -> dict[str, tuple[str, ...]]:
    """Return list name -> its cells in list order (``id``: SS3-SS6; ``static``: ``static``).

    Raises:
        ValueError: If a cell does not hold exactly :data:`EPISODES_PER_CELL` episodes.
    """
    out: dict[str, tuple[str, ...]] = {}
    for name in p8_adapter.list_names():
        eps = list_episodes(name)
        cells = tuple(dict.fromkeys(e.ss for e in eps))
        for c in cells:
            n = sum(e.ss == c for e in eps)
            if n != EPISODES_PER_CELL:
                raise ValueError(f"{name} {c}: {n} episodes, not {EPISODES_PER_CELL}")
        out[name] = cells
    return out


# --------------------------------------------------------------------------- expansion


def read_delay_file(path: Path = deploy_latency_arm.DELAY_FILE) -> dict[str, int]:
    """The latency arm's ``d`` per arm from the committed ``delay.json`` (P3-D1 §7).

    Args:
        path: ``results/latency/delay.json`` (``make bench-score``).

    Returns:
        Arm -> ``d``, control steps (30 Hz; 33.333 ms model scale each).

    Raises:
        LatencyDelayError: If the file does not exist, an arm has no ``d`` (its fallback row
            was not timed), or the file cannot be read.
    """
    if not Path(path).is_file():
        raise LatencyDelayError(
            f"{path} does not exist: the latency_arm block's delay d per arm comes only from "
            "Phase 9's latency study (make bench-score writes it from parity-passing p99s; "
            "P3-D1 §6 H5(a), §7). d is not guessed; refusing to expand the matrix."
        )
    try:
        return dict(deploy_latency_arm.read_delays(Path(path)))
    except (KeyError, TypeError, ValueError) as exc:
        raise LatencyDelayError(f"{path}: no usable d ({exc}); refusing to expand") from exc


def load_matrix(
    path: Path = MATRIX_YAML,
    list_cells: Mapping[str, Sequence[str]] | None = None,
    *,
    alias_ta1_ensemble: bool = True,
    delays: Mapping[str, int] | None = None,
    delay_file: Path = deploy_latency_arm.DELAY_FILE,
    latency_arm: bool = True,
) -> Matrix:
    """Read the frozen matrix (SHA-256 checked) and expand it.

    Args:
        path: ``configs/eval/matrix.yaml``.
        list_cells: List -> cells; read from P8's frozen lists when ``None``.
        alias_ta1_ensemble: Record ``(Ta = 1, on)`` as an alias of ``(Ta = 1, off)``.
        delays: The latency arm's ``d`` per arm (control steps); ``None`` reads
            ``delay_file`` (:func:`read_delay_file`). Tests pass synthetic values.
        delay_file: ``results/latency/delay.json``.
        latency_arm: ``False`` records the ``latency_arm`` block in ``skipped`` instead of
            expanding it (callers needing core arm specs only; module docstring).

    Returns:
        The :class:`Matrix`.

    Raises:
        LatencyDelayError: If ``latency_arm`` and no ``d`` is available (module docstring).
    """
    cells = frozen_list_cells() if list_cells is None else list_cells
    raw = read_matrix_yaml(path)
    if not latency_arm:
        return expand(raw, cells, alias_ta1_ensemble=alias_ta1_ensemble, latency_arm=False)
    d = read_delay_file(delay_file) if delays is None else delays
    return expand(raw, cells, alias_ta1_ensemble=alias_ta1_ensemble, delays=d)


def _cells(
    spec: Any, lists_all: Sequence[str], list_cells: Mapping[str, Sequence[str]]
) -> list[tuple[str, tuple[str, ...], bool]]:
    """``lists: all`` or ``{list: [SS...]}`` -> ``[(list, cells in list order, full)]``.

    Raises:
        ValueError: On an unknown list or cell.
    """
    if spec == "all":
        return [(n, tuple(list_cells[n]), True) for n in lists_all]
    if not isinstance(spec, Mapping):
        raise ValueError(f"lists must be 'all' or a mapping, got {spec!r}")
    out = []
    for name, wanted in spec.items():
        if name not in list_cells:
            raise ValueError(f"unknown list {name!r}")
        have = tuple(list_cells[name])
        unknown = set(wanted) - set(have)
        if unknown:
            raise ValueError(f"{name}: unknown cells {sorted(unknown)}")
        ordered = tuple(c for c in have if c in set(wanted))
        out.append((str(name), ordered, ordered == have))
    return out


def _arm_spec(arm: str, tag: str, seeds: int, inference: Inference | None) -> ArmSpec:
    if is_diffusion(arm) and inference is None:
        raise ValueError(f"{arm}: a diffusion arm needs an inference setting")
    return ArmSpec(
        arm=arm,
        tag=tag,
        seeds=tuple(range(int(seeds))),
        teacher=expected_teacher(arm),
        obs_variant=expected_obs_variant(arm),
        inference=inference if is_diffusion(arm) else None,
    )


def expand(
    raw: Mapping[str, Any],
    list_cells: Mapping[str, Sequence[str]],
    *,
    alias_ta1_ensemble: bool = True,
    delays: Mapping[str, int] | None = None,
    latency_arm: bool = True,
) -> Matrix:
    """Expand a parsed ``matrix.yaml`` into jobs (module docstring).

    Args:
        raw: The parsed YAML (:func:`ila.eval.stats.read_matrix_yaml`).
        list_cells: List -> its cells in list order (:func:`frozen_list_cells`).
        alias_ta1_ensemble: Record ``(Ta = 1, on)`` as an alias of ``(Ta = 1, off)`` (P6-D1
            precedent) instead of flying it.
        delays: The latency arm's ``d`` per arm, control steps (:func:`read_delay_file`).
            Required unless ``latency_arm`` is False.
        latency_arm: Expand the ``latency_arm`` block (``False``: record it in ``skipped``).

    Returns:
        The :class:`Matrix`.

    Raises:
        ValueError: On an unknown top-level key, a missing block, an arm listed twice, or an
            inconsistent sweep or latency block.
        LatencyDelayError: If the latency arm is expanded without a ``d`` for one of its arms,
            or with a negative ``d``.
    """
    unknown = set(raw) - _NON_ARM_KEYS - _ARM_KEYS - {LATENCY_TAG} - set(SKIPPED_BLOCKS)
    if unknown:
        raise ValueError(f"matrix.yaml: keys neither expanded nor skipped: {sorted(unknown)}")
    missing = (_ARM_KEYS | {LATENCY_TAG} | set(SKIPPED_BLOCKS)) - set(raw)
    if missing:
        raise ValueError(f"matrix.yaml: missing blocks {sorted(missing)}")
    if latency_arm and delays is None:
        raise LatencyDelayError(
            "the latency_arm block needs d per arm from results/latency/delay.json "
            "(make bench-score; P3-D1 §7); d is not guessed"
        )
    lists_all = tuple(str(n) for n in raw["lists_all"])
    if set(lists_all) != set(list_cells):
        raise ValueError(f"lists_all {lists_all} != frozen lists {sorted(list_cells)}")

    jobs: list[Job] = []
    shared: list[SharedCell] = []
    arms: list[ArmSpec] = []

    # core
    core_jobs: dict[tuple[str, int, str], Job] = {}
    core_inf: Inference | None = None
    for entry in raw["core_arms"]:
        arm = str(entry["arm"])
        inf = None
        if "Ta" in entry or "K" in entry:
            inf = Inference(int(entry["Ta"]), int(entry["K"]), False)
            if arm == "dp":
                core_inf = inf
        spec = _arm_spec(arm, "core", int(entry["seeds"]), inf)
        arms.append(spec)
        for seed in spec.seeds:
            for name, ss, full in _cells(entry["lists"], lists_all, list_cells):
                job = Job("core", arm, seed, name, ss, full, spec.inference)
                core_jobs[(arm, seed, name)] = job
                jobs.append(job)
    if core_inf is None:
        raise ValueError("matrix.yaml: core dp has no Ta/K")

    # ablations (3 seeds, id SS4-SS6; dp ablations at the core dp inference)
    for entry in raw["ablations"]:
        arm = str(entry["arm"])
        spec = _arm_spec(arm, "ablation", ABLATION_SEEDS, core_inf if is_diffusion(arm) else None)
        arms.append(spec)
        for seed in spec.seeds:
            for name, ss, full in _cells(dict(ABLATION_LISTS), lists_all, list_cells):
                jobs.append(Job("ablation", arm, seed, name, ss, full, spec.inference))

    by_arm = {a.arm: a for a in arms}

    # dp_Ta sweep
    sw = raw["sweeps"]["dp_Ta"]
    dp = by_arm[str(sw["arm"])]
    assert dp.inference is not None
    for seed in dp.seeds:
        for name, ss, full in _cells(sw["lists"], lists_all, list_cells):
            ta1_off: Job | None = None
            for ta in (int(t) for t in sw["Ta"]):
                for ens in (bool(e) for e in sw["ensembling"]):
                    inf = Inference(ta, dp.inference.K, ens)
                    if inf == dp.inference:
                        shared.append(
                            SharedCell(
                                "dp_Ta",
                                dp.arm,
                                seed,
                                name,
                                ss,
                                inf,
                                core_jobs[(dp.arm, seed, name)],
                                "the core dp setting: same checkpoint, episodes and inference "
                                "as the core job (no cell is re-flown)",
                            )
                        )
                        continue
                    job = Job("dp_Ta", dp.arm, seed, name, ss, full, inf)
                    if ta == 1 and ens and alias_ta1_ensemble:
                        if ta1_off is None:
                            raise ValueError("dp_Ta: (Ta=1, on) listed before (Ta=1, off)")
                        shared.append(
                            SharedCell(
                                "dp_Ta",
                                dp.arm,
                                seed,
                                name,
                                ss,
                                inf,
                                ta1_off,
                                "Ta = 1 with ensembling is bit-identical to Ta = 1 without "
                                "(one source of weight 1.0, same noise keys; "
                                "chunking.ensemble_sources, P6-D1 item 3 alias)",
                            )
                        )
                        continue
                    if ta == 1 and not ens:
                        ta1_off = job
                    jobs.append(job)

    # dp_K sweep
    sk = raw["sweeps"]["dp_K"]
    dp = by_arm[str(sk["arm"])]
    assert dp.inference is not None
    for seed in dp.seeds:
        for name, ss, full in _cells(sk["lists"], lists_all, list_cells):
            for k in (int(v) for v in sk["K"]):
                inf = Inference(dp.inference.Ta, k, False)
                if inf == dp.inference:
                    shared.append(
                        SharedCell(
                            "dp_K",
                            dp.arm,
                            seed,
                            name,
                            ss,
                            inf,
                            core_jobs[(dp.arm, seed, name)],
                            "the core dp setting: same checkpoint, episodes and inference "
                            "as the core job (no cell is re-flown)",
                        )
                    )
                    continue
                jobs.append(Job("dp_K", dp.arm, seed, name, ss, full, inf))

    # appearance shift (evaluation only; the arm's seeds; dp at its core inference)
    ap = raw["appearance_shift"]
    if ap["variants"] != "all":
        raise ValueError(f"appearance_shift.variants {ap['variants']!r} != 'all'")
    for arm in (str(a) for a in ap["arms"]):
        spec = by_arm[arm]
        for seed in spec.seeds:
            for name, ss, full in _cells(ap["lists"], lists_all, list_cells):
                for variant in APPEARANCE_SHIFTS:
                    jobs.append(
                        Job("appearance_shift", arm, seed, name, ss, full, spec.inference, variant)
                    )

    # external reference (R1 at the core dp inference, P3-D1 §4)
    ref = raw["reference"]
    if ref.get("tag") != "reference_external":
        raise ValueError(f"reference tag {ref.get('tag')!r}")
    spec = _arm_spec(str(ref["arm"]), "reference_external", int(ref["seeds"]), core_inf)
    arms.append(spec)
    for seed in spec.seeds:
        for name, ss, full in _cells(ref["lists"], lists_all, list_cells):
            jobs.append(Job("reference_external", spec.arm, seed, name, ss, full, spec.inference))

    # latency arm (Phase 9, P3-D1 §7): the core arm specs, one d per arm, id SS4-SS6
    skipped = dict(SKIPPED_BLOCKS)
    if latency_arm:
        assert delays is not None
        jobs.extend(_latency_jobs(raw[LATENCY_TAG], by_arm, delays, lists_all, list_cells))
    else:
        skipped[LATENCY_TAG] = _LATENCY_NOT_EXPANDED

    names = [a.arm for a in arms]
    if len(names) != len(set(names)):
        raise ValueError(f"an arm is listed twice: {names}")
    order = {t: i for i, t in enumerate(TAGS)}
    jobs_sorted = sorted(enumerate(jobs), key=lambda ij: (order[ij[1].tag], ij[0]))
    return Matrix(
        arms=tuple(arms),
        jobs=tuple(j for _, j in jobs_sorted),
        shared=tuple(shared),
        skipped=skipped,
        lists_all=lists_all,
        always_printed=tuple(str(b) for b in raw["always_printed"]),
    )


def _latency_jobs(
    block: Mapping[str, Any],
    by_arm: Mapping[str, ArmSpec],
    delays: Mapping[str, int],
    lists_all: Sequence[str],
    list_cells: Mapping[str, Sequence[str]],
) -> list[Job]:
    """The latency arm's jobs (module docstring; P3-D1 §7).

    Each arm reuses its **core** :class:`ArmSpec` (seeds, teacher, contract, core inference),
    so the checkpoints, the selection pins and the bytes check are the core arm's.

    Raises:
        ValueError: If the block is not P3-D1 §7's (arms, cells, rule, provider, fallback), or
            ``dp``'s core inference is not ``Ta = 8``, ``K = 10``, no ensembling.
        LatencyDelayError: If an arm has no ``d`` or a negative one.
    """
    for key, want in _LATENCY_BLOCK.items():
        if block.get(key) != want:
            raise ValueError(f"latency_arm.{key} {block.get(key)!r} != {want!r} (P3-D1 §7)")
    names = tuple(str(a) for a in block["arms"])
    if names != _LATENCY_ARMS:
        raise ValueError(f"latency_arm.arms {names} != {_LATENCY_ARMS} (P3-D1 §7)")
    if {str(k): tuple(v) for k, v in block["lists"].items()} != dict(_LATENCY_LISTS):
        raise ValueError(f"latency_arm.lists {block['lists']!r} != id SS4-SS6 (P3-D1 §7)")
    out = []
    for arm in names:
        spec = by_arm[arm]
        if spec.tag != "core":
            raise ValueError(f"latency_arm: {arm} is not a core arm")
        if is_diffusion(arm) and spec.inference != Inference(8, 10, False):
            raise ValueError(f"latency_arm: {arm} must fly Ta = 8, K = 10, no ensembling")
        if arm not in delays:
            raise LatencyDelayError(f"latency_arm: no d for {arm} in {dict(delays)}")
        d = int(delays[arm])
        if d < 0:
            raise LatencyDelayError(f"latency_arm: d = {d} for {arm} is negative")
        for seed in spec.seeds:
            for name, ss, full in _cells(block["lists"], lists_all, list_cells):
                out.append(Job(LATENCY_TAG, arm, seed, name, ss, full, spec.inference, None, d))
    return out


# --------------------------------------------------------------------------- checkpoints


@dataclass(frozen=True)
class Checkpoint:
    """The selected checkpoint of one (arm, seed), from its committed ``selection.json``.

    Attributes:
        arm: Arm.
        seed: Training seed.
        kind: ``checkpoint`` (a ``.pt``) or ``onnx_exports`` (R1's export directory).
        path: The checkpoint on disk (the recorded path, rebased onto this checkout's
            ``artifacts/runs/`` if the recorded absolute path does not exist).
        recorded_path: ``selected.candidate.file`` as recorded.
        sha256: ``selected.candidate.sha256`` (a file's SHA-256; an export's ``meta.json``
            bundle hash).
        teacher: Teacher label (``T_ppo``/``T_pid``/``T_mix``), cross-checked.
        selection_json: The record, repository-relative.
        selection_json_sha256: Its SHA-256.
        global_step: Selected checkpoint's global step.
        select_success: Its ``select`` successes (of ``select_n``).
        select_n: ``select`` episodes (150).
        select_mean_steps: Its mean control steps per ``select`` episode (30 Hz).
        selection_wall_clock_s: Wall clock of the whole selection (every candidate), s.
        selection_workers: Worker processes the selection used.
        selection_steps: Control steps flown by the whole selection (sum over candidates of
            ``n x mean_steps``).
        obs_variant: The contract variant the arm must fly (:func:`expected_obs_variant`;
            ``default`` for R1's ONNX export), enforced on the checkpoint itself by
            :func:`verify_checkpoint_bytes`.
    """

    arm: str
    seed: int
    kind: str
    path: Path
    recorded_path: str
    sha256: str
    teacher: str
    selection_json: str
    selection_json_sha256: str
    global_step: int
    select_success: int
    select_n: int
    select_mean_steps: float
    selection_wall_clock_s: float
    selection_workers: int
    selection_steps: float
    obs_variant: str

    @property
    def cpu_s_per_step(self) -> float:
        """Measured cost of one control step on ``select``: CPU-seconds (wall x workers)."""
        return self.selection_wall_clock_s * self.selection_workers / self.selection_steps

    def record(self) -> dict[str, Any]:
        """JSON-serialisable record (sidecars, ``run_info.json``)."""
        return {
            "arm": self.arm,
            "seed": self.seed,
            "kind": self.kind,
            "path": _rel(self.path),
            "sha256": self.sha256,
            "teacher": self.teacher,
            "selection_json": self.selection_json,
            "selection_json_sha256": self.selection_json_sha256,
            "global_step": self.global_step,
            "select_success": self.select_success,
            "select_n": self.select_n,
            "obs_variant": self.obs_variant,
        }


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def selection_path(arm: str, seed: int, results: Path = RESULTS_ROOT) -> Path:
    """Return the one committed ``selection.json`` of (arm, seed).

    Raises:
        ArmResolutionError: If none or more than one of :data:`SELECTION_PHASES` holds it.
    """
    found = [results / ph / "runs" / arm / f"s{seed}" / "selection.json" for ph in SELECTION_PHASES]
    found = [p for p in found if p.is_file()]
    if len(found) != 1:
        raise ArmResolutionError(f"{arm} s{seed}: {len(found)} selection.json files: {found}")
    return found[0]


def _local(recorded: str) -> Path:
    """The recorded checkpoint path, rebased onto this checkout if it does not exist."""
    p = Path(recorded)
    if p.exists():
        return p
    marker = "artifacts/runs/"
    text = p.as_posix()
    if marker in text:
        return p8_adapter.REPO_ROOT / text[text.index(marker) :]
    return p


def resolve_checkpoint(spec: ArmSpec, seed: int, results: Path = RESULTS_ROOT) -> Checkpoint:
    """Resolve (arm, seed) to its selected checkpoint and cross-check the records.

    Reads only the committed ``selection.json`` and ``manifest.json`` (no checkpoint bytes;
    :func:`verify_checkpoint_bytes` does that before a flight).

    Checks (each raising :class:`ArmResolutionError`): one record exists; ``arm``,
    ``run_seed``, ``purpose == "train"``; the list is ``select``; the manifest beside it is
    the same run and ``done``; the teacher (``selection.json`` ``teacher``, else
    :func:`ila.train.select.training_teacher` of the manifest's resolved config) is
    :func:`expected_teacher`; the recorded tag and ``obs_variant``, where recorded, are the
    arm's; a diffusion arm's recorded selection inference is ``Ta = 8, K = 10``, no ensembling
    (R1: its manifest's ``selection.at``), a non-diffusion arm records none; R1 is an ONNX
    export and no other arm is.

    Args:
        spec: The arm.
        seed: Training seed (must be in ``spec.seeds``).
        results: Committed results root.

    Returns:
        The :class:`Checkpoint`.
    """
    if seed not in spec.seeds:
        raise ArmResolutionError(f"{spec.arm}: seed {seed} not in {spec.seeds}")
    path = selection_path(spec.arm, seed, results)
    rec = mf.read_json(path)
    man_path = path.parent / "manifest.json"
    if not man_path.is_file():
        raise ArmResolutionError(f"{man_path}: missing")
    man = mf.read_json(man_path)

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise ArmResolutionError(f"{path}: {what}")

    need(rec.get("arm") == spec.arm and int(rec.get("run_seed", -1)) == seed, "arm/run_seed")
    need(rec.get("purpose") == "train", f"purpose {rec.get('purpose')!r}")
    need(rec.get("list", {}).get("name") == train_select.SELECT_SPLIT, "list is not select")
    need(man.get("arm") == spec.arm and int(man.get("run_seed", -1)) == seed, "manifest run")
    need(man.get("state") == "done", f"manifest state {man.get('state')!r}")
    teacher = rec.get("teacher") or train_select.training_teacher(man.get("config_resolved", {}))
    need(teacher == spec.teacher, f"teacher {teacher!r} != {spec.teacher!r}")
    tag = rec.get("tag", man.get("tag"))
    need(tag is None or tag == spec.tag, f"tag {tag!r} != {spec.tag!r}")
    variant = rec.get("obs_variant", man.get("obs_variant"))
    need(variant is None or variant == spec.obs_variant, f"obs_variant {variant!r}")
    kind = str(rec.get("candidate_kind", "checkpoint"))
    need((kind == train_select.ONNX_EXPORTS) == (spec.arm == R1_ARM), f"candidate kind {kind}")
    inf = rec.get("inference")
    if spec.arm == R1_ARM:
        at = man.get("config_resolved", {}).get("selection", {}).get("at")
        need(at == {"Ta": 8, "K": 10}, f"R1 selection.at {at!r}")
    elif is_diffusion(spec.arm):
        need(inf is None or inf == dict(_SELECT_INFERENCE), f"inference {inf!r}")
    else:
        need(inf is None, f"non-diffusion arm records inference {inf!r}")
    sel = rec["selected"]
    cand = sel["candidate"]
    sha = str(cand["sha256"])
    need(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "sha256 not hex")
    steps = sum(float(c["n"]) * float(c["mean_steps"]) for c in rec["candidates"])
    workers = int(rec.get("workers", 0))
    wall = float(rec.get("wall_clock_s", math.nan))
    need(steps > 0 and workers > 0 and math.isfinite(wall), "selection cost not recorded")
    return Checkpoint(
        arm=spec.arm,
        seed=seed,
        kind=kind,
        path=_local(str(cand["file"])),
        recorded_path=str(cand["file"]),
        sha256=sha,
        teacher=str(teacher),
        selection_json=_rel(path),
        selection_json_sha256=p8_adapter.sha256_file(path),
        global_step=int(cand["global_step"]),
        select_success=int(sel["success"]),
        select_n=int(sel["n"]),
        select_mean_steps=float(sel["mean_steps"]),
        selection_wall_clock_s=wall,
        selection_workers=workers,
        selection_steps=steps,
        obs_variant=spec.obs_variant,
    )


def verify_checkpoint_bytes(ckpt: Checkpoint) -> None:
    """Re-hash the checkpoint on disk, then check it is the arm's, on the arm's contract.

    Called before every flight (:func:`ila.eval.matrix.run_jobs`). The contract variant a flight
    uses is read from the checkpoint by :func:`ila.train.select.fly_checkpoint`, so the
    checkpoint's own ``arm``, ``run_seed`` and ``obs_variant`` must be the matrix arm's
    (privileged-leak review, Phase 8 RISK 1: a ``plus_clock`` checkpoint at a core arm's path
    would otherwise fly with the clock). An ONNX export (R1) always flies the ``default``
    contract, so its arm must expect ``default``.

    Raises:
        ArmResolutionError: If it is missing, its bytes differ, or its ``arm``, ``run_seed`` or
            ``obs_variant`` are not the arm's.
    """
    where = f"{ckpt.arm} s{ckpt.seed}: {ckpt.path}"
    if not ckpt.path.exists():
        raise ArmResolutionError(f"{where} missing")
    actual = train_select.candidate_sha256(ckpt.path)
    if actual != ckpt.sha256:
        raise ArmResolutionError(f"{where} SHA-256 {actual} != recorded {ckpt.sha256}")
    want_variant = expected_obs_variant(ckpt.arm)
    if ckpt.obs_variant != want_variant:
        raise ArmResolutionError(f"{where}: expects obs_variant {ckpt.obs_variant!r}")
    if ckpt.kind == train_select.ONNX_EXPORTS:
        if want_variant != "default":
            raise ArmResolutionError(f"{where}: an ONNX export flies 'default' only")
        return
    raw = read_checkpoint(ckpt.path)
    got = (raw.get("arm"), raw.get("run_seed"), raw.get("obs_variant"))
    if got[0] != ckpt.arm or got[1] is None or int(got[1]) != ckpt.seed or got[2] != want_variant:
        raise ArmResolutionError(
            f"{where}: checkpoint (arm, run_seed, obs_variant) {got} != "
            f"({ckpt.arm!r}, {ckpt.seed}, {want_variant!r})"
        )
