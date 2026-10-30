"""DAgger rollouts: beta schedule, step-by-step teacher shadowing, aggregation (Phase 5).

Plan §1.6 and P3-D1 §2-§3 (``bc_dagger`` and ``bc_matched`` share one schedule). Iteration 0 is
the 1 000 ``ppo_2000`` demos (the ``bc`` run of the same seed); iterations i = 1 .. 5 each fly
the current student on 200 ``train`` episodes and label every visited state with the teacher.
This module flies, records, audits and accounts for those rollouts; training on the aggregate
is ``ila.train.loop``'s.

Episodes. Iteration i flies global indices :meth:`ila.data.episodes.StreamDesign.dagger_indices`
``(i)`` of the ``demos`` stream (rows 1000 + 200(i-1) .. 1000 + 200i - 1 of
``splits/demos_0-1999.parquet``, read through ``MANIFEST.csv``), the same rows for every
student seed. ``limit`` (smoke and test runs only; refused for ``purpose="train"``) takes the
first k.

Per episode (one :class:`DaggerRecorder`, the runner's actor; :func:`ila.eval.runner.fly_chunk`
does the reset, start check, render-after-step with the committed look and the timing
assertion, exactly as for demos):

* the teacher ``T_ppo`` (P8 ``ppo`` seed 0, SB3 label path with frozen ``VecNormalize``, built as
  :func:`ila.data.collect.collect_chunk` builds it) is ``reset(episode_seed)``; the student is
  ``reset(episode_seed)`` and its :class:`ila.eval.runner.StudentWindow` cleared;
* at **every** control step t, in order: ``label[t] = teacher.act(obs[t])`` (shadowing the
  trajectory actually flown, whichever action is executed; float32-checked by
  :func:`ila.data.collect.check_float32_action`); the student's proposal
  ``act_vision(window.push(t, obs[t], frame[t]))`` (frames plus the allowlisted proprio, built by
  the contract; it never sees the 25-d observation); one coin ``u[t] ~ U[0, 1)``;
* the executed action is ``executable(label[t] if u[t] < beta else proposal[t])`` (clipped,
  rounded to float32 **before** ``env.step``; :func:`ila.data.collect.executable`), so the
  stored ``action_exec[t]`` is bit-exactly what ran and ``last_action`` in ``obs[t+1]`` equals it.

The beta coin. A per-episode ``numpy.random.default_rng(SeedSequence(entropy))`` with
``entropy = [run_seed, H("dagger_beta"), iteration, episode_seed]`` (:func:`coin_entropy`;
``H`` is the SHA-256 label entropy of :mod:`ila.data.episodes`, never ``hash()``), the DART
generator's style (:mod:`ila.data.dart`). Exactly one ``rng.random()`` is drawn per control step,
in order, **whatever beta is** (also at beta = 0, where ``u < 0`` never holds, and at beta = 1),
so the coin sequence of an episode depends only on (run_seed, iteration, episode seed) and the
step count. It does not depend on beta, the student or the worker count.

Student. ``student_ckpt`` is a full training checkpoint; its EMA weights fly on the CPU fp32
single-sample path (:func:`ila.train.select.checkpoint_policy`, the selection path, one torch
thread per worker). Rows and stored bytes do not depend on ``workers`` or ``chunk``.

Outputs, under :func:`rollout_dir` ``= <data_root>/dagger/<arm>/s<run_seed>/``:

* ``iter<i>.h5``: an ``ila-demos/1`` store (schema unchanged, :data:`ila.data.store.STORE_KEYS`),
  group ``ep_<global index>``; attributes ``beta`` = the iteration's beta,
  ``dagger_iteration`` = i, ``student_ckpt_sha256``, ``teacher`` ``T_ppo``, ``episode_teacher``
  ``ppo``. Rollouts are kept whatever their outcome (never success-filtered).
* ``iter<i>.sidecar.npz``: per step, the student's proposal (float32 ``(F, 3)``), the coin ``u``
  (float64) and whether the teacher's label was executed; per-episode offsets, global indices
  and episode seeds. It lets teacher agreement be measured on the student's own proposals.
* ``iter<i>.episodes.csv``: the runner rows (P8 columns; on-``train`` rollout outcomes).
* ``iter<i>.manifest.json``, written last: everything needed to identify and re-check the
  iteration (:func:`run_iteration`). The store and sidecar are written to ``*.tmp`` and renamed.
* ``iter<i>.audit.json``: :func:`audit_iteration`.
* ``aggregate_iter<i>.json``: :func:`aggregate_manifest`.

Units: actions normalised (x v_max = 1.5 m/s model scale, norm-capped by the env); times s model
scale (lambda = 1/25: 1 model s = 5 full-scale s); one control step is 1/30 s model scale.
"""

from __future__ import annotations

import functools
import io
import json
import multiprocessing as mp
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
import yaml
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import audit as data_audit
from ila.data import collect
from ila.data import episodes as data_episodes
from ila.data.store import (
    DemoStore,
    DemoWriter,
    EpisodeRecord,
    clear_fraction,
    contract_field_attrs,
)
from ila.eval import runner
from ila.obs.split import ObsContract, load_contract
from ila.policies.registry import read_checkpoint
from ila.render.camera import Frame
from ila.render.config import load_render_config
from ila.train.select import OUTCOME_ORDER, checkpoint_policy

__all__ = [
    "ARM",
    "BETA_LABEL",
    "PURPOSES",
    "TEACHER",
    "TEACHER_POLICY",
    "BetaCoin",
    "DaggerRecorder",
    "DaggerTask",
    "RolloutExistsError",
    "RolloutResult",
    "aggregate_manifest",
    "aggregate_path",
    "audit_iteration",
    "beta_for",
    "coin_entropy",
    "collection_name",
    "dagger_chunk",
    "load_rollout",
    "manifest_path",
    "rollout_dir",
    "rollout_store_path",
    "run_iteration",
    "sidecar_path",
    "write_npz",
]

#: The arm whose rollouts these are.
ARM: Final = "bc_dagger"

#: The label mixed into every beta-coin generator's entropy.
BETA_LABEL: Final[str] = "dagger_beta"

#: The collection teacher ID and the P8 policy that labels every visited state (P3-D1 §3).
TEACHER: Final[str] = "T_ppo"
TEACHER_POLICY: Final[str] = "ppo"

#: ``purpose`` values. ``limit`` is refused for ``train``.
PURPOSES: Final[tuple[str, ...]] = ("train", "smoke", "test")

#: The arm config holding ``labelled_episodes_total`` (frozen, P3-D1 §3).
ARM_CONFIG: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "train" / "bc_dagger.yaml"

#: Fixed zip member timestamp, so the sidecar's bytes depend on its arrays only.
_ZIP_EPOCH: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)


class RolloutExistsError(RuntimeError):
    """An iteration's outputs exist and do not match the request; nothing is overwritten."""


# --------------------------------------------------------------------------- config and paths


@functools.lru_cache(maxsize=4)
def _dagger_cfg(path: Path = data_episodes.DEFAULT_CONFIG) -> tuple[tuple[float, ...], int, int]:
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = raw["dagger"]
    betas = tuple(float(b) for b in d["beta"])
    iterations = int(d["iterations"])
    if len(betas) != iterations or not all(0.0 <= b <= 1.0 for b in betas):
        raise ValueError(f"dagger.beta {betas} must hold {iterations} values in [0, 1]")
    return betas, iterations, int(d["episodes_per_iteration"])


def beta_for(iteration: int) -> float:
    """Return iteration ``iteration``'s beta (``configs/data/demos.yaml`` ``dagger.beta``).

    Args:
        iteration: 1 .. 5.

    Returns:
        Probability per control step that the teacher's label is executed:
        0.5, 0.25, 0.125, 0.0, 0.0.

    Raises:
        ValueError: Outside ``1 .. dagger.iterations``.
    """
    betas, iterations, _ = _dagger_cfg()
    if not 1 <= int(iteration) <= iterations:
        raise ValueError(f"no DAgger iteration {iteration}; expected 1 .. {iterations}")
    return betas[int(iteration) - 1]


def rollout_dir(data_root: Path, run_seed: int, arm: str = ARM) -> Path:
    """Return ``<data_root>/dagger/<arm>/s<run_seed>/``."""
    return Path(data_root) / "dagger" / arm / f"s{int(run_seed)}"


def rollout_store_path(data_root: Path, run_seed: int, iteration: int, arm: str = ARM) -> Path:
    """Return ``<rollout_dir>/iter<iteration>.h5``."""
    return rollout_dir(data_root, run_seed, arm) / f"iter{int(iteration)}.h5"


def manifest_path(data_root: Path, run_seed: int, iteration: int, arm: str = ARM) -> Path:
    """Return ``<rollout_dir>/iter<iteration>.manifest.json``."""
    return rollout_dir(data_root, run_seed, arm) / f"iter{int(iteration)}.manifest.json"


def sidecar_path(data_root: Path, run_seed: int, iteration: int, arm: str = ARM) -> Path:
    """Return ``<rollout_dir>/iter<iteration>.sidecar.npz``."""
    return rollout_dir(data_root, run_seed, arm) / f"iter{int(iteration)}.sidecar.npz"


def aggregate_path(data_root: Path, run_seed: int, iteration: int, arm: str = ARM) -> Path:
    """Return ``<rollout_dir>/aggregate_iter<iteration>.json``."""
    return rollout_dir(data_root, run_seed, arm) / f"aggregate_iter{int(iteration)}.json"


def _rows_path(data_root: Path, run_seed: int, iteration: int, arm: str) -> Path:
    return rollout_dir(data_root, run_seed, arm) / f"iter{int(iteration)}.episodes.csv"


def _audit_path(data_root: Path, run_seed: int, iteration: int, arm: str) -> Path:
    return rollout_dir(data_root, run_seed, arm) / f"iter{int(iteration)}.audit.json"


def collection_name(run_seed: int, iteration: int, arm: str = ARM) -> str:
    """Return the store's ``collection`` attribute, e.g. ``bc_dagger_s0_iter1``."""
    return f"{arm}_s{int(run_seed)}_iter{int(iteration)}"


def _collection(run_seed: int, iteration: int, arm: str, n_planned: int) -> collect.Collection:
    design = data_episodes.load_stream_design()
    seed = int(yaml.safe_load(data_episodes.DEFAULT_CONFIG.read_text("utf-8"))["generator_seed"])
    return collect.Collection(
        name=collection_name(run_seed, iteration, arm),
        teacher=TEACHER,
        n=n_planned,
        split=design.dagger_split,
        stream=design.demo_stream,
        generator_seed=seed,
    )


def _iteration_rows(iteration: int) -> tuple[str, list[tuple[int, p8_adapter.ListedEpisode]]]:
    """Return the split stem and the ``(global index, row)`` pairs iteration ``i`` flies."""
    design = data_episodes.load_stream_design()
    rows = data_episodes.read_split(design.dagger_split)
    indices = design.dagger_indices(iteration)
    if indices.stop > len(rows):
        raise ValueError(f"{design.dagger_split} has {len(rows)} rows < {indices.stop}")
    out = [(j, rows[j]) for j in indices]
    for j, r in out:  # cheap guard; audit_iteration checks pools and frozen lists in full
        if r.ss in data_episodes.FORBIDDEN_SEA_STATES or any(
            abs(float(r.heading_deg) - h) < 1e-9 for h in data_episodes.FORBIDDEN_HEADINGS_DEG
        ):
            raise ValueError(f"demos[{j}] is {r.ss} / {r.heading_deg} deg: never trained on")
    return design.dagger_split, out


# --------------------------------------------------------------------------- the beta coin


def coin_entropy(run_seed: int, iteration: int, episode_seed: int) -> list[int]:
    """Return the entropy words of one episode's beta-coin generator.

    Returns:
        ``[run_seed, H("dagger_beta"), iteration, episode_seed]``.
    """
    return [
        int(run_seed),
        data_episodes.label_entropy(BETA_LABEL),
        int(iteration),
        int(episode_seed),
    ]


class BetaCoin:
    """One episode's coin: call :meth:`draw` once per control step, in order.

    Exactly one ``U[0, 1)`` draw per step whatever ``beta`` is (module docstring).

    Attributes:
        beta: Probability per step that the teacher's label is executed.
        steps: Draws taken so far.
    """

    def __init__(self, beta: float, run_seed: int, iteration: int, episode_seed: int) -> None:
        """Seed the episode's generator.

        Raises:
            ValueError: If ``beta`` is not in ``[0, 1]``.
        """
        if not (np.isfinite(beta) and 0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = float(beta)
        self._rng = np.random.default_rng(
            np.random.SeedSequence(coin_entropy(run_seed, iteration, episode_seed))
        )
        self.steps = 0

    def draw(self) -> tuple[float, bool]:
        """Return ``(u, u < beta)`` for the next step (``True``: execute the teacher's label)."""
        u = float(self._rng.random())
        self.steps += 1
        return u, u < self.beta


# --------------------------------------------------------------------------- the recorder


@dataclass
class _Side:
    """One episode's per-step sidecar columns."""

    proposed: list[NDArray[np.float32]] = field(default_factory=list)
    u: list[float] = field(default_factory=list)
    teacher_exec: list[bool] = field(default_factory=list)


class DaggerRecorder:
    """The runner actor of a DAgger iteration: fly the beta mixture, shadow the teacher.

    Attributes:
        collection: The iteration's collection record (name, stream, split, ``T_ppo``).
        teacher: The labelling teacher (reset per episode, called every step).
        student: The vision student (``reset`` / ``act_vision``).
        window: The student's frame/proprio window (contract-built inputs only).
        run_seed: The DAgger seed.
        iteration: 1 .. 5.
        beta: The iteration's beta.
        student_sha256: The student checkpoint's SHA-256 (episode attribute).
        finished: Completed ``(buffer, sidecar)`` pairs, in flight order.
        needs_frames: True.
        privileged: False.
        needs_motion_feed: False.
    """

    needs_frames: bool = True
    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(
        self,
        collection: collect.Collection,
        teacher: p8_adapter.EvalPolicy,
        student: Any,
        contract: ObsContract,
        demo_index: Mapping[tuple[str, int, int], int],
        *,
        run_seed: int,
        iteration: int,
        beta: float,
        student_sha256: str,
    ) -> None:
        """Hold the policies, the window and the episode -> global index map.

        Args:
            collection: The iteration's collection record.
            teacher: P8 ``ppo`` (or a test double with ``reset``/``act``).
            student: A student with ``reset`` and ``act_vision``.
            contract: The observation contract the student window builds inputs with.
            demo_index: ``(ss, cell index, episode seed)`` -> global demo index.
            run_seed: The DAgger seed.
            iteration: 1 .. 5.
            beta: The iteration's beta.
            student_sha256: The student checkpoint's SHA-256.

        Raises:
            TypeError: If the student has no ``act_vision``.
        """
        if not runner.is_student(student):
            raise TypeError("the DAgger student must have act_vision")
        self.collection = collection
        self.teacher = teacher
        self.student = student
        self.window = runner.StudentWindow(contract)
        self._index = dict(demo_index)
        self.run_seed = int(run_seed)
        self.iteration = int(iteration)
        self.beta = float(beta)
        self.student_sha256 = str(student_sha256)
        self._buf: collect._Buffer | None = None
        self._side: _Side | None = None
        self._coin: BetaCoin | None = None
        self.finished: list[tuple[collect._Buffer, _Side]] = []

    def _close_episode(self) -> None:
        if self._buf is not None and self._side is not None:
            self.finished.append((self._buf, self._side))
        self._buf = self._side = None

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Start an episode: reset teacher, student and window; seed the coin.

        Raises:
            ValueError: If a context is passed (nothing here is privileged).
        """
        if context is not None:
            raise ValueError("DAgger teacher and student are non-privileged; no context")
        self._close_episode()
        seed = int(listed.episode_seed)
        j = self._index[(listed.ss, int(listed.index), seed)]
        self.teacher.reset(seed, None)
        self.window.clear()
        self.student.reset(seed, None)
        self._coin = BetaCoin(self.beta, self.run_seed, self.iteration, seed)
        self._buf = collect._Buffer(listed, j, TEACHER_POLICY, -1)
        self._side = _Side()

    def act(self, step: int, obs: NDArray[Any], frame: Frame | None) -> NDArray[np.float64]:
        """Label ``obs[step]``, ask the student, toss the coin, record, return the executed action.

        Raises:
            RuntimeError: Before :meth:`begin`, or if steps are skipped.
            ValueError: Without a frame, or on a non-float32 teacher action.
        """
        buf, side, coin = self._buf, self._side, self._coin
        if buf is None or side is None or coin is None:
            raise RuntimeError("act() before begin()")
        if frame is None:
            raise ValueError("DAgger needs frames: render mode must be 'full'")
        if step != len(buf.obs):
            raise RuntimeError(f"step {step} after {len(buf.obs)} recorded steps")
        # The teacher shadows every visited state, in order, whichever action runs.
        label = collect.check_float32_action(np.asarray(self.teacher.act(obs), dtype=np.float64))
        si = self.window.push(step, obs, frame)
        proposed = np.asarray(self.student.act_vision(si.images, si.proprio), dtype=np.float32)
        if proposed.shape != (3,) or not np.all(np.isfinite(proposed)):
            raise ValueError(f"student proposed {proposed!r}")
        u, use_teacher = coin.draw()
        executed = collect.executable(label if use_teacher else proposed)
        buf.images.append(np.array(frame.rgb, dtype=np.uint8, copy=True))
        buf.obs.append(np.array(obs, dtype=np.float32, copy=True))
        buf.label.append(label.astype(np.float32))
        buf.executed.append(executed.astype(np.float32))
        buf.clear.append(clear_fraction(frame.rgb))
        side.proposed.append(proposed)
        side.u.append(u)
        side.teacher_exec.append(use_teacher)
        return executed

    def finish(
        self, rows: Sequence[Mapping[str, Any]], prov: Mapping[str, Any]
    ) -> tuple[list[EpisodeRecord], list[dict[str, NDArray[Any]]]]:
        """Pair the recorded episodes with the runner's rows.

        Returns:
            ``(records, sidecars)``, one each per episode in flight order; a sidecar holds
            ``proposed`` float32 ``(T, 3)``, ``u`` float64 ``(T,)``, ``teacher_exec`` bool
            ``(T,)``.

        Raises:
            RuntimeError: If episodes and rows do not pair one to one.
        """
        self._close_episode()
        if len(rows) != len(self.finished):
            raise RuntimeError(f"{len(rows)} rows for {len(self.finished)} recorded episodes")
        records: list[EpisodeRecord] = []
        sides: list[dict[str, NDArray[Any]]] = []
        for (buf, side), row in zip(self.finished, rows, strict=True):
            records.append(
                collect._record(
                    self.collection,
                    buf,
                    row,
                    prov,
                    beta=self.beta,
                    dagger_iteration=self.iteration,
                    student_ckpt_sha256=self.student_sha256,
                )
            )
            sides.append(
                {
                    "proposed": np.stack(side.proposed).astype(np.float32),
                    "u": np.asarray(side.u, dtype=np.float64),
                    "teacher_exec": np.asarray(side.teacher_exec, dtype=bool),
                }
            )
        self.finished = []
        return records, sides


# --------------------------------------------------------------------------- one chunk


@dataclass(frozen=True)
class DaggerTask:
    """One picklable unit of rollout work.

    Attributes:
        collection: The iteration's collection record.
        episodes: ``(global demo index, listed episode)`` pairs, in stream order.
        cfgs: P8's configs.
        prov: :func:`ila.data.collect.provenance`.
        run_seed: The DAgger seed.
        iteration: 1 .. 5.
        beta: The iteration's beta.
        student_ckpt: The checkpoint file (EMA weights are flown).
        student_sha256: Its SHA-256.
        arm: Arm label for the rows.
    """

    collection: collect.Collection
    episodes: tuple[tuple[int, p8_adapter.ListedEpisode], ...]
    cfgs: p8_adapter.EvalConfigs
    prov: Mapping[str, Any]
    run_seed: int
    iteration: int
    beta: float
    student_ckpt: str
    student_sha256: str
    arm: str = ARM


ChunkResult = tuple[list[EpisodeRecord], list[dict[str, NDArray[Any]]], list[dict[str, Any]]]


def dagger_chunk(task: DaggerTask) -> ChunkResult:
    """Fly and record one chunk through the runner (picklable worker entry point).

    Builds the teacher once (``p8_adapter.teacher("ppo", 0)``) and the student once
    (:func:`ila.train.select.checkpoint_policy`: EMA, CPU, fp32) per chunk; both are reset on
    every episode.

    Returns:
        ``(records, sidecars, runner rows)``, each in stream order.
    """
    collect.check_render_config(load_render_config(appearance_shift=None))
    cfgs = task.cfgs
    teacher = p8_adapter.teacher(TEACHER_POLICY, 0, cfgs)
    student = checkpoint_policy(task.student_ckpt, cfgs)
    index = {(e.ss, int(e.index), int(e.episode_seed)): j for j, e in task.episodes}
    method = f"dagger:{task.arm}"
    vtask = runner.VisionTask(
        episodes=[e for _, e in task.episodes],
        # Carries the non-privileged flags only; its build is never called.
        spec=p8_adapter.callable_spec(
            method,
            functools.partial(p8_adapter.teacher, TEACHER_POLICY, 0),
            run_seed=task.run_seed,
        ),
        cfgs=cfgs,
        tags=runner.RowTags(
            arm=method,
            teacher=TEACHER,
            student_seed=task.run_seed,
            ckpt_sha256=task.student_sha256,
        ),
        render="full",
        appearance_shift=None,
    )
    holder: list[DaggerRecorder] = []

    def make(c: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[runner.Actor, object]:
        del c
        rec = DaggerRecorder(
            task.collection,
            teacher,
            student,
            contract,
            index,
            run_seed=task.run_seed,
            iteration=task.iteration,
            beta=task.beta,
            student_sha256=task.student_sha256,
        )
        holder.append(rec)
        return rec, rec

    rows = runner.fly_chunk(vtask, make)
    if not holder:
        return [], [], rows
    records, sides = holder[0].finish(rows, task.prov)
    return records, sides, rows


# --------------------------------------------------------------------------- files


def write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    """Write an uncompressed ``.npz`` whose bytes depend on the arrays only.

    ``numpy.savez`` stamps each member with the current time; this fixes the zip timestamps,
    so equal arrays give equal file SHA-256s. ``numpy.load`` reads it.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH), buf.getvalue())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Insertion order (not sorted): outcome counts stay in P8's order.
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _agreement(
    proposed: NDArray[np.float32], label: NDArray[np.float32], teacher_exec: NDArray[np.bool_]
) -> dict[str, Any]:
    """Teacher agreement on the student's own states: ``||proposal - label||_2`` per frame.

    Normalised action units (x 1.5 m/s model scale). Descriptive only; the Phase 5 report
    and the eval-auditor decide which statistic is scored.
    """
    d = np.linalg.norm(proposed.astype(np.float64) - label.astype(np.float64), axis=1)

    def block(x: NDArray[np.float64]) -> dict[str, Any]:
        if not x.size:
            return {"frames": 0}
        return {
            "frames": int(x.size),
            "mean_l2": float(x.mean()),
            "median_l2": float(np.median(x)),
            "p90_l2": float(np.quantile(x, 0.9)),
            "max_l2": float(x.max()),
            "mse_per_component": float(np.mean(x**2) / 3.0),
        }

    return {
        "metric": "l2(student proposal - teacher label), normalised action units",
        "all": block(d),
        "student_executed": block(d[~teacher_exec]),
        "teacher_executed": block(d[teacher_exec]),
    }


# --------------------------------------------------------------------------- the result


@dataclass(frozen=True)
class RolloutResult:
    """One finished DAgger iteration of one seed (from its manifest).

    Attributes:
        iteration: 1 .. 5.
        beta: The iteration's beta.
        store: ``iter<i>.h5``.
        manifest: ``iter<i>.manifest.json``.
        sidecar: ``iter<i>.sidecar.npz``.
        store_sha256: The store's SHA-256.
        student_ckpt: The checkpoint flown.
        student_ckpt_sha256: Its SHA-256.
        labelled_episodes: Episodes flown and labelled (200, or ``limit``).
        labelled_frames: Control steps labelled (30 Hz).
        outcomes: P8 outcome class -> count, P8's order, zeros included.
        reused: Read back from an existing iteration rather than flown now.
        run_seed: The DAgger seed.
        arm: The arm.
        purpose: ``train``, ``smoke`` or ``test``.
        limit: The episode limit, or ``None`` for a full iteration.
        teacher_agreement: :func:`_agreement` over the iteration.
    """

    iteration: int
    beta: float
    store: Path
    manifest: Path
    sidecar: Path
    store_sha256: str
    student_ckpt: Path
    student_ckpt_sha256: str
    labelled_episodes: int
    labelled_frames: int
    outcomes: dict[str, int]
    reused: bool
    run_seed: int = -1
    arm: str = ARM
    purpose: str = "train"
    limit: int | None = None
    teacher_agreement: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-ready record (paths as strings)."""
        return {
            "arm": self.arm,
            "run_seed": self.run_seed,
            "iteration": self.iteration,
            "beta": self.beta,
            "purpose": self.purpose,
            "limit": self.limit,
            "store": str(self.store),
            "store_sha256": self.store_sha256,
            "manifest": str(self.manifest),
            "sidecar": str(self.sidecar),
            "student_ckpt": str(self.student_ckpt),
            "student_ckpt_sha256": self.student_ckpt_sha256,
            "labelled_episodes": self.labelled_episodes,
            "labelled_frames": self.labelled_frames,
            "outcomes": dict(self.outcomes),
            "success": int(self.outcomes.get("success", 0)),
            "teacher_agreement": dict(self.teacher_agreement),
            "reused": self.reused,
        }


def _result(man: Mapping[str, Any], path: Path, *, reused: bool) -> RolloutResult:
    return RolloutResult(
        iteration=int(man["iteration"]),
        beta=float(man["beta"]),
        store=Path(man["store"]["path"]),
        manifest=path,
        sidecar=Path(man["sidecar"]["path"]),
        store_sha256=str(man["store"]["sha256"]),
        student_ckpt=Path(man["student"]["ckpt"]),
        student_ckpt_sha256=str(man["student"]["sha256"]),
        labelled_episodes=int(man["labelled_episodes"]),
        labelled_frames=int(man["labelled_frames"]),
        outcomes={k: int(man["outcomes"][k]) for k in OUTCOME_ORDER},
        reused=reused,
        run_seed=int(man["run_seed"]),
        arm=str(man["arm"]),
        purpose=str(man["purpose"]),
        limit=None if man["limit"] is None else int(man["limit"]),
        teacher_agreement=dict(man.get("teacher_agreement", {})),
    )


def load_rollout(
    data_root: Path, run_seed: int, iteration: int, *, arm: str = ARM, verify: bool = True
) -> RolloutResult:
    """Read back a finished iteration from its manifest.

    Args:
        data_root: ``${data_root}``.
        run_seed: The DAgger seed.
        iteration: 1 .. 5.
        arm: The arm.
        verify: Re-hash the store and the sidecar against the manifest.

    Returns:
        The :class:`RolloutResult`, ``reused=True``.

    Raises:
        FileNotFoundError: Without a manifest (the iteration never finished).
        RolloutExistsError: If the manifest is not complete, names another seed/iteration/arm,
            or a file's SHA-256 differs from it.
    """
    path = manifest_path(data_root, run_seed, iteration, arm)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no finished DAgger iteration")
    man = _read_json(path)
    ident = (man.get("arm"), man.get("run_seed"), man.get("iteration"))
    if ident != (arm, int(run_seed), int(iteration)) or man.get("complete") is not True:
        raise RolloutExistsError(f"{path}: manifest {ident}, complete={man.get('complete')}")
    if verify:
        for key in ("store", "sidecar"):
            f = Path(man[key]["path"])
            if not f.is_file() or p8_adapter.sha256_file(f) != man[key]["sha256"]:
                raise RolloutExistsError(f"{f}: missing or SHA-256 differs from {path.name}")
    return _result(man, path, reused=True)


def _existing(
    data_root: Path,
    run_seed: int,
    iteration: int,
    arm: str,
    student_sha: str,
    limit: int | None,
    purpose: str,
) -> RolloutResult | None:
    """Return a reusable finished iteration, ``None`` if nothing exists, or raise."""
    man_path = manifest_path(data_root, run_seed, iteration, arm)
    finals = [
        rollout_store_path(data_root, run_seed, iteration, arm),
        sidecar_path(data_root, run_seed, iteration, arm),
    ]
    if not man_path.is_file():
        if leftover := [str(f) for f in finals if f.exists()]:
            raise RolloutExistsError(
                f"{leftover} exist without a manifest (interrupted after the rename); "
                "remove them by hand after inspection -- nothing is overwritten"
            )
        return None
    man = _read_json(man_path)
    if man["student"]["sha256"] != student_sha:
        raise RolloutExistsError(
            f"{man_path}: iteration {iteration} was flown by student "
            f"{man['student']['sha256'][:12]}..., not {student_sha[:12]}...; refusing to overwrite"
        )
    if (man["limit"], man["purpose"]) != (limit, purpose):
        raise RolloutExistsError(
            f"{man_path}: exists with limit={man['limit']} purpose={man['purpose']}, "
            f"asked limit={limit} purpose={purpose}"
        )
    return load_rollout(data_root, run_seed, iteration, arm=arm)


# --------------------------------------------------------------------------- one iteration


def run_iteration(
    *,
    run_seed: int,
    iteration: int,
    student_ckpt: Path,
    data_root: Path,
    workers: int = 24,
    limit: int | None = None,
    purpose: str = "train",
    arm: str = ARM,
    chunk: int = collect.DEFAULT_COLLECT_CHUNK,
    on_progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
) -> RolloutResult:
    """Fly DAgger iteration ``iteration`` of seed ``run_seed`` and write its outputs.

    Idempotent: a finished iteration flown by the same checkpoint (same SHA-256, same limit
    and purpose) is returned with ``reused=True`` without flying; one flown by another
    checkpoint raises :class:`RolloutExistsError` (never overwritten).

    Args:
        run_seed: The DAgger (student) seed.
        iteration: 1 .. 5.
        student_ckpt: The full training checkpoint whose EMA weights fly (iteration 1: the
            ``bc`` run's ``s0_step030000.pt``; later: the last step of iteration i - 1).
        data_root: ``${data_root}``.
        workers: Processes (rows and bytes do not depend on it).
        limit: Smoke/test only: fly the first ``limit`` episodes of the iteration.
        purpose: ``train``, ``smoke`` or ``test``.
        arm: The arm (directory and row label).
        chunk: Episodes per worker task (rows and bytes do not depend on it).
        on_progress: ``(episodes done, total, seconds elapsed)`` after each chunk.
        log: Line logger.

    Returns:
        The :class:`RolloutResult`.

    Raises:
        ValueError: On an unknown purpose, a ``limit`` with ``purpose="train"`` or out of
            range, a bad worker/chunk count, or a checkpoint of another seed or contract.
        RolloutExistsError: See above.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    if limit is not None and purpose == "train":
        raise ValueError("limit is for smoke and test runs only; a train iteration flies all 200")
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    beta = beta_for(iteration)
    _, _, per_iteration = _dagger_cfg()
    n = per_iteration if limit is None else int(limit)
    if not 1 <= n <= per_iteration:
        raise ValueError(f"limit must be in [1, {per_iteration}], got {limit}")
    ckpt = Path(student_ckpt).resolve()
    student_sha = p8_adapter.sha256_file(ckpt)
    found = _existing(data_root, run_seed, iteration, arm, student_sha, limit, purpose)
    if found is not None:
        log(f"{arm} s{run_seed} iter{iteration}: reusing {found.store} (same student SHA-256)")
        return found

    meta = read_checkpoint(ckpt)
    if int(meta["run_seed"]) != int(run_seed):
        raise ValueError(f"{ckpt}: run_seed {meta['run_seed']} != DAgger seed {run_seed}")
    if str(meta["obs_variant"]) != "default" or str(meta["kind"]) != "bc":
        raise ValueError(f"{ckpt}: DAgger flies default-contract BC students only")
    if meta.get("ema") is None:
        raise ValueError(f"{ckpt}: no EMA weights")
    student_info = {
        "ckpt": str(ckpt),
        "sha256": student_sha,
        "arm": str(meta["arm"]),
        "run_seed": int(meta["run_seed"]),
        "stage": int(meta["stage"]),
        "stage_step": int(meta["stage_step"]),
        "global_step": int(meta["global_step"]),
        "purpose": str(meta["purpose"]),
        "config_sha256": str(meta["config_sha256"]),
        "weights": "ema",
        "inference_path": "cpu_fp32_single_sample",
    }
    del meta

    split, pairs = _iteration_rows(iteration)
    pairs = pairs[:n]
    collection = _collection(run_seed, iteration, arm, per_iteration)
    cfgs = p8_adapter.load_eval_configs()
    prov = collect.provenance()
    tasks = [
        DaggerTask(
            collection=collection,
            episodes=tuple(pairs[i : i + chunk]),
            cfgs=cfgs,
            prov=prov,
            run_seed=run_seed,
            iteration=iteration,
            beta=beta,
            student_ckpt=str(ckpt),
            student_sha256=student_sha,
            arm=arm,
        )
        for i in range(0, n, chunk)
    ]
    split_record = next(
        r
        for r in data_episodes.read_manifest(data_episodes.SPLITS_DIR / data_episodes.MANIFEST_NAME)
        if r["file"] == f"{split}.parquet"
    )
    file_attrs = {
        "collection": collection.name,
        "teacher": TEACHER,
        "dart_sigma": 0.0,
        "beta": beta,
        "split_file": f"{split}.parquet",
        "split_file_sha256": split_record["file_sha256"],
        "generator_seed": collection.generator_seed,
        "p8_commit": prov["p8_commit"],
        "ila_commit": prov["ila_commit"],
        "camera_cfg_sha256": prov["camera_cfg_sha256"],
        **contract_field_attrs(),
    }
    out_dir = rollout_dir(data_root, run_seed, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = rollout_store_path(data_root, run_seed, iteration, arm)
    side = sidecar_path(data_root, run_seed, iteration, arm)
    store_tmp = store.with_name(store.name + ".tmp")
    side_tmp = side.with_name(side.name + ".tmp")
    runner.set_single_thread_env()
    tally = collect._Tally()
    all_rows: list[dict[str, Any]] = []
    sides: list[dict[str, NDArray[Any]]] = []
    labels: list[NDArray[np.float32]] = []
    seeds: list[int] = []
    indices: list[int] = []
    start = time.perf_counter()
    log(
        f"{collection.name}: beta {beta}, {n} episodes, {len(tasks)} tasks, {workers} workers, "
        f"student {student_sha[:12]} -> {store}"
    )
    with DemoWriter(
        store_tmp,
        file_attrs,
        expected_camera_sha256=runner.render_cfg_sha256("full", None),
        overwrite=True,  # a stale *.tmp from an interrupted run; the final file is checked above
    ) as writer:

        def consume(result: ChunkResult) -> None:
            records, chunk_sides, chunk_rows = result
            for record, s in zip(records, chunk_sides, strict=True):
                writer.write(record)
                tally.add(record)
                sides.append(s)
                labels.append(np.asarray(record.arrays["label"]))
                seeds.append(record.meta.episode_seed)
                indices.append(record.meta.demo_index)
            all_rows.extend(chunk_rows)
            if on_progress is not None:
                on_progress(writer.n_episodes, n, time.perf_counter() - start)

        if workers == 1 or len(tasks) == 1:
            runner.init_worker_threads()
            for task in tasks:
                consume(dagger_chunk(task))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=min(workers, len(tasks)), initializer=runner.init_worker_threads
            ) as pool:
                for result in pool.imap(dagger_chunk, tasks, chunksize=1):
                    consume(result)
    wall_s = time.perf_counter() - start
    if indices != [j for j, _ in pairs]:
        raise RuntimeError("episodes were not recorded in stream order")
    if p8_adapter.sha256_file(ckpt) != student_sha:
        raise RuntimeError(f"{ckpt} changed while it was being flown")
    lengths = np.asarray([s["u"].shape[0] for s in sides], dtype=np.int64)
    sidecar_arrays: dict[str, NDArray[Any]] = {
        "run_seed": np.asarray(run_seed, dtype=np.int64),
        "iteration": np.asarray(iteration, dtype=np.int64),
        "beta": np.asarray(beta, dtype=np.float64),
        "demo_index": np.asarray(indices, dtype=np.int64),
        "episode_seed": np.asarray(seeds, dtype=np.int64),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "proposed": np.concatenate([s["proposed"] for s in sides]).astype(np.float32),
        "u": np.concatenate([s["u"] for s in sides]).astype(np.float64),
        "teacher_exec": np.concatenate([s["teacher_exec"] for s in sides]).astype(bool),
    }
    write_npz(side_tmp, sidecar_arrays)
    agreement = _agreement(
        sidecar_arrays["proposed"], np.concatenate(labels), sidecar_arrays["teacher_exec"]
    )
    store_tmp.replace(store)
    side_tmp.replace(side)
    rows_file = _rows_path(data_root, run_seed, iteration, arm)
    rows_file.write_text(runner.vision_rows_csv(all_rows), encoding="utf-8")
    summary = tally.summary()
    outcomes = {k: int(summary["outcomes"].get(k, 0)) for k in OUTCOME_ORDER}
    if sum(outcomes.values()) != n:
        raise RuntimeError(f"unknown outcome classes in {summary['outcomes']}")
    teacher_frames = int(sidecar_arrays["teacher_exec"].sum())
    ppo_model = p8_adapter.load_paths()["p8_ppo_seed0_run"] / "final" / "model.zip"
    manifest: dict[str, Any] = {
        "kind": "dagger_rollouts",
        "arm": arm,
        "run_seed": int(run_seed),
        "iteration": int(iteration),
        "beta": beta,
        "purpose": purpose,
        "limit": limit,
        "n_planned": per_iteration,
        "complete": True,
        "full_iteration": limit is None,
        "collection": collection.name,
        "episodes": {
            "stream": collection.stream,
            "split_file": f"splits/{split}.parquet",
            "split_file_sha256": split_record["file_sha256"],
            "split_content_sha256": split_record["content_sha256"],
            "first_index": pairs[0][0],
            "last_index": pairs[-1][0],
            "list": [
                {
                    "demo_index": j,
                    "ss": r.ss,
                    "index": int(r.index),
                    "episode_seed": int(r.episode_seed),
                }
                for j, r in pairs
            ],
        },
        "student": student_info,
        "teacher": {
            "id": TEACHER,
            "policy": TEACHER_POLICY,
            "label_path": "sb3",
            "ppo_model_sha256": p8_adapter.sha256_file(ppo_model),
            "shadowing": "reset per episode; act on every visited obs, in order",
        },
        "beta_coin": {
            "entropy": "[run_seed, H('dagger_beta'), iteration, episode_seed]",
            "rule": (
                "one u ~ U[0,1) per control step whatever beta; "
                "the teacher label is executed iff u < beta"
            ),
        },
        "executed_action": "ila.data.collect.executable (clip, float32 before env.step)",
        "outcomes": outcomes,
        "outcome_order": list(OUTCOME_ORDER),
        "success": outcomes["success"],
        "labelled_episodes": n,
        "labelled_frames": int(lengths.sum()),
        "teacher_executed_frames": teacher_frames,
        "student_executed_frames": int(lengths.sum()) - teacher_frames,
        "teacher_executed_fraction": teacher_frames / max(int(lengths.sum()), 1),
        "teacher_agreement": agreement,
        "by_ss": summary["by_ss"],
        "success_by_ss": summary["success_by_ss"],
        "by_heading_deg": summary["by_heading_deg"],
        "by_speed_kn": summary["by_speed_kn"],
        "steps_mean": summary["steps_mean"],
        "steps_max": summary["steps_max"],
        "clear_frac_max": summary["clear_frac_max"],
        "clear_frames": summary["clear_frames"],
        "success_filtered": False,
        "store": {
            "path": str(store),
            "sha256": p8_adapter.sha256_file(store),
            "bytes": store.stat().st_size,
        },
        "sidecar": {
            "path": str(side),
            "sha256": p8_adapter.sha256_file(side),
            "bytes": side.stat().st_size,
            "keys": sorted(sidecar_arrays),
        },
        "rows_csv": {"path": str(rows_file), "sha256": p8_adapter.sha256_file(rows_file)},
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        **prov,
        "render": "full",
        "appearance_shift": None,
        "workers": workers,
        "chunk": chunk,
        "wall_s": wall_s,
        "wall_s_per_episode": wall_s / n,
    }
    man_path = manifest_path(data_root, run_seed, iteration, arm)
    _write_json_atomic(man_path, manifest)  # last: its presence means the iteration finished
    log(
        f"{collection.name}: done in {wall_s:.1f} s, outcomes {outcomes}, "
        f"{manifest['labelled_frames']} frames ({teacher_frames} teacher-executed)"
    )
    return _result(manifest, man_path, reused=False)


# --------------------------------------------------------------------------- aggregation


def _labelled_total() -> int:
    raw: Mapping[str, Any] = yaml.safe_load(ARM_CONFIG.read_text(encoding="utf-8"))
    return int(raw["labelled_episodes_total"])


def aggregate_manifest(
    *,
    run_seed: int,
    iteration: int,
    data_root: Path,
    demos: Mapping[str, Any],
    rollouts: Sequence[RolloutResult],
    arm: str = ARM,
    strict: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Account for the aggregate iteration ``iteration`` trains on (demos + rollouts 1..i).

    Args:
        run_seed: The DAgger seed.
        iteration: 1 .. 5.
        data_root: ``${data_root}`` (rollout stores must live under :func:`rollout_dir`).
        demos: The demos' accounting (``ila.train.loop.DataBundle.summary()`` style:
            ``labelled_episodes`` required; ``train_frames``, ``kept_episodes``, ``stores``
            copied when present) for ``ppo_2000`` indices 0-999, success-filtered.
        rollouts: Iterations 1 .. ``iteration`` of this seed, in order.
        arm: The arm.
        strict: Require the frozen counts (1 000 demos, 200 per iteration, 2 000 after
            iteration 5). ``False`` only for smoke/test aggregates (``limit`` rollouts); the
            totals must still add up.
        write: Also write it to :func:`aggregate_path` (atomically).

    Returns:
        The JSON-ready aggregate: per-source episodes, frames, stores and SHA-256s, the beta and
        student SHA of every iteration, and ``labelled_episodes`` = 1 000 + 200 i.

    Raises:
        ValueError: On a wrong iteration sequence, beta, seed, store location or count.
    """
    betas, iterations, per_iteration = _dagger_cfg()
    if not 1 <= iteration <= iterations:
        raise ValueError(f"no DAgger iteration {iteration}")
    got = [r.iteration for r in rollouts]
    if got != list(range(1, iteration + 1)):
        raise ValueError(f"aggregate {iteration} needs rollouts 1..{iteration}, got {got}")
    base = rollout_dir(data_root, run_seed, arm)
    for r in rollouts:
        if r.beta != betas[r.iteration - 1]:
            raise ValueError(f"iteration {r.iteration}: beta {r.beta} != {betas[r.iteration - 1]}")
        if r.run_seed not in (-1, int(run_seed)) or r.arm != arm:
            raise ValueError(f"iteration {r.iteration}: seed {r.run_seed} / arm {r.arm}")
        if r.store.parent.resolve() != base.resolve():
            raise ValueError(f"iteration {r.iteration}: store {r.store} not under {base}")
        if strict and (r.labelled_episodes != per_iteration or r.purpose != "train"):
            raise ValueError(
                f"iteration {r.iteration}: {r.labelled_episodes} labelled episodes "
                f"({r.purpose}), expected {per_iteration} (train)"
            )
    design = data_episodes.load_stream_design()
    demo_n = int(demos["labelled_episodes"])
    if strict and demo_n != design.dagger_first_index:
        raise ValueError(f"demos: {demo_n} labelled episodes, expected {design.dagger_first_index}")
    roll_n = sum(r.labelled_episodes for r in rollouts)
    total = demo_n + roll_n
    expected = design.dagger_first_index + per_iteration * iteration
    if strict and total != expected:
        raise ValueError(f"aggregate {iteration}: {total} labelled episodes != {expected}")
    if strict and iteration == iterations and total != _labelled_total():
        raise ValueError(f"final aggregate: {total} != labelled_episodes_total {_labelled_total()}")
    demo_frames = demos.get("train_frames")
    roll_frames = sum(r.labelled_frames for r in rollouts)
    out: dict[str, Any] = {
        "kind": "dagger_aggregate",
        "arm": arm,
        "run_seed": int(run_seed),
        "iteration": int(iteration),
        "strict": strict,
        "demos": dict(demos),
        "rollouts": [r.summary() for r in rollouts],
        "betas": [r.beta for r in rollouts],
        "student_ckpt_sha256": [r.student_ckpt_sha256 for r in rollouts],
        "labelled_episodes": total,
        "labelled_episodes_demos": demo_n,
        "labelled_episodes_rollouts": roll_n,
        "labelled_episodes_expected": expected,
        "labelled_episodes_total_final": _labelled_total(),
        "labelled_frames": {
            "demos_kept": demo_frames,
            "rollouts": roll_frames,
            "total": None if demo_frames is None else int(demo_frames) + roll_frames,
        },
        "rollouts_success_filtered": False,
    }
    if write:
        base.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(aggregate_path(data_root, run_seed, iteration, arm), out)
    return out


# --------------------------------------------------------------------------- audit


def audit_iteration(
    data_root: Path,
    run_seed: int,
    iteration: int,
    n_steps: int = 1000,
    *,
    arm: str = ARM,
    n_replay: int = 20,
) -> dict[str, Any]:
    """Audit one finished iteration and write ``iter<i>.audit.json``.

    Checks (all must hold for ``clean``):

    * the store and sidecar SHA-256s equal the manifest's; the student checkpoint, if still
      on disk, hashes to the manifest's SHA;
    * re-labelling: :func:`ila.data.audit.relabel_audit` on ``n_steps`` random visited states
      (the teacher reset and shadowed in order from step 0): identical;
    * open-loop replay: :func:`ila.data.audit.replay_audit` re-flies the episodes of
      ``n_replay`` random stored steps executing the stored ``action_exec``; the re-rendered
      frames and observations are byte-identical and no episode diverges;
    * attributes: every realization in P8's ``train`` pool, none in ``tune``, the ``select``
      list or any frozen list; no SS6, no 90 deg; each episode equals its split row; the
      global indices are exactly the first ``n`` of ``dagger_indices(i)``; ``beta`` and
      ``dagger_iteration`` attributes (file and episodes) right; every episode's student SHA
      equals the manifest's; teacher ``T_ppo`` / ``ppo``;
    * every episode validates (``last_action[t] == action_exec[t-1]``, zero at t = 0);
    * the sidecar: the coin re-drawn from :class:`BetaCoin` is bit-identical; ``action_exec[t]``
      equals ``executable(label[t])`` where the teacher was executed and
      ``executable(proposed[t])`` elsewhere; per-episode lengths equal the store's;
    * the count equals 200 (or the manifest's ``limit``) and the manifest's frames.

    Returns:
        A JSON-ready dict with a boolean ``clean``.
    """
    man_path = manifest_path(data_root, run_seed, iteration, arm)
    man = _read_json(man_path)
    store = Path(man["store"]["path"])
    side = Path(man["sidecar"]["path"])
    hashes = {
        "store": p8_adapter.sha256_file(store) == man["store"]["sha256"],
        "sidecar": p8_adapter.sha256_file(side) == man["sidecar"]["sha256"],
    }
    ckpt = Path(man["student"]["ckpt"])
    ckpt_ok = p8_adapter.sha256_file(ckpt) == man["student"]["sha256"] if ckpt.is_file() else None
    beta = beta_for(iteration)
    _, _, per_iteration = _dagger_cfg()
    n_expected = per_iteration if man["limit"] is None else int(man["limit"])
    design = data_episodes.load_stream_design()
    want_indices = list(design.dagger_indices(iteration))[:n_expected]
    relabel = data_audit.relabel_audit(store, n_steps=n_steps)
    replay = data_audit.replay_audit(store, n_states=n_replay)

    with DemoStore(store) as s:
        names = s.episodes()
        metas = [s.meta(e) for e in names]
        file_attrs = dict(s.file_attrs)
        invalid: list[str] = []
        last_action_bad = 0
        coin_bad = exec_bad = length_bad = 0
        contract = load_contract()
        with np.load(side, allow_pickle=False) as f:
            npz = {key: f[key] for key in f.files}
        offsets = npz["offsets"]
        if len(npz["demo_index"]) != len(names) or offsets.shape != (len(names) + 1,):
            length_bad += abs(len(npz["demo_index"]) - len(names)) or 1
        for k, (e, m) in enumerate(zip(names, metas, strict=True)):
            arrays = s.audit_arrays(e)
            try:
                s.validate_episode(e)
            except ValueError as err:  # StoreSchemaError
                invalid.append(f"{e}: {err}")
            last = contract.audit_field(arrays["teacher_obs"], "last_action")
            last_action_bad += int(
                np.any(last[0] != 0.0) or not np.array_equal(last[1:], arrays["action_exec"][:-1])
            )
            if k + 1 >= offsets.shape[0] or k >= len(npz["demo_index"]):
                continue
            lo, hi = int(offsets[k]), int(offsets[k + 1])
            if hi - lo != m.steps or int(npz["demo_index"][k]) != m.demo_index:
                length_bad += 1
                continue
            coin = BetaCoin(beta, run_seed, iteration, m.episode_seed)
            redraw = np.asarray([coin.draw() for _ in range(m.steps)], dtype=np.float64)
            u, te = npz["u"][lo:hi], npz["teacher_exec"][lo:hi]
            coin_bad += int(
                redraw[:, 0].tobytes() != u.tobytes() or not np.array_equal(redraw[:, 1] > 0, te)
            )
            proposed = npz["proposed"][lo:hi]
            want = np.stack(
                [
                    collect.executable(arrays["label"][t] if te[t] else proposed[t])
                    for t in range(m.steps)
                ]
            ).astype(np.float32)
            exec_bad += int(want.tobytes() != arrays["action_exec"].tobytes())
    rows = data_episodes.read_split(design.dagger_split)
    checks, identity = data_audit.attr_checks(metas, rows)
    select = data_episodes.read_split(data_episodes.SELECT_NAME)
    select_eps = {(tuple(r.key), int(r.episode_seed)) for r in select}
    select_keys = {tuple(r.key) for r in select}
    checks.update(
        {
            "in_select_key": sum(
                tuple(json.loads(m.realization_key)) in select_keys for m in metas
            ),
            "in_select_episode": sum(
                (tuple(json.loads(m.realization_key)), m.episode_seed) in select_eps for m in metas
            ),
            "demo_indices_not_dagger_slice": int([m.demo_index for m in metas] != want_indices),
            "beta_attr_wrong": sum(m.beta != beta for m in metas)
            + int(float(file_attrs["beta"]) != beta),
            "dagger_iteration_attr_wrong": sum(m.dagger_iteration != iteration for m in metas),
            "student_sha_attr_wrong": sum(
                m.student_ckpt_sha256 != man["student"]["sha256"] for m in metas
            ),
            "teacher_attr_wrong": sum(
                (m.teacher, m.episode_teacher) != (TEACHER, TEACHER_POLICY) for m in metas
            ),
            "dart_sigma_nonzero": sum(m.dart_sigma != 0.0 for m in metas),
            "collection_attr_wrong": int(
                file_attrs["collection"] != collection_name(run_seed, iteration, arm)
            ),
        }
    )
    frames = sum(m.steps for m in metas)
    episodes_ok = len(metas) == n_expected == int(man["labelled_episodes"])
    frames_ok = frames == int(man["labelled_frames"]) == int(offsets[-1])
    count_ok = episodes_ok and frames_ok
    out: dict[str, Any] = {
        "arm": arm,
        "run_seed": int(run_seed),
        "iteration": int(iteration),
        "beta": beta,
        "manifest": str(man_path),
        "hashes_match_manifest": hashes,
        "student_ckpt_matches": ckpt_ok,
        "relabel": relabel,
        "replay": replay,
        "attributes": {"violations": checks, "identity_mismatches": identity[:20]},
        "validate_failures": invalid[:20],
        "n_validate_failures": len(invalid),
        "last_action_mismatch_episodes": last_action_bad,
        "sidecar": {
            "length_mismatch_episodes": length_bad,
            "coin_mismatch_episodes": coin_bad,
            "executed_action_mismatch_episodes": exec_bad,
        },
        "count": {"episodes": len(metas), "expected": n_expected, "frames": frames, "ok": count_ok},
        "outcomes": man["outcomes"],
    }
    out["clean"] = bool(
        all(hashes.values())
        and ckpt_ok is not False
        and relabel["passed"]
        and replay["passed"]
        and not any(checks.values())
        and not invalid
        and last_action_bad == 0
        and length_bad == coin_bad == exec_bad == 0
        and count_ok
    )
    _write_json_atomic(_audit_path(data_root, run_seed, iteration, arm), out)
    return out
