"""Teacher demo collection with paired images and observations (plan §1.5, Phase 3).

Flight path. Every episode is flown by :func:`ila.eval.runner.fly_chunk` (the Phase 2 runner,
unchanged), so collection inherits P8's start check on every reset, render-after-step with the
committed look (``full``), the image/obs ``stepSimulation`` timing assertion and the
frames == steps + 1 count. The runner calls a :class:`DemoRecorder` as its actor. At control
step t the recorder gets ``obs[t]`` (P8's 25-d observation) and ``Frame`` t (pixels only),
asks the episode's teacher for ``teacher.act(obs[t])`` (the label), and returns the executed
action: the teacher's action itself (P8's ``TeacherActor`` path, so trajectories equal P8's),
or ``clip(label + sigma z)`` under DART (:mod:`ila.data.dart`). It stores ``image[t]``,
``teacher_obs[t]``, ``label[t]``, ``action_exec[t] = clip(executed, -1, 1)`` (what the env
executes and what ``last_action`` in ``obs[t+1]`` reports) and ``t_model_s[t] = t0 + t/30``.

Teachers (plan §1.4). ``T_ppo``: P8 ``ppo`` seed 0 on SB3 with frozen ``VecNormalize`` (P2-D1).
``T_pid``: P8 ``pid_feedforward``. ``T_mix``: per episode, a seeded coin flip between ``ppo``
and P8 ``gated`` (:func:`mix_teacher`); the label is that episode's teacher's action. Every
teacher is ``reset`` at the start of every episode and called exactly once per step, in order,
along the trajectory actually flown, never on an isolated state.

Episodes come from a committed ``train`` split (``splits/demos_0-*.parquet``, checked against
``splits/MANIFEST.csv``); collection ``c`` flies its first ``n`` rows. Every episode is kept,
whatever its outcome; the outcome class is recorded. An appearance-shift render config is
refused before anything flies (P1-D2 obligation 2), and the writer refuses any frame whose
camera SHA is not the committed look's.

Units: actions normalised (x v_max = 1.5 m/s model scale, norm-capped by the env), times s
model scale (lambda = 1/25: 1 model s = 5 full-scale s), lengths m model scale.
"""

from __future__ import annotations

import functools
import json
import multiprocessing as mp
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
import yaml
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.data.dart import DartNoise
from ila.data.store import (
    CTRL_HZ,
    DemoWriter,
    EpisodeMeta,
    EpisodeRecord,
    clear_fraction,
    contract_field_attrs,
)
from ila.eval import runner
from ila.obs.split import ObsContract
from ila.render.camera import Frame
from ila.render.config import RenderConfig, load_render_config

__all__ = [
    "MIX_LABEL",
    "TEACHER_POLICIES",
    "AppearanceShiftRefusedError",
    "Collection",
    "CollectTask",
    "DemoRecorder",
    "check_render_config",
    "collect_chunk",
    "load_collections",
    "mix_teacher",
    "provenance",
    "run_collection",
]

#: Collection teacher ID -> the P8 policies it may fly (plan §1.4).
TEACHER_POLICIES: Final[Mapping[str, tuple[str, ...]]] = {
    "T_ppo": ("ppo",),
    "T_pid": ("pid_feedforward",),
    "T_mix": ("ppo", "gated"),
}

#: The label mixed into every ``T_mix`` coin's entropy.
MIX_LABEL: Final[str] = "T_mix"

#: Episodes per worker task (one env and one set of teachers per task). Rows and stored bytes
#: do not depend on it; smaller than the runner's 25 to bound the frames held per result.
DEFAULT_COLLECT_CHUNK: Final[int] = 10


class AppearanceShiftRefusedError(ValueError):
    """A demo writer was handed an evaluation-only appearance-shift render config."""


def check_render_config(cfg: RenderConfig) -> None:
    """Refuse any render config but the committed look (P1-D2 obligation 2).

    Raises:
        AppearanceShiftRefusedError: If ``cfg.appearance_shift`` is set.
    """
    if cfg.appearance_shift is not None:
        raise AppearanceShiftRefusedError(
            f"appearance shift {cfg.appearance_shift!r} is evaluation only (plan §1.3); "
            "no demo, DART or DAgger frame may be rendered with it"
        )


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Collection:
    """One demo collection (``configs/data/demos.yaml`` ``collections``).

    Attributes:
        name: ``ppo_2000``, ``pid_1000``, ``mix_1000``, ``dart_0.05``, ``dart_0.10`` or
            ``dart_0.20``.
        teacher: ``T_ppo``, ``T_pid`` or ``T_mix``.
        n: Episodes (a prefix of the demo stream).
        split: The split file stem the episodes come from (``demos_0-1999`` / ``demos_0-999``).
        stream: The demo stream label.
        generator_seed: The draw seed (20261023).
        dart_sigma: DART noise std (normalised units), or ``None``.
        dart_seed: DART generator seed (used only with ``dart_sigma``).
        mix_seed: ``T_mix`` coin seed (used only for ``T_mix``).
        mix_p_ppo: ``T_mix``'s probability of ``ppo`` (0.5).
    """

    name: str
    teacher: str
    n: int
    split: str
    stream: str
    generator_seed: int
    dart_sigma: float | None = None
    dart_seed: int = 0
    mix_seed: int = 0
    mix_p_ppo: float = 0.5

    @property
    def policies(self) -> tuple[str, ...]:
        """The P8 policies this collection may fly."""
        return TEACHER_POLICIES[self.teacher]


def load_collections(path: Path = data_episodes.DEFAULT_CONFIG) -> dict[str, Collection]:
    """Load every collection, the DART entry expanded per sigma (``dart_0.05`` ...).

    Args:
        path: ``configs/data/demos.yaml``.

    Returns:
        Name -> :class:`Collection`, in YAML order.

    Raises:
        ValueError: On an unknown teacher, a non-0.5/0.5 ``T_mix`` mixture of other
            policies, or a T_ppo label path other than SB3.
    """
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    design = data_episodes.load_stream_design(path)
    seed = int(raw["generator_seed"])
    teachers: Mapping[str, Any] = raw["teachers"]
    if teachers["T_ppo"]["label_path"] != "sb3":
        raise ValueError("P2-D1: T_ppo is flown and labelled on SB3")
    mixture = dict(teachers["T_mix"]["mixture"])
    if set(mixture) != {"T_ppo", "gated"}:
        raise ValueError(f"T_mix mixture must be T_ppo/gated, got {mixture}")
    common: dict[str, Any] = {
        "stream": design.demo_stream,
        "generator_seed": seed,
        "mix_seed": int(teachers["T_mix"]["seed"]),
        "mix_p_ppo": float(mixture["T_ppo"]),
    }
    out: dict[str, Collection] = {}
    for key, spec in raw["collections"].items():
        teacher = str(spec["teacher"])
        if teacher not in TEACHER_POLICIES:
            raise ValueError(f"{key}: unknown teacher {teacher!r}")
        n = int(spec["n"])
        split = data_episodes.train_name(n, stream=design.demo_stream)
        if "sigma_grid" in spec:
            for sigma in spec["sigma_grid"]:
                name = f"{key}_{float(sigma):.2f}"
                out[name] = Collection(
                    name, teacher, n, split, dart_sigma=float(sigma),
                    dart_seed=int(spec["seed"]), **common,
                )  # fmt: skip
        else:
            out[str(key)] = Collection(str(key), teacher, n, split, **common)
    return out


def mix_teacher(collection: Collection, demo_index: int, episode_seed: int) -> str:
    """Return the P8 policy one ``T_mix`` episode flies (``"ppo"`` or ``"gated"``).

    One uniform draw from ``default_rng(SeedSequence([mix_seed, H("T_mix"), demo_index,
    episode_seed]))``; ``ppo`` if it is below ``mix_p_ppo``. A pure function of the episode,
    so a re-flown episode gets the same teacher.

    Args:
        collection: A ``T_mix`` collection.
        demo_index: The episode's global index in the demo stream.
        episode_seed: The env's episode seed.

    Returns:
        The policy name.
    """
    entropy = [
        int(collection.mix_seed),
        data_episodes.label_entropy(MIX_LABEL),
        int(demo_index),
        int(episode_seed),
    ]
    u = float(np.random.default_rng(np.random.SeedSequence(entropy)).random())
    return "ppo" if u < collection.mix_p_ppo else "gated"


def episode_policy(collection: Collection, demo_index: int, episode_seed: int) -> str:
    """Return the P8 policy flown on one episode of a collection."""
    if collection.teacher == "T_mix":
        return mix_teacher(collection, demo_index, episode_seed)
    (only,) = collection.policies
    return only


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=p8_adapter.REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def provenance() -> dict[str, Any]:
    """Return the commits and the camera SHA every store and manifest records.

    Returns:
        ``p8_commit``, ``ila_commit`` (``+dirty`` appended when the tree has uncommitted
        changes), ``ila_dirty_paths`` and ``camera_cfg_sha256`` (the committed look).
    """
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln]
    head = _git("rev-parse", "HEAD")
    return {
        "p8_commit": p8_adapter.p8_commit(),
        "ila_commit": head + ("+dirty" if dirty else ""),
        "ila_dirty_paths": dirty[:50],
        "camera_cfg_sha256": runner.render_cfg_sha256("full", None),
    }


# --------------------------------------------------------------------------- the recorder


@dataclass
class _Buffer:
    listed: p8_adapter.ListedEpisode
    demo_index: int
    policy: str
    noise_seed: int
    images: list[NDArray[np.uint8]] = field(default_factory=list)
    obs: list[NDArray[np.float32]] = field(default_factory=list)
    label: list[NDArray[np.float32]] = field(default_factory=list)
    executed: list[NDArray[np.float32]] = field(default_factory=list)
    clear: list[float] = field(default_factory=list)


class DemoRecorder:
    """The runner actor of a collection: fly the teacher (maybe noised), record every step.

    It declares ``needs_frames`` so the runner insists on render ``full`` and a
    non-privileged spec; every teacher here is non-privileged and gets no context.

    Attributes:
        collection: The collection.
        teachers: P8 policy name -> built policy (one instance per chunk, reset per episode).
        finished: Completed episode buffers, in flight order.
        needs_frames: True.
        privileged: False (checked against the spec by the runner).
        needs_motion_feed: False.
    """

    needs_frames: bool = True
    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(
        self,
        collection: Collection,
        teachers: Mapping[str, p8_adapter.EvalPolicy],
        demo_index: Mapping[tuple[str, int, int], int],
    ) -> None:
        """Hold the teachers and the episode -> global index map.

        Args:
            collection: The collection.
            teachers: Built P8 policies, keyed by name; must cover ``collection.policies``.
            demo_index: ``(ss, cell index, episode seed)`` -> global demo index.
        """
        if missing := set(collection.policies) - set(teachers):
            raise ValueError(f"{collection.name}: teachers {sorted(missing)} not built")
        self.collection = collection
        self.teachers = dict(teachers)
        self._index = dict(demo_index)
        self._buf: _Buffer | None = None
        self._teacher: p8_adapter.EvalPolicy | None = None
        self._noise: DartNoise | None = None
        self.finished: list[_Buffer] = []

    def _close_episode(self) -> None:
        if self._buf is not None:
            self.finished.append(self._buf)
            self._buf = None

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Start an episode: pick and ``reset`` its teacher, seed its DART noise.

        Raises:
            ValueError: If a context is passed (no teacher here is privileged).
        """
        if context is not None:
            raise ValueError("demo teachers are non-privileged; no context")
        self._close_episode()
        j = self._index[(listed.ss, int(listed.index), int(listed.episode_seed))]
        policy = episode_policy(self.collection, j, listed.episode_seed)
        self._teacher = self.teachers[policy]
        self._teacher.reset(listed.episode_seed, None)
        sigma = self.collection.dart_sigma
        noise_seed = -1
        self._noise = None
        if sigma is not None:
            self._noise = DartNoise(sigma, self.collection.dart_seed, j, listed.episode_seed)
            noise_seed = self.collection.dart_seed
        self._buf = _Buffer(listed, j, policy, noise_seed)

    def act(self, step: int, obs: NDArray[Any], frame: Frame | None) -> NDArray[np.float64]:
        """Label ``obs[step]``, record the step, return the action ``env.step`` executes.

        Raises:
            RuntimeError: Before :meth:`begin`, or if steps are skipped.
            ValueError: Without a frame.
        """
        buf, teacher = self._buf, self._teacher
        if buf is None or teacher is None:
            raise RuntimeError("act() before begin()")
        if frame is None:
            raise ValueError("collection needs frames: render mode must be 'full'")
        if step != len(buf.obs):
            raise RuntimeError(f"step {step} after {len(buf.obs)} recorded steps")
        clean = np.asarray(teacher.act(obs), dtype=np.float64)
        executed = clean if self._noise is None else self._noise.executed(clean)
        buf.images.append(np.array(frame.rgb, dtype=np.uint8, copy=True))
        buf.obs.append(np.array(obs, dtype=np.float32, copy=True))
        buf.label.append(clean.astype(np.float32))
        buf.executed.append(np.clip(executed, -1.0, 1.0).astype(np.float32))
        buf.clear.append(clear_fraction(frame.rgb))
        return executed

    def finish(
        self, rows: Sequence[Mapping[str, Any]], prov: Mapping[str, Any]
    ) -> list[EpisodeRecord]:
        """Pair the recorded episodes with the runner's rows and build validated records.

        Args:
            rows: ``fly_chunk``'s rows, one per episode, in flight order.
            prov: :func:`provenance`.

        Returns:
            One :class:`EpisodeRecord` per episode, in flight order.

        Raises:
            RuntimeError: If episodes and rows do not pair one to one.
        """
        self._close_episode()
        if len(rows) != len(self.finished):
            raise RuntimeError(f"{len(rows)} rows for {len(self.finished)} recorded episodes")
        records: list[EpisodeRecord] = []
        for buf, row in zip(self.finished, rows, strict=True):
            records.append(_record(self.collection, buf, row, prov))
        self.finished = []
        return records


def _record(
    collection: Collection, buf: _Buffer, row: Mapping[str, Any], prov: Mapping[str, Any]
) -> EpisodeRecord:
    listed = buf.listed
    ident = (row["ss"], int(row["index"]), int(row["episode_seed"]))
    if ident != (listed.ss, int(listed.index), int(listed.episode_seed)):
        raise RuntimeError(f"row {ident} does not pair with recorded episode {listed}")
    t_len = len(buf.obs)
    if int(row["steps"]) != t_len or int(row["frames_rendered"]) != t_len + 1:
        raise RuntimeError(
            f"{ident}: {t_len} recorded steps, row steps {row['steps']}, "
            f"frames {row['frames_rendered']}"
        )
    key = p8_adapter.realization_key(
        listed.ss, listed.heading_deg, listed.speed_kn, listed.vessel, listed.realization_seed
    )
    clear = np.asarray(buf.clear, dtype=np.float32)
    meta = EpisodeMeta(
        collection=collection.name,
        stream=collection.stream,
        demo_index=buf.demo_index,
        realization_key=json.dumps(list(key)),
        regime=listed.regime,
        ss=listed.ss,
        index=int(listed.index),
        pad=listed.pad,
        vessel=listed.vessel,
        heading_deg=float(listed.heading_deg),
        speed_kn=float(listed.speed_kn),
        realization_seed=int(listed.realization_seed),
        episode_seed=int(listed.episode_seed),
        t0_model_s=float(listed.t0_model_s),
        teacher=collection.teacher,
        episode_teacher=buf.policy,
        beta=1.0,
        dart_sigma=0.0 if collection.dart_sigma is None else float(collection.dart_sigma),
        noise_seed=buf.noise_seed,
        dagger_iteration=0,
        student_ckpt_sha256="",
        outcome=str(row["outcome"]),
        termination_reason=str(row["termination_reason"]),
        success=str(row["outcome"]) == "success",
        steps=t_len,
        p8_commit=str(prov["p8_commit"]),
        ila_commit=str(prov["ila_commit"]),
        camera_cfg_sha256=str(row["render_cfg_sha256"]),
        clear_frac=clear,
        clear_frac_max=float(clear.max()),
        clear_frac_mean=float(clear.mean()),
        clear_frames=int(np.count_nonzero(clear)),
    )
    return EpisodeRecord.build(
        images=np.stack(buf.images),
        teacher_obs=np.stack(buf.obs),
        action_exec=np.stack(buf.executed),
        label=np.stack(buf.label),
        t_model_s=float(listed.t0_model_s) + np.arange(t_len, dtype=np.float64) / CTRL_HZ,
        meta=meta,
    )


# --------------------------------------------------------------------------- one chunk


@dataclass(frozen=True)
class CollectTask:
    """One picklable unit of collection work.

    Attributes:
        collection: The collection.
        episodes: ``(global demo index, listed episode)`` pairs, in stream order.
        cfgs: P8's configs.
        prov: :func:`provenance`.
        appearance_shift: Must be ``None`` (refused otherwise).
    """

    collection: Collection
    episodes: tuple[tuple[int, p8_adapter.ListedEpisode], ...]
    cfgs: p8_adapter.EvalConfigs
    prov: Mapping[str, Any]
    appearance_shift: str | None = None


def _spec(collection: Collection) -> p8_adapter.PolicySpec:
    """The runner spec of a collection (non-privileged; its ``build`` is never called)."""
    first = collection.policies[0]
    return p8_adapter.callable_spec(
        f"demo:{collection.name}", functools.partial(p8_adapter.teacher, first, 0), run_seed=0
    )


def collect_chunk(task: CollectTask) -> tuple[list[EpisodeRecord], list[dict[str, Any]]]:
    """Fly and record one chunk through the runner (picklable worker entry point).

    Args:
        task: The chunk.

    Returns:
        ``(records, runner rows)``, both in stream order.

    Raises:
        AppearanceShiftRefusedError: If the task asks for an appearance shift.
    """
    check_render_config(load_render_config(appearance_shift=task.appearance_shift))
    collection, cfgs = task.collection, task.cfgs
    teachers = {name: p8_adapter.teacher(name, 0, cfgs) for name in collection.policies}
    index = {(e.ss, int(e.index), int(e.episode_seed)): j for j, e in task.episodes}
    recorder = DemoRecorder(collection, teachers, index)
    vtask = runner.VisionTask(
        episodes=[e for _, e in task.episodes],
        spec=_spec(collection),
        cfgs=cfgs,
        tags=runner.RowTags(arm=f"demo:{collection.name}", teacher=collection.teacher),
        render="full",
        appearance_shift=None,
    )

    def make(c: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[runner.Actor, object]:
        del c, contract
        return recorder, recorder

    rows = runner.fly_chunk(vtask, make)
    return recorder.finish(rows, task.prov), rows


# --------------------------------------------------------------------------- a collection


@dataclass
class _Tally:
    outcomes: Counter[str] = field(default_factory=Counter)
    by_ss: Counter[str] = field(default_factory=Counter)
    by_heading: Counter[str] = field(default_factory=Counter)
    by_speed: Counter[str] = field(default_factory=Counter)
    success_by_ss: Counter[str] = field(default_factory=Counter)
    policies: Counter[str] = field(default_factory=Counter)
    success_by_policy: Counter[str] = field(default_factory=Counter)
    frames: int = 0
    kept_frames: int = 0
    clear_max: float = 0.0
    clear_frames: int = 0
    episodes_with_clear: int = 0
    steps: list[int] = field(default_factory=list)

    def add(self, r: EpisodeRecord) -> None:
        m = r.meta
        self.outcomes[m.outcome] += 1
        self.by_ss[m.ss] += 1
        self.by_heading[f"{m.heading_deg:g}"] += 1
        self.by_speed[f"{m.speed_kn:g}"] += 1
        self.policies[m.episode_teacher] += 1
        if m.success:
            self.success_by_ss[m.ss] += 1
            self.success_by_policy[m.episode_teacher] += 1
            self.kept_frames += m.steps
        self.frames += m.steps
        self.steps.append(m.steps)
        self.clear_max = max(self.clear_max, m.clear_frac_max)
        self.clear_frames += m.clear_frames
        self.episodes_with_clear += int(m.clear_frames > 0)

    def summary(self) -> dict[str, Any]:
        n = sum(self.outcomes.values())
        success = self.outcomes.get("success", 0)
        steps = np.asarray(self.steps, dtype=np.float64)
        return {
            "episodes": n,
            "frames": self.frames,
            "outcomes": dict(sorted(self.outcomes.items())),
            "success": success,
            "kept_fraction_success_filter": success / n if n else float("nan"),
            "kept_frames": self.kept_frames,
            "by_ss": dict(sorted(self.by_ss.items())),
            "success_by_ss": dict(sorted(self.success_by_ss.items())),
            "by_heading_deg": dict(sorted(self.by_heading.items())),
            "by_speed_kn": dict(sorted(self.by_speed.items())),
            "episode_teachers": dict(sorted(self.policies.items())),
            "success_by_episode_teacher": dict(sorted(self.success_by_policy.items())),
            "steps_mean": float(steps.mean()) if n else float("nan"),
            "steps_max": int(steps.max()) if n else 0,
            "clear_frac_max": self.clear_max,
            "clear_frames": self.clear_frames,
            "episodes_with_clear": self.episodes_with_clear,
        }


def _rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(runner.vision_rows_csv(rows), encoding="utf-8")


def run_collection(
    collection: Collection,
    out_path: Path,
    *,
    run_dir: Path,
    workers: int = p8_adapter.DEFAULT_WORKERS,
    chunk: int = DEFAULT_COLLECT_CHUNK,
    limit: int | None = None,
    appearance_shift: str | None = None,
    overwrite: bool = False,
    splits_dir: Path = data_episodes.SPLITS_DIR,
    on_progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly one collection into ``out_path`` and return its manifest.

    Workers fly and record chunks (:func:`collect_chunk`) in a spawn pool; this process
    writes each chunk's episodes in stream order, so the file does not depend on ``workers``
    or ``chunk``. The manifest is written to ``<out_path>.manifest.json`` and to
    ``run_dir/manifest.json``; the runner rows to ``run_dir/episodes.csv``.

    Args:
        collection: The collection.
        out_path: The ``.h5`` store to create.
        run_dir: ``artifacts/runs/collect/<name>/`` (rows, manifest).
        workers: Processes (P2-D2 budget: 24).
        chunk: Episodes per task.
        limit: Fly only the first ``limit`` episodes (pilots); recorded in the manifest.
        appearance_shift: Must be ``None``; anything else is refused before flying.
        overwrite: Replace an existing store.
        splits_dir: Where the split files and ``MANIFEST.csv`` live.
        on_progress: ``(episodes done, total, seconds elapsed)`` after each chunk.
        log: Line logger.

    Returns:
        The manifest dict.

    Raises:
        AppearanceShiftRefusedError: If ``appearance_shift`` is set.
        ValueError: If the split is shorter than the collection or ``limit`` is invalid.
    """
    check_render_config(load_render_config(appearance_shift=appearance_shift))
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    rows = data_episodes.read_split(collection.split, splits_dir)
    if len(rows) < collection.n:
        raise ValueError(f"{collection.split} has {len(rows)} rows < n = {collection.n}")
    n = collection.n if limit is None else int(limit)
    if not 1 <= n <= collection.n:
        raise ValueError(f"limit must be in [1, {collection.n}], got {limit}")
    indexed = tuple(enumerate(rows[:n]))
    cfgs = p8_adapter.load_eval_configs()
    prov = provenance()
    tasks = [
        CollectTask(collection, indexed[i : i + chunk], cfgs, prov)
        for i in range(0, len(indexed), chunk)
    ]
    split_record = next(
        r
        for r in data_episodes.read_manifest(splits_dir / data_episodes.MANIFEST_NAME)
        if r["file"] == f"{collection.split}.parquet"
    )
    file_attrs = {
        "collection": collection.name,
        "teacher": collection.teacher,
        "dart_sigma": 0.0 if collection.dart_sigma is None else collection.dart_sigma,
        "beta": 1.0,
        "split_file": f"{collection.split}.parquet",
        "split_file_sha256": split_record["file_sha256"],
        "generator_seed": collection.generator_seed,
        "p8_commit": prov["p8_commit"],
        "ila_commit": prov["ila_commit"],
        "camera_cfg_sha256": prov["camera_cfg_sha256"],
        **contract_field_attrs(),
    }
    runner.set_single_thread_env()
    run_dir.mkdir(parents=True, exist_ok=True)
    tally = _Tally()
    all_rows: list[dict[str, Any]] = []
    write_s = 0.0
    start = time.perf_counter()
    log(f"{collection.name}: {n} episodes, {len(tasks)} tasks, {workers} workers -> {out_path}")
    with DemoWriter(
        out_path,
        file_attrs,
        expected_camera_sha256=runner.render_cfg_sha256("full", None),
        overwrite=overwrite,
    ) as writer:

        def consume(result: tuple[list[EpisodeRecord], list[dict[str, Any]]]) -> None:
            nonlocal write_s
            records, chunk_rows = result
            t_write = time.perf_counter()
            for record in records:
                writer.write(record)
                tally.add(record)
            write_s += time.perf_counter() - t_write
            all_rows.extend(chunk_rows)
            if on_progress is not None:
                on_progress(writer.n_episodes, n, time.perf_counter() - start)

        if workers == 1 or len(tasks) == 1:
            runner.init_worker_threads()
            for task in tasks:
                consume(collect_chunk(task))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=min(workers, len(tasks)), initializer=runner.init_worker_threads
            ) as pool:
                for result in pool.imap(collect_chunk, tasks, chunksize=1):
                    consume(result)
    wall_s = time.perf_counter() - start
    _rows_csv(run_dir / "episodes.csv", all_rows)
    size = out_path.stat().st_size
    run_dir_ppo = p8_adapter.load_paths()["p8_ppo_seed0_run"] / "final" / "model.zip"
    manifest: dict[str, Any] = {
        "collection": collection.name,
        "teacher": collection.teacher,
        "policies": list(collection.policies),
        "ppo_model_sha256": (
            p8_adapter.sha256_file(run_dir_ppo) if "ppo" in collection.policies else None
        ),
        "label_path": "sb3" if "ppo" in collection.policies else "controller",
        "dart_sigma": collection.dart_sigma,
        "dart_seed": collection.dart_seed if collection.dart_sigma is not None else None,
        "mix_seed": collection.mix_seed if collection.teacher == "T_mix" else None,
        "mix_p_ppo": collection.mix_p_ppo if collection.teacher == "T_mix" else None,
        "beta": 1.0,
        "stream": collection.stream,
        "generator_seed": collection.generator_seed,
        "split_file": f"splits/{collection.split}.parquet",
        "split_file_sha256": split_record["file_sha256"],
        "split_content_sha256": split_record["content_sha256"],
        "n_planned": collection.n,
        "limit": limit,
        "complete": n == collection.n,
        **tally.summary(),
        "file": str(out_path),
        "file_bytes": size,
        "bytes_per_episode": size / n,
        "bytes_per_frame": size / max(tally.frames, 1),
        "file_sha256": p8_adapter.sha256_file(out_path),
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "compression": "lzf",
        **prov,
        "proprio_fields": json.loads(file_attrs["proprio_fields"]),
        "render": "full",
        "appearance_shift": None,
        "workers": workers,
        "chunk": chunk,
        "wall_s": wall_s,
        "hdf5_write_s": write_s,
        "wall_s_per_episode": wall_s / n,
    }
    for path in (out_path.with_suffix(".manifest.json"), run_dir / "manifest.json"):
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(
        f"{collection.name}: done in {wall_s:.1f} s ({write_s:.1f} s writing), "
        f"{size / 1e6:.1f} MB, outcomes {manifest['outcomes']}"
    )
    return manifest
