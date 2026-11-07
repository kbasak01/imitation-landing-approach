"""Phase 8 matrix runner: fly :mod:`ila.eval.arms` jobs on P8's frozen lists, resumably.

Everything here is simulation only (PyBullet physics, PyBullet TinyRenderer frames). Times are
seconds of wall clock unless stated; control steps are 1/30 s model scale (lambda = 1/25;
1/6 s full scale).

Path
----
Every job flies through :func:`ila.train.select.fly_checkpoint` -- the selection path -- on the
**CPU fp32 single-sample path** (one torch thread per worker; ``R1_lerobot_dp`` on ONNX
Runtime's CPU path, one thread), EMA weights, render ``full``, with ``diagnostics=True`` and a
shadow teacher. The batched GPU path (:mod:`ila.eval.batched`) is not used by any arm: it was
never wired for trained checkpoints (``batched.shared_net_factory`` builds dummy networks only,
``SharedDPPolicy`` has no ensembling) and it refuses ``modular``. This module does not import it
(``tests/test_eval_matrix.py``). Rows do not depend on ``workers`` or ``chunk`` (P2-D3 item 1).

Latency arm (tag ``latency_arm``, Phase 9, P3-D1 §7): :func:`latency_flight` builds
``deploy-benchmarker``'s interface -- ``functools.partial(delayed_checkpoint_policy, path, d,
n_action_steps=8, num_inference_steps=10)`` (``bc_dagger``: no ``Ta``/``K``) wrapped in
``p8_adapter.callable_spec`` -- and rows tagged ``RowTags(..., latency_steps=d)``, then flies them
through the same runner call as :func:`ila.train.select.fly_checkpoint` (same episodes, render,
contract variant read from the checkpoint, diagnostics, shadow teacher), on the same CPU fp32
single-sample path as the synchronous core rows it is contrasted with (P3-D1 §3; P9-D1 item 10).
At ``d = 0`` the call equals the synchronous core job's except for the build function
(``tests/test_eval_matrix.py``); the delayed student equals the synchronous one row for row at
``d = 0`` (``tests/test_deploy_latency_arm.py``). Checkpoints, selection pins and the bytes check
are the core arm's: the latency arm adds no arm, so ``selection_pins.json`` is unchanged.

Shadow teacher (``teacher_agreement``)
-------------------------------------
:func:`shadow_groups`: ``T_ppo`` arms -> P8 ``ppo`` seed 0; ``T_pid`` arms -> ``pid_feedforward``;
``T_mix`` arms -> per episode, by :func:`ila.eval.h2_select.tmix_assignment` (one
``ila.data.mix.assign_rows`` call per sea-state cell, seed 20261023; the H2 reference's
construction): the job's episodes are split into a ``ppo``-shadowed half and a
``gated``-shadowed half, each half is flown, and the rows are merged back into list order.
The split cannot change any row other than ``teacher_agreement``: every episode is reset from
its listed seed on a re-pointed environment and a freshly ``reset`` policy, so a row does not
depend on which episodes share its chunk (``tests/test_eval_runner.py::
test_student_rows_do_not_depend_on_workers_or_chunk`` and
``::test_diagnostic_rows_do_not_depend_on_workers_or_chunk``; diffusion noise is keyed
``(run_seed, episode_seed, t)``). ``tests/test_eval_matrix.py`` re-checks it on a ``T_mix``
checkpoint.

Outputs (under ``out``, default ``results/e08``)
-----------------------------------------------
* ``rows/<arm>/s<seed>/<stem>.csv.gz`` (:attr:`ila.eval.arms.Job.rel_path`): columns
  :data:`ROW_COLUMNS` -- P8's ``EPISODE_COLUMNS`` in P8's order and P8's text, the P9 columns,
  the render evidence, the diagnostics (``runner.EVAL_EPISODE_COLUMNS``), then ``tag`` and
  ``appearance_shift`` (empty for the committed look). gzip with ``mtime = 0`` and no file name,
  so the bytes are a function of the rows alone.
* ``rows/<arm>/s<seed>/<stem>.json``: the sidecar (job spec, checkpoint record, shadow plan,
  row count, SHA-256 of the CSV text and of the ``.gz``, wall clock, workers, commits). A job is
  skipped when its sidecar's spec and checkpoint SHA-256 match and the ``.gz`` on disk has the
  recorded SHA-256 and row count (:func:`job_done`).
* ``run_info.json``: commits, list hashes, config hashes, committed-row provenance, host, the
  shared cells, the skipped blocks, every invocation, and per-job wall clocks
  (:func:`write_run_info`).
* ``selection_pins.json``: ``{"<arm>/s<seed>": selection.json SHA-256}`` for every (arm, seed)
  of the whole matrix, written by the first non-smoke invocation; every later invocation
  re-resolves them and refuses to fly if any differs (:func:`check_or_pin_selections`;
  privileged-leak review, Phase 8 RISK 2: no selection may change between invocations).
* ``artifacts/runs/eval/status.json`` (:class:`ila.eval.jobs.JobStatus`): state, jobs and
  episodes done / total, ETA.

Checks on every flown job (:func:`check_rows`): one row per expected episode, in the list's
order (``regime``, ``ss``, ``index``, ``episode_seed``); 200 rows per (list, cell) on a frozen
list; every row's ``arm``, ``student_seed``, ``ckpt_sha256``, ``teacher``, ``Ta`` / ``K`` /
``ensemble`` and ``render_cfg_sha256`` are the job's; ``inference_calls`` present; the start
state of every reset is checked by the runner (P8's ``check_start``).

Projection (:func:`project`)
---------------------------
Measured per arm and seed from its committed ``selection.json``: CPU-seconds per control step
``c = wall_clock_s x workers / sum_candidates(150 x mean_steps)`` on ``select`` (the CPU
single-sample path, Ta = 8 / K = 10 for diffusion arms). Assumptions, stated in the dry run:

1. Wall clock scales as CPU-seconds / workers (linear in workers; the bc-family selections ran
   at 16 workers, every other at 24).
2. Steps per episode in a cell = the selected checkpoint's ``select`` mean steps x (P8 ``ppo``
   s0 mean steps in that cell / its mean over ``id`` SS3-SS5, the ``select`` look-alike).
3. Diffusion cost per episode of L steps = ``L s + calls (e + K d)``, with ``calls = L`` under
   ensembling, else ``ceil(L / Ta)``; the ratio to the measured ``Ta = 8, K = 10`` setting uses
   P2-D2's single-thread costs ``s = 11`` ms (render + physics per step), ``e = 31`` ms (encoder
   per call), ``d = 14`` ms (per denoising step) (:data:`P2D2_MS`).
4. Appearance shift costs as the committed look; the shadow teacher's MLP per step and the pool
   start-up per job are not added (both are included in the measured selection rate only to the
   extent selection had them: start-up yes, shadow no).
5. Latency arm (:func:`latency_cost_ratio`): episodes as long as the synchronous arm's; one
   inference call per ``max(d, 1)`` steps for BC and per ``max(d, Ta)`` steps for ``dp``
   (requests never overlap, P3-D1 §7), with :data:`P2D2_MS` costs; BC's head is free.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import gzip
import hashlib
import io
import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.deploy import latency_arm as deploy_latency_arm
from ila.eval import arms as ev_arms
from ila.eval import h2_select, runner
from ila.eval.jobs import JobStatus, host_info, now_utc, rel, write_json
from ila.policies.registry import read_checkpoint
from ila.provenance import EXPECTED_MANIFEST_SHA256
from ila.render.config import RENDER_CONFIG_PATH
from ila.train import manifest as mf
from ila.train import select as train_select

__all__ = [
    "CAMERA_SHA256_PREFIX",
    "CAMERA_SHA256_SUFFIX",
    "EXTRA_COLUMNS",
    "OUT_ROOT",
    "P2D2_MS",
    "PROJECTION_ASSUMPTIONS",
    "ROW_COLUMNS",
    "STATUS_DIR",
    "PINS_FILE",
    "DirtyTreeError",
    "JobProjection",
    "RowCheckError",
    "SelectionPinError",
    "check_or_pin_selections",
    "check_rows",
    "diffusion_cost_ratio",
    "dirty_paths",
    "expected_episodes",
    "filter_jobs",
    "fly_job",
    "gzip_bytes",
    "job_done",
    "latency_cost_ratio",
    "latency_flight",
    "order_jobs",
    "ppo_cell_lengths",
    "project",
    "read_rows_text",
    "rows_csv",
    "run_jobs",
    "selection_pins",
    "shadow_groups",
    "smoke_jobs",
    "static_provenance",
    "text_rows",
    "write_job",
    "write_run_info",
]

#: Default output root (committed: rows, sidecars, ``run_info.json``).
OUT_ROOT: Final[Path] = p8_adapter.REPO_ROOT / "results" / "e08"

#: Selection pins, under the output root (module docstring).
PINS_FILE: Final[str] = "selection_pins.json"

#: ``status.json`` and the log of the background job.
STATUS_DIR: Final[Path] = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "eval"

#: Columns after ``runner.EVAL_EPISODE_COLUMNS``.
EXTRA_COLUMNS: Final[tuple[str, ...]] = ("tag", "appearance_shift")

#: Column order of every rows file.
ROW_COLUMNS: Final[tuple[str, ...]] = (*runner.EVAL_EPISODE_COLUMNS, *EXTRA_COLUMNS)

#: ``configs/render/camera.yaml`` SHA-256 as P3-D1 §1 prints it (``53f80265…0380``).
CAMERA_SHA256_PREFIX: Final[str] = "53f80265"
CAMERA_SHA256_SUFFIX: Final[str] = "0380"

#: P2-D2's single-thread CPU costs of a dummy ``dp`` (``ila.train.dp_sweeps`` docstring), ms:
#: render + physics per control step, encoder per call, per DDIM denoising step.
P2D2_MS: Final[Mapping[str, float]] = {"step": 11.0, "encoder": 31.0, "denoise": 14.0}

#: The projection's assumptions, printed with it (module docstring).
PROJECTION_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "CPU-s per control step measured per (arm, seed) from its committed selection.json: "
    "wall_clock_s x workers / sum over candidates of (150 x mean_steps) on select "
    "(CPU fp32 single-sample path; diffusion arms at Ta = 8, K = 10, no ensembling)",
    "wall clock = CPU-s / workers (linear in workers; bc-family selections ran at 16 workers, "
    "the rest at 24)",
    "steps per episode = the selected checkpoint's select mean steps x (P8 ppo s0 mean steps "
    "in the cell / P8 ppo s0 mean steps over id SS3-SS5)",
    "diffusion: cost(L) = L*11 ms + calls*(31 ms + K*14 ms) (P2-D2 single-thread dummy dp), "
    "calls = L with ensembling else ceil(L/Ta); scaled by its ratio to Ta = 8, K = 10",
    "appearance shift costs as the committed look; shadow-teacher MLP cost not added",
    "latency arm: episodes as long as the synchronous arm's; one call per max(d, 1) steps (BC) "
    "or max(d, Ta) steps (dp), P2-D2 costs, BC head free; ratio to the measured select rate",
)

#: Diffusion inference at which selection measured the rate (P3-D1 §4).
_SELECT_INF: Final[ev_arms.Inference] = ev_arms.Inference(8, 10, False)


class DirtyTreeError(RuntimeError):
    """The checkout has uncommitted changes outside the output root (refused without override)."""


class RowCheckError(AssertionError):
    """A flown job's rows are not the job's episodes, cells or tags."""


class SelectionPinError(RuntimeError):
    """A selection record differs from the one pinned by the first invocation."""


# --------------------------------------------------------------------------- rows out


def rows_csv(rows: Sequence[Mapping[str, Any]], job: ev_arms.Job) -> str:
    """Serialise one job's rows: :func:`runner.vision_rows_csv`, then :data:`EXTRA_COLUMNS`.

    The runner's text is kept byte for byte (P8's columns as P8 writes them); ``tag`` and
    ``appearance_shift`` (empty for ``None``) are appended to every line.

    Args:
        rows: Runner rows with the diagnostic columns, in list order.
        job: The job (its tag and shift).

    Returns:
        CSV text with header :data:`ROW_COLUMNS`.

    Raises:
        ValueError: If the rows lack the diagnostic columns or the text does not split into
            one line per row.
    """
    if not rows or any(c not in r for r in rows for c in runner.DIAGNOSTIC_COLUMNS):
        raise ValueError("matrix rows must carry runner.DIAGNOSTIC_COLUMNS")
    text = runner.vision_rows_csv(rows)
    lines = text.split("\n")
    if lines[-1] != "" or len(lines) != len(rows) + 2:
        raise ValueError("runner CSV does not have one line per row")
    if lines[0] != ",".join(runner.EVAL_EPISODE_COLUMNS):
        raise ValueError("runner CSV header is not EVAL_EPISODE_COLUMNS")
    extra = f",{job.tag},{job.appearance_shift or ''}"
    body = [lines[0] + "," + ",".join(EXTRA_COLUMNS)] + [ln + extra for ln in lines[1:-1]]
    return "\n".join(body) + "\n"


def gzip_bytes(text: str) -> bytes:
    """Compress ``text`` (UTF-8, gzip) reproducibly: ``mtime = 0``, no file name, level 9."""
    return gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- jobs


def expected_episodes(job: ev_arms.Job) -> list[p8_adapter.ListedEpisode]:
    """The job's episodes: its list's rows in its cells, in committed order (frozen lists)."""
    keep = set(job.sea_states)
    return [e for e in ev_arms.list_episodes(job.list_name) if e.ss in keep]


def shadow_groups(
    teacher: str, episodes: Sequence[p8_adapter.ListedEpisode]
) -> list[tuple[str, list[int]]]:
    """Split a job's episode positions by the P8 teacher that shadows them.

    Args:
        teacher: The arm's teacher label (``T_ppo``, ``T_pid``, ``T_mix``).
        episodes: The job's episodes, in order.

    Returns:
        ``[(p8 teacher name, positions)]`` covering every position exactly once (``T_mix``:
        ``ppo`` first, then ``gated``; an empty group is omitted).

    Raises:
        ValueError: For an unknown teacher label.
    """
    if teacher not in ev_arms.SHADOW_TEACHER:
        raise ValueError(f"unknown teacher {teacher!r}")
    name = ev_arms.SHADOW_TEACHER[teacher]
    if name is not None:
        return [(name, list(range(len(episodes))))]
    assigned = h2_select.tmix_assignment(episodes)
    groups = [(p, [i for i, a in enumerate(assigned) if a == p]) for p in ("ppo", "gated")]
    if sorted(i for _, idx in groups for i in idx) != list(range(len(episodes))):
        raise ValueError("T_mix assignment does not cover every episode once")
    return [(p, idx) for p, idx in groups if idx]


def latency_flight(
    job: ev_arms.Job, ckpt: ev_arms.Checkpoint
) -> tuple[p8_adapter.PolicySpec, runner.RowTags]:
    """The latency arm's policy spec and row tags (module docstring; P3-D1 §7).

    Args:
        job: A ``latency_arm`` job (``latency_steps`` = d control steps, 33.333 ms model scale
            each).
        ckpt: Its resolved checkpoint (the core arm's).

    Returns:
        ``(callable_spec(arm, partial(delayed_checkpoint_policy, path, d, ...)), RowTags)``.

    Raises:
        ValueError: If the job is not a latency job, its checkpoint is an ONNX export, or a
            diffusion job asks for ensembling or a non-``Ta = 8``, ``K = 10`` setting.
    """
    d = job.latency_steps
    if job.tag != ev_arms.LATENCY_TAG or d is None:
        raise ValueError(f"{job.rel_path}: not a latency_arm job")
    if ckpt.kind == train_select.ONNX_EXPORTS:
        raise ValueError(f"{job.rel_path}: the latency arm flies .pt checkpoints only")
    if job.inference is None:
        build = functools.partial(deploy_latency_arm.delayed_checkpoint_policy, str(ckpt.path), d)
        tags = runner.RowTags(
            arm=job.arm,
            teacher=ckpt.teacher,
            student_seed=job.seed,
            ckpt_sha256=ckpt.sha256,
            latency_steps=d,
        )
    else:
        ta, k, ensemble = job.inference.as_tuple()
        if ensemble or (ta, k) != (8, 10):
            raise ValueError(f"{job.rel_path}: the latency arm flies dp at Ta = 8, K = 10, off")
        build = functools.partial(
            deploy_latency_arm.delayed_checkpoint_policy,
            str(ckpt.path),
            d,
            n_action_steps=ta,
            num_inference_steps=k,
        )
        tags = runner.RowTags(
            arm=job.arm,
            teacher=ckpt.teacher,
            student_seed=job.seed,
            Ta=ta,
            K=k,
            ensemble=False,
            ckpt_sha256=ckpt.sha256,
            latency_steps=d,
        )
    return p8_adapter.callable_spec(job.arm, build, run_seed=job.seed), tags


def _fly_latency(
    job: ev_arms.Job,
    ckpt: ev_arms.Checkpoint,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    workers: int,
    chunk: int,
    shadow_teacher: p8_adapter.PolicySpec,
) -> list[dict[str, Any]]:
    """Fly a latency job: :func:`ila.train.select.fly_checkpoint`'s runner call, delayed build."""
    spec, tags = latency_flight(job, ckpt)
    return runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="full",
        workers=workers,
        chunk=chunk,
        appearance_shift=job.appearance_shift,
        contract_variant=str(read_checkpoint(ckpt.path)["obs_variant"]),
        diagnostics=True,
        shadow_teacher=shadow_teacher,
    )


def fly_job(
    job: ev_arms.Job,
    ckpt: ev_arms.Checkpoint,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    workers: int,
    chunk: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fly one job (CPU fp32 single-sample path; module docstring).

    Args:
        job: The job.
        ckpt: Its resolved checkpoint (bytes already verified).
        episodes: The episodes, in order.
        workers: Processes (rows do not depend on it).
        chunk: Episodes per task (rows do not depend on it).

    Returns:
        ``(rows in episode order, shadow plan records)``.
    """
    rows: list[dict[str, Any] | None] = [None] * len(episodes)
    plan = []
    for teacher_name, idx in shadow_groups(ckpt.teacher, episodes):
        shadow = p8_adapter.teacher_spec(teacher_name)
        group = [episodes[i] for i in idx]
        if job.latency_steps is not None:
            out = _fly_latency(
                job, ckpt, group, workers=workers, chunk=chunk, shadow_teacher=shadow
            )
        else:
            out = train_select.fly_checkpoint(
                ckpt.path,
                group,
                arm=job.arm,
                run_seed=job.seed,
                sha256=ckpt.sha256,
                workers=workers,
                chunk=chunk,
                inference=None if job.inference is None else job.inference.as_tuple(),
                teacher=ckpt.teacher,
                appearance_shift=job.appearance_shift,
                diagnostics=True,
                shadow_teacher=shadow,
            )
        if len(out) != len(idx):
            raise RowCheckError(f"{job.rel_path}: {len(out)} rows for {len(idx)} episodes")
        for i, r in zip(idx, out, strict=True):
            rows[i] = r
        plan.append({"shadow_teacher": teacher_name, "seed": 0, "episodes": len(idx)})
    if any(r is None for r in rows):
        raise RowCheckError(f"{job.rel_path}: an episode was not flown")
    return [r for r in rows if r is not None], plan


def _blank(value: Any) -> str:
    return "" if value is None else str(value)


def check_rows(
    job: ev_arms.Job,
    ckpt: ev_arms.Checkpoint,
    rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    frozen: bool = True,
) -> None:
    """Check a job's rows against its episodes and spec (module docstring).

    Rows may be live runner rows or rows read back from the CSV (string cells).

    Args:
        job: The job.
        ckpt: Its checkpoint.
        rows: The rows.
        episodes: The expected episodes, in order.
        frozen: A frozen-list job: also require 200 rows per cell of ``job.sea_states``.

    Raises:
        RowCheckError: On any mismatch.
    """
    where = job.rel_path.as_posix()

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise RowCheckError(f"{where}: {what}")

    need(len(rows) == len(episodes), f"{len(rows)} rows for {len(episodes)} episodes")
    got = [(str(r["regime"]), str(r["ss"]), int(r["index"]), int(r["episode_seed"])) for r in rows]
    want = [(e.regime, e.ss, int(e.index), int(e.episode_seed)) for e in episodes]
    need(got == want, "rows are not the expected episodes in list order")
    if frozen:
        counts = Counter(str(r["ss"]) for r in rows)
        need(
            dict(counts) == dict.fromkeys(job.sea_states, ev_arms.EPISODES_PER_CELL),
            f"rows per cell {dict(counts)}",
        )
        need({str(r["regime"]) for r in rows} == {job.list_name}, "regime is not the list")
    inf = job.inference
    want_tags = {
        "arm": job.arm,
        "student_seed": str(job.seed),
        "ckpt_sha256": ckpt.sha256,
        "teacher": ckpt.teacher,
        "Ta": "" if inf is None else str(inf.Ta),
        "K": "" if inf is None else str(inf.K),
        "ensemble": "" if inf is None else str(inf.ensemble),
        "latency_steps": str(job.latency_steps or 0),
        "render_cfg_sha256": runner.render_cfg_sha256("full", job.appearance_shift),
    }
    for col, value in want_tags.items():
        bad = {_blank(r[col]) for r in rows} - {value}
        need(not bad, f"column {col}: {sorted(bad)[:3]} != {value!r}")
    need(all(_blank(r["inference_calls"]) != "" for r in rows), "inference_calls missing")


def write_job(
    out_root: Path,
    job: ev_arms.Job,
    ckpt: ev_arms.Checkpoint,
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Sequence[Mapping[str, Any]],
    wall_clock_s: float,
    workers: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a job's ``.csv.gz`` and then its sidecar (the sidecar marks it done).

    Returns:
        The sidecar record.
    """
    path = out_root / job.rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = rows_csv(rows, job)
    data = gzip_bytes(text)
    tmp = path.with_suffix(".gz.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    sidecar = {
        "job": job.record(),
        "rows_file": job.rel_path.as_posix(),
        "checkpoint": ckpt.record(),
        "shadow": list(plan),
        "n_rows": len(rows),
        "columns": list(ROW_COLUMNS),
        "rows_csv_sha256": _sha(text.encode("utf-8")),
        "rows_gz_sha256": _sha(data),
        "inference_path": "ort_cpu_fp32_single_sample"
        if ckpt.kind == train_select.ONNX_EXPORTS
        else "cpu_fp32_single_sample",
        "wall_clock_s": wall_clock_s,
        "workers": workers,
        "finished_utc": now_utc(),
        **provenance,
    }
    write_json(_sidecar(out_root, job), sidecar)
    return sidecar


def _sidecar(out_root: Path, job: ev_arms.Job) -> Path:
    return out_root / job.rel_path.with_name(job.rel_path.name.removesuffix(".csv.gz") + ".json")


def job_done(out_root: Path, job: ev_arms.Job, ckpt: ev_arms.Checkpoint) -> bool:
    """Whether a job's outputs exist and match (spec, checkpoint, rows SHA-256, row count)."""
    path, side = out_root / job.rel_path, _sidecar(out_root, job)
    if not (path.is_file() and side.is_file()):
        return False
    rec = mf.read_json(side)
    return bool(
        rec.get("job") == job.record()
        and rec.get("checkpoint", {}).get("sha256") == ckpt.sha256
        and rec.get("rows_gz_sha256") == _sha(path.read_bytes())
        and rec.get("n_rows") == len(_read_rows(path))
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- selection of jobs


def filter_jobs(jobs: Iterable[ev_arms.Job], only: Sequence[str] = ()) -> list[ev_arms.Job]:
    """Keep jobs matching ``only`` tokens: a tag, an arm, or a seed ``s<n>``/``<n>``.

    Tokens of one kind are OR-ed, kinds are AND-ed (``core dp s0``: core jobs of ``dp``
    seed 0).

    Raises:
        ValueError: On a token that is no tag, arm or seed of these jobs.
    """
    jobs = list(jobs)
    tags, arm_names, seeds = set(), set(), set()
    known_arms = {j.arm for j in jobs}
    for tok in only:
        if tok in ev_arms.TAGS:
            tags.add(tok)
        elif tok in known_arms:
            arm_names.add(tok)
        elif tok.removeprefix("s").isdigit():
            seeds.add(int(tok.removeprefix("s")))
        else:
            raise ValueError(f"--only token {tok!r} is no tag, arm or seed")
    return [
        j
        for j in jobs
        if (not tags or j.tag in tags)
        and (not arm_names or j.arm in arm_names)
        and (not seeds or j.seed in seeds)
    ]


def order_jobs(
    jobs: Iterable[ev_arms.Job], order: Sequence[str] = ev_arms.DEFAULT_ORDER
) -> list[ev_arms.Job]:
    """Sort jobs by tag in ``order`` (stable within a tag: expansion order).

    Raises:
        ValueError: If ``order`` is not a permutation of :data:`ila.eval.arms.TAGS`.
    """
    if sorted(order) != sorted(ev_arms.TAGS):
        raise ValueError(f"order {list(order)} is not a permutation of {list(ev_arms.TAGS)}")
    rank = {t: i for i, t in enumerate(order)}
    return sorted(jobs, key=lambda j: rank[j.tag])


def smoke_jobs(jobs: Sequence[ev_arms.Job], select_cells: Sequence[str]) -> list[ev_arms.Job]:
    """One job per (tag, arm) -- its first, at its lowest seed -- re-pointed at ``select``.

    Args:
        jobs: Matrix jobs, in order.
        select_cells: Cells of the ``select`` episodes the smoke flies (e.g. ``("SS3",)``).

    Returns:
        Jobs whose ``list_name`` is ``select`` (never a frozen list).
    """
    seen: set[tuple[str, str]] = set()
    out = []
    for j in sorted(jobs, key=lambda j: j.seed):
        if (j.tag, j.arm) in seen:
            continue
        seen.add((j.tag, j.arm))
        out.append(
            dataclasses.replace(
                j, list_name="select", sea_states=tuple(select_cells), full_list=False
            )
        )
    order = {t: i for i, t in enumerate(ev_arms.TAGS)}
    return sorted(out, key=lambda j: order[j.tag])


# --------------------------------------------------------------------------- projection


@dataclass(frozen=True)
class JobProjection:
    """Projected cost of one job.

    Attributes:
        job: The job.
        episodes: Episodes flown.
        steps: Expected control steps (30 Hz, model scale).
        cpu_s: Projected CPU-seconds.
    """

    job: ev_arms.Job
    episodes: int
    steps: float
    cpu_s: float


def ppo_cell_lengths() -> dict[tuple[str, str], float]:
    """P8 ``ppo`` seed 0 mean control steps per (regime, cell), from P8's committed e07 rows."""
    rows = p8_adapter.committed_rows("e07", "ppo", 0).rows
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["regime"], r["ss"]), []).append(float(r["steps"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def diffusion_cost_ratio(inf: ev_arms.Inference, steps: float) -> float:
    """Cost of an episode of ``steps`` control steps at ``inf`` over its cost at Ta 8, K 10.

    Uses :data:`P2D2_MS` (module docstring, assumption 3).
    """

    def cost(i: ev_arms.Inference) -> float:
        calls = steps if i.ensemble else math.ceil(steps / i.Ta)
        return steps * P2D2_MS["step"] + calls * (P2D2_MS["encoder"] + i.K * P2D2_MS["denoise"])

    return cost(inf) / cost(_SELECT_INF)


def latency_cost_ratio(job: ev_arms.Job, steps: float) -> float:
    """Cost of a latency-arm episode of ``steps`` control steps over its synchronous cost.

    Assumption 5 (module docstring): one call per ``max(d, 1)`` steps (BC) or ``max(d, Ta)``
    steps (``dp``), :data:`P2D2_MS` costs; the synchronous reference is one call per step
    (BC) or the measured ``Ta = 8``, ``K = 10`` setting (``dp``). ``d = 0`` gives 1.

    Raises:
        ValueError: If ``job`` is not a latency job.
    """
    d = job.latency_steps
    if d is None:
        raise ValueError(f"{job.rel_path}: not a latency_arm job")
    ms = P2D2_MS
    if job.inference is None:
        sync = steps * (ms["step"] + ms["encoder"])
        return (steps * ms["step"] + math.ceil(steps / max(d, 1)) * ms["encoder"]) / sync
    inf = job.inference
    per_call = ms["encoder"] + inf.K * ms["denoise"]
    delayed = steps * ms["step"] + math.ceil(steps / max(d, inf.Ta)) * per_call
    sync_inf = steps * ms["step"] + math.ceil(steps / inf.Ta) * per_call
    return (delayed / sync_inf) * diffusion_cost_ratio(inf, steps)


def project(
    jobs: Sequence[ev_arms.Job],
    checkpoints: Mapping[tuple[str, int], ev_arms.Checkpoint],
    lengths: Mapping[tuple[str, str], float],
) -> list[JobProjection]:
    """Project every job's CPU cost (module docstring, assumptions 1-4).

    Args:
        jobs: Jobs.
        checkpoints: (arm, seed) -> resolved checkpoint (its selection rate).
        lengths: :func:`ppo_cell_lengths`.

    Returns:
        One :class:`JobProjection` per job.
    """
    proxy = sum(lengths[("id", c)] for c in ("SS3", "SS4", "SS5")) / 3
    out = []
    for j in jobs:
        ck = checkpoints[(j.arm, j.seed)]
        steps = cpu = 0.0
        for cell in j.sea_states:
            ep_steps = ck.select_mean_steps * lengths[(j.list_name, cell)] / proxy
            if j.latency_steps is not None:
                ratio = latency_cost_ratio(j, ep_steps)
            elif j.inference is None:
                ratio = 1.0
            else:
                ratio = diffusion_cost_ratio(j.inference, ep_steps)
            n = ev_arms.EPISODES_PER_CELL
            steps += n * ep_steps
            cpu += n * ep_steps * ck.cpu_s_per_step * ratio
        out.append(JobProjection(j, j.n_episodes, steps, cpu))
    return out


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=p8_adapter.REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def dirty_paths(exclude: Path | None = None) -> list[str]:
    """Uncommitted paths of this checkout (``git status --porcelain``), outside ``exclude``."""
    skip = None
    if exclude is not None:
        try:
            skip = exclude.resolve().relative_to(p8_adapter.REPO_ROOT).as_posix().rstrip("/") + "/"
        except ValueError:
            skip = None
    out = []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].strip()
        if skip is not None and (path + "/").startswith(skip):
            continue
        out.append(line)
    return out


def static_provenance(out_root: Path) -> dict[str, Any]:
    """Everything ``run_info.json`` records that does not change while the matrix flies.

    Raises:
        RuntimeError: If the P8 list manifest or the camera config is not P3-D1's.
    """
    manifest = p8_adapter.manifest_sha256()
    if manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"P8 list MANIFEST {manifest} != P3-D1's {EXPECTED_MANIFEST_SHA256}")
    camera = p8_adapter.sha256_file(RENDER_CONFIG_PATH)
    if not (camera.startswith(CAMERA_SHA256_PREFIX) and camera.endswith(CAMERA_SHA256_SUFFIX)):
        raise RuntimeError(f"camera.yaml {camera} is not P3-D1's 53f80265…0380")
    from ila.eval import stats

    committed = [p8_adapter.committed_rows("e07", "ppo", 0).provenance()]
    for method in ("pid_feedforward", "pid_track_descend", "oracle_gated", "gated"):
        committed.append(p8_adapter.committed_rows("e01", method).provenance())
    return {
        "p8_commit": p8_adapter.p8_commit(),
        "list_manifest_sha256": manifest,
        "lists": {
            n: {"file": rel(p), "sha256": p8_adapter.sha256_file(p)}
            for n, p in p8_adapter.list_paths().items()
        },
        "success_yaml_sha256": p8_adapter.success_yaml_sha256(),
        "camera_yaml_sha256": camera,
        "matrix_yaml_sha256": p8_adapter.sha256_file(stats.MATRIX_YAML),
        "committed_rows": committed,
        "output_root": rel(out_root),
    }


def write_run_info(
    out_root: Path,
    *,
    matrix: ev_arms.Matrix,
    static: Mapping[str, Any],
    invocation: Mapping[str, Any],
    jobs_of_record: Sequence[ev_arms.Job] | None = None,
) -> dict[str, Any]:
    """(Re)write ``run_info.json``: static provenance, invocations, shared cells, per-job sidecars.

    Earlier invocations are kept (read from the existing file); per-job records come from the
    sidecars on disk, so a resumed run keeps every job's wall clock.

    Args:
        out_root: Output root.
        matrix: The expanded matrix.
        static: :func:`static_provenance`.
        invocation: This invocation's record (replaces an earlier one with the same ``id``).
        jobs_of_record: Jobs whose sidecars are summarised (default every matrix job; a smoke
            run passes its own ``select`` jobs).
    """
    path = out_root / "run_info.json"
    old = mf.read_json(path) if path.is_file() else {}
    invocations = [i for i in old.get("invocations", []) if i.get("id") != invocation["id"]]
    invocations.append(dict(invocation))
    of_record = matrix.jobs if jobs_of_record is None else tuple(jobs_of_record)
    jobs = {}
    for j in of_record:
        side = _sidecar(out_root, j)
        if side.is_file():
            rec = mf.read_json(side)
            jobs[j.rel_path.as_posix()] = {
                k: rec.get(k)
                for k in ("n_rows", "rows_gz_sha256", "wall_clock_s", "workers", "ila_commit")
            }
    record = {
        **static,
        "inference_path": (
            "CPU fp32 single-sample for every arm (R1: ONNX Runtime CPU, one thread); the batched "
            "GPU path is not wired for trained checkpoints and refuses modular, so no arm uses it"
        ),
        "skipped_blocks": dict(matrix.skipped),
        "shared_cells": [s.record() for s in matrix.shared],
        "invocations": invocations,
        "jobs_total": len(of_record),
        "jobs_done": len(jobs),
        "jobs": jobs,
    }
    write_json(path, record)
    return record


# --------------------------------------------------------------------------- run


def selection_pins(matrix: ev_arms.Matrix) -> dict[str, str]:
    """``{"<arm>/s<seed>": selection.json SHA-256}`` for every (arm, seed) of the matrix.

    Each is resolved (and cross-checked) by :func:`ila.eval.arms.resolve_checkpoint`.
    """
    return {
        f"{a.arm}/s{seed}": ev_arms.resolve_checkpoint(a, seed).selection_json_sha256
        for a in matrix.arms
        for seed in a.seeds
    }


def check_or_pin_selections(
    out_root: Path, matrix: ev_arms.Matrix, *, write: bool
) -> dict[str, Any]:
    """Compare the matrix's selections with ``<out_root>/selection_pins.json``, or pin them.

    Args:
        out_root: Output root.
        matrix: The expanded matrix (every arm and seed, whatever ``--only`` selects).
        write: Write the pins when the file does not exist (the first non-smoke invocation).

    Returns:
        ``{"file", "sha256", "pins", "created"}`` (``file``/``sha256`` ``None`` when nothing
        was pinned and ``write`` is off).

    Raises:
        SelectionPinError: If a pin file exists and any (arm, seed) is missing, extra, or has
            another ``selection.json`` SHA-256.
    """
    current = selection_pins(matrix)
    path = out_root / PINS_FILE
    if path.is_file():
        pinned = mf.read_json(path)["pins"]
        diff = sorted(k for k in set(pinned) | set(current) if pinned.get(k) != current.get(k))
        if diff:
            raise SelectionPinError(
                f"{path}: selections differ from the pins for {diff[:10]} "
                f"({len(diff)} of {len(pinned)}); nothing flown"
            )
        return {
            "file": rel(path),
            "sha256": _sha(path.read_bytes()),
            "pins": current,
            "created": False,
        }
    if not write:
        return {"file": None, "sha256": None, "pins": current, "created": False}
    write_json(path, {"created_utc": now_utc(), "pins": dict(sorted(current.items()))})
    return {"file": rel(path), "sha256": _sha(path.read_bytes()), "pins": current, "created": True}


def run_jobs(
    jobs: Sequence[ev_arms.Job],
    matrix: ev_arms.Matrix,
    *,
    out_root: Path,
    status_dir: Path,
    workers: int,
    chunk: int = 10,
    allow_dirty: bool = False,
    episodes_of: Callable[[ev_arms.Job], list[p8_adapter.ListedEpisode]] | None = None,
    frozen: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly ``jobs`` in order, skipping finished ones; write rows, sidecars, run info, status.

    Args:
        jobs: Jobs to fly, in order.
        matrix: The expanded matrix (for ``run_info.json``).
        out_root: Output root.
        status_dir: ``status.json`` directory.
        workers: Processes per job.
        chunk: Episodes per task.
        allow_dirty: Fly on a dirty tree (smoke only).
        episodes_of: Job -> episodes (default :func:`expected_episodes`, the frozen list).
        frozen: Frozen-list jobs (200 rows per cell checked).
        log: Line sink.

    Returns:
        The final ``run_info.json`` record.

    Raises:
        DirtyTreeError: On a dirty tree without ``allow_dirty``.
        SelectionPinError: If a selection differs from its pin (before anything is flown).
    """
    dirty = dirty_paths(exclude=out_root)
    if dirty and not allow_dirty:
        raise DirtyTreeError(f"uncommitted changes (use --allow-dirty for a smoke run): {dirty}")
    pins = check_or_pin_selections(out_root, matrix, write=frozen)
    episodes_of = expected_episodes if episodes_of is None else episodes_of
    status = JobStatus(status_dir, "eval")
    ila_commit = _git("rev-parse", "HEAD").strip()
    static = {
        **static_provenance(out_root),
        "selection_pins": {"file": pins["file"], "sha256": pins["sha256"]},
    }
    invocation = {
        "id": status.record["started_utc"],
        "started_utc": status.record["started_utc"],
        "ila_commit": ila_commit,
        "ila_dirty": bool(dirty),
        "ila_dirty_paths": dirty[:50],
        "host": host_info(),
        "workers": workers,
        "chunk": chunk,
        "jobs_requested": len(jobs),
        "selection_pins_created": pins["created"],
    }
    provenance = {
        "ila_commit": ila_commit,
        "ila_dirty": bool(dirty),
        "p8_commit": static["p8_commit"],
    }
    record = None if frozen else list(jobs)
    total_eps = sum(len(episodes_of(j)) for j in jobs)
    done_jobs = done_eps = flown_eps = 0
    flown_s = 0.0
    status.update(
        "running", jobs_total=len(jobs), jobs_done=0, episodes_total=total_eps, episodes_done=0
    )
    try:
        for job in jobs:
            spec = matrix.arm(job.arm)
            ckpt = ev_arms.resolve_checkpoint(spec, job.seed)
            episodes = episodes_of(job)
            if job_done(out_root, job, ckpt):
                log(f"skip {job.rel_path} (done)")
            else:
                if pins["pins"][f"{job.arm}/s{job.seed}"] != ckpt.selection_json_sha256:
                    raise SelectionPinError(f"{job.arm} s{job.seed}: selection changed mid-run")
                ev_arms.verify_checkpoint_bytes(ckpt)  # bytes, arm, seed, obs_variant; every job
                status.update(current_job=job.rel_path.as_posix())
                t0 = time.perf_counter()
                rows, plan = fly_job(job, ckpt, episodes, workers=workers, chunk=chunk)
                wall = time.perf_counter() - t0
                check_rows(job, ckpt, rows, episodes, frozen=frozen)
                write_job(
                    out_root,
                    job,
                    ckpt,
                    rows,
                    plan=plan,
                    wall_clock_s=wall,
                    workers=workers,
                    provenance=provenance,
                )
                flown_eps += len(episodes)
                flown_s += wall
                ok = sum(str(r["outcome"]) == "success" for r in rows)
                log(f"flew {job.rel_path}: {len(rows)} episodes, success {ok}, {wall:.0f} s")
            done_jobs += 1
            done_eps += len(episodes)
            rate = flown_s / flown_eps if flown_eps else None
            status.update(
                jobs_done=done_jobs,
                episodes_done=done_eps,
                eta_s=None if rate is None else rate * (total_eps - done_eps),
            )
            invocation["jobs_done"] = done_jobs
            write_run_info(
                out_root, matrix=matrix, static=static, invocation=invocation, jobs_of_record=record
            )
    except BaseException as exc:
        status.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    invocation["finished_utc"] = now_utc()
    invocation["episodes_flown"] = flown_eps
    invocation["flight_wall_clock_s"] = flown_s
    final = write_run_info(
        out_root, matrix=matrix, static=static, invocation=invocation, jobs_of_record=record
    )
    status.update("done", eta_s=0.0, current_job=None)
    return final


def read_rows_text(path: Path) -> str:
    """Return a rows file's CSV text (for tests and the scorer)."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return f.read()


def text_rows(text: str) -> list[dict[str, str]]:
    """Parse a rows CSV text."""
    return list(csv.DictReader(io.StringIO(text, newline="")))
