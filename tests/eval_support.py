"""Synthetic Phase 8 results trees for the scoring, report, figure and GIF-rule tests.

Every outcome here is synthetic: nothing is flown and no student row of a frozen list is read.
Episode *keys* (index, episode seed, realization) are the first ``n`` of each frozen-list cell
(reading frozen-list keys is allowed, P3-D1 §4 / §8). Rows carry every column the matrix writes
(:data:`ila.eval.matrix.ROW_COLUMNS`) and P8's committed-row columns, with values chosen so that
P8's ``cell_metrics`` accepts them. Speeds m/s, lengths m, times s, all model scale.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import runner
from ila.eval import score as sc

#: outcome -> (touchdown, termination reason, closing speed m/s, lateral m, tilt deg)
_SHAPE: Mapping[str, tuple[bool, str, float, float, float]] = {
    "success": (True, "dwell_complete", 0.26, 0.012, 3.0),
    "timeout": (False, "time_limit", math.nan, math.nan, math.nan),
    "hard_landing": (True, "dwell_complete", 0.62, 0.02, 4.0),
    "bounce": (True, "release", 0.3, 0.03, 5.0),
    "off_pad": (True, "off_plate_strike", 0.3, 0.2, 5.0),
    "crash": (True, "tilt_gt_crash", 0.9, 0.05, 70.0),
}

#: (arm, seed, list, ss, index, shift, Ta) -> outcome
type OutcomeFn = Callable[[str, int, str, str, int, str | None, int | None], str]


@functools.cache
def listed_cell(list_name: str, ss: str, n: int) -> tuple[p8_adapter.ListedEpisode, ...]:
    """The first ``n`` listed episodes of one frozen-list cell (keys only)."""
    return tuple(e for e in ev_arms.list_episodes(list_name) if e.ss == ss)[:n]


def _episode_cells(e: p8_adapter.ListedEpisode, outcome: str, salt: float) -> dict[str, str]:
    td, reason, speed, lateral, tilt = _SHAPE[outcome]
    t = 1.5 + 0.01 * (int(e.index) % 7) + salt if td else math.nan
    return {
        "regime": e.regime,
        "ss": e.ss,
        "index": str(e.index),
        "pad": "aft",
        "vessel": str(e.vessel),
        "heading_deg": repr(float(e.heading_deg)),
        "speed_kn": repr(float(e.speed_kn)),
        "realization_seed": str(e.realization_seed),
        "episode_seed": str(e.episode_seed),
        "in_training_distribution": str(bool(e.in_training_distribution)),
        "init_x_m": repr(float(e.init_x_m)),
        "init_y_m": repr(float(e.init_y_m)),
        "init_z_m": repr(float(e.init_z_m)),
        "outcome": outcome,
        "termination_reason": reason,
        "t0_model_s": repr(float(e.t0_model_s)),
        "steps": "60" if td else "361",
        "return": "1.0",
        "control_effort": "1.0",
        "action_jerk": "1.0",
        "dwell_s": "0.5" if outcome == "success" else "0.0",
        "max_penetration_m": "-0.002",
        "tunnelled": "False",
        "contact_events": "1" if td else "0",
        "touchdown_contact": str(td),
        "touchdown_analytic": str(td),
        "td_t_episode_s": repr(t),
        "td_t_episode_analytic_s": repr(t),
        "rel_vz_normal_m_s": repr(-speed),
        "rel_vz_world_z_m_s": repr(-speed),
        "closing_speed_normal_m_s": repr(speed),
        "lateral_offset_m": repr(lateral),
        "rel_tilt_deg": repr(tilt),
        "abs_tilt_deg": repr(tilt),
        "deck_tilt_deg": "1.0",
        "n_contacts": "1" if td else "0",
        "detectors_disagree": "False",
        "closing_speed_world_z_m_s": repr(speed),
        "time_to_touchdown_s": repr(t),
        "effort_mean_sq": "0.25",
        "action_jerk_mean": "0.2",
        "td_in_quiescent_window": "True" if td else "nan",
    }


def p8_row(e: p8_adapter.ListedEpisode, method: str, outcome: str) -> dict[str, str]:
    """A P8-committed-style row (``EPISODE_COLUMNS``) of ``method`` run seed 0."""
    cells = _episode_cells(e, outcome, 0.0)
    cells.update(method=method, privileged=str(method == "oracle_gated"), run_seed="0")
    return {c: cells[c] for c in p8_adapter.EPISODE_COLUMNS}


def student_row(
    e: p8_adapter.ListedEpisode, job: ev_arms.Job, outcome: str, *, ckpt: str
) -> dict[str, str]:
    """A matrix-style row (:data:`ila.eval.matrix.ROW_COLUMNS`) of ``job``."""
    inf = job.inference
    cells = _episode_cells(e, outcome, 0.001 * job.seed)
    cells.update(
        method=job.arm,
        privileged="False",
        run_seed=str(job.seed),
        arm=job.arm,
        teacher=ev_arms.expected_teacher(job.arm),
        student_seed=str(job.seed),
        Ta="" if inf is None else str(inf.Ta),
        K="" if inf is None else str(inf.K),
        ensemble="" if inf is None else str(inf.ensemble),
        latency_steps="0",
        render_cfg_sha256=runner.render_cfg_sha256("full", job.appearance_shift),
        ckpt_sha256=ckpt,
        frames_rendered="61",
        timed_checks="4",
        inference_calls="8" if inf is not None else "60",
        teacher_agreement="0.05",
        tag=job.tag,
        appearance_shift=job.appearance_shift or "",
    )
    return {c: cells[c] for c in ev_matrix.ROW_COLUMNS}


def rows_text(rows: Sequence[Mapping[str, str]]) -> str:
    """CSV text with header :data:`ila.eval.matrix.ROW_COLUMNS` (LF line ends)."""
    lines = [",".join(ev_matrix.ROW_COLUMNS)]
    lines += [",".join(r[c] for c in ev_matrix.ROW_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def write_job(root: Path, job: ev_arms.Job, rows: Sequence[Mapping[str, str]]) -> None:
    """Write a job's ``.csv.gz`` and its sidecar as :mod:`ila.eval.matrix` does (subset)."""
    path = root / job.rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ev_matrix.gzip_bytes(rows_text(rows))
    path.write_bytes(data)
    side = path.with_name(path.name.removesuffix(".csv.gz") + ".json")
    side.write_text(
        json.dumps(
            {
                "job": job.record(),
                "n_rows": len(rows),
                "rows_gz_sha256": hashlib.sha256(data).hexdigest(),
                "wall_clock_s": 1.0,
                "workers": 1,
                "ila_commit": "synthetic",
            }
        ),
        encoding="utf-8",
    )


def default_outcome(
    arm: str, seed: int, list_name: str, ss: str, i: int, shift: str | None, ta: int | None
) -> str:
    """A deterministic synthetic outcome pattern with every class represented."""
    k = (i + 3 * seed + len(arm) + (ta or 0)) % 23
    if ss == "SS6" and k in (1, 5):
        return "hard_landing"
    if k == 2 and ss in ("SS5", "SS6"):
        return "bounce"
    if k == 7 and arm.startswith("dp"):
        return "off_pad"
    if k == 11 and arm.endswith("n100"):
        return "timeout"
    if k == 13 and list_name == "unseen_heading":
        return "crash"
    return "success"


def baseline_outcome(method: str, list_name: str, ss: str, i: int) -> str:
    """Synthetic committed baselines: the teacher fails a few SS6 episodes."""
    if method == "ppo":
        return "hard_landing" if ss == "SS6" and i % 10 == 3 else "success"
    if method == "gated":
        return "timeout" if i % 5 == 1 else "success"
    if method == "pid_track_descend":
        return "bounce" if i % 6 == 0 else "success"
    return "hard_landing" if ss == "SS6" and i % 8 == 0 else "success"


def synthetic_baselines(n: int, list_cells: Mapping[str, Sequence[str]]) -> sc.Baselines:
    """Committed-style rows of every :data:`ila.eval.score.BASELINE_METHODS` method."""
    rows = {
        m: tuple(
            p8_row(e, m, baseline_outcome(m, ln, ss, int(e.index)))
            for ln, cells in list_cells.items()
            for ss in cells
            for e in listed_cell(ln, ss, n)
        )
        for m in sc.BASELINE_METHODS
    }
    return sc.Baselines(rows=rows, provenance=({"synthetic": True},))


def reduced_matrix(keep: Callable[[ev_arms.Job], bool]) -> ev_arms.Matrix:
    """The frozen matrix restricted to the jobs ``keep`` accepts (and their shared cells)."""
    m = ev_arms.load_matrix()
    jobs = tuple(j for j in m.jobs if keep(j))
    shared = tuple(s for s in m.shared if s.source in jobs)
    return dataclasses.replace(m, jobs=jobs, shared=shared)


def write_run_info(root: Path, m: ev_arms.Matrix) -> None:
    """A minimal ``run_info.json`` with the keys the report reads (synthetic values)."""
    jobs = {
        j.rel_path.as_posix(): {"n_rows": 1, "wall_clock_s": 10.0, "workers": 1}
        for j in m.jobs
        if (root / j.rel_path).is_file()
    }
    record = {
        "p8_commit": "synthetic",
        "list_manifest_sha256": "synthetic",
        "lists": {"id": {"sha256": "synthetic"}},
        "success_yaml_sha256": "synthetic",
        "camera_yaml_sha256": "synthetic",
        "matrix_yaml_sha256": "synthetic",
        "committed_rows": [],
        "inference_path": "synthetic",
        "skipped_blocks": dict(m.skipped),
        "shared_cells": [s.record() for s in m.shared],
        "invocations": [{"id": "1", "started_utc": "t0", "ila_commit": "synthetic"}],
        "jobs_total": len(m.jobs),
        "jobs_done": len(jobs),
        "jobs": jobs,
    }
    (root / "run_info.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def synthetic_tree(
    root: Path,
    *,
    n: int,
    outcome: OutcomeFn = default_outcome,
    skip: Callable[[ev_arms.Job], bool] | None = None,
    matrix: ev_arms.Matrix | None = None,
) -> sc.ScoreInputs:
    """Write every matrix job (minus ``skip``) under ``root`` and return the score inputs.

    Args:
        root: The output root (``rows/`` is written under it).
        n: Episodes per cell (the first ``n`` of each frozen-list cell).
        outcome: Synthetic outcome per (arm, seed, list, ss, index, shift, Ta).
        skip: Jobs not written (to exercise missing cells).
        matrix: The matrix (default: the frozen one).

    Returns:
        :class:`ila.eval.score.ScoreInputs` with synthetic baselines and fixed H1/H4 inputs.
    """
    m = ev_arms.load_matrix() if matrix is None else matrix
    for job in m.jobs:
        if skip is not None and skip(job):
            continue
        inf = job.inference
        rows = [
            student_row(
                e,
                job,
                outcome(
                    job.arm,
                    job.seed,
                    job.list_name,
                    ss,
                    int(e.index),
                    job.appearance_shift,
                    None if inf is None else inf.Ta,
                ),
                ckpt=f"ckpt-{job.arm}-{job.seed}",
            )
            for ss in job.sea_states
            for e in listed_cell(job.list_name, ss, n)
        ]
        write_job(root, job, rows)
    write_run_info(root, m)
    list_cells = ev_arms.frozen_list_cells()
    return sc.ScoreInputs(
        matrix=m,
        root=root,
        baselines=synthetic_baselines(n, list_cells),
        listed=lambda ln, ss: listed_cell(ln, ss, n),
        labelled_frames={
            "bc_dagger": [123_216, 123_013, 123_359, 123_223, 123_242],
            "bc_matched": [122_873] * 5,
            "bc_dart": [60_907] * 5,
            "bc": [61_518] * 5,
        },
        labelled_frames_sources={"synthetic": "none"},
        best_end_to_end=("bc_dart", "synthetic: highest mean select success"),
        best_sources={"synthetic": "none"},
        list_cells=list_cells,
        episodes_per_cell=n,
    )
