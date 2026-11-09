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
from ila.eval import hypotheses as hy
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

#: The latency arm's d per arm in every synthetic tree (control steps; not delay.json's).
SYNTHETIC_DELAYS: Mapping[str, int] = {"bc_dagger": 2, "dp": 3}

#: Synthetic commits recorded by the synthetic latency stages (``run_info.json``, rows).
SYNTHETIC_LATENCY_COMMITS: Mapping[str, str] = {"export": "c0ffee1", "latency": "beef002"}

#: Synthetic closed-loop flights: arm -> provider -> (scored, status, flips as
#: (reference outcome, provider outcome) per flipped episode). n = 10, reference 6 successes.
_CL_N: int = 10
_CL_REF_SUCCESS: int = 6
_LOST = ("success", "off_pad")
_GAINED = ("off_pad", "success")
_OTHER = ("bounce", "off_pad")
_FLIGHTS_OK: Mapping[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {
    "ort_cuda_fp32": ("True", "flown", (_LOST, _GAINED)),
    "ort_trt_fp16": ("True", "refused (open-loop parity not passed)", ()),
    "ort_trt_fp32": ("False", "flown", (_OTHER,)),
}
_FLIGHTS_FAIL: Mapping[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {
    "ort_cpu_fp32": ("True", "flown", (_LOST, _LOST, _GAINED, _GAINED)),
    "ort_cuda_fp32": ("True", "flown", (_LOST, _LOST, _LOST, _GAINED, _OTHER)),
    "torch_cuda_fp32": ("True", "flown", (_LOST, _GAINED, _GAINED, _GAINED)),
    "ort_trt_fp16": ("True", "refused (open-loop parity not passed)", ()),
    "ort_trt_fp32": ("False", "flown", (_LOST,) * 6),
}
#: provider -> (precision, chunk max-abs unrolled, loop) for dp in parity.csv.
_PARITY: Mapping[str, tuple[str, str, str]] = {
    "ort_cpu_fp32": ("fp32", "2.2e-05", "2.3e-05"),
    "ort_cuda_fp32": ("fp32", "2.1e-05", "2.74e-05"),
    "torch_cuda_fp32": ("fp32", "1.9e-05", "2.0e-05"),
    "ort_trt_fp16": ("fp16", "0.03", "0.035"),
    "ort_trt_fp32": ("fp32", "2.2e-05", "2.6e-05"),
}


def synthetic_p99(d: int) -> float:
    """A batch-1 p99 (ms wall clock) whose ceil over 33.333 ms is ``d`` (P3-D1 §7's rule)."""
    return round((d - 0.4) * 33.333, 3)


def write_latency_results(
    root: Path,
    *,
    verdict: str = "supported",
    delays: Mapping[str, int] = SYNTHETIC_DELAYS,
    csv_verdict: str | None = None,
    parity_fails: bool = False,
) -> Path:
    """Synthetic Phase 9 results under ``root`` (the deploy scorer's shapes).

    Writes ``h5a.csv``, ``delay.json``, ``closed_loop.csv``, ``closed_loop/dp/<provider>.json``,
    ``parity.csv``, ``latency.csv`` and ``run_info.json``. Each arm's ``delay.json`` p99 is
    :func:`synthetic_p99` of its ``d``.

    Default: two scored providers (``ort_cuda_fp32`` passes both parities at p99 21.5 ms with 2
    closed-loop flips, ``ort_trt_fp16`` fails open loop) and one unscored. ``parity_fails``: the
    shape of Phase 9's committed result -- three scored fp32 providers pass open loop (p99
    122.9 / 52.6 / 89.4 ms) and fail closed loop (4 / 5 / 4 flips of 10, both directions), so
    the delays fall back to ``torch_cpu_fp32`` (verdict "not supported"). ``csv_verdict``
    overrides the CSV's verdict on scored rows (inconsistency).
    """
    from ila.deploy import latency as lat

    root.mkdir(parents=True, exist_ok=True)
    v = verdict if csv_verdict is None else csv_verdict
    if parity_fails:
        rows = [
            ["H5a", "dp_e2e", "ort_cpu_fp32", "True", "1", "timed", "111.6", "122.911", "128.6",
             "True", "False", "False", "33.3", "False", v],
            ["H5a", "dp_e2e", "ort_cuda_fp32", "True", "1", "timed", "48.3", "52.6441", "54.2",
             "True", "False", "False", "33.3", "False", v],
            ["H5a", "dp_e2e", "torch_cuda_fp32", "True", "1", "timed", "80.7", "89.3851", "93.5",
             "True", "False", "False", "33.3", "False", v],
            ["H5a", "dp_e2e", "ort_trt_fp16", "True", "1", "refused (open-loop parity not passed)",
             "", "", "", "False", "", "False", "33.3", "", v],
            ["H5a", "dp_e2e", "ort_trt_fp32", "False", "1", "timed", "19.8", "25.7", "31.3",
             "True", "False", "False", "33.3", "True", "context (not scored)"],
        ]  # fmt: skip
        flights = _FLIGHTS_FAIL
        provider = {"bc_dagger": "torch_cuda_fp32", "dp": "torch_cpu_fp32"}
        fallback = {"bc_dagger": False, "dp": True}
    else:
        rows = [
            ["H5a", "dp_e2e", "ort_cuda_fp32", "True", "1", "timed", "20.1", "21.5", "30.2",
             "True", "True", "True", "33.3", "True", v],
            ["H5a", "dp_e2e", "ort_trt_fp16", "True", "1", "refused", "", "", "",
             "False", "", "False", "33.3", "", v],
            ["H5a", "dp_e2e", "ort_trt_fp32", "False", "1", "timed", "9.0", "9.9", "12.0",
             "True", "", "False", "33.3", "True", "context (not scored)"],
        ]  # fmt: skip
        flights = _FLIGHTS_OK
        provider = {"bc_dagger": "ort_cuda_fp32", "dp": "ort_cuda_fp32"}
        fallback = {"bc_dagger": False, "dp": False}
    (root / "h5a.csv").write_text(lat.h5a_csv(rows), encoding="utf-8")
    arms = {
        arm: {"d": d, "provider": provider[arm], "row": f"{arm}_e2e",
              "p99_ms": synthetic_p99(d), "fallback": fallback[arm]}
        for arm, d in delays.items()
    }  # fmt: skip
    record = {"arms": arms, "control_period_ms": 33.333, "h5a_verdict": verdict}
    (root / "delay.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commit = SYNTHETIC_LATENCY_COMMITS["latency"]
    cl = [
        "arm,provider,scored,status,n,flips,max_flips,closed_loop_pass,success,"
        "reference_equals_selection_rows,ila_commit",
        f"bc_dagger,torch_cpu_fp32,False,reference,{_CL_N},0,3,reference,9,True,{commit}",
        f"bc_dagger,ort_cuda_fp32,True,flown,{_CL_N},0,3,True,9,,{commit}",
        f"dp,torch_cpu_fp32,False,reference,{_CL_N},0,3,reference,{_CL_REF_SUCCESS},True,{commit}",
    ]
    (root / "closed_loop" / "dp").mkdir(parents=True, exist_ok=True)
    for prov, (scored, status, flipped) in flights.items():
        if status != "flown":
            cl.append(f"dp,{prov},{scored},{status},0,,3,False,,,")
            continue
        lost = sum(r == "success" and p != "success" for r, p in flipped)
        gained = sum(p == "success" and r != "success" for r, p in flipped)
        succ = _CL_REF_SUCCESS - lost + gained
        cl.append(
            f"dp,{prov},{scored},{status},{_CL_N},{len(flipped)},3,{len(flipped) <= 3},{succ},,"
            f"{commit}"
        )
        flips = [
            {"index": str(i), "provider": p, "reference": r, "ss": "SS3"}
            for i, (r, p) in enumerate(flipped)
        ]
        (root / "closed_loop" / "dp" / f"{prov}.json").write_text(
            json.dumps({"arm": "dp", "flipped": flips}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (root / "closed_loop.csv").write_text("\n".join(cl) + "\n", encoding="utf-8")
    par = ["arm,provider,precision,open_loop_pass,chunk_unrolled_max_abs,chunk_loop_max_abs"]
    for prov in flights:
        prec, unrolled, loop = _PARITY[prov]
        par.append(f"bc_dagger,{prov},{prec},{prec == 'fp32'},,")
        par.append(f"dp,{prov},{prec},{prec == 'fp32'},{unrolled},{loop}")
    (root / "parity.csv").write_text("\n".join(par) + "\n", encoding="utf-8")
    latency = ["row,provider,batch_size,threads,p99_ms,ila_commit"]
    for r in rows:
        latency.append(f"dp_e2e,{r[2]},1,1,{r[7]},{commit}")
    latency.append(f"dp_e2e,torch_cpu_fp32,1,1,{synthetic_p99(delays.get('dp', 1))},{commit}")
    latency.append(f"dp_e2e,torch_cpu_fp32,24,4,999.0,{commit}")  # batch 24: never read for d
    (root / "latency.csv").write_text("\n".join(latency) + "\n", encoding="utf-8")
    stage = {
        "export": {"ila_commit": SYNTHETIC_LATENCY_COMMITS["export"], "dirty": True,
                   "dirty_outside_results": False},
        "latency": {"ila_commit": commit, "dirty": False, "dirty_outside_results": False},
    }  # fmt: skip
    (root / "run_info.json").write_text(
        json.dumps({"stages": {k: {"git": g} for k, g in stage.items()}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


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
        latency_steps=str(job.latency_steps or 0),
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
    m = ev_arms.load_matrix(delays=SYNTHETIC_DELAYS)
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
    latency_dir: Path | None = None,
) -> sc.ScoreInputs:
    """Write every matrix job (minus ``skip``) under ``root`` and return the score inputs.

    Args:
        root: The output root (``rows/`` is written under it).
        n: Episodes per cell (the first ``n`` of each frozen-list cell).
        outcome: Synthetic outcome per (arm, seed, list, ss, index, shift, Ta).
        skip: Jobs not written (to exercise missing cells).
        matrix: The matrix (default: the frozen one, latency arm at :data:`SYNTHETIC_DELAYS`).
        latency_dir: H5(a)'s ``results/latency`` stand-in (:func:`write_latency_results`), or
            ``None`` (H5(a) not scored).

    Returns:
        :class:`ila.eval.score.ScoreInputs` with synthetic baselines and fixed H1/H4 inputs.
    """
    m = ev_arms.load_matrix(delays=SYNTHETIC_DELAYS) if matrix is None else matrix
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
        latency_dir=latency_dir,
    )


# --------------------------------------------------------------------------- bc_dart seeds 1-4

#: Failing episodes per ``bc_dart`` seed (of 10) in :func:`dart_s0_tree`: success rates
#: s0 0.9, s1 0.6, s2 0.7, s3 0.8, s4 0.9. Known answers: P8's IQM (25 % trimmed mean) is 0.75
#: on seeds 1-4 (mean of 0.7, 0.8) and 0.80 on seeds 0-4 (mean of 0.7, 0.8, 0.9); with ``bc`` at
#: 0.9 on every seed, success(bc_dart) - success(bc) is -0.15 on seeds 1-4 and -0.12 on 0-4.
DART_FAILS: Mapping[int, int] = {0: 1, 1: 4, 2: 3, 3: 2, 4: 1}

#: ``select`` successes (of 150) per seed: ``bc_dart`` leads on five seeds (146.8 vs ``bc``'s
#: 146.6) and trails on seeds 1-4 (146.0), so the H4 rule picks ``bc`` there.
DART_SELECT: Mapping[str, tuple[int, ...]] = {
    "bc": (147, 146, 147, 146, 147),
    "bc_matched": (140,) * 5,
    "bc_dart": (150, 146, 146, 146, 146),
    "bc_dagger": (140,) * 5,
    "dp": (60,) * 5,
}


def _dart_keep(job: ev_arms.Job) -> bool:
    return (
        job.tag == "core"
        and job.arm in ("bc", "bc_dart", "bc_dagger", "bc_matched")
        and job.list_name == "id"
    )


def dart_outcome(
    arm: str, seed: int, list_name: str, ss: str, i: int, shift: str | None, ta: int | None
) -> str:
    """:data:`DART_FAILS` for ``bc_dart``; one bounce (episode 5) for every other arm.

    Nothing exceeds the synthetic teacher (1.0 at SS3-SS5, 0.9 at SS6).
    """
    if arm == "bc_dart":
        return "hard_landing" if i in (3, 0, 1, 2)[: DART_FAILS[seed]] else "success"
    return "bounce" if i == 5 else "success"


def dart_select_scores() -> dict[str, hy.SelectScore]:
    """Synthetic :data:`DART_SELECT` scores (150 episodes per seed; no rows files needed)."""
    return {
        arm: hy.SelectScore(arm, hy.CORE_SEEDS, succ, (150,) * 5, ((Path("unused.csv"),),) * 5)
        for arm, succ in DART_SELECT.items()
    }


def dart_s0_tree(root: Path, *, n: int = 10) -> sc.ScoreInputs:
    """Core ``bc``, ``bc_dart``, ``bc_dagger``, ``bc_matched`` on ``id`` with known answers."""
    inp = synthetic_tree(root, n=n, outcome=dart_outcome, matrix=reduced_matrix(_dart_keep))
    scores = dart_select_scores()
    return dataclasses.replace(
        inp, select_scores=scores, best_end_to_end=hy.best_end_to_end(scores)
    )
