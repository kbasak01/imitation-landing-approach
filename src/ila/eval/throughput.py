"""Phase 2 throughput (plan Phase 2 item 4): measured costs and the wall-clock budget they imply.

Sections of ``results/throughput.csv`` (one row per measured configuration):

1. ``render``: wall time per frame at 64/96/128 px, ``camera`` (P8's scene) and ``full``
   (the scene edits), single process, a ``pid_feedforward``-flown train-pool episode.
2. ``teacher``: ``pid_feedforward`` and ``ppo`` (P8's SB3 path) flown with render ``full`` at
   96 px through :func:`ila.eval.runner.run_list_vision`, at 1/8/16/24 workers: control
   steps per second of wall clock, worker spawn included.
3. ``micro``: one inference call of the dummy (random-init) ``bc`` and ``dp`` networks
   (``ila.policies.registry.build_dummy``), batch 1/8/24, on CUDA with TF32 off, plus the
   CPU single-sample path (1 thread). ``dp`` per call is one ``sample_batch`` (K DDIM steps);
   its per-step cost is the per-call cost divided by ``Ta``.
4. ``e2e``: lockstep rendering workers + the shared-net server
   (:func:`ila.eval.batched.run_batched`), ``bc`` and ``dp`` at K in {10, 100} x Ta in {1, 8},
   batch cap B = workers = 1/8/24: env steps per second, plus the in-situ per-call latency.

Episodes are P8 ``train``-pool realizations (``ila.data.episodes.train_rows``, stream
:data:`STREAM`), never a frozen list or ``select``. ``results/throughput_projection.csv``
turns the measured rates into hours for demo collection, DAgger and the Phase 8 matrix, with
every assumption in its own column. Times are wall-clock seconds on the stated machine;
episode lengths are 30 Hz control steps, model scale (lambda = 1/25).
"""

from __future__ import annotations

import csv
import functools
import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import batched
from ila.eval.jobs import JobStatus, host_info, rel, write_json
from ila.eval.runner import RowTags, init_worker_threads, run_list_vision, set_single_thread_env
from ila.obs.split import load_contract
from ila.render.bench import render_frame_times

__all__ = ["COLUMNS", "STREAM", "Plan", "episode_lengths", "projection", "run"]

#: Draw stream of the throughput episodes (train pool).
STREAM: Final[str] = "throughput"

#: Columns of ``throughput.csv``.
COLUMNS: Final[tuple[str, ...]] = (
    "section",
    "name",
    "mode",
    "size_px",
    "device",
    "numerics",
    "workers",
    "batch",
    "K",
    "Ta",
    "n",
    "p50_ms",
    "p99_ms",
    "mean_ms",
    "episodes",
    "steps",
    "wall_s",
    "steps_per_s",
    "spawn_s",
    "steady_steps_per_s",
    "mean_batch",
    "note",
)


@dataclass(frozen=True)
class Plan:
    """What is measured (the full run, or a smoke)."""

    sizes: tuple[int, ...] = (64, 96, 128)
    render_frames: int = 300
    worker_counts: tuple[int, ...] = (1, 8, 16, 24)
    episodes_per_worker: int = 16
    teacher_chunk: int = 4
    micro_batches: tuple[int, ...] = (1, 8, 24)
    micro_reps: int = 30
    micro_reps_slow: int = 5
    dp_grid: tuple[tuple[int, int], ...] = ((10, 1), (10, 8), (100, 1), (100, 8))
    e2e_batches: tuple[int, ...] = (1, 8, 24)
    e2e_episodes_per_slot: int = 4
    cpu_micro: bool = True

    @classmethod
    def smoke(cls) -> Plan:
        """A few-second version of every section."""
        return cls(
            sizes=(64, 96),
            render_frames=20,
            worker_counts=(1, 2),
            episodes_per_worker=1,
            teacher_chunk=1,
            micro_batches=(1, 8),
            micro_reps=3,
            micro_reps_slow=1,
            dp_grid=((10, 8),),
            e2e_batches=(1, 2),
            e2e_episodes_per_slot=1,
            cpu_micro=False,
        )


def _row(**kw: Any) -> dict[str, Any]:
    out = dict.fromkeys(COLUMNS, "")
    out.update(kw)
    return out


def _ms(times_s: Sequence[float] | NDArray[np.float64]) -> dict[str, float]:
    t = np.asarray(times_s, dtype=np.float64) * 1e3
    return {
        "p50_ms": round(float(np.quantile(t, 0.5)), 4),
        "p99_ms": round(float(np.quantile(t, 0.99)), 4),
        "mean_ms": round(float(t.mean()), 4),
        "n": int(t.size),
    }


# --------------------------------------------------------------------------- sections


def _render_rows(
    cfgs: p8_adapter.EvalConfigs, listed: p8_adapter.ListedEpisode, plan: Plan
) -> list[dict[str, Any]]:
    rows = []
    for size in plan.sizes:
        for mode in ("camera", "full"):
            policy = p8_adapter.controller_spec("pid_feedforward").build(cfgs)
            times = render_frame_times(
                cfgs, listed, policy, mode=mode, size_px=size, n_frames=plan.render_frames
            )
            rows.append(
                _row(
                    section="render",
                    name="tinyrenderer_frame",
                    mode=mode,
                    size_px=size,
                    device="cpu",
                    workers=1,
                    note="single process; frame time only",
                    **_ms(times),
                )
            )
    return rows


def _warm(i: int) -> int:
    """Pool task for the spawn-overhead measurement (the worker has imported this module)."""
    return i


def spawn_overhead_s(workers: int) -> float:
    """Wall seconds to spawn a pool of ``workers`` that imports what runner workers import.

    ``ila.eval.throughput`` imports the runner, the adapter (``rld``, pybullet, SB3/torch) and
    the render stack, i.e. what a ``run_chunk_vision`` worker unpickles; ``init_worker_threads``
    is the same initializer. 0 for one worker (the in-process path spawns nothing).
    """
    if workers <= 1:
        return 0.0
    import multiprocessing as mp

    start = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=workers, initializer=init_worker_threads) as pool:
        pool.map(_warm, range(workers), chunksize=1)
    return time.perf_counter() - start


def _teacher_spec(name: str) -> p8_adapter.PolicySpec:
    if name == "ppo":
        return p8_adapter.callable_spec(
            "ppo", functools.partial(p8_adapter.teacher, "ppo", 0), run_seed=0
        )
    return p8_adapter.controller_spec(name)


def _teacher_rows(
    cfgs: p8_adapter.EvalConfigs,
    episodes: Sequence[p8_adapter.ListedEpisode],
    plan: Plan,
    tick: Callable[[str], None],
) -> list[dict[str, Any]]:
    rows = []
    for name in ("pid_feedforward", "ppo"):
        for workers in plan.worker_counts:
            eps = list(episodes[: workers * plan.episodes_per_worker])
            tick(f"teacher {name} workers={workers}")
            start = time.perf_counter()
            out = run_list_vision(
                eps,
                _teacher_spec(name),
                cfgs,
                render="full",
                workers=workers,
                chunk=plan.teacher_chunk,
            )
            wall = time.perf_counter() - start
            steps = sum(int(r["steps"]) for r in out)
            spawn = spawn_overhead_s(min(workers, math.ceil(len(eps) / plan.teacher_chunk)))
            rows.append(
                _row(
                    section="teacher",
                    spawn_s=round(spawn, 3),
                    steady_steps_per_s=round(steps / max(wall - spawn, 1e-9), 2),
                    name=name,
                    mode="full",
                    size_px=96,
                    device="cpu",
                    workers=workers,
                    episodes=len(out),
                    steps=steps,
                    wall_s=round(wall, 3),
                    steps_per_s=round(steps / wall, 2),
                    note=f"run_list_vision chunk {plan.teacher_chunk}; steps_per_s includes "
                    "spawn; steady = steps / (wall - separately measured spawn_s)",
                )
            )
    return rows


def _random_inputs(batch: int) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    contract = load_contract()
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (batch, contract.n_obs_steps, 96, 96, 3), dtype=np.uint8)
    proprio = rng.normal(size=(batch, contract.n_obs_steps, contract.proprio_dim))
    return images, proprio.astype(np.float32)


def _time_calls(call: Callable[[int], Any], reps: int, warmup: int = 2) -> list[float]:
    for i in range(warmup):
        call(10_000 + i)  # warm-up keys never reused by the timed calls
    times = []
    for i in range(reps):
        start = time.perf_counter()
        call(i)
        times.append(time.perf_counter() - start)
    return times


def _bc_call(student: Any, images: Any, proprio: Any, i: int) -> Any:
    del i
    return student.act_batch(images, proprio)


def _dp_call(student: Any, images: Any, proprio: Any, b: int, i: int) -> Any:
    return student.sample_batch(images, proprio, [(0, j, i) for j in range(b)])


def _micro_rows(plan: Plan, tick: Callable[[str], None]) -> list[dict[str, Any]]:
    import torch

    from ila.policies.registry import build_dummy

    rows = []
    configs: list[tuple[str, str, int, int | None]] = []  # (device, arm, batch, K)
    if torch.cuda.is_available():
        configs += [("cuda", "bc", b, None) for b in plan.micro_batches]
        configs += [
            ("cuda", "dp", b, k)
            for k in sorted({k for k, _ in plan.dp_grid})
            for b in plan.micro_batches
        ]
    if plan.cpu_micro:
        configs += [("cpu", "bc", 1, None), ("cpu", "dp", 1, 10), ("cpu", "dp", 1, 100)]
    for device, arm, b, k in configs:
        tick(f"micro {device} {arm} B={b} K={k}")
        flags = batched.configure_torch_numerics(device, "strict")
        images, proprio = _random_inputs(b)
        call: Callable[[int], Any]
        if arm == "bc":
            call = functools.partial(_bc_call, build_dummy("bc", 0, device), images, proprio)
            reps = plan.micro_reps
        else:
            student = build_dummy("dp", 0, device, num_inference_steps=k)
            call = functools.partial(_dp_call, student, images, proprio, b)
            reps = plan.micro_reps if (k or 0) <= 10 and device == "cuda" else plan.micro_reps_slow
        times = _time_calls(call, reps)
        rows.append(
            _row(
                section="micro",
                name=f"dummy_{arm}",
                device=device,
                numerics=flags["numerics"],
                batch=b,
                K=k if k is not None else "",
                note=(
                    "one sample_batch call (one chunk per sample)"
                    if arm == "dp"
                    else "one act_batch call"
                )
                + ("; torch threads 1" if device == "cpu" else "; TF32 off"),
                **_ms(times),
            )
        )
    return rows


def _e2e_rows(
    cfgs: p8_adapter.EvalConfigs,
    episodes: Sequence[p8_adapter.ListedEpisode],
    plan: Plan,
    tick: Callable[[str], None],
) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return [_row(section="e2e", note="skipped: no CUDA device")]
    rows = []
    grid: list[tuple[str, int | None, int | None]] = [("bc", None, None)]
    grid += [("dp", k, ta) for k, ta in plan.dp_grid]
    for arm, k, ta in grid:
        for b in plan.e2e_batches:
            tick(f"e2e {arm} K={k} Ta={ta} B={b}")
            eps = list(episodes[: b * plan.e2e_episodes_per_slot])
            spec = p8_adapter.callable_spec(
                f"dummy_{arm}",
                functools.partial(batched.build_dummy_student, arm, 0, "cpu", ta, k),
                run_seed=0,
            )
            inner = batched.shared_net_factory(
                arm, 0, "cuda", n_action_steps=ta, num_inference_steps=k
            )
            built: list[batched.TimedPolicy] = []

            def server(
                c: p8_adapter.EvalConfigs, n: int, inner: Any = inner, built: Any = built
            ) -> Any:
                timed = batched.TimedPolicy(inner(c, n))
                built.append(timed)
                return timed

            arm_spec = batched.BatchedArm(
                spec=spec,
                episodes=eps,
                tags=RowTags(arm=f"dummy_{arm}", Ta=ta, K=k),
                server_build=server,
                batch_cap=b,
            )
            start = time.perf_counter()
            out = batched.run_batched(arm_spec, cfgs, workers=b, chunk=plan.e2e_episodes_per_slot)
            wall = time.perf_counter() - start
            steps = sum(int(r["steps"]) for r in out)
            timed = built[0]
            calls = timed.calls
            call_times = [t for _, t in calls] or [float("nan")]
            window = (
                (timed.last_end - timed.first_start)
                if timed.first_start is not None and timed.last_end is not None
                else float("nan")
            )
            rows.append(
                _row(
                    section="e2e",
                    name=f"dummy_{arm}",
                    mode="full",
                    size_px=96,
                    device="cuda",
                    numerics="strict",
                    workers=b,
                    batch=b,
                    K=k if k is not None else "",
                    Ta=ta if ta is not None else "",
                    episodes=len(out),
                    steps=steps,
                    wall_s=round(wall, 3),
                    steps_per_s=round(steps / wall, 2),
                    spawn_s=round(wall - window, 3),
                    steady_steps_per_s=round(steps / window, 2),
                    mean_batch=round(float(np.mean([n for n, _ in calls])), 2) if calls else "",
                    note="lockstep workers=B; p50/p99 = server act_batch call; steady = steps / "
                    "(first call start .. last call end); "
                    f"mean episode {steps / max(len(out), 1):.1f} steps (random-init net)",
                    **_ms(call_times),
                )
            )
    return rows


# --------------------------------------------------------------------------- projection


def _rows_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(p8_adapter.read_text(path))))


def episode_lengths() -> dict[str, float]:
    """Mean control steps per episode from P8's committed rows (the projection's lengths).

    Returns:
        ``ppo_train_like`` (e07 ppo seed 0, ``id`` SS3-SS5), ``ppo_all`` (all 2 800),
        ``ppo_id_ss4_ss6``, ``ppo_id_ss5``, ``ppo_id_ss5_ss6``, ``pid_train_like`` and
        ``gated_train_like`` (e01, ``id`` SS3-SS5), steps.
    """
    e07 = [
        r
        for r in _rows_csv(p8_adapter.P8_ROOT / "results/e07/matrix/episodes.csv")
        if r["method"] == "ppo" and r["run_seed"] == "0"
    ]
    e01 = _rows_csv(p8_adapter.P8_ROOT / "results/e01/episodes.csv")

    def mean(rows: Sequence[dict[str, str]]) -> float:
        return float(np.mean([int(r["steps"]) for r in rows]))

    def cell(
        rows: Sequence[dict[str, str]], regime: str, sss: Sequence[str]
    ) -> list[dict[str, str]]:
        return [r for r in rows if r["regime"] == regime and r["ss"] in sss]

    train_like = ("SS3", "SS4", "SS5")
    return {
        "ppo_train_like": mean(cell(e07, "id", train_like)),
        "ppo_all": mean(e07),
        "ppo_id_ss4_ss6": mean(cell(e07, "id", ("SS4", "SS5", "SS6"))),
        "ppo_id_ss5": mean(cell(e07, "id", ("SS5",))),
        "ppo_id_ss5_ss6": mean(cell(e07, "id", ("SS5", "SS6"))),
        "pid_train_like": mean(
            [r for r in cell(e01, "id", train_like) if r["method"] == "pid_feedforward"]
        ),
        "gated_train_like": mean(
            [r for r in cell(e01, "id", train_like) if r["method"] == "gated"]
        ),
    }


def _rate(rows: Sequence[dict[str, Any]], section: str, name: str, **match: Any) -> float:
    for r in rows:
        if (
            r["section"] == section
            and r["name"] == name
            and all(str(r[k]) == str(v) for k, v in match.items())
        ):
            steady = r.get("steady_steps_per_s", "")
            return float(steady) if steady not in ("", None) else float(r["steps_per_s"])
    return float("nan")


#: Upper-bound episode length: every episode a timeout (12 s model + 1 step), steps.
TIMEOUT_STEPS: Final[int] = 361


def projection(rows: Sequence[dict[str, Any]], lengths: dict[str, float]) -> list[dict[str, Any]]:
    """Project wall-clock hours per task from the measured rates (see each row's assumptions).

    Rates are the steady-state (spawn excluded) largest-worker measurements (24 workers /
    B = 24, or the largest measured in a smoke). Hours = episodes x steps per episode / rate /
    3600. Training time is never included. ``hours_upper`` uses :data:`TIMEOUT_STEPS` for
    student episodes.
    """
    w_max = max(int(r["workers"]) for r in rows if r["section"] == "teacher")
    b_max = max(
        (int(r["batch"]) for r in rows if r["section"] == "e2e" and r["batch"] != ""), default=0
    )
    r_pid = _rate(rows, "teacher", "pid_feedforward", workers=w_max)
    r_ppo = _rate(rows, "teacher", "ppo", workers=w_max)
    r_bc = _rate(rows, "e2e", "dummy_bc", batch=b_max)

    def r_dp(k: int, ta: int) -> float:
        got = _rate(rows, "e2e", "dummy_dp", batch=b_max, K=k, Ta=ta)
        return (
            got if not math.isnan(got) else _rate(rows, "e2e", "dummy_dp", batch=b_max, K=10, Ta=8)
        )

    src_t = f"teacher-with-render, {w_max} workers, 96 px"
    src_b = f"e2e lockstep B={b_max} (dummy net)"
    tasks: list[tuple[str, int, float, float | None, float, str, str]] = []
    # (task, episodes, steps/episode, upper steps/episode, rate, rate source, assumptions)
    lens = lengths
    tasks += [
        (
            "demo ppo_2000",
            2000,
            lens["ppo_train_like"],
            None,
            r_ppo,
            f"ppo {src_t}",
            "episode length = P8 e07 ppo s0 mean on id SS3-SS5; HDF5 write not timed",
        ),
        (
            "demo pid_1000",
            1000,
            lens["pid_train_like"],
            None,
            r_pid,
            f"pid {src_t}",
            "length = P8 e01 pid_feedforward mean on id SS3-SS5",
        ),
        (
            "demo mix_1000 (ppo half)",
            500,
            lens["ppo_train_like"],
            None,
            r_ppo,
            f"ppo {src_t}",
            "T_mix per-episode 0.5 ppo",
        ),
        (
            "demo mix_1000 (gated half)",
            500,
            lens["gated_train_like"],
            None,
            r_pid,
            f"pid {src_t}",
            "T_mix 0.5 gated: length = e01 gated id SS3-SS5; rate = pid_feedforward's (controller)",
        ),
        (
            "demo DART 3 sigma x 1000",
            3000,
            lens["ppo_train_like"],
            None,
            r_ppo,
            f"ppo {src_t}",
            "DART noise assumed not to change episode length",
        ),
        (
            "DAgger rollouts 5 seeds x 5 it x 200",
            5000,
            lens["ppo_train_like"],
            TIMEOUT_STEPS,
            r_bc,
            f"bc {src_b}",
            "student length = teacher's (upper: all timeouts); teacher label "
            "(ppo MLP, CPU) assumed negligible; training excluded",
        ),
    ]
    core = {
        "bc_state": (r_ppo, f"ppo {src_t}", "state MLP: cost like the teacher, rendered"),
        "bc": (r_bc, f"bc {src_b}", ""),
        "bc_matched": (r_bc, f"bc {src_b}", ""),
        "bc_dart": (r_bc, f"bc {src_b}", ""),
        "bc_dagger": (r_bc, f"bc {src_b}", ""),
        "dp": (r_dp(10, 8), f"dp K10 Ta8 {src_b}", ""),
        "modular": (r_bc, f"bc {src_b}", "estimator assumed to cost like the bc encoder"),
    }
    for arm, (rate, src, note) in core.items():
        tasks.append(
            (
                f"matrix core {arm} 5 seeds x 2800",
                14000,
                lens["ppo_all"],
                TIMEOUT_STEPS,
                rate,
                src,
                "length = P8 ppo s0 mean over all 2800 (upper: all timeouts)"
                + (f"; {note}" if note else ""),
            )
        )
    n_abl_bc, n_abl_dp = 6, 5  # configs/eval/matrix.yaml ablations by arm
    tasks += [
        (
            "matrix ablations bc (6) x 3 seeds x 600",
            n_abl_bc * 3 * 600,
            lens["ppo_id_ss4_ss6"],
            TIMEOUT_STEPS,
            r_bc,
            f"bc {src_b}",
            "id SS4-SS6; length = ppo s0 mean there",
        ),
        (
            "matrix ablations dp (5) x 3 seeds x 600",
            n_abl_dp * 3 * 600,
            lens["ppo_id_ss4_ss6"],
            TIMEOUT_STEPS,
            r_dp(10, 8),
            f"dp K10 Ta8 {src_b}",
            "id SS4-SS6",
        ),
        (
            "sweep dp_Ta (Ta<8 or ensembling: 6 configs) x 5 seeds x 600",
            6 * 5 * 600,
            lens["ppo_id_ss4_ss6"],
            TIMEOUT_STEPS,
            r_dp(10, 1),
            f"dp K10 Ta1 {src_b}",
            "Ta in {1,2,4} and every ensembling config costed at the Ta=1 rate (conservative)",
        ),
        (
            "sweep dp_Ta (Ta>=8 no ensembling: 2 configs) x 5 seeds x 600",
            2 * 5 * 600,
            lens["ppo_id_ss4_ss6"],
            TIMEOUT_STEPS,
            r_dp(10, 8),
            f"dp K10 Ta8 {src_b}",
            "Ta in {8,12}",
        ),
        (
            "sweep dp_K (K<=10: 4) x 5 seeds x 200",
            4 * 5 * 200,
            lens["ppo_id_ss5"],
            TIMEOUT_STEPS,
            r_dp(10, 8),
            f"dp K10 Ta8 {src_b}",
            "id SS5",
        ),
        (
            "sweep dp_K (K 25 and 100: 2) x 5 seeds x 200",
            2 * 5 * 200,
            lens["ppo_id_ss5"],
            TIMEOUT_STEPS,
            r_dp(100, 8),
            f"dp K100 Ta8 {src_b}",
            "K=25 costed as K=100",
        ),
        (
            "appearance shift 3 arms x 5 variants x 5 seeds x 400",
            3 * 5 * 5 * 400,
            lens["ppo_id_ss5_ss6"],
            TIMEOUT_STEPS,
            r_dp(10, 8),
            f"dp K10 Ta8 {src_b}",
            "costed at the dp rate (slowest of bc_dagger/dp/modular)",
        ),
        (
            "latency arm 2 arms x 5 seeds x 600",
            2 * 5 * 600,
            lens["ppo_id_ss4_ss6"],
            TIMEOUT_STEPS,
            r_dp(10, 8),
            f"dp K10 Ta8 {src_b}",
            "costed at the dp rate",
        ),
    ]
    out: list[dict[str, Any]] = []
    for task, n, length, upper, rate, src, note in tasks:
        hours = n * length / rate / 3600.0
        hours_up = "" if upper is None else round(n * upper / rate / 3600.0, 3)
        flag = (
            "FLAG >24 h"
            if hours > 24.0 or (upper is not None and n * upper / rate / 3600 > 24)
            else ""
        )
        out.append(
            {
                "task": task,
                "episodes": n,
                "steps_per_episode": round(length, 2),
                "steps_per_episode_upper": upper if upper is not None else "",
                "rate_steps_per_s": round(rate, 2),
                "rate_source": src,
                "hours": round(hours, 3),
                "hours_upper": hours_up,
                "flag": flag,
                "assumptions": note,
            }
        )
    total = float(sum(float(r["hours"]) for r in out))
    out.append({"task": "TOTAL (central)", "hours": round(total, 3), "assumptions": "sum of hours"})
    return out


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return buffer.getvalue()


PROJECTION_COLUMNS: Final[tuple[str, ...]] = (
    "task",
    "episodes",
    "steps_per_episode",
    "steps_per_episode_upper",
    "rate_steps_per_s",
    "rate_source",
    "hours",
    "hours_upper",
    "flag",
    "assumptions",
)


def run(*, out_dir: Path, run_dir: Path, smoke: bool = False) -> dict[str, Any]:
    """Measure every section, write the CSVs, the projection and ``run_info.json``.

    Args:
        out_dir: Where ``throughput.csv``, ``throughput_projection.csv`` and the run info go.
        run_dir: Where ``status.json`` goes.
        smoke: Run :meth:`Plan.smoke` instead of the full plan.

    Returns:
        The run info record.
    """
    plan = Plan.smoke() if smoke else Plan()
    status = JobStatus(run_dir, "throughput" + ("_smoke" if smoke else ""))
    try:
        threads = set_single_thread_env()
        init_worker_threads()
        cfgs = p8_adapter.load_eval_configs()
        started = time.perf_counter()

        def tick(what: str) -> None:
            status.update(
                "running",
                progress={"section": what, "elapsed_s": round(time.perf_counter() - started, 1)},
            )

        tick("drawing train-pool episodes")
        n_eps = max(
            max(plan.worker_counts) * plan.episodes_per_worker,
            max(plan.e2e_batches) * plan.e2e_episodes_per_slot,
            1,
        )
        split_cfg = data_episodes.load_splits_config()
        api = data_episodes.p8_episode_api(cfgs, workers=8)
        episodes = data_episodes.train_rows(api, split_cfg, n_eps, stream=STREAM)
        rows: list[dict[str, Any]] = []
        tick("render")
        rows += _render_rows(cfgs, episodes[0], plan)
        rows += _teacher_rows(cfgs, episodes, plan, tick)
        rows += _micro_rows(plan, tick)
        rows += _e2e_rows(cfgs, episodes, plan, tick)
        text = _write_csv(out_dir / "throughput.csv", rows, COLUMNS)
        lengths = episode_lengths()
        proj = projection(rows, lengths)
        _write_csv(out_dir / "throughput_projection.csv", proj, PROJECTION_COLUMNS)
        import hashlib

        info = {
            "item": "plan Phase 2 item 4 (throughput)",
            "smoke": smoke,
            "plan": {k: getattr(plan, k) for k in plan.__dataclass_fields__},
            "wall_s": round(time.perf_counter() - started, 1),
            "episodes": {
                "pool": "P8 dev_pool train",
                "stream": STREAM,
                "generator_seed": split_cfg.generator_seed,
                "n": len(episodes),
                "content_sha256": data_episodes.content_sha256(episodes),
            },
            "episode_lengths_from_p8_rows": lengths,
            "threads": {**threads, "torch_num_threads": 1},
            "server_numerics": batched.configure_torch_numerics("cuda", "strict"),
            "throughput_csv_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "out_dir": rel(out_dir),
            **host_info(),
        }
        write_json(out_dir / "throughput_run_info.json", info)
        status.update("done", progress={"elapsed_s": info["wall_s"]})
        return info
    except BaseException as exc:
        status.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise
