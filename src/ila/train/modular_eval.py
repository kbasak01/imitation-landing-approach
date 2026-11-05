"""``modular``'s per-field estimation error (plan Phase 7 item 1; P7-D2 data choice).

For each run's **selected** checkpoint (``selection.json``; EMA weights), the error of the
de-normalised estimate ``priv_hat`` against the true privileged block, per field
(``rel_position``, ``rel_velocity``, ``deck_normal``, ``rel_tilt``, ``height``), per component and,
for the 3-vectors, of the error vector's Euclidean norm:

* **open loop** on :data:`ila.train.loop.HELDOUT` (``ppo_2000`` stream indices 1950-1999,
  ``train`` pool, success-filtered, outside ``modular``'s first 1 000; user decision
  2026-11-02, P7-D2, replacing the plan's "``tune`` demos"): every stored frame, the estimator
  on the CPU in fp32 (eval mode, centre crop), truth = the stored ``priv[t]``;
* **closed loop** on the 150 ``select`` episodes, re-flown on the CPU fp32 single-sample path
  exactly as selection flew them (``act_modular``, P8's frozen ``T_ppo`` given the estimate and
  the true ``time_fraction``), recording at every control step the estimate the action was
  computed from and the true privileged fields of the same observation. The truth is read by
  :func:`ila.train.modular_targets.true_priv` (by field name) **for this log only**; it is
  never passed to the student. A diagnostic: it chooses nothing.

Statistics per (mode, sea state or ``all``, field, component): ``n`` steps, ``bias`` (mean
error), ``rmse``, ``p95_abs`` (95th percentile of ``|error|``, or of the norm; linear
interpolation). Units are model scale (lambda = 1/25): ``rel_position`` and ``height`` m
(full scale x 25), ``rel_velocity`` m/s (full scale x 5 = x 25 lengths / x 5 times),
``deck_normal`` dimensionless (unit vector, drone-yaw frame), ``rel_tilt`` **degrees**
(converted from P8's radians at this report boundary). Full-scale columns are given beside.

Comparison with P8's perception stand-in (P8 P7-D4, ``rld.envs.noise``): zero-mean Gaussian
noise with sigma_p = 0.04 m per world axis on the pad position and sigma_v = 0.2 m/s per world
axis on the pad velocity (model scale), under which P8's PID baselines collapse.
``rel_position`` components and ``height`` are compared with sigma_p, ``rel_velocity``
components with sigma_v; a norm with ``sqrt(3) sigma`` (the RMS norm of isotropic per-axis
noise). Caveats stated with the table: P8 adds noise in the **world** frame while the estimate
is in the drone-yaw frame (z is shared; x/y are rotated, the norm is frame-free), P8's noise is
white and zero-mean while estimation error is correlated in time and can be biased, and P8
also delays the deck sample (latency, hold), which is not modelled here.
"""

from __future__ import annotations

import csv
import functools
import io
import math
import multiprocessing as mp
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.data.store import DemoStore
from ila.eval import runner
from ila.obs.split import ObsContract, load_contract
from ila.policies.modular import ModularEstimator, ModularStudent
from ila.policies.registry import build_net_from_config, read_checkpoint
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import sha256_file
from ila.train.modular_targets import EstimatorDataset, true_priv
from ila.train.sampling import choose_episodes
from ila.train.select import SELECT_SPLIT, checkpoint_policy, summarise_rows

__all__ = [
    "CSV_COLUMNS",
    "FIELDS",
    "SIGMA_P_M",
    "SIGMA_V_M_S",
    "ClosedLoopRecord",
    "FieldInfo",
    "closed_loop",
    "error_rows",
    "open_loop",
    "selected_checkpoint",
    "write_study",
]

#: P8's perception stand-in noise (P8 P7-D4): pad position, m model scale, per world axis.
SIGMA_P_M: Final[float] = 0.04

#: P8's perception stand-in noise: pad velocity, m/s model scale, per world axis.
SIGMA_V_M_S: Final[float] = 0.2

#: Froude factors, lambda = 1/25: lengths x 25, times x 5, velocities x 5.
_LENGTH_FS: Final[float] = 25.0
_VELOCITY_FS: Final[float] = 5.0


@dataclass(frozen=True)
class FieldInfo:
    """How one privileged field is reported.

    Attributes:
        components: Component labels (``x, y, z`` drone-yaw frame, or ``value``).
        unit: Report unit (model scale).
        to_unit: Factor from P8's unit to ``unit`` (radians -> degrees for ``rel_tilt``).
        full_scale: Factor from model to full scale.
        sigma_name: P8 noise compared against (``sigma_p``/``sigma_v``), or ``""``.
        sigma: Its value in ``unit`` (per axis), or NaN.
    """

    components: tuple[str, ...]
    unit: str
    to_unit: float
    full_scale: float
    sigma_name: str
    sigma: float


#: The privileged fields, in contract order.
FIELDS: Final[Mapping[str, FieldInfo]] = {
    "rel_position": FieldInfo(("x", "y", "z"), "m", 1.0, _LENGTH_FS, "sigma_p", SIGMA_P_M),
    "rel_velocity": FieldInfo(("x", "y", "z"), "m/s", 1.0, _VELOCITY_FS, "sigma_v", SIGMA_V_M_S),
    "deck_normal": FieldInfo(("x", "y", "z"), "1", 1.0, 1.0, "", math.nan),
    "rel_tilt": FieldInfo(("value",), "deg", 180.0 / math.pi, 1.0, "", math.nan),
    "height": FieldInfo(("value",), "m", 1.0, _LENGTH_FS, "sigma_p", SIGMA_P_M),
}

#: Columns of ``results/phase7/modular_estimation.csv``.
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "run_seed",
    "checkpoint",
    "checkpoint_sha256",
    "selected",
    "mode",
    "group",
    "field",
    "component",
    "unit",
    "n",
    "bias",
    "rmse",
    "p95_abs",
    "full_scale_factor",
    "rmse_full_scale",
    "p95_abs_full_scale",
    "sigma_name",
    "sigma",
    "rmse_over_sigma",
    "p95_over_sigma",
)


def error_rows(
    estimate: NDArray[np.floating[Any]],
    truth: NDArray[np.floating[Any]],
    contract: ObsContract,
) -> list[dict[str, Any]]:
    """Per-field, per-component and norm error statistics (module docstring).

    Args:
        estimate: ``[N, 11]``, P8 units, contract order.
        truth: ``[N, 11]``, same.
        contract: The default contract (field sizes by name).

    Returns:
        One dict per (field, component) with keys ``field``, ``component``, ``unit``, ``n``,
        ``bias``, ``rmse``, ``p95_abs``, the full-scale and sigma columns of
        :data:`CSV_COLUMNS`.

    Raises:
        ValueError: On mismatched shapes or no rows.
    """
    est = np.asarray(estimate, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    if est.shape != tru.shape or est.ndim != 2 or est.shape[1] != contract.priv_dim:
        raise ValueError(f"estimate {est.shape} and truth {tru.shape} must be [N, 11]")
    if est.shape[0] == 0:
        raise ValueError("no rows")
    out = []
    col = 0
    for name in contract.privileged:
        info = FIELDS[name]
        size = contract.layout[name][1]
        if size != len(info.components):
            raise ValueError(f"{name}: {size} components, expected {len(info.components)}")
        err = (est[:, col : col + size] - tru[:, col : col + size]) * info.to_unit
        col += size
        parts: list[tuple[str, NDArray[np.float64], float]] = [
            (c, err[:, i], info.sigma) for i, c in enumerate(info.components)
        ]
        if size > 1:
            parts.append(("norm", np.linalg.norm(err, axis=1), info.sigma * math.sqrt(size)))
        for comp, e, sigma in parts:
            rmse = float(np.sqrt(np.mean(e**2)))
            p95 = float(np.quantile(np.abs(e), 0.95))
            out.append(
                {
                    "field": name,
                    "component": comp,
                    "unit": info.unit,
                    "n": int(e.size),
                    "bias": float(np.mean(e)) if comp != "norm" else math.nan,
                    "rmse": rmse,
                    "p95_abs": p95,
                    "full_scale_factor": info.full_scale,
                    "rmse_full_scale": rmse * info.full_scale,
                    "p95_abs_full_scale": p95 * info.full_scale,
                    "sigma_name": info.sigma_name
                    if comp != "norm"
                    else (f"sqrt({size})*{info.sigma_name}" if info.sigma_name else ""),
                    "sigma": sigma,
                    "rmse_over_sigma": rmse / sigma if math.isfinite(sigma) else math.nan,
                    "p95_over_sigma": p95 / sigma if math.isfinite(sigma) else math.nan,
                }
            )
    return out


def _grouped(
    estimate: NDArray[np.float32], truth: NDArray[np.float32], groups: Sequence[str]
) -> list[dict[str, Any]]:
    """:func:`error_rows` pooled (``group = all``) and per group label (sea state)."""
    contract = load_contract()
    labels = np.asarray(groups)
    out = [{"group": "all", **r} for r in error_rows(estimate, truth, contract)]
    for g in sorted(set(labels.tolist())):
        mask = labels == g
        out += [{"group": g, **r} for r in error_rows(estimate[mask], truth[mask], contract)]
    return out


# --------------------------------------------------------------------------- checkpoints


def selected_checkpoint(run_dir: Path) -> tuple[Path, str]:
    """Return a run's selected checkpoint and its SHA-256 (re-checked) from ``selection.json``.

    Raises:
        ValueError: If the run is not ``modular``, has no selection, or the bytes changed.
    """
    sel = mf.read_json(run_dir / "selection.json")
    if sel.get("arm") != "modular":
        raise ValueError(f"{run_dir}: arm {sel.get('arm')!r}, not modular")
    cand = sel["selected"]["candidate"]
    path = Path(cand["file"])
    if not path.is_absolute():
        path = run_dir / "checkpoints" / path.name
    if sha256_file(path) != cand["sha256"]:
        raise ValueError(f"{path}: SHA-256 differs from selection.json")
    return path, str(cand["sha256"])


def _estimator(path: Path) -> tuple[ModularEstimator, dict[str, Any]]:
    """The checkpoint's EMA estimator on the CPU, eval mode (no teacher needed open loop)."""
    ckpt = read_checkpoint(path)
    if ckpt["kind"] != "modular":
        raise ValueError(f"{path}: kind {ckpt['kind']!r}, not modular")
    net = build_net_from_config("modular", ckpt["config"], int(ckpt["run_seed"]))
    assert isinstance(net, ModularEstimator)
    net.load_state_dict(ckpt["ema"])
    return net.eval(), ckpt


# --------------------------------------------------------------------------- open loop


def open_loop(
    path: Path,
    data_root: Path,
    heldout: loop.HeldoutSpec = loop.HELDOUT,
    batch: int = 256,
) -> dict[str, Any]:
    """Open-loop error on the held-out demos (CPU fp32, eval mode, centre crop).

    Args:
        path: A ``modular`` checkpoint (EMA weights used).
        data_root: ``${data_root}`` (the store under ``demos/``).
        heldout: The held-out slice (default :data:`ila.train.loop.HELDOUT`).
        batch: Samples per forward (memory only).

    Returns:
        ``{"estimate", "truth"}`` (``float32[N, 11]``), ``groups`` (sea state per frame),
        ``episodes`` (kept episode accounting).
    """
    net, _ = _estimator(path)
    contract = load_contract()
    store_file = loop.store_path(data_root, heldout.collection)
    split_rows = data_episodes.read_split("demos_0-1999")
    with DemoStore(store_file) as store:
        choice = choose_episodes(store, heldout.n, True, start=heldout.start)
        data = EstimatorDataset.from_store(store, choice.names, contract.n_obs_steps)
    ss = np.repeat(
        np.asarray([split_rows[i].ss for i in choice.demo_indices]), data.table.episode_len
    )
    est = []
    with torch.inference_mode():
        for start in range(0, len(data), batch):
            idx = np.arange(start, min(len(data), start + batch), dtype=np.int64)
            b = data.batch(idx)
            z = net(torch.from_numpy(b["images"]), torch.from_numpy(b["proprio"]))
            est.append(net.denormalise(z).numpy().astype(np.float32))
    return {
        "estimate": np.concatenate(est),
        "truth": data.targets,
        "groups": ss.tolist(),
        "episodes": {**choice.summary(), "frames": len(data)},
        "store": str(store_file),
    }


# --------------------------------------------------------------------------- closed loop


@dataclass(frozen=True)
class ClosedLoopRecord:
    """One control step of a closed-loop flight.

    Attributes:
        episode_seed: P8 episode seed.
        ss: Sea state.
        step: Control step t (30 Hz, model scale).
        estimate: ``float32[11]``, the estimate the step's action was computed from.
        truth: ``float32[11]``, the observation's true privileged fields (logged only).
    """

    episode_seed: int
    ss: str
    step: int
    estimate: NDArray[np.float32]
    truth: NDArray[np.float32]


class _RecordingActor(runner.StudentActor):
    """The runner's student actor, plus a log of (estimate, truth) per step.

    The student gets exactly what :class:`ila.eval.runner.StudentActor` gives it (a
    ``ModularInput`` built by the contract); the truth is read from the observation after the
    action is computed and goes to ``records`` only.
    """

    def __init__(
        self, policy: ModularStudent, contract: ObsContract, records: list[ClosedLoopRecord]
    ) -> None:
        super().__init__(policy, contract)
        if not self.modular:
            raise TypeError("the estimation study flies a modular student")
        self._student = policy
        self._records = records
        self._listed: p8_adapter.ListedEpisode | None = None

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Start an episode (the student is reset by the base class)."""
        super().begin(listed, context)
        self._listed = listed

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        """Act through ``act_modular``, then log the estimate used and the true fields."""
        action = super().act(step, obs, frame)
        est = self._student.last_estimate
        assert est is not None and self._listed is not None
        self._records.append(
            ClosedLoopRecord(
                episode_seed=int(self._listed.episode_seed),
                ss=str(self._listed.ss),
                step=step,
                estimate=np.array(est, dtype=np.float32, copy=True),
                truth=true_priv(self.window.contract, np.asarray(obs, np.float32)),
            )
        )
        return action


def _fly_recording(task: runner.VisionTask) -> tuple[list[dict[str, Any]], list[ClosedLoopRecord]]:
    """Fly one chunk with a recording actor (picklable pool entry point)."""
    records: list[ClosedLoopRecord] = []

    def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Any, object]:
        policy: object = task.spec.build(cfgs)
        if not isinstance(policy, ModularStudent):
            raise TypeError(f"{task.spec.method}: not a modular student")
        return _RecordingActor(policy, contract, records), policy

    rows = runner.fly_chunk(task, make)
    return rows, records


def closed_loop(
    path: Path,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    run_seed: int,
    sha256: str,
    workers: int = 24,
    chunk: int = 10,
) -> dict[str, Any]:
    """Re-fly a checkpoint on listed episodes (CPU fp32 single-sample path), logging per step.

    The flight is selection's (:func:`ila.train.select.fly_checkpoint`: ``checkpoint_policy``,
    render ``full``, default contract, one torch thread per worker), so the rows are the
    selection's rows; only the actor adds a log. Rows do not depend on ``workers``/``chunk``.

    Args:
        path: A ``modular`` checkpoint.
        episodes: Listed episodes (``select``), in order.
        run_seed: The run's seed (row column).
        sha256: The checkpoint's SHA-256 (row column).
        workers: Processes.
        chunk: Episodes per task.

    Returns:
        ``rows`` (runner rows), ``records`` (:class:`ClosedLoopRecord`, episode order then step).
    """
    spec = p8_adapter.callable_spec(
        "modular", functools.partial(checkpoint_policy, str(path)), run_seed=run_seed
    )
    tags = runner.RowTags(arm="modular", teacher="T_ppo", student_seed=run_seed, ckpt_sha256=sha256)
    cfgs = p8_adapter.load_eval_configs()
    tasks = [
        runner.VisionTask(
            episodes=list(episodes[i : i + chunk]), spec=spec, cfgs=cfgs, tags=tags, render="full"
        )
        for i in range(0, len(episodes), chunk)
    ]
    runner.set_single_thread_env()
    results: list[tuple[list[dict[str, Any]], list[ClosedLoopRecord]]] = []
    if workers == 1 or len(tasks) <= 1:
        runner.init_worker_threads()
        results = [_fly_recording(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(workers, len(tasks)), initializer=runner.init_worker_threads
        ) as pool:
            results = list(pool.imap(_fly_recording, tasks, chunksize=1))
    return {
        "rows": [r for rows, _ in results for r in rows],
        "records": [rec for _, recs in results for rec in recs],
    }


# --------------------------------------------------------------------------- study


def _rows_match(rows: Sequence[Mapping[str, Any]], rows_csv: Path) -> bool | None:
    """Whether re-flown rows reproduce the selection's stored rows (outcome, steps, speed)."""
    if not rows_csv.is_file():
        return None
    with rows_csv.open(newline="", encoding="utf-8") as f:
        stored = list(csv.DictReader(f))
    keys = ("episode_seed", "outcome", "steps", "closing_speed_normal_m_s")
    if len(stored) != len(rows):
        return False
    return all(str(r[k]) == str(s[k]) for r, s in zip(rows, stored, strict=True) for k in keys)


def write_study(
    runs: Sequence[tuple[Path, Path, str, bool]],
    *,
    data_root: Path,
    out_prefix: Path,
    workers: int = 24,
    select_limit: int | None = None,
    modes: Sequence[str] = ("open_loop", "closed_loop"),
    heldout: loop.HeldoutSpec = loop.HELDOUT,
) -> dict[str, Any]:
    """Run the study on each ``(run_dir, checkpoint, sha256, selected)`` and write CSV + JSON.

    Args:
        runs: One tuple per run: its directory, the checkpoint, its SHA-256 and whether it is
            the run's selected checkpoint (``False`` only for a labelled diagnostic override).
        data_root: ``${data_root}``.
        out_prefix: ``results/phase7/modular_estimation`` (``.csv`` and ``.json`` appended).
        workers: Closed-loop processes.
        select_limit: Tests only: fly the first ``n`` ``select`` episodes (recorded).
        modes: ``open_loop`` and/or ``closed_loop``.
        heldout: The open-loop slice (:data:`ila.train.loop.HELDOUT`; tests: a synthetic one).

    Returns:
        The JSON record.
    """
    t0 = time.perf_counter()
    select = data_episodes.read_split(SELECT_SPLIT)
    if select_limit is not None:
        select = select[:select_limit]
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    table: list[dict[str, Any]] = []
    per_run = []
    for run_dir, path, sha, selected in runs:
        ckpt = read_checkpoint(path)
        seed = int(ckpt["run_seed"])
        base = {
            "arm": str(ckpt["arm"]),
            "run_seed": seed,
            "checkpoint": path.name,
            "checkpoint_sha256": sha,
            "selected": selected,
        }
        info: dict[str, Any] = {**base, "run_dir": str(run_dir), "global_step": ckpt["global_step"]}
        if "open_loop" in modes:
            t1 = time.perf_counter()
            ol = open_loop(path, data_root, heldout)
            table += [{**base, "mode": "open_loop", **r} for r in _grouped(
                ol["estimate"], ol["truth"], ol["groups"]
            )]  # fmt: skip
            info["open_loop"] = {
                "data": f"{heldout.collection} stream indices "
                f"{heldout.start}-{heldout.start + heldout.n - 1} (train pool, "
                "success-filtered; P7-D2)",
                "episodes": ol["episodes"],
                "wall_s": time.perf_counter() - t1,
            }
        if "closed_loop" in modes:
            t2 = time.perf_counter()
            cl = closed_loop(path, select, run_seed=seed, sha256=sha, workers=workers)
            recs: list[ClosedLoopRecord] = cl["records"]
            est = np.stack([r.estimate for r in recs])
            tru = np.stack([r.truth for r in recs])
            table += [{**base, "mode": "closed_loop", **r} for r in _grouped(
                est, tru, [r.ss for r in recs]
            )]  # fmt: skip
            rows_csv = run_dir / "select" / f"{path.stem}_rows.csv"
            info["closed_loop"] = {
                "list": {
                    "name": SELECT_SPLIT,
                    "file_sha256": data_episodes.file_sha256(split),
                    "episodes": len(select),
                    "limited_for_test": select_limit,
                },
                "path": "cpu_fp32_single_sample (act_modular; frozen T_ppo with the true "
                "time_fraction); truth logged via modular_targets.true_priv, never an input",
                "steps": len(recs),
                "summary": summarise_rows(cl["rows"]),
                "rows_match_selection": None if select_limit else _rows_match(cl["rows"], rows_csv),
                "wall_s": time.perf_counter() - t2,
            }
        per_run.append(info)
    record = {
        "what": "modular per-field estimation error vs P8's perception stand-in (sigma_p = 0.04 m, "
        "sigma_v = 0.2 m/s per world axis, model scale); P7-D2",
        "units": {name: f.unit for name, f in FIELDS.items()},
        "scale": "model scale, lambda = 1/25: lengths x 25, velocities x 5, times x 5 full scale",
        "sigma_p_m": SIGMA_P_M,
        "sigma_v_m_s": SIGMA_V_M_S,
        "caveats": [
            "P8's noise is per world axis; the estimate is in the drone-yaw frame (z shared, "
            "x/y rotated; the norm is frame-free).",
            "P8's noise is white and zero-mean; estimation error is time-correlated and may be "
            "biased (bias column).",
            "P8's stand-in also delays and holds the deck sample; not modelled here.",
            "closed loop is a diagnostic on select: it chooses nothing.",
        ],
        "runs": per_run,
        "rows": [{k: _json_value(v) for k, v in r.items()} for r in table],
        "wall_s": time.perf_counter() - t0,
    }
    out_csv = out_prefix.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows([{k: _cell(r.get(k)) for k in CSV_COLUMNS} for r in table])
    out_csv.write_text(buf.getvalue(), encoding="utf-8")
    mf.write_json_atomic(out_prefix.with_suffix(".json"), record)
    return record


def _json_value(value: Any) -> Any:
    """JSON value: NaN as ``null`` (strict JSON)."""
    return None if isinstance(value, float) and math.isnan(value) else value


def _cell(value: Any) -> Any:
    """CSV cell: floats to 6 significant digits, NaN empty."""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return value
