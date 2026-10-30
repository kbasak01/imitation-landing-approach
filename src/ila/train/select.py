"""Closed-loop checkpoint selection on ``splits/select.parquet`` only (P3-D1 §4).

* **List.** ``splits/select.parquet`` (150 ``tune`` episodes, 50 each of SS3-SS5, SHA-256
  checked against ``splits/MANIFEST.csv`` by :func:`ila.data.episodes.read_split`). No frozen
  list is read anywhere in this module (``tests/test_render_access.py`` and
  ``tests/test_train_select.py`` guard it).
* **Contract.** A checkpoint flies with the observation-contract variant it was trained on
  (``obs_variant`` in the checkpoint; ``default`` for every Phase 4 arm).
* **Path.** Every candidate flies through :func:`ila.eval.runner.run_list_vision` on the **CPU
  fp32 single-sample path** (one torch thread per worker; rows are bit-identical for any worker
  count, P2-D3 item 1), render ``full``. Vision students take ``act_vision``; ``bc_state`` takes
  P8's observation (it has no ``act_vision``).
* **Candidates.** Every full checkpoint of the run, EMA weights (the evaluated weights). An
  iterated arm restricts them to its final iteration's checkpoints (P3-D1 §3): any run whose
  config says ``selection.candidates: final_iteration_checkpoints`` (``bc_matched``,
  ``bc_dagger``: iteration 5's 5 k and 10 k) or that has more than one stage. The rule reads the
  run's manifest (its resolved config and stages), never the arm name.
  ``ema_last.pt`` duplicates the last checkpoint's EMA and is not a separate candidate.
  A run whose config says ``selection.candidates: onnx_exports`` (``R1_lerobot_dp``, P6-D3) is
  selected among its ONNX exports ``<run>/onnx/step<N>/``, one per manifest checkpoint (every
  10 k steps), each export's ``meta.json`` naming that checkpoint's ``model.safetensors``
  SHA-256. Such a run has no EMA (LeRobot 0.6.1 has none): the exported raw weights fly through
  :class:`ila.policies.onnx_dp.OnnxDPStudent` on ONNX Runtime's CPU path (one thread), at the
  same ``Ta = 8``, ``K = 10``, with the same sampler and noise keys as ``dp``. Its candidate
  SHA-256 is ``meta.json``'s (which records both graphs' and the source checkpoint's).
  **Parity gate** (P6-D3): every export must carry a ``parity.json``
  (``scripts/r1/parity.py``) written for its exact files; selection refuses a run with any
  export unchecked, and a failed parity bars that checkpoint (listed under
  ``excluded_by_parity`` in ``selection.json``, never flown).
* **Rule.** Highest ``success`` count; ties by the lower pooled p95 closing speed
  (``closing_speed_normal_m_s``, m/s model scale, over touched-down episodes only, P8's
  ``quantile_linear`` at 0.95; no touchdown ranks as +inf); then the lowest step.
* **Refusals.** Timing, smoke and test runs, and runs not ``done``, are never selected.

:func:`choose_sigma` applies the same rule to the three DART trials' selected checkpoints, then
the smaller sigma (P3-D1 §4).
"""

from __future__ import annotations

import csv
import functools
import io
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import runner
from ila.policies.onnx_dp import (
    OnnxDPMeta,
    bundle_sha256,
    export_dirs,
    load_onnx_student,
    parity_passed,
    read_meta,
)
from ila.policies.registry import Student, load_student, read_checkpoint
from ila.train import manifest as mf
from ila.train.config import sha256_file

__all__ = [
    "OUTCOME_ORDER",
    "FINAL_ITERATION",
    "ONNX_EXPORTS",
    "SELECTION_RULE",
    "ResummaryMismatchError",
    "P95",
    "SELECT_SPLIT",
    "Candidate",
    "CandidateResult",
    "candidate_sha256",
    "candidates",
    "checkpoint_policy",
    "onnx_checkpoint_policy",
    "onnx_parity_exclusions",
    "choose",
    "choose_sigma",
    "fly_checkpoint",
    "resummarise_run",
    "select_run",
    "summarise_rows",
    "write_rows_csv",
]

#: The only list selection flies.
SELECT_SPLIT: Final[str] = data_episodes.SELECT_NAME

#: The closing-speed quantile (P8's ``rld.eval.stats.P95_QUANTILE``, not re-exported by the
#: adapter; P3-D1 §1 "relative p95"). The quantile itself is P8's ``quantile_linear``.
P95: Final[float] = 0.95

#: ``selection.candidates`` of the iterated arms' configs (P3-D1 §3).
FINAL_ITERATION: Final[str] = "final_iteration_checkpoints"

#: ``selection.candidates`` of an externally trained arm flown from ONNX (``R1_lerobot_dp``).
ONNX_EXPORTS: Final[str] = "onnx_exports"

#: The rule's text in every ``selection.json``.
SELECTION_RULE: Final[str] = (
    "max success; then min pooled p95 closing_speed_normal_m_s (touched-down); "
    "then min global step (P3-D1 §4)"
)

#: P8's outcome classes, in P8's order (P3-D1 §1).
OUTCOME_ORDER: Final[tuple[str, ...]] = (
    "crash",
    "off_pad",
    "hard_landing",
    "bounce",
    "success",
    "timeout",
)


def checkpoint_policy(
    path: str,
    cfgs: p8_adapter.EvalConfigs,
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
    temporal_ensembling: bool = False,
) -> Any:
    """Picklable ``PolicySpec.build``: a checkpoint's EMA student on the CPU, fp32.

    Args:
        path: The checkpoint file.
        cfgs: P8's configs (P8's ``build(cfgs)`` shape; unused).
        n_action_steps: ``dp`` only: ``Ta`` (config default 8 when ``None``; the inference
            sweeps, :mod:`ila.train.dp_sweeps`, set it).
        num_inference_steps: ``dp`` only: ``K`` (config default 10 when ``None``).
        temporal_ensembling: ``dp`` only (P6-D1).

    Returns:
        The :data:`ila.policies.registry.Student` (``dp``: config defaults ``Ta = 8``,
        ``K = 10`` for selection), typed ``Any`` as P8's ``PolicySpec.build`` result (vision
        students satisfy P8's ``EvalPolicy`` shape through ``reset``/``act``; the runner calls
        ``act_vision``).
    """
    del cfgs
    student: Student = load_student(
        Path(path),
        "cpu",
        weights="ema",
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        temporal_ensembling=temporal_ensembling,
    )
    return student


def onnx_checkpoint_policy(
    path: str,
    run_seed: int,
    cfgs: p8_adapter.EvalConfigs,
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
    temporal_ensembling: bool = False,
) -> Any:
    """Picklable ``PolicySpec.build``: an ONNX export's student on ORT's CPU path (one thread).

    Args:
        path: The export directory ``<run>/onnx/step<N>/``.
        run_seed: The training run's seed (first element of every noise key).
        cfgs: P8's configs (unused).
        n_action_steps: ``Ta`` (8 when ``None``).
        num_inference_steps: ``K`` (10 when ``None``).
        temporal_ensembling: Off for selection (P6-D1).

    Returns:
        The :class:`ila.policies.onnx_dp.OnnxDPStudent`.
    """
    del cfgs
    return load_onnx_student(
        Path(path),
        run_seed,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        temporal_ensembling=temporal_ensembling,
    )


def candidate_sha256(path: Path) -> str:
    """A candidate's SHA-256: the checkpoint file's, or an ONNX export's ``meta.json``'s.

    For an export directory both graphs are verified against ``meta.json`` first
    (:func:`ila.policies.onnx_dp.bundle_sha256`).
    """
    return bundle_sha256(path) if Path(path).is_dir() else sha256_file(Path(path))


@dataclass(frozen=True)
class Candidate:
    """One candidate checkpoint.

    Attributes:
        file: Checkpoint path.
        stage: Stage (iteration) index.
        stage_step: Step within the stage.
        global_step: Step over the whole run (the tie-break's "step").
        sha256: The file's SHA-256 (from the manifest; re-checked before flying).
    """

    file: str
    stage: int
    stage_step: int
    global_step: int
    sha256: str


@dataclass(frozen=True)
class CandidateResult:
    """One candidate's ``select`` outcome.

    Attributes:
        candidate: The checkpoint.
        n: Episodes flown (150).
        success: ``success`` count.
        outcomes: Outcome class -> count, P8's order.
        n_touchdown: Episodes with a touchdown closing speed.
        p95_closing_speed_m_s: Pooled p95 over touched-down episodes, m/s model scale
            (``inf`` without any touchdown).
        mean_steps: Mean control steps per episode (30 Hz, model scale).
        rows_file: Per-episode rows (CSV) next to ``selection.json``.
        n_tunnelled: Episodes P8 flags ``tunnelled`` (the drone passed through the plate
            between physics substeps), any outcome; ``None`` in records written before the
            column was summarised.
        n_detectors_disagree: Episodes whose contact and analytic touchdown detectors
            disagree (P8's ``detectors_disagree``); ``None`` likewise.
    """

    candidate: Candidate
    n: int
    success: int
    outcomes: Mapping[str, int]
    n_touchdown: int
    p95_closing_speed_m_s: float
    mean_steps: float
    rows_file: str
    n_tunnelled: int | None = None
    n_detectors_disagree: int | None = None

    @property
    def success_rate(self) -> float:
        """``success / n``."""
        return self.success / self.n if self.n else 0.0


def summarise_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Success, outcome breakdown and pooled p95 closing speed of runner rows.

    Args:
        rows: Runner rows (P8 columns ``outcome``, ``closing_speed_normal_m_s``, ``steps``).

    Rows may be live runner rows or rows read back from a ``*_rows.csv`` (string cells).

    Returns:
        ``n``, ``success``, ``outcomes`` (P8's order, zeros included), ``n_touchdown``,
        ``p95_closing_speed_m_s`` (``inf`` without a touchdown), ``mean_steps``,
        ``n_tunnelled`` and ``n_detectors_disagree`` (P8's per-episode flags, any outcome).

    Raises:
        ValueError: On an unknown outcome class or a flag that is not a boolean.
    """
    outcomes = dict.fromkeys(OUTCOME_ORDER, 0)
    for r in rows:
        o = str(r["outcome"])
        if o not in outcomes:
            raise ValueError(f"unknown outcome {o!r}")
        outcomes[o] += 1
    speeds = np.asarray([float(r["closing_speed_normal_m_s"]) for r in rows], dtype=np.float64)
    touched = speeds[np.isfinite(speeds)]
    p95 = float(p8_adapter.stats.quantile_linear(touched, P95)) if touched.size else math.inf
    steps = [float(r["steps"]) for r in rows]
    return {
        "n": len(rows),
        "success": outcomes["success"],
        "outcomes": outcomes,
        "n_touchdown": int(touched.size),
        "p95_closing_speed_m_s": p95,
        "mean_steps": float(np.mean(steps)) if steps else float("nan"),
        "n_tunnelled": sum(_flag(r["tunnelled"]) for r in rows),
        "n_detectors_disagree": sum(_flag(r["detectors_disagree"]) for r in rows),
    }


def _flag(value: Any) -> int:
    """Return a P8 boolean cell (``True``/``False``, or ``"True"``/``"False"`` from a CSV) as 0/1.

    Raises:
        ValueError: On anything else (NaN included: these flags are never missing).
    """
    if isinstance(value, str):
        if value in ("True", "False"):
            return int(value == "True")
        raise ValueError(f"not a boolean cell: {value!r}")
    if isinstance(value, bool | np.bool_):
        return int(bool(value))
    raise ValueError(f"not a boolean cell: {value!r}")


def _result(c: Candidate, s: Mapping[str, Any], rows_file: str) -> CandidateResult:
    return CandidateResult(
        candidate=c,
        n=s["n"],
        success=s["success"],
        outcomes=s["outcomes"],
        n_touchdown=s["n_touchdown"],
        p95_closing_speed_m_s=s["p95_closing_speed_m_s"],
        mean_steps=s["mean_steps"],
        rows_file=rows_file,
        n_tunnelled=s["n_tunnelled"],
        n_detectors_disagree=s["n_detectors_disagree"],
    )


def choose(results: Sequence[CandidateResult]) -> CandidateResult:
    """Apply P3-D1 §4: max success, then min pooled p95 closing speed, then min step.

    Raises:
        ValueError: With no candidates.
    """
    if not results:
        raise ValueError("no candidates")
    return min(
        results,
        key=lambda r: (-r.success, r.p95_closing_speed_m_s, r.candidate.global_step),
    )


def candidates(run_dir: Path) -> list[Candidate]:
    """Return a run's candidate checkpoints (module docstring), lowest step first.

    Raises:
        ValueError: If the run is not selectable (timing/smoke/test), not ``done``, has no
            full checkpoint, or a checkpoint's file is missing.
    """
    man = mf.read_json(run_dir / "manifest.json")
    if not man.get("selectable") or man.get("purpose") not in ("train", "tuning"):
        raise ValueError(f"{run_dir}: purpose {man.get('purpose')!r} is never selected")
    if man.get("state") != "done":
        raise ValueError(f"{run_dir}: state {man.get('state')!r}, not done")
    full = [c for c in man["checkpoints"] if c["kind"] == "full"]
    if not full:
        raise ValueError(f"{run_dir}: no checkpoints")
    stages = [s["index"] for s in man["stages"]]
    rule = man.get("config_resolved", {}).get("selection", {}).get("candidates")
    if rule not in (None, FINAL_ITERATION, ONNX_EXPORTS):
        raise ValueError(f"{run_dir}: unknown candidate rule {rule!r}")
    if rule == ONNX_EXPORTS:
        return _onnx_candidates(run_dir, full)
    if rule == FINAL_ITERATION or len(stages) > 1:  # iterated arms: the final iteration only
        full = [c for c in full if c["stage"] == stages[-1]]
    out = []
    for c in sorted(full, key=lambda c: c["global_step"]):
        path = run_dir / "checkpoints" / c["file"]
        if not path.is_file():
            raise ValueError(f"{path} missing")
        out.append(Candidate(str(path), c["stage"], c["stage_step"], c["global_step"], c["sha256"]))
    return out


def onnx_parity_exclusions(run_dir: Path) -> list[dict[str, Any]]:
    """Exports of a run whose ``parity.json`` failed (barred from selection; P6-D3).

    Raises:
        ValueError: If an export has no ``parity.json``, or one written for other files.
    """
    out = []
    for step, bundle in sorted(export_dirs(run_dir).items()):
        passed, reason = parity_passed(bundle)
        if not passed and reason != "parity failed":
            raise ValueError(f"{bundle}: {reason}")
        if not passed:
            out.append({"step": step, "export": str(bundle), "reason": reason})
    return out


def _onnx_candidates(run_dir: Path, full: Sequence[Mapping[str, Any]]) -> list[Candidate]:
    """One ONNX export per manifest checkpoint, lowest step first, parity-passed only.

    Raises:
        ValueError: If a checkpoint has no export, an export has no checkpoint, an export's
            ``meta.json`` names another checkpoint's weights, an export has no valid
            ``parity.json``, or none passed parity.
    """
    exports = export_dirs(run_dir)
    by_step = {int(c["global_step"]): c for c in full}
    if set(exports) != set(by_step):
        raise ValueError(
            f"{run_dir}: exports at steps {sorted(exports)} != checkpoints {sorted(by_step)}"
        )
    barred = {e["step"] for e in onnx_parity_exclusions(run_dir)}
    out = []
    for step in sorted(by_step):
        c, bundle = by_step[step], exports[step]
        if OnnxDPMeta.from_mapping(read_meta(bundle)).source_sha256 != c["sha256"]:
            raise ValueError(f"{bundle}: exported from another checkpoint than {c['file']}")
        if step in barred:
            continue
        out.append(Candidate(str(bundle), c["stage"], c["stage_step"], step, bundle_sha256(bundle)))
    if not out:
        raise ValueError(f"{run_dir}: no export passed parity")
    return out


def fly_checkpoint(
    path: Path,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    arm: str,
    run_seed: int,
    sha256: str,
    workers: int,
    chunk: int = 10,
    inference: tuple[int, int, bool] | None = None,
) -> list[dict[str, Any]]:
    """Fly one checkpoint's EMA student on listed episodes (CPU fp32 single-sample path).

    Args:
        path: The checkpoint, or an ONNX export directory (``R1_lerobot_dp``: ORT CPU, one
            thread, the default observation contract).
        episodes: Listed episodes, in order.
        arm: Arm label for the rows.
        run_seed: The run seed (``dp``: first element of every noise key).
        sha256: Checkpoint SHA-256 (row column).
        workers: Processes (rows do not depend on it).
        chunk: Episodes per task (rows do not depend on it).
        inference: ``dp`` only: ``(Ta, K, temporal_ensembling)``, also written to the rows'
            ``Ta``/``K``/``ensemble`` columns (the inference sweeps). ``None`` (selection):
            the config defaults, columns empty.

    Returns:
        Runner rows, in episode order.
    """
    onnx = Path(path).is_dir()
    base = (
        functools.partial(onnx_checkpoint_policy, str(path), int(run_seed))
        if onnx
        else functools.partial(checkpoint_policy, str(path))
    )
    if inference is None:
        build = base
        tags = runner.RowTags(arm=arm, teacher="T_ppo", student_seed=run_seed, ckpt_sha256=sha256)
    else:
        ta, k, ensemble = inference
        build = functools.partial(
            base,
            n_action_steps=int(ta),
            num_inference_steps=int(k),
            temporal_ensembling=bool(ensemble),
        )
        tags = runner.RowTags(
            arm=arm,
            teacher="T_ppo",
            student_seed=run_seed,
            Ta=int(ta),
            K=int(k),
            ensemble=bool(ensemble),
            ckpt_sha256=sha256,
        )
    spec = p8_adapter.callable_spec(arm, build, run_seed=run_seed)
    # The contract it was trained on (an ONNX export: the default contract, 13-d proprio).
    variant = "default" if onnx else str(read_checkpoint(path)["obs_variant"])
    return runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="full",
        workers=workers,
        chunk=chunk,
        contract_variant=variant,
    )


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write runner rows to ``path`` (:data:`runner.VISION_EPISODE_COLUMNS`; selection's format).

    Args:
        path: The CSV file (overwritten).
        rows: Runner rows.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(runner.VISION_EPISODE_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def select_run(run_dir: Path, *, workers: int = 24) -> dict[str, Any]:
    """Select a run's checkpoint on ``select`` and write ``selection.json``.

    Args:
        run_dir: A finished ``train`` or ``tuning`` run.
        workers: Rollout processes (rows do not depend on it).

    Returns:
        The selection record (also written to ``run_dir/selection.json``).

    Raises:
        ValueError: If a checkpoint's bytes differ from its manifest SHA-256.
    """
    t0 = time.perf_counter()
    man = mf.read_json(run_dir / "manifest.json")
    cands = candidates(run_dir)
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    rows_list = data_episodes.read_split(SELECT_SPLIT)
    out_dir = run_dir / "select"
    out_dir.mkdir(exist_ok=True)
    results: list[CandidateResult] = []
    onnx = man.get("config_resolved", {}).get("selection", {}).get("candidates") == ONNX_EXPORTS
    for c in cands:
        if candidate_sha256(Path(c.file)) != c.sha256:
            raise ValueError(f"{c.file}: SHA-256 differs from the manifest")
        rows = fly_checkpoint(
            Path(c.file),
            rows_list,
            arm=str(man["arm"]),
            run_seed=int(man["run_seed"]),
            sha256=c.sha256,
            workers=workers,
        )
        rows_file = out_dir / f"{Path(c.file).name if onnx else Path(c.file).stem}_rows.csv"
        write_rows_csv(rows_file, rows)
        results.append(_result(c, summarise_rows(rows), rows_file.name))
    best = choose(results)
    record = {
        "run_dir": str(run_dir),
        "arm": man["arm"],
        "run_seed": man["run_seed"],
        "purpose": man["purpose"],
        "sigma": man.get("sigma"),
        "list": {
            "name": SELECT_SPLIT,
            "file": split.name,
            "file_sha256": data_episodes.file_sha256(split),
            "episodes": len(rows_list),
        },
        "rule": SELECTION_RULE,
        "weights": "raw_no_ema (lerobot 0.6.1 has no EMA)" if onnx else "ema",
        "inference_path": "ort_cpu_fp32_single_sample" if onnx else "cpu_fp32_single_sample",
        "candidate_kind": ONNX_EXPORTS if onnx else "checkpoint",
        **({"excluded_by_parity": onnx_parity_exclusions(run_dir)} if onnx else {}),
        "tag": man.get("tag", "core"),
        "render": "full",
        "workers": workers,
        "candidates": [_result_record(r) for r in results],
        "selected": _result_record(best),
        "wall_clock_s": time.perf_counter() - t0,
    }
    mf.write_json_atomic(run_dir / "selection.json", record)
    status = run_dir / "status.json"
    if status.is_file():
        st = mf.read_json(status)
        st["last_select_success"] = best.success_rate
        mf.write_json_atomic(status, st)
    return record


def _result_record(r: CandidateResult) -> dict[str, Any]:
    out = asdict(r)
    out["success_rate"] = r.success_rate
    out["p95_closing_speed_m_s"] = (
        None if math.isinf(r.p95_closing_speed_m_s) else r.p95_closing_speed_m_s
    )
    return out


def _from_record(rec: Mapping[str, Any]) -> CandidateResult:
    p95 = rec["p95_closing_speed_m_s"]
    return CandidateResult(
        candidate=Candidate(**rec["candidate"]),
        n=int(rec["n"]),
        success=int(rec["success"]),
        outcomes=dict(rec["outcomes"]),
        n_touchdown=int(rec["n_touchdown"]),
        p95_closing_speed_m_s=math.inf if p95 is None else float(p95),
        mean_steps=float(rec["mean_steps"]),
        rows_file=str(rec["rows_file"]),
        n_tunnelled=rec.get("n_tunnelled"),
        n_detectors_disagree=rec.get("n_detectors_disagree"),
    )


class ResummaryMismatchError(ValueError):
    """Re-summarising stored rows would change a selection (refused; nothing is written)."""


#: Fields of a candidate that a re-summary must reproduce exactly.
_REPRODUCED: Final[tuple[str, ...]] = (
    "candidate",
    "n",
    "success",
    "outcomes",
    "n_touchdown",
    "p95_closing_speed_m_s",
    "mean_steps",
    "rows_file",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resummarise_run(run_dir: Path, *, write: bool = True) -> dict[str, Any]:
    """Rebuild ``selection.json`` from the stored ``select/*_rows.csv`` -- no flight.

    Every candidate is re-summarised from its rows file (written when it was flown), the rule
    is re-applied, and the result must reproduce the existing record: the same candidates in
    the same order, each with the same success, outcomes, touchdowns, p95 and mean steps, and
    the same selected checkpoint. Only then is the record rewritten, with every original field
    kept (``wall_clock_s``, ``workers``, ... of the flight) and ``n_tunnelled`` /
    ``n_detectors_disagree`` added per candidate and for the selected one.

    Args:
        run_dir: A run with ``selection.json`` and ``select/``.
        write: Write the rebuilt record (``False``: check only).

    Returns:
        The rebuilt record.

    Raises:
        ResummaryMismatchError: If anything above differs, or a rows file does not hold the
            ``select`` list's episodes in order.
    """
    old = mf.read_json(run_dir / "selection.json")
    episodes = data_episodes.read_split(SELECT_SPLIT)
    seeds = [str(e.episode_seed) for e in episodes]
    results = []
    for rec in old["candidates"]:
        prev = _from_record(rec)
        rows = _read_rows(run_dir / "select" / prev.rows_file)
        if [r["episode_seed"] for r in rows] != seeds:
            raise ResummaryMismatchError(f"{prev.rows_file}: not the select episodes in order")
        if any(r["ckpt_sha256"] != prev.candidate.sha256 for r in rows):
            raise ResummaryMismatchError(f"{prev.rows_file}: rows of another checkpoint")
        results.append(_result(prev.candidate, summarise_rows(rows), prev.rows_file))
    for rec, new in zip(old["candidates"], results, strict=True):
        fresh = _result_record(new)
        diff = [k for k in _REPRODUCED if fresh[k] != rec[k]]
        if diff:
            raise ResummaryMismatchError(f"{new.rows_file}: {diff} differ from selection.json")
    best = choose(results)
    sel = _result_record(best)
    if any(sel[k] != old["selected"][k] for k in _REPRODUCED):
        raise ResummaryMismatchError(f"{run_dir}: the selected candidate would change")
    record = {
        **old,
        "candidates": [_result_record(r) for r in results],
        "selected": sel,
        "resummary": {
            "flown": False,
            "source": "select/*_rows.csv",
            "reproduced": list(_REPRODUCED),
            "added": ["n_tunnelled", "n_detectors_disagree"],
        },
    }
    if write:
        mf.write_json_atomic(run_dir / "selection.json", record)
    return record


def choose_sigma(selections: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose sigma* from the DART trials' ``selection.json`` records (P3-D1 §4).

    sigma* is the trial whose selected checkpoint has the highest ``select`` success; ties by
    the lower pooled p95 closing speed, then the lower step, then the **smaller sigma**.

    Args:
        selections: sigma -> that trial's selection record (``purpose`` must be ``tuning``).

    Returns:
        ``{"sigma_star": ..., "trials": [...], "rule": ...}``.

    Raises:
        ValueError: With no trials or a record that is not a ``bc_dart`` tuning trial.
    """
    if not selections:
        raise ValueError("no DART trials")
    trials = []
    for sigma, rec in selections.items():
        if rec.get("purpose") != "tuning" or rec.get("arm") != "bc_dart":
            raise ValueError(f"sigma {sigma}: not a bc_dart tuning trial")
        if rec.get("sigma") is None or not math.isclose(float(rec["sigma"]), sigma):
            raise ValueError(f"sigma {sigma}: record says {rec.get('sigma')}")
        trials.append((float(sigma), _from_record(rec["selected"])))
    sigma_star, best = min(
        trials,
        key=lambda t: (-t[1].success, t[1].p95_closing_speed_m_s, t[1].candidate.global_step, t[0]),
    )
    return {
        "sigma_star": sigma_star,
        "rule": "max select success of the selected checkpoint; then min p95 closing speed; "
        "then min step; then the smaller sigma (P3-D1 §4)",
        "trials": [{"sigma": s, "selected": _result_record(r)} for s, r in sorted(trials)],
        "selected": _result_record(best),
    }
