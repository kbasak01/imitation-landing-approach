"""Per-iteration DAgger report (plan Phase 5 item 3): descriptive only, never a selection.

For each ``bc_dagger`` seed and iteration ``i = 1 .. 5`` (plus an iteration-0 row), one row of
``results/phase5/dagger_iterations.csv`` (and ``.json`` with provenance):

* **Rollouts on ``train``** (iteration ``i``'s 200 episodes): the student checkpoint flown and
  its SHA-256, success ``k/n`` and rate, P8's outcome classes. ``policy_flown`` says what was
  measured: for beta > 0 the **beta-mixture** of teacher and student (the teacher's label is
  executed with probability beta per step), not the student alone; only beta = 0 iterations
  (4, 5) measure the student EMA itself.
* **Labelled data**: episodes and frames added by the iteration, cumulative labelled episodes
  (1 000 demos + 200 i), cumulative rollout frames, and the aggregate's training frames (demos
  after the success filter plus every rollout frame).
* **Teacher agreement on the student's own proposals** (the iteration's sidecar: the student's
  proposed action at every visited state, against the teacher's label there): mean
  ``||a_student - label||_2`` and the fraction of steps with ``||a_student - label||_inf <= 0.1``
  (normalised units; 0.1 normalised = 0.15 m/s model scale, 0.75 m/s full scale), over all
  steps and over the steps where the coin executed the student.
* **Copycat** (:mod:`ila.train.diagnostics`, d action / d last_action diagonal, x/y/z): the
  rollout student on the iteration's own states (pairs with the agreement columns); the EMA at
  the iteration's last step on P4-D5's held-out states and on the iteration's own states.
* **``select``**: the EMA at the iteration's last step flown on ``splits/select.parquet``
  (:func:`ila.train.select.fly_checkpoint`, CPU fp32 single-sample path): success, outcome
  classes, pooled p95 closing speed (m/s model scale), ``n_tunnelled``,
  ``n_detectors_disagree``, mean steps. Rows already flown for the same checkpoint (same
  SHA-256, the ``select`` episodes in order) are reused: the run's own selection rows (iteration
  5's ``s5_step010000.pt``; iteration 0's ``bc`` ``s0_step030000.pt``) or this report's cache.

Iteration 0 is the same-seed ``bc`` run's last step (its EMA): ``select`` and held-out copycat
only. Selection of ``bc_dagger`` stays iteration 5's checkpoints (P3-D1 §3): a curve that rises
on ``train`` but not on ``select`` is a finding, not a reason to pick another iteration. No
frozen list is read.
"""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila.data.store import DemoStore, episode_group
from ila.train import diagnostics as diag
from ila.train import manifest as mf
from ila.train.config import sha256_file
from ila.train.select import OUTCOME_ORDER, SELECT_SPLIT, summarise_rows

__all__ = [
    "AGREE_TOL",
    "COLUMNS",
    "agreement",
    "build_report",
    "policy_flown",
    "select_summary",
    "write_report",
]

#: The agreement tolerance on ``||a_student - label||_inf``, normalised units (0.15 m/s model).
AGREE_TOL: Final[float] = 0.1

#: The arm.
ARM: Final[str] = "bc_dagger"

_AXES: Final[tuple[str, ...]] = ("x", "y", "z")

#: Outcome classes other than ``success``, whose count has its own column (one name per column).
_FAILURES: Final[tuple[str, ...]] = tuple(o for o in OUTCOME_ORDER if o != "success")


def _axes(prefix: str) -> list[str]:
    return [f"{prefix}_{a}" for a in _AXES]


#: CSV columns, in order.
COLUMNS: Final[tuple[str, ...]] = (
    "seed",
    "iteration",
    "beta",
    "policy_flown",
    "rollout_student_ckpt",
    "rollout_student_sha256",
    "rollout_n",
    "rollout_success",
    "rollout_success_rate",
    *(f"rollout_{o}" for o in _FAILURES),
    "labelled_episodes_added",
    "labelled_episodes_cumulative",
    "labelled_frames_added",
    "labelled_frames_rollouts_cumulative",
    "aggregate_train_frames",
    "agree_steps",
    "agree_l2_mean",
    "agree_linf_le_0p1_frac",
    "agree_student_steps",
    "agree_student_l2_mean",
    "agree_student_linf_le_0p1_frac",
    *_axes("copycat_rollout_student_own"),
    *_axes("copycat_heldout"),
    *_axes("copycat_own"),
    "eval_ckpt",
    "eval_ckpt_sha256",
    "select_n",
    "select_success",
    "select_success_rate",
    *(f"select_{o}" for o in _FAILURES),
    "select_p95_closing_speed_m_s",
    "select_n_tunnelled",
    "select_n_detectors_disagree",
    "select_mean_steps",
    "select_rows_source",
)


def policy_flown(beta: float | None) -> str:
    """What an iteration's on-``train`` success measures (module docstring)."""
    if beta is None:
        return "not flown (iteration 0: bc demos)"
    if beta == 0.0:
        return "student EMA"
    return f"beta-mixture (teacher label p={beta:g} per step, else student EMA)"


def agreement(store: Path, sidecar: Path, tol: float = AGREE_TOL) -> dict[str, Any]:
    """Teacher agreement on the student's own proposals from an iteration's sidecar.

    Args:
        store: The iteration's rollout store (labels, student read path).
        sidecar: Its ``iter<i>.sidecar.npz`` (``proposed``, ``teacher_exec``, ``offsets``,
            ``demo_index``).
        tol: ``||proposal - label||_inf`` tolerance, normalised units.

    Returns:
        ``all`` and ``student`` blocks (``steps``, ``l2_mean``, ``linf_le_tol_frac``).

    Raises:
        ValueError: If the sidecar and the store disagree on episodes or lengths.
    """
    with np.load(sidecar, allow_pickle=False) as npz:
        proposed = np.asarray(npz["proposed"], dtype=np.float64)
        teacher_exec = np.asarray(npz["teacher_exec"], dtype=bool)
        offsets = np.asarray(npz["offsets"], dtype=np.int64)
        demo_index = np.asarray(npz["demo_index"], dtype=np.int64)
    labels = []
    with DemoStore(store) as st:
        for k, j in enumerate(demo_index):
            lab = st.labels(episode_group(int(j)))
            if lab.shape[0] != int(offsets[k + 1] - offsets[k]):
                raise ValueError(f"{sidecar}: episode {int(j)} length differs from the store")
            labels.append(lab)
    label = np.concatenate(labels).astype(np.float64)
    if label.shape != proposed.shape or teacher_exec.shape[0] != label.shape[0]:
        raise ValueError(f"{sidecar}: {proposed.shape} proposals, {label.shape} labels")
    diff = proposed - label

    def block(mask: NDArray[np.bool_]) -> dict[str, Any]:
        d = diff[mask]
        if not d.shape[0]:
            return {"steps": 0, "l2_mean": None, "linf_le_tol_frac": None}
        return {
            "steps": int(d.shape[0]),
            "l2_mean": float(np.linalg.norm(d, axis=1).mean()),
            "linf_le_tol_frac": float((np.abs(d).max(axis=1) <= tol).mean()),
        }

    return {
        "tol": tol,
        "units": "normalised action (x 1.5 m/s model scale)",
        "all": block(np.ones(label.shape[0], dtype=bool)),
        "student": block(~teacher_exec),
    }


def _rows_ok(rows: Sequence[Mapping[str, Any]], sha: str, seeds: Sequence[str]) -> bool:
    return [str(r["episode_seed"]) for r in rows] == list(seeds) and all(
        str(r["ckpt_sha256"]) == sha for r in rows
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_summary(
    ckpt: Path,
    sha: str,
    *,
    arm: str,
    run_seed: int,
    cache_dir: Path,
    workers: int,
    fly: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """``select`` success of one checkpoint's EMA, reusing rows already flown for it.

    Looks for ``<run>/select/<stem>_rows.csv`` (a selection's rows) and then
    ``<cache_dir>/<stem>_rows.csv``; rows are reused only if every row carries ``sha`` and the
    episodes are the ``select`` list in order. Otherwise the checkpoint is flown (CPU fp32
    single-sample path) and its rows cached.

    Returns:
        ``(summary, source)`` -- :func:`ila.train.select.summarise_rows` and where the rows came
        from (``"reused:<path>"`` or ``"flown"``); ``(None, "not flown")`` when ``fly`` is
        false and nothing is reusable.

    Raises:
        ValueError: If the checkpoint's bytes differ from ``sha``.
    """
    from ila.data import episodes as data_episodes

    episodes = data_episodes.read_split(SELECT_SPLIT)
    seeds = [str(e.episode_seed) for e in episodes]
    stem = ckpt.stem
    for cand in (
        ckpt.parent.parent / "select" / f"{stem}_rows.csv",
        cache_dir / f"{stem}_rows.csv",
    ):
        if cand.is_file():
            rows = _read_rows(cand)
            if _rows_ok(rows, sha, seeds):
                return summarise_rows(rows), f"reused:{_rel(cand)}"
    if not fly:
        return None, "not flown"
    if sha256_file(ckpt) != sha:
        raise ValueError(f"{ckpt}: SHA-256 differs from {sha}")
    from ila.eval import runner
    from ila.train.select import fly_checkpoint

    live = fly_checkpoint(ckpt, episodes, arm=arm, run_seed=run_seed, sha256=sha, workers=workers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{stem}_rows.csv"
    out.write_text(runner.vision_rows_csv(live), encoding="utf-8")
    return summarise_rows(_read_rows(out)), "flown"


def _rel(path: Path) -> str:
    from ila.policies.registry import REPO_ROOT

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _select_cells(summary: Mapping[str, Any] | None, source: str) -> dict[str, Any]:
    if summary is None:
        return {"select_rows_source": source}
    p95 = summary["p95_closing_speed_m_s"]
    return {
        "select_n": summary["n"],
        "select_success": summary["success"],
        "select_success_rate": summary["success"] / summary["n"] if summary["n"] else None,
        **{f"select_{o}": summary["outcomes"][o] for o in _FAILURES},
        "select_p95_closing_speed_m_s": None if p95 == float("inf") else p95,
        "select_n_tunnelled": summary["n_tunnelled"],
        "select_n_detectors_disagree": summary["n_detectors_disagree"],
        "select_mean_steps": summary["mean_steps"],
        "select_rows_source": source,
    }


def _copycat_cells(prefix: str, rec: Mapping[str, Any] | None) -> dict[str, Any]:
    if rec is None:
        return {}
    return dict(zip(_axes(prefix), rec["diag"], strict=True))


def _stage_steps(man: Mapping[str, Any], stage: int) -> int:
    for s in man["stages"]:
        if int(s["index"]) == stage:
            return int(s["steps"])
    raise ValueError(f"no stage {stage} in the run manifest")


def _checkpoint(run_dir: Path, man: Mapping[str, Any], stage: int) -> tuple[Path, str] | None:
    """Iteration ``stage``'s last full checkpoint and its manifest SHA-256, if written."""
    name = f"s{stage}_step{_stage_steps(man, stage):06d}.pt"
    entries = [c for c in man.get("checkpoints", []) if c["file"] == name and c["kind"] == "full"]
    path = run_dir / "checkpoints" / name
    if not entries or not path.is_file():
        return None
    return path, str(entries[-1]["sha256"])


def _seed_rows(
    seed: int,
    *,
    run_root: Path,
    data_root: Path,
    heldout: Any,
    heldout_idx: NDArray[np.int64],
    workers: int,
    fly: bool,
    log: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    from ila.data import dagger

    run_dir = mf.run_dir(ARM, seed, root=run_root)
    man = mf.read_json(run_dir / "manifest.json")
    cache = run_dir / "dagger" / "select_rows"
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    # Iteration 0: the bc run's last step (init_from), select and held-out copycat only.
    bc_ckpt = Path(man["iteration0"]["file"])
    bc_sha = str(man["init_from_sha256"])
    t0 = time.perf_counter()
    summary, source = select_summary(
        bc_ckpt, bc_sha, arm="bc", run_seed=seed, cache_dir=cache, workers=workers, fly=fly
    )
    row0: dict[str, Any] = {
        "seed": seed,
        "iteration": 0,
        "policy_flown": policy_flown(None),
        "labelled_episodes_cumulative": int(man["iteration0"]["labelled_episodes"]),
        "eval_ckpt": _rel(bc_ckpt),
        "eval_ckpt_sha256": bc_sha,
        **_copycat_cells("copycat_heldout", diag.copycat_record(bc_ckpt, heldout, heldout_idx)),
        **_select_cells(summary, source),
    }
    rows.append(row0)
    log(f"s{seed} iteration 0: {source} ({time.perf_counter() - t0:.0f} s)")

    cumulative_frames = 0
    for i in range(1, len(man["stages"]) + 1):
        t0 = time.perf_counter()
        ckpt = _checkpoint(run_dir, man, i)
        rec_file = run_dir / "dagger" / f"iter{i}.json"
        if ckpt is None or not rec_file.is_file():
            missing.append(f"s{seed} iteration {i}")
            break
        rec = mf.read_json(rec_file)
        r = dagger.load_rollout(data_root, seed, i)
        if r.store_sha256 != rec["rollout"]["store_sha256"]:
            raise ValueError(f"s{seed} iteration {i}: rollout store differs from the run's record")
        cumulative_frames += r.labelled_frames
        agree = agreement(r.store, r.sidecar)
        own = diag.rollout_states(r.store)
        own_idx = diag.sample_indices(len(own), [diag.COPYCAT_SEED, seed, i])
        eval_path, eval_sha = ckpt
        summary, source = select_summary(
            eval_path, eval_sha, arm=ARM, run_seed=seed, cache_dir=cache, workers=workers, fly=fly
        )
        n = r.labelled_episodes
        rows.append(
            {
                "seed": seed,
                "iteration": i,
                "beta": r.beta,
                "policy_flown": policy_flown(r.beta),
                "rollout_student_ckpt": _rel(r.student_ckpt),
                "rollout_student_sha256": r.student_ckpt_sha256,
                "rollout_n": n,
                "rollout_success": r.outcomes["success"],
                "rollout_success_rate": r.outcomes["success"] / n if n else None,
                **{f"rollout_{o}": r.outcomes[o] for o in _FAILURES},
                "labelled_episodes_added": n,
                "labelled_episodes_cumulative": int(rec["labelled_episodes"]),
                "labelled_frames_added": r.labelled_frames,
                "labelled_frames_rollouts_cumulative": cumulative_frames,
                "aggregate_train_frames": int(man["data"][str(i)]["train_frames"]),
                "agree_steps": agree["all"]["steps"],
                "agree_l2_mean": agree["all"]["l2_mean"],
                "agree_linf_le_0p1_frac": agree["all"]["linf_le_tol_frac"],
                "agree_student_steps": agree["student"]["steps"],
                "agree_student_l2_mean": agree["student"]["l2_mean"],
                "agree_student_linf_le_0p1_frac": agree["student"]["linf_le_tol_frac"],
                **_copycat_cells(
                    "copycat_rollout_student_own", diag.copycat_record(r.student_ckpt, own, own_idx)
                ),
                **_copycat_cells(
                    "copycat_heldout", diag.copycat_record(eval_path, heldout, heldout_idx)
                ),
                **_copycat_cells("copycat_own", diag.copycat_record(eval_path, own, own_idx)),
                "eval_ckpt": _rel(eval_path),
                "eval_ckpt_sha256": eval_sha,
                **_select_cells(summary, source),
            }
        )
        log(f"s{seed} iteration {i}: {source} ({time.perf_counter() - t0:.0f} s)")
    return rows, missing


def build_report(
    seeds: Sequence[int],
    *,
    run_root: Path,
    data_root: Path,
    workers: int = 24,
    fly: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """Build every row (module docstring) and the provenance record.

    Args:
        seeds: ``bc_dagger`` seeds.
        run_root: Runs root (``artifacts/runs``).
        data_root: ``${data_root}`` (demo and rollout stores).
        workers: ``select`` rollout processes (rows do not depend on it).
        fly: Fly checkpoints with no reusable ``select`` rows (``False``: leave them empty).
        log: Line logger.

    Returns:
        ``{"rows": [...], "missing": [...], ...provenance}``.
    """
    from ila.data import episodes as data_episodes

    t0 = time.perf_counter()
    heldout = diag.heldout_states(data_root)
    heldout_idx = diag.sample_indices(len(heldout), diag.COPYCAT_SEED)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        seed_rows, seed_missing = _seed_rows(
            seed,
            run_root=run_root,
            data_root=data_root,
            heldout=heldout,
            heldout_idx=heldout_idx,
            workers=workers,
            fly=fly,
            log=log,
        )
        rows.extend(seed_rows)
        missing.extend(seed_missing)
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    return {
        "what": "bc_dagger per-iteration report (plan Phase 5 item 3); descriptive, never "
        "used for selection (iteration 5's checkpoints only, P3-D1 §3)",
        "columns": list(COLUMNS),
        "rows": rows,
        "missing": missing,
        "select_list": {"name": SELECT_SPLIT, "file_sha256": data_episodes.file_sha256(split)},
        "select_path": "cpu_fp32_single_sample, EMA weights, render full",
        "agreement": {
            "tol_linf": AGREE_TOL,
            "units": "normalised action; 0.1 = 0.15 m/s model scale (0.75 m/s full scale)",
            "student_steps": "coin executed the student (teacher_exec false)",
        },
        "copycat": {
            "quantity": "diag of mean d a / d last_action[t], autograd, EMA, centre crop, "
            "fp32 CPU; normalised action per normalised action",
            "heldout": "ppo_2000 1950-1999 every 2nd episode, 512 frames, default_rng(0) (P4-D5)",
            "own": "512 frames of the iteration's rollout store, default_rng([0, seed, i])",
            "rollout_student_own": "the checkpoint that flew iteration i, on its own states",
        },
        "policy_flown": "iterations with beta > 0 measure the beta-mixture, not the student",
        "git": mf.git_state(),
        "wall_clock_s": time.perf_counter() - t0,
    }


def write_report(report: Mapping[str, Any], out_csv: Path) -> Path:
    """Write the CSV (``COLUMNS``; empty cells for not-applicable) and the JSON beside it.

    Returns:
        The JSON path (``out_csv`` with suffix ``.json``).
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in report["rows"]:
        writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in COLUMNS})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(buf.getvalue(), encoding="utf-8")
    out_json = out_csv.with_suffix(".json")
    mf.write_json_atomic(
        out_json, {**report, "csv": out_csv.name, "csv_sha256": sha256_file(out_csv)}
    )
    return out_json
