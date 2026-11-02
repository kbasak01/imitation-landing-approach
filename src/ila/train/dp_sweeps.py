"""``dp`` inference-time sweeps on ``select`` (plan Phase 6 item 4; P6-D1). Reported only.

For each named ``dp`` run, the checkpoint that P3-D1 §4 **already selected** at ``Ta = 8``,
``K = 10`` (``<run>/selection.json``) is flown again on the 150 ``splits/select.parquet``
episodes under other inference settings. Nothing is re-selected, nothing here feeds a choice,
and no frozen list is read (``tests/test_render_access.py`` guards ``ila/train``).

* **Cells** (from ``configs/train/dp.yaml`` ``inference_sweeps``, deduplicated):
  ``Ta in {1, 2, 4, 8, 12}`` x temporal ensembling ``{off, on}`` at the config's ``K = 10``,
  and ``K in {1, 2, 5, 10, 25, 100}`` at the config's ``Ta = 8``, ensembling off: 15 distinct
  settings. ``Ta = 1`` with ensembling is the same policy as ``Ta = 1`` without
  (:func:`ila.policies.dp.chunking.ensemble_sources`; ``tests/test_policies.py`` checks the
  actions bit for bit), so it is flown once and recorded as an alias: 14 flights. ``K = 1`` and
  ``2`` are labelled ``schedule_limited`` (P3-D1 §3: ill-conditioned for any network under
  epsilon prediction at abar_99 = 2.43e-7).
* **Path.** :func:`ila.train.select.fly_checkpoint` with the cell's ``(Ta, K, ensemble)``:
  EMA weights, CPU fp32 single-sample path, render ``full``, the checkpoint's contract variant.
  Rows carry the cell in their ``Ta``/``K``/``ensemble`` columns.
* **Determinism check.** The reference cell ``Ta = 8, K = 10, off`` is the selection setting;
  its rows must equal the run's stored ``select/<checkpoint>_rows.csv`` cell for cell (text),
  except the three inference columns that selection leaves empty
  (:class:`SweepReproductionError` otherwise). It is flown first.
* **Per cell:** ``n``, ``success``, outcome classes (P8's order), ``n_touchdown``, pooled p95
  closing speed (m/s model scale, touched-down episodes), mean steps, ``n_tunnelled``,
  ``n_detectors_disagree`` (:func:`ila.train.select.summarise_rows`).
* **Outputs.** Rows ``results/phase6/sweeps/<arm>/s<n>/<cell>_rows.csv``; summary
  ``results/dp_sweeps.json`` (one entry per run, with the config SHA-256, checkpoint SHA-256,
  the training commit and the sweep's commit). **Resumable per cell:** a cell whose rows file
  holds the 150 ``select`` episodes in order, flown by this checkpoint at this setting, is
  re-summarised, not re-flown. The summary is rewritten after every cell.

Wall clock (P2-D2: 170 ms per K = 10 call on one CPU thread, about 31 ms encoder + 14 ms per
denoising step; render and physics about 11 ms per control step): with ``dp`` episodes as long
as the teacher's (61.5 steps), one run's 14 flights cost about 3.7 CPU-hours, i.e. 15-25 min at
24 workers; if every episode timed out (361 steps) about 6 times that.
"""

from __future__ import annotations

import csv
import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila.data import episodes as data_episodes
from ila.policies.dp import chunking
from ila.policies.registry import REPO_ROOT, read_checkpoint
from ila.train import manifest as mf
from ila.train.config import sha256_file
from ila.train.select import (
    OUTCOME_ORDER,
    SELECT_SPLIT,
    fly_checkpoint,
    summarise_rows,
    write_rows_csv,
)

__all__ = [
    "INFERENCE_COLUMNS",
    "SCHEDULE_LIMITED_K",
    "SUMMARY_JSON",
    "SWEEPS_ROOT",
    "SweepCell",
    "SweepReproductionError",
    "compare_to_selection",
    "run_sweeps",
    "sweep_cells",
]

#: Where the per-cell rows go (``<arm>/s<n>/<cell>_rows.csv``).
SWEEPS_ROOT: Final[Path] = REPO_ROOT / "results" / "phase6" / "sweeps"

#: The committed summary.
SUMMARY_JSON: Final[Path] = REPO_ROOT / "results" / "dp_sweeps.json"

#: K values read as schedule-limited (P3-D1 §3).
SCHEDULE_LIMITED_K: Final[frozenset[int]] = frozenset({1, 2})

#: Row columns that carry the cell (empty in selection rows); excluded from the comparison.
INFERENCE_COLUMNS: Final[tuple[str, ...]] = ("Ta", "K", "ensemble")


class SweepReproductionError(AssertionError):
    """The ``Ta = 8, K = 10, off`` cell does not reproduce the stored selection rows."""


@dataclass(frozen=True)
class SweepCell:
    """One inference setting.

    Attributes:
        Ta: Executed steps per chunk (with ensembling: the ensemble window), control steps
            (1/30 s model scale).
        K: DDIM denoising steps.
        ensemble: Temporal ensembling (P6-D1).
    """

    Ta: int
    K: int
    ensemble: bool

    @property
    def name(self) -> str:
        """File stem, e.g. ``Ta08_K010_off``."""
        return f"Ta{self.Ta:02d}_K{self.K:03d}_{'ens' if self.ensemble else 'off'}"

    @property
    def schedule_limited(self) -> bool:
        """``K`` in :data:`SCHEDULE_LIMITED_K`."""
        return self.K in SCHEDULE_LIMITED_K

    def flown_as(self) -> SweepCell:
        """The cell actually flown: ``Ta = 1`` with ensembling is ``Ta = 1`` without."""
        if self.ensemble and self.Ta == 1:
            return SweepCell(1, self.K, False)
        return self


def sweep_cells(train_cfg: Mapping[str, Any]) -> tuple[SweepCell, list[SweepCell]]:
    """Return the reference cell and every distinct sweep cell (reference first).

    Args:
        train_cfg: The run's resolved train config (``horizons``, ``inference``,
            ``inference_sweeps``).

    Returns:
        ``(reference, cells)``: the selection setting (config ``Ta``, ``K``, off) and the
        deduplicated cells, reference first, then the ``Ta`` x ensembling block, then ``K``.

    Raises:
        ValueError: If the config's ensembling ``m`` is not :data:`chunking.ENSEMBLE_DECAY`
            or a ``Ta`` is inadmissible.
    """
    sweeps = train_cfg["inference_sweeps"]
    m = float(sweeps["temporal_ensembling"]["m"])
    if m != chunking.ENSEMBLE_DECAY:
        raise ValueError(f"config m = {m}, the student implements {chunking.ENSEMBLE_DECAY}")
    hz = train_cfg["horizons"]
    to, tp = int(hz["n_obs_steps"]), int(hz["horizon"])
    ta0 = int(hz["n_action_steps"])
    k0 = int(train_cfg["inference"]["num_inference_steps"])
    reference = SweepCell(ta0, k0, False)
    cells = [reference]
    for ta in sweeps["Ta"]:
        chunking.ChunkSpec(to, tp, int(ta))
        for ensemble in (False, True):
            cells.append(SweepCell(int(ta), k0, ensemble))
    cells += [SweepCell(ta0, int(k), False) for k in sweeps["K"]]
    seen: set[SweepCell] = set()
    unique = []
    for c in cells:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return reference, unique


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_to_selection(
    sweep_rows: Sequence[Mapping[str, str]], selection_rows: Sequence[Mapping[str, str]]
) -> list[str]:
    """Return the differences between the reference cell's rows and the selection rows.

    Every column is compared as text except :data:`INFERENCE_COLUMNS` (selection leaves them
    empty; the sweep writes the cell).

    Returns:
        Human-readable differences (empty when bit-identical).
    """
    out = []
    if len(sweep_rows) != len(selection_rows):
        return [f"{len(sweep_rows)} rows vs {len(selection_rows)} selection rows"]
    for i, (a, b) in enumerate(zip(sweep_rows, selection_rows, strict=True)):
        if list(a) != list(b):
            out.append(f"row {i}: columns differ")
            continue
        for col in a:
            if col not in INFERENCE_COLUMNS and a[col] != b[col]:
                out.append(f"row {i} ({b.get('episode_seed')}): {col} {a[col]!r} != {b[col]!r}")
    return out


def _rows_valid(
    rows: Sequence[Mapping[str, str]], seeds: Sequence[str], sha: str, cell: SweepCell
) -> bool:
    """Whether stored rows are this cell's complete flight of this checkpoint."""
    return (
        [r["episode_seed"] for r in rows] == list(seeds)
        and all(r["ckpt_sha256"] == sha for r in rows)
        and all(
            (r["Ta"], r["K"], r["ensemble"]) == (str(cell.Ta), str(cell.K), str(cell.ensemble))
            for r in rows
        )
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _selected(run_dir: Path) -> dict[str, Any]:
    sel = mf.read_json(run_dir / "selection.json")
    if sel.get("inference_path") != "cpu_fp32_single_sample" or sel.get("weights") != "ema":
        raise ValueError(f"{run_dir}: selection was not on the CPU fp32 path with EMA weights")
    if sel["list"]["name"] != SELECT_SPLIT:
        raise ValueError(f"{run_dir}: selection list {sel['list']['name']!r}")
    return sel


def run_sweeps(
    run_dir: Path,
    *,
    workers: int = 24,
    out_root: Path = SWEEPS_ROOT,
    summary_json: Path = SUMMARY_JSON,
    cells: Sequence[SweepCell] | None = None,
) -> dict[str, Any]:
    """Fly a run's selected checkpoint under every sweep cell on ``select``.

    Args:
        run_dir: A selected ``dp`` run (``manifest.json``, ``selection.json``, ``select/``).
        workers: Rollout processes (rows do not depend on it).
        out_root: Rows root (``<out_root>/<arm>/s<n>/``).
        summary_json: The summary file (this run's entry is replaced, others kept).
        cells: A subset to fly (tests); default every cell of :func:`sweep_cells`. The
            reference cell is always flown first and checked.

    Returns:
        This run's summary entry.

    Raises:
        SweepReproductionError: If the reference cell differs from the selection rows.
        ValueError: If the run is not a selected ``dp`` run or the checkpoint's bytes differ
            from the selection's SHA-256.
    """
    man = mf.read_json(run_dir / "manifest.json")
    sel = _selected(run_dir)
    cand = sel["selected"]["candidate"]
    ckpt = Path(cand["file"])
    sha = str(cand["sha256"])
    if sha256_file(ckpt) != sha:
        raise ValueError(f"{ckpt}: SHA-256 differs from selection.json")
    payload = read_checkpoint(ckpt)
    if payload.get("kind") != "dp":
        raise ValueError(f"{ckpt}: not a diffusion checkpoint ({payload.get('kind')!r})")
    train_cfg = payload["config"]
    reference, all_cells = sweep_cells(train_cfg)
    todo = all_cells if cells is None else [reference, *[c for c in cells if c != reference]]
    arm, seed = str(man["arm"]), int(man["run_seed"])
    episodes = data_episodes.read_split(SELECT_SPLIT)
    seeds = [str(e.episode_seed) for e in episodes]
    split = data_episodes.SPLITS_DIR / f"{SELECT_SPLIT}.parquet"
    out_dir = out_root / arm / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_rows_file = run_dir / "select" / sel["selected"]["rows_file"]
    entry: dict[str, Any] = {
        "run_dir": _rel(run_dir),
        "arm": arm,
        "run_seed": seed,
        "what": "the selected checkpoint (P3-D1 §4, Ta = 8, K = 10) re-flown on select under "
        "other inference settings; reported only, never used to select (plan Phase 6 item 4)",
        "checkpoint": {
            "file": _rel(ckpt),
            "sha256": sha,
            "global_step": cand["global_step"],
        },
        "config": man["config"],
        "train_git": man["git"],
        "sweep_git": mf.git_state(),
        "list": {
            "name": SELECT_SPLIT,
            "file": split.name,
            "file_sha256": data_episodes.file_sha256(split),
            "episodes": len(episodes),
        },
        "inference_path": "cpu_fp32_single_sample",
        "weights": "ema",
        "render": "full",
        "workers": workers,
        "outcome_order": list(OUTCOME_ORDER),
        "units": "p95_closing_speed_m_s: m/s model scale (lambda = 1/25), pooled over "
        "touched-down episodes; mean_steps: 30 Hz control steps",
        "reference_cell": reference.name,
        "selection_rows": {"file": _rel(sel_rows_file), "sha256": _file_sha256(sel_rows_file)},
        "cells": {},
    }
    t_run = time.perf_counter()
    for cell in todo:
        flown = cell.flown_as()
        path = out_dir / f"{flown.name}_rows.csv"
        if flown != cell:
            # Alias: flown under the equivalent cell (P6-D1); fly that one if still missing.
            if flown.name not in entry["cells"]:
                entry["cells"][flown.name] = _fly_cell(
                    ckpt, episodes, seeds, arm, seed, sha, flown, path, workers
                )
            rec = dict(entry["cells"][flown.name])
            rec.update(Ta=cell.Ta, K=cell.K, ensemble=cell.ensemble, alias_of=flown.name)
            entry["cells"][cell.name] = rec
        elif cell.name not in entry["cells"]:
            entry["cells"][cell.name] = _fly_cell(
                ckpt, episodes, seeds, arm, seed, sha, cell, path, workers
            )
        if cell == reference:
            diffs = compare_to_selection(_read_rows(path), _read_rows(sel_rows_file))
            entry["reproduces_selection_rows"] = not diffs
            if diffs:
                entry["reproduction_diffs"] = diffs[:20]
                _write_summary(summary_json, arm, seed, entry)
                raise SweepReproductionError(
                    f"{run_dir}: {reference.name} differs from {sel_rows_file.name}: {diffs[:3]}"
                )
        entry["wall_clock_s"] = time.perf_counter() - t_run
        _write_summary(summary_json, arm, seed, entry)
        c = entry["cells"][cell.name]
        print(
            f"{arm}/s{seed} {cell.name}: success {c['success']}/{c['n']} "
            f"outcomes {c['outcomes']} p95 {c['p95_closing_speed_m_s']} "
            f"({'reused' if c.get('reused') else 'flown'}, {c.get('wall_s', 0.0):.0f} s)",
            flush=True,
        )
    return entry


def _fly_cell(
    ckpt: Path,
    episodes: Sequence[Any],
    seeds: Sequence[str],
    arm: str,
    seed: int,
    sha: str,
    cell: SweepCell,
    path: Path,
    workers: int,
) -> dict[str, Any]:
    """Fly (or reuse) one cell and return its summary record."""
    reused = False
    t0 = time.perf_counter()
    if path.is_file() and _rows_valid(_read_rows(path), seeds, sha, cell):
        reused = True
    else:
        rows = fly_checkpoint(
            ckpt,
            episodes,
            arm=arm,
            run_seed=seed,
            sha256=sha,
            workers=workers,
            inference=(cell.Ta, cell.K, cell.ensemble),
        )
        tmp = path.with_suffix(".tmp")
        write_rows_csv(tmp, rows)
        tmp.replace(path)
        if not _rows_valid(_read_rows(path), seeds, sha, cell):
            raise ValueError(f"{path}: rows written do not match the cell")
    summary = summarise_rows(_read_rows(path))
    p95 = summary["p95_closing_speed_m_s"]
    return {
        "Ta": cell.Ta,
        "K": cell.K,
        "ensemble": cell.ensemble,
        "schedule_limited": cell.schedule_limited,
        **summary,
        "success_rate": summary["success"] / summary["n"] if summary["n"] else 0.0,
        "p95_closing_speed_m_s": None if p95 == float("inf") else p95,
        "rows_file": _rel(path),
        "rows_sha256": _file_sha256(path),
        "reused": reused,
        "wall_s": time.perf_counter() - t0,
    }


def _write_summary(path: Path, arm: str, seed: int, entry: Mapping[str, Any]) -> None:
    """Replace this run's entry in the summary JSON (other runs kept)."""
    doc: dict[str, Any] = mf.read_json(path) if path.is_file() else {}
    doc.setdefault(
        "what",
        "dp inference-time sweeps on select (plan Phase 6 item 4; P6-D1): the selected "
        "checkpoint of each run re-flown; reported only, never used for selection",
    )
    runs = dict(doc.get("runs", {}))
    runs[f"{arm}/s{seed}"] = dict(entry)
    doc["runs"] = dict(sorted(runs.items()))
    mf.write_json_atomic(path, doc)
