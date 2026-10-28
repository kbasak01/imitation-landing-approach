"""P3-D1 §5 training-budget projection from the timing runs (Phase 4).

Before the first real Phase 4 run, a 2 000-step ``dp`` seed-0 run and a 2 000-step ``bc`` run are
timed (purpose ``timing``, wall clock only, no ``select`` read). This module projects **every
planned training run** (core arms, the DART trials, DAgger, ``dp``, ``modular``, the ablations,
R1) from those measurements and compares the total with **120 GPU-hours**. If it exceeds 120 h,
it computes the single cut P3-D1 §5 allows: one factor for every family, the largest that brings
the projection to <= 120 h (so the smallest cut), with checkpoint cadence and warm-up scaled by
it. It never applies the cut: the factor, the projection and the decision go in a dated
protocol entry (P4-D1), written before any evaluation number exists.

Model per run: ``hours = (steps / rate + overhead) / 3600`` with ``rate`` the timing run's
loop-only steps/s (data loading and startup excluded) and ``overhead`` its non-loop wall clock
(startup, decompressing the store), added once per run and never scaled. Checkpoint saves and
held-out evaluations, which a timing run skips, are not in the rate (stated as an assumption;
about 1 s per checkpoint and 1 s per held-out pass at the measured rates).

Families and their rate sources:

* ``bc`` (``bc``, ``bc_matched``, ``bc_dart`` trials and seeds, ``bc_dagger``, ``modular``, the
  ``bc_*`` ablations): the ``bc`` timing run. ``modular``'s estimator is the ``bc`` encoder with
  an 11-wide head, costed as ``bc`` (assumption).
* ``dp`` (``dp``, the ``dp_*`` ablations, R1): the ``dp`` timing run. R1 (LeRobot's
  ``DiffusionPolicy``, Phase 6 config not yet written) is costed as ``dp`` at ``dp``'s steps
  (assumption).
* ``state`` (``bc_state``): a ``bc_state`` timing manifest if one exists, else the
  ``bc_state`` smoke's measured rate (same model, batch and precision; labelled as such).

Iterated arms: ``bc_matched`` and ``bc_dagger`` add 5 x 10 000 steps after iteration 0, which
is the same-seed ``bc`` run and is not re-trained. DAgger's rollouts and every selection rollout
fly the CPU and are not GPU time (reported separately when measured, never in the 120 h).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila.policies.registry import REPO_ROOT
from ila.train.config import TrainConfig, load_train_yaml

__all__ = [
    "BUDGET_HOURS",
    "PlannedRun",
    "Rate",
    "cut_factor",
    "planned_runs",
    "project",
    "projection_csv",
    "rate_from_manifest",
    "rate_from_smoke",
]

#: P3-D1 §5: the GPU-hours every planned run must fit in.
BUDGET_HOURS: Final[float] = 120.0

_TRAIN: Final[Path] = REPO_ROOT / "configs" / "train"

#: Ablation configs (P3-D1 §3), 3 seeds each, by family.
_ABLATIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "bc": (
        "bc_mix",
        "bc_pid",
        "bc_n100",
        "bc_n300",
        "bc_minus_last_action",
        "bc_plus_clock",
    ),
    "dp": ("dp_mix", "dp_pid", "dp_n100", "dp_n300", "dp_minus_last_action"),
}


@dataclass(frozen=True)
class PlannedRun:
    """One line of the projection.

    Attributes:
        label: What is trained.
        family: Rate family (``bc``, ``dp``, ``state``).
        runs: Number of runs.
        steps: Optimiser steps per run (as configured, before any cut).
        warmup: Warm-up steps per stage (scaled with a cut).
        checkpoint_every: Cadence (scaled with a cut).
        note: Assumption or source.
    """

    label: str
    family: str
    runs: int
    steps: int
    warmup: int
    checkpoint_every: int
    note: str = ""


@dataclass(frozen=True)
class Rate:
    """A measured training rate.

    Attributes:
        family: Rate family.
        steps_per_s: Loop-only optimiser steps per second.
        overhead_s: Non-loop wall clock per run (startup, data), seconds.
        source: Where it was measured.
    """

    family: str
    steps_per_s: float
    overhead_s: float
    source: str


def _cfg(rel: str) -> TrainConfig:
    return TrainConfig.load(_TRAIN / rel)


def planned_runs() -> list[PlannedRun]:
    """Every training run P3-D1 plans, from the frozen configs (steps, seeds, cadence).

    Returns:
        The lines, core arms first.
    """
    out: list[PlannedRun] = []

    def add(label: str, cfg: TrainConfig, runs: int, note: str = "", steps: int = 0) -> None:
        family = "dp" if cfg.kind == "dp" else ("state" if cfg.kind == "state" else "bc")
        out.append(
            PlannedRun(
                label,
                family,
                runs,
                steps or cfg.steps,
                cfg.optim.warmup_steps,
                cfg.checkpoint_every,
                note,
            )
        )

    out.append(PlannedRun("timing bc (2k)", "bc", 1, 2000, 500, 0, "P3-D1 §5, already spent"))
    out.append(PlannedRun("timing dp (2k)", "dp", 1, 2000, 500, 0, "P3-D1 §5, already spent"))
    bc_state, bc = _cfg("bc_state.yaml"), _cfg("bc.yaml")
    add("bc_state", bc_state, len(bc_state.seeds))
    add("bc", bc, len(bc.seeds))
    matched = _cfg("bc_matched.yaml")
    it = matched.iterations
    assert it is not None
    iter_steps = it.n * it.steps_per_iteration
    add(
        "bc_matched (iterations 1-5)",
        matched,
        len(matched.seeds),
        "iteration 0 = the bc run (not re-trained)",
        steps=iter_steps,
    )
    dart = _cfg("bc_dart.yaml")
    add("bc_dart sigma trials", dart, 3, "seed 0 at each sigma, tuning (P3-D1 §4)")
    add("bc_dart", dart, len(dart.seeds), "sigma*, seed 0 re-trained")
    # bc_dagger.yaml's per-source success filter is Phase 5's; its schedule is bc_matched's
    # (P3-D1 §3), so it is costed from bc_matched's iterations and its own seed list.
    dagger_raw, _ = load_train_yaml(_TRAIN / "bc_dagger.yaml")
    add(
        "bc_dagger (iterations 1-5)",
        matched,
        len(dagger_raw["seeds"]),
        "bc_matched's schedule (P3-D1 §3); rollouts on the CPU, not counted",
        steps=iter_steps,
    )
    dp = _cfg("dp.yaml")
    add("dp", dp, len(dp.seeds))
    modular = _cfg("modular.yaml")
    add("modular", modular, len(modular.seeds), "estimator costed as the bc encoder")
    for family, arms in _ABLATIONS.items():
        for arm in arms:
            c = _cfg(f"ablations/{arm}.yaml")
            assert (c.kind == "dp") == (family == "dp"), arm
            add(f"ablation {arm}", c, len(c.seeds))
    out.append(
        PlannedRun(
            "R1_lerobot_dp",
            "dp",
            3,
            dp.steps,
            dp.optim.warmup_steps,
            dp.checkpoint_every,
            "Phase 6 config not written: costed as dp at dp's steps (assumption)",
        )
    )
    return out


def rate_from_manifest(path: Path, family: str) -> Rate:
    """Read a timing run's manifest.

    Raises:
        ValueError: If it is not a finished timing run.
    """
    import json

    man: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if man.get("purpose") != "timing" or man.get("state") != "done":
        raise ValueError(f"{path}: not a finished timing run")
    loop_s = float(man["loop_s_total"])
    steps = int(man["steps_timed_total"])
    overhead = max(0.0, float(man["wall_clock_s"]) - loop_s)
    gpu = man.get("host", {}).get("gpu", "?")
    return Rate(family, steps / loop_s, overhead, f"{path} ({steps} steps, {gpu})")


def rate_from_smoke(path: Path, family: str) -> Rate:
    """Read a smoke summary's training rate (used for ``bc_state`` without a timing run)."""
    import json

    rec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    train = rec["train"]
    overhead = max(0.0, float(train["wall_s"]) - float(train["loop_s"]))
    return Rate(family, float(train["steps_per_s"]), overhead, f"{path} (smoke, 8 episodes)")


def project(
    runs: Sequence[PlannedRun], rates: Mapping[str, Rate], factor: float = 1.0
) -> list[dict[str, Any]]:
    """Project GPU-hours per line at a step factor (timing runs are never scaled).

    Raises:
        KeyError: If a family has no rate.
    """
    rows = []
    for r in runs:
        rate = rates[r.family]
        f = 1.0 if r.label.startswith("timing") else factor
        steps = int(round(r.steps * f))
        hours = r.runs * (steps / rate.steps_per_s + rate.overhead_s) / 3600.0
        rows.append(
            {
                "label": r.label,
                "family": r.family,
                "runs": r.runs,
                "steps_per_run": steps,
                "warmup": int(round(r.warmup * f)),
                "checkpoint_every": int(round(r.checkpoint_every * f)),
                "steps_per_s": rate.steps_per_s,
                "overhead_s_per_run": rate.overhead_s,
                "gpu_hours": hours,
                "rate_source": rate.source,
                "note": r.note,
            }
        )
    return rows


def cut_factor(
    runs: Sequence[PlannedRun], rates: Mapping[str, Rate], budget_h: float = BUDGET_HOURS
) -> float:
    """Return P3-D1 §5's factor: 1 if the projection fits, else the largest factor that does.

    The factor is rounded **down** to 3 decimals (so the cut projection is <= the budget).

    Raises:
        ValueError: If even the unscalable part (timing runs, overheads) exceeds the budget.
    """
    fixed = 0.0
    scalable = 0.0
    for r in runs:
        rate = rates[r.family]
        fixed += r.runs * rate.overhead_s / 3600.0
        per = r.runs * r.steps / rate.steps_per_s / 3600.0
        if r.label.startswith("timing"):
            fixed += per
        else:
            scalable += per
    if fixed + scalable <= budget_h:
        return 1.0
    if fixed >= budget_h:
        raise ValueError(f"the unscalable part alone ({fixed:.1f} h) exceeds {budget_h} h")
    return math.floor((budget_h - fixed) / scalable * 1000.0) / 1000.0


def projection_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows (plus a TOTAL line) as CSV text."""
    buf = io.StringIO()
    fields = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    total = sum(float(r["gpu_hours"]) for r in rows)
    w.writerow({**dict.fromkeys(fields, ""), "label": "TOTAL", "gpu_hours": f"{total:.4f}"})
    return buf.getvalue()
