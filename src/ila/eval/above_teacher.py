"""Touchdown audit of every student cell above its own teacher (CLAUDE.md non-negotiable 8).

Simulation only (PyBullet physics, PyBullet TinyRenderer frames). Speeds m/s, lengths m, times
s, all model scale (lambda = 1/25: x5 for speeds and times, x25 for lengths at full scale);
angles degrees.

For each row of ``results/e08/above_teacher.csv`` (:func:`ila.eval.score.above_teacher_rows`),
:func:`audit` compares the student's rows in that cell (every present seed) with its teacher's
committed rows in the same cell (P8 ``ppo`` seed 0 for ``T_ppo`` arms, ``pid_feedforward`` for
``*_pid``, the ``T_mix`` reference for ``*_mix``) and records, against the frozen success
criteria read from P8's ``configs/env/success.yaml`` (closing speed along the deck normal
<= 0.5 m/s, lateral offset <= 0.10 m, relative tilt <= 15 deg, contact >= 0.5 s):

* touchdown distributions (contact-detected touchdowns only, never imputed): closing speed
  along the deck normal and along world z, lateral offset, relative tilt -- p50 / p95 / max and
  the count over each threshold;
* margin-to-threshold histograms, margin = (threshold - value) / threshold, in fixed bins;
* successes within :data:`NEAR_FRACTION` (5 %) of any threshold, episode by episode, and the
  shortest successful dwell;
* tunnelling and detector-disagreement counts, and which episodes;
* the outcome-class shift: a teacher-class x student-class table pooled over seeds, and every
  episode the student turned from a teacher failure into a success, and vice versa.

Nothing here decides whether the cell is reported; the audit goes to ``results-skeptic``
first. Outputs: ``above_teacher/<group slug>.json`` and ``above_teacher/audit.md``, both
deterministic.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import score as sc
from ila.eval import stats as st

__all__ = ["BINS", "NEAR_FRACTION", "audit", "render_md", "run_audits", "thresholds"]

#: A success is "near" a threshold when its margin is below this fraction of the threshold.
NEAR_FRACTION: Final[float] = 0.05

#: Margin bins, as fractions of the threshold (left-closed; < 0 = over the threshold).
BINS: Final[tuple[float, ...]] = (-math.inf, 0.0, 0.05, 0.10, 0.25, 0.50, math.inf)

#: Metric -> (row column or derivation, success.yaml key, unit).
_METRICS: Final[tuple[tuple[str, str, str], ...]] = (
    ("closing_speed_normal_m_s", "rel_vertical_velocity_max_m_s", "m/s model scale"),
    ("closing_speed_world_z_m_s", "rel_vertical_velocity_max_m_s", "m/s model scale"),
    ("lateral_offset_m", "lateral_offset_max_m", "m model scale"),
    ("rel_tilt_deg", "relative_tilt_max_deg", "deg"),
)


def thresholds() -> dict[str, float]:
    """The four success thresholds from P8's frozen ``success.yaml`` (SHA-256 ``c7fbdcc4…``)."""
    raw = yaml.safe_load(p8_adapter.success_yaml_path().read_text(encoding="utf-8"))
    keys = (
        "rel_vertical_velocity_max_m_s",
        "lateral_offset_max_m",
        "relative_tilt_max_deg",
        "contact_dwell_min_s",
    )
    return {k: float(raw[k]) for k in keys}


def _value(row: Mapping[str, Any], metric: str) -> float:
    """A touchdown metric of a touched-down row (P8's closing-speed construction)."""
    if metric == "closing_speed_normal_m_s":
        return max(0.0, -sc.as_float(row["rel_vz_normal_m_s"]))
    if metric == "closing_speed_world_z_m_s":
        return max(0.0, -sc.as_float(row["rel_vz_world_z_m_s"]))
    return sc.as_float(row[metric])


def _touched(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if sc.as_bool(r["touchdown_contact"])]


def _ep(row: Mapping[str, Any], seed_col: str) -> dict[str, Any]:
    return {
        "seed": int(float(row[seed_col])),
        "index": int(row["index"]),
        "episode_seed": int(row["episode_seed"]),
        "outcome": str(row["outcome"]),
    }


def _side(
    rows: Sequence[Mapping[str, Any]], seed_col: str, limits: Mapping[str, float]
) -> dict[str, Any]:
    touched = _touched(rows)
    out: dict[str, Any] = {
        "n_episodes": len(rows),
        "n_touchdowns": len(touched),
        "outcomes": {o: sum(str(r["outcome"]) == o for r in rows) for o in sc.OUTCOMES},
        "metrics": {},
    }
    near: list[dict[str, Any]] = []
    for metric, key, unit in _METRICS:
        lim = limits[key]
        vals = np.asarray([_value(r, metric) for r in touched], dtype=np.float64)
        margins = (lim - vals) / lim
        hist = np.histogram(margins, bins=np.asarray(BINS))[0] if vals.size else np.zeros(6)
        out["metrics"][metric] = {
            "unit": unit,
            "threshold": lim,
            "p50": st.quantile(vals, 0.5) if vals.size else math.nan,
            "p95": st.quantile(vals, 0.95) if vals.size else math.nan,
            "max": float(vals.max()) if vals.size else math.nan,
            "n_over_threshold": int(np.sum(vals > lim)),
            "margin_histogram": {
                "bin_edges_fraction_of_threshold": [sc.fmt(b) for b in BINS],
                "counts": [int(c) for c in hist],
            },
        }
        for r, m in zip(touched, margins, strict=True):
            if str(r["outcome"]) == "success" and m < NEAR_FRACTION:
                near.append({**_ep(r, seed_col), "metric": metric, "margin_fraction": float(m)})
    succ = [r for r in rows if str(r["outcome"]) == "success"]
    dwell = [sc.as_float(r["dwell_s"]) for r in succ]
    out["successes"] = len(succ)
    out["success_min_dwell_s"] = min(dwell) if dwell else math.nan
    out["successes_near_threshold"] = sorted(
        near, key=lambda d: (d["seed"], d["index"], d["metric"])
    )
    out["n_successes_near_threshold"] = len({(d["seed"], d["index"]) for d in near})
    out["tunnelled"] = [_ep(r, seed_col) for r in rows if sc.as_bool(r["tunnelled"])]
    out["detectors_disagree"] = [
        _ep(r, seed_col) for r in rows if sc.as_bool(r["detectors_disagree"])
    ]
    out["n_tunnelled"] = len(out["tunnelled"])
    out["n_detectors_disagree"] = len(out["detectors_disagree"])
    return out


def audit(
    group: sc.GroupKey,
    index: sc.CellIndex,
    inp: sc.ScoreInputs,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Touchdown audit of one above-teacher cell (module docstring).

    Args:
        group: The student cell.
        index: :func:`ila.eval.score.load_cells`.
        inp: The score inputs (baselines, listed episodes).
        row: Its ``above_teacher.csv`` row (string or typed cells).

    Returns:
        The JSON-serialisable audit.
    """
    limits = thresholds()
    teacher = ev_arms.expected_teacher(group.arm)
    trows = sc.teacher_rows(teacher, group.list_name, group.ss, inp.baselines, inp.listed)
    t_by = {int(r["index"]): r for r in trows}
    cells = [c for c in index.group_cells(group) if c.rows is not None]
    srows = [r for c in cells for r in (c.rows or ())]
    table: Counter[tuple[str, str]] = Counter()
    gained, lost = [], []
    for r in srows:
        t = t_by[int(r["index"])]
        if int(t["episode_seed"]) != int(r["episode_seed"]):
            raise sc.ScoreInputError(f"{group.label}: teacher and student episodes differ")
        to, so = str(t["outcome"]), str(r["outcome"])
        table[(to, so)] += 1
        ep = {**_ep(r, "student_seed"), "teacher_outcome": to}
        if to != "success" and so == "success":
            gained.append(ep)
        elif to == "success" and so != "success":
            lost.append(ep)
    return {
        "simulation_only": "PyBullet physics; model scale lambda = 1/25",
        "cell": group.record(),
        "label": group.label,
        "above_teacher_row": {k: sc.fmt(v) for k, v in row.items()},
        "teacher": teacher,
        "teacher_source": row.get("teacher_source", ""),
        "seeds": [c.key.seed for c in cells],
        "thresholds_success_yaml": limits,
        "success_yaml_sha256": p8_adapter.success_yaml_sha256(),
        "near_fraction": NEAR_FRACTION,
        "student": _side(srows, "student_seed", limits),
        "teacher_rows": _side(trows, "run_seed", limits),
        "outcome_shift": {
            "table_teacher_by_student_pooled_over_seeds": [
                {"teacher": a, "student": b, "n": table[(a, b)]}
                for a in sc.OUTCOMES
                for b in sc.OUTCOMES
                if table[(a, b)]
            ],
            "teacher_failure_to_student_success": sorted(
                gained, key=lambda d: (d["index"], d["seed"])
            ),
            "teacher_success_to_student_failure": sorted(
                lost, key=lambda d: (d["index"], d["seed"])
            ),
        },
        "status": "for results-skeptic review; not reported until reviewed",
    }


def _m(side: Mapping[str, Any], metric: str, key: str) -> str:
    return sc.fmt(side["metrics"][metric][key])


def render_md(audits: Sequence[Mapping[str, Any]]) -> str:
    """The combined markdown of every audit (deterministic)."""
    lines = [
        "# Above-teacher touchdown audits (Phase 8) -- simulation only",
        "",
        "Generated by `scripts/above_teacher.py` from `results/e08/above_teacher.csv`, the "
        "committed rows and P8's committed teacher rows. Not reported until `results-skeptic` "
        "has reviewed them (CLAUDE.md non-negotiable 8). Speeds m/s, lengths m, model scale "
        "(lambda = 1/25); angles deg. Touchdown metrics over contact-detected touchdowns only.",
        "",
    ]
    if not audits:
        lines.append("No student cell is above its teacher in the scored rows.")
        return "\n".join(lines) + "\n"
    for a in audits:
        s, t = a["student"], a["teacher_rows"]
        row = a["above_teacher_row"]
        lines += [
            f"## {a['label']}",
            "",
            f"- Teacher: {a['teacher']} ({a['teacher_source']}); "
            f"{row.get('teacher_k')}/{row.get('teacher_n')} = {row.get('teacher_rate')}.",
            f"- Student seeds {', '.join(str(x) for x in a['seeds'])}: IQM "
            f"{row.get('success_iqm') or 'n/a'} (margin {row.get('iqm_margin') or 'n/a'}); "
            f"seeds above: {row.get('seeds_above') or 'none'}.",
            f"- Successes within {int(100 * a['near_fraction'])} % of a threshold: student "
            f"{s['n_successes_near_threshold']} of {s['successes']}, teacher "
            f"{t['n_successes_near_threshold']} of {t['successes']}; shortest successful dwell "
            f"student {sc.fmt(s['success_min_dwell_s'])} s, teacher "
            f"{sc.fmt(t['success_min_dwell_s'])} s.",
            f"- Tunnelling: student {s['n_tunnelled']}, teacher {t['n_tunnelled']}. Detector "
            f"disagreement: student {s['n_detectors_disagree']}, teacher "
            f"{t['n_detectors_disagree']}.",
            f"- Teacher failure -> student success: "
            f"{len(a['outcome_shift']['teacher_failure_to_student_success'])} seed-episodes; "
            f"teacher success -> student failure: "
            f"{len(a['outcome_shift']['teacher_success_to_student_failure'])}.",
            "",
            "| metric | threshold | student p50 / p95 / max | over | teacher p50 / p95 / max "
            "| over |",
            "|---|---|---|---|---|---|",
        ]
        for metric, _, unit in _METRICS:
            lines.append(
                f"| {metric} ({unit}) | {_m(s, metric, 'threshold')} | "
                f"{_m(s, metric, 'p50')} / {_m(s, metric, 'p95')} / {_m(s, metric, 'max')} | "
                f"{_m(s, metric, 'n_over_threshold')} | {_m(t, metric, 'p50')} / "
                f"{_m(t, metric, 'p95')} / {_m(t, metric, 'max')} | "
                f"{_m(t, metric, 'n_over_threshold')} |"
            )
        lines += ["", "Outcome shift (teacher class -> student class, pooled over seeds):", ""]
        for e in a["outcome_shift"]["table_teacher_by_student_pooled_over_seeds"]:
            lines.append(f"- {e['teacher']} -> {e['student']}: {e['n']}")
        lines += ["", f"Details: `{row.get('audit_file')}`.", ""]
    return "\n".join(lines) + "\n"


def run_audits(inp: sc.ScoreInputs, out_dir: Path | None = None) -> list[dict[str, Any]]:
    """Audit every row of ``above_teacher.csv`` under ``out_dir`` (default ``inp.root``).

    Returns:
        The audits, in ``above_teacher.csv`` order.
    """
    out = inp.root if out_dir is None else out_dir
    index = sc.load_cells(inp.root, inp.matrix, episodes_per_cell=inp.episodes_per_cell)
    rows = sc.read_csv(out / "above_teacher.csv")
    by_label = {g.slug: g for g in index.groups}
    audits = []
    target = out / "above_teacher"
    target.mkdir(parents=True, exist_ok=True)
    slugs = [r["audit_file"].removeprefix("above_teacher/").removesuffix(".json") for r in rows]
    for stale in sorted(target.glob("*.json")):
        if stale.stem not in slugs:
            stale.unlink()  # a cell no longer above its teacher keeps no audit
    for r, slug in zip(rows, slugs, strict=True):
        a = audit(by_label[slug], index, inp, r)
        sc.dump_json(target / f"{slug}.json", a)
        audits.append(a)
    (target / "audit.md").write_text(render_md(audits), encoding="utf-8")
    return audits
