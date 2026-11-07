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
* successes within :data:`NEAR_FRACTION` (5 %) of a success-criterion threshold, episode by
  episode. The near test uses the touchdown criteria only: closing speed along the deck normal,
  lateral offset and relative tilt. World-z closing speed is printed beside them but is not a
  criterion (``vz_component: deck_normal``), so it is not in the test. Dwell is a criterion but
  is not a margin: a success ends at ``dwell_complete``, so its dwell sits just above 0.5 s by
  construction; the observed dwell range and any success not ending at ``dwell_complete`` are
  reported instead (:data:`DWELL_NOTE`);
* tunnelling and detector-disagreement counts, and which episodes;
* a per-seed stricter recount, student and teacher: successes, then successes minus tunnelled
  and detector-disagreement successes, then also minus near-threshold successes; and the exact
  McNemar p (:func:`ila.eval.metrics.mcnemar_exact_p`) from each seed's discordant counts
  against the teacher, descriptive only;
* grace-used successes: successes with more than one contact event (``contact_events > 1``;
  P8 counts a new event only when contact returns after a gap longer than ``success.yaml``'s
  ``contact_loss_grace_s``, 50 ms model scale); counted, not removed from any recount;
* the outcome-class shift: a teacher-class x student-class table pooled over seeds, and every
  episode the student turned from a teacher failure into a success, and vice versa. For ``T_mix``
  cells each such episode also carries its assigned policy (``ppo`` or ``gated``) and the other
  teacher's committed outcome, with per-seed counts by assigned policy.

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
from ila.eval.metrics import mcnemar_exact_p

__all__ = [
    "BINS",
    "DWELL_NOTE",
    "NEAR_FRACTION",
    "REVIEWED_JSON",
    "audit",
    "recount",
    "render_md",
    "run_audits",
    "thresholds",
]

#: The committed results-skeptic review record beside the audits; never an audit file.
REVIEWED_JSON: Final[str] = "reviewed.json"

#: A success is "near" a threshold when its margin is below this fraction of the threshold.
NEAR_FRACTION: Final[float] = 0.05

#: Margin bins, as fractions of the threshold (left-closed; < 0 = over the threshold).
BINS: Final[tuple[float, ...]] = (-math.inf, 0.0, 0.05, 0.10, 0.25, 0.50, math.inf)

#: Metric -> (row column or derivation, success.yaml key, unit, is a success criterion).
#: World-z closing speed is printed beside the deck-normal criterion but is not one
#: (``success.yaml`` ``vz_component: deck_normal``), so the near test never uses it.
_METRICS: Final[tuple[tuple[str, str, str, bool], ...]] = (
    ("closing_speed_normal_m_s", "rel_vertical_velocity_max_m_s", "m/s model scale", True),
    ("closing_speed_world_z_m_s", "rel_vertical_velocity_max_m_s", "m/s model scale", False),
    ("lateral_offset_m", "lateral_offset_max_m", "m model scale", True),
    ("rel_tilt_deg", "relative_tilt_max_deg", "deg", True),
)

#: Why dwell is reported but not in the near test.
DWELL_NOTE: Final[str] = (
    "dwell (contact >= 0.5 s) is a success criterion but not in the near test: a success ends "
    "at termination_reason dwell_complete, so its dwell sits just above 0.5 s by construction "
    "and carries no landing-quality margin; the observed range and the count of successes not "
    "ending at dwell_complete are reported instead"
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


def _ep_detail(row: Mapping[str, Any], seed_col: str) -> dict[str, Any]:
    """:func:`_ep` plus the touchdown detail of a tunnelled or disagreement episode.

    Times are seconds of episode time and penetration metres, model scale; the two touchdown
    times are the contact and analytic detectors' (NaN when that detector did not fire).
    """
    return {
        **_ep(row, seed_col),
        "td_t_episode_contact_s": sc.as_float(row["td_t_episode_s"]),
        "td_t_episode_analytic_s": sc.as_float(row["td_t_episode_analytic_s"]),
        "max_penetration_m": sc.as_float(row["max_penetration_m"]),
        "closing_speed_normal_m_s": _value(row, "closing_speed_normal_m_s")
        if sc.as_bool(row["touchdown_contact"])
        else math.nan,
        "lateral_offset_m": sc.as_float(row["lateral_offset_m"]),
        "rel_tilt_deg": sc.as_float(row["rel_tilt_deg"]),
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
    for metric, key, unit, criterion in _METRICS:
        lim = limits[key]
        vals = np.asarray([_value(r, metric) for r in touched], dtype=np.float64)
        margins = (lim - vals) / lim
        hist = np.histogram(margins, bins=np.asarray(BINS))[0] if vals.size else np.zeros(6)
        out["metrics"][metric] = {
            "unit": unit,
            "success_criterion": criterion,
            "in_near_test": criterion,
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
        if not criterion:
            continue
        for r, m in zip(touched, margins, strict=True):
            if str(r["outcome"]) == "success" and m < NEAR_FRACTION:
                near.append({**_ep(r, seed_col), "metric": metric, "margin_fraction": float(m)})
    succ = [r for r in rows if str(r["outcome"]) == "success"]
    dwell = [sc.as_float(r["dwell_s"]) for r in succ]
    out["successes"] = len(succ)
    out["success_min_dwell_s"] = min(dwell) if dwell else math.nan
    out["success_dwell"] = {
        "unit": "s model scale",
        "success_criterion": True,
        "in_near_test": False,
        "threshold": limits["contact_dwell_min_s"],
        "min": min(dwell) if dwell else math.nan,
        "max": max(dwell) if dwell else math.nan,
        "n_successes_not_ending_dwell_complete": sum(
            str(r["termination_reason"]) != "dwell_complete" for r in succ
        ),
        "note": DWELL_NOTE,
    }
    out["successes_near_threshold"] = sorted(
        near, key=lambda d: (d["seed"], d["index"], d["metric"])
    )
    out["n_successes_near_threshold"] = len({(d["seed"], d["index"]) for d in near})
    out["tunnelled"] = [_ep_detail(r, seed_col) for r in rows if sc.as_bool(r["tunnelled"])]
    out["detectors_disagree"] = [
        _ep_detail(r, seed_col) for r in rows if sc.as_bool(r["detectors_disagree"])
    ]
    out["n_tunnelled"] = len(out["tunnelled"])
    out["n_detectors_disagree"] = len(out["detectors_disagree"])
    out["grace_used_successes"] = [_ep(r, seed_col) for r in succ if _grace_used(r)]
    out["n_grace_used_successes"] = len(out["grace_used_successes"])
    return out


def _grace_used(row: Mapping[str, Any]) -> bool:
    """A success with more than one P8 contact event (``contact_events > 1``)."""
    return str(row["outcome"]) == "success" and int(float(row["contact_events"])) > 1


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
    # T_mix: which policy each episode was assigned, and the other teacher's committed row.
    assigned: dict[int, str] = {}
    other: dict[int, Mapping[str, Any]] = {}
    if teacher == "T_mix":
        mix = sc.tmix_reference(inp.listed(group.list_name, group.ss), inp.baselines)
        alt = {
            p: {int(r["index"]): r for r in inp.baselines.cell(p, group.list_name, group.ss)}
            for p in _TMIX_POLICIES
        }
        for policy, r in mix:
            i = int(r["index"])
            assigned[i] = policy
            other[i] = alt[_other_policy(policy)][i]
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
        ep: dict[str, Any] = {**_ep(r, "student_seed"), "teacher_outcome": to}
        if assigned:
            i = int(r["index"])
            ep["assigned_policy"] = assigned[i]
            ep["other_teacher"] = _other_policy(assigned[i])
            ep["other_teacher_outcome"] = str(other[i]["outcome"])
        if to != "success" and so == "success":
            gained.append(ep)
        elif to == "success" and so != "success":
            lost.append(ep)
    flagged = _flagged_seeds(row.get("seeds_above", ""))
    t_count = recount(trows, limits)
    per_seed = []
    for c in cells:
        rows_c = list(c.rows or ())
        s_count = recount(rows_c, limits)
        b = sum(
            str(r["outcome"]) == "success" and str(t_by[int(r["index"])]["outcome"]) != "success"
            for r in rows_c
        )
        d = sum(
            str(r["outcome"]) != "success" and str(t_by[int(r["index"])]["outcome"]) == "success"
            for r in rows_c
        )
        rec: dict[str, Any] = {
            "seed": c.key.seed,
            "flagged_above_teacher": c.key.seed in flagged,
            "student": s_count,
            "teacher": t_count,
            "discordant_teacher_failure_to_student_success": b,
            "discordant_teacher_success_to_student_failure": d,
            "mcnemar_exact_p_two_sided": mcnemar_exact_p(b, d),
        }
        if assigned:
            rec["by_assigned_policy"] = {
                p: {
                    "n_episodes": sum(assigned[int(r["index"])] == p for r in rows_c),
                    "gained": sum(
                        e["seed"] == c.key.seed and e["assigned_policy"] == p for e in gained
                    ),
                    "lost": sum(
                        e["seed"] == c.key.seed and e["assigned_policy"] == p for e in lost
                    ),
                    "student_successes": sum(
                        assigned[int(r["index"])] == p and str(r["outcome"]) == "success"
                        for r in rows_c
                    ),
                    "reference_successes": sum(
                        v == p and str(t_by[i]["outcome"]) == "success" for i, v in assigned.items()
                    ),
                }
                for p in _TMIX_POLICIES
            }
        per_seed.append(rec)
    out_mix: dict[str, Any] = {}
    if assigned:
        out_mix = {
            "tmix_assignment": {p: sum(v == p for v in assigned.values()) for p in _TMIX_POLICIES},
            "tmix_reference_outcomes_by_policy": {
                p: {
                    o: sum(v == p and str(t_by[i]["outcome"]) == o for i, v in assigned.items())
                    for o in sc.OUTCOMES
                }
                for p in _TMIX_POLICIES
            },
        }
    return {
        **out_mix,
        "per_seed_recount": per_seed,
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


_TMIX_POLICIES: Final[tuple[str, str]] = ("ppo", "gated")


def _other_policy(policy: str) -> str:
    return "gated" if policy == "ppo" else "ppo"


def _flagged_seeds(seeds_above: Any) -> frozenset[int]:
    """Seeds named in an ``above_teacher.csv`` ``seeds_above`` cell (``s0:0.03|s1:0.07``)."""
    out = set()
    for tok in str(seeds_above or "").split("|"):
        if tok:
            out.add(int(tok.split(":")[0].removeprefix("s")))
    return frozenset(out)


def _near_episodes(rows: Sequence[Mapping[str, Any]], limits: Mapping[str, float]) -> set[int]:
    """Indices of successes within :data:`NEAR_FRACTION` of a touchdown-criterion threshold."""
    near = set()
    for r in _touched(rows):
        if str(r["outcome"]) != "success":
            continue
        for metric, key, _, criterion in _METRICS:
            lim = limits[key]
            if criterion and (lim - _value(r, metric)) / lim < NEAR_FRACTION:
                near.add(int(r["index"]))
    return near


def recount(rows: Sequence[Mapping[str, Any]], limits: Mapping[str, float]) -> dict[str, int]:
    """The stricter success recount of one run's rows in one cell (episodes).

    Args:
        rows: One seed's rows (or the teacher's single row set) in one cell.
        limits: :func:`thresholds`.

    Returns:
        ``successes``; tunnelled and detector-disagreement successes; ``successes`` minus the
        union of those two; near-threshold successes (touchdown criteria only, module
        docstring); ``successes`` minus the union of all three; and grace-used successes
        (``contact_events > 1``, counted only, removed from nothing).
    """
    succ = {int(r["index"]) for r in rows if str(r["outcome"]) == "success"}
    tun = {int(r["index"]) for r in rows if sc.as_bool(r["tunnelled"])} & succ
    dis = {int(r["index"]) for r in rows if sc.as_bool(r["detectors_disagree"])} & succ
    near = _near_episodes(rows, limits)
    grace = {int(r["index"]) for r in rows if _grace_used(r)}
    return {
        "n_episodes": len(rows),
        "grace_used_successes": len(grace),
        "successes": len(succ),
        "tunnelled_successes": len(tun),
        "disagreement_successes": len(dis),
        "successes_minus_tunnelled_or_disagreement": len(succ - tun - dis),
        "near_threshold_successes": len(near),
        "successes_minus_tunnelled_disagreement_or_near": len(succ - tun - dis - near),
    }


def _m(side: Mapping[str, Any], metric: str, key: str) -> str:
    return sc.fmt(side["metrics"][metric][key])


def _recount_md(a: Mapping[str, Any]) -> list[str]:
    """The per-seed stricter recount, McNemar p and (``T_mix``) mixture decomposition."""
    lines = [
        "",
        "Per-seed recount (successes of N; student / teacher). A = as scored; B = minus "
        "tunnelled and detector-disagreement successes; C = B also minus near-threshold "
        "successes. Grace-used: successes with contact_events > 1 (P8 counts a new contact "
        "event only after a gap longer than the contact-loss grace), counted only, removed "
        "from nothing. McNemar: exact two-sided p from the "
        "discordant counts (teacher failure -> student success / the reverse), descriptive, no "
        "multiplicity correction.",
        "",
        "| seed | flagged | A | B | C | grace-used | discordant +/- | McNemar p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in a["per_seed_recount"]:
        s, t = p["student"], p["teacher"]
        lines.append(
            f"| s{p['seed']} | {'yes' if p['flagged_above_teacher'] else 'no'} | "
            f"{s['successes']} / {t['successes']} | "
            f"{s['successes_minus_tunnelled_or_disagreement']} / "
            f"{t['successes_minus_tunnelled_or_disagreement']} | "
            f"{s['successes_minus_tunnelled_disagreement_or_near']} / "
            f"{t['successes_minus_tunnelled_disagreement_or_near']} | "
            f"{s['grace_used_successes']} / {t['grace_used_successes']} | "
            f"+{p['discordant_teacher_failure_to_student_success']} / "
            f"-{p['discordant_teacher_success_to_student_failure']} | "
            f"{p['mcnemar_exact_p_two_sided']:.3f} |"
        )
    if "tmix_assignment" in a:
        asg = a["tmix_assignment"]
        ref = a["tmix_reference_outcomes_by_policy"]
        lines += [
            "",
            "T_mix decomposition by assigned policy (assign_rows over P8 `ppo` s0 + `gated`): "
            + "; ".join(
                f"{p} {asg[p]} episodes, reference "
                + ", ".join(f"{o} {n}" for o, n in ref[p].items() if n)
                for p in _TMIX_POLICIES
            )
            + ".",
            "",
            "| seed | policy | episodes | student successes | reference successes | gained "
            "| lost |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in a["per_seed_recount"]:
            for pol, d in p["by_assigned_policy"].items():
                lines.append(
                    f"| s{p['seed']} | {pol} | {d['n_episodes']} | {d['student_successes']} | "
                    f"{d['reference_successes']} | {d['gained']} | {d['lost']} |"
                )
        gained = a["outcome_shift"]["teacher_failure_to_student_success"]
        combos = Counter(
            (e["assigned_policy"], e["teacher_outcome"], e["other_teacher_outcome"]) for e in gained
        )
        lines += ["", "Teacher failure -> student success, by (assigned policy, its outcome, "
                  "the other teacher's outcome):", ""]  # fmt: skip
        lines += [
            f"- {pol}: {to}; other teacher: {oo} -- {n}"
            for (pol, to, oo), n in sorted(combos.items())
        ]
    return lines


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
            f"- Successes within {int(100 * a['near_fraction'])} % of a success-criterion "
            "threshold (closing speed along the deck normal, lateral offset, relative tilt; "
            "world-z closing speed and dwell not in the test): student "
            f"{s['n_successes_near_threshold']} of {s['successes']}, teacher "
            f"{t['n_successes_near_threshold']} of {t['successes']}; shortest successful dwell "
            f"student {sc.fmt(s['success_min_dwell_s'])} s, teacher "
            f"{sc.fmt(t['success_min_dwell_s'])} s.",
            f"- Tunnelling: student {s['n_tunnelled']}, teacher {t['n_tunnelled']}. Detector "
            f"disagreement: student {s['n_detectors_disagree']}, teacher "
            f"{t['n_detectors_disagree']}. Grace-used successes (contact_events > 1): student "
            f"{s['n_grace_used_successes']} of {s['successes']}, teacher "
            f"{t['n_grace_used_successes']} of {t['successes']}.",
            f"- Teacher failure -> student success: "
            f"{len(a['outcome_shift']['teacher_failure_to_student_success'])} seed-episodes; "
            f"teacher success -> student failure: "
            f"{len(a['outcome_shift']['teacher_success_to_student_failure'])}.",
            "",
            "| metric | threshold | student p50 / p95 / max | over | teacher p50 / p95 / max "
            "| over |",
            "|---|---|---|---|---|---|",
        ]
        for metric, _, unit, criterion in _METRICS:
            name = metric if criterion else f"{metric}, not a criterion"
            lines.append(
                f"| {name} ({unit}) | {_m(s, metric, 'threshold')} | "
                f"{_m(s, metric, 'p50')} / {_m(s, metric, 'p95')} / {_m(s, metric, 'max')} | "
                f"{_m(s, metric, 'n_over_threshold')} | {_m(t, metric, 'p50')} / "
                f"{_m(t, metric, 'p95')} / {_m(t, metric, 'max')} | "
                f"{_m(t, metric, 'n_over_threshold')} |"
            )
        sd = s["success_dwell"]
        lines += [
            "",
            f"Dwell (criterion, not in the near test): successful dwell {sc.fmt(sd['min'])}–"
            f"{sc.fmt(sd['max'])} s (student), {sc.fmt(t['success_dwell']['min'])}–"
            f"{sc.fmt(t['success_dwell']['max'])} s (teacher); successes not ending at "
            f"dwell_complete: student {sd['n_successes_not_ending_dwell_complete']}, teacher "
            f"{t['success_dwell']['n_successes_not_ending_dwell_complete']}.",
        ]
        lines += _recount_md(a)
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
        if stale.name != REVIEWED_JSON and stale.stem not in slugs:
            stale.unlink()  # a cell no longer above its teacher keeps no audit
    for r, slug in zip(rows, slugs, strict=True):
        a = audit(by_label[slug], index, inp, r)
        sc.dump_json(target / f"{slug}.json", a)
        audits.append(a)
    (target / "audit.md").write_text(render_md(audits), encoding="utf-8")
    return audits
