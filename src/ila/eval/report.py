"""``results/results.md`` rendered from committed CSV/JSON only (Phase 8).

Simulation only: PyBullet physics and PyBullet TinyRenderer frames; no sentence here refers to
a real camera, deck or flight. Lengths, speeds and times are model scale (lambda = 1/25; x25 for
lengths, x5 for times and speeds at full scale). One control step is 33.3 ms model scale.

Inputs (all under ``results/e08``; nothing is recomputed from rows and no P8 file is read):
``run_info.json``, ``score_info.json``, ``summary.csv``, ``baselines.csv``, ``aggregate.csv``,
``contrasts.csv``, ``hypotheses.json``, ``above_teacher.csv``, ``missing.csv`` and, when the
review exists, ``above_teacher/reviewed.json``. :func:`render` is a pure function of those
bytes (no clock, no environment, no dict-order dependence: every table follows the CSV row
order, which :mod:`ila.eval.score` sorts), so ``make report`` re-renders byte-identically.

Withholding rule (CLAUDE.md non-negotiable 8; eval-auditor rule 5). A student cell listed in
``above_teacher.csv`` is printed as *withheld* -- and so is every contrast or hypothesis whose
``inputs`` include it -- until ``above_teacher/reviewed.json`` releases it::

    {"released": [{"label": "<GroupKey.label>", "reviewed_by": "results-skeptic",
                   "date": "YYYY-MM-DD", "finding": "..."}]}

A released cell prints its number with a ``‡`` marker, and the table it sits in is followed by
its finding as a footnote (and §8 lists every finding). Every student table prints the teacher
and the three P8 baselines (:data:`ila.eval.score.ALWAYS_PRINTED`; CLAUDE.md non-negotiable 7)
beside the students; the ablation table adds ``gated`` and the ``T_mix`` reference for the
``*_mix`` arms.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.eval import score as sc

__all__ = [
    "ARM_NOTES",
    "LOSS_ABBREV",
    "RESULTS_MD",
    "ReportInputs",
    "label_of",
    "load_inputs",
    "render",
    "withheld_inputs",
    "write_report",
]

#: The rendered file.
RESULTS_MD: Final[Path] = p8_adapter.REPO_ROOT / "results" / "results.md"

#: Loss-class abbreviations used in the compact cells (P8's classes, P8's order).
LOSS_ABBREV: Final[Mapping[str, str]] = {
    "crash": "C",
    "off_pad": "O",
    "hard_landing": "H",
    "bounce": "B",
    "timeout": "T",
}

_TEACHER_NAME: Final[Mapping[str, str]] = {
    "T_ppo": "P8 `ppo` seed 0",
    "T_pid": "P8 `pid_feedforward`",
    "T_mix": "`T_mix` (P8 `ppo` seed 0 + `gated`, assign_rows)",
}

#: What each arm is, with its teacher named (claim-trace: teacher named for every student).
ARM_NOTES: Final[Mapping[str, str]] = {
    "bc_state": "state input (P8's 25-d obs incl. `time_fraction`), BC",
    "bc": "vision BC, 1 000 demos",
    "bc_matched": "vision BC, 2 000 demos on `bc_dagger`'s schedule",
    "bc_dart": "vision BC on DART σ* = 0.10 demos",
    "bc_dagger": "vision BC + DAgger, 2 000 labelled episodes",
    "dp": "vision Diffusion Policy, Ta = 8, K = 10",
    "modular": "estimator + frozen teacher (true `time_fraction` to the teacher)",
    "R1_lerobot_dp": "LeRobot DiffusionPolicy (external reference)",
}

_BASELINE_LABEL: Final[Mapping[str, str]] = {
    "ppo": "teacher `ppo` s0 (P8 e07)",
    "pid_feedforward": "`pid_feedforward` (P8 e01)",
    "pid_track_descend": "`pid_track_descend` (P8 e01)",
    "oracle_gated": "`oracle_gated` (P8 e01; commit-timing oracle, privileged, not a bound)",
    "gated": "`gated` (P8 e01)",
    sc.TMIX: "`T_mix` reference (assign_rows over P8 `ppo` s0 + `gated`)",
}

_HYP_TITLES: Final[Mapping[str, str]] = {
    "H1": "H1 — compounding error: `bc_dagger` − `bc_matched`",
    "H1-SS6": "H1 secondary (SS6): `bc_dagger` − `bc_matched`",
    "H1-DART-SS5": "H1 DART secondary (SS5): `bc_dart` − `bc`",
    "H1-DART-SS6": "H1 DART secondary (SS6): `bc_dart` − `bc`",
    "H2": "H2 — multimodality: W₁(`bc_mix`, ref) − W₁(`dp_mix`, ref)",
    "H3": "H3 — chunk length: `dp` Ta = 1 − Ta = 8 (same checkpoints)",
    "H4": "H4 — where the shortfall comes from: D = P − I",
    "H5(a)": "H5(a) — deployment latency (Phase 9)",
    "H5(b)": "H5(b) — latency in the loop (Phase 9)",
}

_DAGGER: Final[str] = "†"
_RELEASED: Final[str] = "‡"
_OUTSIDE: Final[str] = (
    f"{_DAGGER} outside the training distribution: no SS6 and no 90° heading in any training, "
    "tuning or selection (P3-D1 §2); `unseen_heading` is the 90° list. `unseen_vessel` and "
    "`static` are also not in the `train` pool (frigate SS3–SS5 only) and are not marked."
)
_SIM: Final[str] = (
    "Simulation only: PyBullet physics, images rendered by PyBullet's TinyRenderer; model "
    "scale, λ = 1/25."
)


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class ReportInputs:
    """The committed files the report is rendered from (parsed, string cells).

    Attributes:
        run_info: ``run_info.json``.
        score_info: ``score_info.json``.
        summary: ``summary.csv`` rows.
        baselines: ``baselines.csv`` rows.
        aggregate: ``aggregate.csv`` rows.
        contrasts: ``contrasts.csv`` rows.
        hypotheses: ``hypotheses.json``.
        above: ``above_teacher.csv`` rows.
        missing: ``missing.csv`` rows.
        released: Labels released by ``above_teacher/reviewed.json`` -> its record.
    """

    run_info: Mapping[str, Any]
    score_info: Mapping[str, Any]
    summary: Sequence[Mapping[str, str]]
    baselines: Sequence[Mapping[str, str]]
    aggregate: Sequence[Mapping[str, str]]
    contrasts: Sequence[Mapping[str, str]]
    hypotheses: Mapping[str, Any]
    above: Sequence[Mapping[str, str]]
    missing: Sequence[Mapping[str, str]]
    released: Mapping[str, Mapping[str, Any]]

    @property
    def withheld(self) -> frozenset[str]:
        """Labels of above-teacher cells not yet released by the review."""
        return frozenset(label_of(r) for r in self.above) - frozenset(self.released)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(root: Path = sc.OUT_ROOT) -> ReportInputs:
    """Read the committed inputs under ``root`` (``results/e08``)."""
    reviewed = root / "above_teacher" / "reviewed.json"
    released = {}
    if reviewed.is_file():
        for rec in _json(reviewed).get("released", []):
            released[str(rec["label"])] = rec
    return ReportInputs(
        run_info=_json(root / "run_info.json"),
        score_info=_json(root / "score_info.json"),
        summary=_csv(root / "summary.csv"),
        baselines=_csv(root / "baselines.csv"),
        aggregate=_csv(root / "aggregate.csv"),
        contrasts=_csv(root / "contrasts.csv"),
        hypotheses=_json(root / "hypotheses.json"),
        above=_csv(root / "above_teacher.csv"),
        missing=_csv(root / "missing.csv"),
        released=released,
    )


# --------------------------------------------------------------------------- formatting


def _opt_int(v: str) -> int | None:
    return None if v in ("", "None") else int(v)


def _opt_bool(v: str) -> bool | None:
    return None if v in ("", "None") else v == "True"


def _group(r: Mapping[str, str]) -> sc.GroupKey:
    return sc.GroupKey(
        r["tag"],
        r["arm"],
        r["list"],
        r["ss"],
        _opt_int(r["Ta"]),
        _opt_int(r["K"]),
        _opt_bool(r["ensemble"]),
        r["appearance_shift"] or None,
    )


def label_of(r: Mapping[str, str]) -> str:
    """A CSV row's cell label (:attr:`ila.eval.score.GroupKey.label`)."""
    return _group(r).label


def _x(v: Any) -> float:
    if v is None:
        return math.nan
    text = str(v).strip()
    return math.nan if text in ("", "None") else float(text)


def _pct(v: Any) -> str:
    x = _x(v)
    return "n/a" if math.isnan(x) else f"{100.0 * x:.1f}"


def _num(v: Any) -> str:
    """An integer-valued count as an integer, else one decimal."""
    x = _x(v)
    if math.isnan(x):
        return "n/a"
    return str(round(x)) if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def _f3(v: Any) -> str:
    x = _x(v)
    return "n/a" if math.isnan(x) else f"{x:.3f}"


def _pts(v: Any) -> str:
    x = _x(v)
    return "n/a" if math.isnan(x) else f"{100.0 * x:+.1f}"


def _losses(r: Mapping[str, str], prefix: str) -> str:
    parts = [
        f"{LOSS_ABBREV[o]} {_num(r[f'{prefix}{o}'])}"
        for o in sc.LOSS_CLASSES
        if _x(r[f"{prefix}{o}"]) != 0.0
    ]
    return " · ".join(parts) if parts else "0"


def _learned_cell(
    r: Mapping[str, str] | None, withheld: frozenset[str], *, show_n: bool = False
) -> str:
    """``IQM [CI]; seeds min–max; losses/1 000`` (P8's learned-cell format).

    With ``show_n`` the cell also prints N = seeds × episodes per seed (from the aggregate row),
    so a per-1 000 loss rate over 600 seed-episodes is not read as a count.
    """
    if r is None:
        return "—"
    if label_of(r) in withheld:
        return "withheld (above teacher; under review)"
    if r["status"] != "complete":
        return f"missing ({r['n_seeds']}/{r['n_seeds_expected']} seeds)"
    n_txt = ""
    if show_n:
        n_seeds, n_eps = int(r["n_seeds"]), int(r["n_episodes_per_seed"])
        n_txt = f"; N = {n_seeds * n_eps} ({n_seeds} seeds × {n_eps})"
    return (
        f"**{_pct(r['success_iqm'])}** [{_pct(r['success_iqm_lo'])}, "
        f"{_pct(r['success_iqm_hi'])}]; seeds {_pct(r['success_seed_min'])}–"
        f"{_pct(r['success_seed_max'])}; losses/1 000: {_losses(r, 'per_1000_')}{n_txt}"
    )


def _cell(
    r: Mapping[str, str] | None,
    inp: ReportInputs,
    notes: list[str] | None = None,
    *,
    show_n: bool = False,
) -> str:
    """:func:`_learned_cell`, marked ``‡`` when it is a released above-teacher cell.

    The marked cell's label is appended to ``notes`` (once) for :func:`_footnotes`.
    """
    text = _learned_cell(r, inp.withheld, show_n=show_n)
    if r is not None and label_of(r) in inp.released:
        text = f"{text} {_RELEASED}"
        if notes is not None and label_of(r) not in notes:
            notes.append(label_of(r))
    return text


def _footnotes(inp: ReportInputs, labels: Sequence[str]) -> list[str]:
    """The reviewed findings of the released above-teacher cells marked in a table."""
    out = []
    for lab in labels:
        rec = inp.released[lab]
        out += [
            "",
            f"{_RELEASED} `{lab}` is above its own teacher in at least one seed; released by "
            f"{rec.get('reviewed_by', '')} ({rec.get('date', '')}) after the touchdown audit "
            f"(`results/e08/above_teacher/audit.md`): {rec.get('finding', '')}",
        ]
    return out


def _baseline_rows(
    inp: ReportInputs,
    list_name: str,
    cells: Sequence[str],
    methods: Sequence[str] = sc.ALWAYS_PRINTED,
) -> list[list[str]]:
    """One row per baseline (P8's committed rows) for ``cells`` of ``list_name``."""
    base = _base_index(inp)
    return [
        [_BASELINE_LABEL[m]] + [_baseline_cell(base.get((m, list_name, ss))) for ss in cells]
        for m in methods
    ]


def _baseline_cell(r: Mapping[str, str] | None) -> str:
    """``rate [Wilson] k/N; losses per N`` (P8's baseline format)."""
    if r is None:
        return "—"
    n = r["n_episodes"]
    return (
        f"{_pct(r['success_rate'])} [{_pct(r['success_wilson_lo'])}, "
        f"{_pct(r['success_wilson_hi'])}] {r['n_success']}/{n}; losses/{n}: "
        f"{_losses(r, 'n_')}"
    )


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c.replace("|", "\\|") for c in row) + " |" for row in rows]
    return out


def _ss_head(list_name: str, ss: str) -> str:
    return f"{ss} {_DAGGER}" if sc.outside_training(list_name, ss) else ss


def _student_name(arm: str, teacher: str) -> str:
    note = ARM_NOTES.get(arm, "ablation")
    return f"`{arm}` ({note}; from {_TEACHER_NAME.get(teacher, teacher)})"


# --------------------------------------------------------------------------- sections


def _lists_and_cells(inp: ReportInputs) -> list[tuple[str, list[str]]]:
    order: dict[str, list[str]] = {}
    for r in inp.aggregate:
        if r["tag"] == "core":
            cells = order.setdefault(r["list"], [])
            if r["ss"] not in cells:
                cells.append(r["ss"])
    return list(order.items())


def _agg_index(inp: ReportInputs) -> dict[tuple[str, ...], Mapping[str, str]]:
    return {
        (
            r["tag"],
            r["arm"],
            r["list"],
            r["ss"],
            r["Ta"],
            r["K"],
            r["ensemble"],
            r["appearance_shift"],
        ): r
        for r in inp.aggregate
    }


def _base_index(inp: ReportInputs) -> dict[tuple[str, str, str], Mapping[str, str]]:
    return {(r["method"], r["list"], r["ss"]): r for r in inp.baselines}


def _arms(inp: ReportInputs, tag: str) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for r in inp.aggregate:
        if r["tag"] == tag and r["arm"] not in seen:
            seen[r["arm"]] = r["teacher"]
    return list(seen.items())


def _header(inp: ReportInputs) -> list[str]:
    si = inp.score_info
    lines = [
        "# Results — Phase 8: evaluation matrix and hypothesis scoring (simulation only)",
        "",
        "<!-- Generated by scripts/report.py from committed files under results/e08/ only; "
        "do not edit by hand. `make report` re-renders it byte-identically. -->",
        "",
        _SIM,
        "Every student here is distilled from a privileged, state-based P8 teacher and flies "
        "P8's frozen episode lists under P8's frozen success criteria (closing speed along the "
        "deck normal ≤ 0.5 m/s, lateral offset ≤ 0.10 m, relative tilt ≤ 15°, contact ≥ 0.5 s), "
        "all model scale.",
        "",
    ]
    if not si.get("complete"):
        lines += [
            f"**PARTIAL MATRIX:** {si.get('cells_present')} of {si.get('cells_expected')} "
            f"expected cells are present and {si.get('cells_missing')} are missing "
            "(`results/e08/missing.csv`). Missing cells are marked in every table; nothing is "
            "pooled over a missing seed.",
            "",
        ]
    lines += [
        "Reading rules (P3-D1 §1, §6): learned cells are `IQM [stratified-bootstrap 95 % CI]` "
        "over seeds; baselines are `rate [Wilson 95 %] k/N`; contrasts are paired bootstraps "
        "over identical episodes; a difference exists only when its 95 % CI excludes 0; "
        f"magnitudes are judged on the point estimate. {hy_no_multiplicity(inp)} Sea states "
        "are never pooled. Every SS6 contrast carries P8's one-ulp flip floor: "
        f"{p8_adapter.FLIP_FLOOR_SS6.sentence}.",
        "",
        f"Losses per 1 000 seed-episodes (learned) or per N (baselines): "
        f"{', '.join(f'{v} {k}' for k, v in LOSS_ABBREV.items())}.",
        "",
        _OUTSIDE,
        "",
    ]
    return lines


def hy_no_multiplicity(inp: ReportInputs) -> str:
    """The no-multiplicity sentence, from ``hypotheses.json``."""
    return str(inp.hypotheses.get("no_multiplicity", "No multiplicity correction is applied."))


def _provenance(inp: ReportInputs) -> list[str]:
    ri, si = inp.run_info, inp.score_info
    jobs = ri.get("jobs", {})
    wall = sum(_x(j.get("wall_clock_s")) for j in jobs.values() if j.get("wall_clock_s"))
    commits = sorted({str(i.get("ila_commit")) for i in ri.get("invocations", [])})
    dirty = any(bool(i.get("ila_dirty")) for i in ri.get("invocations", []))
    workers = sorted({str(j.get("workers")) for j in jobs.values()})
    rows = [
        ("P8 commit", f"`{ri.get('p8_commit')}`"),
        ("ila commit(s) that flew rows", ", ".join(f"`{c}`" for c in commits) + (
            " (a dirty tree was recorded)" if dirty else " (clean tree)")),
        ("P8 list MANIFEST SHA-256", f"`{ri.get('list_manifest_sha256')}`"),
        ("success.yaml SHA-256", f"`{ri.get('success_yaml_sha256')}`"),
        ("camera.yaml (render) SHA-256", f"`{ri.get('camera_yaml_sha256')}`"),
        ("matrix.yaml SHA-256", f"`{ri.get('matrix_yaml_sha256')}`"),
        ("inference path", str(ri.get("inference_path", ""))),
        ("jobs with rows", f"{ri.get('jobs_done')} of {ri.get('jobs_total')}"),
        ("summed per-job wall clock", f"{wall / 3600.0:.2f} h on this machine (workers "
         f"{', '.join(workers)})"),
        ("shared cells (not re-flown)", str(len(ri.get("shared_cells", [])))),
        ("statistics", f"bootstrap seed {si.get('statistics', {}).get('bootstrap_seed')}; "
         f"IQM CIs {si.get('statistics', {}).get('iqm_reps')} replicates; contrasts "
         f"{si.get('statistics', {}).get('paired_reps')} replicates"),
    ]  # fmt: skip
    for name, rec in sorted(ri.get("lists", {}).items()):
        rows.append((f"list `{name}`", f"`{rec.get('sha256')}`"))
    for rec in ri.get("committed_rows", []):
        rows.append(
            (
                f"P8 committed rows `{rec.get('method')}`",
                f"`{rec.get('file')}` SHA-256 `{rec.get('file_sha256')}` ({rec.get('n_rows')} "
                "rows)",
            )
        )
    for block, why in sorted(ri.get("skipped_blocks", {}).items()):
        rows.append((f"not flown: `{block}`", str(why)))
    best = si.get("best_end_to_end", {})
    rows.append(("H4 best end-to-end student", f"`{best.get('arm')}`: {best.get('reason')}"))
    lines = ["## 1. Provenance", ""]
    lines += _table(("item", "value"), rows)
    invs = ri.get("invocations", [])
    if invs:
        lines += ["", "Matrix invocations (from `run_info.json`):", ""]
        lines += _table(
            ("started (UTC)", "finished (UTC)", "ila commit", "host CPUs"),
            [
                (
                    str(i.get("started_utc", "")),
                    str(i.get("finished_utc", "")),
                    f"`{str(i.get('ila_commit', ''))[:12]}`",
                    str(i.get("host", {}).get("cpu_count", "")),
                )
                for i in invs
            ],
        )
    lines += [
        "",
        "`latency_arm` and H5 are Phase 9's: the delay d comes from Phase 9's latency study "
        "(P3-D1 §6 H5, §7, §8).",
        "",
        "Sources: `results/e08/run_info.json`, `results/e08/score_info.json`.",
        "",
    ]
    return lines


def _core(inp: ReportInputs) -> list[str]:
    agg = _agg_index(inp)
    lines = [
        "## 2. Success per list and sea state (core arms; simulated landings)",
        "",
        "Teacher and the three P8 baselines are printed beside every student, from P8's "
        "committed rows. Never pooled across sea states.",
        "",
    ]
    for list_name, cells in _lists_and_cells(inp):
        lines += [f"### `{list_name}`", ""]
        header = ["arm", *(_ss_head(list_name, ss) for ss in cells)]
        rows = []
        notes: list[str] = []
        for arm, teacher in _arms(inp, "core"):
            rows.append(
                [_student_name(arm, teacher)]
                + [
                    _cell(agg.get(("core", arm, list_name, ss, *_inf(arm), "")), inp, notes)
                    for ss in cells
                ]
            )
        rows += _baseline_rows(inp, list_name, cells)
        lines += _table(header, rows)
        lines += _footnotes(inp, notes)
        lines.append("")
    lines += [
        "Sources: `results/e08/aggregate.csv` (tag `core`), `results/e08/baselines.csv`.",
        "",
    ]
    return lines


def _inf(arm: str) -> tuple[str, str, str]:
    """The core inference columns of an arm's aggregate key (``dp``: Ta 8, K 10, off)."""
    return ("8", "10", "False") if arm == "dp" else ("", "", "")


def _breakdown(inp: ReportInputs) -> list[str]:
    agg, base, withheld = _agg_index(inp), _base_index(inp), inp.withheld
    lines = [
        "## 3. Outcome breakdown (all six classes, counts pooled over seeds within a cell; % of N)",
        "",
    ]
    header = ["arm", "list", "SS", "N", *sc.OUTCOMES]
    rows = []
    for list_name, cells in _lists_and_cells(inp):
        for ss in cells:
            for arm, _ in _arms(inp, "core"):
                r = agg.get(("core", arm, list_name, ss, *_inf(arm), ""))
                if r is None:
                    continue
                if label_of(r) in withheld:
                    rows.append([f"`{arm}`", list_name, ss, "withheld", *([""] * 6)])
                    continue
                if r["status"] != "complete":
                    rows.append([f"`{arm}`", list_name, ss, "missing", *([""] * 6)])
                    continue
                n = int(r["n_seeds"]) * int(r["n_episodes_per_seed"])
                mark = f" {_RELEASED}" if label_of(r) in inp.released else ""
                rows.append(
                    [f"`{arm}`{mark}", list_name, _ss_head(list_name, ss), str(n)]
                    + [f"{r[f'n_{o}']} ({100.0 * int(r[f'n_{o}']) / n:.1f})" for o in sc.OUTCOMES]
                )
            for m in sc.ALWAYS_PRINTED:
                b = base.get((m, list_name, ss))
                if b is None:
                    continue
                n = int(b["n_episodes"])
                rows.append(
                    [_BASELINE_LABEL[m], list_name, _ss_head(list_name, ss), str(n)]
                    + [f"{b[f'n_{o}']} ({100.0 * int(b[f'n_{o}']) / n:.1f})" for o in sc.OUTCOMES]
                )
    lines += _table(header, rows)
    lines += ["", "Sources: `results/e08/aggregate.csv`, `results/e08/baselines.csv`.", ""]
    return lines


_SECONDARY: Final[tuple[tuple[str, str, str], ...]] = (
    ("rel_vz_normal_p50_m_s", "closing p50 (normal)", "mean"),
    ("rel_vz_normal_p95_m_s", "closing p95 (normal)", "mean"),
    ("rel_vz_world_z_p95_m_s", "closing p95 (world z)", "mean"),
    ("lateral_p50_m", "lateral p50", "mean"),
    ("lateral_p95_m", "lateral p95", "mean"),
    ("rel_tilt_p95_deg", "rel. tilt p95 (deg)", "mean"),
    ("time_to_touchdown_p50_s", "touchdown p50 (s)", "mean"),
    ("effort_mean_sq", "effort", "mean"),
    ("action_jerk_mean", "jerk", "mean"),
    ("disagreement_n", "detector disagree n", "sum"),
    ("tunnelling_n", "tunnelling n", "sum"),
    ("teacher_agreement_mean", "teacher agreement", "mean"),
    ("inference_calls_mean", "inference calls / ep", "mean"),
)


def _secondary(inp: ReportInputs) -> list[str]:
    withheld = inp.withheld
    per: dict[str, list[Mapping[str, str]]] = {}
    order: list[str] = []
    for r in inp.summary:
        if r["tag"] != "core":
            continue
        lab = label_of(r)
        if lab not in per:
            order.append(lab)
        per.setdefault(lab, []).append(r)
    lines = [
        "## 4. Secondary metrics (core arms)",
        "",
        "Learned arms: the mean over seeds of each seed's cell value (sums for counts); "
        "baselines: the cell value. Touchdown metrics over contact-detected touchdowns only, "
        "never imputed. Speeds m/s, lengths m, times s, model scale; effort and jerk in P8's "
        "normalised action units; teacher agreement is the mean ‖a_student − a_teacher‖ on the "
        "student's own states.",
        "",
    ]
    header = ["arm", "list", "SS", *(h for _, h, _ in _SECONDARY)]
    rows = []
    for lab in order:
        rs = per[lab]
        r0 = rs[0]
        if lab in withheld:
            rows.append(
                [f"`{r0['arm']}`", r0["list"], r0["ss"], "withheld"] + [""] * (len(_SECONDARY) - 1)
            )
            continue
        if any(r["status"] == "missing" for r in rs):
            rows.append(
                [f"`{r0['arm']}`", r0["list"], r0["ss"], "missing"] + [""] * (len(_SECONDARY) - 1)
            )
            continue
        vals = []
        for col, _, how in _SECONDARY:
            xs = [_x(r[col]) for r in rs]
            finite = [x for x in xs if not math.isnan(x)]
            if how == "sum":
                vals.append(_num(sum(finite)) if finite else "n/a")
            else:
                vals.append(_f3(sum(finite) / len(finite)) if finite else "n/a")
        mark = f" {_RELEASED}" if lab in inp.released else ""
        rows.append([f"`{r0['arm']}`{mark}", r0["list"], _ss_head(r0["list"], r0["ss"]), *vals])
    for b in inp.baselines:
        if b["method"] not in sc.ALWAYS_PRINTED:
            continue
        vals = [
            (_num(b[col]) if how == "sum" else _f3(b[col])) if col in b else "—"
            for col, _, how in _SECONDARY
        ]
        rows.append([_BASELINE_LABEL[b["method"]], b["list"], _ss_head(b["list"], b["ss"]), *vals])
    lines += _table(header, rows)
    lines += ["", "Sources: `results/e08/summary.csv`, `results/e08/baselines.csv`.", ""]
    return lines


def withheld_inputs(inputs: str | Sequence[str], withheld: frozenset[str]) -> list[str]:
    """The entries of ``inputs`` (``; ``-joined labels, or a list) that are withheld."""
    items = inputs.split("; ") if isinstance(inputs, str) else list(inputs)
    return [i for i in items if i in withheld]


def _hyp_estimate(rec: Mapping[str, Any]) -> str:
    units = str(rec.get("units", ""))
    est, lo, hi = rec.get("estimate"), rec.get("ci_lo"), rec.get("ci_hi")
    if units.startswith("seconds"):
        return f"{_x(est):+.3f} s [{_x(lo):+.3f}, {_x(hi):+.3f}] (model scale; ×5 full scale)"
    return f"{_pts(est)} points [{_pts(lo)}, {_pts(hi)}]"


def verdict_line(verdict: str, note: str) -> str:
    """The bold verdict with its note, the note only where it adds a reason.

    ``supported`` / ``supported`` prints the verdict alone; a note that starts with the verdict
    word prints only its remainder; any other note follows a dash.
    """
    head = f"**Verdict: {verdict}**"
    if note == verdict or not note:
        return f"{head}."
    if note.startswith(verdict):
        return f"{head} {note[len(verdict) :].strip()}."
    return f"{head} — {note}."


def _hypotheses(inp: ReportInputs) -> list[str]:
    withheld = inp.withheld
    lines = [
        "## 5. Hypotheses (scored exactly as frozen in P3-D1 §6)",
        "",
        "Verdict words are P3-D1's: supported / not supported / inconclusive. Nulls are "
        f"reported here, in the body. {hy_no_multiplicity(inp)}",
        "",
    ]
    for rec in inp.hypotheses.get("hypotheses", []):
        name = str(rec["hypothesis"])
        lines += [f"### {_HYP_TITLES.get(name, name)} — `{rec.get('cell')}`", ""]
        status = str(rec.get("status", ""))
        if status != "scored":
            lines += [f"**{status}**", ""]
            if str(rec.get("cell", "")).endswith("SS6"):
                lines += [f"SS6 flip floor: {p8_adapter.FLIP_FLOOR_SS6.sentence}.", ""]
            continue
        held = withheld_inputs(rec.get("inputs", []), withheld)
        if held:
            lines += [
                "**withheld**: an input cell is above its teacher and under `results-skeptic` "
                f"review ({', '.join(f'`{h}`' for h in held)}; "
                "`results/e08/above_teacher/audit.md`).",
                "",
            ]
            if str(rec.get("cell", "")).endswith("SS6"):
                lines += [f"SS6 flip floor: {p8_adapter.FLIP_FLOOR_SS6.sentence}.", ""]
            continue
        mag = rec.get("magnitude")
        mag_txt = "D > 0" if mag is None else (
            f"{_x(mag):.3f} s" if str(rec.get("units", "")).startswith("seconds")
            else f"+{100.0 * _x(mag):.0f} points"
        )  # fmt: skip
        lines += [
            f"- {verdict_line(str(rec['verdict']), str(rec['verdict_note']))}",
            f"- Estimate: {_hyp_estimate(rec)}; magnitude {mag_txt}; in simulation.",
            f"- Contrast: {rec['contrast']}; statistic: {rec['statistic']}.",
            f"- N: {rec['n_episodes']} paired episodes; seeds "
            + ", ".join(f"`{s['arm']}` {s['seeds']}" for s in rec.get("n_seeds", []))
            + ".",
            f"- Inputs: {', '.join(f'`{i}`' for i in rec.get('inputs', []))}.",
        ]
        for b in rec.get("beside", []):
            lines.append(f"- {b['label']}: {b['text']}.")
        if str(rec.get("cell", "")).endswith("SS6") and not any(
            b["label"] == "SS6 flip floor" for b in rec.get("beside", [])
        ):
            lines.append(f"- SS6 flip floor: {p8_adapter.FLIP_FLOOR_SS6.sentence}.")
        lines += ["", "Stated with the result (verbatim from the scorer):", ""]
        lines += [f"- {c}" for c in rec.get("caveats", [])]
        lines.append("")
    lines += ["Sources: `results/e08/hypotheses.json`, `results/e08/hypotheses.csv`.", ""]
    return lines


def _contrast_value(r: Mapping[str, str]) -> str:
    if r["status"].startswith("missing"):
        return r["status"]
    if r["units"].startswith("success"):
        lo, hi = r["ci_lo"], r["ci_hi"]
        ci = f" [{_pts(lo)}, {_pts(hi)}]" if lo not in ("", "nan") else " (no CI)"
        return f"{_pts(r['estimate'])} points{ci}"
    if r["units"].startswith("seconds"):
        return f"{_x(r['estimate']):+.3f} s [{_x(r['ci_lo']):+.3f}, {_x(r['ci_hi']):+.3f}]"
    return f"{_f3(r['estimate'])} [{_f3(r['ci_lo'])}, {_f3(r['ci_hi'])}]"


#: Input tags that never enter a core section (P3-D1 §3 reporting rules; skill
#: landing-protocol-p8): their contrasts go to §6b.
TAGGED_INPUTS: Final[frozenset[str]] = frozenset(
    ("ablation", "appearance_shift", "reference_external", "latency_arm")
)


def contrast_tags(inputs: str) -> list[str]:
    """The tags of a contrast's input cells, in input order (``P8 rows`` for P8's rows).

    ``+clock`` is added when ``bc_plus_clock`` is an input (the labelled ``+clock`` ablation).
    """
    tags: list[str] = []
    for lab in (i for i in inputs.split("; ") if i):
        first = lab.split(" ")[0]
        tag = "P8 rows" if first in ("P8", "T_mix") else first
        if tag not in tags:
            tags.append(tag)
    if " bc_plus_clock " in f" {inputs} ":
        tags.append("+clock")
    return tags


def is_tagged(inputs: str) -> bool:
    """Whether any input cell carries a tag that is never mixed into a core section."""
    return any(t in TAGGED_INPUTS or t == "+clock" for t in contrast_tags(inputs))


def _contrast_n(r: Mapping[str, str]) -> str:
    """``seeds a / seeds b × episodes`` of a contrast row (blank when missing)."""
    a, b, n = r.get("n_seeds_a", ""), r.get("n_seeds_b", ""), r.get("n_episodes", "")
    if "" in (a, b, n):
        return ""
    return f"{a} / {b} × {n}"


def _contrasts(
    inp: ReportInputs, keep: Callable[[Mapping[str, str]], bool], title: str
) -> list[str]:
    withheld = inp.withheld
    rows = []
    for r in inp.contrasts:
        if not keep(r):
            continue
        held = withheld_inputs(r.get("inputs", ""), withheld)
        value = "withheld (input under above-teacher review)" if held else _contrast_value(r)
        note = r["note"]
        if r["flip_floor"]:
            note = f"{note} SS6 flip floor: {r['flip_floor']}.".strip()
        rows.append(
            [
                r["name"],
                ", ".join(f"`{t}`" for t in contrast_tags(r.get("inputs", ""))),
                f"{r['list']} {r['ss']}",
                _contrast_n(r),
                value,
                "" if held or r["status"].startswith("missing") else r["reading"],
                note,
            ]
        )
    lines = [title, ""]
    if not rows:
        return [*lines, "None.", ""]
    header = (
        "contrast",
        "input tags",
        "cell",
        "N: seeds a / b × episodes",
        "estimate [95 % CI]",
        "reading",
        "notes",
    )
    lines += _table(header, rows)
    lines += ["", "Sources: `results/e08/contrasts.csv` (`inputs` column for the tags).", ""]
    return lines


def _unscored(inp: ReportInputs) -> list[str]:
    """§6: unscored contrasts, core inputs (6a) apart from tagged inputs (6b)."""

    def unscored(r: Mapping[str, str]) -> bool:
        return r["kind"] not in ("scored", "component of the scored H4")

    lines = [
        "## 6. Unscored contrasts: post hoc, descriptive and teacher checks (no verdicts)",
        "",
    ]
    lines += _contrasts(
        inp,
        lambda r: unscored(r) and not is_tagged(r.get("inputs", "")),
        "### 6a. Core inputs (tags `core`, `dp_Ta`; P8's committed rows)",
    )
    lines += _contrasts(
        inp,
        lambda r: unscored(r) and is_tagged(r.get("inputs", "")),
        "### 6b. Contrasts with a tagged input (ablation, `+clock`; never mixed into 6a)",
    )
    return lines


def _tcell(r: Mapping[str, str] | None, inp: ReportInputs, notes: list[str] | None = None) -> str:
    """A §7 learned cell: :func:`_cell` with N printed (ablations and R1 have 3 seeds)."""
    return _cell(r, inp, notes, show_n=True)


def _tagged(inp: ReportInputs) -> list[str]:
    agg = _agg_index(inp)
    notes: list[str] = []
    lines = [
        "## 7. Tagged sections (never mixed into the core tables)",
        "",
        "### 7a. `dp` chunk length Ta, with and without temporal ensembling (tag `dp_Ta`; "
        "`dp` distilled from P8 `ppo` seed 0; same checkpoints, K = 10; unscored curve)",
        "",
        "Ta = 8 without ensembling is the core `dp` row (shared, not re-flown); Ta = 1 with "
        "ensembling is recorded as an alias of Ta = 1 without (one source of weight 1.0, same "
        "noise keys; P6-D1 item 3).",
        "",
        "`dp` is reported at its frozen Ta = 8, K = 10 (P3-D1 §3); nothing is re-selected from "
        "this curve. Stated with it, from Phase 6 (`docs/protocol.md` P6-D5, items 3–5 and its "
        "README requirements; sources under `results/phase6/` and `results/dp_sweeps.json`): on "
        "`select`, `dp` at Ta = 8 already landed far below the BC family, mostly a chunking "
        "effect that did not depend on sea state; its checkpoint selection is noise-dominated "
        "and the selected `select` scores are optimistic; and held-out denoising loss is not a "
        "selection proxy (success does not follow it). H3 (§5) scores Ta = 1 − Ta = 8 at `id` "
        "SS6 with its caveats: checkpoints selected at Ta = 8, mode switching between replans, "
        "the `dp_minus_last_action` input-freshness confound, and P6-D5's: a Ta contrast cannot "
        "attribute the loss to deck motion.",
        "",
    ]
    tas = sorted({int(r["Ta"]) for r in inp.aggregate if r["tag"] == "dp_Ta"})
    sss = [
        ss
        for ss in ("SS4", "SS5", "SS6")
        if any(r["tag"] == "dp_Ta" and r["ss"] == ss for r in inp.aggregate)
    ]
    for ss in sss:
        lines += [f"`id` {_ss_head('id', ss)}:", ""]
        rows = [
            [
                f"Ta = {ta}",
                *(
                    _tcell(agg.get(("dp_Ta", "dp", "id", ss, str(ta), "10", e, "")), inp, notes)
                    for e in ("False", "True")
                ),
            ]
            for ta in tas
        ]
        lines += _table(("Ta", "no ensembling", "ACT ensembling (m = 0.01)"), rows)
        lines.append("")
    if sss:
        lines += ["Beside it, the teacher and the three P8 baselines (P8's committed rows):", ""]
        lines += _table(
            ("baseline", *(f"id {_ss_head('id', ss)}" for ss in sss)),
            _baseline_rows(inp, "id", sss),
        )
        lines += _footnotes(inp, notes)
        lines.append("")
        notes = []
    lines += [
        "### 7b. `dp` DDIM steps K (tag `dp_K`; `id` SS5; Ta = 8, no ensembling)",
        "",
        "K = 1 and K = 2 are schedule-limited: under ε-prediction at ᾱ₉₉ = 2.43e-7 they are "
        "ill-conditioned for any network (P3-D1 §3, §8). K = 10 is the core `dp` row.",
        "",
    ]
    ks = sorted({int(r["K"]) for r in inp.aggregate if r["tag"] == "dp_K"})
    k_rows = []
    for k in ks:
        r = agg.get(("dp_K", "dp", "id", "SS5", "8", str(k), "False", ""))
        name = f"K = {k}" + (" (schedule-limited)" if k in (1, 2) else "")
        k_rows.append([name, _tcell(r, inp, notes)])
    k_rows += _baseline_rows(inp, "id", ("SS5",))
    lines += _table(("K", "id SS5"), k_rows)
    lines += _footnotes(inp, notes)
    notes = []
    lines += [
        "",
        "### 7c. Appearance shift (tag `appearance_shift`; evaluation-only render variants of "
        "the rendered (PyBullet TinyRenderer) frames; `id` SS5–SS6)",
        "",
    ]
    shift_rows = []
    for arm, teacher in _arms(inp, "appearance_shift"):
        inf = _inf(arm)
        shift_rows.append(
            [f"{_student_name(arm, teacher)}, committed look"]
            + [
                _tcell(agg.get(("core", arm, "id", ss, *inf, "")), inp, notes)
                for ss in ("SS5", "SS6")
            ]
        )
        variants = sorted(
            {
                r["appearance_shift"]
                for r in inp.aggregate
                if r["tag"] == "appearance_shift" and r["arm"] == arm
            }
        )
        for v in variants:
            shift_rows.append(
                [f"`{arm}`, `{v}`"]
                + [
                    _tcell(agg.get(("appearance_shift", arm, "id", ss, *inf, v)), inp, notes)
                    for ss in ("SS5", "SS6")
                ]
            )
    shift_rows += _baseline_rows(inp, "id", ("SS5", "SS6"))
    lines += _table(("arm, look", "id SS5", f"id SS6 {_DAGGER}"), shift_rows)
    lines += _footnotes(inp, notes)
    notes = []
    lines += [
        "",
        "### 7d. Ablations (tag `ablation`; 3 seeds; `id` SS4–SS6; each with its own teacher's "
        "committed rows)",
        "",
        "`bc_plus_clock` is the labelled `+clock` ablation (`time_fraction` in the input). "
        "Beside the students: the teacher and the three P8 baselines, plus `gated` and the "
        "`T_mix` reference, the two references of the `*_mix` arms (`T_mix` assigns each "
        "episode to `ppo` s0 or `gated`; the student cannot observe the assignment). "
        "`pid_feedforward` is the `*_pid` arms' teacher.",
        "",
    ]
    abl_rows = []
    for arm, teacher in _arms(inp, "ablation"):
        abl_rows.append(
            [_student_name(arm, teacher)]
            + [
                _tcell(agg.get(("ablation", arm, "id", ss, *_inf_abl(arm), "")), inp, notes)
                for ss in ("SS4", "SS5", "SS6")
            ]
        )
    abl_rows += _baseline_rows(
        inp, "id", ("SS4", "SS5", "SS6"), (*sc.ALWAYS_PRINTED, "gated", sc.TMIX)
    )
    lines += _table(("arm", "id SS4", "id SS5", f"id SS6 {_DAGGER}"), abl_rows)
    lines += _footnotes(inp, notes)
    notes = []
    lines += [
        "",
        "### 7e. External reference R1 (tag `reference_external`; not gating; 3 seeds; "
        "`id` SS3–SS6)",
        "",
    ]
    ref_rows = []
    for arm, teacher in _arms(inp, "reference_external"):
        ref_rows.append(
            [_student_name(arm, teacher)]
            + [
                _tcell(
                    agg.get(("reference_external", arm, "id", ss, "8", "10", "False", "")),
                    inp,
                    notes,
                )
                for ss in ("SS3", "SS4", "SS5", "SS6")
            ]
        )
    ref_rows.append(
        [_student_name("dp", "T_ppo") + " (core, beside it)"]
        + [
            _tcell(agg.get(("core", "dp", "id", ss, "8", "10", "False", "")), inp, notes)
            for ss in ("SS3", "SS4", "SS5", "SS6")
        ]
    )
    ref_rows += _baseline_rows(inp, "id", ("SS3", "SS4", "SS5", "SS6"))
    lines += _table(("arm", "id SS3", "id SS4", "id SS5", f"id SS6 {_DAGGER}"), ref_rows)
    lines += _footnotes(inp, notes)
    lines += ["", "Sources: `results/e08/aggregate.csv`, `results/e08/baselines.csv`.", ""]
    return lines


def _inf_abl(arm: str) -> tuple[str, str, str]:
    return ("8", "10", "False") if arm.startswith("dp_") else ("", "", "")


def _above(inp: ReportInputs) -> list[str]:
    lines = [
        "## 8. Students above their own teacher (investigated before reporting)",
        "",
        "A cell is listed when its IQM, or any single seed's success rate, exceeds its own "
        "teacher's committed rate in that cell. Its numbers are withheld from every table above "
        "until `results-skeptic` has reviewed the touchdown audit (CLAUDE.md non-negotiable 8). "
        f"A released cell is marked {_RELEASED} wherever it is printed; its finding follows each "
        "success table that prints it and is listed below.",
        "",
    ]
    if not inp.above:
        return [*lines, "No cell is above its teacher in the scored rows.", ""]
    rows = []
    for r in inp.above:
        lab = label_of(r)
        rel = inp.released.get(lab)
        status = (
            f"released by {rel.get('reviewed_by', '')} ({rel.get('date', '')}): "
            f"{rel.get('finding', '')}"
            if rel
            else "under review (withheld)"
        )
        n_seeds = len([s for s in r["seeds_above"].split("|") if s])
        rows.append(
            [
                f"`{lab}`",
                _TEACHER_NAME.get(r["teacher"], r["teacher"]),
                "yes" if r["iqm_above"] == "True" else "no",
                str(n_seeds),
                f"`results/e08/{r['audit_file']}`",
                status,
            ]
        )
    lines += _table(("cell", "teacher", "IQM above", "seeds above", "audit", "status"), rows)
    lines += [
        "",
        "Combined audit: `results/e08/above_teacher/audit.md`. Sources: "
        "`results/e08/above_teacher.csv`, `results/e08/above_teacher/reviewed.json` (if present).",
        "",
    ]
    return lines


def _missing(inp: ReportInputs) -> list[str]:
    lines = ["## 9. Missing cells", ""]
    if not inp.missing:
        return [*lines, "None: every expected cell has rows.", ""]
    counts: dict[tuple[str, str], int] = {}
    for r in inp.missing:
        counts[(r["tag"], r["arm"])] = counts.get((r["tag"], r["arm"]), 0) + 1
    lines += [
        f"{len(inp.missing)} expected (cell, seed) entries have no rows yet; each is listed "
        "with its reason in `results/e08/missing.csv`.",
        "",
    ]
    lines += _table(
        ("tag", "arm", "missing (cell, seed) entries"),
        [[t, f"`{a}`", str(n)] for (t, a), n in counts.items()],
    )
    return [*lines, ""]


def render(inp: ReportInputs) -> str:
    """The full ``results.md`` text (deterministic in its inputs)."""
    lines: list[str] = []
    lines += _header(inp)
    lines += _provenance(inp)
    lines += _core(inp)
    lines += _breakdown(inp)
    lines += _secondary(inp)
    lines += _hypotheses(inp)
    lines += _unscored(inp)
    lines += _tagged(inp)
    lines += _above(inp)
    lines += _missing(inp)
    lines += [
        "## 10. Scope",
        "",
        f"{_SIM} A rendered frame is not a camera image, and no result here is a statement "
        "about hardware. Teacher-relative statements are conditional on P8 teacher seed 0. "
        f"{hy_no_multiplicity(inp)}",
        "",
    ]
    return "\n".join(lines)


def write_report(root: Path = sc.OUT_ROOT, out: Path = RESULTS_MD) -> str:
    """Render ``results.md`` from the committed files under ``root``; return the text."""
    text = render(load_inputs(root))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
