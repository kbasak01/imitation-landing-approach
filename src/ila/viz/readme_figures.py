"""README figures: seven uncluttered static figures from committed CSV/JSON only.

Simulation only (PyBullet physics; images rendered by PyBullet's TinyRenderer). Success in
percent; times in seconds at model scale (lambda = 1/25; x5 full scale); latency in
milliseconds of wall clock on the benchmark machine (RTX A4000, WSL2), not scaled.

Every figure ``<name>`` is written as ``<name>.png`` (light), ``<name>_dark.png`` (dark, for a
GitHub ``<picture>`` with ``prefers-color-scheme``) and ``<name>.csv``, the exact values plotted.
Each CSV row names the committed file it came from (``source_file``), the row inside it
(``source_row``: ``key=value`` pairs separated by ``;`` that select exactly one row, or a JSON
key path) and the columns (``value_col``, ``lo_col``, ``hi_col``, ``n_col``; ``n_col`` may be a
``*``-product of columns). ``value = scale * source[value_col]`` (likewise ``lo`` and ``hi``);
``text`` is a verbatim source cell (``text_col``). Nothing is retyped or recomputed except the
scale factor (x100 for percent and points).

Inputs (never a row file, never P8): ``results/e08/{aggregate,baselines,hypotheses}.csv``,
``results/e08/hypotheses.json`` (H3's selection caveat), ``results/e08/figures/{h3_ta_curve,
h4_gaps}.csv``, ``results/latency/{latency,parity,closed_loop,h5a}.csv`` and
``results/latency/delay.json``. Cells withheld by the above-teacher rule
(:attr:`ila.eval.report.ReportInputs.withheld`) or not complete are not plotted; their CSV row
carries an empty value and the reason.

Determinism: Agg backend, fixed DPI, DejaVu Sans (bundled with matplotlib), no metadata stamps
(``metadata={"Software": None}``), fixed ``svg.hashsalt``; identical inputs give identical bytes.

Design: 10 in wide at 160 dpi (1600 px), so text set at 12 pt is about 11 pt when GitHub shows
the image in its ~880 px column. Arm colours are fixed (:data:`ARM_SLOT`) across every figure;
the teacher is solid ink, P8 baselines are grey dashed/dotted lines; text is in ink or the
secondary text colour, never in a series colour.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.eval import report as rp
from ila.eval import score as sc
from ila.viz.figures import _plt, _write_table, plain

__all__ = [
    "ARM_SLOT",
    "COLUMNS",
    "DARK",
    "FIGURES",
    "LIGHT",
    "OUT_DIR",
    "Theme",
    "make_readme_figures",
    "tables",
]

#: Output directory, repo-relative.
OUT_DIR: Final[str] = "docs/figures"

#: Figure names, in README order.
FIGURES: Final[tuple[str, ...]] = (
    "success_id",
    "conditions",
    "outcomes",
    "hypotheses",
    "chunking",
    "gaps",
    "latency",
)

#: Plot-table columns (module docstring).
COLUMNS: Final[tuple[str, ...]] = (
    "panel",
    "series",
    "x",
    "value",
    "lo",
    "hi",
    "n",
    "unit",
    "scale",
    "text",
    "note",
    "source_file",
    "source_row",
    "value_col",
    "lo_col",
    "hi_col",
    "n_col",
    "text_col",
)

_DPI: Final[int] = 160
_W: Final[float] = 10.0  # inches; 1600 px at _DPI
_FS: Final[float] = 12.0  # pt; ~11 pt on GitHub's ~880 px column
_LW: Final[float] = 1.7  # pt; ~2 px on GitHub
_MS: Final[float] = 7.5  # pt; ~9 px on GitHub
_SIM_NOTE: Final[str] = "Simulation (PyBullet; TinyRenderer images) · model scale λ = 1/25"
_TEACHER_NOTE: Final[str] = "Every student is distilled from P8 ppo seed 0."

_E08: Final[str] = "results/e08"
_LAT: Final[str] = "results/latency"
_AGG: Final[str] = f"{_E08}/aggregate.csv"
_BASE: Final[str] = f"{_E08}/baselines.csv"
_HYP: Final[str] = f"{_E08}/hypotheses.csv"
_HYP_JSON: Final[str] = f"{_E08}/hypotheses.json"
_H3: Final[str] = f"{_E08}/figures/h3_ta_curve.csv"
_H4: Final[str] = f"{_E08}/figures/h4_gaps.csv"
_LATENCY: Final[str] = f"{_LAT}/latency.csv"
_PARITY: Final[str] = f"{_LAT}/parity.csv"
_CLOSED: Final[str] = f"{_LAT}/closed_loop.csv"
_DELAY: Final[str] = f"{_LAT}/delay.json"
_H5A: Final[str] = f"{_LAT}/h5a.csv"
#: Froude time scale for lambda = 1/25: full-scale time = model time x sqrt(25) (derived).
_TIME_SCALE: Final[float] = math.sqrt(25.0)

#: Core students in canonical order; slot i of the categorical palette is arm i's colour.
STUDENTS: Final[tuple[str, ...]] = (
    "bc_state",
    "bc",
    "bc_matched",
    "bc_dart",
    "bc_dagger",
    "dp",
    "modular",
)
#: Arm -> categorical palette slot, identical in every figure.
ARM_SLOT: Final[Mapping[str, int]] = {a: i for i, a in enumerate(STUDENTS)}
_MARKER: Final[Mapping[str, str]] = {
    "bc_state": "D",
    "bc": "o",
    "bc_matched": "s",
    "bc_dart": "^",
    "bc_dagger": "v",
    "dp": "o",
    "modular": "s",
}
_ARM_LABEL: Final[Mapping[str, str]] = {"bc_state": "bc_state (privileged state)"}
_BASELINES: Final[tuple[str, ...]] = sc.ALWAYS_PRINTED  # ppo, pid_ff, pid_td, oracle_gated
_BASE_LABEL: Final[Mapping[str, str]] = {
    "ppo": "teacher (P8 ppo s0)",
    "pid_feedforward": "pid_feedforward",
    "pid_track_descend": "pid_track_descend",
    "oracle_gated": "oracle_gated",
}
_BASE_LS: Final[Mapping[str, Any]] = {
    "ppo": "-",
    "pid_feedforward": (0, (5, 2.5)),
    "pid_track_descend": (0, (1, 2)),
    "oracle_gated": (0, (6, 2, 1, 2)),
}
#: P8's six outcome classes in the order stacked (success first, then losses).
_OUTCOME_ORDER: Final[tuple[str, ...]] = (
    "success",
    "off_pad",
    "hard_landing",
    "bounce",
    "crash",
    "timeout",
)
#: Outcome class -> palette slot (green success; distinct slots for the losses).
_OUTCOME_SLOT: Final[Mapping[str, int]] = {
    "success": 5,
    "off_pad": 1,
    "hard_landing": 3,
    "bounce": 6,
    "crash": 7,
    "timeout": 0,
}


@dataclass(frozen=True)
class Theme:
    """A colour theme (light or dark); all colours are hex sRGB strings.

    Attributes:
        name: ``"light"`` or ``"dark"``.
        suffix: File-name suffix (``""`` or ``"_dark"``).
        surface: Background.
        ink: Primary text and the teacher's line.
        secondary: Secondary text (notes, tick marks).
        baseline: P8 baselines' grey line colour.
        grid: Recessive grid colour.
        shade: Outside-training shading.
        palette: The eight categorical slots (blue, orange, aqua, yellow, magenta, green,
            violet, red).
    """

    name: str
    suffix: str
    surface: str
    ink: str
    secondary: str
    baseline: str
    grid: str
    shade: str
    palette: tuple[str, ...]

    def arm(self, arm: str) -> str:
        """The arm's fixed colour."""
        return self.palette[ARM_SLOT[arm]]


LIGHT: Final[Theme] = Theme(
    name="light",
    suffix="",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    baseline="#8b8a85",
    grid="#e6e5e1",
    shade="#efeeea",
    palette=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
)
DARK: Final[Theme] = Theme(
    name="dark",
    suffix="_dark",
    surface="#1a1a19",
    ink="#f0efec",
    secondary="#c3c2b7",
    baseline="#8e8d86",
    grid="#2f2f2d",
    shade="#2a2a28",
    palette=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
)
THEMES: Final[tuple[Theme, ...]] = (LIGHT, DARK)

Row = Mapping[str, str]
Rec = dict[str, Any]


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class _Inputs:
    report: rp.ReportInputs
    h3: Sequence[Row]
    h4: Sequence[Row]
    hyp_csv: Sequence[Row]
    latency: Sequence[Row]
    parity: Sequence[Row]
    closed: Sequence[Row]
    delay: Mapping[str, Any]
    h5a: Sequence[Row]


def _load(repo: Path) -> _Inputs:
    return _Inputs(
        report=rp.load_inputs(repo / _E08),
        h3=sc.read_csv(repo / _H3),
        h4=sc.read_csv(repo / _H4),
        hyp_csv=sc.read_csv(repo / _HYP),
        latency=sc.read_csv(repo / _LATENCY),
        parity=sc.read_csv(repo / _PARITY),
        closed=sc.read_csv(repo / _CLOSED),
        delay=json.loads((repo / _DELAY).read_text(encoding="utf-8")),
        h5a=sc.read_csv(repo / _H5A),
    )


def _f(v: Any) -> float:
    text = "" if v is None else str(v).strip()
    return math.nan if text in ("", "None", "nan") else float(text)


def _one(rows: Sequence[Row], **key: str) -> Row | None:
    found = [r for r in rows if all(r.get(k) == v for k, v in key.items())]
    if len(found) > 1:
        raise ValueError(f"key {key} matches {len(found)} rows")
    return found[0] if found else None


def _key(**key: str) -> str:
    return ";".join(f"{k}={v}" for k, v in key.items())


def _rec(
    panel: str,
    series: str,
    x: str,
    source_file: str,
    row: Row | None,
    key: Mapping[str, str],
    *,
    value_col: str = "",
    lo_col: str = "",
    hi_col: str = "",
    n_col: str = "",
    text_col: str = "",
    scale: float = 1.0,
    unit: str = "",
    note: str = "",
    plot: bool = True,
) -> Rec:
    """One plot-table row traced to ``source_file``/``key``; value = scale * cell."""
    rec: Rec = {
        "panel": panel,
        "series": series,
        "x": x,
        "unit": unit,
        "scale": scale,
        "note": note,
        "source_file": source_file,
        "source_row": _key(**key),
        "value_col": value_col,
        "lo_col": lo_col,
        "hi_col": hi_col,
        "n_col": n_col,
        "text_col": text_col,
    }
    if row is None:
        rec["note"] = note or "missing (no source row)"
        return rec
    if text_col:
        rec["text"] = row[text_col]
    if not plot:
        return rec
    for out, col in (("value", value_col), ("lo", lo_col), ("hi", hi_col)):
        if col and str(row[col]).strip() not in ("", "nan", "None"):
            rec[out] = scale * _f(row[col])
    if n_col:
        rec["n"] = math.prod(int(row[c]) for c in n_col.split("*"))
    return rec


def _learned(
    inp: rp.ReportInputs, panel: str, arm: str, list_name: str, ss: str, *, col: str = "iqm"
) -> Rec:
    """A core student's success at (list, ss) in percent: IQM [stratified-bootstrap CI]."""
    key = {"tag": "core", "arm": arm, "list": list_name, "ss": ss}
    row = _one(inp.aggregate, **key)
    x = ss if list_name in ("id", "static") else f"{list_name} {ss}"
    if row is not None and rp.label_of(row) in inp.withheld:
        return _rec(panel, arm, x, _AGG, row, key, plot=False,
                    note="withheld (above-teacher review pending)")  # fmt: skip
    if row is not None and row["status"] != "complete":
        return _rec(panel, arm, x, _AGG, row, key, plot=False, note=f"missing ({row['status']})")
    if col == "mean":
        return _rec(panel, arm, x, _AGG, row, key, value_col="success_mean",
                    n_col="n_seeds*n_episodes_per_seed", scale=100.0, unit="%",
                    note="mean success over seeds x episodes (H4's S)")  # fmt: skip
    return _rec(
        panel, arm, x, _AGG, row, key,
        value_col="success_iqm", lo_col="success_iqm_lo", hi_col="success_iqm_hi",
        n_col="n_seeds*n_episodes_per_seed", scale=100.0, unit="%",
        note="IQM over seeds, stratified-bootstrap 95 % CI",
    )  # fmt: skip


def _baseline(inp: rp.ReportInputs, panel: str, method: str, list_name: str, ss: str) -> Rec:
    """A P8 committed baseline's success at (list, ss) in percent: rate [Wilson 95 %]."""
    key = {"method": method, "list": list_name, "ss": ss}
    x = ss if list_name in ("id", "static") else f"{list_name} {ss}"
    return _rec(
        panel, method, x, _BASE, _one(inp.baselines, **key), key,
        value_col="success_rate", lo_col="success_wilson_lo", hi_col="success_wilson_hi",
        n_col="n_episodes", scale=100.0, unit="%", note="P8 committed rows, Wilson 95 % CI",
    )  # fmt: skip


def _meta(
    series: str,
    source_file: str,
    rows: Sequence[Row],
    keys: Sequence[Mapping[str, str]],
    col: str,
) -> Rec:
    """A count quoted in a figure's text (seeds, episodes), uniform over the plotted rows.

    Raises:
        ValueError: If ``col`` differs between the plotted rows (the text would be wrong).
    """
    found = [(k, r) for k in keys if (r := _one(rows, **k)) is not None]
    vals = sorted({r[col] for _, r in found})
    if len(vals) != 1:
        raise ValueError(f"{series}: {col} is not uniform over the plotted rows: {vals}")
    key, row = found[0]
    return _rec("meta", series, "", source_file, row, key, value_col=col,
                note=f"uniform over the {len(found)} plotted rows")  # fmt: skip


def _design_meta(
    inp: rp.ReportInputs,
    arms: Sequence[str],
    cells: Sequence[tuple[str, str]],
    *,
    tag: str = "core",
    baselines: bool = True,
) -> list[Rec]:
    """Seeds and episodes per seed (students) and episodes (P8 rows) for the figure text."""
    keys = [{"tag": tag, "arm": a, "list": lst, "ss": ss} for a in arms for lst, ss in cells]
    out = [
        _meta("n_seeds", _AGG, inp.aggregate, keys, "n_seeds"),
        _meta("n_episodes_per_seed", _AGG, inp.aggregate, keys, "n_episodes_per_seed"),
    ]
    if baselines:
        bkeys = [{"method": m, "list": lst, "ss": ss} for m in _BASELINES for lst, ss in cells]
        out.append(_meta("n_episodes_p8", _BASE, inp.baselines, bkeys, "n_episodes"))
    return out


def _period_meta(inp: _Inputs) -> Rec:
    return {
        "panel": "meta",
        "series": "control period",
        "x": "",
        "value": float(inp.delay["control_period_ms"]),
        "unit": "ms (model scale)",
        "scale": 1.0,
        "source_file": _DELAY,
        "source_row": "",
        "value_col": "control_period_ms",
    }


# --------------------------------------------------------------------------- tables

_F1_PANELS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("(a) behaviour-cloning family", ("bc", "bc_matched", "bc_dart", "bc_dagger")),
    ("(b) other routes", ("bc_state", "modular", "dp")),
)
_ID_SS: Final[tuple[str, ...]] = ("SS3", "SS4", "SS5", "SS6")


def success_id_table(inp: _Inputs) -> list[Rec]:
    """F1: ``id`` success per sea state; students IQM [CI], baselines rate [Wilson], %."""
    r = inp.report
    out: list[Rec] = []
    for panel, arms in _F1_PANELS:
        for m in _BASELINES:
            out += [_baseline(r, panel, m, "id", ss) for ss in _ID_SS]
        for arm in arms:
            out += [_learned(r, panel, arm, "id", ss) for ss in _ID_SS]
    return out + _design_meta(r, STUDENTS, [("id", ss) for ss in _ID_SS])


#: F2 columns (list, ss); all exist in the frozen lists.
_F2_COLS: Final[tuple[tuple[str, str], ...]] = (
    ("static", "static"),
    ("id", "SS5"),
    ("id", "SS6"),
    ("unseen_heading", "SS5"),
    ("unseen_heading", "SS6"),
    ("unseen_vessel", "SS5"),
    ("unseen_vessel", "SS6"),
    ("unseen_seastate", "SS6"),
)


def conditions_table(inp: _Inputs) -> list[Rec]:
    """F2: arms x conditions, success %, with CI and N in the table."""
    r = inp.report
    out: list[Rec] = []
    for m in _BASELINES:
        out += [_baseline(r, "conditions", m, lst, ss) for lst, ss in _F2_COLS]
    for arm in STUDENTS:
        out += [_learned(r, "conditions", arm, lst, ss) for lst, ss in _F2_COLS]
    for rec, (lst, ss) in zip(out, _F2_COLS * (len(out) // len(_F2_COLS)), strict=True):
        rec["x"] = f"{lst} {ss}" if lst != "static" else "static"
    return out + _design_meta(r, STUDENTS, _F2_COLS)


def outcomes_table(inp: _Inputs) -> list[Rec]:
    """F3: outcome counts per arm at ``id`` SS5 / SS6 (students pooled over seeds x episodes)."""
    r = inp.report
    out: list[Rec] = []
    for ss in ("SS5", "SS6"):
        panel = f"id {ss}"
        for m in _BASELINES:
            key = {"method": m, "list": "id", "ss": ss}
            row = _one(r.baselines, **key)
            out += [
                _rec(panel, m, o, _BASE, row, key, value_col=f"n_{o}", n_col="n_episodes",
                     unit="episodes", note="plotted as 100 * value / n %; P8 committed rows")
                for o in _OUTCOME_ORDER
            ]  # fmt: skip
        for arm in STUDENTS:
            key = {"tag": "core", "arm": arm, "list": "id", "ss": ss}
            row = _one(r.aggregate, **key)
            held = row is not None and (
                rp.label_of(row) in r.withheld or row["status"] != "complete"
            )
            out += [
                _rec(panel, arm, o, _AGG, row, key, value_col=f"n_{o}",
                     n_col="n_seeds*n_episodes_per_seed", unit="episodes", plot=not held,
                     note="withheld or incomplete" if held else
                     "plotted as 100 * value / n %; pooled over seeds x episodes")
                for o in _OUTCOME_ORDER
            ]  # fmt: skip
    return out + _design_meta(r, STUDENTS, [("id", "SS5"), ("id", "SS6")])


#: F4 rows: hypothesis -> short contrast label (each arm named appears in ``contrast``).
_H_LABEL: Final[Mapping[str, str]] = {
    "H1": "bc_dagger − bc_matched",
    "H1-SS6": "bc_dagger − bc_matched",
    "H1-DART-SS5": "bc_dart − bc",
    "H1-DART-SS6": "bc_dart − bc",
    "H3": "dp: Ta = 1 − Ta = 8",
    "H4": "D = P − I (perception − imitation)",
    "H5(b)": "dp: latency arm − synchronous",
    "H2": "W₁(bc_mix) − W₁(dp_mix) to T_mix",
    "H5(a)": "dp batch-1 p99 on a parity-passing provider",
}
_H_POINTS: Final[tuple[str, ...]] = (
    "H1",
    "H1-SS6",
    "H1-DART-SS5",
    "H1-DART-SS6",
    "H3",
    "H4",
    "H5(b)",
)


def hypotheses_table(inp: _Inputs) -> list[Rec]:
    """F4: each scored contrast (points, or seconds for H2) with its 95 % CI, margin, verdict."""
    out: list[Rec] = []
    for h in (*_H_POINTS, "H2", "H5(a)"):
        key = {"hypothesis": h}
        row = _one(inp.hyp_csv, **key)
        points = h in _H_POINTS
        panel = "points" if points else ("seconds" if h == "H2" else "text")
        scale = 100.0 if points else 1.0
        unit = "points" if points else ("s (model scale)" if h == "H2" else "ms")
        cell = row["cell"] if row is not None else ""
        if h == "H5(a)":
            out.append(_rec(panel, h, cell, _HYP, row, key, text_col="verdict", plot=False,
                            note="no CI; verdict only"))  # fmt: skip
            out.append(_rec(panel, f"{h} margin", cell, _HYP, row, key, value_col="magnitude",
                            unit=unit, note="threshold, ms"))  # fmt: skip
            for r5 in inp.h5a:
                if r5["scored"] != "True":
                    continue
                k5 = {"hypothesis": r5["hypothesis"], "row": r5["row"],
                      "provider": r5["provider"], "batch_size": r5["batch_size"]}  # fmt: skip
                timed = r5["latency_status"] == "timed"
                out.append(_rec(panel, "H5(a) scored p99", r5["provider"], _H5A, r5, k5,
                                value_col="p99_ms", text_col="parity_pass", unit="ms",
                                plot=timed, note="scored; text = parity_pass"))  # fmt: skip
            continue
        out.append(_rec(panel, h, cell, _HYP, row, key, value_col="estimate", lo_col="ci_lo",
                        hi_col="ci_hi", text_col="verdict", scale=scale, unit=unit,
                        note="95 % CI"))  # fmt: skip
        if row is not None and row["magnitude"].strip():
            out.append(_rec(panel, f"{h} margin", cell, _HYP, row, key, value_col="magnitude",
                            scale=scale, unit=unit, note="pre-registered margin"))  # fmt: skip
        if row is not None and row["flip_floor"].strip():
            out.append(_rec(panel, f"{h} flip floor", cell, _HYP, row, key,
                            text_col="flip_floor", plot=False,
                            note="P8 one-ulp flip rate (SS6 contrast)"))  # fmt: skip
    return out


def _selected_ta(inp: _Inputs) -> Rec:
    rec = next(
        (h for h in inp.report.hypotheses.get("hypotheses", []) if h.get("hypothesis") == "H3"),
        None,
    )
    found = [str(c) for c in (rec or {}).get("caveats", []) if "selected at Ta" in str(c)]
    out: Rec = {
        "panel": "all",
        "series": "selected Ta",
        "x": "",
        "unit": "control steps",
        "scale": 1.0,
        "note": "Ta parsed from the caveat text",
        "source_file": _HYP_JSON,
        "source_row": "hypothesis=H3",
        "text_col": "caveats[]",
    }
    if len(found) == 1:
        m = re.search(r"Ta = (\d+)", found[0])
        out["text"] = found[0]
        if m:
            out["value"] = float(m.group(1))
    return out


def chunking_table(inp: _Inputs) -> list[Rec]:
    """F5: H3's Ta curve (dp, same checkpoints) at ``id`` SS4-SS6, with and without ensembling."""
    out: list[Rec] = []
    for r in inp.h3:
        key = {"panel": r["panel"], "x": r["x"], "series": r["series"]}
        if r["point"].strip():
            out.append(_rec(r["panel"], r["series"], r["x"], _H3, r, key, value_col="point",
                            lo_col="lo", hi_col="hi", unit="%", note=r["note"]))  # fmt: skip
        else:
            out.append(_rec(r["panel"], r["series"], r["x"], _H3, r, key, plot=False,
                            note=r["note"]))  # fmt: skip
    out.append(_selected_ta(inp))
    keys = [{"tag": "dp_Ta", "arm": "dp", "list": "id", "ss": ss, "Ta": ta, "ensemble": e}
            for ss in ("SS4", "SS5", "SS6") for ta in ("1", "2", "4", "8", "12")
            for e in ("True", "False")]  # fmt: skip
    out.append(_meta("n_seeds", _AGG, inp.report.aggregate, keys, "n_seeds"))
    return out


def gaps_table(inp: _Inputs) -> list[Rec]:
    """F6: H4 at ``id`` SS6: levels (teacher rate, students' mean) and P, I, D with CIs."""
    r = inp.report
    out = [_baseline(r, "levels", "ppo", "id", "SS6")]
    out[0]["note"] = "teacher P8 ppo s0 success rate (H4's S(teacher)), Wilson 95 %"
    out += [_learned(r, "levels", a, "id", "SS6", col="mean") for a in ("bc_state", "bc")]
    for x in ("H4 I", "H4 P", "H4"):
        key = {"x": x}
        row = _one(inp.h4, **key)
        out.append(_rec("gaps", x, "id SS6", _H4, row, key, value_col="point", lo_col="lo",
                        hi_col="hi", unit="points", text_col="series",
                        note="three-arm bootstrap 95 % CI"))  # fmt: skip
    key = {"hypothesis": "H4"}
    row = _one(inp.hyp_csv, **key)
    out.append(_rec("gaps", "H4 verdict", "id SS6", _HYP, row, key, text_col="verdict",
                    plot=False))  # fmt: skip
    out.append(_rec("gaps", "H4 flip floor", "id SS6", _HYP, row, key, text_col="flip_floor",
                    plot=False, note="P8 one-ulp flip rate (SS6 contrast)"))  # fmt: skip
    out.append(_rec("gaps", "H4 notes", "id SS6", _HYP, row, key, text_col="notes",
                    plot=False))  # fmt: skip
    return out + _design_meta(r, ("bc_state", "bc"), [("id", "SS6")])


_PROVIDERS: Final[tuple[str, ...]] = (
    "torch_cpu_fp32",
    "ort_cpu_fp32",
    "torch_cuda_fp32",
    "ort_cuda_fp32",
    "ort_trt_fp32",
    "ort_trt_fp16",
    "ort_cuda_fp16",
)
_E2E: Final[Mapping[str, str]] = {"bc_dagger": "bc_e2e", "dp": "dp_e2e"}


def latency_table(inp: _Inputs) -> list[Rec]:
    """F7: batch-1 end-to-end p99 (ms) per provider with parity status; control period."""
    out: list[Rec] = []
    for arm, row_name in _E2E.items():
        for prov in _PROVIDERS:
            lkey = {"label": f"{row_name}-{prov}-b1-t1"}
            lat = _one(inp.latency, **lkey)
            timed = lat is not None and lat["status"] == "timed"
            out.append(_rec(arm, prov, "p99", _LATENCY, lat, lkey, value_col="p99_ms",
                            n_col="timed_iters" if timed else "", text_col="status", unit="ms",
                            plot=timed, note="batch 1, end-to-end, wall clock"))  # fmt: skip
            out.append(_rec(arm, prov, "scored", _LATENCY, lat, lkey, text_col="scored",
                            plot=False))  # fmt: skip
            key = {"arm": arm, "provider": prov}
            par = _one(inp.parity, **key)
            if par is not None:  # the torch_cpu_fp32 reference has no parity row
                out.append(_rec(arm, prov, "open_loop_pass", _PARITY, par, key,
                                text_col="open_loop_pass", plot=False))  # fmt: skip
            cl = _one(inp.closed, **key)
            out.append(_rec(arm, prov, "closed_loop_pass", _CLOSED, cl, key,
                            text_col="closed_loop_pass", plot=False))  # fmt: skip
        delay = inp.delay["arms"][arm]
        out.append(
            {
                "panel": arm,
                "series": str(delay["provider"]),
                "x": "latency-arm delay d",
                "value": float(delay["d"]),
                "unit": "control steps",
                "scale": 1.0,
                "note": "provider used for the latency arm",
                "source_file": _DELAY,
                "source_row": f"arms.{arm}",
                "value_col": "d",
                "text_col": "provider",
                "text": str(delay["provider"]),
            }
        )
    out.append(_period_meta(inp))
    return out


def tables(repo: Path = p8_adapter.REPO_ROOT) -> dict[str, list[Rec]]:
    """Every figure's plot table from the committed inputs under ``repo``."""
    inp = _load(repo)
    return {
        "success_id": success_id_table(inp),
        "conditions": conditions_table(inp),
        "outcomes": outcomes_table(inp),
        "hypotheses": hypotheses_table(inp),
        "chunking": chunking_table(inp),
        "gaps": gaps_table(inp),
        "latency": latency_table(inp),
    }


# --------------------------------------------------------------------------- drawing helpers


def _rc(t: Theme) -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": _FS,
        "svg.hashsalt": "ila",
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "text.color": t.ink,
        "axes.labelcolor": t.ink,
        "axes.edgecolor": t.secondary,
        "axes.linewidth": 0.8,
        "axes.titlesize": _FS + 1,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": t.secondary,
        "ytick.color": t.secondary,
        "xtick.labelcolor": t.ink,
        "ytick.labelcolor": t.ink,
        "xtick.labelsize": _FS,
        "ytick.labelsize": _FS,
        "legend.fontsize": _FS,
        "legend.frameon": False,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "hatch.linewidth": 1.4,
        "axes.unicode_minus": True,
    }


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, metadata={"Software": None})
    _plt().close(fig)


def _note(fig: Any, t: Theme, text: str, y: float = 0.012) -> None:
    fig.text(0.012, y, text, fontsize=_FS - 0.5, color=t.secondary, ha="left", va="bottom")


def _title(fig: Any, text: str, y: float = 0.985) -> None:
    fig.text(0.012, y, text, fontsize=_FS + 3, fontweight="bold", ha="left", va="top")


def _pts(recs: Sequence[Rec], panel: str, series: str) -> list[Rec]:
    return [r for r in recs if r["panel"] == panel and r["series"] == series and "value" in r]


def _spread(ys: Sequence[float], gap: float) -> list[float]:
    """Push label positions apart (keeping order) so neighbours are >= ``gap`` apart."""
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        out[cur] = min(out[cur], out[prev] - gap)
    return out


def _mv(recs: Sequence[Rec], series: str) -> str:
    """A meta count (``panel == "meta"``) as text, e.g. ``"5"``; ``"?"`` if absent."""
    r = next((r for r in recs if r["panel"] == "meta" and r["series"] == series), None)
    if r is None or "value" not in r:
        return "?"
    v = float(r["value"])
    return f"{v:.0f}" if v == int(v) else f"{v:.1f}"


def _row_label(arm: str) -> str:
    if arm == "ppo":
        return _BASE_LABEL[arm]
    if arm in _BASE_LABEL:
        return _BASE_LABEL[arm] + " (P8)"
    return _ARM_LABEL.get(arm, arm)


def _fmt1(v: float) -> str:
    return f"{v:.1f}".replace("-", "−")


# --------------------------------------------------------------------------- F1 success_id


def _draw_success_id(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    from matplotlib.lines import Line2D

    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 6.8))
        axes = [
            fig.add_axes((0.075, 0.15, 0.405, 0.62)),
            fig.add_axes((0.585, 0.15, 0.405, 0.62)),
        ]
        pos = {ss: i for i, ss in enumerate(_ID_SS)}
        for ax, (panel, arms) in zip(axes, _F1_PANELS, strict=True):
            ax.axvspan(2.5, 3.5, color=t.shade, lw=0, zorder=0)
            ax.grid(True, axis="y", zorder=0)
            for m in _BASELINES:
                b = _pts(recs, panel, m)
                # The teacher is drawn above the students (thinner) so it is never hidden.
                ax.plot([pos[r["x"]] for r in b], [r["value"] for r in b],
                        color=t.ink if m == "ppo" else t.baseline, ls=_BASE_LS[m],
                        lw=_LW * (0.8 if m == "ppo" else 1.0),
                        zorder=4 if m == "ppo" else 2)  # fmt: skip
            n = len(arms)
            lows = [100.0]
            for k, arm in enumerate(arms):
                a = _pts(recs, panel, arm)
                off = (k - (n - 1) / 2) * 0.1
                xs = [pos[r["x"]] + off for r in a]
                ys = [r["value"] for r in a]
                lows += [r["lo"] for r in a]
                err = [[max(0.0, r["value"] - r["lo"]) for r in a],
                       [max(0.0, r["hi"] - r["value"]) for r in a]]  # fmt: skip
                ax.errorbar(xs, ys, yerr=err, color=t.arm(arm), lw=_LW, elinewidth=_LW,
                            capsize=0, marker=_MARKER[arm], ms=_MS, mec=t.surface, mew=0.9,
                            label=_ARM_LABEL.get(arm, arm), zorder=3)  # fmt: skip
            lows += [r["value"] for m in _BASELINES for r in _pts(recs, panel, m)]
            ax.set_ylim(math.floor((min(lows) - 4.0) / 10.0) * 10.0, 101.5)
            ax.set_xticks(range(len(_ID_SS)), list(_ID_SS))
            for ss in _ID_SS:
                if sc.outside_training("id", ss):
                    ax.text(pos[ss], 1.01, "outside\ntraining", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=_FS - 1, color=t.secondary,
                            linespacing=1.0)  # fmt: skip
            ax.set_xlim(-0.5, 3.5)
            ax.set_title(panel, pad=8)
            ax.set_ylabel("success, %")
            ax.set_xlabel("sea state")
            low_panel = min(lows) < 50.0
            ax.legend(loc="upper left" if low_panel else "lower left",
                      bbox_to_anchor=(0.0, 0.74) if low_panel else None, ncol=1,
                      handlelength=1.6, borderaxespad=0.3)  # fmt: skip
        handles = [Line2D([], [], color=t.ink if m == "ppo" else t.baseline, ls=_BASE_LS[m],
                          lw=_LW * (0.8 if m == "ppo" else 1.0)) for m in _BASELINES]  # fmt: skip
        fig.legend(handles, [_BASE_LABEL[m] for m in _BASELINES], loc="upper left", ncol=4,
                   bbox_to_anchor=(0.005, 0.895), handlelength=2.4, columnspacing=1.4,
                   handletextpad=0.5)  # fmt: skip
        _title(fig, "Simulated landing success on the id episode list, by sea state")
        fig.text(0.012, 0.925, f"Students (colour): IQM over {_mv(recs, 'n_seeds')} seeds, "
                 "95 % CI.  Lines: teacher and P8 baselines (P8 rows).  y-ranges differ.",
                 fontsize=_FS, color=t.secondary, ha="left", va="top")  # fmt: skip
        _note(fig, t, f"{_SIM_NOTE} · {_TEACHER_NOTE}")
        _save(fig, path)


# --------------------------------------------------------------------------- F2 conditions


def _mix(a: str, b: str, w: float) -> tuple[float, float, float]:
    ca = [int(a[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, bl = (x + (y - x) * w for x, y in zip(ca, cb, strict=True))
    return (r, g, bl)


def _lum(c: tuple[float, float, float]) -> float:
    def lin(u: float) -> float:
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])


_GROUP_LABEL: Final[Mapping[str, str]] = {
    "static": "static",
    "unseen_heading": "unseen heading\n(90°)",
    "unseen_vessel": "unseen vessel",
    "unseen_seastate": "unseen\nseastate",
}
_HEAT_BREAK: Final[float] = 70.0  # % ; break of the two-segment colour scale
_HEAT_W_BREAK: Final[float] = 0.36  # colour weight at the break (0 = surface, 1 = full hue)


def _heat_weight(v: float) -> float:
    """Two linear segments: 0-70 % -> weight 0.04-0.36, 70-100 % -> 0.36-1.0 (stretched)."""
    v = min(100.0, max(0.0, v))
    if v <= _HEAT_BREAK:
        return 0.04 + (_HEAT_W_BREAK - 0.04) * v / _HEAT_BREAK
    return _HEAT_W_BREAK + (1.0 - _HEAT_W_BREAK) * (v - _HEAT_BREAK) / (100.0 - _HEAT_BREAK)


def _draw_conditions(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    rows = [*_BASELINES, *STUDENTS]
    cols = [f"{lst} {ss}" if lst != "static" else "static" for lst, ss in _F2_COLS]
    cell = {(r["series"], r["x"]): r for r in recs}
    hue = t.palette[0]
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 7.4))
        ax = fig.add_axes((0.26, 0.10, 0.72, 0.62))
        for i, arm in enumerate(rows):
            for j, c in enumerate(cols):
                r = cell.get((arm, c))
                if r is None or "value" not in r:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=t.surface,
                                               edgecolor=t.grid, lw=1.0))  # fmt: skip
                    ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center", color=t.secondary)
                    continue
                v = float(r["value"])
                fill = _mix(t.surface, hue, _heat_weight(v))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fill, edgecolor=t.surface,
                                           lw=2.0))  # fmt: skip
                surf_l = _lum(_mix(t.surface, t.surface, 0.0))
                ink_l = _lum(_mix(t.ink, t.ink, 0.0))
                fl = _lum(fill)
                contrast_ink = (max(fl, ink_l) + 0.05) / (min(fl, ink_l) + 0.05)
                contrast_surf = (max(fl, surf_l) + 0.05) / (min(fl, surf_l) + 0.05)
                colour = t.ink if contrast_ink >= contrast_surf else t.surface
                ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center", color=colour,
                        fontsize=_FS)  # fmt: skip
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(rows), 0)
        ax.set_yticks([i + 0.5 for i in range(len(rows))], [_row_label(a) for a in rows])
        ax.tick_params(length=0)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(len(_BASELINES), color=t.ink, lw=1.2)
        # Column headers: list group (spanning its columns) over sea state; outside-training
        # columns get a shaded header box.
        for j, (lst, ss) in enumerate(_F2_COLS):
            if sc.outside_training(lst, ss):
                ax.add_patch(plt.Rectangle((j + 0.03, -0.62), 0.94, 0.56, facecolor=t.grid,
                                           edgecolor="none", clip_on=False))  # fmt: skip
            ax.text(j + 0.5, -0.33, "deck" if lst == "static" else ss, ha="center",
                    va="center", fontsize=_FS, fontweight="bold")  # fmt: skip
        groups: dict[str, list[int]] = {}
        for j, (lst, _) in enumerate(_F2_COLS):
            groups.setdefault(lst, []).append(j)
        for lst, js in groups.items():
            a, b = min(js), max(js) + 1
            ax.plot([a + 0.08, b - 0.08], [-0.72, -0.72], color=t.secondary, lw=0.8,
                    clip_on=False)  # fmt: skip
            ax.text((a + b) / 2, -0.8, _GROUP_LABEL.get(lst, lst), ha="center", va="bottom",
                    fontsize=_FS - 0.5, linespacing=1.05)  # fmt: skip
        _title(fig, "Simulated landing success (%) across lists and sea states")
        fig.text(0.012, 0.925, f"Students: IQM over {_mv(recs, 'n_seeds')} seeds; P8 rows: "
                 f"rate over {_mv(recs, 'n_episodes_p8')} episodes.  CIs and N in "
                 "conditions.csv.", fontsize=_FS, color=t.secondary, ha="left",
                 va="top")  # fmt: skip
        fig.text(0.012, 0.885, "Shaded = SS6 or 90° heading (outside training, P3-D1); "
                 "unseen_vessel is a vessel never trained on.", fontsize=_FS,
                 color=t.secondary, ha="left", va="top")  # fmt: skip
        _note(fig, t, f"Colour: two linear segments, 0–{_HEAT_BREAK:.0f} % and "
              f"{_HEAT_BREAK:.0f}–100 % (upper segment stretched).  — = withheld or missing.",
              y=0.045)  # fmt: skip
        _note(fig, t, f"{_SIM_NOTE} · {_TEACHER_NOTE}", y=0.012)
        _save(fig, path)


# --------------------------------------------------------------------------- F3 outcomes


def _draw_outcomes(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    rows = [*_BASELINES, *STUDENTS]
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 6.8))
        axes = [fig.add_axes((0.25, 0.145, 0.325, 0.60))]
        axes.append(fig.add_axes((0.635, 0.145, 0.33, 0.60), sharey=axes[0]))
        for ax, panel in zip(axes, ("id SS5", "id SS6"), strict=True):
            if panel.endswith("SS6"):
                ax.set_facecolor(t.shade)
            for i, arm in enumerate(rows):
                mine = {r["x"]: r for r in recs if r["panel"] == panel and r["series"] == arm}
                left = 0.0
                for o in _OUTCOME_ORDER:
                    r = mine.get(o)
                    if r is None or "value" not in r:
                        continue
                    w = 100.0 * float(r["value"]) / float(r["n"])
                    if w > 0:
                        ax.barh(i, w, left=left, height=0.72,
                                color=t.palette[_OUTCOME_SLOT[o]], edgecolor=t.surface,
                                lw=1.0, zorder=2)  # fmt: skip
                    left += w
                if not mine or all("value" not in r for r in mine.values()):
                    ax.text(1, i, "—", va="center", color=t.secondary)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_xticks([25, 75], minor=True)
            ax.set_xlabel("share of episodes, %")
            ax.set_title(panel + ("  (outside training)" if panel.endswith("SS6") else ""),
                         pad=8)  # fmt: skip
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0].set_ylim(len(rows) - 0.5, -0.5)
        axes[0].set_yticks(range(len(rows)), [_row_label(a) for a in rows])
        axes[1].tick_params(labelleft=False)
        for ax in axes:
            ax.axhline(len(_BASELINES) - 0.5, color=t.ink, lw=1.0)
        handles = [plt.Rectangle((0, 0), 1, 1, color=t.palette[_OUTCOME_SLOT[o]])
                   for o in _OUTCOME_ORDER]  # fmt: skip
        fig.legend(handles, list(_OUTCOME_ORDER), loc="upper left", ncol=6,
                   bbox_to_anchor=(0.005, 0.89), handlelength=1.1, columnspacing=1.2,
                   handletextpad=0.4)  # fmt: skip
        _title(fig, "How simulated episodes end: P8's six outcome classes")
        sub = (
            f"Students: pooled over {_mv(recs, 'n_seeds')} seeds × "
            f"{_mv(recs, 'n_episodes_per_seed')} episodes.  Teacher and P8 baselines: "
            f"{_mv(recs, 'n_episodes_p8')} episodes each (P8 rows)."
        )
        fig.text(0.012, 0.925, sub, fontsize=_FS, color=t.secondary, ha="left", va="top")
        _note(fig, t, f"{_SIM_NOTE} · {_TEACHER_NOTE}")
        _save(fig, path)


# --------------------------------------------------------------------------- F4 hypotheses


def _h5a_text(recs: Sequence[Rec], margin: Rec | None) -> str:
    """H5(a)'s text row from the scored providers in ``h5a.csv`` (parity, p99 range)."""
    scored = [r for r in recs if r["series"] == "H5(a) scored p99"]
    timed = [float(r["value"]) for r in scored if "value" in r]
    thr = float(margin["value"]) if margin and "value" in margin else math.nan
    no_parity = bool(scored) and all(r.get("text") == "False" for r in scored)
    if not (no_parity and timed and all(v > thr for v in timed)):
        return "no estimate (see hypotheses.csv H5(a))"
    return (
        "no estimate: no scored provider passes both parities, and every scored p99 is "
        f"above {thr:.1f} ms ({min(timed):.1f}–{max(timed):.1f} ms)"
    )


def _flip_sentence(text: str) -> str:
    """P8's flip floor in full (P3-D1 §1), verbatim from ``flip_floor`` (backticks dropped)."""
    return f"P8 one-ulp flip floor: {plain(text)}" if text.strip() else ""


def _wrap(text: str, width: int) -> str:
    # A no-break space keeps a number with its unit ("1.9 %", "33.3 ms") on one line.
    text = re.sub(r"(\d) (%|ms)", "\\1\u00a0\\2", text)
    return "\n".join(textwrap.wrap(text, width=width, break_on_hyphens=False))


def _draw_hypotheses(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    by = {r["series"]: r for r in recs}
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 8.8))
        top = fig.add_axes((0.36, 0.445, 0.44, 0.43))
        bot = fig.add_axes((0.36, 0.28, 0.44, 0.06))
        flip = ""
        for ax, names in ((top, _H_POINTS), (bot, ("H2",))):
            labels = []
            for i, h in enumerate(names):
                r = by.get(h)
                dag = ""
                if f"{h} flip floor" in by:
                    dag = " †"
                    flip = flip or _flip_sentence(by[f"{h} flip floor"].get("text", ""))
                labels.append(f"{h} · {r['x'] if r else ''}{dag}\n{_H_LABEL[h]}")
                if r is None or "value" not in r:
                    ax.text(0.5, i, "not scored", transform=ax.get_yaxis_transform(),
                            ha="center", va="center", color=t.secondary)  # fmt: skip
                    continue
                ax.plot([r["lo"], r["hi"]], [i, i], color=t.ink, lw=_LW * 1.2,
                        solid_capstyle="butt", zorder=3)  # fmt: skip
                ax.plot([r["value"]], [i], "o", color=t.ink, ms=_MS, mec=t.surface, zorder=4)
                mg = by.get(f"{h} margin")
                if mg is not None and "value" in mg:
                    ax.plot([mg["value"]], [i], marker="|", ms=_MS * 2.2, mew=_LW,
                            color=t.palette[7], zorder=2)  # fmt: skip
                ax.text(1.03, i, r.get("text", ""), transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=_FS, fontweight="bold")  # fmt: skip
            ax.set_yticks(range(len(names)), labels, fontsize=_FS - 0.5, linespacing=1.15)
            ax.set_ylim(len(names) - 0.5, -0.5)
            ax.axvline(0.0, color=t.secondary, lw=1.0, zorder=1)
            ax.grid(True, axis="x", zorder=0)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        top.set_xlabel("difference in success, points (95 % CI)")
        bot.set_xlabel("difference in W₁ distance, s (model scale; ×5 full scale)")
        xs = [v for r in recs if r["panel"] == "seconds" for k in ("lo", "hi", "value")
              if (v := r.get(k)) is not None]  # fmt: skip
        if xs:
            lim = max(abs(v) for v in xs) * 1.25
            bot.set_xlim(-lim, lim)
        a5 = by.get("H5(a)")
        a5m = by.get("H5(a) margin")
        thr = f"{a5m['value']:.1f}" if a5m and "value" in a5m else "?"
        y5 = 0.155
        fig.text(0.352, y5, f"H5(a) · dp_e2e, batch 1\np99 < {thr} ms, parity-passing provider",
                 fontsize=_FS - 0.5, ha="right", va="center", linespacing=1.15)  # fmt: skip
        fig.text(0.58, y5, _wrap(_h5a_text(recs, a5m), 42),
                 fontsize=_FS - 0.5, ha="center", va="center", color=t.secondary,
                 linespacing=1.15)  # fmt: skip
        fig.text(0.8132, y5, a5.get("text", "") if a5 else "", fontsize=_FS,
                 fontweight="bold", ha="left", va="center")  # fmt: skip
        fig.add_artist(plt.Line2D([0.03, 0.97], [0.215, 0.215], color=t.grid, lw=0.8))
        _title(fig, "Pre-registered hypotheses: estimate, 95 % CI, verdict")
        fig.text(0.012, 0.925, "Dot = estimate, bar = 95 % CI, red tick = pre-registered "
                 "margin; verdict judged on point estimate and CI.", fontsize=_FS,
                 color=t.secondary, ha="left", va="top")  # fmt: skip
        if flip:
            _note(fig, t, _wrap(f"† SS6 contrast. {flip}.", 118), y=0.042)
        _note(fig, t, _SIM_NOTE + " · no multiplicity correction")
        _save(fig, path)


# --------------------------------------------------------------------------- F5 chunking


def _draw_chunking(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    sel = next((r for r in recs if r["series"] == "selected Ta"), None)
    sel_ta = sel.get("value") if sel else None
    colour = t.arm("dp")
    styles: dict[str, dict[str, Any]] = {
        "ensembling": {"ls": (0, (5, 2.5)), "mfc": t.surface, "marker": "s",
                       "label": "with temporal ensembling"},
        "no ensembling": {"ls": "-", "mfc": colour, "marker": "o", "label": "no ensembling"},
    }  # fmt: skip
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 6.6))
        big = fig.add_axes((0.075, 0.185, 0.52, 0.575))
        s4 = fig.add_axes((0.70, 0.525, 0.28, 0.235), sharey=big)
        s6 = fig.add_axes((0.70, 0.185, 0.28, 0.235), sharey=big)
        for ax, panel, large in ((big, "id SS5", True), (s4, "id SS4", False),
                                 (s6, "id SS6", False)):  # fmt: skip
            if panel.endswith("SS6"):
                ax.set_facecolor(t.shade)
            ax.grid(True, axis="y", zorder=0)
            for series, st in styles.items():
                pts = sorted(_pts(recs, panel, series), key=lambda r: int(r["x"]))
                xs = [int(r["x"]) for r in pts]
                ys = [r["value"] for r in pts]
                err = [[max(0.0, r["value"] - r["lo"]) for r in pts],
                       [max(0.0, r["hi"] - r["value"]) for r in pts]]  # fmt: skip
                ax.errorbar(xs, ys, yerr=err, color=colour, ls=st["ls"], lw=_LW,
                            elinewidth=_LW, capsize=0, marker=st["marker"],
                            ms=_MS if large else _MS * 0.8, mfc=st["mfc"], mec=colour,
                            mew=1.4, zorder=3)  # fmt: skip
                if large and pts:
                    above = series == "ensembling"
                    ax.annotate(st["label"], xy=(xs[-1], ys[-1]),
                                xytext=(4, 16 if above else -22), textcoords="offset points",
                                va="bottom" if above else "top", ha="right", fontsize=_FS,
                                annotation_clip=False)  # fmt: skip
            if sel_ta is not None:
                ax.axvline(sel_ta, color=t.secondary, lw=1.0, ls=(0, (1, 2)), zorder=1)
            ax.set_xscale("log", base=2)
            ticks = sorted({int(r["x"]) for r in recs if r["panel"] == panel and "value" in r})
            ax.set_xticks(ticks, [str(k) for k in ticks])
            ax.minorticks_off()
            ax.set_ylim(0, 102)
            title = panel + ("  (outside training)" if panel.endswith("SS6") else "")
            ax.set_title(title, pad=6, fontsize=_FS + (1 if large else 0))
            if not large:
                ax.tick_params(labelsize=_FS - 1)
        if sel_ta is not None:
            big.text(sel_ta * 0.94, 4, f"checkpoints selected at Ta = {sel_ta:.0f}",
                     ha="right", va="bottom", fontsize=_FS - 0.5, color=t.secondary)  # fmt: skip
        big.set_xlabel("Ta (log scale)\nno ensembling: actions executed per chunk; "
                       "ensembling: window", linespacing=1.3)  # fmt: skip
        big.set_ylabel(f"success, %  (IQM over {_mv(recs, 'n_seeds')} seeds, 95 % CI)")
        s6.set_xlabel("Ta")
        _title(fig, "dp: success against Ta (same checkpoints)")
        sub = (
            "No ensembling: Ta actions executed per chunk.  Ensembling: replans every step, "
            "Ta = averaging window (identical at Ta = 1).  Competing explanation: stale "
            "last_action input (P7-D3 item 7)."
        )
        fig.text(0.012, 0.93, _wrap(sub, 100), fontsize=_FS, color=t.secondary, ha="left",
                 va="top", linespacing=1.3)  # fmt: skip
        _note(fig, t, _SIM_NOTE)
        _save(fig, path)


# --------------------------------------------------------------------------- F6 gaps


def _draw_gaps(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    lev = {r["series"]: r for r in recs if r["panel"] == "levels"}
    gap = {r["series"]: r for r in recs if r["panel"] == "gaps"}
    order = ("ppo", "bc_state", "bc")
    names = {"ppo": "teacher\n(P8 ppo s0)", "bc_state": "bc_state\n(privileged state)",
             "bc": "bc\n(rendered images)"}  # fmt: skip
    colours = {"ppo": t.ink, "bc_state": t.arm("bc_state"), "bc": t.arm("bc")}
    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 6.6))
        ax = fig.add_axes((0.09, 0.215, 0.56, 0.61))
        ax.set_facecolor(t.shade)
        ax.grid(True, axis="y", zorder=0)
        vals: dict[str, float] = {}
        for i, a in enumerate(order):
            r = lev.get(a)
            if r is None or "value" not in r:
                ax.text(i, 90, "—", ha="center", color=t.secondary)
                continue
            v = float(r["value"])
            vals[a] = v
            ax.plot([i - 0.32, i + 0.32], [v, v], color=colours[a], lw=_LW * 2.4,
                    solid_capstyle="butt", zorder=3)  # fmt: skip
            ax.text(i, v + 0.45, f"{v:.1f} %", ha="center", va="bottom", fontsize=_FS)
            if i > 0 and order[i - 1] in vals:
                ax.plot([i - 0.68, i - 0.32], [vals[order[i - 1]], v], color=t.secondary,
                        lw=1.0, ls=(0, (2, 2)), zorder=2)  # fmt: skip
        g_i, g_p = gap.get("H4 I"), gap.get("H4 P")
        if g_i is not None and "value" in g_i and {"ppo", "bc_state"} <= set(vals):
            y = min(vals["ppo"], vals["bc_state"]) - 0.9
            ax.text(0.5, y, f"I = {_fmt1(g_i['value'])} pts\n[{_fmt1(g_i['lo'])}, "
                    f"{_fmt1(g_i['hi'])}]", ha="center", va="top", fontsize=_FS)  # fmt: skip
        if g_p is not None and "value" in g_p and {"bc_state", "bc"} <= set(vals):
            y0, y1 = vals["bc"], vals["bc_state"]
            x = 2.42
            ax.plot([x - 0.06, x, x, x - 0.06], [y1, y1, y0, y0], color=t.ink, lw=1.2,
                    zorder=3)  # fmt: skip
            ax.text(2.0, (y0 + y1) / 2 + 0.6, f"P = {_fmt1(g_p['value'])} pts\n"
                    f"[{_fmt1(g_p['lo'])}, {_fmt1(g_p['hi'])}]", ha="center", va="center",
                    fontsize=_FS)  # fmt: skip
        ax.set_xticks(range(len(order)), [names[a] for a in order])
        ax.set_xlim(-0.5, 2.6)
        lo = min(vals.values(), default=80.0)
        ax.set_ylim(math.floor((lo - 4.0) / 5.0) * 5.0, 101.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.set_ylabel("mean success, %")
        ax.text(0.015, 0.02, "id SS6: outside training", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=_FS - 0.5, color=t.secondary)  # fmt: skip
        # Right column: the scored statistic and the verdict.
        d = gap.get("H4")
        verdict = gap.get("H4 verdict", {}).get("text", "")
        flip = _flip_sentence(gap.get("H4 flip floor", {}).get("text", ""))
        x0 = 0.69
        fig.text(x0, 0.78, "H4 scores D = P − I", fontsize=_FS + 1, fontweight="bold")
        if d is not None and "value" in d:
            fig.text(x0, 0.715, f"D = {_fmt1(d['value'])} points", fontsize=_FS + 1)
            fig.text(x0, 0.665, f"95 % CI [{_fmt1(d['lo'])}, {_fmt1(d['hi'])}]", fontsize=_FS)
        fig.text(x0, 0.60, f"verdict: {verdict}", fontsize=_FS + 1, fontweight="bold")
        lines = [
            "P: perception gap",
            "  (privileged state → images);",
            "  also includes: no time_fraction;",
            "  bf16 vs fp32 training",
            "I: imitation gap",
            "  (teacher → state student)",
        ]
        if "ceiling-limited" in gap.get("H4 notes", {}).get("text", ""):
            lines += ["", "I is ceiling-limited (the", "  teacher is near 100 %)."]
        fig.text(x0, 0.53, "\n".join(lines), fontsize=_FS - 0.5, color=t.secondary, va="top",
                 linespacing=1.3)  # fmt: skip
        _title(fig, "id SS6: from teacher to state student to image student")
        fig.text(0.012, 0.925, f"Levels: mean over {_mv(recs, 'n_seeds')} seeds × "
                 f"{_mv(recs, 'n_episodes_per_seed')} episodes (teacher: "
                 f"{_mv(recs, 'n_episodes_p8')}).  Gaps: three-arm bootstrap 95 % CI.",
                 fontsize=_FS, color=t.secondary, ha="left", va="top")  # fmt: skip
        if flip:
            _note(fig, t, _wrap(flip + ".", 118), y=0.045)
        _note(fig, t, f"{_SIM_NOTE} · {_TEACHER_NOTE}")
        _save(fig, path)


# --------------------------------------------------------------------------- F7 latency


def _draw_latency(recs: Sequence[Rec], t: Theme, path: Path) -> None:
    plt = _plt()
    period_rec = next((r for r in recs if r["series"] == "control period"), None)
    period = float(period_rec["value"]) if period_rec and "value" in period_rec else None
    box = {"boxstyle": "square,pad=0.15", "facecolor": t.surface, "edgecolor": "none"}

    def get(arm: str, prov: str, x: str) -> Rec | None:
        return next((r for r in recs if (r["panel"], r["series"], r["x"]) == (arm, prov, x)),
                    None)  # fmt: skip

    with plt.rc_context(_rc(t)):
        fig = plt.figure(figsize=(_W, 8.2))
        axes = {
            "bc_dagger": fig.add_axes((0.25, 0.535, 0.53, 0.29)),
            "dp": fig.add_axes((0.25, 0.135, 0.53, 0.30)),
        }
        for arm, ax in axes.items():
            colour = t.arm(arm)
            labels = []
            vmax = 0.0
            used = get(arm, next((r["series"] for r in recs if r["panel"] == arm and
                                  r["x"] == "latency-arm delay d"), ""),
                       "latency-arm delay d")  # fmt: skip
            for i, prov in enumerate(_PROVIDERS):
                lat = get(arm, prov, "p99")
                scored = (get(arm, prov, "scored") or {}).get("text", "") == "True"
                cl = (get(arm, prov, "closed_loop_pass") or {}).get("text", "")
                ref = cl == "reference"
                labels.append(prov + ("  (reference)" if ref else "" if scored else
                                      "  (not scored)"))  # fmt: skip
                if lat is None or "value" not in lat:
                    status = (lat or {}).get("text", "") or "missing"
                    why = "not timed: fails open-loop parity" if "parity" in status else status
                    ax.text(0.0, i, " " + why, va="center", ha="left", fontsize=_FS - 0.5,
                            color=t.secondary, bbox=box, zorder=5)  # fmt: skip
                    continue
                v = float(lat["value"])
                vmax = max(vmax, v)
                fails = cl == "False"
                ax.barh(i, v, height=0.62, color=t.surface if fails else colour,
                        edgecolor=colour, hatch="////" if fails else None, lw=1.4,
                        zorder=2)  # fmt: skip
                txt = f"{v:.1f} ms"
                if fails:
                    txt += " · fails closed-loop parity"
                if used is not None and used.get("text") == prov:
                    txt += f" · latency arm, d = {used['value']:.0f}"
                ax.annotate(txt, xy=(v, i), xytext=(6, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=_FS - 0.5, bbox=box,
                            zorder=5, annotation_clip=False)  # fmt: skip
            if period is not None:
                ax.axvline(period, color=t.ink, lw=1.2, ls=(0, (4, 2)), zorder=4)
            ax.set_yticks(range(len(_PROVIDERS)), labels)
            ax.set_ylim(len(_PROVIDERS) - 0.5, -0.5)
            ax.set_xlim(0, max(vmax, period or 0.0) * 1.08)
            ax.grid(True, axis="x", zorder=0)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel("p99 latency, ms (batch 1, end-to-end)")
            ax.set_title(arm, pad=8)
        _title(fig, "Inference latency vs the control period")
        if period is not None:
            sub = (
                f"Dashed line: one {1000.0 / period:.0f} Hz control step, {period:.1f} ms model "
                f"scale = {period * _TIME_SCALE:.1f} ms full scale (derived: ×5 for λ = 1/25)."
                "\nHatched: fails closed-loop parity."
            )
        else:
            sub = "Hatched: fails closed-loop parity."
        fig.text(0.012, 0.935, sub, fontsize=_FS, color=t.secondary, ha="left", va="top",
                 linespacing=1.3)  # fmt: skip
        _note(fig, t, "Wall clock on this machine (RTX A4000, WSL2), not scaled · simulation "
              "(PyBullet; TinyRenderer images)")  # fmt: skip
        _save(fig, path)


# --------------------------------------------------------------------------- entry

_DRAW: Final[Mapping[str, Callable[[Sequence[Rec], Theme, Path], None]]] = {
    "success_id": _draw_success_id,
    "conditions": _draw_conditions,
    "outcomes": _draw_outcomes,
    "hypotheses": _draw_hypotheses,
    "chunking": _draw_chunking,
    "gaps": _draw_gaps,
    "latency": _draw_latency,
}


def make_readme_figures(
    repo: Path = p8_adapter.REPO_ROOT, out_dir: Path | None = None
) -> dict[str, list[Path]]:
    """Write every README figure (light PNG, dark PNG, CSV); return name -> written paths.

    Args:
        repo: Repository root holding the committed ``results/`` inputs.
        out_dir: Output directory; ``None`` -> ``repo / docs/figures``.

    Returns:
        Figure name -> ``[<name>.png, <name>_dark.png, <name>.csv]``.
    """
    out = repo / OUT_DIR if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for name, recs in tables(repo).items():
        csv_path = out / f"{name}.csv"
        _write_table(csv_path, COLUMNS, recs)
        paths = []
        for theme in THEMES:
            p = out / f"{name}{theme.suffix}.png"
            _DRAW[name](recs, theme, p)
            paths.append(p)
        written[name] = [*paths, csv_path]
    return written
