"""Phase 8 figures from committed CSVs only (``results/e08``); each PNG beside the CSV it plots.

Simulation only (PyBullet physics, PyBullet TinyRenderer frames). Success in percent; times
seconds model scale (lambda = 1/25; x5 full scale). Every figure is drawn from a small "plot
table" (written next to the PNG as ``<name>.csv``) derived from the committed
``aggregate.csv``, ``baselines.csv``, ``contrasts.csv`` and ``h2_touchdown_times.csv``; nothing
reads a row file or P8. Cells withheld by the above-teacher rule
(:attr:`ila.eval.report.ReportInputs.withheld`) are left out of the plot tables and named in
each figure's footnote. Missing cells are omitted and named the same way.

Style: Okabe-Ito colours (colour-blind safe), the teacher solid black, P8 baselines in grey
line styles; SS6 and the 90° (``unseen_heading``) cells shaded as outside the training
distribution; learned arms as IQM with the stratified-bootstrap 95 % CI, baselines as rate with
the Wilson 95 % CI. PNGs are written with the Agg backend, a fixed DPI and no metadata stamps,
so the same inputs give the same bytes.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.eval import report as rp
from ila.eval import score as sc

__all__ = ["FIGURES", "FIG_DIR", "h4_caption", "make_figures"]

FIG_DIR: Final[str] = "figures"

#: Figure names (each ``<name>.png`` + ``<name>.csv``).
FIGURES: Final[tuple[str, ...]] = (
    "success_vs_seastate",
    "h4_gaps",
    "h3_ta_curve",
    "h2_touchdown_times",
    "dp_k_sweep",
)

_ARM_COLOURS: Final[Mapping[str, str]] = {
    "bc_state": "#56B4E9",
    "bc": "#E69F00",
    "bc_matched": "#009E73",
    "bc_dart": "#0072B2",
    "bc_dagger": "#D55E00",
    "dp": "#CC79A7",
    "modular": "#8C6D00",
}
_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D", "v", "P", "X")
_BASE_STYLE: Final[Mapping[str, Mapping[str, Any]]] = {
    "ppo": {"color": "#000000", "ls": "-", "lw": 2.0},
    "pid_feedforward": {"color": "#777777", "ls": "--", "lw": 1.2},
    "pid_track_descend": {"color": "#777777", "ls": ":", "lw": 1.2},
    "oracle_gated": {"color": "#777777", "ls": "-.", "lw": 1.2},
}
_BASE_NAME: Final[Mapping[str, str]] = {
    "ppo": "teacher ppo s0 (P8)",
    "pid_feedforward": "pid_feedforward (P8)",
    "pid_track_descend": "pid_track_descend (P8)",
    "oracle_gated": "oracle_gated (P8; commit-timing oracle, not a bound)",
}
_SHADE: Final[str] = "0.9"
_PLOT_COLUMNS: Final[tuple[str, ...]] = (
    "panel",
    "x",
    "series",
    "kind",
    "point",
    "lo",
    "hi",
    "note",
)


def _x(v: Any) -> float:
    text = "" if v is None else str(v).strip()
    return math.nan if text in ("", "None") else float(text)


def _write_table(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([sc.fmt(r.get(c)) for c in columns])
    path.write_text(buf.getvalue(), encoding="utf-8")


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=120, metadata={"Software": None})
    _plt().close(fig)


def _footnote(fig: Any, notes: Sequence[str]) -> None:
    text = (
        "Simulation only (PyBullet; rendered by PyBullet TinyRenderer); model scale, λ = 1/25. "
        + " ".join(notes)
    )
    fig.text(0.01, 0.005, text, fontsize=6.5, ha="left", va="bottom", wrap=True)


def _rc() -> dict[str, Any]:
    return {"font.family": "DejaVu Sans", "font.size": 8, "svg.hashsalt": "ila"}


# --------------------------------------------------------------------------- tables


def _learned_ok(r: Mapping[str, str], withheld: frozenset[str]) -> str:
    if rp.label_of(r) in withheld:
        return "withheld"
    if r["status"] != "complete":
        return "missing"
    return ""


def success_table(inp: rp.ReportInputs) -> list[dict[str, Any]]:
    """Core arms (IQM [CI]) and the four baselines (rate [Wilson]) per list x sea state, %."""
    withheld = inp.withheld
    out: list[dict[str, Any]] = []
    for r in inp.aggregate:
        if r["tag"] != "core":
            continue
        why = _learned_ok(r, withheld)
        rec = {"panel": r["list"], "x": r["ss"], "series": r["arm"], "kind": "learned"}
        if why:
            out.append({**rec, "note": why})
            continue
        out.append(
            {
                **rec,
                "point": 100.0 * _x(r["success_iqm"]),
                "lo": 100.0 * _x(r["success_iqm_lo"]),
                "hi": 100.0 * _x(r["success_iqm_hi"]),
                "note": "IQM over 5 seeds, stratified-bootstrap 95 % CI",
            }
        )
    for b in inp.baselines:
        if b["method"] not in sc.ALWAYS_PRINTED:
            continue
        out.append(
            {
                "panel": b["list"],
                "x": b["ss"],
                "series": b["method"],
                "kind": "baseline",
                "point": 100.0 * _x(b["success_rate"]),
                "lo": 100.0 * _x(b["success_wilson_lo"]),
                "hi": 100.0 * _x(b["success_wilson_hi"]),
                "note": f"{b['n_success']}/{b['n_episodes']}, Wilson 95 %",
            }
        )
    return out


def _sweep_table(inp: rp.ReportInputs, tag: str) -> list[dict[str, Any]]:
    withheld = inp.withheld
    out: list[dict[str, Any]] = []
    for r in inp.aggregate:
        if r["tag"] != tag:
            continue
        x = r["Ta"] if tag == "dp_Ta" else r["K"]
        series = (
            ("ensembling" if r["ensemble"] == "True" else "no ensembling")
            if tag == "dp_Ta"
            else "Ta = 8, no ensembling"
        )
        rec = {"panel": f"{r['list']} {r['ss']}", "x": x, "series": series, "kind": "learned"}
        why = _learned_ok(r, withheld)
        if why:
            out.append({**rec, "note": why})
            continue
        out.append(
            {
                **rec,
                "point": 100.0 * _x(r["success_iqm"]),
                "lo": 100.0 * _x(r["success_iqm_lo"]),
                "hi": 100.0 * _x(r["success_iqm_hi"]),
                "note": "schedule-limited (P3-D1 §3)"
                if tag == "dp_K" and x in ("1", "2")
                else "IQM over 5 seeds",
            }
        )
    return out


def h4_table(inp: rp.ReportInputs) -> list[dict[str, Any]]:
    """H4's P, I and D (points) with their 95 % CIs, from ``contrasts.csv``."""
    withheld = inp.withheld
    names = {
        "H4 P": "P = S(bc_state) - S(bc)",
        "H4 I": "I = S(teacher) - S(bc_state)",
        "H4": "D = P - I",
    }
    out: list[dict[str, Any]] = []
    for r in inp.contrasts:
        if r["name"] not in names:
            continue
        rec = {"panel": "id SS6", "x": r["name"], "series": names[r["name"]], "kind": r["kind"]}
        if r["status"].startswith("missing"):
            out.append({**rec, "note": "missing"})
        elif rp.withheld_inputs(r.get("inputs", ""), withheld):
            out.append({**rec, "note": "withheld"})
        else:
            out.append(
                {
                    **rec,
                    "point": 100.0 * _x(r["estimate"]),
                    "lo": 100.0 * _x(r["ci_lo"]),
                    "hi": 100.0 * _x(r["ci_hi"]),
                    "note": "three-arm bootstrap, 10 000 reps",
                }
            )
    return out


def h2_table(root: Path, inp: rp.ReportInputs) -> list[dict[str, Any]]:
    """Touched-down ``time_to_touchdown_s`` at ``id`` SS4 (s model scale), one row per episode."""
    path = root / "h2_touchdown_times.csv"
    if not path.is_file():
        return []
    held = {lab.split(" ")[1] for lab in inp.withheld if lab.endswith("id SS4")}
    out: list[dict[str, Any]] = [
        {"panel": "id SS4", "x": "", "series": arm, "kind": "", "note": "withheld"}
        for arm in sorted(held & {"bc_mix", "dp_mix"})
    ]
    for r in sc.read_csv(path):
        group = r["group"].split(":")[0] if r["group"].startswith("reference") else r["group"]
        if group in held:
            continue
        out.append(
            {
                "panel": "id SS4",
                "x": r["time_to_touchdown_s"],
                "series": r["group"],
                "kind": "touchdown" if r["touchdown_contact"] == "True" else "no touchdown",
                "point": None,
                "lo": None,
                "hi": None,
                "note": f"seed {r['seed']} index {r['index']} {r['outcome']}",
            }
        )
    return out


# --------------------------------------------------------------------------- drawing


def _errorbar(ax: Any, xs: Sequence[float], rows: Sequence[Mapping[str, Any]], **kw: Any) -> None:
    pts = [r["point"] for r in rows]
    lo = [max(0.0, r["point"] - r["lo"]) for r in rows]
    hi = [max(0.0, r["hi"] - r["point"]) for r in rows]
    ax.errorbar(xs, pts, yerr=[lo, hi], capsize=2.5, **kw)


def _notes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    bad = [r for r in rows if r.get("note") in ("missing", "withheld")]
    if not bad:
        return []
    items = sorted({f"{r['series']} {r['panel']} {r['x']} ({r['note']})" for r in bad})
    return ["Not plotted: " + "; ".join(items) + "."]


def _draw_success(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _plt()
    panels = list(dict.fromkeys(r["panel"] for r in rows))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(
            1, max(1, len(panels)), figsize=(3.2 * max(1, len(panels)), 5.2), sharey=True,
            squeeze=False,
        )  # fmt: skip
        for ax, panel in zip(axes[0], panels, strict=False):
            mine = [r for r in rows if r["panel"] == panel]
            cells = list(dict.fromkeys(r["x"] for r in mine))
            pos = {c: i for i, c in enumerate(cells)}
            for c in cells:
                if sc.outside_training(panel, c):
                    ax.axvspan(pos[c] - 0.5, pos[c] + 0.5, color=_SHADE, lw=0, zorder=0)
            for m, style in _BASE_STYLE.items():
                b = [r for r in mine if r["series"] == m and r.get("point") is not None]
                if b:
                    ax.plot([pos[r["x"]] for r in b], [r["point"] for r in b],
                            label=_BASE_NAME[m], marker="_", ms=10, **style)  # fmt: skip
            arms = list(dict.fromkeys(r["series"] for r in mine if r["kind"] == "learned"))
            n = max(1, len(arms))
            for k, arm in enumerate(arms):
                a = [r for r in mine if r["series"] == arm and r.get("point") is not None]
                if not a:
                    continue
                off = (k - (n - 1) / 2) * 0.06
                _errorbar(
                    ax, [pos[r["x"]] + off for r in a], a,
                    color=_ARM_COLOURS.get(arm, "#444444"), marker=_MARKERS[k % len(_MARKERS)],
                    ms=4, lw=1.0, label=f"{arm} (IQM, 95 % CI)",
                )  # fmt: skip
            ax.set_xticks(range(len(cells)), cells)
            ax.set_xlim(-0.5, len(cells) - 0.5)
            ax.set_title(panel + (" (90° heading)" if panel == "unseen_heading" else ""))
            ax.grid(True, axis="y", lw=0.3, alpha=0.6)
        axes[0, 0].set_ylabel("success, % (simulated landings)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes[0, 1:]:
            for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.95))  # fmt: skip
        fig.suptitle(
            "Success per sea state, students distilled from P8 ppo seed 0, beside the teacher "
            "and P8 baselines (shaded: outside the training distribution)", fontsize=9, y=0.995,
        )  # fmt: skip
        _footnote(fig, _notes(rows))
        fig.subplots_adjust(left=0.05, right=0.99, top=0.72, bottom=0.17, wspace=0.08)
        _save(fig, path)


def h4_caption(inp: rp.ReportInputs) -> list[str]:
    """H4's caption: I's ceiling (from the teacher's committed ``id`` SS6 rate) and the flip floor.

    I = S(teacher) - S(bc_state) >= S(teacher) - 1, in points; the teacher's rate is P8 ``ppo``
    s0's ``id`` SS6 row of ``baselines.csv``.
    """
    out = []
    for b in inp.baselines:
        if (b["method"], b["list"], b["ss"]) == ("ppo", "id", "SS6"):
            rate = _x(b["success_rate"])
            out.append(
                f"I is ceiling-limited: the teacher (P8 ppo s0) is at {100.0 * rate:.1f} % "
                f"({b['n_success']}/{b['n_episodes']}) at id SS6, so I >= "
                f"{100.0 * (rate - 1.0):+.1f} points."
            )
    out.append(f"SS6 flip floor: {p8_adapter.FLIP_FLOOR_SS6.sentence}.")
    return out


def _draw_bars(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    title: str,
    ylabel: str,
    caption: Sequence[str] = (),
) -> None:
    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ok = [r for r in rows if r.get("point") is not None]
        if ok:
            xs = list(range(len(ok)))
            ax.bar(xs, [r["point"] for r in ok], color=["#0072B2", "#E69F00", "#009E73"][: len(ok)])
            _errorbar(ax, xs, ok, fmt="none", ecolor="#000000", lw=1.0)
            ax.set_xticks(xs, [r["series"] for r in ok], fontsize=7)
            ax.axhline(0.0, color="#000000", lw=0.6)
        else:
            ax.text(0.5, 0.5, "not available (missing or withheld)", ha="center", va="center",
                    transform=ax.transAxes)  # fmt: skip
            ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(True, axis="y", lw=0.3, alpha=0.6)
        _footnote(fig, [*caption, *_notes(rows)])
        fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.3)
        _save(fig, path)


def _draw_sweep(rows: Sequence[Mapping[str, Any]], path: Path, *, title: str, xlabel: str,
                log: bool) -> None:  # fmt: skip
    plt = _plt()
    panels = list(dict.fromkeys(r["panel"] for r in rows)) or ["(no rows)"]
    colours = {
        "no ensembling": "#0072B2",
        "ensembling": "#D55E00",
        "Ta = 8, no ensembling": "#CC79A7",
    }
    width = max(3.4 * len(panels), 5.0)  # inches; a single panel needs room for its labels
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.8), sharey=True,
                                 squeeze=False)  # fmt: skip
        for ax, panel in zip(axes[0], panels, strict=True):
            if panel.endswith("SS6"):
                ax.set_facecolor(_SHADE)
            mine = [r for r in rows if r["panel"] == panel]
            for series in dict.fromkeys(r["series"] for r in mine):
                s = sorted(
                    (r for r in mine if r["series"] == series and r.get("point") is not None),
                    key=lambda r: int(r["x"]),
                )
                if s:
                    _errorbar(ax, [int(r["x"]) for r in s], s, color=colours.get(series, "#444"),
                              marker="o", ms=4, lw=1.2, label=series)  # fmt: skip
            if log:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(panel + (" (outside training)" if panel.endswith("SS6") else ""))
            ax.grid(True, lw=0.3, alpha=0.6)
        axes[0, 0].set_ylabel("success, % (IQM over 5 seeds, 95 % CI)")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(title, fontsize=9, wrap=True)
        _footnote(fig, _notes(rows))
        fig.subplots_adjust(left=0.75 / width, right=0.98, top=0.82, bottom=0.2, wspace=0.08)
        _save(fig, path)


def _draw_h2(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import numpy as np

    plt = _plt()
    groups = (
        ("reference", "T_mix reference (P8 rows)", "#000000"),
        ("bc_mix", "bc_mix (3 seeds pooled)", "#009E73"),
        ("dp_mix", "dp_mix, Ta = 8, K = 10 (3 seeds pooled)", "#CC79A7"),
    )
    with plt.rc_context(_rc()):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True)
        finite = [_x(r["x"]) for r in rows if r["kind"] == "touchdown"]
        x_max = max(math.ceil(max(finite) + 0.1), 1.0) if finite else 1.0
        bins = np.arange(0.0, x_max + 0.05, 0.1)
        notes = []
        for key, label, colour in groups:
            mine = [r for r in rows if r["series"].split(":")[0] == key and r["note"] != "withheld"]
            t = np.sort([_x(r["x"]) for r in mine if r["kind"] == "touchdown"])
            if not mine:
                notes.append(f"{key}: not plotted")
                continue
            notes.append(f"{key}: {t.size}/{len(mine)} touched down")
            if t.size:
                y = np.arange(1, t.size + 1) / t.size
                top.step(np.concatenate([[0.0], t]), np.concatenate([[0.0], y]), where="post",
                         color=colour, lw=1.4, label=label)  # fmt: skip
                h, _ = np.histogram(t, bins=bins)
                bottom.stairs(h / len(mine), bins, color=colour, lw=1.3, label=label)
        top.set_ylabel("ECDF over touched-down episodes")
        bottom.set_ylabel("fraction of all episodes per 0.1 s")
        bottom.set_xlabel("time to touchdown, s (model scale; ×5 full scale)")
        top.legend(fontsize=7, frameon=False)
        for ax in (top, bottom):
            ax.grid(True, lw=0.3, alpha=0.6)
        top.set_title("H2: touchdown-time distributions at id SS4 (simulated)", fontsize=9)
        _footnote(fig, ["; ".join(notes) + ".", *_notes(rows)])
        fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.14, hspace=0.08)
        _save(fig, path)


# --------------------------------------------------------------------------- entry


def make_figures(root: Path = sc.OUT_ROOT) -> dict[str, Path]:
    """Draw every figure from the committed CSVs under ``root``; return name -> PNG path."""
    inp = rp.load_inputs(root)
    out_dir = root / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]]] = {
        "success_vs_seastate": success_table(inp),
        "h4_gaps": h4_table(inp),
        "h3_ta_curve": _sweep_table(inp, "dp_Ta"),
        "h2_touchdown_times": h2_table(root, inp),
        "dp_k_sweep": _sweep_table(inp, "dp_K"),
    }
    paths = {}
    for name in FIGURES:
        _write_table(out_dir / f"{name}.csv", _PLOT_COLUMNS, tables[name])
        paths[name] = out_dir / f"{name}.png"
    _draw_success(tables["success_vs_seastate"], paths["success_vs_seastate"])
    _draw_bars(
        tables["h4_gaps"],
        paths["h4_gaps"],
        "H4 at id SS6: perception gap P, imitation gap I, D = P - I (points, 95 % CI)",
        "points of success (simulated)",
        h4_caption(inp),
    )
    _draw_sweep(
        tables["h3_ta_curve"],
        paths["h3_ta_curve"],
        title="H3 curve (unscored): dp success vs executed chunk length Ta (same checkpoints)",
        xlabel="Ta (control steps of 33.3 ms model scale)",
        log=False,
    )
    _draw_h2(tables["h2_touchdown_times"], paths["h2_touchdown_times"])
    _draw_sweep(
        tables["dp_k_sweep"],
        paths["dp_k_sweep"],
        title="dp DDIM steps K at id SS5 (K = 1, 2 schedule-limited)",
        xlabel="K (DDIM denoising steps)",
        log=True,
    )
    return paths
