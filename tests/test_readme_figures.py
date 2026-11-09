"""README figures (``ila.viz.readme_figures``): every plotted value traces; bytes are stable.

The trace test re-reads each ``source_file`` named in a figure CSV with the standard library
(no project helpers), selects the row named by ``source_row`` and checks ``value``/``lo``/``hi``
= ``scale`` x the named cells, ``n`` = the product of the ``n_col`` cells and ``text`` = the
``text_col`` cell. Inputs are the committed ``results/`` files (no P8, no row files). The
withheld test copies those inputs, un-releases one above-teacher cell and checks that the cell
is not plotted anywhere and carries the reason.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ila import p8_adapter
from ila.viz import readme_figures as rf

REPO = p8_adapter.REPO_ROOT

#: Every committed input the README figures read (repo-relative).
INPUTS = (
    "results/e08/run_info.json",
    "results/e08/score_info.json",
    "results/e08/summary.csv",
    "results/e08/baselines.csv",
    "results/e08/aggregate.csv",
    "results/e08/contrasts.csv",
    "results/e08/hypotheses.csv",
    "results/e08/hypotheses.json",
    "results/e08/above_teacher.csv",
    "results/e08/above_teacher/reviewed.json",
    "results/e08/missing.csv",
    "results/e08/bc_dart_s1_4.csv",
    "results/e08/figures/h3_ta_curve.csv",
    "results/e08/figures/h4_gaps.csv",
    "results/latency/latency.csv",
    "results/latency/parity.csv",
    "results/latency/closed_loop.csv",
    "results/latency/delay.json",
    "results/latency/h5a.csv",
)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Path]]:
    return rf.make_readme_figures(REPO, tmp_path_factory.mktemp("readme_a"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(text: str) -> float:
    return math.nan if text.strip() in ("", "nan", "None") else float(text)


def _source_row(source_file: str, source_row: str) -> Mapping[str, Any]:
    """The one source row (CSV) or object (JSON) that ``source_row`` names."""
    path = REPO / source_file
    if path.suffix == ".csv":
        key = dict(kv.split("=", 1) for kv in source_row.split(";"))
        rows = [r for r in _read_csv(path) if all(r[k] == v for k, v in key.items())]
        assert len(rows) == 1, f"{source_file} [{source_row}] matches {len(rows)} rows"
        return rows[0]
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if source_file.endswith("hypotheses.json"):
        name = source_row.split("=", 1)[1]
        recs = [h for h in data["hypotheses"] if h.get("hypothesis") == name]
        assert len(recs) == 1
        return dict(recs[0])
    for part in filter(None, source_row.split(".")):
        data = data[part]
    return dict(data)


def test_every_plotted_value_equals_its_source_cell(rendered: dict[str, list[Path]]) -> None:
    assert set(rendered) == set(rf.FIGURES)
    checked = 0
    for name, paths in rendered.items():
        rows = _read_csv(paths[-1])
        assert rows, name
        assert list(rows[0]) == list(rf.COLUMNS), name
        for r in rows:
            assert r["source_file"] in INPUTS, (name, r["source_file"])
            if r["note"] == "missing (no source row)":
                assert r["value"] == "" and r["text"] == "", r
                continue
            src = _source_row(r["source_file"], r["source_row"])
            scale = float(r["scale"] or 1.0)
            for out, col in (("value", "value_col"), ("lo", "lo_col"), ("hi", "hi_col")):
                if not r[col]:
                    # A value parsed from a verbatim text cell is checked below (caveats[]).
                    assert r[out] == "" or r["text_col"] == "caveats[]", (name, r)
                    continue
                want = scale * _num(str(src[r[col]]))
                if r[out] == "":
                    # Not plotted: the source is empty, or the reason is stated.
                    assert math.isnan(want) or r["note"].startswith(("withheld", "missing")), r
                    continue
                assert float(r[out]) == pytest.approx(want, rel=1e-10, abs=1e-10), (name, r)
                checked += 1
            if r["n_col"]:
                want_n = math.prod(int(src[c]) for c in r["n_col"].split("*"))
                assert int(r["n"]) == want_n, (name, r)
            if r["text_col"] == "caveats[]":
                assert r["text"] in [str(c) for c in src["caveats"]], r
                assert float(r["value"]) == float(r["text"].split("Ta = ")[1].split(" ")[0])
            elif r["text_col"]:
                assert r["text"] == str(src[r["text_col"]]), (name, r)
    assert checked > 300


def test_tables_cover_the_requested_cells(rendered: dict[str, list[Path]]) -> None:
    tab = {n: _read_csv(p[-1]) for n, p in rendered.items()}
    baselines = {"ppo", "pid_feedforward", "pid_track_descend", "oracle_gated"}
    # F1: both panels carry the teacher and the three P8 baselines; students IQM, P8 rows rate.
    for panel, arms in (
        ("(a) behaviour-cloning family", {"bc", "bc_matched", "bc_dart", "bc_dagger"}),
        ("(b) other routes", {"bc_state", "modular", "dp"}),
    ):
        mine = [r for r in tab["success_id"] if r["panel"] == panel]
        assert {r["series"] for r in mine} == baselines | arms
        for r in mine:
            want = "success_rate" if r["series"] in baselines else "success_iqm"
            assert r["value_col"] == want and r["x"] in ("SS3", "SS4", "SS5", "SS6"), r
    # F2: 11 rows x 8 conditions.
    cells = {(r["series"], r["x"]) for r in tab["conditions"] if r["panel"] == "conditions"}
    assert len(cells) == 11 * 8
    # F3: six classes per arm and sea state; they sum to N.
    for ss in ("id SS5", "id SS6"):
        for arm in baselines | set(rf.STUDENTS):
            mine = [r for r in tab["outcomes"] if r["panel"] == ss and r["series"] == arm]
            assert sorted(r["x"] for r in mine) == sorted(
                ("success", "off_pad", "hard_landing", "bounce", "crash", "timeout")
            )
            assert sum(int(r["value"]) for r in mine) == int(mine[0]["n"]), (ss, arm)
    # F4: every scored hypothesis; H2 in seconds (scale 1), the rest in points (scale 100).
    hyp = {r["series"]: r for r in tab["hypotheses"]}
    for h in ("H1", "H1-SS6", "H1-DART-SS5", "H1-DART-SS6", "H3", "H4", "H5(b)"):
        assert hyp[h]["unit"] == "points" and float(hyp[h]["scale"]) == 100.0
    assert hyp["H2"]["unit"].startswith("s") and float(hyp["H2"]["scale"]) == 1.0
    assert hyp["H5(a)"]["value"] == "" and hyp["H5(a)"]["text"] == "not supported"
    # Every SS6 contrast carries P8's flip floor.
    for h in ("H1-SS6", "H1-DART-SS6", "H3", "H4"):
        assert f"{h} flip floor" in hyp, h
    # F7: both arms, every provider, the control period.
    lat = tab["latency"]
    assert {r["panel"] for r in lat} == {"bc_dagger", "dp", "meta"}
    assert any(r["series"] == "control period" and r["value"] for r in lat)


def test_withheld_cell_is_not_plotted(tmp_path: Path) -> None:
    for rel in INPUTS:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
    reviewed = tmp_path / "results/e08/above_teacher/reviewed.json"
    data = json.loads(reviewed.read_text(encoding="utf-8"))
    label = "core bc_state id SS6"
    data["released"] = [r for r in data["released"] if r["label"] != label]
    reviewed.write_text(json.dumps(data), encoding="utf-8")
    tabs = rf.tables(tmp_path)
    for name in ("success_id", "conditions", "gaps"):
        held = [
            r
            for r in tabs[name]
            if r["series"] == "bc_state"
            and r.get("source_row") == "tag=core;arm=bc_state;list=id;ss=SS6"
        ]
        assert held, name
        assert all("value" not in r and r["note"].startswith("withheld") for r in held), name
    # Drawing copes with the gap (a dash, no crash).
    out = tmp_path / "out"
    out.mkdir()
    for name in ("success_id", "conditions", "gaps", "outcomes"):
        rf._DRAW[name](tabs[name], rf.LIGHT, out / f"{name}.png")
        assert (out / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_is_byte_identical(
    rendered: dict[str, list[Path]], tmp_path_factory: pytest.TempPathFactory
) -> None:
    again = rf.make_readme_figures(REPO, tmp_path_factory.mktemp("readme_b"))
    for name, paths in rendered.items():
        assert [p.name for p in paths] == [f"{name}.png", f"{name}_dark.png", f"{name}.csv"]
        for a, b in zip(paths, again[name], strict=True):
            assert a.read_bytes() == b.read_bytes(), f"{a.name} differs between renders"


def test_committed_figures_match_a_fresh_render(rendered: dict[str, list[Path]]) -> None:
    committed = REPO / rf.OUT_DIR
    if not committed.is_dir():
        pytest.skip("docs/figures not generated yet")
    for paths in rendered.values():
        for p in paths:
            mine = committed / p.name
            assert mine.is_file(), f"{mine} missing: run scripts/make_readme_figures.py"
            assert mine.read_bytes() == p.read_bytes(), f"{mine} is stale: re-run the script"
