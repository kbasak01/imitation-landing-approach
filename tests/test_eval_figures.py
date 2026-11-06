"""Phase 8 figures (``ila.viz.figures``) and the GIF rule (``ila.viz.gifs``).

Figures are drawn from synthetic scored trees (``tests/eval_support.py``). The GIF rule is tested
on synthetic outcomes over the frozen ``id`` list's keys (keys only); the one re-flight test flies
``select`` episodes only, never a frozen list, and compares against committed/selection rows.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

import eval_support as es
from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import runner
from ila.eval import score as sc
from ila.train import select as train_select
from ila.viz import figures, gifs

N = 10
REPO = p8_adapter.REPO_ROOT


def _keep(job: object) -> bool:
    tag, arm = getattr(job, "tag", ""), getattr(job, "arm", "")
    return (tag == "core" and arm in ("bc", "dp") and getattr(job, "list_name", "") == "id") or (
        tag in ("dp_Ta", "dp_K") and getattr(job, "seed", -1) >= 0
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e08")
    inp = es.synthetic_tree(root, n=N, matrix=es.reduced_matrix(_keep))
    sc.run_score(inp)
    return root


def test_figures_exist_with_their_csvs_and_are_deterministic(scored: Path) -> None:
    paths = figures.make_figures(scored)
    assert set(paths) == set(figures.FIGURES)
    first = {n: p.read_bytes() for n, p in paths.items()}
    for name, p in paths.items():
        assert p.is_file() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
        assert (p.parent / f"{name}.csv").is_file(), name
    again = figures.make_figures(scored)
    for name, p in again.items():
        assert p.read_bytes() == first[name], f"{name}: PNG bytes differ between renders"


def test_success_plot_table_has_baselines_and_marks(scored: Path) -> None:
    figures.make_figures(scored)
    with (scored / "figures" / "success_vs_seastate.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    series = {r["series"] for r in rows}
    assert set(sc.ALWAYS_PRINTED) <= series and {"bc", "dp"} <= series
    assert any(r["note"] == "withheld" for r in rows)  # synthetic above-teacher SS6 cells
    assert all(r["point"] == "" for r in rows if r["note"] in ("withheld", "missing"))


def test_rule_md_states_the_rule_verbatim() -> None:
    text = (REPO / "results" / "e08" / "gifs" / "RULE.md").read_text(encoding="utf-8")
    assert gifs.rule_from_md(text) == gifs.RULE
    assert "hand-picked" in text and "simulation" in text.lower()


def _rows(
    listed: list[p8_adapter.ListedEpisode], special: Mapping[tuple[str, str, int], str]
) -> dict[str, list[dict[str, str]]]:
    """Every panel succeeds on id SS5-SS6 except ``special[(panel, ss, index)]``."""
    out: dict[str, list[dict[str, str]]] = {p: [] for p in gifs.PANELS}
    for e in listed:
        if e.ss not in gifs.CELLS:
            continue
        for p in gifs.PANELS:
            out[p].append(
                {
                    "regime": "id",
                    "ss": e.ss,
                    "index": str(e.index),
                    "episode_seed": str(e.episode_seed),
                    "outcome": special.get((p, e.ss, int(e.index)), "success"),
                }
            )
    return out


def test_gif_rule_picks_lowest_position_per_category() -> None:
    listed = list(ev_arms.list_episodes("id"))
    special = {
        ("dp", "SS5", 0): "bounce",  # (b) students split at the lowest position
        ("dp", "SS6", 5): "off_pad",
        ("dp", "SS5", 7): "off_pad",  # (c): lower list position than SS6 5
        ("teacher", "SS5", 2): "hard_landing",  # not SS6: not (d); and not (a)
        ("teacher", "SS6", 4): "hard_landing",  # (d)
        ("teacher", "SS6", 9): "timeout",
    }
    picks = gifs.select_episodes(listed, _rows(listed, special))
    got = {c: (p.ss, p.index) for c, p in picks.items() if p is not None}
    assert got == {
        "a_all_four_succeed": ("SS5", 1),
        "b_students_split": ("SS5", 0),
        "c_dp_off_pad": ("SS5", 7),
        "d_teacher_ss6_failure": ("SS6", 4),
    }
    pos = {p.position for p in picks.values() if p is not None}
    assert len(pos) == 4
    assert picks["a_all_four_succeed"] is not None
    assert listed[picks["a_all_four_succeed"].position].index == 1


def test_gif_rule_episodes_are_distinct_and_empty_categories_stay_empty() -> None:
    listed = list(ev_arms.list_episodes("id"))
    special = {("dp", "SS5", 0): "off_pad", ("dp", "SS5", 3): "off_pad"}
    picks = gifs.select_episodes(listed, _rows(listed, special))
    b, c = picks["b_students_split"], picks["c_dp_off_pad"]
    assert b is not None and c is not None
    assert (b.ss, b.index) == ("SS5", 0)
    assert (c.ss, c.index) == ("SS5", 3)  # SS5 0 already picked by (b)
    assert picks["d_teacher_ss6_failure"] is None  # the teacher never fails here


def test_gif_rule_refuses_a_missing_row() -> None:
    listed = list(ev_arms.list_episodes("id"))
    rows = _rows(listed, {})
    rows["bc"] = rows["bc"][1:]
    with pytest.raises(ValueError, match="no committed row"):
        gifs.select_episodes(listed, rows)


def test_compose_and_gif_are_deterministic(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    e = ev_arms.list_episodes("id")[0]
    results = [
        gifs.PanelResult(
            gifs.Panel(p, e),
            {},
            [rng.integers(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(3 + k)],
        )
        for k, p in enumerate(gifs.PANELS)
    ]
    frames = gifs.compose(results, [f"{p} · id SS5 ep 0 · success" for p in gifs.PANELS])
    assert len(frames) == 6  # the longest panel; shorter ones hold their last frame
    assert frames[0].shape == (96 * gifs.SCALE + 30, 4 * 96 * gifs.SCALE, 3)
    a = gifs.write_gif(frames, tmp_path / "a.gif")
    b = gifs.write_gif(frames, tmp_path / "b.gif")
    assert a == b and (tmp_path / "a.gif").read_bytes()[:6] == b"GIF89a"


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.checkpoints
def test_reflight_matches_committed_select_rows() -> None:
    """GIF re-flight path on ``select`` only, against committed or selection rows.

    The teacher vs the committed T_mix reference rows; ``bc`` s0 vs its selection rows; frames
    recorded, one per control step.
    """
    from ila.data.episodes import read_split

    select = read_split("select")
    ref_path = REPO / "results" / "phase7" / "h2" / "tmix_select_rows.csv"
    with ref_path.open(encoding="utf-8", newline="") as f:
        ref = next(r for r in csv.DictReader(f) if r["tmix_assigned"] == "ppo")
    e = select[int(ref["select_position"])]
    t = gifs.fly_panel(gifs.Panel("teacher", e))
    text = p8_adapter.episode_rows_csv(runner.p8_rows([t.row]))
    mine = next(csv.DictReader(text.splitlines()))
    for col in p8_adapter.EPISODE_COLUMNS:
        assert mine[col] == ref[col], col
    assert len(t.frames) == int(mine["steps"])

    sel = json.loads(
        (REPO / "results" / "phase4" / "runs" / "bc" / "s0" / "selection.json").read_text()
    )
    rows_file = Path(sel["run_dir"]) / "select" / sel["selected"]["rows_file"]
    if not rows_file.is_file():
        pytest.skip(f"selection rows not on this machine: {rows_file}")
    with rows_file.open(encoding="utf-8", newline="") as f:
        committed = next(csv.DictReader(f))
    e0 = next(
        x for x in select if int(x.index) == int(committed["index"]) and x.ss == committed["ss"]
    )
    panel = gifs.student_panel("bc", e0)
    r = gifs.fly_panel(panel)
    tmp = REPO / "artifacts" / "tmp_gif_reflight_rows.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Selection's own writer, so both sides share one text format.
        train_select.write_rows_csv(tmp, [{c: r.row[c] for c in runner.VISION_EPISODE_COLUMNS}])
        with tmp.open(encoding="utf-8", newline="") as f:
            mine = next(csv.DictReader(f))
    finally:
        tmp.unlink(missing_ok=True)
    for col in runner.VISION_EPISODE_COLUMNS:
        assert mine[col] == committed[col], col
    assert len(r.frames) == int(mine["steps"])
