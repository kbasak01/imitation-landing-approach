"""H2 touchdown-time plots on ``select`` (plan Phase 7 item 3; ``ila.eval.h2_select``).

Synthetic rows only, except ``splits/select.parquet`` (keys, for the ``T_mix`` draw and the
row-order checks). Covers: W1 on cases of known value (and agreement with scipy and with the
demo audit's copy); the per-cell ``T_mix`` draw on ``select``; pooling over seeds, quantiles,
outcome counts and W1 in the summary; per-seed rows, the inter-mode gap fraction and the JSON's
``per_seed`` block on seeds in different regimes; missing-arm reporting; verification of a
student run's rows against its ``selection.json`` (accepts a faithful run, refuses tampered
rows); the reference file's checks; byte-identical re-rendering; and that neither the module
nor its script names a P8 frozen-list accessor. Nothing is flown.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from ila.data import audit
from ila.data import episodes as data_episodes
from ila.eval import h2_select as h2
from ila.eval import runner
from ila.eval.metrics import wasserstein_1
from ila.train import manifest as mf
from ila.train import select as train_select
from test_render_access import FROZEN_LIST_NAMES, _references

NAN = float("nan")


# --------------------------------------------------------------------------- W1


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0.0, 1.0], [2.0, 3.0], 2.0),  # a shift by 2
        ([0.0], [0.0, 1.0], 0.5),  # half the mass moves by 1
        ([1.0, 1.0, 3.0, 3.0], [1.0, 3.0, 3.0], 1.0 / 3.0),  # |1/2 - 1/3| over [1, 3)
        ([1.5, 2.5, 4.0], [1.5, 2.5, 4.0], 0.0),
    ],
)
def test_w1_known_values(a: list[float], b: list[float], expected: float) -> None:
    assert wasserstein_1(a, b) == pytest.approx(expected, abs=1e-15)
    assert wasserstein_1(b, a) == pytest.approx(expected, abs=1e-15)


def test_w1_matches_scipy_and_the_demo_audit() -> None:
    rng = np.random.default_rng(20261023)
    for _ in range(20):
        a = rng.gamma(2.0, 1.0, size=int(rng.integers(1, 60))).tolist()
        b = (rng.normal(3.0, 1.0, size=int(rng.integers(1, 60)))).tolist()
        assert wasserstein_1(a, b) == pytest.approx(wasserstein_distance(a, b), rel=1e-12)
        assert wasserstein_1(a, b) == audit.wasserstein_1(a, b)
        assert wasserstein_1(a, [x + 0.7 for x in a]) == pytest.approx(0.7, rel=1e-12)


def test_w1_empty_is_nan_and_non_finite_refused() -> None:
    assert math.isnan(wasserstein_1([], [1.0]))
    assert math.isnan(wasserstein_1([1.0], []))
    with pytest.raises(ValueError, match="finite"):
        wasserstein_1([1.0, NAN], [1.0])
    with pytest.raises(ValueError, match="finite"):
        wasserstein_1([1.0], [math.inf])


# --------------------------------------------------------------------------- T_mix draw


@pytest.fixture(scope="module")
def select_rows() -> list[Any]:
    return data_episodes.read_split(data_episodes.SELECT_NAME)


def test_tmix_assignment_on_select_is_balanced_per_cell(select_rows: list[Any]) -> None:
    assigned = h2.tmix_assignment(select_rows)
    per = Counter((r.ss, p) for r, p in zip(select_rows, assigned, strict=True))
    assert dict(per) == {(ss, p): 25 for ss in h2.CELLS for p in ("ppo", "gated")}
    assert h2.tmix_seed_from_config() == h2.TMIX_SEED == 20261023


# --------------------------------------------------------------------------- summary


def _rec(arm: str, seed: int | None, mode: str, pos: int, ss: str, t: float, outcome: str) -> Any:
    return h2.EpisodeRecord(arm, seed, mode, pos, ss, pos, 1000 + pos, outcome, t, "")


def _synthetic() -> tuple[list[h2.EpisodeRecord], list[h2.StudentStatus]]:
    """Reference SS4: ppo half at 1 s, gated half at 3 s; bc_mix s0 [1, timeout], s1 [3, 3]."""
    ref = [
        _rec("T_mix", None, "ppo", 0, "SS4", 1.0, "success"),
        _rec("T_mix", None, "ppo", 1, "SS4", 1.0, "success"),
        _rec("T_mix", None, "gated", 2, "SS4", 3.0, "success"),
        _rec("T_mix", None, "gated", 3, "SS4", 3.0, "bounce"),
    ]
    bc = [
        _rec("bc_mix", 0, "", 0, "SS4", 1.0, "success"),
        _rec("bc_mix", 0, "", 1, "SS4", NAN, "timeout"),
        _rec("bc_mix", 1, "", 0, "SS4", 3.0, "off_pad"),
        _rec("bc_mix", 1, "", 1, "SS4", 3.0, "success"),
    ]
    statuses = [
        h2.StudentStatus("bc_mix", 0, "present"),
        h2.StudentStatus("bc_mix", 1, "present"),
        h2.StudentStatus("bc_mix", 2, "missing", "no selection.json (manifest state 'running')"),
        *(h2.StudentStatus("dp_mix", s, "missing", "run directory absent") for s in (0, 1, 2)),
    ]
    return ref + bc, statuses


def _row(
    rows: list[dict[str, Any]], group: str, cell: str, level: str = "pooled", seeds: str = ""
) -> Any:
    found = [x for x in rows if (x["group"], x["cell"], x["level"]) == (group, cell, level)]
    if level == "seed":
        found = [x for x in found if x["seeds"] == seeds]
    (r,) = found
    return r


def test_summary_pools_seeds_and_reports_known_values() -> None:
    records, statuses = _synthetic()
    rows = h2.summarise(records, statuses)
    bc = _row(rows, "bc_mix", "SS4")
    assert bc["seeds"] == "0 1" and bc["status"] == "partial (seeds 2 missing)"
    assert (bc["n_episodes"], bc["n_touchdown"], bc["n_without_touchdown"]) == (4, 3, 1)
    assert bc["frac_without_touchdown"] == 0.25
    assert (bc["n_success"], bc["n_off_pad"], bc["n_timeout"], bc["n_crash"]) == (2, 1, 1, 0)
    # Pooled touched-down times [1, 3, 3]; type-7 quantiles; W1 to [1, 1, 3, 3] is 1/3.
    assert bc["p10_model_s"] == pytest.approx(1.4)
    assert bc["p50_model_s"] == 3.0 and bc["p90_model_s"] == 3.0
    assert bc["w1_to_tmix_ref_model_s"] == pytest.approx(1.0 / 3.0)
    # Per seed: s0 has one touchdown at 1 s -> W1 to the reference is 1.0.
    s0 = _row(rows, "bc_mix", "SS4", "seed", "0")
    assert s0["seeds"] == "0" and s0["n_touchdown"] == 1
    assert s0["w1_to_tmix_ref_model_s"] == pytest.approx(1.0)
    # The reference and its two modes.
    ref = _row(rows, "T_mix", "SS4")
    assert ref["n_episodes"] == 4 and ref["n_bounce"] == 1
    assert math.isnan(ref["w1_to_tmix_ref_model_s"])
    assert _row(rows, "T_mix:ppo", "SS4")["w1_to_tmix_ref_model_s"] == pytest.approx(1.0)
    assert _row(rows, "T_mix:gated", "SS4")["p50_model_s"] == 3.0
    # "all" pools the sea states; an empty cell has no quantiles.
    assert _row(rows, "bc_mix", "all")["n_episodes"] == 4
    assert _row(rows, "bc_mix", "SS3")["n_episodes"] == 0
    assert math.isnan(_row(rows, "bc_mix", "SS3")["p50_model_s"])


def test_per_seed_rows_separate_regimes_that_pooling_hides() -> None:
    """Two seeds in different regimes: s0 averages (all mass in the gap), s1 commits.

    Reference SS4: ppo half [1.0, 1.2], gated half [3.0, 3.4] -> type-7 p90 of the fast half
    1.18, p10 of the slow half 3.04: gap (1.18, 3.04). s0 lands at [2.0, 2.2] (gap 1.0); s1 at
    [1.1, 3.3] (gap 0.0). Pooled [1.1, 2.0, 2.2, 3.3]: gap 0.5, which describes neither seed.
    """
    ref = [
        _rec("T_mix", None, "ppo", 0, "SS4", 1.0, "success"),
        _rec("T_mix", None, "ppo", 1, "SS4", 1.2, "success"),
        _rec("T_mix", None, "gated", 2, "SS4", 3.0, "success"),
        _rec("T_mix", None, "gated", 3, "SS4", 3.4, "success"),
    ]
    bc = [
        _rec("bc_mix", 0, "", 0, "SS4", 2.0, "success"),
        _rec("bc_mix", 0, "", 1, "SS4", 2.2, "success"),
        _rec("bc_mix", 1, "", 0, "SS4", 1.1, "success"),
        _rec("bc_mix", 1, "", 1, "SS4", 3.3, "off_pad"),
    ]
    statuses = [
        h2.StudentStatus("bc_mix", 0, "present", selected_global_step=5000, select_success=2),
        h2.StudentStatus("bc_mix", 1, "present", selected_global_step=20000, select_success=1),
        *(h2.StudentStatus("dp_mix", s, "missing", "run directory absent") for s in (0, 1, 2)),
    ]
    records = ref + bc
    lo, hi = h2.mode_gap(records, "SS4")
    assert (lo, hi) == (pytest.approx(1.18), pytest.approx(3.04))
    assert math.isnan(h2.mode_gap(records, "SS3")[0])  # no reference in SS3: no gap
    rows = h2.summarise(records, statuses)
    s0 = _row(rows, "bc_mix", "SS4", "seed", "0")
    s1 = _row(rows, "bc_mix", "SS4", "seed", "1")
    pooled = _row(rows, "bc_mix", "SS4")
    assert s0["frac_td_in_ref_mode_gap"] == 1.0 and s1["frac_td_in_ref_mode_gap"] == 0.0
    assert pooled["frac_td_in_ref_mode_gap"] == 0.5
    # The reference's own tails reach into the gap: 1.2 > 1.18 and 3.0 < 3.04.
    assert _row(rows, "T_mix", "SS4")["frac_td_in_ref_mode_gap"] == 0.5
    assert s0["p50_model_s"] == pytest.approx(2.1) and s1["p50_model_s"] == pytest.approx(2.2)
    # W1 of s0 to [1.0, 1.2, 3.0, 3.4], integral of |F_a - F_b|: 0.05 + 0.4 + 0 + 0.4 + 0.1.
    assert s0["w1_to_tmix_ref_model_s"] == pytest.approx(0.95)
    block = h2.per_seed_block(rows, statuses)
    assert set(block) == set(h2.STUDENT_ARMS) and block["dp_mix"]["seeds"] == {}
    seeds = block["bc_mix"]["seeds"]
    assert list(seeds) == ["0", "1"]
    assert seeds["0"]["selected_global_step"] == 5000
    assert seeds["1"]["cells"]["SS4"]["outcomes"]["off_pad"] == 1
    assert seeds["0"]["cells"]["SS4"]["frac_td_in_ref_mode_gap"] == 1.0
    rng = block["bc_mix"]["range_across_seeds"]["SS4"]
    assert rng["frac_td_in_ref_mode_gap"] == [0.0, 1.0]
    assert rng["p50_model_s"] == [pytest.approx(2.1), pytest.approx(2.2)]
    assert all(math.isnan(v) for v in block["dp_mix"]["range_across_seeds"]["SS4"]["p50_model_s"])


def test_gap_fraction_is_strict_and_nan_without_a_gap() -> None:
    assert h2.gap_fraction([1.0, 1.5, 2.0], (1.0, 2.0)) == pytest.approx(1.0 / 3.0)
    assert math.isnan(h2.gap_fraction([], (1.0, 2.0)))
    assert math.isnan(h2.gap_fraction([1.5], (NAN, NAN)))
    overlap = [
        _rec("T_mix", None, "ppo", 0, "SS4", 1.0, "success"),
        _rec("T_mix", None, "ppo", 1, "SS4", 3.0, "success"),
        _rec("T_mix", None, "gated", 2, "SS4", 1.5, "success"),
        _rec("T_mix", None, "gated", 3, "SS4", 3.5, "success"),
    ]
    assert all(math.isnan(v) for v in h2.mode_gap(overlap, "SS4"))  # halves overlap: no gap


def test_missing_arm_is_reported_never_skipped() -> None:
    records, statuses = _synthetic()
    rows = h2.summarise(records, statuses)
    dp = [r for r in rows if r["group"] == "dp_mix"]
    assert [r["cell"] for r in dp] == [*h2.CELLS, h2.ALL_CELL]
    assert all(r["status"] == "missing (seeds 0 1 2)" and r["n_episodes"] == 0 for r in dp)
    assert {r["group"] for r in rows} == set(h2.GROUPS)


def test_load_student_reports_missing_runs(tmp_path: Path) -> None:
    status, recs = h2.load_student(tmp_path, "dp_mix", 1)
    assert (status.status, status.reason, recs) == ("missing", "run directory absent", [])
    run = tmp_path / "bc_mix" / "s2"
    run.mkdir(parents=True)
    status, _ = h2.load_student(tmp_path, "bc_mix", 2)
    assert status.reason == "no selection.json (no manifest)"
    mf.write_json_atomic(run / "manifest.json", {"state": "running"})
    status, _ = h2.load_student(tmp_path, "bc_mix", 2)
    assert status.reason == "no selection.json (manifest state 'running')"


# --------------------------------------------------------------------------- verification


def _fake_rows(episodes: list[Any], arm: str, seed: int, sha: str, k: int) -> list[dict[str, Any]]:
    rows = []
    for i, e in enumerate(episodes):
        timeout = (i + k) % 37 == 0
        row = dict.fromkeys(runner.VISION_EPISODE_COLUMNS, "")
        row.update(
            ss=e.ss,
            index=e.index,
            episode_seed=e.episode_seed,
            outcome="timeout" if timeout else ("success" if (i + k) % 5 else "off_pad"),
            closing_speed_normal_m_s=NAN if timeout else 0.2 + 0.001 * ((i * 7 + k) % 50),
            time_to_touchdown_s=NAN if timeout else 1.4 + 0.01 * ((i * 3 + k) % 90),
            steps=361 if timeout else 60 + (i % 9),
            tunnelled=False,
            detectors_disagree=False,
            arm=arm,
            student_seed=seed,
            ckpt_sha256=sha,
        )
        rows.append(row)
    return rows


def _fake_run(root: Path, arm: str, seed: int, episodes: list[Any]) -> Path:
    """A finished run with two candidates and a selection written by select's own code."""
    run = root / arm / f"s{seed}"
    (run / "select").mkdir(parents=True)
    results = []
    for k, step in enumerate((5000, 10000)):
        sha = f"{k:064x}"
        name = f"s{seed}_step{step:06d}_rows.csv"
        rows = _fake_rows(episodes, arm, seed, sha, k)
        train_select.write_rows_csv(run / "select" / name, rows)
        cand = train_select.Candidate(f"/x/{name}", 0, step, step, sha)
        results.append(train_select._result(cand, train_select.summarise_rows(rows), name))
    best = train_select.choose(results)
    split = data_episodes.SPLITS_DIR / "select.parquet"
    config = f"configs/train/ablations/{arm}.yaml"
    mf.write_json_atomic(
        run / "manifest.json",
        {
            "arm": arm,
            "run_seed": seed,
            "state": "done",
            "config": {
                "all_frozen": True,
                "files": [{"path": config, "sha256": mf.FROZEN_CONFIG_SHA256[config]}],
            },
        },
    )
    mf.write_json_atomic(
        run / "selection.json",
        {
            "arm": arm,
            "run_seed": seed,
            "list": {
                "name": "select",
                "file": split.name,
                "file_sha256": data_episodes.file_sha256(split),
                "episodes": len(episodes),
            },
            "weights": "ema",
            "inference_path": "cpu_fp32_single_sample",
            "inference": dict(h2.DP_INFERENCE) if arm == "dp_mix" else None,
            "tag": "ablation",
            "obs_variant": "default",
            "teacher": "T_mix",
            "candidates": [train_select._result_record(r) for r in results],
            "selected": train_select._result_record(best),
        },
    )
    return run


def test_verify_selection_accepts_a_faithful_run(tmp_path: Path, select_rows: list[Any]) -> None:
    _fake_run(tmp_path, "dp_mix", 0, select_rows)
    status, recs = h2.load_student(tmp_path, "dp_mix", 0)
    assert status.status == "present" and len(recs) == 150
    rec = mf.read_json(tmp_path / "dp_mix" / "s0" / "selection.json")
    assert status.ckpt_sha256 == rec["selected"]["candidate"]["sha256"]
    assert status.rows_file == f"select/{rec['selected']['rows_file']}"
    assert sum(not r.touched_down for r in recs) == rec["selected"]["outcomes"]["timeout"]
    assert [r.episode_seed for r in recs] == [e.episode_seed for e in select_rows]


@pytest.mark.parametrize("tamper", ["outcome", "ckpt", "order", "inference", "teacher"])
def test_verify_selection_refuses_tampering(
    tmp_path: Path, select_rows: list[Any], tamper: str
) -> None:
    run = _fake_run(tmp_path, "dp_mix", 1, select_rows)
    rec = mf.read_json(run / "selection.json")
    rows_path = run / "select" / rec["selected"]["rows_file"]
    with rows_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if tamper == "outcome":
        rows[3]["outcome"] = "crash" if rows[3]["outcome"] != "crash" else "success"
    elif tamper == "ckpt":
        rows[0]["ckpt_sha256"] = "f" * 64
    elif tamper == "order":
        rows[0], rows[1] = rows[1], rows[0]
    if tamper in ("outcome", "ckpt", "order"):
        train_select.write_rows_csv(rows_path, rows)
    else:
        rec[tamper] = (
            {"Ta": 1, "K": 10, "temporal_ensembling": False} if tamper == "inference" else "T_ppo"
        )
        mf.write_json_atomic(run / "selection.json", rec)
    with pytest.raises(h2.H2SourceError):
        h2.load_student(tmp_path, "dp_mix", 1)


def test_reference_file_checks(tmp_path: Path, select_rows: list[Any]) -> None:
    assigned = h2.tmix_assignment(select_rows)
    path = tmp_path / h2.REFERENCE_FILE
    rows = []
    for i, (e, p) in enumerate(zip(select_rows, assigned, strict=True)):
        r = dict.fromkeys(h2.REFERENCE_COLUMNS, "")
        r.update(select_position=i, tmix_assigned=p, method=p, ss=e.ss, index=e.index,
                 episode_seed=e.episode_seed, outcome="success",
                 time_to_touchdown_s=1.5 if p == "ppo" else 3.5)  # fmt: skip
        rows.append(r)

    def write(rs: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(h2.REFERENCE_COLUMNS), lineterminator="\n")
            w.writeheader()
            w.writerows(rs)

    write(rows)
    recs = h2.load_reference(path)
    assert Counter(r.tmix_assigned for r in recs) == {"ppo": 75, "gated": 75}
    flipped = [dict(r) for r in rows]
    other = "gated" if flipped[5]["method"] == "ppo" else "ppo"
    flipped[5]["method"] = flipped[5]["tmix_assigned"] = other
    write(flipped)
    with pytest.raises(h2.H2SourceError, match="assigned"):
        h2.load_reference(path)
    write(rows[:-1])
    with pytest.raises(h2.H2SourceError, match="149 rows"):
        h2.load_reference(path)


# --------------------------------------------------------------------------- outputs


def _inputs(out: Path) -> None:
    records, statuses = _synthetic()
    h2.write_episodes(out / h2.EPISODES_FILE, records)
    sources = {
        "reference": {"file": h2.REFERENCE_FILE, "sha256": "0" * 64, "tmix_seed": h2.TMIX_SEED},
        "run_root": "/runs",
        "students": [s.__dict__ for s in statuses],
        "missing": [f"{s.arm} s{s.seed}: {s.reason}" for s in statuses if s.status != "present"],
        "select": {"file": "select.parquet", "file_sha256": "1" * 64},
    }
    (out / h2.SOURCES_FILE).write_text(json.dumps(sources), encoding="utf-8")


def test_render_is_byte_identical_and_marks_partial(tmp_path: Path) -> None:
    outs = [tmp_path / "a", tmp_path / "b"]
    for out in outs:
        out.mkdir()
        _inputs(out)
        record = h2.render(out)
        assert record["partial"] is True and record["scored"] is False
        assert len(record["missing"]) == 4
    for name in (h2.SUMMARY_CSV, h2.SUMMARY_JSON, h2.FIGURE_FILE):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    h2.render(outs[0])  # re-render in place
    assert (outs[0] / h2.FIGURE_FILE).read_bytes() == (outs[1] / h2.FIGURE_FILE).read_bytes()
    with (outs[0] / h2.SUMMARY_CSV).open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == h2.SUMMARY_COLUMNS
    text = (outs[0] / h2.SUMMARY_JSON).read_text(encoding="utf-8")
    assert "descriptive" in text and "id SS4 in Phase 8" in text and "lambda = 1/25" in text
    assert "post hoc" in text and "pools seeds as P3-D1" in text
    rec = json.loads(text)
    assert list(rec["per_seed"]["bc_mix"]["seeds"]) == ["0", "1"]
    assert rec["per_seed"]["bc_mix"]["seeds"]["0"]["cells"]["SS4"]["n_touchdown"] == 1
    assert rec["per_seed"]["dp_mix"]["seeds"] == {}
    assert set(rec["reference"]["mode_gap_model_s"]) == {*h2.CELLS, h2.ALL_CELL}


def test_episode_table_round_trips(tmp_path: Path) -> None:
    records, _ = _synthetic()
    h2.write_episodes(tmp_path / "e.csv", records)
    back = h2.read_episodes(tmp_path / "e.csv")
    assert len(back) == len(records)
    for a, b in zip(records, back, strict=True):
        assert (a.arm, a.seed, a.tmix_assigned, a.outcome) == (
            b.arm,
            b.seed,
            b.tmix_assigned,
            b.outcome,
        )
        assert a.time_to_touchdown_s == b.time_to_touchdown_s or (
            math.isnan(a.time_to_touchdown_s) and math.isnan(b.time_to_touchdown_s)
        )


def test_no_frozen_list_accessor_is_named() -> None:
    repo = Path(h2.__file__).resolve().parents[3]
    for path in (Path(h2.__file__), repo / "scripts" / "h2_touchdown_plot.py"):
        assert _references(path.read_text(encoding="utf-8"), FROZEN_LIST_NAMES) == [], path
