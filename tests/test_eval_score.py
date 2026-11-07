"""Phase 8 scoring (``ila.eval.score``, ``ila.eval.above_teacher``) on synthetic results trees.

Every student row is synthetic (``tests/eval_support.py``): nothing is flown and no student row of
a frozen list is read. Frozen-list *keys* and committed Phase 4/5 manifests are read (allowed,
P3-D1 §4 / §8). Speeds m/s, times s, model scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import rld.eval.metrics as p8_metrics

import eval_support as es
from ila import p8_adapter
from ila.eval import above_teacher as at
from ila.eval import score as sc
from ila.eval.metrics import mcnemar_exact_p

N = 10
OUTPUTS = (
    "summary.csv",
    "baselines.csv",
    "aggregate.csv",
    "contrasts.csv",
    "hypotheses.csv",
    "hypotheses.json",
    "above_teacher.csv",
    "missing.csv",
    "h2_touchdown_times.csv",
    "score_info.json",
)


def _small(job: object) -> bool:
    """Core bc / bc_state / modular on id, plus static for bc (seconds to score)."""
    return (
        getattr(job, "tag", "") == "core"
        and getattr(job, "arm", "") in ("bc", "bc_state", "modular")
        and getattr(job, "list_name", "") in ("id", "static")
    )


def _skip(job: object) -> bool:
    return getattr(job, "arm", "") == "modular" and getattr(job, "seed", -1) == 3


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> tuple[sc.ScoreInputs, Path]:
    root = tmp_path_factory.mktemp("e08")
    inp = es.synthetic_tree(root, n=N, matrix=es.reduced_matrix(_small), skip=_skip)
    sc.run_score(inp)
    return inp, root


def test_fmt_is_fixed() -> None:
    assert sc.fmt(None) == ""
    assert sc.fmt(True) == "True"
    assert sc.fmt(3) == "3"
    assert sc.fmt(0.1 + 0.2) == "0.3"
    assert sc.fmt(1 / 3) == "0.333333333333"
    assert sc.fmt(math.nan) == "nan"
    assert sc.fmt(-0.0) == "0"


def test_cell_metrics_is_p8s_own() -> None:
    assert p8_adapter.cell_metrics is p8_metrics.cell_metrics
    assert p8_adapter.CELL_METRIC_COLUMNS is p8_metrics.CELL_METRIC_COLUMNS
    assert set(p8_adapter.CELL_METRIC_COLUMNS) <= set(sc.SUMMARY_COLUMNS)


def test_no_duplicate_summary_columns(small: tuple[sc.ScoreInputs, Path]) -> None:
    """M8: ``n_success`` (six counts and P8's columns) is written once."""
    _, root = small
    for cols, name in ((sc.SUMMARY_COLUMNS, "summary.csv"), (sc.BASELINE_COLUMNS, "baselines.csv")):
        assert len(cols) == len(set(cols)), name
        header = (root / name).read_text(encoding="utf-8").splitlines()[0].split(",")
        assert len(header) == len(set(header)) and header.count("n_success") == 1, name


def test_score_is_byte_deterministic(small: tuple[sc.ScoreInputs, Path], tmp_path: Path) -> None:
    inp, root = small
    sc.run_score(inp, tmp_path)
    for name in OUTPUTS:
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes(), name


def test_summary_is_p8_cell_metrics(small: tuple[sc.ScoreInputs, Path]) -> None:
    inp, root = small
    index = sc.load_cells(root, inp.matrix, episodes_per_cell=N)
    rows = {(r["arm"], r["seed"], r["list"], r["ss"]): r for r in sc.read_csv(root / "summary.csv")}
    for key, cell in index.cells.items():
        if cell.rows is None:
            continue
        m = p8_adapter.cell_metrics(cell.rows)
        r = rows[(key.group.arm, str(key.seed), key.group.list_name, key.group.ss)]
        assert r["n_success"] == str(m.n_success)
        assert r["success_wilson_lo"] == sc.fmt(m.success_wilson_lo)
        assert r["rel_vz_normal_p95_m_s"] == sc.fmt(m.rel_vz_normal_p95_m_s)
        assert sum(int(r[f"n_{o}"]) for o in sc.OUTCOMES) == N


def test_missing_cells_are_listed_never_skipped(small: tuple[sc.ScoreInputs, Path]) -> None:
    _, root = small
    missing = sc.read_csv(root / "missing.csv")
    assert missing and {(r["arm"], r["seed"]) for r in missing} == {("modular", "3")}
    assert all(r["reason"].startswith("not yet flown") for r in missing)
    summary = [r for r in sc.read_csv(root / "summary.csv") if r["status"] == "missing"]
    assert len(summary) == len(missing)
    agg = [r for r in sc.read_csv(root / "aggregate.csv") if r["arm"] == "modular"]
    assert agg and all(r["status"].startswith("incomplete: seeds 3 missing") for r in agg)
    assert all(r["success_iqm"] == "" for r in agg)
    # A contrast needing the incomplete arm says so instead of disappearing.
    con = [r for r in sc.read_csv(root / "contrasts.csv") if "modular - ppo" in r["name"]]
    assert con and all(r["status"].startswith("missing") for r in con)
    hyp = {r["hypothesis"]: r for r in sc.read_csv(root / "hypotheses.csv")}
    assert hyp["H1"]["status"].startswith("not scored: missing")
    assert hyp["H5(a)"]["status"].startswith("not scored in Phase 8 (Phase 9)")


def test_aggregate_iqm_and_losses(small: tuple[sc.ScoreInputs, Path]) -> None:
    _, root = small
    agg = {(r["arm"], r["list"], r["ss"]): r for r in sc.read_csv(root / "aggregate.csv")}
    r = agg[("bc", "id", "SS6")]
    assert r["status"] == "complete" and r["n_seeds"] == "5"
    total = 5 * N
    for o in sc.LOSS_CLASSES:
        assert float(r[f"per_1000_{o}"]) == pytest.approx(1000.0 * int(r[f"n_{o}"]) / total)
    rates = [float(x) for x in r["success_per_seed"].split("|")]
    assert float(r["success_seed_min"]) == min(rates)
    assert float(r["success_iqm_lo"]) <= float(r["success_iqm"]) <= float(r["success_iqm_hi"])
    assert r["outside_training"] == "True"
    assert agg[("bc", "id", "SS5")]["outside_training"] == "False"


def test_baselines_cover_every_cell(small: tuple[sc.ScoreInputs, Path]) -> None:
    inp, root = small
    rows = sc.read_csv(root / "baselines.csv")
    cells = {(ln, ss) for ln, cs in inp.list_cells.items() for ss in cs}
    for m in sc.ALWAYS_PRINTED:
        assert {(r["list"], r["ss"]) for r in rows if r["method"] == m} == cells, m


def test_tmix_reference_is_balanced(small: tuple[sc.ScoreInputs, Path]) -> None:
    inp, _ = small
    ref = sc.tmix_reference(inp.listed("id", "SS5"), inp.baselines)
    assert len(ref) == N
    assert abs(sum(p == "ppo" for p, _ in ref) - N / 2) <= 1
    assert all(r["method"] == p for p, r in ref)


def test_ss6_contrasts_carry_the_flip_floor(small: tuple[sc.ScoreInputs, Path]) -> None:
    _, root = small
    rows = sc.read_csv(root / "contrasts.csv")
    assert any(r["ss"] == "SS6" for r in rows)
    for r in rows:
        want = p8_adapter.FLIP_FLOOR_SS6.sentence if r["ss"] == "SS6" else ""
        assert r["flip_floor"] == want, r["name"]
    for r in sc.read_csv(root / "hypotheses.csv"):
        if r["cell"].endswith("SS6"):
            assert r["flip_floor"] == p8_adapter.FLIP_FLOOR_SS6.sentence


def test_above_teacher_scan_and_audit(small: tuple[sc.ScoreInputs, Path]) -> None:
    inp, root = small
    rows = {r["audit_file"]: r for r in sc.read_csv(root / "above_teacher.csv")}
    # Synthetic teacher fails id SS6 episode 3 (of 10); bc's synthetic pattern lands it.
    key = "above_teacher/core__bc__id__SS6.json"
    assert key in rows
    r = rows[key]
    assert r["teacher"] == "T_ppo" and r["teacher_k"] == "9" and r["teacher_n"] == str(N)
    assert float(r["max_margin"]) > 0.0
    audits = at.run_audits(inp)
    assert len(audits) == len(rows)
    a = next(x for x in audits if x["label"] == "core bc id SS6")
    gained = a["outcome_shift"]["teacher_failure_to_student_success"]
    assert gained and all(
        g["index"] == 3 and g["teacher_outcome"] == "hard_landing" for g in gained
    )
    assert a["thresholds_success_yaml"] == {
        "rel_vertical_velocity_max_m_s": 0.5,
        "lateral_offset_max_m": 0.1,
        "relative_tilt_max_deg": 15.0,
        "contact_dwell_min_s": 0.5,
    }
    assert (root / key).is_file() and (root / "above_teacher" / "audit.md").is_file()
    assert a["teacher_rows"]["metrics"]["closing_speed_normal_m_s"]["n_over_threshold"] == 1
    # Near test: success criteria only (world-z printed, not tested; dwell reported, not tested).
    assert not a["student"]["metrics"]["closing_speed_world_z_m_s"]["in_near_test"]
    assert a["student"]["metrics"]["closing_speed_normal_m_s"]["in_near_test"]
    assert not a["student"]["success_dwell"]["in_near_test"]
    # Per-seed recount and exact McNemar from the discordant counts.
    flagged = {int(s.split(":")[0][1:]) for s in r["seeds_above"].split("|") if s}
    assert {p["seed"] for p in a["per_seed_recount"] if p["flagged_above_teacher"]} == flagged
    for p in a["per_seed_recount"]:
        assert p["teacher"]["successes"] == 9
        b, c = (
            p["discordant_teacher_failure_to_student_success"],
            p["discordant_teacher_success_to_student_failure"],
        )
        assert p["mcnemar_exact_p_two_sided"] == mcnemar_exact_p(b, c)
        assert p["student"]["successes"] - p["teacher"]["successes"] == b - c
    assert a["student"]["n_grace_used_successes"] == 0  # synthetic rows: one contact event
    assert "| grace-used |" in (root / "above_teacher" / "audit.md").read_text(encoding="utf-8")
    # Re-running is byte-identical, and the committed review record is never deleted.
    reviewed = root / "above_teacher" / at.REVIEWED_JSON
    reviewed.write_text('{"released": []}', encoding="utf-8")
    before = (root / "above_teacher" / "audit.md").read_bytes()
    at.run_audits(inp)
    assert (root / "above_teacher" / "audit.md").read_bytes() == before
    assert reviewed.read_text(encoding="utf-8") == '{"released": []}'
    reviewed.unlink()


def _row(index: int, outcome: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "index": index,
        "outcome": outcome,
        "touchdown_contact": "True",
        "tunnelled": "False",
        "detectors_disagree": "False",
        "contact_events": "1",
        "rel_vz_normal_m_s": -0.2,
        "rel_vz_world_z_m_s": -0.2,
        "lateral_offset_m": 0.01,
        "rel_tilt_deg": 3.0,
    }
    return {**base, **kw}


def test_recount_known_answer() -> None:
    """Synthetic rows with a known stricter recount (speeds m/s, lengths m, model scale)."""
    limits = {
        "rel_vertical_velocity_max_m_s": 0.5,
        "lateral_offset_max_m": 0.1,
        "relative_tilt_max_deg": 15.0,
        "contact_dwell_min_s": 0.5,
    }
    rows = [
        _row(0, "success"),
        _row(1, "success", tunnelled="True"),
        _row(2, "success", detectors_disagree="True", tunnelled="True"),
        _row(3, "success", rel_tilt_deg=14.5),  # margin 3.3 % -> near
        _row(4, "success", rel_vz_world_z_m_s=-0.49),  # world z only: not a criterion
        _row(5, "success", lateral_offset_m=0.0951, contact_events="2"),  # near; grace-used
        _row(6, "hard_landing", rel_vz_normal_m_s=-0.6, tunnelled="True", contact_events="3"),
        _row(7, "timeout", touchdown_contact="False", contact_events="0"),
    ]
    assert at.recount(rows, limits) == {
        "n_episodes": 8,
        "grace_used_successes": 1,  # successes only: the hard landing's 3 events do not count
        "successes": 6,
        "tunnelled_successes": 2,
        "disagreement_successes": 1,
        "successes_minus_tunnelled_or_disagreement": 4,
        "near_threshold_successes": 2,
        "successes_minus_tunnelled_disagreement_or_near": 2,
    }


def test_labelled_frames_from_committed_manifests() -> None:
    frames, sources = sc.load_labelled_frames()
    assert set(frames) == {"bc_dagger", "bc_matched", "bc_dart", "bc"}
    assert all(len(v) == 5 for v in frames.values())
    assert frames["bc_dagger"][0] == 123_216  # results/phase5/dagger_iterations.csv, s0 it 5
    assert "results/phase5/dagger_iterations.csv" in sources


def test_corrupt_rows_file_is_refused(tmp_path: Path) -> None:
    m = es.reduced_matrix(lambda j: j.tag == "core" and j.arm == "bc" and j.list_name == "static")
    inp = es.synthetic_tree(tmp_path, n=N, matrix=m)
    job = m.jobs[0]
    data = bytearray((tmp_path / job.rel_path).read_bytes())
    data[-5] ^= 0xFF
    (tmp_path / job.rel_path).write_bytes(bytes(data))
    with pytest.raises(sc.ScoreInputError, match="SHA-256"):
        sc.load_cells(tmp_path, inp.matrix, episodes_per_cell=N)


@pytest.mark.slow
def test_full_matrix_scores_every_hypothesis(tmp_path: Path) -> None:
    """The frozen matrix, every job present (synthetic): H1-H4 scored, H5 deferred."""
    inp = es.synthetic_tree(tmp_path, n=N)
    info = sc.run_score(inp)
    assert info["complete"] and info["cells_missing"] == 0
    hyp = {r["hypothesis"]: r for r in sc.read_csv(tmp_path / "hypotheses.csv")}
    for h in ("H1", "H1-SS6", "H1-DART-SS5", "H1-DART-SS6", "H2", "H3", "H4"):
        assert hyp[h]["status"] == "scored", h
        assert hyp[h]["verdict"] in ("supported", "not supported", "inconclusive")
        assert hyp[h]["inputs"], h
    names = {r["name"] for r in sc.read_csv(tmp_path / "contrasts.csv")}
    for want in (
        "H1 post hoc IQM",
        "H4 P",
        "H4 I",
        "H4 P' (unscored, tag +clock)",
        "H4 modular vs best end-to-end (unscored)",
        "bc_dart - bc_matched (SS6, unscored)",
        "H2 secondary: drop(bc) - drop(dp) (unscored)",
        "teacher check: dp_pid - pid_feedforward (id SS5)",
        "teacher check: dp_minus_last_action - ppo (id SS6)",
        "teacher check: modular - ppo (unseen_heading SS6)",
    ):
        assert want in names, want
    shared = [
        r
        for r in sc.read_csv(tmp_path / "summary.csv")
        if r["tag"] == "dp_Ta" and r["Ta"] == "8" and r["ensemble"] == "False"
    ]
    assert shared and all(r["status"] == "shared" and r["rows_from"] for r in shared)
