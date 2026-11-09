"""``results/results.md`` rendering (``ila.eval.report``) on synthetic scored trees.

Synthetic rows only (``tests/eval_support.py``); nothing is flown. Checks: byte-identical
re-render; the teacher and the three P8 baselines beside every student; SS6 rows carry P8's flip
floor; missing cells reported; above-teacher cells withheld until reviewed; simulation wording.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

import eval_support as es
from ila import p8_adapter
from ila.eval import hypotheses as hy
from ila.eval import report as rp
from ila.eval import score as sc

N = 10
FORBIDDEN = (
    "real flight",
    "flight-tested",
    "real camera",
    "real images",
    "sim-to-real",
    "at sea",
    "sea trials",
    "state of the art",
    "guarantee",
)


def _keep(job: object) -> bool:
    return (
        getattr(job, "tag", "") == "core"
        and getattr(job, "arm", "") in ("bc", "bc_state", "modular")
        and getattr(job, "list_name", "") in ("id", "static", "unseen_heading")
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e08")
    inp = es.synthetic_tree(
        root,
        n=N,
        matrix=es.reduced_matrix(_keep),
        skip=lambda j: j.arm == "modular" and j.seed == 3,
    )
    sc.run_score(inp)
    return root


def _section(text: str, title: str) -> str:
    start = text.index(title)
    nxt = text.find("\n## ", start + 1)
    return text[start : nxt if nxt != -1 else len(text)]


def test_rerender_is_byte_identical(scored: Path, tmp_path: Path) -> None:
    a = rp.write_report(scored, tmp_path / "a.md")
    b = rp.write_report(scored, tmp_path / "b.md")
    assert a == b
    assert (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()
    # No clock: the only timestamps are the committed run_info's (synthetic "t0" here).
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", a)


def test_baselines_beside_every_student(scored: Path) -> None:
    text = rp.render(rp.load_inputs(scored))
    core = _section(text, "## 2. Success per list")
    tables = core.split("### ")[1:]
    assert tables
    for t in tables:
        rows = [ln for ln in t.splitlines() if ln.startswith("| ")][1:]
        names = [ln.split(" | ")[0] for ln in rows]
        students = [n for n in names if "from P8" in n]
        assert students, t[:40]
        for m in sc.ALWAYS_PRINTED:
            assert any(rp._BASELINE_LABEL[m] in n for n in names), (m, t[:40])
        assert names.index(next(n for n in names if "teacher `ppo` s0" in n)) > names.index(
            students[-1]
        )
    _tagged_tables_print_baselines(text)


def _tagged_tables_print_baselines(text: str) -> None:
    """Every tagged table (7a-7f) with a student prints the four; 7d adds the mix references.

    7f (the latency arm) is present only when ``latency_arm`` rows were scored.
    """
    tagged = _section(text, "## 7. Tagged sections").split("### ")[1:]
    heads = [s[:2] for s in tagged]
    assert heads in (["7a", "7b", "7c", "7d", "7e"], ["7a", "7b", "7c", "7d", "7e", "7f"])
    for sub in tagged:
        if "losses/1 000" not in sub:
            continue  # no student in this subsection (reduced matrix)
        for m in sc.ALWAYS_PRINTED:
            assert rp._BASELINE_LABEL[m] in sub, (m, sub[:20])
    if "losses/1 000" in tagged[3]:
        assert rp._BASELINE_LABEL["gated"] in tagged[3]
        assert rp._BASELINE_LABEL[sc.TMIX] in tagged[3]


def test_learned_and_baseline_formats(scored: Path) -> None:
    text = rp.render(rp.load_inputs(scored))
    assert re.search(
        r"\*\*\d+\.\d\*\* \[\d+\.\d, \d+\.\d\]; seeds \d+\.\d–\d+\.\d; losses/1 000:", text
    )
    assert re.search(r"\d+\.\d \[\d+\.\d, \d+\.\d\] \d+/10; losses/10:", text)


def test_ss6_rows_carry_the_flip_floor(scored: Path) -> None:
    inp = rp.load_inputs(scored)
    text = rp.render(inp)
    six = _section(text, "## 6. Unscored contrasts")
    ss6 = [ln for ln in six.splitlines() if ln.startswith("| ") and " SS6 |" in ln]
    assert ss6
    for ln in ss6:
        assert p8_adapter.FLIP_FLOOR_SS6.sentence in ln
    hyp = _section(text, "## 5. Hypotheses")
    for block in hyp.split("### ")[1:]:
        if "SS6`" in block.splitlines()[0]:
            assert p8_adapter.FLIP_FLOOR_SS6.sentence in block


def test_verdict_line_suppresses_redundant_notes() -> None:
    """M9: the note is printed only where it adds a reason."""
    assert rp.verdict_line("supported", "supported") == "**Verdict: supported**."
    assert rp.verdict_line("inconclusive", "inconclusive (CI excludes 0; below +5)") == (
        "**Verdict: inconclusive** (CI excludes 0; below +5)."
    )
    assert rp.verdict_line("not supported", "not supported (CI includes 0)") == (
        "**Verdict: not supported** (CI includes 0)."
    )
    assert rp.verdict_line("not supported", "odd note") == "**Verdict: not supported** — odd note."
    # A note that already ends a sentence is not given a second period.
    assert rp.verdict_line("not supported", "not supported (x). More.") == (
        "**Verdict: not supported** (x). More."
    )


def test_hypotheses_print_no_redundant_verdict(scored: Path) -> None:
    hyp = _section(rp.render(rp.load_inputs(scored)), "## 5. Hypotheses")
    for verdict in ("supported", "not supported", "inconclusive"):
        assert f"**Verdict: {verdict}** — {verdict}" not in hyp


def test_section6_splits_tagged_inputs() -> None:
    """M5: contrasts with an ablation / +clock input never sit in the core subsection."""
    assert rp.contrast_tags("core bc id SS5; core bc_matched id SS5") == ["core"]
    assert rp.contrast_tags("ablation dp_pid id SS4 Ta8 K10 ens0; P8 committed ppo id SS4") == [
        "ablation",
        "P8 rows",
    ]
    assert rp.contrast_tags("core bc_state id SS6; ablation bc_plus_clock id SS6") == [
        "core",
        "ablation",
        "+clock",
    ]
    assert not rp.is_tagged("dp_Ta dp id SS6 Ta1 K10 ens0; dp_Ta dp id SS6 Ta8 K10 ens0")
    assert rp.is_tagged("core bc id SS4; ablation bc_mix id SS4")


@pytest.mark.slow
def test_full_matrix_section6_and_section7_n(tmp_path: Path) -> None:
    """M5 and M6 on the full synthetic matrix: 6a core only, 6b tagged; section 7 prints N."""
    inp = es.synthetic_tree(tmp_path, n=N)
    sc.run_score(inp)
    text = rp.render(rp.load_inputs(tmp_path))
    six = _section(text, "## 6. Unscored contrasts")
    a, b = six.split("### 6b.")
    rows_a = [ln for ln in a.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    rows_b = [ln for ln in b.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    assert rows_a and rows_b
    assert not any("`ablation`" in ln for ln in rows_a)
    assert all("`ablation`" in ln or "`latency_arm`" in ln for ln in rows_b)
    assert any(ln.startswith("| H5(b) context: bc_dagger latency arm") for ln in rows_b)
    assert "`latency_arm`; never mixed into 6a)" in b.splitlines()[0]
    assert not any("`latency_arm`" in ln for ln in rows_a)
    assert any(ln.startswith("| teacher check: dp_pid") for ln in rows_b)
    assert any("`+clock`" in ln for ln in rows_b)
    assert any(" 5 / 5 × 10 |" in ln for ln in rows_a)
    seven = _section(text, "## 7. Tagged sections")
    learned = [ln for ln in seven.splitlines() if "losses/1 000" in ln]
    assert learned and all("; N = " in ln for ln in learned)
    abl = seven.split("### 7d.")[1].split("### 7e.")[0]
    assert "; N = 30 (3 seeds × 10)" in abl
    assert "; N = 50 (5 seeds × 10)" in seven.split("### 7a.")[1]
    core = _section(text, "## 2. Success per list")
    # 5-seed core cells print no N; the bc_dart seeds 1-4 sensitivity row (P4-D5 item 2) is
    # 4 seeds, so its losses/1 000 are over 4 x N and it prints N.
    sens = f"| `bc_dart` {rp.S0_MARK} seeds 1–4 only"
    rows = [ln for ln in core.splitlines() if ln.startswith("| ")]
    assert any(ln.startswith(sens) for ln in rows)
    for ln in rows:
        if ln.startswith(sens):
            cells = ln.strip(" |").split(" | ")[1:]
            assert cells and all(
                c.startswith("withheld") or c.endswith("; N = 40 (4 seeds × 10)") for c in cells
            ), ln
        else:
            assert "; N = " not in ln


def test_missing_cells_are_reported(scored: Path) -> None:
    text = rp.render(rp.load_inputs(scored))
    assert "**PARTIAL MATRIX:**" in text
    assert "missing (4/5 seeds)" in text  # modular seed 3 skipped
    miss = _section(text, "## 9. Missing cells")
    assert "| core | `modular` |" in miss
    assert "results/e08/missing.csv" in miss


def test_above_teacher_cells_are_withheld_until_released(scored: Path, tmp_path: Path) -> None:
    inp = rp.load_inputs(scored)
    assert "core bc id SS6" in inp.withheld
    text = rp.render(inp)
    assert "withheld (above teacher; under review)" in text
    assert "under review (withheld)" in _section(text, "## 8. Students above")
    # Release one cell: its number appears and it leaves the withheld set.
    review = scored / "above_teacher" / "reviewed.json"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(
        json.dumps(
            {
                "released": [
                    {
                        "label": "core bc id SS6",
                        "reviewed_by": "results-skeptic",
                        "date": "2026-11-06",
                        "finding": "synthetic",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        inp2 = rp.load_inputs(scored)
        assert "core bc id SS6" not in inp2.withheld
        text2 = rp.render(inp2)
        assert "released by results-skeptic" in text2
        # Marked where printed, with the finding as a footnote under the success table.
        core = _section(text2, "## 2. Success per list")
        assert " ‡ |" in core
        assert (
            "‡ `core bc id SS6` is above its own teacher in at least one seed; released by " in core
        )
        assert "synthetic" in core
    finally:
        review.unlink()


def test_simulation_wording_and_no_forbidden_claims(scored: Path) -> None:
    text = rp.render(rp.load_inputs(scored))
    assert "Simulation only" in text and "TinyRenderer" in text and "λ = 1/25" in text
    assert "No multiplicity correction is applied" in text
    low = text.lower()
    for word in FORBIDDEN:
        assert word not in low, word


def _latency_keep(job: object) -> bool:
    tag = getattr(job, "tag", "")
    return tag == "latency_arm" or (
        tag == "core"
        and getattr(job, "arm", "") in ("bc_dagger", "dp")
        and getattr(job, "list_name", "") in ("id", "static")
    )


def test_latency_arm_renders_h5_and_section_7f(tmp_path: Path) -> None:
    lat_dir = es.write_latency_results(tmp_path / "latency")
    root = tmp_path / "e08"
    inp = es.synthetic_tree(root, n=N, matrix=es.reduced_matrix(_latency_keep), latency_dir=lat_dir)
    sc.run_score(inp)
    text = rp.write_report(root, tmp_path / "results.md")
    assert text.startswith("# Results — Phases 8–9: evaluation matrix, latency arm")
    assert rp.write_report(root, tmp_path / "again.md") == text  # byte-identical re-render
    prov = _section(text, "## 1. Provenance")
    assert "`latency_arm` (Phase 9) flew with `bc_dagger` d = 2, `dp` d = 3 control steps" in prov
    # M6: the latency stages' commits beside the commits that flew rows.
    assert (
        "| ila commit(s) of the Phase 9 latency stages (`results/latency/run_info.json`, rows' "
        "`ila_commit`) | `beef002` (closed_loop.csv rows, latency, latency.csv rows; clean "
        "tree); `c0ffee1` (export; dirty only under `results/`) |"
    ) in prov
    assert "copied there by the scorer from `results/latency/run_info.json`" in prov
    hyp = _section(text, "## 5. Hypotheses")
    h5a = hyp.split("### H5(a)")[1].split("### H5(b)")[0]
    assert "**Verdict: supported**." in h5a
    assert "21.50 ms (no CI: a p99 of timed calls); bound p99 < 33.3 ms" in h5a
    assert (
        "bound p99 < 33.3 ms, the model-scale 30 Hz control period (166.7 ms full scale; P3-D1 "
        "§6); the p99 is wall clock on this machine (RTX A4000, WSL2) and is not scaled."
    ) in h5a
    assert "ort_trt_fp32 (unscored)" in h5a and "no embedded target" in h5a
    h5b = hyp.split("### H5(b)")[1]
    assert "**Verdict: supported** (non-inferior: CI lower bound >= -3 points)." in h5b
    assert "non-inferiority margin: CI lower bound ≥ -3 points" in h5b
    assert "`latency_arm dp id SS5 Ta8 K10 ens0`, `core dp id SS5 Ta8 K10 ens0`" in h5b
    assert "`results/latency/h5a.csv`" in hyp
    assert "`results/latency/closed_loop.csv`" in hyp.split("Sources:")[-1]
    # The dp context rows in 6b carry the measured dp flip rate next to P8's flip floor.
    six_b = _section(text, "## 6. Unscored contrasts").split("### 6b.")[1]
    dp6 = next(
        ln for ln in six_b.splitlines()
        if ln.startswith("| H5(b) context: dp latency arm - synchronous (SS6, unscored) |")
    )  # fmt: skip
    assert f"{hy.DP_FLIP_LABEL}: 2/10 episodes" in dp6
    assert dp6.index(hy.DP_FLIP_LABEL) < dp6.index("SS6 flip floor")
    six = _section(text, "## 6. Unscored contrasts")
    a, b = six.split("### 6b.")
    assert "H5(b) context" not in a  # tagged inputs never in 6a
    assert "`latency_arm`, `core`" in b and "H5(b) context: dp latency arm" in b
    seven = _section(text, "## 7. Tagged sections")
    f = seven.split("### 7f.")[1]
    assert "latency arm d = 2" in f and "latency arm d = 3" in f
    assert "synchronous (core, beside it)" in f and "; N = 50 (5 seeds × 10)" in f
    for m in sc.ALWAYS_PRINTED:
        assert f.count(rp._BASELINE_LABEL[m]) == 1 + 3, m  # success row + SS4-SS6 outcome rows
    assert (
        "| arm, side | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |" in f
    )
    assert "| `dp` latency d = 3 | SS5 | 50 |" in f and "| `dp` sync | SS5 | 50 |" in f
    # The synthetic SS6 students beat the synthetic teacher: latency cells are withheld too.
    assert "| `dp` latency d = 3 | SS6 † | withheld |" in f
    assert "latency_arm dp id SS6 Ta8 K10 ens0" in rp.load_inputs(root).withheld
    assert "PyBullet does not advance" in f and "RTX A4000, WSL2" in f
    assert (
        "Simulated delay of d control steps of sim time (33.3 ms model scale each, 166.7 ms at "
        "full scale), d from the batch-1 p99 wall clock measured on this machine (RTX A4000, "
        "WSL2; not scaled;"
    ) in f
    _tagged_tables_print_baselines(text)
    low = text.lower()
    for word in FORBIDDEN:
        assert word not in low, word


def test_parity_failure_renders_in_section5(tmp_path: Path) -> None:
    """S3, S4, M4, M5 in results.md from the not-supported synthetic latency results."""
    lat_dir = es.write_latency_results(
        tmp_path / "latency", verdict="not supported", parity_fails=True
    )
    root = tmp_path / "e08"
    inp = es.synthetic_tree(root, n=N, matrix=es.reduced_matrix(_latency_keep), latency_dir=lat_dir)
    sc.run_score(inp)
    text = rp.write_report(root, tmp_path / "results.md")
    assert rp.write_report(root, tmp_path / "again.md") == text
    hyp = _section(text, "## 5. Hypotheses")
    h5a = hyp.split("### H5(a)")[1].split("### H5(b)")[0]
    verdict = next(ln for ln in h5a.splitlines() if ln.startswith("- **Verdict"))
    assert verdict.startswith(
        "- **Verdict: not supported** (no scored provider passed both parities and was timed). "
        "dp's closed-loop parity failure reflects outcome sensitivity of this dp, not a provider "
        "error: torch_cuda_fp32 runs the identical Python sampler"
    )
    assert "all > 33.3 ms, the model-scale 30 Hz control period (166.7 ms full scale)" in verdict
    assert verdict.endswith("results/latency/closed_loop.csv, parity.csv).")  # one period
    assert not verdict.endswith("..")
    h5b = hyp.split("### H5(b)")[1]
    assert "- why d: d = 3 is the torch_cpu_fp32 (1 thread) fallback delay" in h5b
    assert "would have given d = ceil(52.6441 / 33.333) = 2" in h5b
    assert "those delays were not flown. The verdict is unchanged" in h5b
    assert "- zero-action start: the contrast includes the d = 3 zero-action steps" in h5b
    assert f"- {hy.DP_FLIP_LABEL}: 4/10–5/10 episodes (40.0–50.0 %)" in h5b
    assert "- delay: d = 3 control steps = 3 × 33.3 ms = 100.0 ms of model-scale sim time" in h5b
    six_b = _section(text, "## 6. Unscored contrasts").split("### 6b.")[1]
    bc = [ln for ln in six_b.splitlines() if ln.startswith("| H5(b) context: bc_dagger")]
    success_rows = [ln for ln in bc if "relative p95" not in ln]
    assert len(success_rows) == 3
    for ln in success_rows:
        assert "the cost of a two-control-step delay (ceil of a 53.33 ms p99 to two 33.3 ms" in ln
        assert "not of 53 ms of compute" in ln
        assert "a feedback channel the student never saw in training" in ln
        assert "no teacher-under-delay reference exists to attribute the loss" in ln
    low = text.lower()
    for word in FORBIDDEN:
        assert word not in low, word


@pytest.mark.slow
def test_full_matrix_report_sections(tmp_path: Path) -> None:
    inp = es.synthetic_tree(tmp_path, n=N)
    sc.run_score(inp)
    text = rp.write_report(tmp_path, tmp_path / "results.md")
    for title in (
        "## 1. Provenance",
        "## 2. Success per list",
        "## 3. Outcome breakdown",
        "## 4. Secondary metrics",
        "## 5. Hypotheses",
        "## 6. Unscored contrasts",
        "### 7a.",
        "### 7b.",
        "### 7c.",
        "### 7d.",
        "### 7e.",
        "### 7f.",
        "## 8. Students above",
        "## 9. Missing cells",
    ):
        assert title in text, title
    assert "K = 1 (schedule-limited)" in text
    _tagged_tables_print_baselines(text)
    assert all("losses/1 000" in s for s in _section(text, "## 7.").split("### ")[1:])
    assert "None: every expected cell has rows." in text
    assert rp.write_report(tmp_path, tmp_path / "again.md") == text


# --------------------------------------------------------------------------- bc_dart seeds 1-4

_POST_HOC = "post hoc: added after `select`, before any frozen-list read"


@pytest.fixture(scope="module")
def dart_scored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e08_dart")
    sc.run_score(es.dart_s0_tree(root))
    return root


def _tables(text: str) -> list[list[str]]:
    """Every markdown table (its lines) with the lines that follow it up to the next table."""
    out: list[list[str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("| ") and (i == 0 or not lines[i - 1].startswith("|")):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(lines[i:j] + lines[j : j + 2])
            i = j
        else:
            i += 1
    return out


def test_every_bc_dart_row_carries_the_s0_footnote(dart_scored: Path, tmp_path: Path) -> None:
    text = rp.write_report(dart_scored, tmp_path / "results.md")
    assert rp.write_report(dart_scored, tmp_path / "again.md") == text
    marked = 0
    for table in _tables(text):
        rows = [ln for ln in table if ln.startswith("| ")]
        dart_rows = [ln for ln in rows if "bc_dart" in ln.split(" | ")[0]]
        for ln in dart_rows:
            assert rp.S0_MARK in ln.split(" | ")[0], ln
        if dart_rows:
            marked += 1
            assert table[-1] == rp.S0_NOTE, table[0]
    assert marked >= 4  # §1 provenance, §2 id, §3, §4, §6a
    for name in ("H1-DART-SS5", "H1-DART-SS6"):
        block = _section(text, f"### H1 DART secondary ({name[-3:]}): `bc_dart` {rp.S0_MARK}")
        assert rp.S0_NOTE in block.split("\n### ")[0]
    assert "P4-D5 item 2" in rp.S0_NOTE and "σ = 0.10" in rp.S0_NOTE and _POST_HOC in rp.S0_NOTE


def test_section2_prints_seeds_1_4_beside_never_instead(dart_scored: Path, tmp_path: Path) -> None:
    text = rp.write_report(dart_scored, tmp_path / "results.md")
    core = _section(text, "## 2. Success per list")
    rows = [ln for ln in core.splitlines() if ln.startswith("| `bc_dart`")]
    assert len(rows) == 2
    five, four = rows
    assert "seeds 1–4" not in five and four.startswith(
        f"| `bc_dart` {rp.S0_MARK} seeds 1–4 only — sensitivity ({_POST_HOC}); beside the "
        "5-seed row above, never substituted for it |"
    )
    # Known answers (tests/eval_support.py DART_FAILS): IQM 0.80 on seeds 0-4, 0.75 on 1-4.
    assert five.count("**80.0**") == 4 and "N = " not in five
    assert four.count("**75.0**") == 4 and four.count("N = 40 (4 seeds × 10)") == 4
    assert core.index(five) < core.index(four)
    assert f"{rp.S0_MARK} seeds 1–4 row, SS6 cells: SS6 flip floor: " in core
    assert p8_adapter.FLIP_FLOOR_SS6.sentence in core
    assert "`results/e08/bc_dart_s1_4.csv` (`kind` iqm)" in core


def test_section5_and_h4_print_seeds_1_4_beside(dart_scored: Path, tmp_path: Path) -> None:
    text = rp.write_report(dart_scored, tmp_path / "results.md")
    hyp = _section(text, "## 5. Hypotheses")
    for ss in ("SS5", "SS6"):
        block = hyp.split(f"### H1 DART secondary ({ss})")[1].split("\n### ")[0]
        verdict = block.index("**Verdict:")
        bullet = [ln for ln in block.splitlines() if ln.startswith(f"- {rp.S0_MARK} Sensitivity")]
        assert len(bullet) == 1, ss
        assert _POST_HOC in bullet[0] and "never substituted" in bullet[0]
        assert "no verdict is given on it" in bullet[0]
        assert "= -15.0 points [" in bullet[0] and "seeds `bc_dart` 4, `bc` 5" in bullet[0]
        assert "seeds 1–4 75.0 [" in bullet[0] and "five seeds as frozen 80.0 [" in bullet[0]
        assert (p8_adapter.FLIP_FLOOR_SS6.sentence in bullet[0]) == (ss == "SS6")
        # The scored 5-seed result comes first and is unchanged.
        assert "- Estimate: -12.0 points [" in block and verdict < block.index(bullet[0])
        assert "seeds `bc_dart` 5, `bc` 5." in block
    prov = _section(text, "## 1. Provenance")
    row = next(ln for ln in prov.splitlines() if ln.startswith("| H4 best end-to-end student"))
    assert row.startswith(f"| H4 best end-to-end student | `bc_dart` {rp.S0_MARK}: highest")
    assert "all five seeds, as frozen" in row and _POST_HOC in row
    assert "seeds 1–4 has 97.33 % mean select success" in row
    assert "picks bc (highest mean select success (97.73 %))" in row


def test_seeds_1_4_rows_are_withheld_with_their_cell(dart_scored: Path) -> None:
    """An above-teacher bc_dart cell withholds its 5-seed and seeds 1-4 numbers alike."""
    inp = rp.load_inputs(dart_scored)
    agg = next(r for r in inp.aggregate if (r["arm"], r["ss"]) == ("bc_dart", "SS5"))
    above = {c: agg[c] for c in sc.GROUP_COLUMNS}
    above.update(teacher="T_ppo", seeds_above="s1:0.01", iqm_above="False", audit_file="x.json")
    held = dataclasses.replace(inp, above=[above])
    text = rp.render(held)
    core = _section(text, "## 2. Success per list")
    rows = [ln for ln in core.splitlines() if ln.startswith("| `bc_dart`")]
    assert all(r.count("withheld (above teacher; under review)") == 1 for r in rows)
    block = _section(text, "### H1 DART secondary (SS5)").split("\n### ")[0]
    assert "**withheld**" in block and "Sensitivity" not in block


def test_report_without_dart_file_renders_as_before(dart_scored: Path) -> None:
    inp = rp.load_inputs(dart_scored)
    text = rp.render(dataclasses.replace(inp, dart_s1_4=()))
    assert "seeds 1–4" not in text.replace(rp.S0_NOTE, "")
