"""``results/results.md`` rendering (``ila.eval.report``) on synthetic scored trees.

Synthetic rows only (``tests/eval_support.py``); nothing is flown. Checks: byte-identical
re-render; the teacher and the three P8 baselines beside every student; SS6 rows carry P8's flip
floor; missing cells reported; above-teacher cells withheld until reviewed; simulation wording.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import eval_support as es
from ila import p8_adapter
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
    """Every tagged table (7a-7e) with a student prints the four; 7d adds the mix references."""
    tagged = _section(text, "## 7. Tagged sections").split("### ")[1:]
    assert [s[:2] for s in tagged] == ["7a", "7b", "7c", "7d", "7e"]
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
    assert all("`ablation`" in ln for ln in rows_b)
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
    assert "; N = " not in core


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
        "## 8. Students above",
        "## 9. Missing cells",
    ):
        assert title in text, title
    assert "K = 1 (schedule-limited)" in text
    _tagged_tables_print_baselines(text)
    assert all("losses/1 000" in s for s in _section(text, "## 7.").split("### ")[1:])
    assert "None: every expected cell has rows." in text
    assert rp.write_report(tmp_path, tmp_path / "again.md") == text
