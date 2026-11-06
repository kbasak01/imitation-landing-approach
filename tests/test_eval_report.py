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
        assert "released by results-skeptic" in rp.render(inp2)
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
    assert "None: every expected cell has rows." in text
    assert rp.write_report(tmp_path, tmp_path / "again.md") == text
