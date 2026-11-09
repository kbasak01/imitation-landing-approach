"""Phase 9 closed-loop records: ``reference_equals_selection_rows`` is a byte comparison (N2).

The 3b2ae85 flight compared the reference rows text as written (``csv``: CRLF row endings) with
the selection rows file read back through newline translation (LF), so byte-identical files were
recorded ``false``. These tests pin the cause, the byte comparison, and the no-flight re-check
that corrects such a record. No PyBullet, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deploy_support import tiny_settings
from ila.data import episodes as data_episodes
from ila.deploy import closed_loop as cl
from ila.deploy import pipeline
from ila.deploy.export import sha256_file
from ila.eval import runner


def _quiet(msg: str) -> None:
    del msg


def _rows(n: int = 3, outcome: str = "success") -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        r: dict[str, Any] = {c: f"{c}{i}" for c in runner.VISION_EPISODE_COLUMNS}
        r.update(regime="id", ss=3, index=i, episode_seed=100 + i, outcome=outcome)
        rows.append(r)
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cl.rows_text(rows), "utf-8")  # as run_closed_loop and selection write them
    return path


def test_newline_translation_was_the_cause(tmp_path: Path) -> None:
    text = cl.rows_text(_rows())
    assert "\r\n" in text  # csv's default row ending
    a = _write(tmp_path / "ref.csv", _rows())
    b = _write(tmp_path / "sel.csv", _rows())
    assert a.read_bytes() == b.read_bytes()
    # The old comparison: written text vs read_text (newline-translated) -> False.
    assert text != b.read_text("utf-8")
    # The fixed ones: bytes, and text read without translation.
    assert cl.rows_files_identical(a, b)
    assert cl._read_exact(b) == text


def test_rows_files_identical_sees_one_byte(tmp_path: Path) -> None:
    a = _write(tmp_path / "ref.csv", _rows())
    b = _write(tmp_path / "sel.csv", _rows())
    b.write_bytes(b.read_bytes().replace(b"\r\n", b"\n", 1))
    assert not cl.rows_files_identical(a, b)
    c = _write(tmp_path / "flip.csv", _rows(outcome="off_pad"))
    assert not cl.rows_files_identical(a, c)


def _reference_record(tmp_path: Path, arm: str, sel: Path) -> tuple[Any, Path, dict[str, Any]]:
    s = tiny_settings(tmp_path)
    adir = s.results_dir / "closed_loop" / arm
    _write(adir / f"{s.reference}_rows.csv", _rows())
    rec = {
        "arm": arm,
        "provider": s.reference,
        "status": "reference",
        "n": 3,
        "flips": 0,
        "rows_text_differ": 0,
        "reference_equals_selection_rows": False,
        "selection_rows_file": str(sel),
        "selection_rows_sha256": sha256_file(sel),
        "selection_rows_flips": 0,
        "outcomes": {"success": 3},
        "workers": 24,
        "wall_s": 44.9,
        "git": {"dirty": False, "dirty_paths": [], "ila_commit": "3b2ae85"},
        "checkpoint_sha256": "b" * 64,
    }
    path = adir / f"{s.reference}.json"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")
    return s, path, rec


def test_recheck_rewrites_only_that_field_and_records_it(tmp_path: Path) -> None:
    sel = _write(tmp_path / "sel" / "rows.csv", _rows())
    s, path, before = _reference_record(tmp_path, "dp", sel)
    out = cl.recheck_reference_rows(s, arms=["dp"], selection_rows={"dp": sel}, log=_quiet)
    after = json.loads(path.read_text("utf-8"))
    assert out["dp"] == after
    assert after["reference_equals_selection_rows"] is True
    note = after.pop("reference_equals_selection_rows_recheck")
    assert note["previous"] is False
    assert note["fields_changed"] == ["reference_equals_selection_rows"]
    assert note["method"].startswith("bytes")
    assert note["selection_rows_sha256"] == sha256_file(sel)
    assert note["reference_rows_sha256"] == sha256_file(path.with_name(f"{s.reference}_rows.csv"))
    # Nothing else changed, the flight's git included.
    assert {k: v for k, v in after.items() if k != "reference_equals_selection_rows"} == {
        k: v for k, v in before.items() if k != "reference_equals_selection_rows"
    }
    # A second re-check keeps the first previous value.
    cl.recheck_reference_rows(s, arms=["dp"], selection_rows={"dp": sel}, log=_quiet)
    again = json.loads(path.read_text("utf-8"))
    assert again["reference_equals_selection_rows_recheck"]["previous"] is False
    assert again["reference_equals_selection_rows"] is True


def test_recheck_reports_a_real_difference(tmp_path: Path) -> None:
    sel = _write(tmp_path / "sel" / "rows.csv", _rows(outcome="off_pad"))
    s, path, _ = _reference_record(tmp_path, "dp", sel)
    cl.recheck_reference_rows(s, arms=["dp"], selection_rows={"dp": sel}, log=_quiet)
    assert json.loads(path.read_text("utf-8"))["reference_equals_selection_rows"] is False


def test_recheck_refuses_another_selection_file(tmp_path: Path) -> None:
    sel = _write(tmp_path / "sel" / "rows.csv", _rows())
    s, path, _ = _reference_record(tmp_path, "dp", sel)
    other = _write(tmp_path / "other.csv", _rows(n=2))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="selection_rows_sha256"):
        cl.recheck_reference_rows(s, arms=["dp"], selection_rows={"dp": other}, log=_quiet)
    assert path.read_bytes() == before


def test_recheck_refuses_a_provider_record(tmp_path: Path) -> None:
    sel = _write(tmp_path / "sel" / "rows.csv", _rows())
    s, path, rec = _reference_record(tmp_path, "dp", sel)
    path.write_text(json.dumps({**rec, "status": "flown"}), "utf-8")
    with pytest.raises(ValueError, match="not a reference record"):
        cl.recheck_reference_rows(s, arms=["dp"], selection_rows={"dp": sel}, log=_quiet)


def test_stage_recheck_rerenders_closed_loop_csv_and_check_stays_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sel = _write(tmp_path / "sel" / "rows.csv", _rows())
    s, _, _ = _reference_record(tmp_path, "dp", sel)
    _reference_record(tmp_path, "bc_dagger", sel)
    monkeypatch.setattr(cl, "_selection_rows_path", lambda _selection_json: sel)
    csv_path = s.results_dir / "closed_loop.csv"
    csv_path.write_text(cl.closed_loop_csv(s), "utf-8")
    assert ",False," in csv_path.read_text("utf-8")
    pipeline.stage_recheck_reference(s, allow_dirty=True, log=_quiet)
    text = csv_path.read_text("utf-8")
    assert pipeline.check(s) == []
    lines = text.splitlines()
    col = lines[0].split(",").index("reference_equals_selection_rows")
    assert [ln.split(",")[col] for ln in lines[1:]] == ["True", "True"]
    info = json.loads((s.results_dir / "run_info.json").read_text("utf-8"))
    assert info["stages"]["closed_loop_recheck"]["arms"]["dp"] == {
        "reference_equals_selection_rows": True,
        "previous": False,
    }


def test_reference_flight_records_byte_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The flight itself (stubbed): identical CRLF files must be recorded True.
    from types import SimpleNamespace

    sel = _write(tmp_path / "sel" / "rows.csv", _rows())
    s = tiny_settings(tmp_path)
    ckpt = SimpleNamespace(sha256="c" * 64)
    monkeypatch.setattr(cl, "load_arm", lambda *_a: SimpleNamespace(ckpt=ckpt))
    monkeypatch.setattr(cl, "_selection_rows", lambda _c: (sel, cl._read_exact(sel)))
    monkeypatch.setattr(cl, "fly_provider", lambda *_a, **_k: _rows())
    monkeypatch.setattr(data_episodes, "read_split", lambda _name: [])
    cl.run_closed_loop(s, arms=["dp"], providers=[], graph_dir=tmp_path, open_loop=None, log=_quiet)
    rec = json.loads((s.results_dir / "closed_loop" / "dp" / f"{s.reference}.json").read_text())
    assert rec["reference_equals_selection_rows"] is True
    assert rec["reference_rows_sha256"] == rec["selection_rows_sha256"] == sha256_file(sel)
    assert rec["selection_rows_flips"] == 0
