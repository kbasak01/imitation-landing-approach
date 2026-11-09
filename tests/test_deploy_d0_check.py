"""The d = 0 check on the real arms (ila.deploy.d0_check): build, comparison and record plumbing.

No flight and no checkpoint: an injected flyer returns synthetic runner rows, and the
"selection rows" are written beside them in a temporary results root. The real flight is
``make bench-d0-check``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from deploy_support import tiny_settings
from ila import p8_adapter
from ila.deploy import closed_loop as cl
from ila.deploy import d0_check as d0
from ila.deploy import latency_arm as la
from ila.deploy.settings import BenchSettings
from ila.eval import runner

GIT = {"ila_commit": "c" * 40, "dirty": False, "dirty_paths": [], "dirty_outside_results": False}


def _rows(arm: str, n: int = 3) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        r: dict[str, Any] = dict.fromkeys(runner.VISION_EPISODE_COLUMNS, "")
        r.update(
            regime="select",
            ss=4,
            index=i,
            episode_seed=1000 + i,
            outcome="success",
            arm=arm,
            student_seed=0,
            latency_steps=0,
            closing_speed_normal_m_s=float("nan") if i == 1 else 0.25,
        )
        out.append(r)
    return out


def _flyer(selection: dict[str, Path], rows: dict[str, list[dict[str, Any]]]) -> d0.Flyer:
    def fly(
        arm: str,
        settings: BenchSettings,
        episodes: Sequence[p8_adapter.ListedEpisode],
        workers: int,
    ) -> d0.D0Flight:
        del settings, episodes, workers
        return d0.D0Flight(rows[arm], "a" * 64, selection[arm], {"delay": 0})

    return fly


def _setup(
    tmp_path: Path, selection_text: dict[str, str] | None = None
) -> tuple[BenchSettings, dict[str, Path], dict[str, list[dict[str, Any]]]]:
    s = tiny_settings(tmp_path)
    rows = {arm: _rows(arm) for arm in ("bc_dagger", "dp")}
    sel: dict[str, Path] = {}
    for arm, r in rows.items():
        p = tmp_path / "selection" / arm / "rows.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        # As ila.train.select.write_rows_csv writes selection rows (CRLF from csv).
        p.write_text((selection_text or {}).get(arm, cl.rows_text(r)), "utf-8")
        sel[arm] = p
    return s, sel, rows


def _run(
    s: BenchSettings,
    sel: dict[str, Path],
    rows: dict[str, list[dict[str, Any]]],
    arms: Sequence[str] = ("bc_dagger", "dp"),
) -> dict[str, Any]:
    return d0.run_d0_check(
        s, git=GIT, arms=arms, workers=2, episodes=[], fly=_flyer(sel, rows), log=lambda _m: None
    )


def test_d0_spec_is_the_latency_build_at_d0_with_selection_tags(tmp_path: Path) -> None:
    s = tiny_settings(tmp_path)
    for arm, kw in (
        ("dp", {"n_action_steps": 8, "num_inference_steps": 10}),
        ("bc_dagger", {}),
    ):
        spec, tags, build = d0.d0_spec(
            arm, Path("/ck.pt"), run_seed=0, sha256="f" * 64, teacher="T_ppo", settings=s
        )
        part = spec.build
        assert isinstance(part, functools.partial)
        assert part.func is la.delayed_checkpoint_policy
        assert part.args == ("/ck.pt", 0) and part.keywords == kw
        assert spec.run_seed == 0 and not spec.privileged
        # Selection's tags (ila.train.select.fly_checkpoint with inference=None).
        assert tags == runner.RowTags(
            arm=arm, teacher="T_ppo", student_seed=0, ckpt_sha256="f" * 64
        )
        assert tags.Ta is None and tags.K is None and tags.ensemble is None
        assert tags.latency_steps == 0 and build["delay"] == 0
    with pytest.raises(ValueError):
        d0.d0_spec("bc", Path("/x"), run_seed=0, sha256="", teacher="", settings=s)


def test_identical_rows_record_and_verify(tmp_path: Path) -> None:
    s, sel, rows = _setup(tmp_path)
    rec = _run(s, sel, rows)
    assert rec["identical"] is True
    assert rec["ila_commit"] == GIT["ila_commit"] and rec["dirty"] is False
    for arm in ("bc_dagger", "dp"):
        e = rec["arms"][arm]
        assert e["identical"] is True and e["n"] == 3 and e["selection_n"] == 3
        assert e["rows_sha256"] == e["selection_rows_sha256"]
        assert e["outcome_flips"] == 0 and e["rows_text_differ"] == 0
        assert e["ila_commit"] == GIT["ila_commit"] and e["dirty"] is False
        path = s.results_dir / d0.D0_DIR / f"{arm}_rows.csv"
        assert path.read_bytes() == sel[arm].read_bytes()
        assert b"\r\n" in path.read_bytes()  # written as the reference flight writes it
    on_disk = json.loads((s.results_dir / d0.D0_FILE).read_text("utf-8"))
    assert on_disk == rec
    assert d0.verify_record(s) == []


def test_differing_outcome_is_not_identical_and_check_says_so(tmp_path: Path) -> None:
    s, sel, rows = _setup(tmp_path)
    changed = [dict(r) for r in rows["dp"]]
    changed[2]["outcome"] = "off_pad"
    sel["dp"].write_text(cl.rows_text(changed), "utf-8")
    rec = _run(s, sel, rows)
    assert rec["identical"] is False
    assert rec["arms"]["bc_dagger"]["identical"] is True
    e = rec["arms"]["dp"]
    assert e["identical"] is False and e["outcome_flips"] == 1 and e["rows_text_differ"] == 1
    assert e["rows_sha256"] != e["selection_rows_sha256"]
    bad = d0.verify_record(s)
    assert any("dp rows differ" in b for b in bad)
    assert not any("bc_dagger" in b for b in bad)


def test_bytes_not_text_lf_selection_file_is_not_identical(tmp_path: Path) -> None:
    # The 3b2ae85 trap (closed_loop.RECHECK_REASON): same rows, other line endings.
    lf = cl.rows_text(_rows("dp")).replace("\r\n", "\n")
    s, sel, rows = _setup(tmp_path, {"dp": lf})
    e = _run(s, sel, rows)["arms"]["dp"]
    assert e["identical"] is False and e["outcome_flips"] == 0 and e["rows_text_differ"] == 0


def test_partial_flight_keeps_the_other_arm_and_verify_catches_tampering(tmp_path: Path) -> None:
    s, sel, rows = _setup(tmp_path)
    assert d0.verify_record(s) == []  # no record yet: nothing to verify
    rec = _run(s, sel, rows, arms=("dp",))
    assert set(rec["arms"]) == {"dp"} and rec["identical"] is False  # bc_dagger not flown
    assert any("bc_dagger not flown" in b for b in d0.verify_record(s))
    rec = _run(s, sel, rows, arms=("bc_dagger",))
    assert list(rec["arms"]) == ["bc_dagger", "dp"] and rec["identical"] is True
    assert d0.verify_record(s) == []
    path = s.results_dir / d0.D0_DIR / "bc_dagger_rows.csv"
    path.write_bytes(path.read_bytes().replace(b"success", b"timeout", 1))
    bad = d0.verify_record(s)
    assert any("bc_dagger rows file differs from the recorded SHA-256" in b for b in bad)
    assert any("bc_dagger identical disagrees with the files' bytes" in b for b in bad)
    with pytest.raises(ValueError):
        _run(s, sel, rows, arms=("bc",))
