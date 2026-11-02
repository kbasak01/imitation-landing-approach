"""Phase 6 ``dp`` jobs on a tiny trained checkpoint (CPU): smoke MSE, sweeps, diagnostics.

* :func:`ila.train.smoke.dp_sampled_mse`: one sample per frame with the flight's noise key,
  executed indices ``To-1 ... To+6`` against the label chunk.
* :mod:`ila.train.dp_sweeps`: the cell list from the frozen ``dp.yaml`` (15 distinct settings,
  ``Ta = 1`` with ensembling aliased), the selection-row comparison, and an end-to-end sweep of
  one ``select`` episode: the reference cell reproduces the stored selection rows, a tampered
  selection row is caught, a re-run reuses every cell.
* :mod:`ila.train.dp_diagnostics`: boundary/within classification tables, spread keys and
  global-RNG independence, the open-loop noise keys, and an end-to-end run whose recorded
  closed-loop flight reproduces the selection rows and P8's ``action_jerk_mean``.

The checkpoint is ``dp`` trained 2 CPU steps on a synthetic store (purpose ``test``); it crashes
within about 45 control steps, which keeps each flight short. Flights use one ``select`` episode
(never a frozen list).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ila.data import episodes as data_episodes
from ila.data.store import DemoStore
from ila.policies import registry
from ila.policies.dp import chunking
from ila.policies.dp.policy import DPStudent
from ila.train import dp_diagnostics as dd
from ila.train import dp_sweeps as sw
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import sha256_file
from ila.train.dataset import BCWindowDataset, DPChunkDataset, FrameTable
from ila.train.select import SELECT_SPLIT, fly_checkpoint, write_rows_csv
from ila.train.smoke import dp_sampled_mse

RUN_SEED = 3


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _chunk_data(root: Path, n: int = 2) -> DPChunkDataset:
    with DemoStore(root / "demos" / "ppo_2000.h5") as store:
        names = list(store.episodes())[:n]
        table = FrameTable.from_store(store, names)
    return DPChunkDataset(table, 2, 16)


@pytest.fixture(scope="module")
def dp_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A selected ``dp`` run: 2 training steps, its selection flown on one ``select`` episode."""
    import test_train_loop as tl

    tmp = tmp_path_factory.mktemp("dp_run")
    root = tmp / "data"
    tl.write_store(root / "demos" / "ppo_2000.h5", [tl.synthetic_record(j, 6, j) for j in range(4)])
    run = tmp / "runs" / "dp" / f"s{RUN_SEED}"
    spec = tl._spec(
        tl._cfg("dp", batch_size=2),
        run,
        root,
        steps_override=2,
        checkpoint_every_override=2,
        heldout=None,
    )
    loop.train_run(spec, loop.build_stages(spec))
    ckpt = run / "checkpoints" / "s0_step000002.pt"
    sha = sha256_file(ckpt)
    episodes = data_episodes.read_split(SELECT_SPLIT)[:1]
    rows = fly_checkpoint(ckpt, episodes, arm="dp", run_seed=RUN_SEED, sha256=sha, workers=1)
    (run / "select").mkdir()
    rows_file = run / "select" / f"{ckpt.stem}_rows.csv"
    write_rows_csv(rows_file, rows)
    selection = {
        "arm": "dp",
        "run_seed": RUN_SEED,
        "inference_path": "cpu_fp32_single_sample",
        "weights": "ema",
        "list": {"name": SELECT_SPLIT},
        "selected": {
            "candidate": {"file": str(ckpt), "sha256": sha, "global_step": 2},
            "rows_file": rows_file.name,
        },
    }
    mf.write_json_atomic(run / "selection.json", selection)
    return {"run": run, "root": root, "ckpt": ckpt, "sha": sha, "episodes": episodes}


@pytest.fixture
def one_select_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    real = data_episodes.read_split

    def first(name: str) -> Any:
        assert name == SELECT_SPLIT  # nothing but select is ever read here
        return real(name)[:1]

    monkeypatch.setattr(data_episodes, "read_split", first)


def test_load_student_raw_weights_are_the_model_key(dp_run: dict[str, Any]) -> None:
    # P6-D4 side finding: "raw" must read the checkpoint's "model" entry, not a "raw" key.
    ckpt = torch.load(dp_run["ckpt"], map_location="cpu", weights_only=False)
    raw = registry.load_student(dp_run["ckpt"], "cpu", weights="raw")
    ema = registry.load_student(dp_run["ckpt"], "cpu")
    assert isinstance(raw, DPStudent) and isinstance(ema, DPStudent)
    for k, v in raw.net.state_dict().items():
        assert torch.equal(v, ckpt["model"][k])
    for k, v in ema.net.state_dict().items():
        assert torch.equal(v, ckpt["ema"][k])


# --------------------------------------------------------------------------- smoke criterion


def test_dp_sampled_mse_uses_flight_noise_keys(dp_run: dict[str, Any]) -> None:
    student = registry.load_student(dp_run["ckpt"], "cpu")
    assert isinstance(student, DPStudent)
    data = _chunk_data(dp_run["root"])
    seeds = [101, 202]
    rec = dp_sampled_mse(student, data, seeds, num_inference_steps=2, batch=5)
    assert rec["frames"] == len(data) and rec["Ta"] == 8 and len(rec["per_episode"]) == 2
    # By hand, frame t of episode e: key (run_seed, seeds[e], t), indices To-1 ... To+6.
    errs = []
    for e in range(2):
        start, t_len = int(data.table.episode_start[e]), int(data.table.episode_len[e])
        for t in range(t_len):
            b = data.batch(np.asarray([start + t]))
            chunk = student.sample_batch(b["images"], b["proprio"], [(RUN_SEED, seeds[e], t)], 2)
            errs.append((chunk[0, 1:9].astype(np.float64) - b["target"][0, 1:9]) ** 2)
    np.testing.assert_allclose(rec["value"], np.mean(errs), rtol=1e-6)
    assert rec["first_step"] == pytest.approx(np.mean([e[0] for e in errs]), rel=1e-6)
    with pytest.raises(ValueError, match="seeds"):
        dp_sampled_mse(student, data, [1], num_inference_steps=2)


# --------------------------------------------------------------------------- sweeps


def test_sweep_cells_from_the_frozen_config() -> None:
    reference, cells = sw.sweep_cells(registry.load_train_config("dp"))
    assert reference == sw.SweepCell(8, 10, False) and cells[0] == reference
    assert len(cells) == len(set(cells)) == 15
    grid = {(c.Ta, c.K, c.ensemble) for c in cells}
    assert grid == {(ta, 10, e) for ta in (1, 2, 4, 8, 12) for e in (False, True)} | {
        (8, k, False) for k in (1, 2, 5, 10, 25, 100)
    }
    flown = {c.flown_as() for c in cells}
    assert len(flown) == 14 and sw.SweepCell(1, 10, True).flown_as() == sw.SweepCell(1, 10, False)
    assert {c.K for c in cells if c.schedule_limited} == {1, 2}
    assert sw.SweepCell(8, 10, False).name == "Ta08_K010_off"
    assert sw.SweepCell(12, 10, True).name == "Ta12_K010_ens"


def test_sweep_cells_refuse_a_config_m_the_student_does_not_implement() -> None:
    cfg = registry.load_train_config("dp")
    cfg["inference_sweeps"] = {**cfg["inference_sweeps"], "temporal_ensembling": {"m": 0.02}}
    with pytest.raises(ValueError, match="m = 0.02"):
        sw.sweep_cells(cfg)


def test_compare_to_selection_ignores_only_the_inference_columns() -> None:
    sel = [{"episode_seed": "1", "outcome": "crash", "Ta": "", "K": "", "ensemble": ""}]
    ours = [{"episode_seed": "1", "outcome": "crash", "Ta": "8", "K": "10", "ensemble": "False"}]
    assert sw.compare_to_selection(ours, sel) == []
    bad = [{**ours[0], "outcome": "success"}]
    assert sw.compare_to_selection(bad, sel) != []
    assert sw.compare_to_selection(ours + ours, sel) != []


@pytest.mark.pybullet
def test_sweep_reproduces_selection_rows_and_resumes(
    dp_run: dict[str, Any], tmp_path: Path, one_select_episode: None
) -> None:
    cells = [sw.SweepCell(8, 2, False), sw.SweepCell(1, 2, True)]  # K = 2: schedule-limited
    out, summary = tmp_path / "sweeps", tmp_path / "dp_sweeps.json"
    entry = sw.run_sweeps(dp_run["run"], workers=1, out_root=out, summary_json=summary, cells=cells)
    assert entry["reproduces_selection_rows"] is True
    names = list(entry["cells"])
    # The alias's flown cell is recorded too (in the full sweep it comes first anyway).
    assert names == ["Ta08_K010_off", "Ta08_K002_off", "Ta01_K002_off", "Ta01_K002_ens"]
    alias = entry["cells"]["Ta01_K002_ens"]
    assert alias["alias_of"] == "Ta01_K002_off" and alias["ensemble"] is True
    assert entry["cells"]["Ta01_K002_off"]["rows_file"] == alias["rows_file"]
    assert entry["cells"]["Ta08_K002_off"]["schedule_limited"] is True
    rows = _read(out / "dp" / f"s{RUN_SEED}" / "Ta08_K002_off_rows.csv")
    assert [(r["Ta"], r["K"], r["ensemble"]) for r in rows] == [("8", "2", "False")]
    doc = json.loads(summary.read_text())
    assert doc["runs"][f"dp/s{RUN_SEED}"]["checkpoint"]["sha256"] == dp_run["sha"]
    assert doc["runs"][f"dp/s{RUN_SEED}"]["config"]["files"][0]["path"].endswith("dp.yaml")
    # Resumable per cell: nothing is flown again.
    again = sw.run_sweeps(dp_run["run"], workers=1, out_root=out, summary_json=summary, cells=cells)
    assert all(c["reused"] for c in again["cells"].values())
    # A selection row that differs in any non-inference column is caught.
    sel_file = dp_run["run"] / "select" / "s0_step000002_rows.csv"
    original = sel_file.read_text()
    try:
        sel_rows = _read(sel_file)
        sel_rows[0]["steps"] = str(int(sel_rows[0]["steps"]) + 1)
        write_rows_csv(sel_file, sel_rows)
        with pytest.raises(sw.SweepReproductionError):
            sw.run_sweeps(dp_run["run"], workers=1, out_root=out, summary_json=summary, cells=[])
    finally:
        sel_file.write_text(original)


# --------------------------------------------------------------------------- diagnostics


def test_chunk_jerk_classifies_boundaries() -> None:
    # Ta = 4: chunks at t = 0, 4, 8; constant within a chunk, a jump at each boundary.
    a = np.zeros((10, 3))
    a[4:8, 0] = 1.0
    a[8:, 0] = 3.0
    j = dd.chunk_jerk(a, 4)
    assert j["d1_boundary"] == [1.0, 2.0]  # t = 4, 8
    assert j["d1_within"] == [0.0] * 7  # t = 1, 2, 3, 5, 6, 7, 9
    # d2 at t is a boundary when t or t - 1 starts a chunk: t = 4, 5, 8, 9.
    assert j["d2_boundary"] == [1.0, 1.0, 2.0, 2.0]  # t = 4, 5, 8, 9
    assert j["d2_within"] == [0.0, 0.0, 0.0, 0.0]  # t = 2, 3, 6, 7
    s = dd.jerk_summary([j])
    assert s["d1_boundary"]["mean"] == 1.5 and s["d1_ratio_boundary_over_within"] is None
    ta1 = dd.chunk_jerk(a, 1)
    assert ta1["d1_within"] == [] and len(ta1["d1_boundary"]) == 9


def test_spread_keys_are_explicit_and_global_rng_untouched(dp_run: dict[str, Any]) -> None:
    student = registry.load_student(dp_run["ckpt"], "cpu")
    assert isinstance(student, DPStudent)
    data = _chunk_data(dp_run["root"])
    b = data.batch(np.arange(3))
    torch.manual_seed(9)
    expected = torch.rand(3)
    torch.manual_seed(9)
    one = dd.spread_and_clip(
        student.net, RUN_SEED, b["images"], b["proprio"], b["target"], n_samples=4,
        num_inference_steps=2,
    )  # fmt: skip
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
    two = dd.spread_and_clip(
        student.net, RUN_SEED, b["images"], b["proprio"], b["target"], n_samples=4,
        num_inference_steps=2,
    )  # fmt: skip
    assert one == two
    assert len(one["spread_executed"]) == 3 and all(s > 0 for s in one["spread_executed"])
    assert 0.0 <= one["clip_chunk"] <= 1.0 and 0.0 <= one["label_clip_executed"] <= 1.0
    # The keys are (run_seed, -1, 4 i + j): another run seed draws other noise.
    other = dd.spread_and_clip(
        student.net, RUN_SEED + 1, b["images"], b["proprio"], n_samples=4, num_inference_steps=2
    )
    assert other["spread_executed"] != one["spread_executed"] and "bias_executed" not in other


def test_open_loop_replay_uses_negative_episode_keys(dp_run: dict[str, Any]) -> None:
    student = registry.load_student(dp_run["ckpt"], "cpu", num_inference_steps=2)
    assert isinstance(student, DPStudent)
    data = _chunk_data(dp_run["root"])
    out = dd.open_loop_actions(student, data)
    assert [len(a) for a, _ in out] == [int(n) for n in data.table.episode_len]
    for e, (acts, labels) in enumerate(out):
        start = int(data.table.episode_start[e])
        b = data.batch(np.asarray([start]))
        chunk = student.sample_batch(
            b["images"], b["proprio"], [(RUN_SEED, dd.OPEN_LOOP_KEY_BASE - e, 0)]
        )[0]
        ex = chunking.executed_indices(2, 16, 8)
        np.testing.assert_array_equal(acts[: len(acts)], chunk[list(ex)][: len(acts)])
        np.testing.assert_array_equal(labels, data.table.label[start : start + len(acts)])


@pytest.mark.pybullet
def test_run_diagnostics_end_to_end(
    dp_run: dict[str, Any],
    tmp_path: Path,
    one_select_episode: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ila.train import diagnostics

    def held(root: Path, **_: Any) -> BCWindowDataset:
        return BCWindowDataset(_chunk_data(dp_run["root"]).table, 2)

    monkeypatch.setattr(diagnostics, "heldout_states", held)
    rec = dd.run_diagnostics(dp_run["run"], data_root=dp_run["root"], workers=1, out_root=tmp_path)
    written = json.loads((tmp_path / "dp" / f"s{RUN_SEED}.json").read_text())
    assert written["checkpoint"]["sha256"] == dp_run["sha"]
    assert rec["select"]["rows_match_selection"] and rec["select"]["action_jerk_mean_matches_rows"]
    assert rec["select"]["episodes"] == 1 and rec["select"]["replan_states"] >= 1
    held_rec = rec["heldout"]
    assert held_rec["episodes"] == 2 and held_rec["spread_states"] == held_rec["frames"]
    assert held_rec["open_loop_jerk"]["d1_boundary"]["n"] >= 0
    assert set(held_rec["teacher_label_jerk"]) == set(held_rec["open_loop_jerk"])
    assert rec["settings"]["Ta"] == 8 and rec["settings"]["K"] == 10
