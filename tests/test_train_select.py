"""Phase 4 selection and budget (CPU, no rollouts): the P3-D1 §4 rule, sigma*, candidates, §5.

The rule: highest ``select`` success; ties by the lower pooled p95 closing speed (touched-down
episodes; none ranks last); then the lowest step. sigma*: the same over the three DART trials'
selected checkpoints, then the smaller sigma. Candidates: every full checkpoint (EMA), the final
iteration's only for an iterated arm; timing/smoke/test runs and unfinished runs refused. The
budget: P3-D1 §5's single cut factor.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.train import budget
from ila.train import manifest as mf
from ila.train.select import (
    OUTCOME_ORDER,
    P95,
    SELECT_SPLIT,
    Candidate,
    CandidateResult,
    _result_record,
    candidates,
    choose,
    choose_sigma,
    summarise_rows,
)


def _result(step: int, success: int, p95: float, n: int = 150) -> CandidateResult:
    return CandidateResult(
        candidate=Candidate(f"s0_step{step:06d}.pt", 0, step, step, "x"),
        n=n,
        success=success,
        outcomes={"success": success, "crash": n - success},
        n_touchdown=n,
        p95_closing_speed_m_s=p95,
        mean_steps=60.0,
        rows_file="r.csv",
    )


def test_rule_success_then_p95_then_lowest_step() -> None:
    a = _result(5000, 148, 0.30)
    b = _result(10000, 149, 0.40)
    assert choose([a, b]) is b  # success first
    c = _result(15000, 149, 0.35)
    assert choose([a, b, c]) is c  # tie on success -> lower p95
    d = _result(20000, 149, 0.35)
    assert choose([d, c, b]) is c  # tie on both -> lowest step
    e = _result(25000, 149, math.inf)  # no touchdown ranks last on p95
    assert choose([e, b]) is b
    with pytest.raises(ValueError):
        choose([])


def _trial(sigma: float, success: int, p95: float, step: int = 30000) -> dict[str, Any]:
    return {
        "arm": "bc_dart",
        "purpose": "tuning",
        "sigma": sigma,
        "selected": _result_record(_result(step, success, p95)),
    }


def test_sigma_star_tie_goes_to_the_smaller_sigma() -> None:
    trials = {
        0.05: _trial(0.05, 140, 0.3),
        0.10: _trial(0.10, 145, 0.4),
        0.20: _trial(0.20, 145, 0.4),
    }
    assert choose_sigma(trials)["sigma_star"] == 0.10
    trials[0.20] = _trial(0.20, 145, 0.35)
    assert choose_sigma(trials)["sigma_star"] == 0.20  # p95 breaks the tie first
    trials[0.05] = _trial(0.05, 145, 0.35)
    assert choose_sigma(trials)["sigma_star"] == 0.05  # full tie -> smaller sigma
    with pytest.raises(ValueError, match="tuning"):
        choose_sigma({0.05: {**_trial(0.05, 1, 0.1), "purpose": "train"}})
    with pytest.raises(ValueError, match="record says"):
        choose_sigma({0.05: _trial(0.10, 1, 0.1)})


#: P8's two per-episode flags, both off (every runner row carries them).
NO_FLAGS: dict[str, Any] = {"tunnelled": False, "detectors_disagree": False}


def test_summarise_rows_pools_touched_down_closing_speeds() -> None:
    speeds = [0.2, float("nan"), 0.5, 0.3, float("nan")]
    outcomes = ["success", "timeout", "hard_landing", "success", "crash"]
    rows = [
        {"outcome": o, "closing_speed_normal_m_s": v, "steps": 60, **NO_FLAGS}
        for o, v in zip(outcomes, speeds, strict=True)
    ]
    s = summarise_rows(rows)
    assert s["success"] == 2 and s["n"] == 5 and s["n_touchdown"] == 3
    assert list(s["outcomes"]) == list(OUTCOME_ORDER)
    assert s["outcomes"]["timeout"] == 1 and s["outcomes"]["bounce"] == 0
    expected = float(np.percentile([0.2, 0.5, 0.3], 95))
    assert math.isclose(s["p95_closing_speed_m_s"], expected)
    assert math.isclose(
        s["p95_closing_speed_m_s"],
        float(p8_adapter.stats.quantile_linear(np.array([0.2, 0.5, 0.3]), P95)),
    )
    none = summarise_rows(
        [{"outcome": "crash", "closing_speed_normal_m_s": float("nan"), "steps": 9, **NO_FLAGS}]
    )
    assert none["p95_closing_speed_m_s"] == math.inf
    with pytest.raises(ValueError, match="unknown outcome"):
        summarise_rows(
            [{"outcome": "landed", "closing_speed_normal_m_s": 0.1, "steps": 1, **NO_FLAGS}]
        )


def _fake_run(root: Path, purpose: str = "train", state: str = "done", stages: int = 1) -> Path:
    (root / "checkpoints").mkdir(parents=True)
    ckpts = []
    idx = [0] if stages == 1 else list(range(1, stages + 1))
    for stage in idx:
        for k in (5000, 10000):
            name = f"s{stage}_step{k:06d}.pt"
            (root / "checkpoints" / name).write_bytes(name.encode())
            ckpts.append(
                {
                    "file": name,
                    "kind": "full",
                    "stage": stage,
                    "stage_step": k,
                    "global_step": (max(stage, 1) - 1) * 10000 + k,
                    "sha256": "x",
                }
            )
    ckpts.append({"file": "ema_last.pt", "kind": "ema_last", "stage": 0, "stage_step": 0,
                  "global_step": 10000, "sha256": "y"})  # fmt: skip
    mf.write_json_atomic(
        root / "manifest.json",
        {
            "purpose": purpose,
            "selectable": purpose in ("train", "tuning"),
            "state": state,
            "checkpoints": ckpts,
            "stages": [{"index": i} for i in idx],
        },
    )
    return root


def test_candidates(tmp_path: Path) -> None:
    single = candidates(_fake_run(tmp_path / "a"))
    assert [c.stage_step for c in single] == [5000, 10000]  # ema_last is not a candidate
    iterated = candidates(_fake_run(tmp_path / "b", stages=5))
    assert {c.stage for c in iterated} == {5} and [c.stage_step for c in iterated] == [5000, 10000]
    for purpose in ("timing", "smoke", "test"):
        with pytest.raises(ValueError, match="never selected"):
            candidates(_fake_run(tmp_path / purpose, purpose=purpose))
    with pytest.raises(ValueError, match="not done"):
        candidates(_fake_run(tmp_path / "running", state="running"))
    (tmp_path / "a" / "checkpoints" / "s0_step005000.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        candidates(tmp_path / "a")


def test_selection_list_is_select_only() -> None:
    assert SELECT_SPLIT == "select"


# --------------------------------------------------------------------------- budget


def _rates(bc: float, dp: float) -> dict[str, budget.Rate]:
    return {
        "bc": budget.Rate("bc", bc, 30.0, "t"),
        "dp": budget.Rate("dp", dp, 30.0, "t"),
        "state": budget.Rate("state", 300.0, 5.0, "t"),
    }


def test_budget_covers_every_planned_run() -> None:
    runs = budget.planned_runs()
    labels = {r.label for r in runs}
    for arm in ("bc_state", "bc", "bc_dart", "dp", "modular", "R1_lerobot_dp"):
        assert arm in labels
    assert {
        "bc_matched (iterations 1-5)",
        "bc_dagger (iterations 1-5)",
        "bc_dart sigma trials",
    } <= labels
    assert sum(1 for r in runs if r.label.startswith("ablation")) == 11
    by = {r.label: r for r in runs}
    assert by["bc_matched (iterations 1-5)"].steps == 50_000 and by["dp"].steps == 100_000
    assert by["bc_dart sigma trials"].runs == 3 and by["ablation dp_mix"].runs == 3


def test_budget_cut_factor_is_the_smallest_cut() -> None:
    runs = budget.planned_runs()
    fast = _rates(1000.0, 1000.0)
    assert budget.cut_factor(runs, fast) == 1.0
    slow = _rates(8.0, 4.0)
    f = budget.cut_factor(runs, slow)
    assert 0 < f < 1
    total = sum(r["gpu_hours"] for r in budget.project(runs, slow, f))
    assert total <= budget.BUDGET_HOURS
    above = sum(r["gpu_hours"] for r in budget.project(runs, slow, f + 0.002))
    assert above > budget.BUDGET_HOURS  # the largest factor (3 decimals) that fits
    cut = {r["label"]: r for r in budget.project(runs, slow, 0.5)}
    assert cut["bc"]["steps_per_run"] == 15_000 and cut["bc"]["checkpoint_every"] == 2500
    assert cut["bc"]["warmup"] == 250
    assert cut["timing dp (2k)"]["steps_per_run"] == 2000  # spent, never scaled
    with pytest.raises(ValueError, match="unscalable"):
        budget.cut_factor(
            runs, {k: dataclasses.replace(v, overhead_s=1e5) for k, v in slow.items()}
        )


# --------------------------------------------------------------------------- end to end


@pytest.mark.pybullet
@pytest.mark.parametrize(
    ("arm", "workers"), [("bc_state", 2), pytest.param("bc", 1, marks=pytest.mark.slow)]
)
def test_trained_checkpoint_flies_on_the_cpu_path(arm: str, workers: int, tmp_path: Path) -> None:
    """A tiny trained checkpoint flies train episodes through the runner (bc_state: spawned)."""
    import test_train_loop as tl
    from ila.data import episodes as data_episodes
    from ila.train import loop
    from ila.train.config import sha256_file
    from ila.train.select import fly_checkpoint

    root = tmp_path / "data"
    tl.write_store(root / "demos" / "ppo_2000.h5", [tl.synthetic_record(j, 6, j) for j in range(4)])
    spec = tl._spec(tl._cfg(arm), tmp_path / "run", root, steps_override=2, heldout=None)
    loop.train_run(spec, loop.build_stages(spec))
    ckpt = tmp_path / "run" / "ema_last.pt"
    listed = data_episodes.read_split("demos_0-999")[:workers]
    rows = fly_checkpoint(
        ckpt, listed, arm=arm, run_seed=3, sha256=sha256_file(ckpt), workers=workers, chunk=1
    )
    assert [r["episode_seed"] for r in rows] == [e.episode_seed for e in listed]
    assert all(r["ckpt_sha256"] == sha256_file(ckpt) for r in rows)
    assert all(r["outcome"] in OUTCOME_ORDER for r in rows)
    assert summarise_rows(rows)["n"] == workers


def test_fly_checkpoint_uses_the_checkpoints_contract_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint flies with the obs_variant it was trained on (leak NOTE 6)."""
    import torch

    from ila.eval import runner
    from ila.policies.registry import CHECKPOINT_FORMAT
    from ila.train import select

    seen: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(runner, "run_list_vision", fake)
    for variant in ("default", "plus_clock", "minus_last_action"):
        path = tmp_path / f"{variant}.pt"
        torch.save({"format": CHECKPOINT_FORMAT, "obs_variant": variant}, path)
        select.fly_checkpoint(path, [], arm="bc", run_seed=0, sha256="x", workers=1)
        assert seen["contract_variant"] == variant and seen["render"] == "full"


def test_fly_checkpoint_passes_the_appearance_shift_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 8: an evaluation-only appearance shift reaches the runner; an unknown one raises.

    The row tags (``ckpt_sha256``, ``Ta``/``K``/``ensemble``) do not depend on the shift, and
    the runner's ``render_cfg_sha256`` cell differs for every shifted config.
    """
    import torch

    from ila.eval import runner
    from ila.policies.registry import CHECKPOINT_FORMAT
    from ila.render.config import APPEARANCE_SHIFTS
    from ila.train import select

    seen: dict[str, Any] = {}

    def fake(episodes: Any, spec: Any, cfgs: Any, tags: Any, **kwargs: Any) -> list[Any]:
        seen.clear()
        seen.update(kwargs, tags=tags)
        return []

    monkeypatch.setattr(runner, "run_list_vision", fake)
    path = tmp_path / "dp.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "obs_variant": "default"}, path)
    kw: dict[str, Any] = {"arm": "dp", "run_seed": 4, "sha256": "abc", "workers": 1}
    select.fly_checkpoint(path, [], **kw)
    assert seen["appearance_shift"] is None
    plain = seen["tags"]
    for shift in APPEARANCE_SHIFTS:
        select.fly_checkpoint(path, [], appearance_shift=shift, **kw)
        assert seen["appearance_shift"] == shift and seen["render"] == "full"
        assert seen["contract_variant"] == "default"
        assert seen["tags"] == plain and plain.ckpt_sha256 == "abc"
        select.fly_checkpoint(path, [], appearance_shift=shift, inference=(8, 10, False), **kw)
        t = seen["tags"]
        assert (t.Ta, t.K, t.ensemble, t.ckpt_sha256) == (8, 10, False, "abc")
    shas = {runner.render_cfg_sha256("full", s) for s in (None, *APPEARANCE_SHIFTS)}
    assert len(shas) == 1 + len(APPEARANCE_SHIFTS)
    seen.clear()
    for bad in ("nope", "", "LIGHT_ROT_60"):
        with pytest.raises(ValueError, match="unknown appearance shift"):
            select.fly_checkpoint(path, [], appearance_shift=bad, **kw)
    assert not seen  # refused before anything flew


def test_smoke_records_repo_relative_checkpoint_paths() -> None:
    from ila.policies.registry import REPO_ROOT
    from ila.train.smoke import _repo_relative

    assert _repo_relative(REPO_ROOT / "artifacts" / "runs" / "smoke" / "bc" / "ema_last.pt") == (
        "artifacts/runs/smoke/bc/ema_last.pt"
    )


@pytest.mark.data
@pytest.mark.checkpoints
@pytest.mark.pybullet
def test_smoke_replay_matches_a_fresh_teacher_rollout() -> None:
    """Stored steps 0-3 of ppo_2000 episode 0 == a fresh SB3 teacher rollout (leak NOTE 5)."""
    from ila.data import episodes as data_episodes
    from ila.data.store import DemoStore
    from ila.train.loop import store_path
    from ila.train.smoke import replay_check

    path = store_path(p8_adapter.load_paths()["data_root"], "ppo_2000")
    rows = data_episodes.read_split("demos_0-1999")
    with DemoStore(path) as store:
        good = replay_check(store, "ep_00000", rows[0], "ppo")
        bad = replay_check(store, "ep_00000", rows[1], "ppo")  # negative control
    assert good["passed"] and good["steps"] == 4
    assert good["images_identical"] == good["proprio_identical"] == good["labels_identical"] == 4
    assert good["max_abs_label_diff"] == 0.0
    assert not bad["passed"]


# --------------------------------------------------------------------------- re-summary (S2)


def test_summarise_rows_counts_tunnelled_and_detector_disagreements() -> None:
    def row(t: Any, d: Any) -> dict[str, Any]:
        return {"outcome": "success", "closing_speed_normal_m_s": 0.3, "steps": 60,
                "tunnelled": t, "detectors_disagree": d}  # fmt: skip

    live = summarise_rows([row(True, False), row(False, True), row(np.bool_(True), False)])
    assert (live["n_tunnelled"], live["n_detectors_disagree"]) == (2, 1)
    stored = summarise_rows([row("True", "False"), row("False", "True"), row("True", "False")])
    assert (stored["n_tunnelled"], stored["n_detectors_disagree"]) == (2, 1)
    for bad in (float("nan"), "nan", "", 1):
        with pytest.raises(ValueError, match="boolean"):
            summarise_rows([row(bad, False)])


def _select_rows(sha: str, success: int, tunnelled: int) -> list[dict[str, Any]]:
    from ila.data import episodes as data_episodes

    out = []
    for i, e in enumerate(data_episodes.read_split(SELECT_SPLIT)):
        ok = i < success
        out.append(
            {
                "episode_seed": e.episode_seed,
                "ckpt_sha256": sha,
                "outcome": "success" if ok else "hard_landing",
                "closing_speed_normal_m_s": 0.2 + 0.001 * i,
                "steps": 60 + i % 3,
                "tunnelled": i < tunnelled,
                "detectors_disagree": i == 7,
            }
        )
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _flown_run(root: Path) -> Path:
    """A run as select_run left it before S2: rows CSVs and a record without the flag counts."""
    from ila.train.select import _result, _result_record

    (root / "select").mkdir(parents=True)
    results = []
    for step, success, tun in ((5000, 147, 2), (10000, 149, 1), (15000, 149, 3)):
        c = Candidate(f"s0_step{step:06d}.pt", 0, step, step, f"sha{step}")
        rows = _select_rows(c.sha256, success, tun)
        _write_rows(root / "select" / f"s0_step{step:06d}_rows.csv", rows)
        results.append(_result(c, summarise_rows(rows), f"s0_step{step:06d}_rows.csv"))
    best = choose(results)

    def old(r: CandidateResult) -> dict[str, Any]:
        rec = _result_record(r)
        del rec["n_tunnelled"], rec["n_detectors_disagree"]
        return rec

    record = {"arm": "bc", "workers": 24, "wall_clock_s": 12.5, "rule": "r",
              "candidates": [old(r) for r in results], "selected": old(best)}  # fmt: skip
    mf.write_json_atomic(root / "selection.json", record)
    return root


def test_resummarise_adds_counts_without_changing_the_selection(tmp_path: Path) -> None:
    from ila.train.select import resummarise_run

    run = _flown_run(tmp_path / "run")
    before = mf.read_json(run / "selection.json")
    new = resummarise_run(run)
    assert mf.read_json(run / "selection.json") == new
    assert new["selected"]["candidate"]["global_step"] == 10000  # same rule, same choice
    assert new["selected"]["n_tunnelled"] == 1 and new["selected"]["n_detectors_disagree"] == 1
    assert [c["n_tunnelled"] for c in new["candidates"]] == [2, 1, 3]
    assert new["wall_clock_s"] == 12.5 and new["workers"] == 24  # the flight's fields kept
    assert new["resummary"]["flown"] is False
    strip = {"n_tunnelled", "n_detectors_disagree"}
    assert [{k: v for k, v in c.items() if k not in strip} for c in new["candidates"]] == before[
        "candidates"
    ]
    assert resummarise_run(run) == new  # idempotent


@pytest.mark.parametrize("tamper", ["outcome", "order", "checkpoint", "selected"])
def test_resummarise_refuses_any_change(tmp_path: Path, tamper: str) -> None:
    import csv

    from ila.train.select import ResummaryMismatchError, resummarise_run

    run = _flown_run(tmp_path / "run")
    rows_file = run / "select" / "s0_step010000_rows.csv"
    with rows_file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if tamper == "outcome":
        rows[0]["outcome"] = "crash"
    elif tamper == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif tamper == "checkpoint":
        rows[3]["ckpt_sha256"] = "other"
    _write_rows(rows_file, rows)
    if tamper == "selected":
        rec = mf.read_json(run / "selection.json")
        rec["selected"] = rec["candidates"][2]
        mf.write_json_atomic(run / "selection.json", rec)
    before = (run / "selection.json").read_bytes()
    with pytest.raises(ResummaryMismatchError):
        resummarise_run(run)
    assert (run / "selection.json").read_bytes() == before  # nothing written


def test_inference_settings_recorded_for_diffusion_arms_only() -> None:
    # Pre-gate review M8: selection.json states the Ta / K it flew.
    from ila.train.select import inference_settings

    dp = {
        "config_resolved": {
            "horizons": {"n_action_steps": 8},
            "inference": {"num_inference_steps": 10},
        }
    }
    r1 = {
        "config_resolved": {
            "policy": {"n_action_steps": 8},
            "inference": {"num_inference_steps": 10},
        }
    }
    assert inference_settings(dp) == {"Ta": 8, "K": 10, "temporal_ensembling": False}
    assert inference_settings(r1) == {"Ta": 8, "K": 10, "temporal_ensembling": False}
    assert inference_settings({"config_resolved": {"steps": 30000}}) is None
