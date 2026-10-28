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


def test_summarise_rows_pools_touched_down_closing_speeds() -> None:
    speeds = [0.2, float("nan"), 0.5, 0.3, float("nan")]
    outcomes = ["success", "timeout", "hard_landing", "success", "crash"]
    rows = [
        {"outcome": o, "closing_speed_normal_m_s": v, "steps": 60}
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
        [{"outcome": "crash", "closing_speed_normal_m_s": float("nan"), "steps": 9}]
    )
    assert none["p95_closing_speed_m_s"] == math.inf
    with pytest.raises(ValueError, match="unknown outcome"):
        summarise_rows([{"outcome": "landed", "closing_speed_normal_m_s": 0.1, "steps": 1}])


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
