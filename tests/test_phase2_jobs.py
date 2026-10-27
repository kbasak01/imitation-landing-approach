"""Phase 2 job plumbing: status files, repo-relative paths, comparisons, projection, numerics.

Fast unit tests on synthetic inputs. The long jobs themselves are exercised by their
``--smoke`` runs; these pin the arithmetic that turns their measurements into verdicts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.eval import batched, determinism, jobs, label_parity, reproduce, throughput

# --------------------------------------------------------------------------- jobs


def test_rel_is_repo_relative() -> None:
    assert jobs.rel(p8_adapter.P8_ROOT / "results/e01/episodes.csv") == (
        "external/rl-land-on-moving-deck/results/e01/episodes.csv"
    )
    assert jobs.rel(p8_adapter.REPO_ROOT / "artifacts/runs/x") == "artifacts/runs/x"
    assert jobs.rel("/definitely/elsewhere") == "/definitely/elsewhere"
    assert reproduce._rel(p8_adapter.REPO_ROOT / "artifacts") == "artifacts"


def test_job_status_writes_state_and_pid(tmp_path: Path) -> None:
    status = jobs.JobStatus(tmp_path, "unit")
    status.update("running", progress={"done": 1})
    record = json.loads((tmp_path / "status.json").read_text())
    assert record["state"] == "running" and record["progress"] == {"done": 1}
    assert isinstance(record["pid"], int) and record["job"] == "unit"


# --------------------------------------------------------------------------- determinism


def _row(index: int, outcome: str = "crash", steps: int = 30, ret: float = 1.0) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(p8_adapter.EPISODE_COLUMNS, 0.0)
    row.update(regime="select", ss="SS3", index=index, outcome=outcome, steps=steps)
    row.update({"return": ret, "episode_seed": 100 + index, "closing_speed_normal_m_s": math.nan})
    return row


def test_compare_runs_counts_text_diffs_and_flips() -> None:
    ref = [_row(0), _row(1), _row(2)]
    same, diffs = determinism.compare_runs(ref, [dict(r) for r in ref])
    assert same["bit_identical"] and same["n_text_differ"] == 0 and diffs == []
    other = [_row(0), _row(1, ret=1.0 + 1e-12), _row(2, outcome="timeout", steps=361)]
    summary, diffs = determinism.compare_runs(ref, other)
    assert summary["n_text_differ"] == 2 and summary["n_outcome_flips"] == 1
    assert summary["flips_by_class"] == {"crash->timeout": 1}
    assert [d["index"] for d in diffs] == [1, 2]
    assert diffs[0]["differing_columns"] == "return" and not diffs[0]["flip"]
    assert diffs[1]["absdiff_steps"] == 331.0
    assert diffs[1]["absdiff_closing_speed_normal_m_s"] == ""  # NaN in both: not a difference
    with pytest.raises(ValueError, match="order"):
        determinism.compare_runs(ref, [ref[1], ref[0], ref[2]])


def test_p8_floor_is_cited() -> None:
    assert determinism.P8_FLOOR["flipped"] == 53 and determinism.P8_FLOOR["of"] == 800
    path = p8_adapter.P8_ROOT / "docs" / "protocol.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "53 of 800" in lines[5314 - 1] and "6.6 %" in lines[5314 - 1]


# --------------------------------------------------------------------------- throughput


def _trow(**kw: Any) -> dict[str, Any]:
    out = dict.fromkeys(throughput.COLUMNS, "")
    out.update(kw)
    return out


def test_projection_arithmetic() -> None:
    rows = [
        _trow(section="teacher", name="pid_feedforward", workers=24, steps_per_s=1000.0),
        _trow(
            section="teacher", name="ppo", workers=24, steps_per_s=800.0, steady_steps_per_s=900.0
        ),
        _trow(section="e2e", name="dummy_bc", batch=24, steps_per_s=500.0),
        _trow(section="e2e", name="dummy_dp", batch=24, K=10, Ta=8, steps_per_s=200.0),
        _trow(section="e2e", name="dummy_dp", batch=24, K=10, Ta=1, steps_per_s=50.0),
    ]
    lengths = dict.fromkeys(
        [
            "ppo_train_like",
            "ppo_all",
            "ppo_id_ss4_ss6",
            "ppo_id_ss5",
            "ppo_id_ss5_ss6",
            "pid_train_like",
            "gated_train_like",
        ],
        100.0,
    )
    proj = {r["task"]: r for r in throughput.projection(rows, lengths)}
    # 2000 episodes x 100 steps / 900 steps/s (the steady rate wins) / 3600.
    assert proj["demo ppo_2000"]["hours"] == pytest.approx(2000 * 100 / 900 / 3600, abs=1e-3)
    dp = proj["matrix core dp 5 seeds x 2800"]
    assert dp["hours"] == pytest.approx(14000 * 100 / 200 / 3600, abs=1e-3)
    assert dp["hours_upper"] == pytest.approx(14000 * 361 / 200 / 3600, abs=1e-3)
    assert dp["flag"] == ""  # 7.0 h even with every episode a timeout
    ta1 = proj["sweep dp_Ta (Ta<8 or ensembling: 6 configs) x 5 seeds x 600"]
    assert ta1["rate_steps_per_s"] == 50.0  # costed at the Ta=1 rate
    assert ta1["flag"] == "FLAG >24 h"  # 18000 x 361 / 50 / 3600 = 36 h upper bound
    # K=100 was not measured here: falls back to the K10/Ta8 rate.
    assert proj["sweep dp_K (K 25 and 100: 2) x 5 seeds x 200"]["rate_steps_per_s"] == 200.0
    assert "TOTAL (central)" in proj


def test_episode_lengths_from_p8_rows() -> None:
    lengths = throughput.episode_lengths()
    assert set(lengths) >= {"ppo_train_like", "ppo_all", "pid_train_like", "gated_train_like"}
    assert all(30.0 < v < 361.0 for v in lengths.values())


# --------------------------------------------------------------------------- batched numerics


def test_numerics_modes() -> None:
    torch = pytest.importorskip("torch")
    strict = batched.configure_torch_numerics("cpu", "strict")
    assert strict["cudnn_allow_tf32"] is False and strict["matmul_allow_tf32"] is False
    default = batched.configure_torch_numerics("cpu", "torch_default")
    assert default["cudnn_allow_tf32"] is True and default["matmul_allow_tf32"] is False
    batched.configure_torch_numerics("cpu", "strict")
    assert torch.backends.cudnn.allow_tf32 is False
    with pytest.raises(ValueError, match="numerics"):
        batched.configure_torch_numerics("cpu", "fast")


def test_timed_policy_forwards_and_times() -> None:
    class Inner:
        run_seed = 3
        resets: list[tuple[int, int]] = []

        def reset_slot(self, slot: int, episode_seed: int) -> None:
            Inner.resets.append((slot, episode_seed))

        def act_batch(self, requests: Any) -> np.ndarray:
            return np.zeros((len(requests), 3))

    timed = batched.TimedPolicy(Inner())  # type: ignore[arg-type]
    timed.reset_slot(1, 7)
    out = timed.act_batch([object(), object()])  # type: ignore[list-item]
    assert out.shape == (2, 3) and Inner.resets == [(1, 7)] and timed.run_seed == 3
    assert timed.calls[0][0] == 2 and timed.first_start is not None
    assert timed.last_end is not None and timed.last_end >= timed.first_start


# --------------------------------------------------------------------------- label parity


def test_diff_stats_and_obs_sha() -> None:
    a = np.zeros((4, 3), dtype=np.float32)
    b = a.copy()
    b[2, 1] = 1e-7
    where = [("SS3", 0, t) for t in range(4)]
    stats = label_parity.diff_stats(a, b, where)
    assert stats["argmax_row"] == 2 and stats["argmax_component"] == 1
    assert stats["max_abs"] == pytest.approx(1e-7, rel=1e-6)
    assert stats["argmax_episode"] == {"ss": "SS3", "index": 0, "step": 2}
    assert stats["fraction_bit_equal"] == pytest.approx(11 / 12)
    assert label_parity.obs_sha256(a) == label_parity.obs_sha256(a.copy())
    assert label_parity.obs_sha256(a) != label_parity.obs_sha256(a.reshape(3, 4))


# --------------------------------------------------------------------------- render bench


@pytest.mark.pybullet
@pytest.mark.parametrize("mode", ["camera", "full"])
def test_render_bench_times_frames(mode: str) -> None:
    from ila.render.bench import render_frame_times, sized_config

    assert sized_config(64).camera.width_px == 64 == sized_config(64).camera.height_px
    cfgs = p8_adapter.load_eval_configs()
    listed = p8_adapter.read_list(p8_adapter.list_path("static"))[0]
    policy = p8_adapter.controller_spec("pid_feedforward").build(cfgs)
    times = render_frame_times(cfgs, listed, policy, mode=mode, size_px=64, n_frames=5)
    assert times.shape == (5,) and np.all(times > 0)
