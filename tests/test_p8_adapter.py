"""Phase 1: the P8 adapter re-exports P8's objects by reference and exposes scene handles."""

from __future__ import annotations

import inspect

import numpy as np
import pybullet as pyb
import pytest
import rld.eval.stats as p8_stats
from rld.envs.observation import obs_fields as p8_obs_fields

from ila import p8_adapter
from ila.data.episodes import read_split


def _env_after_reset(ss: str = "SS5") -> tuple[p8_adapter.DeckLandingAviary, np.ndarray]:
    """Reset on a ``train``-pool row of the committed ``demos`` split (P1-D2 obligation 1)."""
    cfgs = p8_adapter.load_eval_configs()
    listed = next(r for r in read_split("demos_0-1999") if r.ss == ss)
    env = p8_adapter.make_env(cfgs, listed)
    obs, _ = env.reset(seed=listed.episode_seed)
    return env, obs


# --------------------------------------------------------------------------- stats


def test_stats_namespace_covers_p8_all_exactly() -> None:
    exported = {n for n in dir(p8_adapter.stats) if not n.startswith("_")}
    assert exported == set(p8_stats.__all__)


@pytest.mark.parametrize("name", sorted(p8_stats.__all__))
def test_stats_reexport_is_identity(name: str) -> None:
    assert getattr(p8_adapter.stats, name) is getattr(p8_stats, name)
    assert getattr(p8_adapter.P8Stats, name) is getattr(p8_stats, name)


def test_stats_namespace_is_read_only() -> None:
    with pytest.raises(AttributeError):
        p8_adapter.stats.iqm = np.mean
    with pytest.raises(AttributeError):
        p8_adapter.P8Stats.iqm = np.mean


def test_stats_call_through() -> None:
    assert p8_adapter.stats.wilson_interval(5, 10) == p8_stats.wilson_interval(5, 10)


# --------------------------------------------------------------------------- obs layout


def test_obs_fields_is_p8_layout_under_eval_config() -> None:
    cfgs = p8_adapter.load_eval_configs()
    assert p8_adapter.obs_fields() == p8_obs_fields(cfgs.observation)
    assert sum(f.size for f in p8_adapter.obs_fields()) == 25


def test_list_paths_are_p8_lists() -> None:
    paths = p8_adapter.list_paths()
    assert list(paths) == ["id", "unseen_seastate", "unseen_heading", "unseen_vessel", "static"]
    assert all(p.is_file() for p in paths.values())


# --------------------------------------------------------------------------- teachers


def _assert_eval_policy(policy: p8_adapter.EvalPolicy) -> None:
    for method in ("reset", "act"):
        assert callable(getattr(policy, method, None)), method
    params = list(inspect.signature(policy.reset).parameters)
    assert params[:2] == ["seed", "context"]
    assert getattr(policy, "privileged", False) is False


def _assert_acts(policy: p8_adapter.EvalPolicy) -> None:
    env, obs = _env_after_reset()
    try:
        policy.reset(0, None)
        action = np.asarray(policy.act(obs))
    finally:
        env.close()
    assert action.shape == (3,)
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0)


@pytest.mark.pybullet
@pytest.mark.parametrize("name", ["pid_feedforward", "gated"])
def test_controller_teacher_is_eval_policy(name: str) -> None:
    policy = p8_adapter.teacher(name, 0)
    _assert_eval_policy(policy)
    _assert_acts(policy)


@pytest.mark.checkpoints
@pytest.mark.pybullet
def test_ppo_teacher_is_eval_policy() -> None:
    policy = p8_adapter.teacher("ppo", 0)
    _assert_eval_policy(policy)
    _assert_acts(policy)


@pytest.mark.parametrize(
    ("name", "seed"), [("ppo", 1), ("pid_feedforward", 3), ("oracle_gated", 0), ("bc", 0)]
)
def test_teacher_rejects_unavailable(name: str, seed: int) -> None:
    with pytest.raises(ValueError, match="teacher"):
        p8_adapter.teacher(name, seed)


# --------------------------------------------------------------------------- scene


@pytest.mark.pybullet
def test_scene_handles_find_p8_bodies() -> None:
    env, _ = _env_after_reset()
    try:
        h = p8_adapter.scene_handles(env)
        assert len({h.plane, h.drone, h.plate, h.pad_marker}) == 4
        marker = pyb.getVisualShapeData(h.pad_marker, physicsClientId=h.client)
        assert len(marker) == 1 and marker[0][2] == pyb.GEOM_CYLINDER
        assert tuple(marker[0][7]) == pytest.approx(p8_adapter.PAD_MARKER_RGBA)
        assert pyb.getCollisionShapeData(h.pad_marker, -1, physicsClientId=h.client) == ()
        plate = pyb.getVisualShapeData(h.plate, physicsClientId=h.client)
        assert plate[0][2] == pyb.GEOM_BOX
        # URDF_IGNORE_VISUAL_SHAPES drops cf2.dae, but PyBullet then renders the collision
        # cylinder (length 0.025 m, radius 0.06 m, white) as the drone's visual (P1-D1).
        drone = pyb.getVisualShapeData(h.drone, physicsClientId=h.client)
        assert [(v[1], v[2]) for v in drone] == [(-1, pyb.GEOM_CYLINDER)]
        assert tuple(drone[0][3][:2]) == pytest.approx((0.025, 0.06))
    finally:
        env.close()


@pytest.mark.pybullet
def test_physics_step_count() -> None:
    env, _ = _env_after_reset()
    try:
        assert p8_adapter.physics_step_count(env) == 0
        env.step(np.zeros(3))
        assert p8_adapter.physics_step_count(env) == 8
        env.step(np.zeros(3))
        assert p8_adapter.physics_step_count(env) == 16
    finally:
        env.close()


# ------------------------------------------------------------ runner contract (Phase 2)


def _p8_runner_objects() -> dict[str, object]:
    """Every Phase 2 re-export, keyed by adapter name, as P8 defines it (pinned commit)."""
    import dmf.data.splits
    import rld.control.base
    import rld.deck.splits
    import rld.deploy.onnx_policy
    import rld.deploy.parity
    import rld.envs.landing_env
    import rld.eval.envs
    import rld.eval.episodes
    import rld.eval.reproduce
    import rld.eval.runner
    import rld.eval.storage
    import rld.eval.truth

    runner = rld.eval.runner
    return {
        "EPISODE_COLUMNS": runner.EPISODE_COLUMNS,
        "LIST_ROW_COLUMNS": runner.LIST_ROW_COLUMNS,
        "RECORD_COLUMNS": runner.RECORD_COLUMNS,
        "DEFAULT_CHUNK": runner.DEFAULT_CHUNK,
        "DEFAULT_WORKERS": runner.DEFAULT_WORKERS,
        "PolicySpec": runner.PolicySpec,
        "controller_spec": runner.controller_spec,
        "callable_spec": runner.callable_spec,
        "episode_rows_csv": runner.episode_rows_csv,
        "feed_skip_reason": runner.feed_skip_reason,
        "split_runnable": runner.split_runnable,
        "EpisodeListMismatchError": runner.EpisodeListMismatchError,
        "EvalPolicy": runner.EvalPolicy,
        "motion_for": rld.eval.envs.motion_for,
        "pad_offset_for": rld.eval.envs.pad_offset_for,
        "PrivilegedContext": rld.control.base.PrivilegedContext,
        "td_in_quiescent_window": rld.eval.truth.td_in_quiescent_window,
        "touchdown_rule": rld.eval.truth.touchdown_rule,
        "ACTION_DIM": rld.envs.landing_env.ACTION_DIM,
        "compare_to_reference": rld.eval.reproduce.compare_to_reference,
        "read_text": rld.eval.storage.read_text,
        "dev_pool": rld.deck.splits.dev_pool,
        "EpisodeDraw": rld.eval.episodes.EpisodeDraw,
        "resolve_draws": rld.eval.episodes.resolve_draws,
        "label_entropy": rld.eval.episodes.label_entropy,
        "cell_order": rld.eval.episodes.cell_order,
        "canonical_csv": rld.eval.episodes.canonical_csv,
        "content_sha256": rld.eval.episodes.content_sha256,
        "DEFAULT_PAD": rld.eval.episodes.DEFAULT_PAD,
        "reference_actions": rld.deploy.parity.reference_actions,
        "open_session": rld.deploy.parity.open_session,
        "check_ort": rld.deploy.parity.check_ort,
        "PARITY_TOLERANCE": rld.deploy.parity.PARITY_TOLERANCE,
        "cpu_session": rld.deploy.onnx_policy.cpu_session,
        "attach_onnx": rld.deploy.onnx_policy.attach_onnx,
        # P8's runner imports dmf's function; the adapter re-exports that same object.
        "realization_key": dmf.data.splits.realization_key,
    }


@pytest.mark.parametrize("name", sorted(_p8_runner_objects()))
def test_runner_reexport_is_identity(name: str) -> None:
    assert getattr(p8_adapter, name) is _p8_runner_objects()[name]
    assert name in p8_adapter.__all__


def test_realization_key_is_the_one_p8_runner_uses() -> None:
    import rld.eval.runner

    assert p8_adapter.realization_key is rld.eval.runner.realization_key


def test_resolve_draws_signature_is_p8s() -> None:
    import inspect as _inspect

    params = _inspect.signature(p8_adapter.resolve_draws).parameters
    assert list(params)[:2] == ["draws", "cfgs"]
    assert {"workers", "chunk"} <= set(params)
    assert params["workers"].kind is _inspect.Parameter.KEYWORD_ONLY
    assert p8_adapter.DEFAULT_PAD == "aft"


def test_runner_constants_are_p8_values() -> None:
    # Values as read in P8's source (rld/eval/runner.py, rld/envs/landing_env.py).
    assert p8_adapter.DEFAULT_CHUNK == 25
    assert p8_adapter.DEFAULT_WORKERS == 24
    assert p8_adapter.ACTION_DIM == 3
    assert len(p8_adapter.EPISODE_COLUMNS) == 13 + 3 + 23 + 6
    assert p8_adapter.EPISODE_COLUMNS[: len(p8_adapter.LIST_ROW_COLUMNS)] == (
        p8_adapter.LIST_ROW_COLUMNS
    )


@pytest.mark.pybullet
def test_set_motion_matches_p8_run_chunk() -> None:
    """``set_motion`` re-points a reused env the way P8's ``run_chunk`` does (hull switch)."""
    cfgs = p8_adapter.load_eval_configs()
    first = p8_adapter.read_list(p8_adapter.list_path("static"))[0]
    moving = p8_adapter.read_list(p8_adapter.list_path("unseen_vessel"))[0]
    assert moving.vessel != first.vessel
    env = p8_adapter.make_env(cfgs, first)
    try:
        for listed in (first, moving, first):
            pad, offset = p8_adapter.set_motion(env, cfgs, listed)
            assert pad == listed.pad
            assert offset == p8_adapter.pad_offset_for(cfgs, listed.vessel, listed.pad)
            assert env.pad == pad and tuple(env.pad_offset_m) == offset
            _, info = env.reset(seed=listed.episode_seed)
            key = p8_adapter.realization_key(
                listed.ss,
                listed.heading_deg,
                listed.speed_kn,
                listed.vessel,
                listed.realization_seed,
            )
            assert info["realization_key"] == key
            assert float(env.record.t0_model_s) == listed.t0_model_s
    finally:
        env.close()


@pytest.mark.parametrize("name", ["ppo", "pid_feedforward", "gated"])
def test_teacher_spec_builds_the_teacher(name: str) -> None:
    import functools
    import pickle

    spec = p8_adapter.teacher_spec(name)
    assert spec.method == name and spec.run_seed == 0
    assert spec.privileged is False and spec.needs_motion_feed is False
    assert isinstance(spec.build, functools.partial)
    assert spec.build.func is p8_adapter.teacher and spec.build.args == (name, 0)
    assert pickle.loads(pickle.dumps(spec)).build.args == (name, 0)  # spawn-pool safe


@pytest.mark.parametrize(("name", "seed"), [("ppo", 1), ("oracle_gated", 0), ("bc", 0)])
def test_teacher_spec_rejects_unavailable(name: str, seed: int) -> None:
    with pytest.raises(ValueError, match="teacher"):
        p8_adapter.teacher_spec(name, seed)


# --------------------------------------------------------------------------- committed rows
# P8's committed rows are read-only evaluation references (P3-D1 §4 allows reading them at any
# time). The ceilings below are P3-D1 §6's, "measured before any student exists".

_COMMITTED = [
    *(("e07", "ppo", s) for s in range(5)),
    *(("e01", m, 0) for m in ("pid_feedforward", "pid_track_descend", "gated", "oracle_gated")),
]


@pytest.mark.parametrize(("experiment", "method", "seed"), _COMMITTED)
def test_committed_rows_shape_and_order(experiment: str, method: str, seed: int) -> None:
    got = p8_adapter.committed_rows(experiment, method, seed)
    assert len(got.rows) == 2800
    assert got.columns == p8_adapter.EPISODE_COLUMNS
    assert all(tuple(r) == p8_adapter.EPISODE_COLUMNS for r in got.rows)
    assert {r["method"] for r in got.rows} == {method}
    assert {r["run_seed"] for r in got.rows} == {str(seed)}
    assert {r["pad"] for r in got.rows} == {"aft"}
    # File order is P8's list order: the five lists in MANIFEST order, 200 per cell.
    cells = list(dict.fromkeys((r["regime"], r["ss"]) for r in got.rows))
    assert [c[0] for c in cells][0] == "id" and cells[-1] == ("static", "static")
    assert len(cells) == 14
    listed = [
        e for n in p8_adapter.list_names() for e in p8_adapter.read_list(p8_adapter.list_path(n))
    ]
    assert [(r["regime"], r["ss"], r["index"]) for r in got.rows] == [
        (e.regime, e.ss, str(e.index)) for e in listed
    ]


def test_committed_rows_records_what_it_read() -> None:
    e07 = p8_adapter.committed_rows("e07", "ppo", 0)
    assert e07.logical_path == "results/e07/matrix/episodes.csv"
    assert e07.file == "results/e07/matrix/episodes.csv.gz"
    sidecar = (p8_adapter.P8_ROOT / "results/e07/matrix/episodes.csv.sha256").read_text()
    assert e07.text_sha256 == sidecar.split()[0]
    assert e07.file_sha256 == p8_adapter.sha256_file(p8_adapter.P8_ROOT / e07.file)
    e01 = p8_adapter.committed_rows("e01", "gated")
    assert e01.file == e01.logical_path == "results/e01/episodes.csv"
    assert (
        e01.file_sha256
        == e01.text_sha256
        == p8_adapter.sha256_file(p8_adapter.P8_ROOT / "results/e01/episodes.csv")
    )
    prov = e07.provenance()
    assert prov["n_rows"] == 2800 and prov["text_sha256"] == e07.text_sha256
    assert "rows" not in prov


def test_committed_rows_are_fresh_copies() -> None:
    first = p8_adapter.committed_rows("e01", "gated", 0, regime="id", ss="SS3")
    first.rows[0]["outcome"] = "tampered"
    again = p8_adapter.committed_rows("e01", "gated", 0, regime="id", ss="SS3")
    assert again.rows[0]["outcome"] != "tampered"


def test_committed_rows_all_seeds_and_errors() -> None:
    every = p8_adapter.committed_rows("e07", "ppo")
    assert len(every.rows) == 5 * 2800 and every.run_seed is None
    assert [r["run_seed"] for r in every.rows][::2800] == ["0", "1", "2", "3", "4"]
    with pytest.raises(ValueError, match="experiment"):
        p8_adapter.committed_rows("e99", "ppo")
    with pytest.raises(ValueError, match="no rows"):
        p8_adapter.committed_rows("e07", "pid_feedforward")
    with pytest.raises(ValueError, match="no rows"):
        p8_adapter.committed_rows("e01", "gated", 1)


def _successes(rows: tuple[dict[str, str], ...]) -> int:
    return sum(r["outcome"] == "success" for r in rows)


def test_committed_ppo_ceiling() -> None:
    """P3-D1 §6: T_ppo s0 100 % at id SS3-SS5; 197/200 at id SS6, the 3 losses hard landings."""
    for ss in ("SS3", "SS4", "SS5"):
        assert (
            _successes(p8_adapter.committed_rows("e07", "ppo", 0, regime="id", ss=ss).rows) == 200
        )
    ss6 = p8_adapter.committed_rows("e07", "ppo", 0, regime="id", ss="SS6").rows
    assert len(ss6) == 200 and _successes(ss6) == 197
    assert sorted(r["outcome"] for r in ss6 if r["outcome"] != "success") == ["hard_landing"] * 3


@pytest.mark.parametrize(
    ("method", "per_ss"),
    [("gated", (200, 197, 179, 126)), ("pid_feedforward", (200, 200, 198, 181))],
)
def test_committed_baseline_ceilings(method: str, per_ss: tuple[int, ...]) -> None:
    """P3-D1 §6: gated 100 / 98.5 / 89.5 / 63.0 %; pid_feedforward 100 / 100 / 99.0 / 90.5 %."""
    for ss, expected in zip(("SS3", "SS4", "SS5", "SS6"), per_ss, strict=True):
        rows = p8_adapter.committed_rows("e01", method, 0, regime="id", ss=ss).rows
        assert len(rows) == 200 and _successes(rows) == expected, (ss, _successes(rows))


# --------------------------------------------------------------------------- flip floor


def test_flip_floor_matches_p8_and_p3_d1() -> None:
    floor = p8_adapter.FLIP_FLOOR_SS6
    assert (floor.flipped, floor.of, floor.draws) == (53, 800, 20)
    assert floor.single_draw_median_pct == 1.9
    assert floor.rate == 53 / 800 and f"{100 * floor.rate:.1f} %" == "6.6 %"
    p8_lines = (
        (p8_adapter.P8_ROOT / "docs" / "protocol.md").read_text(encoding="utf-8").splitlines()
    )
    assert "median 1.9 %" in p8_lines[5311 - 1]
    assert "53 of 800" in p8_lines[5314 - 1] and "6.6 %" in p8_lines[5314 - 1]
    assert "53 of 800" in p8_lines[5455 - 1]
    # P3-D1 §1's sentence, verbatim (the protocol wraps it; compare with whitespace folded).
    ours = (p8_adapter.REPO_ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
    start = ours.index("### P3-D1")
    block = " ".join(ours[start : ours.index("\n### ", start + 1)].split())
    assert f'"{floor.sentence}"' in block


def test_determinism_floor_is_the_adapter_constant() -> None:
    from ila.eval import determinism

    floor = p8_adapter.FLIP_FLOOR_SS6
    assert {
        "flipped": floor.flipped,
        "of": floor.of,
        "rate": floor.rate,
        "cell": floor.cell,
        "definition": floor.definition,
        "source": floor.source,
    } == determinism.P8_FLOOR
