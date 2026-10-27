"""Phase 2: the vision runner keeps P8's contract, and rendering changes no row.

Flown on P8's own ``tiny_list`` fixture (``runner_support.build_tiny_list``: dev-pool train and
static episodes, never a frozen evaluation list). The re-implemented private helpers are
compared with P8's own (imported here directly; only ``src/`` is restricted to the adapter).

(a) helper equality with P8's privates and P8's ``run_list``; (b) rows independent of workers
and chunk; (c) render off/camera/full give identical P8-row text; (d) a start mismatch raises;
(e) P1-D2 Phase 2 obligation 1: no ``RenderMeta`` and no physics step count in any public
rollout or record type; (f) obligation 2: students receive only a contract-built
``StudentInput`` with ``Frame.rgb`` pixels; (g) ``needs_motion_feed`` specs are refused.
Units: times seconds model scale, lengths metres model scale; actions normalised.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import rld.eval.runner as p8_runner
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.eval.episodes import ListedEpisode

from ila import p8_adapter
from ila.eval import batched, runner
from ila.obs.split import load_contract
from ila.render import scene
from ila.render.camera import Frame, RenderMeta
from runner_support import build_pixel_student, build_tiny_list, pixel_action, replay_inputs

pytestmark = [pytest.mark.pybullet, pytest.mark.slow]


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    return load_eval_configs()


@pytest.fixture(scope="module")
def tiny_list(cfgs: EvalConfigs) -> list[ListedEpisode]:
    return build_tiny_list(cfgs)


def _p8_csv(rows: list[dict[str, Any]]) -> str:
    return p8_adapter.episode_rows_csv(runner.p8_rows(rows))


# --------------------------------------------------------------------------- (a) helpers


def test_none_to_nan_equals_p8() -> None:
    for value in (None, 0, 1.5, "x", True, False, float("inf")):
        ours, theirs = runner.none_to_nan(value), p8_runner._none_to_nan(value)
        assert type(ours) is type(theirs)
        assert ours == theirs or (math.isnan(ours) and math.isnan(theirs))


class _Flagged:
    def __init__(self, privileged: bool | None, feed: bool | None) -> None:
        if privileged is not None:
            self.privileged = privileged
        if feed is not None:
            self.needs_motion_feed = feed


def _outcome(fn: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # compared by type and message
        return type(exc).__name__, str(exc)
    return "ok", ""


@pytest.mark.parametrize("spec_priv", [False, True])
@pytest.mark.parametrize("spec_feed", [False, True])
@pytest.mark.parametrize("declared_priv", [None, False, True])
@pytest.mark.parametrize("declared_feed", [None, False, True])
def test_check_privilege_equals_p8(
    spec_priv: bool, spec_feed: bool, declared_priv: bool | None, declared_feed: bool | None
) -> None:
    spec = p8_adapter.PolicySpec("spy", spec_priv, build_pixel_student, 0, spec_feed)
    policy = _Flagged(declared_priv, declared_feed)
    assert _outcome(runner.check_privilege, spec, policy) == _outcome(
        p8_runner._check_privilege, spec, policy
    )


def _perturbations(listed: ListedEpisode) -> list[tuple[ListedEpisode, str, Any]]:
    """Listed rows that must (or must not) trip the start check, with the pad/offset to pass."""
    return [
        (listed, listed.pad, None),
        (replace(listed, t0_model_s=listed.t0_model_s + 1e-12), listed.pad, None),
        (replace(listed, init_z_m=float(np.nextafter(listed.init_z_m, 9.0))), listed.pad, None),
        (replace(listed, init_x_m=listed.init_x_m + 1e-9), listed.pad, None),
        (replace(listed, realization_seed=listed.realization_seed + 1), listed.pad, None),
        (replace(listed, heading_deg=listed.heading_deg + 45.0), listed.pad, None),
        (listed, "cg", None),
        (listed, listed.pad, (0.123, 0.0, 0.0)),
    ]


def test_check_start_equals_p8(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    env = p8_adapter.make_env(cfgs, tiny_list[0])
    n_raised = 0
    try:
        for listed in tiny_list:
            pad, offset = p8_adapter.set_motion(env, cfgs, listed)
            _, info = env.reset(seed=listed.episode_seed)
            for row, eff_pad, eff_offset in _perturbations(listed):
                off = offset if eff_offset is None else eff_offset
                for check_t0 in (True, False):
                    ours = _outcome(
                        runner.check_start, env, info, row, eff_pad, off, check_t0=check_t0
                    )
                    theirs = _outcome(
                        p8_runner._check_start, env, info, row, eff_pad, off, check_t0=check_t0
                    )
                    assert ours == theirs, (listed.index, row, eff_pad)
                    n_raised += ours[0] == "EpisodeListMismatchError"
            assert pad == listed.pad
    finally:
        env.close()
    assert n_raised > 0


def test_episode_row_equals_p8(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    """Same finished env, same actions: our row and P8's ``_episode_row`` are the same text."""
    spec = p8_adapter.controller_spec("pid_feedforward")
    policy = spec.build(cfgs)
    rule = p8_adapter.touchdown_rule(cfgs)
    stride = int(cfgs.landing.pyb_steps_per_ctrl)
    env = p8_adapter.make_env(cfgs, tiny_list[0])
    try:
        for listed in tiny_list:
            pad, _ = p8_adapter.set_motion(env, cfgs, listed)
            obs, _ = env.reset(seed=listed.episode_seed)
            policy.reset(listed.episode_seed, None)
            actions = []
            for _ in range(400):
                action = np.asarray(policy.act(obs), dtype=np.float64)
                actions.append(np.clip(action, -1.0, 1.0))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            contact = env.record.contact_record
            truth = p8_adapter.td_in_quiescent_window(
                p8_adapter.PrivilegedContext.from_env(env),
                rule,
                None if contact is None else float(contact.t_episode_s),
                stride,
            )
            ours = runner.episode_row(listed, pad, spec, env, actions, truth)
            theirs = p8_runner._episode_row(listed, pad, spec, env, actions, truth)
            assert tuple(ours) == tuple(theirs) == p8_adapter.EPISODE_COLUMNS
            assert p8_adapter.episode_rows_csv([ours]) == p8_adapter.episode_rows_csv([theirs])
            for key in ours:
                assert type(ours[key]) is type(theirs[key]), key
            # One action only: jerk is NaN in both.
            short = runner.episode_row(listed, pad, spec, env, actions[:1], truth)
            assert math.isnan(short["action_jerk_mean"])
            assert math.isnan(
                p8_runner._episode_row(listed, pad, spec, env, actions[:1], truth)[
                    "action_jerk_mean"
                ]
            )
    finally:
        env.close()


@pytest.mark.parametrize("name", ["pid_feedforward", "oracle_gated", "pid_track_descend"])
def test_run_chunk_loop_equals_p8_run_list(
    name: str, tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    """The loop as a whole: P8's ``run_list`` and ours (render full) give the same text.

    ``oracle_gated`` covers the privileged-context path.
    """
    spec = p8_adapter.controller_spec(name)
    theirs = p8_runner.run_list(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list))
    ours = runner.run_list_vision(
        tiny_list, spec, cfgs, render="full", workers=1, chunk=len(tiny_list)
    )
    assert _p8_csv(ours) == p8_adapter.episode_rows_csv(theirs)
    assert all(row["privileged"] is (name == "oracle_gated") for row in ours)


# --------------------------------------------------------------------------- (b) workers, chunk


def test_rows_do_not_depend_on_workers_or_chunk(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = p8_adapter.controller_spec("pid_feedforward")
    one = runner.run_list_vision(tiny_list, spec, cfgs, workers=1, chunk=1)
    three = runner.run_list_vision(tiny_list, spec, cfgs, workers=3, chunk=2)
    whole = runner.run_list_vision(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list))
    assert runner.vision_rows_csv(one) == runner.vision_rows_csv(three)
    assert runner.vision_rows_csv(one) == runner.vision_rows_csv(whole)
    assert [row["index"] for row in three] == [row.index for row in tiny_list]


def test_student_rows_do_not_depend_on_workers_or_chunk(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=0)
    one = runner.run_list_vision(tiny_list, spec, cfgs, workers=1, chunk=1)
    three = runner.run_list_vision(tiny_list, spec, cfgs, workers=3, chunk=2)
    assert runner.vision_rows_csv(one) == runner.vision_rows_csv(three)


# --------------------------------------------------------------------------- (c) render modes


def test_render_off_camera_full_identical_rows(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = p8_adapter.controller_spec("pid_feedforward")
    texts = {
        mode: _p8_csv(runner.run_list_vision(tiny_list, spec, cfgs, render=mode, workers=1))
        for mode in runner.RENDER_MODES
    }
    assert texts["off"] == texts["camera"] == texts["full"]


def test_render_cfg_column(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    spec = p8_adapter.controller_spec("pid_feedforward")
    full = runner.run_list_vision(tiny_list[:1], spec, cfgs, render="full", workers=1)
    off = runner.run_list_vision(tiny_list[:1], spec, cfgs, render="off", workers=1)
    assert full[0]["render_cfg_sha256"] == p8_adapter.sha256_file(
        p8_adapter.REPO_ROOT / "configs" / "render" / "camera.yaml"
    )
    assert off[0]["render_cfg_sha256"] == ""
    assert full[0]["arm"] == "pid_feedforward" and full[0]["latency_steps"] == 0


# --------------------------------------------------------------------------- render evidence


class _RenderSpy:
    """Counts every TinyRenderer call and records the step of every timed frame (in-process)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pybullet as pyb

        self.camera_calls = 0
        self.timed_steps: list[int] = []
        real_image = pyb.getCameraImage
        real_timed = runner._timed_frame

        def image(*args: Any, **kwargs: Any) -> Any:
            self.camera_calls += 1
            return real_image(*args, **kwargs)

        def timed(env: Any, sc: Any, cfg: Any, step: int) -> Any:
            self.timed_steps.append(step)
            return real_timed(env, sc, cfg, step)

        monkeypatch.setattr(pyb, "getCameraImage", image)
        monkeypatch.setattr(runner, "_timed_frame", timed)

    def timed_by_episode(self) -> list[set[int]]:
        episodes: list[set[int]] = []
        for step in self.timed_steps:
            if step == 0:
                episodes.append(set())
            episodes[-1].add(step)
        return episodes


@pytest.mark.parametrize("mode", runner.RENDER_MODES)
def test_frames_actually_rendered_per_step(
    mode: str,
    tiny_list: list[ListedEpisode],
    cfgs: EvalConfigs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One TinyRenderer frame after the reset and after every step (terminal included)."""
    spy = _RenderSpy(monkeypatch)
    spec = p8_adapter.controller_spec("pid_feedforward")
    episodes = tiny_list[:2]  # static, then frigate: a reset between them
    rows = runner.run_list_vision(episodes, spec, cfgs, render=mode, workers=1, chunk=2)
    steps = [int(r["steps"]) for r in rows]
    if mode == "off":
        assert spy.camera_calls == 0 and spy.timed_steps == []
        assert all(r["frames_rendered"] == 0 and r["timed_checks"] == 0 for r in rows)
        return
    assert spy.camera_calls == sum(n + 1 for n in steps)
    assert [r["frames_rendered"] for r in rows] == [n + 1 for n in steps]
    every = runner.DEFAULT_TIMING_EVERY
    expected = [{0, n} | set(range(every, n + 1, every)) for n in steps]
    assert spy.timed_by_episode() == expected
    assert [r["timed_checks"] for r in rows] == [len(e) for e in expected]


def test_negative_control_no_op_render_trips_the_frame_count(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If rendering were skipped for a teacher (frames dropped anyway), the run must fail."""

    def no_frame(self: Any, step: int, timed: bool) -> None:
        del self, step, timed

    monkeypatch.setattr(runner._Renderer, "frame", no_frame)
    spec = p8_adapter.controller_spec("pid_feedforward")
    with pytest.raises(runner.FrameCountError, match="0 frames"):
        runner.run_list_vision(tiny_list[:1], spec, cfgs, render="full", workers=1)
    # With render off, no frames is exactly what is expected.
    rows = runner.run_list_vision(tiny_list[:1], spec, cfgs, render="off", workers=1)
    assert rows[0]["frames_rendered"] == 0


def test_render_evidence_columns_are_not_p8_columns(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    assert not set(runner.RENDER_EVIDENCE_COLUMNS) & set(p8_adapter.EPISODE_COLUMNS)
    assert runner.VISION_EPISODE_COLUMNS[-2:] == runner.RENDER_EVIDENCE_COLUMNS
    rows = runner.run_list_vision(
        tiny_list[:1], p8_adapter.controller_spec("pid_feedforward"), cfgs, workers=1
    )
    assert tuple(runner.p8_rows(rows)[0]) == p8_adapter.EPISODE_COLUMNS


# --------------------------------------------------------------------------- (d) start check


def test_every_reset_is_checked_against_the_list(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = p8_adapter.controller_spec("pid_track_descend")
    mismatch = p8_adapter.EpisodeListMismatchError
    bad_t0 = [tiny_list[0], replace(tiny_list[1], t0_model_s=tiny_list[1].t0_model_s + 1e-12)]
    with pytest.raises(mismatch, match="t0"):
        runner.run_list_vision(bad_t0, spec, cfgs, workers=1)
    bad_init = [replace(tiny_list[2], init_z_m=np.nextafter(tiny_list[2].init_z_m, 9.0))]
    with pytest.raises(mismatch, match="init"):
        runner.run_list_vision(bad_init, spec, cfgs, workers=1)
    bad_seed = [replace(tiny_list[3], episode_seed=tiny_list[3].episode_seed + 1)]
    with pytest.raises(mismatch):
        runner.run_list_vision(bad_seed, spec, cfgs, workers=1)
    # Through the student path and the pool too.
    student = p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=0)
    with pytest.raises(mismatch, match="t0"):
        runner.run_list_vision(bad_t0, student, cfgs, workers=2, chunk=1)


# --------------------------------------------------------------------------- timing


def test_timing_assertion_fires_on_a_mismatched_step(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    from ila.render.config import load_render_config

    env = p8_adapter.make_env(cfgs, tiny_list[1])
    try:
        env.reset(seed=tiny_list[1].episode_seed)
        sc = scene.attach(env)
        cfg = load_render_config()
        assert isinstance(runner._timed_frame(env, sc, cfg, 0), Frame)
        with pytest.raises(runner.RenderTimingError):
            runner._timed_frame(env, sc, cfg, 1)  # the frame is of step 0, not 1
        env.step(np.zeros(3))
        assert isinstance(runner._timed_frame(env, sc, cfg, 1), Frame)
        assert isinstance(runner._timed_frame(env, None, cfg, 1), Frame)  # camera mode
        with pytest.raises(runner.RenderTimingError):
            runner._timed_frame(env, None, cfg, 0)
    finally:
        env.close()


def test_timing_every_step(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    spec = p8_adapter.controller_spec("pid_feedforward")
    every = runner.run_list_vision(tiny_list[:2], spec, cfgs, workers=1, timing_every=1)
    sparse = runner.run_list_vision(tiny_list[:2], spec, cfgs, workers=1)
    # Checking every frame changes nothing but the count of checks (render evidence).
    assert _p8_csv(every) == _p8_csv(sparse)
    assert [r["frames_rendered"] for r in every] == [r["frames_rendered"] for r in sparse]
    assert all(r["timed_checks"] == r["frames_rendered"] == r["steps"] + 1 for r in every)
    assert all(r["timed_checks"] < r["frames_rendered"] for r in sparse)


# --------------------------------------------------------------------------- (e) obligation 1


_FORBIDDEN_FIELD_WORDS = ("substep", "step_count", "physics", "meta", "view", "projection")


def _annotation_text(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        return " ".join(str(f.type) for f in dataclasses.fields(obj))
    if isinstance(obj, type) and hasattr(obj, "_fields"):  # NamedTuple
        return " ".join(str(v) for v in getattr(obj, "__annotations__", {}).values())
    if callable(obj):
        try:
            return str(inspect.signature(obj).return_annotation)
        except (TypeError, ValueError):
            return ""
    return ""


@pytest.mark.parametrize("module", [runner, batched])
def test_public_types_carry_no_render_meta_or_step_count(module: Any) -> None:
    for name in module.__all__:
        obj = getattr(module, name)
        assert "RenderMeta" not in _annotation_text(obj), f"{module.__name__}.{name}"
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                assert not any(w in f.name.lower() for w in _FORBIDDEN_FIELD_WORDS), (name, f.name)
    assert not any(any(w in c.lower() for w in _FORBIDDEN_FIELD_WORDS) for c in runner.P9_COLUMNS)
    # Explicit allowlist (leak audit RISK 1): the noise key is the only carrier of the clock
    # and the episode seed, and a slot request carries nothing else; where each field may be
    # read is pinned by tests/test_step_guard.py.
    assert batched.NoiseKey._fields == ("run_seed", "episode_seed", "step")
    assert tuple(f.name for f in dataclasses.fields(batched.SlotRequest)) == (
        "slot",
        "key",
        "images",
        "proprio",
    )
    # The only P8 column with "step" in it is P8's control-step count of the episode record.
    steps = [c for c in runner.VISION_EPISODE_COLUMNS if "step" in c.lower()]
    assert steps == ["steps", "latency_steps"]


def _walk(value: Any) -> list[Any]:
    out = [value]
    if isinstance(value, dict):
        for v in value.values():
            out += _walk(v)
    elif isinstance(value, list | tuple):
        for v in value:
            out += _walk(v)
    return out


def test_rows_hold_no_render_meta_or_frames(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    student = p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=0)
    rows = runner.run_list_vision(tiny_list[:2], student, cfgs, workers=1)
    for row in rows:
        assert tuple(row) == runner.VISION_EPISODE_COLUMNS
        for value in _walk(row):
            assert not isinstance(value, RenderMeta | Frame | np.ndarray), type(value)
        for value in row.values():  # scalar cells only: nothing nested can hide a meta
            assert value is None or isinstance(value, str | int | float | bool), type(value)


# --------------------------------------------------------------------------- (f) obligation 2


class _SpyStudent:
    """Records exactly what the runner hands a student, and acts like ``pixel_action``."""

    privileged = False
    calls: list[tuple[Any, ...]] = []
    resets: list[tuple[int, Any]] = []

    def reset(self, seed: int, context: Any = None) -> None:
        _SpyStudent.resets.append((seed, context))

    def act_vision(self, image_stack: Any, proprio: Any) -> Any:  # exactly two positionals
        _SpyStudent.calls.append((image_stack, proprio))
        return pixel_action(image_stack, proprio)

    def act(self, obs: Any) -> Any:
        raise AssertionError(f"the runner called act() on a student with {np.shape(obs)}")


def _build_spy(cfgs: EvalConfigs) -> _SpyStudent:
    del cfgs
    return _SpyStudent()


def test_student_receives_only_contract_input_and_pixels(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    contract = load_contract()
    episodes = tiny_list[:2]  # static, then a frigate realization: a hull switch
    _SpyStudent.calls, _SpyStudent.resets = [], []
    spec = p8_adapter.callable_spec("spy", _build_spy, run_seed=0)
    rows = runner.run_list_vision(episodes, spec, cfgs, workers=1, chunk=len(episodes))
    assert _SpyStudent.resets == [(e.episode_seed, None) for e in episodes]
    assert len(_SpyStudent.calls) == sum(int(r["steps"]) for r in rows)
    start = 0
    for listed, row in zip(episodes, rows, strict=True):
        n = int(row["steps"])
        expected = replay_inputs(listed, cfgs, n)
        assert len(expected) == n
        for (images, proprio), (exp_images, exp_proprio) in zip(
            _SpyStudent.calls[start : start + n], expected, strict=True
        ):
            assert images.dtype == np.uint8 and images.shape == (contract.n_obs_steps, 96, 96, 3)
            assert proprio.dtype == np.float32
            assert proprio.shape == (contract.n_obs_steps, contract.proprio_dim)
            assert not images.flags.writeable and not proprio.flags.writeable  # StudentInput's
            assert images.tobytes() == exp_images.tobytes()
            assert proprio.tobytes() == exp_proprio.tobytes()
        start += n


def test_student_needs_full_render_and_no_privilege(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = p8_adapter.callable_spec("pixel", build_pixel_student, run_seed=0)
    for mode in ("off", "camera"):
        with pytest.raises(ValueError, match="render 'full'"):
            runner.run_list_vision(tiny_list[:1], spec, cfgs, render=mode, workers=1)
    privileged = p8_adapter.PolicySpec("pixel", True, build_pixel_student)
    with pytest.raises(ValueError, match="privileged"):
        runner.run_list_vision(tiny_list[:1], privileged, cfgs, workers=1)


# --------------------------------------------------------------------------- (g) feeds


def test_feed_specs_are_refused(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    feed = p8_adapter.PolicySpec("feed", False, build_pixel_student, needs_motion_feed=True)
    with pytest.raises(runner.MotionFeedRefusedError):
        runner.run_list_vision(tiny_list[1:2], feed, cfgs, workers=1)
    with pytest.raises(runner.MotionFeedRefusedError):
        runner.run_list_vision(
            tiny_list[1:2], p8_adapter.controller_spec("gated_forecast"), cfgs, workers=2
        )
    task = runner.VisionTask(tiny_list[1:2], feed, cfgs, runner.RowTags("feed"))
    with pytest.raises(runner.MotionFeedRefusedError):
        runner.run_chunk_vision(task)


# --------------------------------------------------------------------------- PPO (checkpoint)


@pytest.mark.checkpoints
def test_ppo_teacher_reproduces_committed_rows(cfgs: EvalConfigs) -> None:
    """Two listed id SS5 episodes, P8's SB3 path, render full: P8's e07 rows exactly."""
    from functools import partial

    episodes = [r for r in p8_adapter.read_list(p8_adapter.list_path("id")) if r.ss == "SS5"][:2]
    spec = p8_adapter.callable_spec("ppo", partial(p8_adapter.teacher, "ppo", 0), run_seed=0)
    rows = runner.run_list_vision(episodes, spec, cfgs, workers=1)
    result = p8_adapter.compare_to_reference(
        runner.p8_rows(rows), p8_adapter.P8_ROOT / "results/e07/matrix/episodes.csv", pad="aft"
    )
    assert result["all_identical"], result
    assert result["methods"]["ppo"]["n_compared"] == 2
