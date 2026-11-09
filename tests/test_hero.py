"""README hero GIFs (``ila.viz.hero``): picks, composition, timing, manifest, physics invariance.

Fast tests use synthetic frames. The scripted-action invariance test flies a ``train``-pool
episode (P1-D2), never a frozen-list row. The one full re-flight needs P8's PPO checkpoint.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageSequence

import render_support as rsup
from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.render import scene as render_scene
from ila.viz import gifs, hero

MANIFEST = hero.SOURCE_MANIFEST


def _pick(category: str = "a_all_four_succeed", outcome: str = "success") -> gifs.Pick:
    return gifs.Pick(category, "SS5", 2, 953709142, 402, dict.fromkeys(gifs.PANELS, outcome))


def _column(panel: str, n_steps: int, stride: int, outcome: str, seed: int = 0) -> hero.Column:
    rng = np.random.default_rng(seed)
    steps = list(range(0, n_steps, stride))
    h, w = hero.CAMERA.height_px, hero.CAMERA.width_px
    return hero.Column(
        panel=panel,
        outcome=outcome,
        n_steps=n_steps,
        steps=steps,
        outside=[np.full((h, w, 3), 40 + s % 50, dtype=np.uint8) for s in steps],
        onboard=[rng.integers(0, 255, (96, 96, 3), dtype=np.uint8) for _ in steps],
    )


def _badge_pixels(img: Image.Image, col: int, rgb: tuple[int, int, int]) -> int:
    """Count ``rgb`` pixels in a column's label bar."""
    w = hero.CAMERA.width_px
    bar = np.asarray(img)[hero.HEADER_H : hero.HEADER_H + hero.LABEL_H, col * w : (col + 1) * w]
    return int(np.all(bar == np.asarray(rgb, dtype=np.uint8), axis=2).sum())


# --------------------------------------------------------------------------- picks


def test_picks_are_the_committed_gallery_manifest() -> None:
    picks, sha, record = hero.load_picks(MANIFEST)
    assert sha == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert list(picks) == [c for c in gifs.CATEGORIES if record["picks"][c] is not None]
    for cat, p in picks.items():
        src = record["picks"][cat]
        assert (p.category, p.ss, p.index, p.episode_seed, p.position) == (
            cat,
            src["ss"],
            src["index"],
            src["episode_seed"],
            src["list_position"],
        )
        assert dict(p.outcomes) == src["outcomes"]


def test_hero_has_no_selection_logic() -> None:
    src = inspect.getsource(hero)
    for name in ("select_episodes", "_qualifies", "run_gallery"):
        assert name not in src, name


def test_check_picks_against_committed_rows_and_list() -> None:
    picks, _, record = hero.load_picks(MANIFEST)
    rows, sources = gifs.committed_panel_rows(hero.ROWS_ROOT)
    listed = ev_arms.list_episodes(gifs.LIST_NAME)
    hero.check_picks(picks, record, listed, rows, sources)  # passes on the committed state
    bad = dict(picks)
    p = bad["a_all_four_succeed"]
    bad["a_all_four_succeed"] = gifs.Pick(
        p.category, p.ss, p.index, p.episode_seed, p.position, {**p.outcomes, "dp": "crash"}
    )
    with pytest.raises(AssertionError, match="committed outcome"):
        hero.check_picks(bad, record, listed, rows, sources)
    with pytest.raises(AssertionError, match="committed rows changed"):
        hero.check_picks(picks, record, listed, rows, {**sources, "x": "0"})
    moved = {"a": gifs.Pick("a_all_four_succeed", "SS5", 2, 953709142, 401, p.outcomes)}
    with pytest.raises(AssertionError, match="list position"):
        hero.check_picks(moved, record, listed, rows, sources)


# --------------------------------------------------------------------------- composition


def test_header_name_and_labels() -> None:
    pick = _pick()
    assert hero.header_text(pick) == (
        "(a) all four panels succeed · id SS5 episode 2 · hand-picked by stated rule · "
        "simulation (PyBullet, TinyRenderer)"
    )
    assert hero.gif_name(pick) == "hero_a_all_four_succeed__id_SS5_ep2.gif"
    assert set(hero.TITLES) == set(gifs.CATEGORIES)
    assert set(hero.ARM_LABELS) == set(gifs.PANELS)
    assert hero._font(16).size >= 16 and hero._font(18, True).size >= 16


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_durations_track_model_scale_real_time(stride: int) -> None:
    n = 50
    d = hero.durations_ms(n, stride, hold_ms=0)
    assert all(x % 10 == 0 and x > 0 for x in d)
    assert abs(sum(d) - n * stride * 1000.0 / hero.CTRL_HZ) <= 5.0
    assert hero.durations_ms(n, stride)[-1] == d[-1] + hero.HOLD_MS


def test_compose_holds_finished_panels_and_badges_from_committed_outcome() -> None:
    s = 2
    cols = [
        _column("teacher", 9, s, "success", 0),  # frames at 0..8 -> 5 frames
        _column("bc", 14, s, "off_pad", 1),  # 7 frames
        _column("bc_dagger", 4, s, "hard_landing", 2),  # 2 frames
        _column("dp", 13, s, "success", 3),  # 7 frames
    ]
    frames = hero.compose(_pick(), cols, s)
    assert len(frames) == hero.n_composite_frames(cols) == 7 + 1
    w, h = frames[0].size
    assert (w, h) == (4 * hero.CAMERA.width_px, hero._layout(cols)[3])
    # No badge at t = 0; every badge on the final (held) frame, in the outcome's colour.
    for k, c in enumerate(cols):
        rgb = hero.SUCCESS_RGB if c.outcome == "success" else hero.FAILURE_RGB
        assert _badge_pixels(frames[0], k, rgb) == 0
        assert _badge_pixels(frames[-1], k, rgb) > 100
    # bc_dagger (4 steps) is over at frame 2 (t = 4 steps); bc (14 steps) is not.
    assert _badge_pixels(frames[2], 2, hero.FAILURE_RGB) > 100
    assert _badge_pixels(frames[2], 1, hero.FAILURE_RGB) == 0
    # A finished panel holds its last outside frame (inside the 3 px column separators).
    oh = hero.CAMERA.height_px
    y0 = hero.HEADER_H + hero.LABEL_H
    view = np.asarray(frames[5])[y0 : y0 + oh, 2 * w // 4 : 3 * w // 4]
    assert np.array_equal(view[:, 2:-2], np.asarray(cols[2].outside[-1])[:, 2:-2])


def test_encode_is_deterministic_and_real_time() -> None:
    s = 2
    cols = [_column(p, 10 + 2 * k, s, "success", k) for k, p in enumerate(gifs.PANELS)]
    frames = hero.compose(_pick(), cols, s)
    durations = hero.durations_ms(len(frames), s)
    a = hero.encode_gif(frames, durations)
    b = hero.encode_gif(frames, durations)
    assert a == b and a[:6] == b"GIF89a"
    with Image.open(io.BytesIO(a)) as im:
        got = [int(f.info["duration"]) for f in ImageSequence.Iterator(im)]
    assert got == durations


def test_column_of_subsamples_and_checks_coverage() -> None:
    e = ev_arms.list_episodes("id")[0]
    n = 11
    frames = [np.full((96, 96, 3), k, dtype=np.uint8) for k in range(n)]
    outside = [np.full((4, 4, 3), k, dtype=np.uint8) for k in range(n)]
    res = hero.HeroResult(
        gifs.PanelResult(gifs.Panel("bc", e), {"steps": n, "outcome": "success"}, frames),
        list(range(n)),
        outside,
    )
    col = hero.column_of(res, 2)
    assert list(col.steps) == [0, 2, 4, 6, 8, 10]
    assert [int(f[0, 0, 0]) for f in col.onboard] == list(col.steps)
    assert [int(f[0, 0, 0]) for f in col.outside] == list(col.steps)
    short = hero.HeroResult(res.result, list(range(n - 3)), outside[: n - 3])
    with pytest.raises(ValueError):
        hero.column_of(short, 2)


# --------------------------------------------------------------------------- the job


def _fake_flights(
    panels: Sequence[gifs.Panel], workers: int, stride: int
) -> Iterator[hero.HeroResult]:
    del workers
    for k, p in enumerate(panels):
        n = 7 + k
        steps = list(range(0, n, stride))
        h, w = hero.CAMERA.height_px, hero.CAMERA.width_px
        yield hero.HeroResult(
            gifs.PanelResult(
                p,
                {"steps": n, "outcome": "success"},
                [np.zeros((96, 96, 3), dtype=np.uint8)] * n,
            ),
            steps,
            [np.full((h, w, 3), 30 + s, dtype=np.uint8) for s in steps],
        )


def test_manifest_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema only: flights and row checks are stubbed (the checks are tested elsewhere)."""
    monkeypatch.setattr(hero, "_fly_all", _fake_flights)
    checked: list[str] = []

    def check(r: gifs.PanelResult, committed: object, job: object) -> None:
        del committed, job
        checked.append(r.panel.panel)

    monkeypatch.setattr(gifs, "check_panel", check)
    rec = hero.run_hero(
        out_dir=tmp_path, only="a_all_four_succeed", panels=["teacher"], require_clean=False
    )
    assert checked == ["teacher"]
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(rec))
    for key in (
        "source_manifest",
        "picks",
        "camera",
        "frames",
        "gifs",
        "total_bytes",
        "ila_commit",
        "dirty",
        "p8_commit",
        "committed_row_sources",
        "rule_sha256",
        "smoke",
    ):
        assert key in rec, key
    src = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert rec["picks"] == {"a_all_four_succeed": src["picks"]["a_all_four_succeed"]}
    assert rec["source_manifest"]["sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert rec["smoke"] is True and len(rec["ila_commit"]) == 40 and len(rec["p8_commit"]) == 40
    assert on_disk["camera"]["target_m"] == list(hero.CAMERA.target_m)
    (g,) = rec["gifs"]
    data = (tmp_path / g["file"]).read_bytes()
    assert g["sha256"] == hashlib.sha256(data).hexdigest() and g["bytes"] == len(data)
    assert [r["panel"] for r in g["reflight"]] == ["teacher"]
    assert all(r["equal"] for r in g["reflight"])


def test_full_run_refuses_a_dirty_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ila.eval import matrix as ev_matrix

    def dirty(exclude: Path | None = None) -> list[str]:
        del exclude
        return [" M src/x.py"]

    monkeypatch.setattr(ev_matrix, "dirty_paths", dirty)
    with pytest.raises(ev_matrix.DirtyTreeError):
        hero.run_hero(out_dir=tmp_path)
    assert not any(tmp_path.iterdir())


def test_env_tap_returns_the_same_env_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    calls: list[tuple[Any, Any]] = []

    def fake(cfgs: Any, listed: Any) -> Any:
        calls.append((cfgs, listed))
        return sentinel

    monkeypatch.setattr(p8_adapter, "make_env", fake)
    with hero._env_tap() as built:
        assert p8_adapter.make_env(1, 2) is sentinel  # type: ignore[arg-type]
    assert built == [sentinel] and calls == [(1, 2)]
    assert p8_adapter.make_env is fake


def test_outside_camera_is_p8s() -> None:
    from rld.viz.gifs import Camera

    assert p8_adapter.OutsideCamera is Camera
    assert isinstance(hero.CAMERA, Camera)


# --------------------------------------------------------------------------- physics


class _Null:
    needs_frames = False

    def begin(self, listed: Any, context: Any) -> None:
        del listed, context

    def act(self, step: int, obs: Any, frame: Any) -> np.ndarray:
        del step, obs, frame
        return np.zeros(3)


@pytest.mark.slow
@pytest.mark.pybullet
def test_outside_grab_changes_no_physics_and_no_onboard_pixel() -> None:
    """A scripted descent flown twice, with and without outside grabs, is bit-identical."""
    cfgs = p8_adapter.load_eval_configs()
    e = rsup.train_listed("SS5", 0)
    action = np.array([0.1, -0.05, -0.4])

    def fly(grab: bool) -> tuple[list[bytes], list[bytes], int]:
        env = p8_adapter.make_env(cfgs, e)
        try:
            p8_adapter.set_motion(env, cfgs, e)
            obs, _ = env.reset(seed=e.episode_seed)
            sc = render_scene.attach(env)
            rec = hero._OutsideRecorder(_Null(), [env], hero.CAMERA, 1)
            rec.begin(e, None)
            obs_log, pix = [obs.tobytes()], [sc.render().rgb.tobytes()]
            for k in range(40):
                if grab:
                    rec.act(k, obs, None)
                obs, _, term, trunc, _ = env.step(action)
                obs_log.append(obs.tobytes())
                pix.append(sc.render().rgb.tobytes())
                if term or trunc:
                    break
            return obs_log, pix, len(rec.outside)
        finally:
            env.close()

    o0, p0, n0 = fly(False)
    o1, p1, n1 = fly(True)
    assert n0 == 0 and n1 == len(o1) - 1 > 10
    assert o0 == o1
    assert p0 == p1


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.checkpoints
def test_teacher_reflight_with_outside_view_matches_committed() -> None:
    """Pick (a)'s teacher panel, flown with the outside view, equals P8's committed row."""
    picks, _, _ = hero.load_picks(MANIFEST)
    p = picks["a_all_four_succeed"]
    e = ev_arms.list_episodes(gifs.LIST_NAME)[p.position]
    res = hero.fly_hero_panel(gifs.Panel("teacher", e), stride=3)
    gifs.check_panel(res.result, "", None)
    n = int(res.result.row["steps"])
    assert res.steps == list(range(0, n, 3)) and len(res.result.frames) == n
    assert res.outside[0].shape == (hero.CAMERA.height_px, hero.CAMERA.width_px, 3)
    assert p8_adapter.make_env.__module__ == "ila.p8_adapter"
