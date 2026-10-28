"""Phase 3: demo collection through the runner (plan §1.4-§1.6, il-conventions alignment).

Small real collections (2-3 ``train`` episodes per teacher) are flown once per module into a
temporary data root and checked:

* every episode validates (``last_action[t] == action_exec[t-1]``, by-name proprio/priv,
  ``t_model_s``), and every label equals the teacher re-run in order from ``reset``;
* open-loop replay of the stored ``action_exec`` through the same runner reproduces every stored
  observation and frame byte for byte (``image[t]``/``obs[t]`` are the state before
  ``action_exec[t]``, and ``action_exec[t]`` is what produced ``obs[t+1]``);
* DART: the label is the clean teacher action, ``action_exec = clip(label + sigma z)`` with the
  episode's own noise; ``T_mix``: the balanced per-SS assignment decides the teacher, and
  ``ppo`` episodes equal the ``ppo_2000`` episodes at the same index;
* the file bytes do not depend on workers or chunking;
* an appearance-shift render is refused before anything flies (P1-D2 obligation 2).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.data import audit, collect
from ila.data import episodes as ep
from ila.data import mix as mix_module
from ila.data.dart import DartNoise
from ila.data.store import DemoStore
from ila.eval import runner
from ila.obs.split import ObsContract
from ila.render.config import APPEARANCE_SHIFTS, load_render_config

LIMITS = {"ppo_2000": 3, "pid_1000": 2, "mix_1000": 3, "dart_0.20": 2}


@pytest.fixture(scope="module")
def collections() -> dict[str, collect.Collection]:
    return collect.load_collections()


@pytest.fixture(scope="module")
def stores(
    collections: dict[str, collect.Collection], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("demos")
    out: dict[str, Path] = {}
    for name, limit in LIMITS.items():
        path = root / f"{name}.h5"
        collect.run_collection(
            collections[name],
            path,
            run_dir=root / "runs" / name,
            workers=1,
            limit=limit,
            log=lambda _line: None,
        )
        out[name] = path
    return out


# --------------------------------------------------------------------------- config (no physics)


def test_collections(collections: dict[str, collect.Collection]) -> None:
    assert list(collections) == [
        "ppo_2000",
        "pid_1000",
        "mix_1000",
        "dart_0.05",
        "dart_0.10",
        "dart_0.20",
    ]
    c = collections
    assert (c["ppo_2000"].split, c["ppo_2000"].n, c["ppo_2000"].policies) == (
        "demos_0-1999",
        2000,
        ("ppo",),
    )
    for name in ("pid_1000", "mix_1000", "dart_0.05", "dart_0.10", "dart_0.20"):
        assert c[name].split == "demos_0-999" and c[name].n == 1000
    assert c["pid_1000"].policies == ("pid_feedforward",)
    assert c["mix_1000"].policies == ("ppo", "gated")
    assert [c[f"dart_{s}"].dart_sigma for s in ("0.05", "0.10", "0.20")] == [0.05, 0.10, 0.20]
    assert all(c[n].teacher == "T_ppo" for n in ("ppo_2000", "dart_0.05"))
    assert all(v.generator_seed == 20261023 and v.stream == "demos" for v in c.values())


def test_mix_policies_balanced_per_ss(collections: dict[str, collect.Collection]) -> None:
    mix = collections["mix_1000"]
    rows = ep.read_split(mix.split)
    policies = collect.collection_policies(mix, rows)
    assert policies == mix_module.assign_rows(20261023, rows)
    assert Counter(policies) == {"ppo": 500, "gated": 500}
    for name in ("ppo_2000", "pid_1000", "dart_0.10"):
        c = collections[name]
        assert set(collect.collection_policies(c, ep.read_split(c.split))) == set(c.policies)
    with pytest.raises(ValueError):
        collect.collection_policies(mix, rows[:999])


def test_dart_noise_is_per_episode_and_shared_across_sigma() -> None:
    clean = np.array([0.1, -0.2, 0.3])
    a, b = DartNoise(0.05, 20261023, 7, 99), DartNoise(0.20, 20261023, 7, 99)
    other = DartNoise(0.05, 20261023, 8, 99)
    xa = [a.executed(clean) for _ in range(5)]
    xb = [b.executed(clean) for _ in range(5)]
    for u, v in zip(xa, xb, strict=True):  # same z: (exec - clean) / sigma equal
        np.testing.assert_allclose((u - clean) / 0.05, (v - clean) / 0.20, rtol=1e-9)
    assert not np.allclose(xa[0], other.executed(clean))
    again = DartNoise(0.05, 20261023, 7, 99)
    assert all(np.array_equal(u, again.executed(clean)) for u in xa)
    huge = DartNoise(50.0, 1, 0, 0)
    assert np.all(np.abs(huge.executed(clean)) <= 1.0)
    with pytest.raises(ValueError):
        DartNoise(0.0, 1, 0, 0)


@pytest.mark.parametrize("shift", APPEARANCE_SHIFTS)
def test_appearance_shift_refused(
    shift: str, collections: dict[str, collect.Collection], tmp_path: Path
) -> None:
    with pytest.raises(collect.AppearanceShiftRefusedError):
        collect.check_render_config(load_render_config(appearance_shift=shift))
    with pytest.raises(collect.AppearanceShiftRefusedError):
        collect.run_collection(
            collections["ppo_2000"],
            tmp_path / "x.h5",
            run_dir=tmp_path / "run",
            limit=1,
            appearance_shift=shift,
        )
    row = ep.read_split("demos_0-999")[0]
    task = collect.CollectTask(
        collections["ppo_2000"], ((0, row, "ppo"),), p8_adapter.load_eval_configs(), {}, shift
    )
    with pytest.raises(collect.AppearanceShiftRefusedError):
        collect.collect_chunk(task)
    assert not (tmp_path / "x.h5").exists()
    collect.check_render_config(load_render_config())


# --------------------------------------------------------------------------- collected stores


@pytest.mark.pybullet
def test_every_episode_validates_and_is_kept(stores: dict[str, Path]) -> None:
    for name, path in stores.items():
        with DemoStore(path) as s:
            assert s.episodes() == [f"ep_{j:05d}" for j in range(LIMITS[name])]
            for e in s.episodes():
                s.validate_episode(e)
                m = s.meta(e)
                assert m.outcome in {"crash", "off_pad", "hard_landing", "bounce", "success",
                                     "timeout"}  # fmt: skip
                assert m.camera_cfg_sha256 == runner.render_cfg_sha256("full", None)
                assert 0.0 <= m.clear_frac_max <= 1.0
        manifest = (path.with_suffix(".manifest.json")).read_text(encoding="utf-8")
        for key in ("p8_commit", "ila_commit", "camera_cfg_sha256", "file_sha256", "outcomes"):
            assert f'"{key}"' in manifest


@pytest.mark.pybullet
def test_labels_equal_teacher_in_order(stores: dict[str, Path]) -> None:
    for path in stores.values():
        result = audit.relabel_audit(path, n_steps=10_000)
        assert result["passed"] and result["max_abs_diff"] == 0.0, result


@pytest.mark.pybullet
def test_split_audit(stores: dict[str, Path]) -> None:
    for path in stores.values():
        assert audit.split_audit(path)["passed"]


class _Replay:
    """Open-loop replay of stored actions; records what it is shown."""

    needs_frames = True
    privileged = False
    needs_motion_feed = False

    def __init__(self, actions: dict[int, np.ndarray]) -> None:
        self.actions = actions
        self.seen: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._seed = -1

    def begin(self, listed: Any, context: Any) -> None:
        del context
        self._seed = int(listed.episode_seed)
        self.seen[self._seed] = []

    def act(self, step: int, obs: Any, frame: Any) -> np.ndarray:
        self.seen[self._seed].append((np.array(obs, np.float32), np.array(frame.rgb)))
        return self.actions[self._seed][step].astype(np.float64)


@pytest.mark.pybullet
def test_open_loop_replay_reproduces_obs_and_frames(stores: dict[str, Path]) -> None:
    """``action_exec[t]`` produced ``obs[t+1]``/``image[t+1]`` (ppo: float32 actions, exact)."""
    rows = ep.read_split("demos_0-1999")
    with DemoStore(stores["ppo_2000"]) as s:
        names = s.episodes()
        metas = [s.meta(e) for e in names]
        arrays = [s.audit_arrays(e) for e in names]
    replay = _Replay({m.episode_seed: a["action_exec"] for m, a in zip(metas, arrays, strict=True)})
    task = runner.VisionTask(
        episodes=[rows[m.demo_index] for m in metas],
        spec=p8_adapter.callable_spec("replay", lambda _c: replay, run_seed=0),  # type: ignore[arg-type,return-value]
        cfgs=p8_adapter.load_eval_configs(),
        tags=runner.RowTags(arm="replay"),
        render="full",
    )

    def make(c: Any, contract: ObsContract) -> tuple[Any, object]:
        return replay, replay

    out = runner.fly_chunk(task, make)
    for m, a, row in zip(metas, arrays, out, strict=True):
        seen = replay.seen[m.episode_seed]
        assert len(seen) == m.steps and row["outcome"] == m.outcome
        assert np.array_equal(np.stack([o for o, _ in seen]), a["teacher_obs"])
        assert np.array_equal(np.stack([f for _, f in seen]), a["images"])


@pytest.mark.pybullet
def test_dart_executes_noised_label(
    stores: dict[str, Path], collections: dict[str, collect.Collection]
) -> None:
    c = collections["dart_0.20"]
    with DemoStore(stores["dart_0.20"]) as s:
        for e in s.episodes():
            m = s.meta(e)
            assert (m.dart_sigma, m.noise_seed, m.teacher) == (0.20, c.dart_seed, "T_ppo")
            noise = DartNoise(0.20, c.dart_seed, m.demo_index, m.episode_seed)
            label = s.labels(e).astype(np.float64)
            expected = np.stack([noise.executed(x) for x in label]).astype(np.float32)
            assert np.array_equal(s.executed_actions(e), expected)
            assert not np.array_equal(s.executed_actions(e), np.clip(s.labels(e), -1, 1))


@pytest.mark.pybullet
def test_mix_follows_assignment_and_pairs_with_ppo(
    stores: dict[str, Path], collections: dict[str, collect.Collection]
) -> None:
    mix = collections["mix_1000"]
    policies = collect.collection_policies(mix, ep.read_split(mix.split))
    with DemoStore(stores["mix_1000"]) as s, DemoStore(stores["ppo_2000"]) as p:
        for e in s.episodes():
            m = s.meta(e)
            assert m.episode_teacher == policies[m.demo_index]
            if m.episode_teacher == "ppo":
                a, b = s.audit_arrays(e), p.audit_arrays(e)
                assert all(a[k].tobytes() == b[k].tobytes() for k in a)


@pytest.mark.pybullet
def test_bytes_independent_of_workers(
    stores: dict[str, Path], collections: dict[str, collect.Collection], tmp_path: Path
) -> None:
    path = tmp_path / "ppo.h5"
    collect.run_collection(
        collections["ppo_2000"],
        path,
        run_dir=tmp_path / "run",
        workers=2,
        chunk=1,
        limit=LIMITS["ppo_2000"],
        log=lambda _line: None,
    )
    assert path.read_bytes() == stores["ppo_2000"].read_bytes()


@pytest.mark.pybullet
def test_refly_audit(stores: dict[str, Path]) -> None:
    result = audit.refly_audit(stores["pid_1000"], n_states=5)
    assert result["passed"], result


def test_recorder_refuses_context_and_skips(collections: dict[str, collect.Collection]) -> None:
    row = ep.read_split("demos_0-999")[0]
    rec = collect.DemoRecorder(
        collections["pid_1000"],
        {"pid_feedforward": p8_adapter.teacher("pid_feedforward")},
        {(row.ss, int(row.index), int(row.episode_seed)): (0, "pid_feedforward")},
    )
    with pytest.raises(ValueError, match="context"):
        rec.begin(row, object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        rec.act(0, np.zeros(25, np.float32), None)
    with pytest.raises(ValueError, match="not built"):
        collect.DemoRecorder(dataclasses.replace(collections["mix_1000"]), {}, {})
    with pytest.raises(ValueError, match="not in the collection"):
        collect.DemoRecorder(
            collections["pid_1000"],
            {"pid_feedforward": p8_adapter.teacher("pid_feedforward")},
            {(row.ss, int(row.index), int(row.episode_seed)): (0, "gated")},
        )
