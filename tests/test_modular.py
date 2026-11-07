"""Phase 7: ``modular`` (estimator + frozen teacher) and the 11 ablations (P3-D1 §3).

``modular`` (CPU; a stand-in teacher unless marked ``checkpoints``):

* the estimator's ``forward`` has no clock parameter (leak RISK 2);
* the 11 privileged fields of the env observation set to NaN -> the action is bit-identical,
  both through ``ObsContract.modular_input`` and through the runner's ``StudentActor``;
* the teacher's observation is P8's layout: proprio of step t, the estimate, the true clock of
  step t; ``act_vision`` (no clock), a ``StudentInput`` or a raw observation raise;
* :class:`EstimatorDataset`: inputs equal the BC student path's, target = stored ``priv[t]``,
  identical batches on a store whose ``teacher_obs`` privileged/clock columns are poisoned;
* training writes the target normaliser into every checkpoint and the manifest.

Registry: every entry resolves; config SHA-256s equal the frozen table (and P3-D1 §3 via
``test_frozen_config_sha256_match_files_and_protocol``); kind/variant/tag as declared. Ablations:
each trains with its variant (``plus_clock`` 14-d, ``minus_last_action`` 10-d) and the student
built from its checkpoint flies the same variant; stage data and the normaliser are identical on
a store poisoned per variant (only ``plus_clock`` leaves ``time_fraction`` clean).

Estimation study: :func:`error_rows` on known errors, and the open- and closed-loop study on an
untrained (1-step) checkpoint.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

import ila.policies.modular as modular_mod
from ila.data.store import DemoStore
from ila.eval import runner
from ila.obs.split import CLOCK_FIELD, NEVER_STUDENT, load_contract
from ila.policies import registry
from ila.policies.modular import ModularEstimator, ModularStudent
from ila.render.camera import Frame
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import TrainConfig, sha256_file
from ila.train.dataset import SAMPLE_KEYS, BCWindowDataset, FrameTable
from ila.train.modular_eval import FIELDS, error_rows, open_loop, write_study
from ila.train.modular_targets import EstimatorDataset, true_priv
from ila.train.sampling import choose_episodes
from ila.train.select import training_teacher
from test_data_store import SCRAMBLED_ATTRS, synthetic_record, write_store

CONTRACT = load_contract()
N_EPISODES = 8


def _set(obs: np.ndarray, name: str, value: Any) -> None:
    offset, size = CONTRACT.layout[name]
    obs[..., offset : offset + size] = value


class StubTeacher:
    """Deterministic stand-in for ``T_ppo``: ``tanh(W obs)``; NaN anywhere -> NaN action."""

    privileged = False
    needs_motion_feed = False

    def __init__(self) -> None:  # noqa: D107
        self.w = np.random.default_rng(0).standard_normal((3, CONTRACT.obs_dim)) / 5.0
        self.seen: list[np.ndarray] = []
        self.resets: list[int] = []

    def reset(self, seed: int, context: Any = None) -> None:  # noqa: D102
        assert context is None
        self.resets.append(seed)

    def act(self, obs: Any) -> np.ndarray:  # noqa: D102
        x = np.asarray(obs, dtype=np.float32)
        assert x.shape == (CONTRACT.obs_dim,) and x.dtype == np.float32
        self.seen.append(x.copy())
        return np.tanh(self.w @ x.astype(np.float64)).astype(np.float32)


@pytest.fixture
def stub_teacher(monkeypatch: pytest.MonkeyPatch) -> StubTeacher:
    """Every ModularStudent built in this test (load_student included) gets the stub."""
    teacher = StubTeacher()
    monkeypatch.setattr(modular_mod, "build_frozen_teacher", lambda: teacher)
    return teacher


def _student(teacher: StubTeacher, seed: int = 0) -> ModularStudent:
    net = registry.build_net("modular", seed)
    assert isinstance(net, ModularEstimator)
    with torch.no_grad():  # a non-trivial output normaliser
        net.target_mean.copy_(torch.linspace(-1.0, 1.0, 11))
        net.target_std.copy_(torch.linspace(0.1, 2.0, 11))
    return ModularStudent(net, "cpu", teacher=teacher)


def _window(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, (CONTRACT.n_obs_steps, 96, 96, 3), dtype=np.uint8)
    obs = rng.standard_normal((CONTRACT.n_obs_steps, CONTRACT.obs_dim)).astype(np.float32)
    _set(obs, CLOCK_FIELD, np.array([[0.25], [0.5]], np.float32))
    return images, obs


# --------------------------------------------------------------------------- modular: contract


def test_estimator_forward_has_no_clock_parameter() -> None:
    params = [p for p in inspect.signature(ModularEstimator.forward).parameters if p != "self"]
    assert params == ["images", "proprio", "crop_generator"]
    assert not any("time" in p or "clock" in p or "step" in p for p in params)
    for name in ("estimate", "estimate_batch"):
        sig = inspect.signature(getattr(ModularStudent, name)).parameters
        assert list(sig)[1:] in (["image_stack", "proprio"], ["images", "proprio"]), name


def test_nan_privileged_fields_leave_the_action_bit_identical() -> None:
    teacher = StubTeacher()
    student = _student(teacher)
    images, obs = _window()
    student.reset(5)
    clean = student.act_modular(CONTRACT.modular_input(images, obs))
    clean_est = student.last_estimate
    poisoned = obs.copy()
    for name in sorted(NEVER_STUDENT):
        _set(poisoned, name, np.nan)
    assert np.isnan(true_priv(CONTRACT, poisoned)).all()
    student.reset(6)
    again = student.act_modular(CONTRACT.modular_input(images, poisoned))
    assert np.isfinite(clean).all()
    assert again.tobytes() == clean.tobytes()
    assert student.last_estimate is not None and clean_est is not None
    assert student.last_estimate.tobytes() == clean_est.tobytes()
    assert teacher.seen[0].tobytes() == teacher.seen[1].tobytes()


def test_nan_privileged_fields_through_the_runner_actor() -> None:
    """The runner's StudentActor (act_modular path), clean vs poisoned env observations."""
    actions = {}
    for label in ("clean", "poisoned"):
        teacher = StubTeacher()
        actor = runner.StudentActor(_student(teacher), CONTRACT)
        assert actor.modular
        listed = dataclasses.make_dataclass("L", [("episode_seed", int)])(7)
        actor.begin(listed, None)  # type: ignore[arg-type]
        out = []
        for t in range(4):
            images, obs = _window(t)
            o = obs[-1].copy()
            _set(o, CLOCK_FIELD, t / 10.0)
            if label == "poisoned":
                for name in sorted(NEVER_STUDENT):
                    _set(o, name, np.nan)
            out.append(actor.act(t, o, Frame(images[-1])))
        actions[label] = np.stack(out)
    assert np.isfinite(actions["clean"]).all()
    assert actions["clean"].tobytes() == actions["poisoned"].tobytes()


def test_inference_calls_count_estimator_calls_per_episode() -> None:
    """One estimator call per ``act_modular``; ``reset`` zeroes the counter."""
    student = _student(StubTeacher())
    images, obs = _window()
    student.reset(1)
    for _ in range(4):
        student.act_modular(CONTRACT.modular_input(images, obs))
    assert student.inference_calls == 4
    student.reset(2)
    assert student.inference_calls == 0


def test_teacher_observation_is_p8_layout_with_the_true_clock() -> None:
    teacher = StubTeacher()
    student = _student(teacher)
    images, obs = _window(2)
    mi = CONTRACT.modular_input(images, obs)
    student.reset(1)
    student.act_modular(mi)
    seen = teacher.seen[-1]
    proprio, priv, extras = CONTRACT.split_privileged(seen)
    assert proprio.tobytes() == CONTRACT.proprio(obs[-1]).tobytes()  # step t's proprio
    assert student.last_estimate is not None
    assert priv.tobytes() == student.last_estimate.tobytes()  # the estimate, not the truth
    assert not np.array_equal(priv, true_priv(CONTRACT, obs[-1]))
    assert extras.tolist() == [0.5]  # time_fraction of step t (the window's last entry)
    est = student.estimate(images, CONTRACT.proprio(obs))
    assert est.tobytes() == student.last_estimate.tobytes()
    assert teacher.resets == [1]


def test_modular_refuses_inputs_without_a_clock() -> None:
    student = _student(StubTeacher())
    images, obs = _window()
    si = CONTRACT.student_input(images, obs)
    with pytest.raises(TypeError, match="ModularInput"):
        student.act_vision(si.images, si.proprio)
    with pytest.raises(TypeError, match="ModularInput"):
        student.act(si)
    with pytest.raises(TypeError, match="ModularInput"):
        student.act(obs[-1])
    with pytest.raises(ValueError, match="privileged"):
        student.reset(1, context=object())
    assert runner.is_modular(student) and not student.privileged


def test_estimate_batch_matches_single() -> None:
    student = _student(StubTeacher())
    windows = [_window(s) for s in range(3)]
    images = np.stack([w[0] for w in windows])
    proprio = np.stack([CONTRACT.proprio(w[1]) for w in windows])
    batch = student.estimate_batch(images, proprio)
    single = np.stack([student.estimate(i, p) for i, p in zip(images, proprio, strict=True)])
    np.testing.assert_allclose(batch, single, rtol=1e-5, atol=1e-5)


@pytest.mark.checkpoints
def test_frozen_teacher_is_p8_ppo_with_frozen_vecnormalize() -> None:
    teacher = modular_mod.build_frozen_teacher()
    assert teacher.normalizer is not None and teacher.normalizer.training is False  # type: ignore[attr-defined]
    student = ModularStudent(registry.build_net("modular", 0), "cpu")  # type: ignore[arg-type]
    images, obs = _window()
    student.reset(3)
    a = student.act_modular(CONTRACT.modular_input(images, obs))
    poisoned = obs.copy()
    for name in sorted(NEVER_STUDENT):
        _set(poisoned, name, np.nan)
    b = student.act_modular(CONTRACT.modular_input(images, poisoned))
    assert a.shape == (3,) and a.tobytes() == b.tobytes()


# --------------------------------------------------------------------------- data


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    for collection in ("ppo_2000", "pid_1000", "mix_1000"):
        records = [synthetic_record(j, 5 + j % 4, seed=200 + j) for j in range(N_EPISODES)]
        write_store(root / "demos" / f"{collection}.h5", records)
    return root


def _poison_copy(src: Path, dst: Path, variant: str, *, is_modular: bool) -> Path:
    """Copy the stores and poison everything a student must not read.

    NaN the privileged fields (+ the clock unless ``plus_clock``) of every ``teacher_obs``, NaN
    ``t_model_s``, scramble the attributes. ``priv`` is kept only for ``modular`` (its target);
    for any other arm it is NaN too, so a dataset that reads it cannot pass.
    """
    import shutil

    shutil.copytree(src, dst)
    poison = sorted(NEVER_STUDENT | (set() if variant == "plus_clock" else {CLOCK_FIELD}))
    for path in (dst / "demos").glob("*.h5"):
        with h5py.File(path, "r+") as f:
            for name in f:
                g = f[name]
                obs = g["teacher_obs"][()]
                for field in poison:
                    _set(obs, field, np.nan)
                g["teacher_obs"][...] = obs
                g["t_model_s"][...] = np.nan
                if not is_modular:
                    g["priv"][...] = np.nan
                for key, value in SCRAMBLED_ATTRS.items():
                    if key != "outcome":  # the success filter reads it
                        g.attrs[key] = value
    return dst


def test_estimator_dataset_inputs_are_the_student_path_and_target_is_priv(
    data_root: Path, tmp_path: Path
) -> None:
    path = data_root / "demos" / "ppo_2000.h5"
    with DemoStore(path) as store:
        names = choose_episodes(store, 4, True).names
        est = EstimatorDataset.from_store(store, names, CONTRACT.n_obs_steps)
        bc = BCWindowDataset(FrameTable.from_store(store, names), CONTRACT.n_obs_steps)
        priv = np.concatenate([store.estimator_targets(n) for n in names])
    idx = np.arange(len(est), dtype=np.int64)
    a, b = est.batch(idx), bc.batch(idx)
    assert tuple(a) == SAMPLE_KEYS
    assert a["images"].tobytes() == b["images"].tobytes()
    assert a["proprio"].tobytes() == b["proprio"].tobytes()
    assert a["proprio"].shape[-1] == 13
    assert a["target"].tobytes() == priv.tobytes()  # priv[t] at the window's anchor t
    mean, std = est.target_stats()
    np.testing.assert_allclose(mean, priv.astype(np.float64).mean(0), rtol=1e-6)
    assert mean.shape == std.shape == (11,)
    poisoned = _poison_copy(data_root, tmp_path / "p", "default", is_modular=True)
    with DemoStore(poisoned / "demos" / "ppo_2000.h5") as store:
        again = EstimatorDataset.from_store(store, names, CONTRACT.n_obs_steps).batch(idx)
    for key in SAMPLE_KEYS:
        assert again[key].tobytes() == a[key].tobytes(), key


def _cfg(arm: str, **change: Any) -> TrainConfig:
    cfg = TrainConfig.load(registry.REGISTRY[arm].config)
    return dataclasses.replace(cfg, **{"batch_size": 4, "precision": "fp32", **change})


def _spec(cfg: TrainConfig, run_dir: Path, data_root: Path, **kw: Any) -> loop.RunSpec:
    base: dict[str, Any] = {
        "cfg": cfg,
        "run_seed": 1,
        "run_dir": run_dir,
        "data_root": data_root,
        "device": "cpu",
        "purpose": "test",
        "first_n_override": 4,
        "steps_override": 2,
        "checkpoint_every_override": 2,
        "heldout": loop.HeldoutSpec("ppo_2000", 4, 2),
        "log_every": 1,
        "prefetch": 0,
        "hash_store": False,
    }
    base.update(kw)
    return loop.RunSpec(**base)


@pytest.fixture(scope="module")
def modular_run(data_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.set_num_threads(1)
    run_dir = tmp_path_factory.mktemp("runs") / "modular"
    spec = _spec(_cfg("modular"), run_dir, data_root)
    result = loop.train_run(spec, loop.build_stages(spec))
    assert result.state == "done"
    return run_dir


def test_modular_training_writes_the_target_normaliser(data_root: Path, modular_run: Path) -> None:
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        names = choose_episodes(store, 4, True).names
        mean, std = EstimatorDataset.from_store(store, names, 2).target_stats()
    ckpt = registry.read_checkpoint(modular_run / "checkpoints" / "s0_step000002.pt")
    assert ckpt["kind"] == "modular" and ckpt["obs_variant"] == "default"
    for key in ("model", "ema"):
        np.testing.assert_array_equal(ckpt[key]["target_mean"].numpy(), mean)
        np.testing.assert_array_equal(ckpt[key]["target_std"].numpy(), std)
    man = mf.read_json(modular_run / "manifest.json")
    assert man["normaliser"]["target_mean"] == pytest.approx(mean.tolist())
    assert (man["kind"], man["tag"], man["obs_variant"]) == ("modular", "core", "default")
    held = [
        json.loads(x)
        for x in (modular_run / "train_log.jsonl").read_text().splitlines()
        if json.loads(x)["event"] == "heldout"
    ]
    assert held and all(math.isfinite(h["ema"]["loss"]) for h in held)


def test_load_student_builds_modular_with_the_frozen_teacher(
    modular_run: Path, stub_teacher: StubTeacher
) -> None:
    student = registry.load_student(modular_run / "ema_last.pt")
    assert isinstance(student, ModularStudent) and student.teacher is stub_teacher
    images, obs = _window()
    student.reset(2)
    action = student.act_modular(CONTRACT.modular_input(images, obs))
    assert action.shape == (3,) and np.isfinite(action).all()


# --------------------------------------------------------------------------- registry, ablations


def test_registry_entries_resolve_and_configs_are_frozen() -> None:
    assert len(registry.ABLATIONS) == 11
    for arm in ("modular", *registry.ABLATIONS):
        entry = registry.REGISTRY[arm]
        rel = entry.config.relative_to(registry.REPO_ROOT).as_posix()
        assert mf.FROZEN_CONFIG_SHA256[rel] == sha256_file(entry.config), arm
        cfg = TrainConfig.load(entry.config)
        assert (cfg.arm, cfg.kind, cfg.obs_variant) == (arm, entry.kind, entry.obs_variant)
        assert mf.check_config_frozen(cfg.files_sha256)["all_frozen"], arm
        assert registry.load_train_config(arm) == dict(cfg.raw)
        assert entry.tag == ("core" if arm == "modular" else "ablation")
        assert cfg.seeds == ((0, 1, 2, 3, 4) if arm == "modular" else (0, 1, 2))
    variants = {a: registry.REGISTRY[a].obs_variant for a in registry.ABLATIONS}
    assert {a for a, v in variants.items() if v != "default"} == {
        "bc_minus_last_action",
        "dp_minus_last_action",
        "bc_plus_clock",
    }
    assert variants["bc_plus_clock"] == "plus_clock"
    assert variants["dp_minus_last_action"] == variants["bc_minus_last_action"]
    assert variants["bc_minus_last_action"] == "minus_last_action"
    with pytest.raises(ValueError, match="no dummy"):
        registry.build_dummy("modular", 0)


def test_training_teacher_tags() -> None:
    expect = {"bc_pid": "T_pid", "dp_pid": "T_pid", "bc_mix": "T_mix", "dp_mix": "T_mix"}
    for arm in ("modular", "bc", "bc_dart", "bc_dagger", "dp", *registry.ABLATIONS):
        cfg = registry.load_train_config(arm)
        assert training_teacher(cfg) == expect.get(arm, "T_ppo"), arm


#: Proprio width each ablation trains and flies with.
WIDTH = {"default": 13, "plus_clock": 14, "minus_last_action": 10}


@pytest.mark.parametrize("arm", registry.ABLATIONS)
def test_ablation_trains_and_flies_its_variant(arm: str, data_root: Path, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    entry = registry.REGISTRY[arm]
    cfg = _cfg(arm)
    spec = _spec(cfg, tmp_path / arm, data_root, steps_override=1, checkpoint_every_override=1)
    stages = loop.build_stages(spec)
    bundle = stages[0].data(False)
    train = bundle.train
    assert isinstance(train, BCWindowDataset)
    assert type(train) is not EstimatorDataset  # an action target, never the priv target
    assert train.batch(np.arange(len(train), dtype=np.int64))["target"].shape[-1] == 3
    assert train.table.variant == entry.obs_variant
    with DemoStore(data_root / "demos" / f"{cfg.collection}.h5") as store:
        names = choose_episodes(store, 4, True).names
        expected = np.concatenate(
            [store.student_inputs(n, entry.obs_variant).proprio for n in names]
        )
    assert train.table.proprio.tobytes() == expected.tobytes()
    assert train.table.proprio.shape[1] == WIDTH[entry.obs_variant]
    if entry.obs_variant == "plus_clock":  # the labelled +clock input is time_fraction, last
        with DemoStore(data_root / "demos" / f"{cfg.collection}.h5") as store:
            obs = np.concatenate([store.privileged_observations(n) for n in names])
        np.testing.assert_array_equal(
            train.table.proprio[:, -1:], CONTRACT.audit_field(obs, CLOCK_FIELD)
        )
    result = loop.train_run(spec, stages)
    assert result.state == "done"
    assert result.net.encoder.cfg.proprio_dim == WIDTH[entry.obs_variant]  # type: ignore[union-attr]
    ckpt_path = tmp_path / arm / "checkpoints" / "s0_step000001.pt"
    ckpt = registry.read_checkpoint(ckpt_path)
    assert ckpt["obs_variant"] == entry.obs_variant and ckpt["arm"] == arm
    man = mf.read_json(tmp_path / arm / "manifest.json")
    assert (man["tag"], man["obs_variant"]) == ("ablation", entry.obs_variant)
    student = registry.load_student(ckpt_path)
    assert student.cfg.proprio_dim == WIDTH[entry.obs_variant]  # type: ignore[union-attr]
    si = load_contract(variant=entry.obs_variant).student_input(*_window())
    student.reset(4)
    action = student.act_vision(si.images, si.proprio)  # type: ignore[union-attr]
    assert action.shape == (3,) and np.isfinite(action).all()


@pytest.mark.parametrize("arm", registry.ABLATIONS)
def test_ablation_stage_data_identical_on_a_poisoned_store(
    arm: str, data_root: Path, tmp_path: Path
) -> None:
    """Per variant: privileged fields (+ the clock unless ``plus_clock``) NaN -> same batches."""
    entry = registry.REGISTRY[arm]
    poisoned = _poison_copy(data_root, tmp_path / "poisoned", entry.obs_variant, is_modular=False)
    cfg = _cfg(arm)
    out = {}
    for label, root in (("clean", data_root), ("poisoned", poisoned)):
        spec = _spec(cfg, tmp_path / label, root)
        bundle = loop.load_stage_data(spec, [(0, 5)], hash_store=False)
        train = bundle.train
        assert isinstance(train, BCWindowDataset)
        assert type(train) is not EstimatorDataset  # an action target, never the priv target
        b = train.batch(np.arange(len(train), dtype=np.int64))
        assert tuple(b) == SAMPLE_KEYS
        assert b["target"].shape[-1] == 3
        out[label] = {k: v.tobytes() for k, v in b.items()}
        assert all(np.isfinite(v).all() for v in b.values() if v.dtype.kind == "f")
    assert out["clean"] == out["poisoned"]


# --------------------------------------------------------------------------- estimation study


def test_error_rows_on_known_errors() -> None:
    n = 200
    truth = np.zeros((n, 11), np.float32)
    est = truth.copy()
    est[:, 0] = 0.04  # rel_position x: constant +4 cm
    est[:, 3:6] = np.where(np.arange(n)[:, None] % 2 == 0, 0.2, -0.2)  # rel_velocity +-0.2
    est[:, 9] = math.radians(1.0)  # rel_tilt: 1 degree
    rows = {(r["field"], r["component"]): r for r in error_rows(est, truth, CONTRACT)}
    px = rows[("rel_position", "x")]
    assert px["rmse"] == pytest.approx(0.04) and px["bias"] == pytest.approx(0.04)
    assert px["rmse_over_sigma"] == pytest.approx(1.0)
    assert px["rmse_full_scale"] == pytest.approx(1.0)  # 4 cm model = 1 m full scale
    assert rows[("rel_position", "norm")]["sigma"] == pytest.approx(0.04 * math.sqrt(3))
    vz = rows[("rel_velocity", "z")]
    assert vz["rmse"] == pytest.approx(0.2) and vz["bias"] == pytest.approx(0.0)
    assert vz["rmse_full_scale"] == pytest.approx(1.0)  # x 5 for velocities
    assert rows[("rel_velocity", "norm")]["rmse"] == pytest.approx(0.2 * math.sqrt(3))
    assert rows[("rel_tilt", "value")]["rmse"] == pytest.approx(1.0)  # degrees
    assert rows[("rel_tilt", "value")]["unit"] == "deg"
    assert math.isnan(rows[("deck_normal", "x")]["sigma"])
    assert rows[("height", "value")]["rmse"] == 0.0
    assert set(FIELDS) == set(CONTRACT.privileged)


def test_open_loop_study_on_an_untrained_checkpoint(data_root: Path, modular_run: Path) -> None:
    ckpt = modular_run / "ema_last.pt"
    ol = open_loop(ckpt, data_root, heldout=loop.HeldoutSpec("ppo_2000", 4, 3))
    assert ol["estimate"].shape == ol["truth"].shape and ol["estimate"].shape[1] == 11
    assert len(ol["groups"]) == ol["estimate"].shape[0] and np.isfinite(ol["estimate"]).all()
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        names = choose_episodes(store, 3, True, start=4).names
        truth = np.concatenate([store.estimator_targets(n) for n in names])
    assert ol["truth"].tobytes() == truth.tobytes()


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.checkpoints
def test_study_end_to_end_on_an_untrained_checkpoint(
    data_root: Path, modular_run: Path, tmp_path: Path
) -> None:
    """Open loop (synthetic held-out) + closed loop (2 select episodes, real T_ppo, CPU path)."""
    ckpt = modular_run / "ema_last.pt"
    out = tmp_path / "modular_estimation"
    rec = write_study(
        [(modular_run, ckpt, sha256_file(ckpt), False)],
        data_root=data_root,
        out_prefix=out,
        workers=1,
        select_limit=2,
        heldout=loop.HeldoutSpec("ppo_2000", 4, 2),
    )
    run = rec["runs"][0]
    assert run["closed_loop"]["steps"] > 0 and run["closed_loop"]["list"]["episodes"] == 2
    modes = {r["mode"] for r in rec["rows"]}
    assert modes == {"open_loop", "closed_loop"}
    text = out.with_suffix(".csv").read_text()
    assert text.splitlines()[0].startswith("arm,run_seed,checkpoint")
    json.loads(out.with_suffix(".json").read_text())  # strict JSON (no NaN)
