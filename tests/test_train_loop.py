"""Phase 4 training loop on a tiny synthetic store (CPU; no gpu/checkpoints/data markers).

Covers: episode choice ("first N" before the success filter, kept fraction), deterministic
loader seeding, the cosine schedule, both EMA rules (constant; diffusion_policy's power
warm-up), resume bit-identity, the fixed ``bc_state`` normaliser, the purpose gates
(``--max-steps`` is for timing runs only), the iterated ``bc_matched`` schedule's accounting
(2 000 labelled episodes; warm start; AdamW reset) and the frozen config SHA-256s.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ila.data.store import DemoStore
from ila.policies.dp.policy import DiffusionPolicyNet, ddpm_loss
from ila.policies.registry import (
    REPO_ROOT,
    build_net,
    build_net_from_config,
    load_student,
    read_checkpoint,
)
from ila.policies.state_mlp import StateMLP, fit_normaliser
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import EMAConfig, IterationSchedule, TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset, DPChunkDataset, FrameTable
from ila.train.sampling import (
    STREAM_CROP,
    STREAM_NOISE,
    EpochSampler,
    choose_episodes,
    loader_seed,
)
from ila.train.state_dataset import StateFrames
from test_data_store import synthetic_record, write_store

N_EPISODES = 8
BOUNCE = 2  # stream index of the one non-success episode


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    records = []
    for j in range(N_EPISODES):
        r = synthetic_record(j, 5 + j % 4, seed=100 + j)
        if j == BOUNCE:
            meta = dataclasses.replace(r.meta, outcome="bounce", success=False)
            r = dataclasses.replace(r, meta=meta)
        records.append(r)
    write_store(root / "demos" / "ppo_2000.h5", records)
    return root


def _cfg(arm: str, **change: Any) -> TrainConfig:
    cfg = TrainConfig.load(REPO_ROOT / "configs" / "train" / f"{arm}.yaml")
    return dataclasses.replace(cfg, **{"batch_size": 4, "precision": "fp32", **change})


def _spec(cfg: TrainConfig, run_dir: Path, data_root: Path, **kw: Any) -> loop.RunSpec:
    base: dict[str, Any] = {
        "cfg": cfg,
        "run_seed": 3,
        "run_dir": run_dir,
        "data_root": data_root,
        "device": "cpu",
        "purpose": "test",
        "first_n_override": 4,
        "steps_override": 6,
        "checkpoint_every_override": 3,
        "heldout": loop.HeldoutSpec("ppo_2000", 4, 2),
        "log_every": 1,
        "eval_every": 1000,
        "prefetch": 2,
        "hash_store": False,
    }
    base.update(kw)
    return loop.RunSpec(**base)


# --------------------------------------------------------------------------- data choice


def test_first_n_is_taken_before_the_success_filter(data_root: Path) -> None:
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        kept = choose_episodes(store, 4, True)
        assert kept.attempted == 4 and kept.kept == 3 and kept.kept_fraction == 0.75
        assert kept.demo_indices == (0, 1, 3) and dict(kept.outcomes) == {"bounce": 1, "success": 3}
        unfiltered = choose_episodes(store, 4, False)
        assert unfiltered.demo_indices == (0, 1, 2, 3)
        assert choose_episodes(store, 3, True, start=4).demo_indices == (4, 5, 6)
        with pytest.raises(ValueError, match="missing"):
            choose_episodes(store, 3, True, start=7)


def test_loader_seeding_is_a_pure_function_of_the_step() -> None:
    assert loader_seed(0, 0, 0, 0, 0, 0) == loader_seed(0, 0, 0, 0, 0, 0)
    keys = {(r, s, e, w, b, k) for r in (0, 1) for s in (0, 2) for e in (0, 3) for w in (0, 1)
            for b in (-1, 0, 5) for k in (STREAM_CROP, STREAM_NOISE)}  # fmt: skip
    assert len({loader_seed(*k) for k in keys}) == len(keys)
    a = EpochSampler(50, 8, run_seed=1)
    b = EpochSampler(50, 8, run_seed=1)
    assert a.batches_per_epoch == 6
    steps = [13, 0, 7, 6, 12]
    for s in steps:  # any order, any instance: the same batch
        np.testing.assert_array_equal(a.indices(s), b.indices(s))
    epoch0 = np.concatenate([a.indices(s) for s in range(6)])
    assert len(set(epoch0.tolist())) == 48  # without replacement within an epoch
    assert not np.array_equal(a.indices(0), a.indices(6))  # a new permutation per epoch
    assert not np.array_equal(a.indices(0), EpochSampler(50, 8, run_seed=1, stage=1).indices(0))
    assert a.seed(3, STREAM_CROP) != a.seed(3, STREAM_NOISE)


def test_prefetch_threads_do_not_change_batches(data_root: Path) -> None:
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        names = choose_episodes(store, 6, True).names
        ds = DPChunkDataset(FrameTable.from_store(store, names), 2, 16)
    sampler = EpochSampler(len(ds), 4, run_seed=0)
    make = lambda s: loop._make_input(ds, sampler, s, (4, 16, 3, 100), False)  # noqa: E731
    inline = list(loop._Prefetch(make, 0, 12, 0))
    threaded = list(loop._Prefetch(make, 0, 12, 4))
    for (s1, x), (s2, y) in zip(inline, threaded, strict=True):
        assert s1 == s2
        for k in x.batch:
            assert torch.equal(x.batch[k], y.batch[k])
        assert x.noise is not None and y.noise is not None and torch.equal(x.noise, y.noise)
        assert x.timestep is not None and y.timestep is not None
        assert torch.equal(x.timestep, y.timestep)
    # DDPM training noise comes from the loader's STREAM_NOISE generator (P2-D3 item 4).
    g = torch.Generator().manual_seed(sampler.seed(5, STREAM_NOISE))
    torch.testing.assert_close(inline[5][1].noise, torch.randn((4, 16, 3), generator=g))


# --------------------------------------------------------------------------- schedule, EMA


def test_cosine_with_warmup() -> None:
    assert loop.lr_multiplier(0, 100, 10) == 0.0
    assert loop.lr_multiplier(5, 100, 10) == 0.5
    assert loop.lr_multiplier(10, 100, 10) == 1.0
    assert math.isclose(loop.lr_multiplier(55, 100, 10), 0.5)
    assert loop.lr_multiplier(100, 100, 10) == 0.0


def test_ema_decay_rules() -> None:
    const = EMAConfig.from_value(0.999)
    assert [const.decay_at(n) for n in (1, 2, 1000)] == [0.999] * 3
    power = EMAConfig.from_value({"power": 0.75, "max_value": 0.9999})
    # diffusion_policy EMAModel: get_decay(optimization_step), optimization_step = n - 1.
    assert power.decay_at(1) == 0.0 and power.decay_at(2) == 0.0
    assert math.isclose(power.decay_at(3), 1 - 2**-0.75)
    assert math.isclose(power.decay_at(12), 1 - 11**-0.75)
    assert power.decay_at(10**7) == 0.9999


def test_ema_update_math() -> None:
    net = build_net("bc_state", 0)
    assert isinstance(net, StateMLP)
    ema = loop.EMA(net, EMAConfig.from_value(0.9))
    w0 = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
        net.obs_mean.fill_(2.0)
    ema.update(net)
    for k, v in ema.state_dict().items():
        if k.startswith("mlp."):
            torch.testing.assert_close(v, 0.9 * w0[k] + 0.1 * (w0[k] + 1.0))
    torch.testing.assert_close(ema.model.obs_mean, net.obs_mean)  # buffers copied
    assert ema.updates == 1


# --------------------------------------------------------------------------- runs


def _states(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _losses(run_dir: Path) -> dict[int, float]:
    out = {}
    for line in (run_dir / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "train":
            out[r["step"]] = r["loss_last"]
    return out


def test_resume_is_bit_identical(data_root: Path, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cfg = _cfg("bc")
    full_spec = _spec(cfg, tmp_path / "full", data_root)
    full = loop.train_run(full_spec, loop.build_stages(full_spec))
    assert full.state == "done" and full.global_step == 6
    cut_spec = _spec(cfg, tmp_path / "cut", data_root)
    first = loop.train_run(cut_spec, loop.build_stages(cut_spec), stop_after=3)
    assert first.state == "stopped" and first.global_step == 3
    second = loop.train_run(cut_spec, loop.build_stages(cut_spec))
    assert second.state == "done" and second.global_step == 6
    a, b = _states(full.net), _states(second.net)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ea, eb = _states(full.ema.model), _states(second.ema.model)
    assert all(torch.equal(ea[k], eb[k]) for k in ea)
    assert _losses(tmp_path / "full") == _losses(tmp_path / "cut")
    man = mf.read_json(tmp_path / "cut" / "manifest.json")
    assert [r["from"] for r in man["resumes"]] == ["s0_step000003.pt"]
    files = {c["file"]: c for c in man["checkpoints"]}
    assert set(files) == {"s0_step000003.pt", "s0_step000006.pt", "ema_last.pt"}
    for name, entry in files.items():
        path = tmp_path / "cut" / ("checkpoints" if name != "ema_last.pt" else "") / name
        assert sha256_file(path) == entry["sha256"]
    # The last-step EMA is kept and loads as the evaluated student.
    last = read_checkpoint(tmp_path / "cut" / "ema_last.pt")
    assert "model" not in last and last["global_step"] == 6
    student = load_student(tmp_path / "cut" / "ema_last.pt")
    torch.testing.assert_close(_states(student.net)["head.0.weight"], eb["head.0.weight"])
    held = [json.loads(x) for x in (tmp_path / "full" / "train_log.jsonl").read_text().split("\n")
            if x and json.loads(x)["event"] == "heldout"]  # fmt: skip
    assert [h["step"] for h in held] == [3, 6]
    assert all(math.isfinite(h["raw"]["loss"]) and math.isfinite(h["ema"]["loss"]) for h in held)
    assert held[0]["overlap_episodes"] == 0


def test_bc_state_normaliser_is_fixed_from_its_training_frames(
    data_root: Path, tmp_path: Path
) -> None:
    spec = _spec(_cfg("bc_state"), tmp_path / "state", data_root, steps_override=4)
    result = loop.train_run(spec, loop.build_stages(spec))
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        frames = StateFrames(store, choose_episodes(store, 4, True).names)
    mean, std = fit_normaliser(frames.obs)
    net = result.net
    assert isinstance(net, StateMLP)
    np.testing.assert_array_equal(net.obs_mean.numpy(), mean)  # unchanged by training
    np.testing.assert_array_equal(net.obs_std.numpy(), std)
    ckpt = read_checkpoint(tmp_path / "state" / "checkpoints" / "s0_step000004.pt")
    np.testing.assert_array_equal(ckpt["model"]["obs_mean"].numpy(), mean)
    np.testing.assert_array_equal(ckpt["ema"]["obs_std"].numpy(), std)
    man = mf.read_json(tmp_path / "state" / "manifest.json")
    assert man["normaliser"]["obs_mean"] == pytest.approx(mean.tolist())
    student = load_student(tmp_path / "state" / "ema_last.pt")
    assert not hasattr(student, "act_vision")
    action = student.act(frames.obs[0])
    assert action.shape == (3,) and np.all(np.abs(action) <= 1.0)
    with pytest.raises(ValueError, match="privileged"):
        student.reset(1, context=object())


def test_fit_normaliser() -> None:
    x = np.array([[1.0, 5.0, 0.0], [3.0, 5.0, 2.0]], np.float32)
    mean, std = fit_normaliser(x)
    np.testing.assert_array_equal(mean, [2.0, 5.0, 1.0])
    np.testing.assert_array_equal(std, [1.0, 1.0, 1.0])  # the constant column is not scaled
    with pytest.raises(ValueError):
        fit_normaliser(np.full((2, 3), np.nan, np.float32))


def test_vision_proprio_normaliser_is_fitted(data_root: Path, tmp_path: Path) -> None:
    spec = _spec(_cfg("bc"), tmp_path / "bc", data_root, steps_override=1)
    result = loop.train_run(spec, loop.build_stages(spec))
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        table = FrameTable.from_store(store, choose_episodes(store, 4, True).names)
    mean, std = table.proprio_stats()
    enc = result.net.encoder  # type: ignore[union-attr]
    np.testing.assert_array_equal(enc.proprio_mean.numpy(), mean)
    np.testing.assert_array_equal(result.ema.model.encoder.proprio_std.numpy(), std)  # type: ignore[union-attr]


def test_purpose_gates(data_root: Path, tmp_path: Path) -> None:
    cfg = _cfg("bc")
    with pytest.raises(ValueError, match="timing"):
        loop.RunSpec(cfg, 0, tmp_path, data_root, purpose="train", max_steps=10)
    with pytest.raises(ValueError, match="timing"):
        loop.RunSpec(cfg, 0, tmp_path, data_root, purpose="timing")
    with pytest.raises(ValueError, match="smoke"):
        loop.RunSpec(cfg, 0, tmp_path, data_root, purpose="train", steps_override=10)
    with pytest.raises(ValueError, match="tuning"):
        loop.RunSpec(cfg, 0, tmp_path, data_root, purpose="tuning")
    with pytest.raises(ValueError, match="DART trials"):
        mf.run_dir("bc_dart", 1, sigma=0.1, tuning=True)
    assert mf.run_dir("bc_dart", 0, sigma=0.1, tuning=True).parts[-3:] == (
        "bc_dart_tune",
        "sigma_0.10",
        "s0",
    )
    assert mf.run_dir("dp", 0, timing=True).parts[-2:] == ("dp_timing", "s0")


def test_timing_run_writes_no_checkpoint(data_root: Path, tmp_path: Path) -> None:
    cfg = _cfg("bc")
    spec = loop.RunSpec(
        cfg,
        0,
        tmp_path / "timing",
        data_root,
        device="cpu",
        purpose="timing",
        max_steps=2,
        heldout=None,
        hash_store=False,
        log_every=1,
        allow_dirty=True,
    )
    # A timing run takes the real config's data (first 1 000): give it a 1-episode config.
    spec = dataclasses.replace(spec, cfg=dataclasses.replace(cfg, first_n=4))
    result = loop.train_run(spec, loop.build_stages(spec))
    man = mf.read_json(tmp_path / "timing" / "manifest.json")
    assert result.state == "done" and result.global_step == 2
    assert man["purpose"] == "timing" and man["selectable"] is False and man["max_steps"] == 2
    assert man["checkpoints"] == [] and man["total_steps"] == cfg.steps
    assert man["steps_per_s_loop"] > 0
    assert not list((tmp_path / "timing" / "checkpoints").iterdir())


def test_dp_trains_one_step_on_cpu(data_root: Path, tmp_path: Path) -> None:
    """The DP timing-run path: chunk dataset, loader noise, ddpm_loss, power EMA."""
    cfg = _cfg("dp", batch_size=2)
    spec = _spec(cfg, tmp_path / "dp", data_root, steps_override=2, checkpoint_every_override=2)
    result = loop.train_run(spec, loop.build_stages(spec))
    assert result.state == "done" and isinstance(result.net, DiffusionPolicyNet)
    assert result.ema.updates == 2 and result.ema.cfg.kind == "power"
    held = [json.loads(x) for x in (tmp_path / "dp" / "train_log.jsonl").read_text().split("\n")
            if x and json.loads(x)["event"] == "heldout"]  # fmt: skip
    assert set(held[0]["raw"]) == {
        "loss",
        "loss_k0_24",
        "loss_k25_49",
        "loss_k50_74",
        "loss_k75_99",
    }


def test_ddpm_loss_contract() -> None:
    net = build_net("dp", 0)
    assert isinstance(net, DiffusionPolicyNet)
    net.eval()
    torch.testing.assert_close(net.ddpm().alphas_cumprod, net.ddim(10).alphas_cumprod)
    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 256, (2, 2, 96, 96, 3), dtype=np.uint8))
    proprio = torch.zeros(2, 2, 13)
    target = torch.rand(2, 16, 3) * 2 - 1
    noise = torch.randn(2, 16, 3)
    with torch.no_grad():
        per = ddpm_loss(net, images, proprio, target, noise, torch.tensor([0, 99]))
    assert per.shape == (2,) and torch.isfinite(per).all()
    with pytest.raises(ValueError, match="timestep"):
        ddpm_loss(net, images, proprio, target, noise, torch.tensor([0, 100]))
    with pytest.raises(ValueError, match="target"):
        ddpm_loss(net, images, proprio, target[:, :8], noise, torch.tensor([0, 1]))


# --------------------------------------------------------------------------- iterations


def test_bc_matched_schedule_is_the_dagger_schedule(tmp_path: Path) -> None:
    cfg = TrainConfig.load(REPO_ROOT / "configs" / "train" / "bc_matched.yaml")
    it = cfg.iterations
    assert it == IterationSchedule(5, 200, 1000, 10_000, 500, 2000)
    assert cfg.first_n == 1000 and cfg.collection == "ppo_2000" and cfg.success_filter
    spec = loop.RunSpec(cfg, 0, tmp_path, tmp_path, init_from=tmp_path / "bc.pt")
    stages = loop.build_stages(spec)
    assert [(s.index, s.steps, s.warmup, s.checkpoint_every) for s in stages] == [
        (i, 10_000, 500, 5000) for i in range(1, 6)
    ]
    last_ranges = stages[-1].data.args[1]  # type: ignore[attr-defined]
    assert last_ranges == [(0, 1000), (1000, 200), (1200, 200), (1400, 200), (1600, 200),
                           (1800, 200)]  # fmt: skip
    assert sum(n for _, n in last_ranges) == it.labelled_episodes_total == 2000
    assert 30_000 + sum(s.steps for s in stages) == 80_000  # P3-D1 §3
    with pytest.raises(ValueError, match="init-from"):
        loop.build_stages(dataclasses.replace(spec, init_from=None))


def test_iterated_run_warm_starts_and_counts_labelled_episodes(
    data_root: Path, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    bc_spec = _spec(_cfg("bc"), tmp_path / "bc", data_root, first_n_override=3, heldout=None,
                    steps_override=2, checkpoint_every_override=2)  # fmt: skip
    bc = loop.train_run(bc_spec, loop.build_stages(bc_spec))
    bc_final = tmp_path / "bc" / "checkpoints" / "s0_step000002.pt"
    schedule = IterationSchedule(2, 2, 3, 3, 1, labelled_episodes_total=7)
    cfg = _cfg("bc_matched", iterations=schedule, first_n=3)
    spec = _spec(
        cfg,
        tmp_path / "matched",
        data_root,
        init_from=bc_final,
        heldout=None,
        first_n_override=None,
        steps_override=None,
        checkpoint_every_override=None,
    )
    spec = dataclasses.replace(spec, cfg=dataclasses.replace(cfg, checkpoint_every=3))
    result = loop.train_run(spec, loop.build_stages(spec))
    assert result.state == "done" and result.global_step == 6
    man = mf.read_json(tmp_path / "matched" / "manifest.json")
    assert man["iteration0"]["labelled_episodes"] == 3
    assert man["data"]["1"]["labelled_episodes_total"] == 5
    assert man["data"]["2"]["labelled_episodes_total"] == 7
    assert man["data"]["2"]["kept_episodes"] == 6  # the bounce (index 2) is filtered
    assert man["init_from_sha256"] == sha256_file(bc_final)
    assert "normaliser" not in man  # the bc run's normaliser is inherited, not refitted
    # Warm start copies weights and EMA; AdamW is reset per iteration.
    first = read_checkpoint(tmp_path / "matched" / "checkpoints" / "s1_step000003.pt")
    final = read_checkpoint(tmp_path / "matched" / "checkpoints" / "s2_step000003.pt")
    steps = {int(v["step"]) for v in final["optimizer"]["state"].values()}
    assert steps == {3}  # 3 AdamW steps in iteration 2, not 6
    assert first["ema_updates"] == bc.ema.updates + 3
    torch.testing.assert_close(
        first["model"]["encoder.proprio_mean"], _states(bc.net)["encoder.proprio_mean"]
    )
    bad = dataclasses.replace(cfg, checkpoint_every=3, iterations=dataclasses.replace(
        schedule, labelled_episodes_total=8))  # fmt: skip
    with pytest.raises(RuntimeError, match="labelled episodes"):
        loop.train_run(dataclasses.replace(spec, cfg=bad, run_dir=tmp_path / "bad"),
                       loop.build_stages(dataclasses.replace(spec, cfg=bad)))  # fmt: skip


def test_iteration_zero_must_be_a_finished_same_seed_bc_run(
    data_root: Path, tmp_path: Path
) -> None:
    bc_spec = _spec(_cfg("bc"), tmp_path / "bc", data_root, first_n_override=3, heldout=None,
                    steps_override=2, checkpoint_every_override=1)  # fmt: skip
    loop.train_run(bc_spec, loop.build_stages(bc_spec))
    cfg = _cfg("bc_matched", iterations=IterationSchedule(1, 2, 3, 2, 1, 5), first_n=3)
    mid = tmp_path / "bc" / "checkpoints" / "s0_step000001.pt"
    spec = _spec(
        cfg,
        tmp_path / "m",
        data_root,
        init_from=mid,
        heldout=None,
        first_n_override=None,
        steps_override=None,
        checkpoint_every_override=None,
    )
    with pytest.raises(ValueError, match="last step"):
        loop.train_run(spec, loop.build_stages(spec))
    other_seed = dataclasses.replace(spec, run_seed=4, init_from=mid.with_name("s0_step000002.pt"))
    with pytest.raises(ValueError, match="arm/seed"):
        loop.train_run(other_seed, loop.build_stages(other_seed))


# --------------------------------------------------------------------------- frozen configs


def test_frozen_config_sha256_match_files_and_protocol() -> None:
    protocol = (REPO_ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
    for rel, sha in mf.FROZEN_CONFIG_SHA256.items():
        assert sha256_file(REPO_ROOT / rel) == sha, rel
        assert f"{sha[:8]}…{sha[-4:]}" in protocol or f"`{sha}`" in protocol, rel
    check = mf.check_config_frozen(
        _cfg("bc").files_sha256
    )  # dataclasses.replace keeps the file SHAs
    assert check["all_frozen"]


def test_bc_window_dataset_feeds_the_encoder(data_root: Path) -> None:
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        ds = BCWindowDataset(FrameTable.from_store(store, ["ep_00000", "ep_00001"]), 2)
    b = ds.batch(np.arange(4))
    assert b["images"].shape == (4, 2, 96, 96, 3) and b["proprio"].shape == (4, 2, 13)
    net = build_net("bc", 0).eval()
    with torch.no_grad():
        out = net(torch.from_numpy(b["images"]), torch.from_numpy(b["proprio"]))
    assert out.shape == (4, 3)


# --------------------------------------------------------------------------- poison (leak NOTE 3)


def _poison_copy(src: Path, dst_root: Path) -> Path:
    """Copy a data root's ppo_2000 store and poison it (``tests/test_data_store.py``'s rule).

    NaN: the five privileged fields and ``time_fraction`` of ``teacher_obs`` (by name), the whole
    ``priv`` block, ``t_model_s``. Scrambled: ``t0_model_s``, ``steps``, ``ss``, ``heading_deg``,
    ``speed_kn``. ``outcome`` and ``demo_index`` are left alone: they are exactly what
    ``choose_episodes`` is allowed to read (the success filter and "first N"), outside the
    batchers.
    """
    import shutil

    import h5py

    from ila.obs.split import CLOCK_FIELD, NEVER_STUDENT
    from test_data_store import SCRAMBLED_ATTRS, _set

    path = dst_root / "demos" / "ppo_2000.h5"
    path.parent.mkdir(parents=True)
    shutil.copy(src / "demos" / "ppo_2000.h5", path)
    with h5py.File(path, "r+") as f:
        for name in f:
            g = f[name]
            obs = g["teacher_obs"][()]
            for field in NEVER_STUDENT | {CLOCK_FIELD}:
                _set(obs, field, np.nan)
            g["teacher_obs"][...] = obs
            g["priv"][...] = np.nan
            g["t_model_s"][...] = np.nan
            for key, value in SCRAMBLED_ATTRS.items():
                if key != "outcome":
                    g.attrs[key] = value
    return dst_root


def _bundle_bytes(bundle: loop.DataBundle) -> dict[str, bytes]:
    out = {}
    for part, ds in (("train", bundle.train), ("heldout", bundle.heldout)):
        assert isinstance(ds, BCWindowDataset)
        t = ds.table
        out.update(
            {
                f"{part}.images": t.images.tobytes(),
                f"{part}.proprio": t.proprio.tobytes(),
                f"{part}.label": t.label.tobytes(),
                f"{part}.episode_len": t.episode_len.tobytes(),
                f"{part}.window_index": ds.window_index.tobytes(),
                f"{part}.batch": b"".join(
                    v.tobytes() for v in ds.batch(np.arange(len(ds), dtype=np.int64)).values()
                ),
            }
        )
        if isinstance(ds, DPChunkDataset):
            out[f"{part}.target_index"] = ds.target_index.tobytes()
    out["names"] = repr([c.names for c in bundle.choices]).encode()
    return out


#: The vision encoder's fitted normaliser buffers.
NORMALISER_KEYS = ("encoder.proprio_mean", "encoder.proprio_std")


@pytest.mark.parametrize("arm", ["bc_matched", "dp"])
def test_poisoned_store_gives_identical_stage_data_and_normaliser(
    arm: str, data_root: Path, tmp_path: Path
) -> None:
    """Stage data and the fitted normaliser are byte-identical on a poisoned store.

    ``load_stage_data`` (with ``FrameTable.concat`` for an iterated arm, ``DPChunkDataset`` for
    ``dp``), its held-out slice and ``_fit_normaliser`` (privileged-leak NOTE 3).
    """
    poisoned = _poison_copy(data_root, tmp_path / "poisoned")
    if arm == "bc_matched":
        cfg = _cfg(arm, iterations=IterationSchedule(2, 2, 3, 2, 1, 7), first_n=3)
        ranges = [(0, 3), (3, 2), (5, 2)]  # iteration 2's aggregate: concat of three slices
    else:
        cfg = _cfg(arm)
        ranges = [(0, 5)]
    held = loop.HeldoutSpec("ppo_2000", 6, 2)
    out = {}
    buffers = {}
    for label, root in (("clean", data_root), ("poisoned", poisoned)):
        spec = _spec(cfg, tmp_path / label, root, heldout=held)
        bundle = loop.load_stage_data(spec, ranges, hash_store=False)
        out[label] = _bundle_bytes(bundle)
        net = build_net_from_config(cfg.kind, cfg.raw, 0)
        ema = loop.EMA(net, cfg.ema)
        loop._fit_normaliser(net, ema, bundle.train, torch.device("cpu"))
        state = net.state_dict()
        buffers[label] = {k: state[k].numpy().tobytes() for k in NORMALISER_KEYS}
        assert all(np.isfinite(np.frombuffer(v, np.float32)).all() for v in buffers[label].values())
    with DemoStore(poisoned / "demos" / "ppo_2000.h5") as store:  # the poison took
        assert np.isnan(store.privileged_observations("ep_00000")).any()
        assert np.isnan(store.estimator_targets("ep_00000")).all()
    assert out["clean"].keys() == out["poisoned"].keys()
    for key in out["clean"]:
        assert out["clean"][key] == out["poisoned"][key], key
    assert buffers["clean"] == buffers["poisoned"] and len(buffers["clean"]) == 2


def test_only_bc_state_reads_teacher_obs(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "configs" / "train" / "bc.yaml").read_text())
    raw["data"] = {**raw["data"], "input": "teacher_obs"}  # a vision arm asking for teacher_obs
    path = tmp_path / "bc.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="only bc_state"):
        TrainConfig.load(path)
    state = yaml.safe_load((REPO_ROOT / "configs" / "train" / "bc_state.yaml").read_text())
    del state["data"]["input"]
    path.write_text(yaml.safe_dump(state))
    with pytest.raises(ValueError, match="only bc_state"):
        TrainConfig.load(path)
