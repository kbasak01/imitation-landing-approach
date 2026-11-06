"""``OnnxDPStudent`` (``R1_lerobot_dp``, P6-D3) on a tiny synthetic ONNX pair (CPU, ORT).

The pair is generated in-test by ``tests/onnx_support.py`` with the R1 graph contract. Checked:
the student contract (shapes, dtype, range, refusals), the sampler being ``dp``'s (a pure-torch
reference through ``ila.policies.dp.policy.ddim_sample`` agrees), chunk execution from
``To - 1`` with replanning every ``Ta`` and the noise key ``(run_seed, episode_seed, t)``,
determinism and reset, LeRobot's MIN_MAX unnormalisation, the ``meta.json`` and graph guards,
the registry entry, and selection's ONNX-export candidates. Actions are normalised world-frame
velocity setpoints in [-1, 1] (x 1.5 m/s model scale); one step is 1/30 s model scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import onnx_support as ox
from ila.policies import onnx_dp
from ila.policies.dp.policy import (
    ChunkedDiffusionStudent,
    DPStudent,
    ddim_sample,
    ddim_scheduler,
    initial_noise,
)
from ila.policies.onnx_dp import OnnxDPStudent, load_onnx_student


def load_dp_cfg() -> Any:
    from ila.policies.dp.policy import DPConfig
    from ila.policies.registry import load_train_config

    return DPConfig.from_mapping(load_train_config("dp"))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ox.write_bundle(tmp_path_factory.mktemp("r1") / "onnx" / "step10000")


@pytest.fixture
def student(bundle: Path) -> OnnxDPStudent:
    torch.set_num_threads(1)
    return load_onnx_student(bundle, run_seed=3)


def _windows(n: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        (
            rng.integers(0, 256, (ox.TO, 96, 96, 3), dtype=np.uint8),
            rng.standard_normal((ox.TO, ox.D)).astype(np.float32),
        )
        for _ in range(n)
    ]


# --------------------------------------------------------------------------- contract


def test_contract_shapes_dtype_range(student: OnnxDPStudent) -> None:
    student.reset(11)
    for img, prop in _windows(10):
        a = student.act_vision(img, prop)
        assert a.shape == (3,) and a.dtype == np.float32
        assert np.all(np.abs(a) <= 1.0) and np.all(np.isfinite(a))
    assert student.chunk_spec.n_action_steps == 8 and student.num_inference_steps == 10
    assert not student.temporal_ensembling


def test_refuses_privileged_input(student: OnnxDPStudent) -> None:
    with pytest.raises(ValueError, match="privileged"):
        student.reset(1, context=object())
    student.reset(1)
    with pytest.raises(TypeError, match="act_vision"):
        student.act(np.zeros(25, np.float32))
    img, _ = _windows(1)[0]
    with pytest.raises(ValueError, match="proprio"):
        student.act_vision(img, np.zeros((ox.TO, 25), np.float32))
    with pytest.raises(ValueError, match="image_stack"):
        student.act_vision(img.astype(np.float32), np.zeros((ox.TO, ox.D), np.float32))  # type: ignore[arg-type]


def test_act_before_reset_raises(bundle: Path) -> None:
    img, prop = _windows(1)[0]
    with pytest.raises(RuntimeError, match="reset"):
        load_onnx_student(bundle, 0).act_vision(img, prop)


def test_shares_dp_chunk_executor() -> None:
    """One executor: neither subclass overrides the clocked methods."""
    for cls in (DPStudent, OnnxDPStudent):
        assert issubclass(cls, ChunkedDiffusionStudent)
        for name in ("act_vision", "_ensembled", "_reset_episode"):
            assert getattr(cls, name) is getattr(ChunkedDiffusionStudent, name), (cls, name)


# --------------------------------------------------------------------------- sampler


def test_sample_batch_matches_torch_reference(bundle: Path, student: OnnxDPStudent) -> None:
    """ORT graphs + ila's DDIM == the torch modules + ila's DDIM (same noise, same K)."""
    enc, den = ox.modules(0)
    wins = _windows(3, seed=4)
    img = np.stack([w[0] for w in wins])
    prop = np.stack([w[1] for w in wins])
    seeds = [(3, 100 + i, 7 * i) for i in range(3)]
    got = student.sample_batch(img, prop, seeds)
    with torch.inference_mode():
        cond = enc(torch.from_numpy(img), torch.from_numpy(prop))
        np.testing.assert_allclose(student.encode(img, prop), cond.numpy(), atol=1e-6)
        sched = student.ddim(10)
        want_t = [int(k) for k in ddim_scheduler(load_dp_cfg(), 10).timesteps]
        assert [int(k) for k in sched.timesteps] == want_t  # dp's trailing schedule
        x = ddim_sample(lambda s, k: den(s, k, cond), initial_noise(seeds, ox.TP, ox.A), sched)
        lo = torch.tensor(ox.ACTION_MIN)
        ref = ((x + 1) / 2 * (torch.tensor(ox.ACTION_MAX) - lo) + lo).clamp(-1, 1).numpy()
    assert got.shape == (3, ox.TP, ox.A) and got.dtype == np.float32
    # The first trailing step multiplies an epsilon difference by ~2029 (P3-D1 §8).
    np.testing.assert_allclose(got, ref, atol=2e-3)


def test_slot_noise_is_its_own_key(student: OnnxDPStudent) -> None:
    wins = _windows(2, seed=5)
    img = np.stack([w[0] for w in wins])
    prop = np.stack([w[1] for w in wins])
    both = student.sample_batch(img, prop, [(3, 1, 0), (3, 2, 0)])
    one = student.sample_batch(img[1:], prop[1:], [(3, 2, 0)])
    np.testing.assert_allclose(both[1], one[0], atol=1e-4)
    again = student.sample_batch(img, prop, [(3, 1, 0), (3, 2, 0)])
    assert np.array_equal(both, again)  # fixed batch: bit-identical


def test_unnormalise_min_max_with_zero_range(tmp_path: Path) -> None:
    b = ox.write_bundle(
        tmp_path / "step10000", action_min=[-0.5, 0.2, -2.0], action_max=[0.5, 0.2, 2.0]
    )
    st = load_onnx_student(b, 0)
    x = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.5, 0.25]], np.float32)
    got = st.unnormalise(x)
    lo = np.array([-0.5, 0.2, -2.0], np.float32)
    d = np.array([1.0, 1e-8, 4.0], np.float32)
    want = np.clip((x + 1) / 2 * d + lo, -1, 1)
    np.testing.assert_array_equal(got, want.astype(np.float32))
    assert got[0, 2] == -1.0 and got[1, 2] == 1.0  # the action bound clips the third dim


def test_does_not_touch_global_rng(student: OnnxDPStudent) -> None:
    state = torch.get_rng_state()
    np_state = np.random.get_state()[1].copy()
    student.reset(2)
    for img, prop in _windows(3):
        student.act_vision(img, prop)
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(np_state, np.random.get_state()[1])


# --------------------------------------------------------------------------- execution


def test_executes_from_to_minus_one_and_replans_every_ta(
    student: OnnxDPStudent, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys: list[tuple[int, int, int]] = []

    def fake(images: Any, proprio: Any, seeds: Any, num_inference_steps: Any = None) -> Any:
        keys.extend(seeds)
        t = seeds[0][2]
        chunk = np.zeros((1, ox.TP, ox.A), np.float32)
        chunk[0, :, 0] = np.arange(ox.TP)  # chunk index
        chunk[0, :, 1] = t  # replanning step
        return chunk

    monkeypatch.setattr(student, "sample_batch", fake)
    student.reset(42)
    img, prop = _windows(1)[0]
    acts = [student.act_vision(img, prop) for _ in range(20)]
    assert keys == [(3, 42, 0), (3, 42, 8), (3, 42, 16)]
    for t, a in enumerate(acts):
        assert a[0] == ox.TO - 1 + t % 8, t  # executed indices To-1 ... To-2+Ta
        assert a[1] == t - t % 8


def test_determinism_and_reset(bundle: Path) -> None:
    wins = _windows(12, seed=7)
    a = load_onnx_student(bundle, 3)
    b = load_onnx_student(bundle, 3)
    a.reset(9)
    b.reset(9)
    first = [a.act_vision(*w) for w in wins]
    assert all(np.array_equal(x, b.act_vision(*w)) for x, w in zip(first, wins, strict=True))
    a.reset(9)
    assert all(np.array_equal(x, a.act_vision(*w)) for x, w in zip(first, wins, strict=True))
    a.reset(10)
    assert not np.array_equal(first[0], a.act_vision(*wins[0]))  # the seed keys the noise


def test_inference_calls_count_chunks_sampled(bundle: Path) -> None:
    """R1 ONNX: one call per chunk sampled (Ta = 8: steps 0, 8, 16 of 20); reset zeroes it."""
    st = load_onnx_student(bundle, run_seed=1, num_inference_steps=2)
    st.reset(4)
    for img, prop in _windows(20):
        st.act_vision(img, prop)
    assert st.inference_calls == 3
    st.reset(5)
    assert st.inference_calls == 0


def test_temporal_ensembling_ta_1_equals_plain(bundle: Path) -> None:
    wins = _windows(5, seed=8)
    plain = load_onnx_student(bundle, 0, n_action_steps=1)
    ens = load_onnx_student(bundle, 0, n_action_steps=1, temporal_ensembling=True)
    plain.reset(4)
    ens.reset(4)
    assert all(np.array_equal(plain.act_vision(*w), ens.act_vision(*w)) for w in wins)


# --------------------------------------------------------------------------- guards


def _rewrite_meta(bundle: Path, **changes: Any) -> None:
    meta = json.loads((bundle / "meta.json").read_text())
    meta.update(changes)
    (bundle / "meta.json").write_text(json.dumps(meta))


def test_meta_guards(tmp_path: Path) -> None:
    cases: dict[str, tuple[dict[str, Any], str]] = {
        "format": ({"format": "other/1"}, "format"),
        "prediction": ({"prediction_type": "sample"}, "epsilon"),
        "mean_std": (
            {"action_unnormalisation": {"mode": "MEAN_STD", "mean": [0] * 3, "std": [1] * 3}},
            "MIN_MAX",
        ),
    }
    for name, (change, match) in cases.items():
        b = ox.write_bundle(tmp_path / name / "step10000")
        _rewrite_meta(b, **change)
        with pytest.raises((ValueError, RuntimeError), match=match):
            load_onnx_student(b, 0)
    b = ox.write_bundle(tmp_path / "spacing" / "step10000")
    meta = json.loads((b / "meta.json").read_text())
    meta["scheduler"]["ila_inference"]["timestep_spacing"] = "leading"
    (b / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="leading"):
        load_onnx_student(b, 0)


def test_tampered_graph_and_rng_op_refused(tmp_path: Path) -> None:
    b = ox.write_bundle(tmp_path / "t" / "step10000")
    data = bytearray((b / "denoiser.onnx").read_bytes())
    data[-1] ^= 1
    (b / "denoiser.onnx").write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="SHA-256"):  # r1_parity.R1ExportError
        load_onnx_student(b, 0)
    with pytest.raises(ValueError, match="RNG|rng"):
        load_onnx_student(ox.write_bundle(tmp_path / "r" / "step10000", rng_op=True), 0)


def test_graph_contract_checked(tmp_path: Path) -> None:
    b = ox.write_bundle(tmp_path / "c" / "step10000")
    meta = json.loads((b / "meta.json").read_text())
    meta["C"] = ox.C + 1  # the graph's global_cond width no longer matches
    (b / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="global_cond"):
        load_onnx_student(b, 0)


def test_bundle_step_and_export_dirs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for step in (10000, 20000):
        ox.write_bundle(run / "onnx" / f"step{step}", step=step)
    assert sorted(onnx_dp.export_dirs(run)) == [10000, 20000]
    _rewrite_meta(run / "onnx" / "step20000", step=30000)
    with pytest.raises(ValueError, match="step"):
        onnx_dp.export_dirs(run)


# --------------------------------------------------------------------------- registry, selection


def test_registry_entry() -> None:
    from ila.policies import registry
    from ila.train import manifest as mf
    from ila.train.config import sha256_file

    entry = registry.REGISTRY["R1_lerobot_dp"]
    assert (entry.kind, entry.tag, entry.is_vision, entry.has_ila_network) == (
        "onnx_dp",
        "reference_external",
        True,
        False,
    )
    rel = entry.config.relative_to(registry.REPO_ROOT).as_posix()
    assert mf.FROZEN_CONFIG_SHA256[rel] == sha256_file(entry.config)
    cfg = registry.load_train_config("R1_lerobot_dp")
    assert cfg["tag"] == "reference_external" and cfg["seeds"] == [0, 1, 2]
    assert cfg["selection"]["at"] == {"Ta": 8, "K": 10}
    assert cfg["inference"]["timestep_spacing"] == "trailing"
    assert cfg["data"]["input_features"] == ["observation.image", "observation.state"]
    # P6-D3: 12 loader workers, explicit; LeRobot's default (4) recorded as overridden.
    assert cfg["train"]["num_workers"] == 12
    assert cfg["lerobot_defaults_overridden"] == {"train": {"num_workers": 4}}
    assert "num_workers" not in cfg["lerobot_defaults_kept"]["train"]
    for fn in (registry.build_net, registry.build_dummy):
        with pytest.raises(ValueError, match="no ila network"):
            fn("R1_lerobot_dp", 0)
    tags = {n: e.tag for n, e in registry.REGISTRY.items() if n != "R1_lerobot_dp"}
    assert {n for n, t in tags.items() if t == "ablation"} == set(registry.ABLATIONS)
    assert all(t == "core" for n, t in tags.items() if n not in registry.ABLATIONS)


def test_r1_config_matches_dp() -> None:
    """The matched hyper-parameters equal dp.yaml's (P3-D1 §3)."""
    from ila.policies.registry import load_train_config

    r1, dp = load_train_config("R1_lerobot_dp"), load_train_config("dp")
    p = r1["policy"]
    assert (p["n_obs_steps"], p["horizon"], p["n_action_steps"]) == tuple(
        dp["horizons"][k] for k in ("n_obs_steps", "horizon", "n_action_steps")
    )
    assert (
        p["down_dims"] == dp["unet"]["down_dims"] and p["kernel_size"] == dp["unet"]["kernel_size"]
    )
    assert p["diffusion_step_embed_dim"] == dp["unet"]["diffusion_step_embed_dim"]
    ns = dp["noise_scheduler"]
    for k in ("num_train_timesteps", "beta_schedule", "prediction_type", "clip_sample"):
        assert p[k] == ns[k], k
    assert p["crop_shape"] == [dp["augment"]["crop"]["size"]] * 2 and p["crop_is_random"]
    assert p["pretrained_backbone_weights"] is None and p["use_group_norm"]
    assert p["spatial_softmax_num_keypoints"] == dp["encoder"]["spatial_softmax_keypoints"]
    assert (r1["train"]["steps"], r1["train"]["batch_size"], r1["train"]["save_freq"]) == (
        dp["steps"],
        dp["batch_size"],
        dp["checkpoint_every"],
    )


def _onnx_run(root: Path, steps: tuple[int, ...] = (10000, 20000)) -> Path:
    from ila.train import manifest as mf

    ckpts = []
    for step in steps:
        sha = f"{step:064d}"
        (root / "checkpoints" / f"{step:06d}").mkdir(parents=True)
        ckpts.append(
            {
                "kind": "full",
                "file": f"{step:06d}",
                "stage": 0,
                "stage_step": step,
                "global_step": step,
                "sha256": sha,
            }  # fmt: skip
        )
        ox.write_parity(
            ox.write_bundle(root / "onnx" / f"step{step}", step=step, source_sha256=sha)
        )
    mf.write_json_atomic(
        root / "manifest.json",
        {
            "arm": "R1_lerobot_dp",
            "run_seed": 1,
            "purpose": "train",
            "selectable": True,
            "state": "done",
            "tag": "reference_external",
            "checkpoints": ckpts,
            "stages": [{"index": 0}],
            "config_resolved": {"selection": {"candidates": "onnx_exports"}},
        },
    )
    return root


def test_select_candidates_are_onnx_exports(tmp_path: Path) -> None:
    from ila.train.select import candidate_sha256, candidates

    run = _onnx_run(tmp_path / "a")
    got = candidates(run)
    assert [Path(c.file).name for c in got] == ["step10000", "step20000"]
    assert [c.global_step for c in got] == [10000, 20000]
    assert all(c.sha256 == candidate_sha256(Path(c.file)) for c in got)
    assert all(c.sha256 == onnx_dp._sha256(Path(c.file) / "meta.json") for c in got)
    # An export of another checkpoint, or a missing export, is refused.
    bad = _onnx_run(tmp_path / "b")
    _rewrite_meta(bad / "onnx" / "step20000", source={"checkpoint_sha256": "f" * 64})
    ox.write_parity(bad / "onnx" / "step20000")
    with pytest.raises(ValueError, match="another checkpoint"):
        candidates(bad)
    missing = _onnx_run(tmp_path / "c")
    for f in (missing / "onnx" / "step10000").iterdir():
        f.unlink()
    (missing / "onnx" / "step10000").rmdir()
    with pytest.raises(ValueError, match="exports at steps"):
        candidates(missing)


def test_select_parity_gate(tmp_path: Path) -> None:
    """No parity.json refuses the run; a failed parity bars that checkpoint (P6-D3)."""
    from ila.train.select import candidates, onnx_parity_exclusions

    run = _onnx_run(tmp_path / "a")
    (run / "onnx" / "step10000" / "parity.json").unlink()
    with pytest.raises(ValueError, match="no parity.json"):
        candidates(run)
    ox.write_parity(run / "onnx" / "step10000", passed=False)
    assert [Path(c.file).name for c in candidates(run)] == ["step20000"]
    assert [e["step"] for e in onnx_parity_exclusions(run)] == [10000]
    ox.write_parity(run / "onnx" / "step20000", passed=False)
    with pytest.raises(ValueError, match="no export passed parity"):
        candidates(run)
    # A parity.json written for other files does not count.
    ox.write_parity(run / "onnx" / "step20000")
    _rewrite_meta(run / "onnx" / "step20000", note="changed after parity")
    with pytest.raises(ValueError, match="another meta.json"):
        candidates(run)


def test_fly_checkpoint_builds_the_onnx_student(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ila import p8_adapter
    from ila.eval import runner
    from ila.train import select

    seen: dict[str, Any] = {}

    def fake(episodes: Any, spec: Any, cfgs: Any, tags: Any, **kwargs: Any) -> list[Any]:
        seen.update(kwargs, spec=spec, tags=tags)
        return []

    monkeypatch.setattr(runner, "run_list_vision", fake)
    b = ox.write_bundle(tmp_path / "onnx" / "step10000")
    select.fly_checkpoint(b, [], arm="R1_lerobot_dp", run_seed=2, sha256="s", workers=1)
    assert seen["contract_variant"] == "default" and seen["render"] == "full"
    st = seen["spec"].build(p8_adapter.load_eval_configs())
    assert isinstance(st, OnnxDPStudent) and st.run_seed == 2
    assert (st.chunk_spec.n_action_steps, st.num_inference_steps) == (8, 10)
    select.fly_checkpoint(
        b, [], arm="R1_lerobot_dp", run_seed=2, sha256="s", workers=1, inference=(4, 5, True)
    )
    st = seen["spec"].build(p8_adapter.load_eval_configs())
    assert (st.chunk_spec.n_action_steps, st.num_inference_steps, st.temporal_ensembling) == (
        4,
        5,
        True,
    )


@pytest.mark.pybullet
def test_onnx_student_flies_train_episodes_on_the_cpu_path(tmp_path: Path) -> None:
    """The synthetic export flies through the runner; rows do not depend on the worker count."""
    from ila.data import episodes as data_episodes
    from ila.train.select import candidate_sha256, fly_checkpoint

    b = ox.write_bundle(tmp_path / "onnx" / "step10000")
    listed = data_episodes.read_split("demos_0-999")[:2]
    sha = candidate_sha256(b)
    rows = [
        fly_checkpoint(b, listed, arm="R1_lerobot_dp", run_seed=0, sha256=sha, workers=w, chunk=1)
        for w in (1, 2)
    ]
    assert [{k: repr(v) for k, v in r.items()} for r in rows[0]] == [
        {k: repr(v) for k, v in r.items()} for r in rows[1]
    ]  # repr: NaN cells compare equal
    assert [r["episode_seed"] for r in rows[0]] == [e.episode_seed for e in listed]
    assert all(r["ckpt_sha256"] == sha for r in rows[0])
