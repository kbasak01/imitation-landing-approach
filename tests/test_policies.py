"""Phase 2 dummy students: the real ``bc`` and ``dp`` architectures with seeded random init.

CPU tests: encoder forward at 96 px, the crop convention, the ``EvalPolicy`` shape (a raw
25-d observation raises), seeded initialisation, per-slot noise seeding independent of batch
position and size, chunk execution from index ``To - 1``, and a parameter-count print. GPU
tests are marked ``gpu``.
"""

from __future__ import annotations

import functools
import math
import pickle
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ila import p8_adapter
from ila.obs.split import load_contract
from ila.policies import registry
from ila.policies.bc import BCNet, BCStudent
from ila.policies.dp import chunking
from ila.policies.dp.policy import (
    DiffusionPolicyNet,
    DPConfig,
    DPStudent,
    initial_noise,
    noise_seed,
)
from ila.policies.dp.unet1d import ConditionalUnet1D, UNetConfig
from ila.policies.encoder import (
    EncoderConfig,
    VisionEncoder,
    centre_crop,
    count_parameters,
    random_crop,
)

TO, PX, D = 2, 96, 13


def _window(seed: int = 0, b: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape: tuple[int, ...] = (TO,) if b is None else (b, TO)
    img = rng.integers(0, 256, (*shape, PX, PX, 3), dtype=np.uint8)
    prop = rng.normal(size=(*shape, D)).astype(np.float32)
    return img, prop


#: Network-level agreement across batch sizes and devices (fp32 last ulps; measured <= 2.3e-6
#: on the CPU at 1-16 threads for the dummy ``dp``).
EPS_ATOL = 1e-5


def _first_step_eps(
    student: DPStudent, img: np.ndarray, prop: np.ndarray, seeds: list[tuple[int, int, int]]
) -> np.ndarray:
    """Return epsilon-hat at the sampler's first timestep: the network output, no DDIM step."""
    net = student.net
    k0 = int(net.ddim(student.num_inference_steps).timesteps[0])
    noise = initial_noise(seeds, net.cfg.horizon, net.cfg.action_dim).to(student.device)
    with torch.inference_mode():
        cond = net.encoder(
            torch.from_numpy(img).to(student.device), torch.from_numpy(prop).to(student.device)
        )
        k = torch.full((len(seeds),), k0, dtype=torch.long, device=student.device)
        eps: np.ndarray = net.predict_noise(noise, k, cond).cpu().numpy()
    return eps


def _x0_gain(student: DPStudent) -> float:
    """``sqrt((1 - abar_k0) / abar_k0)``: d x0_hat / d eps_hat at the first DDIM timestep."""
    sched = student.net.ddim(student.num_inference_steps)
    abar = float(sched.alphas_cumprod[int(sched.timesteps[0])])
    return math.sqrt((1.0 - abar) / abar)


@pytest.fixture(scope="module")
def bc() -> BCStudent:
    s = registry.build_dummy("bc", 0)
    assert isinstance(s, BCStudent)
    return s


@pytest.fixture(scope="module")
def dp() -> DPStudent:
    s = registry.build_dummy("dp", 0)
    assert isinstance(s, DPStudent)
    return s


# ----------------------------------------------------------------------------- configs


def test_configs_as_read() -> None:
    enc = registry.encoder_config(registry.load_train_config("dp"))
    assert enc == EncoderConfig(
        groups=16, keypoints=32, proprio_dim=13, proprio_mlp=(64, 64), input_px=96, crop_px=88
    )
    assert enc.feature_dim == 256
    dp_cfg = DPConfig.from_mapping(registry.load_train_config("dp"))
    assert (dp_cfg.n_obs_steps, dp_cfg.horizon, dp_cfg.n_action_steps) == (2, 16, 8)
    assert dp_cfg.unet == UNetConfig(down_dims=(128, 256, 512), kernel_size=5)
    assert dp_cfg.num_inference_steps == 10 and dp_cfg.num_train_timesteps == 100
    assert dp_cfg.timestep_spacing == "trailing"  # P3-D1; the config is the authority
    # The policy's To must equal the observation contract's.
    assert dp_cfg.n_obs_steps == load_contract().n_obs_steps
    assert registry.encoder_config(registry.load_train_config("bc")) == enc


def test_dp_timestep_spacing_default_and_validation() -> None:
    # Without the key, a config gets P3-D1's trailing spacing, not diffusers' leading.
    cfg = registry.load_train_config("dp")
    inf = {k: v for k, v in cfg["inference"].items() if k != "timestep_spacing"}
    assert DPConfig.from_mapping({**cfg, "inference": inf}).timestep_spacing == "trailing"
    assert DPConfig().timestep_spacing == "trailing"
    with pytest.raises(ValueError, match="timestep_spacing"):
        DPConfig(timestep_spacing="trailling")


def test_repo_root_matches_adapter() -> None:
    assert registry.REPO_ROOT == p8_adapter.REPO_ROOT


# ----------------------------------------------------------------------------- encoder


def test_encoder_is_groupnorm_resnet18() -> None:
    enc = VisionEncoder(EncoderConfig())
    mods = list(enc.modules())
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in mods)
    gns = [m for m in enc.trunk.modules() if isinstance(m, nn.GroupNorm)]
    assert len(gns) == 20 and all(m.num_groups == 16 for m in gns)  # ResNet-18: 20 norms


def test_encoder_forward_96px() -> None:
    enc = VisionEncoder(EncoderConfig()).eval()
    img, prop = _window(b=3)
    with torch.inference_mode():
        trunk = enc.trunk(torch.zeros(1, 3, 88, 88))
        out = enc(torch.from_numpy(img), torch.from_numpy(prop))
    assert trunk.shape == (1, 512, 3, 3)
    assert out.shape == (3, 256) and out.dtype == torch.float32
    assert torch.isfinite(out).all()
    kp = out[:, : TO * 64]
    assert kp.abs().max() <= 1.0  # spatial-softmax expectations lie in [-1, 1]


def test_encoder_rejects_wrong_input() -> None:
    enc = VisionEncoder(EncoderConfig()).eval()
    img, prop = _window(b=1)
    with pytest.raises(ValueError, match="uint8"):
        enc(torch.from_numpy(img).float(), torch.from_numpy(prop))
    with pytest.raises(ValueError, match="images must be"):
        enc(torch.from_numpy(img[:, :, :88, :88]), torch.from_numpy(prop))
    with pytest.raises(ValueError, match="proprio must be"):
        enc(torch.from_numpy(img), torch.zeros(1, TO, 25))


def test_centre_crop_is_centred_at_eval() -> None:
    x = torch.arange(96 * 96, dtype=torch.float32).reshape(1, 1, 96, 96)
    c = centre_crop(x, 88)
    assert c.shape == (1, 1, 88, 88)
    assert c[0, 0, 0, 0] == x[0, 0, 4, 4] and c[0, 0, -1, -1] == x[0, 0, 91, 91]


def test_eval_mode_uses_centre_crop() -> None:
    # Pixels outside the centre 88 px must not change the eval-mode output.
    enc = VisionEncoder(EncoderConfig()).eval()
    img, prop = _window(b=1)
    img2 = img.copy()
    img2[..., :4, :, :] = 0
    img2[..., -4:, :, :] = 255
    img2[..., :, :4, :] = 7
    with torch.inference_mode():
        a = enc(torch.from_numpy(img), torch.from_numpy(prop))
        b = enc(torch.from_numpy(img2), torch.from_numpy(prop))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_random_crop_needs_generator_and_stays_in_bounds() -> None:
    enc = VisionEncoder(EncoderConfig()).train()
    img, prop = _window(b=2)
    with pytest.raises(ValueError, match="crop_generator"):
        enc(torch.from_numpy(img), torch.from_numpy(prop))
    x = torch.arange(96 * 96, dtype=torch.float32).reshape(1, 1, 96, 96).repeat(50, 1, 1, 1)
    g = torch.Generator().manual_seed(0)
    c = random_crop(x, 88, g)
    corners = c[:, 0, 0, 0].long()
    rows, cols = corners // 96, corners % 96
    assert c.shape == (50, 1, 88, 88)
    assert rows.min() >= 0 and rows.max() <= 8 and cols.min() >= 0 and cols.max() <= 8
    assert len(set(zip(rows.tolist(), cols.tolist(), strict=True))) > 1
    g2 = torch.Generator().manual_seed(0)
    torch.testing.assert_close(random_crop(x, 88, g2), c, rtol=0, atol=0)


# ----------------------------------------------------------------------------- EvalPolicy shape


@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_act_with_raw_obs_raises(arm: str, request: pytest.FixtureRequest) -> None:
    student = request.getfixturevalue(arm)
    student.reset(1)
    with pytest.raises(TypeError, match="StudentInput"):
        student.act(np.zeros(25, dtype=np.float32))
    with pytest.raises(TypeError):
        student.act(np.zeros((TO, 25), dtype=np.float32))
    img, _ = _window()
    with pytest.raises(ValueError, match="proprio"):
        student.act_vision(img, np.zeros((TO, 25), dtype=np.float32))
    with pytest.raises(ValueError, match="proprio"):
        student.act_vision(img, np.zeros(25, dtype=np.float32))


@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_reset_refuses_privileged_context(arm: str, request: pytest.FixtureRequest) -> None:
    student = request.getfixturevalue(arm)
    with pytest.raises(ValueError, match="privileged"):
        student.reset(1, context=object())
    with pytest.raises(ValueError, match="privileged"):
        student.reset(1, None, object())


@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_act_accepts_student_input(arm: str, request: pytest.FixtureRequest) -> None:
    student = request.getfixturevalue(arm)
    contract = load_contract()
    img, _ = _window()
    obs = np.random.default_rng(3).normal(size=(TO, contract.obs_dim)).astype(np.float32)
    inp = contract.student_input(img, obs)
    student.reset(5)
    a = student.act(inp)
    student.reset(5)
    b = student.act_vision(inp.images, inp.proprio)
    assert a.shape == (3,) and a.dtype == np.float32
    assert np.all(np.abs(a) <= 1.0)
    np.testing.assert_array_equal(a, b)


def test_bc_act_batch_matches_single(bc: BCStudent) -> None:
    img, prop = _window(b=4)
    batch = bc.act_batch(img, prop)
    assert batch.shape == (4, 3) and np.all(np.abs(batch) <= 1.0)
    for i in range(4):
        np.testing.assert_allclose(bc.act_vision(img[i], prop[i]), batch[i], rtol=0, atol=1e-5)


# ----------------------------------------------------------------------------- noise seeding


def test_noise_seed_is_stable_and_distinct() -> None:
    s = noise_seed(0, 1490732229, 0)
    assert s == noise_seed(0, 1490732229, 0)
    assert 0 <= s < 2**64
    keys = {(r, e, t) for r in range(3) for e in (0, 1, 2**31) for t in range(20)}
    assert len({noise_seed(*k) for k in keys}) == len(keys)
    with pytest.raises(ValueError):
        noise_seed(0, 0, -1)


def test_noise_seed_pinned_value() -> None:
    # Pinned so a refactor cannot silently change every DP episode's samples.
    assert noise_seed(0, 0, 0) == int.from_bytes(
        __import__("hashlib").blake2b(bytes(24), digest_size=8).digest(), "little"
    )


def test_noise_depends_only_on_slot_seed_tuple() -> None:
    key = (3, 1490732229, 16)
    others = [(3, 17, 0), (4, 1490732229, 16), (3, 1490732229, 8), (0, 0, 0), (9, 9, 9)]
    alone = initial_noise([key], 16, 3)[0]
    for size in (1, 2, 8, 24):
        for pos in sorted({0, size // 2, size - 1}):
            seeds = [others[i % len(others)] for i in range(size)]
            seeds[pos] = key
            batch = initial_noise(seeds, 16, 3)
            assert batch.shape == (size, 16, 3)
            torch.testing.assert_close(batch[pos], alone, rtol=0, atol=0)
    assert not torch.equal(initial_noise([others[0]], 16, 3)[0], alone)


def test_noise_ignores_global_rng() -> None:
    key = [(0, 42, 8)]
    torch.manual_seed(1)
    a = initial_noise(key, 16, 3)
    torch.manual_seed(2)
    torch.rand(100)
    b = initial_noise(key, 16, 3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sampling_does_not_touch_global_rng(dp: DPStudent) -> None:
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    img, prop = _window(b=1)
    dp.sample_batch(img, prop, [(0, 1, 0)])
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)


def test_sample_slot_independent_of_batch(dp: DPStudent) -> None:
    img, prop = _window(seed=1, b=8)
    seeds = [(0, 100 + i, 8 * i) for i in range(8)]
    full = dp.sample_batch(img, prop, seeds)
    assert full.shape == (8, 16, 3) and np.all(np.abs(full) <= 1.0)

    # 1. At a fixed batch size a slot's chunk is bit-identical wherever it sits and whatever the
    #    other slots hold (inputs and noise keys). A slot-mixing bug (noise drawn per batch, a
    #    wrong generator index, cross-slot normalisation) fails here at O(1).
    perm = [5, 2, 7, 0, 3, 6, 1, 4]
    shuffled = dp.sample_batch(img[perm], prop[perm], [seeds[i] for i in perm])
    np.testing.assert_array_equal(shuffled, full[perm])
    other_img, other_prop = _window(seed=9, b=8)
    other_img[3], other_prop[3] = img[3], prop[3]
    other_seeds = [(5, 7 + i, i) for i in range(8)]
    other_seeds[3] = seeds[3]
    np.testing.assert_array_equal(dp.sample_batch(other_img, other_prop, other_seeds)[3], full[3])

    # 2. The CPU single-sample path is bit-identical when repeated (P2-D3 item 1).
    single = np.stack(
        [dp.sample_batch(img[i : i + 1], prop[i : i + 1], [seeds[i]])[0] for i in range(8)]
    )
    np.testing.assert_array_equal(dp.sample_batch(img[:1], prop[:1], [seeds[0]]), single[:1])

    # 3. Across batch sizes the same noise goes in and the network output agrees to the last
    #    ulps (CPU kernels differ by batch size) ...
    eps_single = np.concatenate(
        [_first_step_eps(dp, img[i : i + 1], prop[i : i + 1], [seeds[i]]) for i in range(8)]
    )
    np.testing.assert_allclose(eps_single, _first_step_eps(dp, img, prop, seeds), atol=EPS_ATOL)
    # ... but the chunk does not, by a bounded amount. DDIM with trailing spacing (P3-D1)
    # starts at k = 99, where squaredcos_cap_v2 over 100 steps has abar_99 = 2.43e-7 (beta_99
    # capped at 0.999). x0_hat = (x - sqrt(1 - abar) eps_hat) / sqrt(abar), so an eps_hat
    # difference reaches an x0_hat element times sqrt((1 - abar) / abar) = 2029. clip_sample
    # clips x0_hat to [-1, 1]; only elements with |x - sqrt(1 - abar) eps_hat| < 4.9e-4 escape
    # the clip (random init: about 1 in 400 to 3000). Later steps neither compound nor shrink
    # the x0_hat gap. Measured here (1 thread): eps_hat gap 1.25e-6 -> chunk gap 3.74e-4,
    # 7 of 384 elements above 1e-5, all in slot 5, the slot with the one unclipped element.
    # (With leading spacing the first step is k = 90, gain 7.1, chunk gap 2.9e-6.)
    gain = _x0_gain(dp)
    assert 2000.0 < gain < 2100.0  # pins the mechanism to the configured schedule
    np.testing.assert_allclose(single, full, rtol=0, atol=gain * EPS_ATOL)


# ----------------------------------------------------------------------------- chunk execution


def test_dp_executes_from_to_minus_one_and_replans_every_ta() -> None:
    student = registry.build_dummy("dp", 7, n_action_steps=8)
    assert isinstance(student, DPStudent)
    img, prop = _window(seed=2)
    episode = 1234
    student.reset(episode)
    actions = np.stack([student.act_vision(img, prop) for _ in range(20)])
    for anchor in (0, 8, 16):
        chunk = student.sample_batch(img[None], prop[None], [(7, episode, anchor)])[0]
        n = min(8, 20 - anchor)
        np.testing.assert_array_equal(actions[anchor : anchor + n], chunk[1 : 1 + n])
        assert not np.array_equal(actions[anchor], chunk[0])  # not from index 0
    # The same episode after reset is bit-identical; another episode seed differs.
    student.reset(episode)
    again = np.stack([student.act_vision(img, prop) for _ in range(20)])
    np.testing.assert_array_equal(actions, again)
    student.reset(episode + 1)
    other = student.act_vision(img, prop)
    assert not np.array_equal(other, actions[0])


@pytest.mark.parametrize("ta", [1, 15])
def test_dp_ta_inference_time(ta: int) -> None:
    student = registry.build_dummy("dp", 0, n_action_steps=ta, num_inference_steps=2)
    assert isinstance(student, DPStudent)
    assert student.chunk_spec.executed == chunking.executed_slice(2, 16, ta)
    img, prop = _window()
    student.reset(0)
    for _ in range(ta + 1):
        a = student.act_vision(img, prop)
        assert a.shape == (3,)


def test_dp_inference_args_validated() -> None:
    with pytest.raises(ValueError, match="Ta must satisfy"):
        registry.build_dummy("dp", 0, n_action_steps=16)
    with pytest.raises(ValueError, match="K must be"):
        registry.build_dummy("dp", 0, num_inference_steps=101)
    with pytest.raises(ValueError, match="K must be"):
        registry.build_dummy("dp", 0, num_inference_steps=0)
    with pytest.raises(ValueError, match="not a diffusion arm"):
        registry.build_dummy("bc", 0, n_action_steps=8)


def test_dp_act_before_reset_raises() -> None:
    student = registry.build_dummy("dp", 0)
    img, prop = _window()
    with pytest.raises(RuntimeError, match="reset"):
        student.act_vision(img, prop)


def test_ddim_timesteps_and_k(dp: DPStudent) -> None:
    # P3-D1: trailing spacing, so every K starts at k = 99 (leading gives [0] at K = 1).
    assert dp.net.cfg.timestep_spacing == "trailing"
    assert dp.net.ddim(10).timesteps.tolist() == [99, 89, 79, 69, 59, 49, 39, 29, 19, 9]
    assert dp.net.ddim(1).timesteps.tolist() == [99]
    assert dp.net.ddim(2).timesteps.tolist() == [99, 49]
    assert dp.net.ddim(100).timesteps.tolist() == list(range(99, -1, -1))
    img, prop = _window(b=1)
    for k in (1, 2, 5):
        out = dp.sample_batch(img, prop, [(0, 0, 0)], num_inference_steps=k)
        assert out.shape == (1, 16, 3) and np.all(np.abs(out) <= 1.0)


def test_unet_shapes_and_horizon_check() -> None:
    unet = ConditionalUnet1D(3, 256, UNetConfig())
    x = torch.randn(2, 16, 3)
    out = unet(x, torch.tensor([3, 50]), torch.randn(2, 256))
    assert out.shape == (2, 16, 3)
    out1 = unet(x, torch.tensor(7), torch.randn(2, 256))
    assert out1.shape == (2, 16, 3)
    with pytest.raises(ValueError, match="multiple of 4"):
        unet(torch.randn(1, 14, 3), torch.tensor(1), torch.randn(1, 256))


# ----------------------------------------------------------------------------- build_dummy


def _state(net: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_build_dummy_seeded_init(arm: str) -> None:
    a = _state(registry.build_net(arm, 0))
    b = _state(registry.build_net(arm, 0))
    c = _state(registry.build_net(arm, 1))
    assert a.keys() == b.keys() == c.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_build_dummy_leaves_global_rng_alone() -> None:
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    registry.build_net("bc", 0)
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)


def test_build_dummy_factory_pickles() -> None:
    factory = functools.partial(registry.build_dummy, "dp", 3, "cpu", n_action_steps=4)
    clone: Any = pickle.loads(pickle.dumps(factory))
    student = clone()
    assert isinstance(student, DPStudent)
    assert student.run_seed == 3 and student.chunk_spec.n_action_steps == 4
    assert isinstance(pickle.loads(pickle.dumps(registry.build_dummy))("bc", 0), BCStudent)


def test_parameter_counts(capsys: pytest.CaptureFixture[str]) -> None:
    bc_net = registry.build_net("bc", 0)
    dp_net = registry.build_net("dp", 0)
    assert isinstance(bc_net, BCNet) and isinstance(dp_net, DiffusionPolicyNet)
    counts = {
        "encoder": count_parameters(bc_net.encoder),
        "encoder.trunk": count_parameters(bc_net.encoder.trunk),
        "bc.head": count_parameters(bc_net.head),
        "bc.total": count_parameters(bc_net),
        "dp.unet": count_parameters(dp_net.unet),
        "dp.total": count_parameters(dp_net),
    }
    with capsys.disabled():
        print("\nparameter counts: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    assert counts["encoder"] == count_parameters(dp_net.encoder)  # one encoder for every arm
    assert counts["bc.total"] == counts["encoder"] + counts["bc.head"]
    assert counts["dp.total"] == counts["encoder"] + counts["dp.unet"]
    assert 11_000_000 < counts["encoder.trunk"] < 11_300_000  # ResNet-18 minus fc


# ----------------------------------------------------------------------------- GPU


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("arm", ["bc", "dp"])
def test_gpu_matches_cpu(arm: str) -> None:
    cpu = registry.build_dummy(arm, 0, "cpu")
    gpu = registry.build_dummy(arm, 0, "cuda")
    sc, sg = _state(cpu.net), _state(gpu.net)
    assert all(torch.equal(sc[k], sg[k]) for k in sc)  # same init on any device
    img, prop = _window(seed=4, b=4)
    # cuDNN's default TF32 convolutions move a random-init DP chunk by up to ~3e-3 over 10
    # DDIM steps (measured with leading spacing); with fp32 convolutions the network outputs
    # agree to ~2e-6. The flag is the determinism contract's to set (P2-D3), so the policy
    # leaves it alone.
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        if isinstance(cpu, DPStudent) and isinstance(gpu, DPStudent):
            seeds = [(0, i, 0) for i in range(4)]
            np.testing.assert_allclose(
                _first_step_eps(gpu, img, prop, seeds),
                _first_step_eps(cpu, img, prop, seeds),
                atol=EPS_ATOL,
            )
            # The chunk gap is that times up to sqrt((1 - abar_99) / abar_99) = 2029 on any
            # x0_hat element the k = 99 clip does not catch (see
            # test_sample_slot_independent_of_batch). These 4 slots: 9.8e-6; over 64 slots,
            # 7.7e-4 in one slot.
            np.testing.assert_allclose(
                gpu.sample_batch(img, prop, seeds),
                cpu.sample_batch(img, prop, seeds),
                atol=_x0_gain(cpu) * EPS_ATOL,
            )
        elif isinstance(cpu, BCStudent) and isinstance(gpu, BCStudent):
            np.testing.assert_allclose(
                gpu.act_batch(img, prop), cpu.act_batch(img, prop), atol=1e-5
            )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_slot_noise_independent_of_batch() -> None:
    gpu = registry.build_dummy("dp", 0, "cuda")
    assert isinstance(gpu, DPStudent)
    img, prop = _window(seed=5, b=24)
    seeds = [(0, 1000 + i, 0) for i in range(24)]
    perm = [int(i) for i in np.random.default_rng(0).permutation(24)]
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        full = gpu.sample_batch(img, prop, seeds)
        shuffled = gpu.sample_batch(img[perm], prop[perm], [seeds[i] for i in perm])
        one = gpu.sample_batch(img[10:11], prop[10:11], seeds[10:11])
        eps_one = _first_step_eps(gpu, img[10:11], prop[10:11], seeds[10:11])
        eps_full = _first_step_eps(gpu, img, prop, seeds)
    # Fixed batch size: bit-identical under a slot permutation (measured).
    np.testing.assert_array_equal(shuffled, full[perm])
    # Across batch sizes: network output to the last ulps, chunk within the k = 99 gain.
    np.testing.assert_allclose(eps_one[0], eps_full[10], atol=EPS_ATOL)
    np.testing.assert_allclose(one[0], full[10], atol=_x0_gain(gpu) * EPS_ATOL)
