"""The GPU-path server adapters reproduce the single-sample student, request by request.

``SharedDPPolicy`` / ``SharedBCPolicy`` hold one shared dummy network (random init,
``ila.policies.registry.build_dummy``) and serve many slots. Fed synthetic lockstep requests,
on CPU with one request per call they must equal each slot's own ``act_vision`` sequence bit
for bit (same noise keys, same executed indices, same replan steps). Batched calls are not
claimed bit-identical; they are checked to a tolerance only. No PyBullet needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ila.eval import batched
from ila.obs.split import load_contract
from ila.policies.registry import build_dummy

TA, K, STEPS, SLOTS = 4, 2, 9, 2


def _requests(run_seed: int) -> list[list[batched.SlotRequest]]:
    """Lockstep rounds: ``SLOTS`` episodes of ``STEPS`` steps with random pixels/proprio."""
    contract = load_contract()
    rng = np.random.default_rng(0)
    rounds = []
    for t in range(STEPS):
        rnd = []
        for slot in range(SLOTS):
            images = rng.integers(0, 256, (contract.n_obs_steps, 96, 96, 3), dtype=np.uint8)
            proprio = rng.normal(size=(contract.n_obs_steps, contract.proprio_dim)).astype(
                np.float32
            )
            key = batched.NoiseKey(run_seed, 1000 + slot, t)
            rnd.append(batched.SlotRequest(slot, key, images, proprio))
        rounds.append(rnd)
    return rounds


def _reference(arm: str, rounds: list[list[batched.SlotRequest]]) -> np.ndarray:
    """Each slot flown alone through its own student's ``act_vision`` (CPU single-sample)."""
    kw = {"n_action_steps": TA, "num_inference_steps": K} if arm == "dp" else {}
    out = np.empty((STEPS, SLOTS, 3))
    for slot in range(SLOTS):
        student = build_dummy(arm, 0, "cpu", **kw)
        student.reset(1000 + slot)
        for t in range(STEPS):
            req = rounds[t][slot]
            out[t, slot] = student.act_vision(req.images, req.proprio)
    return out


def _served(
    policy: batched.BatchPolicy, rounds: list[list[batched.SlotRequest]], cap: int
) -> np.ndarray:
    out = np.empty((STEPS, SLOTS, 3))
    for slot in range(SLOTS):
        policy.reset_slot(slot, 1000 + slot)
    for t, rnd in enumerate(rounds):
        for i in range(0, len(rnd), cap):
            out[t, i : i + cap] = policy.act_batch(rnd[i : i + cap])
    return out


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["dp", "bc"])
def test_shared_net_equals_single_sample_on_cpu(arm: str) -> None:
    rounds = _requests(0)
    kw = {"n_action_steps": TA, "num_inference_steps": K} if arm == "dp" else {}
    factory = batched.shared_net_factory(arm, 0, "cpu", **kw)
    policy = factory(None, SLOTS)  # type: ignore[arg-type]
    assert policy.numerics["cudnn_allow_tf32"] is False
    assert policy.numerics["matmul_allow_tf32"] is False
    ref = _reference(arm, rounds)
    one = _served(policy, rounds, cap=1)
    assert one.tobytes() == ref.tobytes()
    if isinstance(policy, batched.SharedDPPolicy):
        # Replans at t = 0, Ta, 2 Ta for every slot, one sample per call at cap 1.
        assert policy.sample_calls == [1] * (SLOTS * len(range(0, STEPS, TA)))
    batch = _served(factory(None, SLOTS), rounds, cap=SLOTS)  # type: ignore[arg-type]
    np.testing.assert_allclose(batch, ref, rtol=0, atol=1e-5)


def test_shared_dp_refuses_out_of_order_steps() -> None:
    class _Fake:
        run_seed = 0
        num_inference_steps = 1
        chunk_spec = type("S", (), {"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8})()

        def sample_batch(self, images, proprio, seeds, num_inference_steps=None):  # type: ignore[no-untyped-def]
            del proprio, num_inference_steps
            return np.zeros((len(seeds), 16, 3), dtype=np.float32) + images.shape[0]

    policy = batched.SharedDPPolicy(_Fake())  # type: ignore[arg-type]
    rnd = _requests(0)
    policy.reset_slot(0, 1000)
    policy.act_batch([rnd[0][0]])
    with pytest.raises(ValueError, match="expected 1"):
        policy.act_batch([rnd[2][0]])


@pytest.mark.gpu
@pytest.mark.slow
def test_shared_dp_on_gpu_is_close_to_cpu() -> None:
    """Measured, not bit-identical: GPU (TF32 off) vs CPU single-sample, dp chunk actions."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rounds = _requests(0)
    factory = batched.shared_net_factory("dp", 0, "cuda", n_action_steps=TA, num_inference_steps=K)
    gpu = _served(factory(None, SLOTS), rounds, cap=SLOTS)  # type: ignore[arg-type]
    ref = _reference("dp", rounds)
    gap = float(np.max(np.abs(gpu - ref)))
    print(f"dp GPU batch {SLOTS} vs CPU single-sample max |diff| = {gap:.3g}")
    assert gap < 1e-4
