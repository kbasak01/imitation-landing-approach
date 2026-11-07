"""Phase 9 open-loop parity: the judged quantities, refusal of a perturbed provider, inputs draw.

CPU tests on tiny networks; GPU providers are exercised only when CUDA is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deploy_support import tiny_arms, tiny_inputs, tiny_settings
from ila.deploy import export as ex
from ila.deploy import inputs as inp
from ila.deploy import parity as par
from ila.deploy.backends import OrtBackend, TorchBackend


class _Shifted:
    """A backend whose epsilon-hat (or action) is off by ``delta`` (a failing provider)."""

    def __init__(self, inner: Any, eps_delta: float = 0.0, bc_delta: float = 0.0) -> None:
        self.inner, self.eps_delta, self.bc_delta = inner, eps_delta, bc_delta
        self.provider = "shifted"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def bc(self, images: Any, proprio: Any) -> Any:
        return self.inner.bc(images, proprio) + np.float32(self.bc_delta)

    def denoise(self, sample: Any, timestep: Any, global_cond: Any) -> Any:
        return self.inner.denoise(sample, timestep, global_cond) + np.float32(self.eps_delta)


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("parity")
    s = tiny_settings(tmp)
    arms = tiny_arms()
    x = tiny_inputs(3)
    ref = par.reference(arms, x, s)
    return {"s": s, "arms": arms, "x": x, "ref": ref, "tmp": tmp}


def test_reference_records_the_sampler_path(setup: dict[str, Any]) -> None:
    ref, s = setup["ref"], setup["s"]
    assert ref.x.shape == (10, 3, 16, 3) and ref.eps.shape == ref.x.shape
    assert list(ref.timesteps) == [99, 89, 79, 69, 59, 49, 39, 29, 19, 9]
    # Step 0's sample is the initial noise keyed (run_seed, -100, i).
    noise = par.parity_noise(0, s.noise_episode_key, 3, 16, 3).numpy()
    assert np.array_equal(ref.x[0], noise) and np.array_equal(ref.noise, noise)
    assert np.all(np.abs(ref.chunk) <= 1.0)


def test_reference_against_itself_passes(setup: dict[str, Any]) -> None:
    b = TorchBackend("torch_cpu_fp32", setup["arms"])
    rec = par.provider_parity(b, setup["x"], setup["ref"], setup["s"])
    assert rec["bc_dagger"]["judged_max_abs"] == 0.0 and rec["bc_dagger"]["open_loop_pass"]
    assert rec["dp"]["judged_max_abs"] == 0.0 and rec["dp"]["open_loop_pass"]
    assert rec["dp"]["chunk_unrolled_max_abs"] == 0.0


@pytest.mark.parametrize(("eps", "bc", "fails"), [(2e-4, 0.0, "dp"), (0.0, 2e-4, "bc_dagger")])
def test_a_provider_off_by_more_than_1e4_fails(
    setup: dict[str, Any], eps: float, bc: float, fails: str
) -> None:
    b = _Shifted(TorchBackend("torch_cpu_fp32", setup["arms"]), eps, bc)
    rec = par.provider_parity(b, setup["x"], setup["ref"], setup["s"])  # type: ignore[arg-type]
    assert not rec[fails]["open_loop_pass"]
    other = "bc_dagger" if fails == "dp" else "dp"
    assert rec[other]["open_loop_pass"]


def test_chunk_gap_is_reported_not_judged(setup: dict[str, Any]) -> None:
    """An epsilon gap under 1e-4 passes even when the chunk moves by more than 1e-3."""
    b = _Shifted(TorchBackend("torch_cpu_fp32", setup["arms"]), 9e-5)
    rec = par.provider_parity(b, setup["x"], setup["ref"], setup["s"])  # type: ignore[arg-type]
    assert rec["dp"]["open_loop_pass"]
    assert "chunk_loop_max_abs" in rec["dp"] and rec["dp"]["chunk_reported_bound"] == 1e-3


def test_ort_cpu_on_tiny_export_passes(setup: dict[str, Any]) -> None:
    out = setup["tmp"] / "graphs"
    meta = ex.export_loaded(setup["arms"], setup["s"], out)
    b = OrtBackend("ort_cpu_fp32", out, ex.dp_config_from_meta(meta))
    rec = par.provider_parity(b, setup["x"], setup["ref"], setup["s"])
    assert rec["bc_dagger"]["open_loop_pass"] and rec["dp"]["open_loop_pass"], rec
    csv_text = par.parity_csv(
        {"ort_cpu_fp32": rec, "ort_cuda_fp32": {"status": "failed: x"}}, setup["s"]
    )
    lines = csv_text.splitlines()
    assert lines[0].split(",")[:3] == ["arm", "provider", "scored"]
    failed = [ln for ln in lines if ",ort_cuda_fp32," in ln]
    assert len(failed) == 2 and all(",failed: x,,," in ln for ln in failed)
    assert par.steps_csv({"ort_cpu_fp32": rec}, setup["s"]).count("\n") == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("provider", ["ort_cuda_fp32", "torch_cuda_fp32"])
def test_gpu_fp32_providers_pass_on_tiny_export(setup: dict[str, Any], provider: str) -> None:
    from ila.deploy.backends import make_backend

    out = setup["tmp"] / "graphs_gpu"
    meta = ex.export_loaded(setup["arms"], setup["s"], out)
    b = make_backend(
        provider,
        graph_dir=out,
        dp_cfg=ex.dp_config_from_meta(meta),
        arms_loader=lambda: tiny_arms(),
    )
    rec = par.provider_parity(b, setup["x"], setup["ref"], setup["s"])
    assert rec["dp"]["open_loop_pass"] and rec["bc_dagger"]["open_loop_pass"], rec


def test_input_draw_is_seeded_sorted_without_replacement() -> None:
    a = inp.draw(9000, 1000, 20261023)
    assert np.array_equal(a, inp.draw(9000, 1000, 20261023))
    assert len(np.unique(a)) == 1000 and np.all(np.diff(a) > 0)
    assert not np.array_equal(a, inp.draw(9000, 1000, 1))
    with pytest.raises(ValueError):
        inp.draw(10, 11, 0)


def test_array_hash_sees_every_array(setup: dict[str, Any]) -> None:
    arrays = setup["ref"].arrays()
    h = par.array_sha256(arrays)
    changed = dict(arrays)
    changed["chunk"] = arrays["chunk"].copy()
    changed["chunk"][0, 0, 0] += 1
    assert par.array_sha256(changed) != h


def test_deploy_reads_no_frozen_list() -> None:
    """``ila/deploy`` reads ``select`` only: no frozen-list reader is named in it."""
    import ila.deploy

    root = Path(ila.deploy.__file__).parent
    banned = (
        "list_episodes",
        "read_list",
        "list_paths",
        "list_path(",
        "committed_rows",
        "frozen_list_cells",
        "load_matrix",
    )
    for f in sorted(root.glob("*.py")):
        text = f.read_text("utf-8")
        assert not [b for b in banned if b in text], f.name
