"""Phase 9 export: settings never relax P3-D1, the unrolled DDIM is ila's sampler, graph checks.

CPU only, tiny networks (``tests/deploy_support.py``), no PyBullet.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch

from deploy_support import tiny_arms, tiny_inputs, tiny_settings
from ila.deploy import export as ex
from ila.deploy import r1_parity
from ila.deploy.backends import OrtBackend, TorchBackend
from ila.deploy.settings import load_settings
from ila.policies.dp.policy import ddim_sample

#: The step guard's allowlist (tests/test_step_guard.py): what a network entry point may take.
ALLOWED = {"images", "proprio", "sample", "timestep", "global_cond", "noise"}


def test_bench_yaml_loads_with_p3d1_numbers() -> None:
    s = load_settings()
    assert (s.eps_max_abs, s.action_max_abs, s.closed_loop_max_flips) == (1e-4, 1e-4, 3)
    assert (s.warmup_iters, s.timed_iters, s.opset, s.n_inputs) == (200, 2000, 17, 1000)
    assert s.input_list == "select" and s.reference == "torch_cpu_fp32"
    assert set(s.scored) == {"ort_cpu_fp32", "ort_cuda_fp32", "torch_cuda_fp32", "ort_trt_fp16"}
    assert s.control_period_ms == 33.333 and s.h5a_threshold_ms == 33.3
    assert 1 in s.batch_sizes and s.eval_batch == 24


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("parity.eps_max_abs", 2e-4),
        ("parity.action_max_abs", 1e-3),
        ("parity.closed_loop_max_flips", 4),
        ("latency.timed_iters", 1999),
        ("latency.warmup_iters", 100),
        ("inputs.list", "id"),
        ("inputs.n", 500),
        ("export.opset", 18),
        ("control_period_ms", 33.3),
        ("providers.scored", ["ort_cpu_fp32", "ort_cuda_fp32", "torch_cuda_fp32"]),
    ],
)
def test_settings_refuse_relaxed_thresholds(tmp_path: Path, key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        tiny_settings(tmp_path, **{key: value})


@pytest.mark.parametrize("k", [1, 5, 10, 25])
def test_unrolled_ddim_is_ilas_sampler_bit_for_bit(k: int) -> None:
    dp = tiny_arms()["dp"].net
    g = torch.Generator().manual_seed(3)
    cond = torch.randn((3, dp.encoder.cfg.feature_dim), generator=g)
    noise = torch.randn((3, 16, 3), generator=g)
    rec = ex.unrolled_matches_sampler(dp, k, cond, noise)  # type: ignore[arg-type]
    assert rec["bit_identical"] and rec["ok"], rec
    un = ex.UnrolledDDIM(dp, k)  # type: ignore[arg-type]
    assert list(un.timesteps) == [int(t) for t in dp.ddim(k).timesteps]  # type: ignore[union-attr]
    assert un.timesteps[0] == 99  # trailing


@pytest.mark.parametrize("cls", [ex.BCGraph, ex.EncoderGraph, ex.DenoiseStepGraph, ex.UnrolledDDIM])
def test_graph_entry_points_take_only_allowlisted_inputs(cls: type) -> None:
    params = [p for p in inspect.signature(cls.forward).parameters if p != "self"]
    assert set(params) <= ALLOWED, (cls.__name__, params)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], Any]:
    tmp = tmp_path_factory.mktemp("deploy")
    s = tiny_settings(tmp)
    arms = tiny_arms()
    out = tmp / "graphs"
    meta = ex.export_loaded(arms, s, out)
    return out, meta, arms


def test_export_writes_every_graph_opset17_no_rng(
    exported: tuple[Path, dict[str, Any], Any],
) -> None:
    out, meta, _ = exported
    names = {Path(ex.graph_path(out, k, p)).name for k in ex.GRAPH_FILES for p in ("fp32", "fp16")}
    assert set(meta["graphs"]) == names
    assert {"bc_dagger.onnx", "dp_encoder.onnx", "dp_denoise_step.onnx", "dp_K10.onnx"} <= names
    for name, rec in meta["graphs"].items():
        model = onnx.load(str(out / name))
        assert {o.domain or "ai.onnx": o.version for o in model.opset_import}["ai.onnx"] == 17
        assert r1_parity.rng_ops(model) == [] and rec["rng_ops"] == []
        assert rec["sha256"] == ex.sha256_file(out / name)
    ex.verify_graphs(out)
    assert meta["dp"]["timesteps"] == [99, 89, 79, 69, 59, 49, 39, 29, 19, 9]
    assert meta["dp"]["unrolled_vs_ddim_sample_torch_cpu"]["bit_identical"]


def test_graph_io_contract_is_kept_in_fp16(exported: tuple[Path, dict[str, Any], Any]) -> None:
    _, meta, _ = exported
    for key, (ins, outs) in ex.GRAPH_IO.items():
        for prec in ("fp32", "fp16"):
            rec = meta["graphs"][Path(ex.graph_path(Path(), key, prec)).name]
            assert [i["name"] for i in rec["inputs"]] == list(ins)
            assert [o["name"] for o in rec["outputs"]] == list(outs)
            types = {i["name"]: i["elem_type"] for i in rec["inputs"]}
            assert types.get("images", "UINT8") == "UINT8"
            assert types.get("timestep", "INT64") == "INT64"
            assert all(o["elem_type"] == "FLOAT" for o in rec["outputs"])
            assert all(i["shape"][0] == "batch" for i in rec["inputs"])
    # The unrolled graph's inputs are the conditioning and the initial noise only.
    assert [i["name"] for i in meta["graphs"]["dp_K10.onnx"]["inputs"]] == ["global_cond", "noise"]


def test_tampered_graph_is_refused(
    exported: tuple[Path, dict[str, Any], Any], tmp_path: Path
) -> None:
    out, meta, _ = exported
    copy = tmp_path / "g"
    copy.mkdir()
    for name in meta["graphs"]:
        (copy / name).write_bytes((out / name).read_bytes())
    (copy / ex.META_FILE).write_text(json.dumps(meta))
    with (copy / "dp_K10.onnx").open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ex.ExportError, match="SHA-256"):
        ex.verify_graphs(copy)


def test_ort_cpu_matches_torch_on_tiny_graphs(exported: tuple[Path, dict[str, Any], Any]) -> None:
    out, meta, arms = exported
    ort = OrtBackend("ort_cpu_fp32", out, ex.dp_config_from_meta(meta))
    ref = TorchBackend("torch_cpu_fp32", arms)
    x = tiny_inputs(3)
    noise = np.random.default_rng(0).standard_normal((3, 16, 3)).astype(np.float32)
    c = ref.encode(x.images, x.proprio)
    assert np.abs(ort.bc(x.images, x.proprio) - ref.bc(x.images, x.proprio)).max() <= 1e-5
    assert np.abs(ort.encode(x.images, x.proprio) - c).max() <= 1e-4
    t = np.full((3,), 99, np.int64)
    assert np.abs(ort.denoise(noise, t, c) - ref.denoise(noise, t, c)).max() <= 1e-5
    assert np.abs(ort.unrolled(c, noise) - ref.unrolled(c, noise)).max() <= 1e-3
    assert np.abs(ort.loop(c, noise, 10) - ort.unrolled(c, noise)).max() <= 1e-3
    # torch's unrolled module and ila's loop are the same arithmetic, bit for bit.
    assert np.array_equal(ref.unrolled(c, noise), ref.loop(c, noise, 10))
    dp = arms["dp"].net
    with torch.inference_mode():
        loop = ddim_sample(
            lambda xx, kk: dp.predict_noise(xx, kk, torch.from_numpy(c)),
            torch.from_numpy(noise),
            dp.ddim(10),
        )
    assert np.array_equal(loop.numpy(), ref.loop(c, noise, 10))


def test_pin_mismatch_is_refused_before_bytes_are_read(tmp_path: Path) -> None:
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"pins": {"bc_dagger/s0": "f" * 64}}))
    with pytest.raises(ex.ExportError, match="pin"):
        ex.resolve_pinned("bc_dagger", 0, pins)


def test_committed_pins_name_the_selection_records() -> None:
    """The two exported (arm, seed) pins equal the committed selection.json hashes."""
    s = load_settings()
    pins = json.loads(s.selection_pins.read_text())["pins"]
    from ila.eval import arms as ev_arms

    for arm in ("bc_dagger", "dp"):
        path = ev_arms.selection_path(arm, 0)
        assert pins[f"{arm}/s0"] == ex.sha256_file(path)
