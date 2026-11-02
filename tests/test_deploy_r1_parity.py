"""R1 ONNX export and parity (``scripts/r1/export_onnx.py`` + ``ila.deploy.r1_parity``).

Two layers:

* fast unit tests (no lerobot): the RNG-op scan (main graph, subgraphs, functions), the MIN_MAX
  unnormalisation against LeRobot's formula, ``meta.json`` format and SHA-256 checks;
* a fixture export (``data`` + ``slow``): ``.venv-lerobot`` builds a random-init R1-shaped
  checkpoint with the R1 dataset's statistics (``export_onnx.py fixture``), exports it on the CPU
  with 8 reference inputs, and the ``.venv`` side checks the graph contract (names, dtypes,
  shapes, symbolic batch axis, opset 17, no RNG op), dynamic batch, idempotency, tampering and
  numeric parity (per-step epsilon max-abs <= 1e-4, P3-D1 §6 H5(a)). About 45 s on one core.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ila import p8_adapter
from ila.deploy import r1_parity

REPO = Path(__file__).resolve().parents[1]
LEROBOT_PY = REPO / ".venv-lerobot" / "bin" / "python"
EXPORT = REPO / "scripts" / "r1" / "export_onnx.py"
DATASET = "r1_ppo_2000_first1000_lerobot_v3"
N_REF = 8


# ------------------------------------------------------------------------------ unit tests
def _model(nodes: list[Any], functions: list[Any] | None = None) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    g = helper.make_graph(nodes, "g", [x], [y])
    return helper.make_model(
        g, opset_imports=[helper.make_opsetid("", 17)], functions=functions or []
    )


def test_rng_scan_finds_random_ops_anywhere() -> None:
    clean = _model([helper.make_node("Relu", ["x"], ["y"])])
    assert r1_parity.rng_ops(clean) == []
    top = _model([helper.make_node("RandomNormalLike", ["x"], ["y"])])
    assert r1_parity.rng_ops(top) == ["ai.onnx::RandomNormalLike"]
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])
    branch = helper.make_graph(
        [helper.make_node("RandomUniformLike", ["x"], ["z"])], "b", [], [out]
    )
    plain = helper.make_graph([helper.make_node("Identity", ["x"], ["z"])], "c", [], [out])
    nested = _model([helper.make_node("If", ["x"], ["y"], then_branch=branch, else_branch=plain)])
    assert r1_parity.rng_ops(nested) == ["ai.onnx::RandomUniformLike"]
    fn = helper.make_function(
        "local",
        "f",
        ["a"],
        ["b"],
        [helper.make_node("Dropout", ["a"], ["b"])],
        [helper.make_opsetid("", 17)],
    )
    in_fn = _model([helper.make_node("f", ["x"], ["y"], domain="local")], [fn])
    assert r1_parity.rng_ops(in_fn) == ["ai.onnx::Dropout"]


def test_unnormalise_is_lerobot_min_max_inverse() -> None:
    meta = {
        "action_unnormalisation": {
            "mode": "MIN_MAX",
            "min": [-0.5, 0.25, 0.0],
            "max": [0.5, 0.75, 0.0],  # axis 2: min == max -> eps denominator
            "eps": 1e-8,
        }
    }
    x = np.array([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.3]], np.float32)
    got = r1_parity.unnormalise(meta, x)
    assert got.dtype == np.float32
    lo = np.array([-0.5, 0.25, 0.0], np.float32)
    d = np.array([1.0, 0.5, 1e-8], np.float32)
    want = (x + np.float32(1)) / np.float32(2) * d + lo
    assert np.array_equal(got, want)
    assert np.allclose(got[:, :2], [[-0.5, 0.5], [0.5, 0.25]])
    with pytest.raises(r1_parity.R1ExportError):
        r1_parity.unnormalise({"action_unnormalisation": {"mode": "MEAN_STD"}}, x)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_meta_format_and_hashes_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(r1_parity.R1ExportError):
        r1_parity.load_meta(tmp_path)
    for name in ("encoder.onnx", "denoiser.onnx", "reference.npz"):
        (tmp_path / name).write_bytes(name.encode())
    meta = {
        "format": r1_parity.FORMAT,
        "onnx_sha256": {n: _sha(tmp_path / n) for n in ("encoder.onnx", "denoiser.onnx")},
        "reference_sha256": _sha(tmp_path / "reference.npz"),
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    loaded = r1_parity.load_meta(tmp_path)
    r1_parity.verify_files(tmp_path, loaded)
    (tmp_path / "denoiser.onnx").write_bytes(b"tampered")
    with pytest.raises(r1_parity.R1ExportError, match="SHA-256"):
        r1_parity.verify_files(tmp_path, loaded)
    (tmp_path / "meta.json").write_text(json.dumps({**meta, "format": "other/0"}))
    with pytest.raises(r1_parity.R1ExportError, match="format"):
        r1_parity.load_meta(tmp_path)


# --------------------------------------------------------------------- the fixture export
def _dataset_root() -> Path:
    return Path(p8_adapter.load_paths()["data_root"]) / "lerobot" / DATASET


def _lerobot(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    out = subprocess.run(
        [str(LEROBOT_PY), str(EXPORT), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if out.returncode != 0:
        raise AssertionError(f"export_onnx.py {args[0]} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


@pytest.fixture(scope="module")
def step_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not LEROBOT_PY.is_file():
        pytest.skip(".venv-lerobot is not installed")
    try:
        ds = _dataset_root()
    except FileNotFoundError:
        pytest.skip("configs/paths.local.yaml missing")
    if not (ds / "meta" / "info.json").is_file():
        pytest.skip(f"R1 dataset not built at {ds}")
    run = tmp_path_factory.mktemp("r1_fixture_run")
    _lerobot("fixture", "--out", str(run), "--dataset-root", str(ds))
    _lerobot("export", "--run", str(run), "--dataset-root", str(ds), "--n-ref", str(N_REF))
    return run / "onnx" / "step0"


@pytest.mark.data
@pytest.mark.slow
def test_graph_contract(step_dir: Path) -> None:
    meta = r1_parity.load_meta(step_dir)
    r1_parity.verify_files(step_dir, meta)
    to, tp, c = meta["n_obs_steps"], meta["horizon"], meta["C"]
    assert (to, tp, meta["n_action_steps"], meta["action_dim"], meta["state_dim"]) == (
        2,
        16,
        8,
        3,
        13,
    )
    assert c == 2 * (13 + 64)  # per step: state + 32 keypoints x 2
    assert meta["prediction_type"] == "epsilon"
    assert meta["versions"]["opset"] == 17
    assert meta["scheduler"]["ila_inference"]["timesteps_at_default_k"] == list(range(99, 0, -10))
    assert meta["action_unnormalisation"]["mode"] == "MIN_MAX"
    want = {
        "encoder.onnx": (
            {"image": (TensorProto.UINT8, [to, 96, 96, 3]), "state": (TensorProto.FLOAT, [to, 13])},
            {"global_cond": (TensorProto.FLOAT, [c])},
        ),
        "denoiser.onnx": (
            {
                "sample": (TensorProto.FLOAT, [tp, 3]),
                "timestep": (TensorProto.INT64, []),
                "global_cond": (TensorProto.FLOAT, [c]),
            },
            {"eps": (TensorProto.FLOAT, [tp, 3])},
        ),
    }
    for name, (ins, outs) in want.items():
        model = onnx.load(str(step_dir / name))
        assert {o.domain or "ai.onnx": o.version for o in model.opset_import} == {"ai.onnx": 17}
        assert model.ir_version <= r1_parity.MAX_IR_VERSION
        for vals, spec in ((model.graph.input, ins), (model.graph.output, outs)):
            got = {v.name: v.type.tensor_type for v in vals}
            assert set(got) == set(spec), name
            for key, (dtype, rest) in spec.items():
                dims = got[key].shape.dim
                assert got[key].elem_type == dtype, (name, key)
                assert dims[0].dim_param and not dims[0].dim_value, (name, key, "batch axis")
                assert [d.dim_value for d in dims[1:]] == rest, (name, key)


@pytest.mark.data
@pytest.mark.slow
def test_no_rng_op_in_graphs(step_dir: Path) -> None:
    checks = r1_parity.graph_checks(step_dir)
    assert checks["ok"], checks
    for name in r1_parity.GRAPHS:
        assert checks[name]["rng_ops"] == []
        assert checks[name]["n_functions"] == 0


@pytest.mark.data
@pytest.mark.slow
def test_dynamic_batch_and_determinism(step_dir: Path) -> None:
    with np.load(step_dir / "reference.npz") as z:
        image, state, x = z["image"], z["state"], z["ddim_x"][0]
    enc = r1_parity.cpu_session(step_dir / "encoder.onnx")
    den = r1_parity.cpu_session(step_dir / "denoiser.onnx")
    one = np.concatenate(
        [r1_parity.encode(enc, image[i : i + 1], state[i : i + 1]) for i in range(N_REF)]
    )
    for b in (1, 3, N_REF):
        gc = r1_parity.encode(enc, image[:b], state[:b])
        assert gc.shape == (b, one.shape[1])
        assert np.max(np.abs(gc - one[:b])) <= 1e-5
        k = np.full((b,), 99, np.int64)
        eps = r1_parity.denoise(den, x[:b], k, gc)
        assert eps.shape == (b, 16, 3)
        assert np.array_equal(eps, r1_parity.denoise(den, x[:b], k, gc))  # same inputs, same eps
    # The noise is an input: a different x_k gives a different eps.
    gc = one[:1]
    k = np.full((1,), 99, np.int64)
    assert not np.array_equal(
        r1_parity.denoise(den, x[:1], k, gc), r1_parity.denoise(den, x[:1] + 0.5, k, gc)
    )


@pytest.mark.data
@pytest.mark.slow
def test_parity_against_torch_reference(step_dir: Path) -> None:
    rec = r1_parity.run_parity(step_dir, write=False)
    assert rec["pass"], {k: v for k, v in rec["judged"].items() if not v}
    assert rec["n_inputs"] == N_REF
    assert len(rec["eps_per_step_max_abs"]) == 10
    assert max(rec["eps_per_step_max_abs"]) <= r1_parity.EPS_TOL
    assert max(rec["eps_probe_max_abs"].values()) <= r1_parity.EPS_TOL
    assert rec["sampler_replay_bit_identical"] and rec["unnormalise_replay_bit_identical"]
    assert rec["scheduler_match"] == {"timesteps": True, "alphas_cumprod_bit_identical": True}
    # LeRobot's own select_action path equals the export's torch reference exactly.
    assert rec["lerobot_path_check"]["bit_identical"]
    assert rec["chunk_executed_vs_lerobot_select_action_max_abs"] < 1e-3
    assert np.isfinite(rec["chunk_norm_max_abs"])


@pytest.mark.data
@pytest.mark.slow
def test_export_is_idempotent_and_tampering_is_caught(step_dir: Path, tmp_path: Path) -> None:
    run = step_dir.parent.parent
    before = {p.name: p.stat().st_mtime_ns for p in step_dir.iterdir()}
    out = _lerobot(
        "export", "--run", str(run), "--dataset-root", str(_dataset_root()), "--n-ref", str(N_REF)
    )
    assert "up to date" in out
    assert {p.name: p.stat().st_mtime_ns for p in step_dir.iterdir()} == before
    copy = tmp_path / "step0"
    shutil.copytree(step_dir, copy)
    data = bytearray((copy / "encoder.onnx").read_bytes())
    data[-1] ^= 0xFF
    (copy / "encoder.onnx").write_bytes(bytes(data))
    with pytest.raises(r1_parity.R1ExportError, match="SHA-256"):
        r1_parity.run_parity(copy, write=False)
