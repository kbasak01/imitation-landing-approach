"""R1's intermediate export (``ila.data.export_lerobot``; Phase 6, ``.venv`` side).

The exported episode set must be ``dp``'s training set, chosen by the ``dp`` training path itself
(``loop.build_stages`` -> ``load_stage_data`` -> ``choose_episodes``), and the arrays must be the
store's student inputs and clean labels, byte for byte (never ``action_exec``, never a privileged
array). Synthetic store as in ``test_train_loop.py``; the ``data``-marked test re-checks the real
export when it exists.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.data import export_lerobot as ex
from ila.data.store import DemoStore
from ila.train import loop
from ila.train.config import TrainConfig
from ila.train.sampling import choose_episodes
from test_data_store import CAMERA, synthetic_record, write_store

N_EPISODES = 8
BOUNCE = 2  # stream index of the one non-success episode
FIRST_N = 4
PROVENANCE: dict[str, Any] = {
    "p8_commit": "p8",
    "ila_commit": "ila",
    "ila_dirty_paths": [],
    "camera_cfg_sha256": CAMERA,
}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    records = []
    for j in range(N_EPISODES):
        r = synthetic_record(j, 5 + j % 4, seed=200 + j)
        if j == BOUNCE:
            meta = dataclasses.replace(r.meta, outcome="bounce", success=False)
            r = dataclasses.replace(r, meta=meta)
        records.append(r)
    write_store(root / "demos" / "ppo_2000.h5", records)
    return root


def _export(data_root: Path, out: Path, **kw: Any) -> ex.ExportResult:
    base: dict[str, Any] = {
        "first_n": FIRST_N,
        "hash_store": False,
        "require_committed_store": False,
        "provenance": PROVENANCE,
    }
    base.update(kw)
    return ex.export_r1(data_root, out, **base)


@pytest.fixture(scope="module")
def exported(data_root: Path, tmp_path_factory: pytest.TempPathFactory) -> ex.ExportResult:
    return _export(data_root, tmp_path_factory.mktemp("export") / ex.EXPORT_NAME)


def _episodes(out: Path, manifest: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    return [ex.load_episode(out / e["file"]) for e in manifest["episodes"]]


def test_export_is_dp_training_data(data_root: Path, exported: ex.ExportResult) -> None:
    """Same episodes, same order, same bytes as the dp stage-0 dataset (FrameTable)."""
    cfg = TrainConfig.load(ex.DP_CONFIG)
    assert cfg.arm == "dp" and cfg.success_filter and cfg.first_n == 1000
    spec = loop.RunSpec(
        cfg=cfg,
        run_seed=0,
        run_dir=data_root / "unused_run",
        data_root=data_root,
        device="cpu",
        purpose="test",
        first_n_override=FIRST_N,
        heldout=None,
        hash_store=False,
    )
    stages = loop.build_stages(spec)
    assert len(stages) == 1
    bundle = stages[0].data(False)
    (choice,) = bundle.choices
    m = exported.manifest
    assert [e["store_group"] for e in m["episodes"]] == list(choice.names)
    assert [e["demo_index"] for e in m["episodes"]] == list(choice.demo_indices) == [0, 1, 3]
    assert m["kept"] == choice.kept == 3 and m["attempted"] == choice.attempted == FIRST_N
    table = bundle.train.table  # type: ignore[union-attr]
    eps = _episodes(exported.out_dir, m)
    assert [int(e["episode_index"]) for e in eps] == list(range(m["kept"]))
    assert [len(e["action"]) for e in eps] == table.episode_len.tolist()
    for key, col in (("images", table.images), ("proprio", table.proprio), ("action", table.label)):
        cat = np.concatenate([e[key] for e in eps])
        assert cat.dtype == col.dtype and cat.shape == col.shape
        assert cat.tobytes() == col.tobytes(), key
    assert m["frames"] == table.n_frames


def test_arrays_are_student_inputs_and_labels(data_root: Path, exported: ex.ExportResult) -> None:
    m = exported.manifest
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        differs = False
        for e, arrays in zip(m["episodes"], _episodes(exported.out_dir, m), strict=True):
            inputs = store.student_inputs(e["store_group"])
            label = store.labels(e["store_group"])
            assert arrays["images"].tobytes() == inputs.images.tobytes()
            assert arrays["proprio"].tobytes() == inputs.proprio.tobytes()
            assert arrays["action"].tobytes() == label.tobytes()
            assert arrays["images"].dtype == np.uint8 and arrays["images"].shape[1:] == (96, 96, 3)
            assert arrays["proprio"].dtype == np.float32 and arrays["proprio"].shape[1] == 13
            assert arrays["action"].dtype == np.float32 and arrays["action"].shape[1] == 3
            differs |= not np.array_equal(
                arrays["action"], store.executed_actions(e["store_group"])
            )
        assert differs, "synthetic labels exceed [-1, 1], so action must differ from action_exec"


def test_episode_files_hold_exactly_the_declared_arrays(exported: ex.ExportResult) -> None:
    for e in exported.manifest["episodes"]:
        with np.load(exported.out_dir / e["file"], allow_pickle=False) as z:
            assert sorted(z.files) == sorted(ex.ARRAY_KEYS)
    files = sorted(p.name for p in exported.out_dir.iterdir())
    assert files == sorted([e["file"] for e in exported.manifest["episodes"]] + ["manifest.json"])


def test_manifest_records_provenance_and_contract(exported: ex.ExportResult) -> None:
    m = json.loads((exported.out_dir / "manifest.json").read_text("utf-8"))
    assert m["format"] == ex.FORMAT and m["fps"] == 30
    assert m["proprio_fields"] == [
        "attitude",
        "body_rates",
        "velocity",
        "last_action",
        "in_contact",
    ]
    assert len(m["proprio_dim_names"]) == 13
    assert m["selection"]["success_filter"] is True and m["selection"]["first_n_is_config"] is False
    assert m["selection"]["outcomes"] == {"bounce": 1, "success": 3}
    assert m["provenance"]["camera_cfg_sha256"] == CAMERA
    assert m["store_file_attrs"]["camera_cfg_sha256"] == CAMERA
    assert m["dp_config"]["sha256"] == TrainConfig.load(ex.DP_CONFIG).sha256
    assert "never action_exec" in m["arrays"]["action"]["source"]


def test_verify_passes_then_catches_tampering(
    data_root: Path, tmp_path: Path, exported: ex.ExportResult
) -> None:
    report = ex.verify_export(exported.out_dir, data_root)
    assert report["kept"] == 3 and report["episodes_compared"] == 3
    out = _export(data_root, tmp_path / "x").out_dir
    m = json.loads((out / "manifest.json").read_text("utf-8"))
    first = m["episodes"][0]
    arrays = ex.load_episode(out / first["file"])
    arrays["action"] = arrays["action"] + np.float32(1e-3)
    np.savez_compressed(out / first["file"], **arrays)  # type: ignore[arg-type]
    with pytest.raises(ex.ExportMismatchError):
        ex.verify_export(out, data_root)  # file SHA-256
    first["file_sha256"] = ex.sha256_file(out / first["file"])
    (out / "manifest.json").write_text(json.dumps(m), "utf-8")
    with pytest.raises(ex.ExportMismatchError):
        ex.verify_export(out, data_root)  # bytes differ from the store


def test_refuses_overwrite_and_foreign_camera(data_root: Path, exported: ex.ExportResult) -> None:
    with pytest.raises(FileExistsError):
        _export(data_root, exported.out_dir)
    with pytest.raises(ValueError, match="camera"):
        _export(
            data_root,
            exported.out_dir.parent / "other",
            provenance={**PROVENANCE, "camera_cfg_sha256": "0" * 64},
        )


#: Store accessors and names that are not the student path (privileged, clock, render metadata).
FORBIDDEN = frozenset(
    {
        "privileged_observations",
        "model_times",
        "audit_arrays",
        "estimator_targets",
        "executed_actions",
        "audit_field",
        "meta",
        "render_with_meta",
        "RenderMeta",
        "physics_step_count",
        "teacher_obs",
        "priv",
        "t_model_s",
        "action_exec",
        # P8 frozen-list accessors (leak audit R3; ``tests/test_render_access.py``
        # FROZEN_LIST_NAMES): the export reads ``splits/`` rows only, never a frozen list.
        "read_list",
        "list_path",
        "list_paths",
        "list_names",
        "manifest_path",
    }
)


def test_export_module_reads_only_the_student_path() -> None:
    """No privileged accessor, no ``meta()`` (only ``choose_episodes`` reads it), no lerobot."""
    src = Path(ex.__file__).read_text("utf-8")
    tree = ast.parse(src)
    seen: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            seen.add(node.value)
        elif isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (seen & FORBIDDEN), seen & FORBIDDEN
    assert imported <= {
        "__future__",
        "hashlib",
        "json",
        "shutil",
        "time",
        "collections",
        "dataclasses",
        "pathlib",
        "typing",
        "numpy",
        "ila",
    }, imported
    assert {"student_inputs", "labels", "choose_episodes"} <= seen


@pytest.mark.data
@pytest.mark.slow
def test_real_export_matches_dp_choice() -> None:
    """The committed-data export: dp's choice on the real store, 25 episodes byte-compared."""
    data_root = p8_adapter.load_paths()["data_root"]
    out = ex.export_dir(data_root)
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no export at {out}")
    cfg = TrainConfig.load(ex.DP_CONFIG)
    with DemoStore(loop.store_path(data_root, cfg.collection)) as store:
        choice = choose_episodes(store, cfg.first_n, cfg.success_filter)
    m = json.loads((out / "manifest.json").read_text("utf-8"))
    assert m["selection"]["first_n_is_config"] is True and m["store"]["matches"] is True
    assert [e["store_group"] for e in m["episodes"]] == list(choice.names)
    rng = np.random.default_rng(20261023)
    sample = sorted(rng.choice(choice.kept, size=25, replace=False).tolist())
    report = ex.verify_export(out, data_root, episodes=sample)
    assert report["kept"] == choice.kept and report["episodes_compared"] == 25
