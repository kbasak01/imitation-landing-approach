"""Export of ``dp``'s training episodes to a neutral intermediate for reference arm R1 (Phase 6).

R1 (``R1_lerobot_dp``, P3-D1 §3: "same 1 000") trains LeRobot 0.6.1's DiffusionPolicy in
``.venv-lerobot``, where ``ila`` is not importable, on exactly the data ``dp`` trains on. This
module runs in ``.venv`` and never imports ``lerobot``: it writes plain ``.npz`` files and a JSON
manifest; ``scripts/r1/build_lerobot_dataset.py`` (``.venv-lerobot`` only) turns them into a
LeRobotDataset v3.

Episode choice is **not** reimplemented. It is the ``dp`` training path's own:
:class:`ila.train.config.TrainConfig` (``configs/train/dp.yaml``: ``data.collection``,
``data.first_n``, ``success_filter``), :func:`ila.train.loop.store_path` and
:func:`ila.train.sampling.choose_episodes` with ``start = 0`` -- exactly what
:func:`ila.train.loop.build_stages` passes to :func:`ila.train.loop.load_stage_data` for a
single-stage arm. ``choose_episodes`` is the only reader of ``DemoStore.meta()`` here; this
module itself never calls it.

Arrays come only from the student read path, as :meth:`ila.train.dataset.FrameTable.from_store`
reads them: :meth:`DemoStore.student_inputs` (``images``, the default variant's 13-d
``proprio``) and :meth:`DemoStore.labels` (the clean teacher label). No ``priv``,
``teacher_obs``, ``action_exec``, ``t_model_s`` or render metadata is read or written. The
store's file attributes are read for provenance only (camera-config SHA-256, commits).

Layout under ``<data_root>/lerobot/<EXPORT_NAME>/``:

* ``ep_<k:04d>.npz`` for ``k = 0 .. kept - 1`` (LeRobot episode index ``k``, in stream order):
  ``images`` ``uint8[T, 96, 96, 3]`` (HWC RGB, the stored frame, uncropped; the 96 -> 88 px crop
  belongs to the policy), ``proprio`` ``float32[T, 13]`` (fields in ``student.yaml`` order;
  P8 units, lengths MODEL scale, lambda = 1/25), ``action`` ``float32[T, 3]`` (= ``label``:
  normalised world-frame velocity setpoint, x v_max = 1.5 m/s MODEL scale; never
  ``action_exec``), ``episode_index`` ``int64`` scalar ``k``.
* ``manifest.json``: :data:`FORMAT`, provenance, the episode list, counts and per-file SHA-256.

Alignment is the store's (il-conventions): row ``t`` of an episode is control step ``t`` at
30 Hz (1/30 s MODEL scale per step, 1/6 s full scale); ``images[t]``, ``proprio[t]`` are what
the policy sees before ``action_exec[t]``; ``action[t] = label[t] = teacher(teacher_obs[t])``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila.data.store import ACTION_DIM, CTRL_HZ, IMAGE_SHAPE, DemoStore
from ila.obs.split import load_contract
from ila.train.config import TrainConfig, sha256_file
from ila.train.loop import store_path
from ila.train.sampling import EpisodeChoice, choose_episodes

__all__ = [
    "ARRAY_KEYS",
    "DP_CONFIG",
    "EXPORT_NAME",
    "FORMAT",
    "ExportMismatchError",
    "ExportResult",
    "content_sha256",
    "dp_choice",
    "export_dir",
    "export_r1",
    "load_episode",
    "proprio_dim_names",
    "verify_export",
]

#: Format tag of the intermediate (bump on any layout change).
FORMAT: Final[str] = "ila-r1-intermediate/1"
#: Directory name under ``<data_root>/lerobot/``.
EXPORT_NAME: Final[str] = "r1_ppo_2000_first1000"
#: The config whose data R1 must match (P3-D1 §3).
DP_CONFIG: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / "train" / "dp.yaml"
#: The arrays of every episode file, in this order, and nothing else.
ARRAY_KEYS: Final[tuple[str, ...]] = ("images", "proprio", "action", "episode_index")


def export_dir(data_root: Path) -> Path:
    """Return ``<data_root>/lerobot/<EXPORT_NAME>``."""
    return data_root / "lerobot" / EXPORT_NAME


def proprio_dim_names() -> list[str]:
    """Return one name per proprio column, ``<field>_<i>``, in ``student.yaml`` order (13)."""
    c = load_contract()
    return [f"{f}_{i}" for f in c.student_proprio for i in range(c.layout[f][1])]


def content_sha256(arrays: dict[str, NDArray[Any]]) -> str:
    """SHA-256 over :data:`ARRAY_KEYS` (name, dtype, shape, C-order bytes), zip-independent."""
    h = hashlib.sha256()
    for k in ARRAY_KEYS:
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dp_choice(cfg: TrainConfig, store: DemoStore, first_n: int | None = None) -> EpisodeChoice:
    """Return ``dp``'s stage-0 episode choice: ``choose_episodes(store, first_n, filter, 0)``.

    Args:
        cfg: ``dp``'s config.
        store: Its collection's store.
        first_n: Tests only: replaces ``cfg.first_n`` (as ``RunSpec.first_n_override`` does).

    Returns:
        The choice (names in stream order).
    """
    n = cfg.first_n if first_n is None else first_n
    return choose_episodes(store, n, cfg.success_filter, start=0)


@dataclass(frozen=True)
class ExportResult:
    """What one export wrote.

    Attributes:
        out_dir: The intermediate directory.
        manifest: Its manifest (as written).
    """

    out_dir: Path
    manifest: dict[str, Any]


def _read_episode(store: DemoStore, name: str, k: int) -> dict[str, NDArray[Any]]:
    """Read one episode through the student path (as ``FrameTable.from_store`` does)."""
    inputs = store.student_inputs(name, "default")
    label = store.labels(name)
    t_len = int(label.shape[0])
    if inputs.images.shape != (t_len, *IMAGE_SHAPE) or inputs.proprio.shape[0] != t_len:
        raise ValueError(f"{name}: images/proprio/label lengths differ")
    if label.shape != (t_len, ACTION_DIM):
        raise ValueError(f"{name}: label shape {label.shape}")
    if not (np.all(np.isfinite(inputs.proprio)) and np.all(np.isfinite(label))):
        raise ValueError(f"{name}: non-finite proprio or label")
    return {
        "images": np.ascontiguousarray(inputs.images, dtype=np.uint8),
        "proprio": np.ascontiguousarray(inputs.proprio, dtype=np.float32),
        "action": np.ascontiguousarray(label, dtype=np.float32),
        "episode_index": np.asarray(k, dtype=np.int64),
    }


def export_r1(
    data_root: Path,
    out_dir: Path | None = None,
    *,
    config: Path = DP_CONFIG,
    first_n: int | None = None,
    overwrite: bool = False,
    hash_store: bool = True,
    require_committed_store: bool = True,
    provenance: dict[str, Any] | None = None,
) -> ExportResult:
    """Write ``dp``'s training episodes to the intermediate (module docstring).

    Args:
        data_root: ``${data_root}`` (store at ``demos/<collection>.h5``).
        out_dir: Output directory (default :func:`export_dir`).
        config: ``dp``'s train config.
        first_n: Tests only: episodes attempted (default ``cfg.first_n`` = 1 000).
        overwrite: Replace an existing output directory.
        hash_store: Hash the store file (tests may skip it).
        require_committed_store: Refuse a store whose SHA-256 differs from the one committed
            in ``results/demos_summary.json`` (tests on synthetic stores turn it off).
        provenance: Tests only: replaces :func:`ila.data.collect.provenance` (git, camera).

    Returns:
        The :class:`ExportResult`.

    Raises:
        FileExistsError: If ``out_dir`` exists and ``overwrite`` is false.
        RuntimeError: If ``require_committed_store`` and the store is not the committed one.
        ValueError: If the store's camera SHA-256 or proprio fields differ from the committed
            ones, or an episode is inconsistent.
    """
    t_start = time.perf_counter()
    cfg = TrainConfig.load(config.resolve())
    if cfg.kind != "dp" or cfg.iterations is not None:
        raise ValueError(f"{config}: R1 mirrors a single-stage dp config, got {cfg.arm}")
    out = export_dir(data_root) if out_dir is None else out_dir
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass overwrite)")
    if provenance is None:
        from ila.data.collect import provenance as git_provenance  # lazy: imports the runner

        provenance = git_provenance()
    collection = cfg.collection_for(None)
    path = store_path(data_root, collection)
    from ila.train import manifest as mf  # lazy: only for the committed store SHA

    record = mf.store_record(collection, path, hash_file=hash_store or require_committed_store)
    if require_committed_store and not record["matches"]:
        raise RuntimeError(
            f"{path}: SHA-256 {record['sha256']} != committed {record['expected_sha256']}"
        )
    contract = load_contract()
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    with DemoStore(path) as store:
        attrs = store.file_attrs
        if attrs["camera_cfg_sha256"] != provenance["camera_cfg_sha256"]:
            raise ValueError(f"{path}: camera SHA-256 differs from the committed camera config")
        if json.loads(attrs["proprio_fields"]) != list(contract.student_proprio):
            raise ValueError(f"{path}: proprio fields differ from student.yaml")
        choice = dp_choice(cfg, store, first_n)
        for k, (name, demo_index) in enumerate(zip(choice.names, choice.demo_indices, strict=True)):
            arrays = _read_episode(store, name, k)
            fname = f"ep_{k:04d}.npz"
            np.savez_compressed(tmp / fname, **arrays)  # type: ignore[arg-type]
            episodes.append(
                {
                    "episode_index": k,
                    "store_group": name,
                    "demo_index": demo_index,
                    "frames": int(arrays["action"].shape[0]),
                    "file": fname,
                    "file_sha256": sha256_file(tmp / fname),
                    "content_sha256": content_sha256(arrays),
                }
            )
        store_attrs = {
            k: attrs[k] for k in ("collection", "teacher", "p8_commit", "ila_commit")
        } | {"camera_cfg_sha256": attrs["camera_cfg_sha256"]}
    frames = sum(int(e["frames"]) for e in episodes)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "name": out.name,
        "purpose": "R1_lerobot_dp training data (P3-D1 §3: the same episodes as dp)",
        "dp_config": {"path": str(config), "sha256": cfg.sha256, "arm": cfg.arm},
        "selection": {
            "function": "ila.train.sampling.choose_episodes",
            "collection": collection,
            "first_n": choice.attempted,
            "first_n_is_config": first_n is None,
            "success_filter": cfg.success_filter,
            "start": 0,
            **{k: v for k, v in choice.summary().items() if k != "store"},
        },
        "store": record,
        "store_file_attrs": store_attrs,
        "read_path": ["DemoStore.student_inputs(name, 'default')", "DemoStore.labels(name)"],
        "arrays": {
            "images": {"dtype": "uint8", "shape": ["T", *IMAGE_SHAPE], "layout": "HWC RGB"},
            "proprio": {"dtype": "float32", "shape": ["T", contract.proprio_dim]},
            "action": {
                "dtype": "float32",
                "shape": ["T", ACTION_DIM],
                "source": "label (clean teacher action), never action_exec",
            },
            "episode_index": {"dtype": "int64", "shape": []},
        },
        "proprio_fields": list(contract.student_proprio),
        "proprio_dim_names": proprio_dim_names(),
        "obs_contract_variant": "default",
        "fps": int(CTRL_HZ),
        "kept": len(episodes),
        "attempted": choice.attempted,
        "frames": frames,
        "episodes": episodes,
        "provenance": provenance,
        "numpy": np.__version__,
        "wall_s": time.perf_counter() - t_start,
    }
    (tmp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", "utf-8"
    )
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    return ExportResult(out_dir=out, manifest=manifest)


class ExportMismatchError(AssertionError):
    """The intermediate differs from the store or from ``dp``'s episode choice."""


def _check(ok: bool, what: object) -> None:
    """Raise :class:`ExportMismatchError` unless ``ok``."""
    if not ok:
        raise ExportMismatchError(str(what))


def load_episode(path: Path) -> dict[str, NDArray[Any]]:
    """Load one intermediate episode file (exactly :data:`ARRAY_KEYS`).

    Raises:
        ValueError: If the file holds other arrays.
    """
    with np.load(path, allow_pickle=False) as z:
        if tuple(sorted(z.files)) != tuple(sorted(ARRAY_KEYS)):
            raise ValueError(f"{path}: arrays {z.files} != {list(ARRAY_KEYS)}")
        return {k: np.asarray(z[k]) for k in ARRAY_KEYS}


def verify_export(
    out_dir: Path,
    data_root: Path,
    *,
    config: Path = DP_CONFIG,
    episodes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Re-derive ``dp``'s choice and compare the intermediate to the store, exactly.

    Checks: the manifest's episode list equals ``dp``'s choice (names, stream indices, kept,
    attempted) at the manifest's ``first_n``; every listed file's SHA-256; and, for
    ``episodes`` (default: all), ``images``/``proprio``/``action`` equal to
    ``student_inputs``/``labels`` byte for byte.

    Args:
        out_dir: The intermediate directory.
        data_root: ``${data_root}``.
        config: ``dp``'s train config.
        episodes: Episode indices to compare array-wise (default: all).

    Returns:
        ``{"kept", "frames", "episodes_compared", "files_checked"}``.

    Raises:
        ExportMismatchError: On any mismatch.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    _check(manifest["format"] == FORMAT, manifest["format"])
    cfg = TrainConfig.load(config.resolve())
    first_n = int(manifest["selection"]["first_n"])
    path = store_path(data_root, cfg.collection_for(None))
    listed = manifest["episodes"]
    with DemoStore(path) as store:
        choice = dp_choice(cfg, store, first_n)
        _check([e["store_group"] for e in listed] == list(choice.names), "episode set differs")
        _check([e["demo_index"] for e in listed] == list(choice.demo_indices), "stream indices")
        _check(manifest["kept"] == choice.kept, "kept count")
        _check(manifest["attempted"] == choice.attempted, "attempted count")
        _check([e["episode_index"] for e in listed] == list(range(choice.kept)), "episode indices")
        files = sorted(p.name for p in out_dir.glob("ep_*.npz"))
        _check(files == sorted(e["file"] for e in listed), "stray or missing episode files")
        for e in listed:
            _check(sha256_file(out_dir / e["file"]) == e["file_sha256"], e["file"])
        which = range(len(listed)) if episodes is None else episodes
        frames = 0
        for k in which:
            e = listed[k]
            got = load_episode(out_dir / e["file"])
            want = _read_episode(store, e["store_group"], k)
            for key in ARRAY_KEYS:
                _check(got[key].dtype == want[key].dtype, (e["file"], key))
                _check(got[key].shape == want[key].shape, (e["file"], key))
                _check(got[key].tobytes() == want[key].tobytes(), (e["file"], key))
            _check(content_sha256(got) == e["content_sha256"], (e["file"], "content_sha256"))
            frames += int(got["action"].shape[0])
    _check(sum(int(e["frames"]) for e in listed) == manifest["frames"], "frame total")
    return {
        "kept": manifest["kept"],
        "frames": manifest["frames"],
        "episodes_compared": len(which),
        "frames_compared": frames,
        "files_checked": len(listed),
    }
