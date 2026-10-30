"""Build R1's LeRobotDataset v3 from the neutral intermediate (Phase 6; ``.venv-lerobot`` ONLY).

Step 2 of 2 for ``R1_lerobot_dp``. Step 1 (``.venv``) is ``scripts/r1/export_dataset.py``,
which writes ``${data_root}/lerobot/r1_ppo_2000_first1000/`` (``ep_<k>.npz`` + ``manifest.json``)
from ``dp``'s exact training episodes. This script never imports ``ila`` (not installed in this
venv) and reads nothing but that directory; it uses lerobot 0.6.1's own writer API
(``LeRobotDataset.create`` -> ``add_frame`` -> ``save_episode`` -> ``finalize``):

* ``fps`` 30 (one row per control step, 1/30 s MODEL scale, lambda = 1/25);
* ``observation.image``: ``image`` dtype, ``(96, 96, 3)`` HWC, ``use_videos=False``: lerobot
  writes each frame as PNG and embeds the PNG bytes in the data parquet (lossless; no video);
* ``observation.state``: ``float32[13]``, the student proprio (``student.yaml`` order);
* ``action``: ``float32[3]``, the clean teacher label (never ``action_exec``).

Then it reloads the dataset from disk (offline) and compares every frame of every episode to
the intermediate exactly (decoded image bytes, state and action bytes, episode boundaries,
``frame_index``, ``timestamp``), plus a few frames through ``LeRobotDataset.__getitem__``
(``float32`` CHW in [0, 1]: ``round(x * 255)`` must give the stored bytes back). A build record
goes to ``${data_root}/lerobot/<dataset>.build.json``. Example::

    .venv-lerobot/bin/python scripts/r1/build_lerobot_dataset.py
    .venv-lerobot/bin/python scripts/r1/build_lerobot_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never touch the Hub: local root only

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"
INTERMEDIATE_FORMAT = "ila-r1-intermediate/1"
INTERMEDIATE = "r1_ppo_2000_first1000"
DATASET = "r1_ppo_2000_first1000_lerobot_v3"
REPO_ID = "ila/r1_ppo_2000_first1000"
LEROBOT_VERSION = "0.6.1"
TASK = "land on the heaving ship deck (simulation)"
ROBOT_TYPE = "pybullet_quadrotor_sim"
ACTION_NAMES = ["vx_setpoint_world", "vy_setpoint_world", "vz_setpoint_world"]
ARRAY_KEYS = ("images", "proprio", "action", "episode_index")
GETITEM_SAMPLES = 32
SEED = 20261023


def data_root_from_paths(path: Path = PATHS_FILE) -> Path:
    """Resolve ``data_root`` as ``ila.p8_adapter.load_paths`` does (``${key}``, repo-relative)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = re.sub(r"\$\{(\w+)\}", lambda m: resolved[m.group(1)], str(value))
    return (REPO_ROOT / resolved["data_root"]).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(ARRAY_KEYS):
            raise ValueError(f"{path}: arrays {z.files}")
        return {k: np.asarray(z[k]) for k in ARRAY_KEYS}


def check_venv() -> str:
    """Refuse to run outside ``.venv-lerobot`` or with another lerobot."""
    if Path(sys.prefix).name != ".venv-lerobot":
        sys.exit(f"run with .venv-lerobot/bin/python (sys.prefix is {sys.prefix})")
    import lerobot

    if lerobot.__version__ != LEROBOT_VERSION:
        sys.exit(f"lerobot {lerobot.__version__} != pinned {LEROBOT_VERSION}")
    return str(lerobot.__version__)


def features(state_names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "observation.image": {
            "dtype": "image",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (13,), "names": state_names},
        "action": {"dtype": "float32", "shape": (3,), "names": ACTION_NAMES},
    }


def build(src: Path, root: Path, manifest: dict[str, Any], threads: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=int(manifest["fps"]),
        features=features(list(manifest["proprio_dim_names"])),
        root=root,
        robot_type=ROBOT_TYPE,
        use_videos=False,
        image_writer_threads=threads,
    )
    for e in manifest["episodes"]:
        arrays = load_episode(src / e["file"])
        if int(arrays["episode_index"]) != int(e["episode_index"]):
            raise ValueError(f"{e['file']}: episode_index mismatch")
        for t in range(arrays["action"].shape[0]):
            ds.add_frame(
                {
                    "observation.image": arrays["images"][t],
                    "observation.state": arrays["proprio"][t],
                    "action": arrays["action"][t],
                    "task": TASK,
                }
            )
        ds.save_episode()
        if e["episode_index"] % 100 == 0:
            print(f"  episode {e['episode_index']} saved", flush=True)
    ds.finalize()


def _column(hf: Any, key: str) -> np.ndarray:
    return np.asarray(hf[key])


def verify(src: Path, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Reload from disk and compare to the intermediate exactly (module docstring)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root)
    meta = ds.meta
    n_ep, n_frames = int(manifest["kept"]), int(manifest["frames"])
    problems: list[str] = []
    if (ds.num_episodes, ds.num_frames, len(ds)) != (n_ep, n_frames, n_frames):
        problems.append(f"counts {ds.num_episodes}/{ds.num_frames} != {n_ep}/{n_frames}")
    if int(meta.fps) != int(manifest["fps"]):
        problems.append(f"fps {meta.fps}")
    if meta.video_keys:
        problems.append(f"video keys {meta.video_keys}")
    raw = ds.hf_dataset.with_format("numpy")
    state = _column(raw, "observation.state").astype(np.float32, copy=False)
    action = _column(raw, "action").astype(np.float32, copy=False)
    ep_col = _column(raw, "episode_index")
    fi_col = _column(raw, "frame_index")
    ts_col = _column(raw, "timestamp")
    images_checked = 0
    expected_offsets: list[tuple[int, int]] = []
    start = 0
    for e in manifest["episodes"]:
        k = int(e["episode_index"])
        arrays = load_episode(src / e["file"])
        t_len = arrays["action"].shape[0]
        sl = slice(start, start + t_len)
        expected_offsets.append((start, start + t_len))
        rec = meta.episodes[k]
        if (int(rec["dataset_from_index"]), int(rec["dataset_to_index"])) != (sl.start, sl.stop):
            problems.append(f"episode {k}: bounds {rec['dataset_from_index']}..")
        if state[sl].tobytes() != arrays["proprio"].tobytes():
            problems.append(f"episode {k}: observation.state differs")
        if action[sl].tobytes() != arrays["action"].tobytes():
            problems.append(f"episode {k}: action differs")
        if not (np.all(ep_col[sl] == k) and np.array_equal(fi_col[sl], np.arange(t_len))):
            problems.append(f"episode {k}: episode_index/frame_index")
        if not np.allclose(ts_col[sl], np.arange(t_len) / float(manifest["fps"]), atol=1e-6):
            problems.append(f"episode {k}: timestamp")
        imgs = np.stack([np.asarray(im) for im in raw[sl]["observation.image"]])
        if imgs.dtype != np.uint8 or imgs.tobytes() != arrays["images"].tobytes():
            problems.append(f"episode {k}: images differ ({imgs.dtype}, {imgs.shape})")
        images_checked += t_len
        start += t_len
    # The training read path (__getitem__): float32 CHW in [0, 1].
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(n_frames, size=GETITEM_SAMPLES, replace=False).tolist())
    flat_ep = np.repeat(np.arange(n_ep), [b - a for a, b in expected_offsets])
    getitem_dtype = None
    for i in idx:
        item = ds[i]
        k = int(flat_ep[i])
        t = i - expected_offsets[k][0]
        arrays = load_episode(src / manifest["episodes"][k]["file"])
        img = item["observation.image"]
        getitem_dtype = str(img.dtype)
        back = torch.round(img.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
        if back.tobytes() != arrays["images"][t].tobytes():
            problems.append(f"__getitem__ {i}: image")
        if item["observation.state"].numpy().tobytes() != arrays["proprio"][t].tobytes():
            problems.append(f"__getitem__ {i}: state")
        if item["action"].numpy().tobytes() != arrays["action"][t].tobytes():
            problems.append(f"__getitem__ {i}: action")
    stats = meta.stats or {}
    return {
        "ok": not problems,
        "problems": problems[:50],
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "frames_image_compared": images_checked,
        "frames_state_action_compared": int(state.shape[0]),
        "getitem_frames_compared": len(idx),
        "getitem_image_dtype": getitem_dtype,
        "features": {k: {kk: v[kk] for kk in ("dtype", "shape")} for k, v in meta.features.items()},
        "stats_keys": {k: sorted(v) for k, v in stats.items()},
        "image_stats": {
            k: np.asarray(v).reshape(-1).round(6).tolist()
            for k, v in stats.get("observation.image", {}).items()
            if k in ("mean", "std", "min", "max", "count")
        },
        "codebase_version": meta.info.get("codebase_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="lerobot image-writer threads")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    version = check_venv()

    data_root = data_root_from_paths() if args.data_root is None else args.data_root
    src = data_root / "lerobot" / INTERMEDIATE
    root = data_root / "lerobot" / DATASET
    record_path = data_root / "lerobot" / f"{DATASET}.build.json"
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest["format"] != INTERMEDIATE_FORMAT:
        sys.exit(f"{manifest_path}: format {manifest['format']!r}")
    for e in manifest["episodes"]:
        if sha256_file(src / e["file"]) != e["file_sha256"]:
            sys.exit(f"{src / e['file']}: SHA-256 differs from the intermediate manifest")

    t0 = time.perf_counter()
    if not args.verify_only:
        if root.exists():
            if not args.overwrite:
                sys.exit(f"{root} exists (pass --overwrite)")
            shutil.rmtree(root)
        print(f"building {root} from {src} ({manifest['kept']} episodes)", flush=True)
        build(src, root, manifest, args.threads)
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    report = verify(src, root, manifest)
    t_verify = time.perf_counter() - t1
    files = sorted(p for p in root.rglob("*") if p.is_file())
    import datasets
    import torch

    record = {
        "dataset_root": str(root),
        "repo_id": REPO_ID,
        "intermediate": str(src),
        "intermediate_manifest_sha256": sha256_file(manifest_path),
        "dp_config": manifest["dp_config"],
        "store": manifest["store"],
        "kept": manifest["kept"],
        "attempted": manifest["attempted"],
        "frames": manifest["frames"],
        "fps": manifest["fps"],
        "task": TASK,
        "robot_type": ROBOT_TYPE,
        "use_videos": False,
        "lerobot": version,
        "datasets": datasets.__version__,
        "torch": torch.__version__,
        "ila_commit_of_intermediate": manifest["provenance"]["ila_commit"],
        "ila_commit_at_build": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "camera_cfg_sha256": manifest["provenance"]["camera_cfg_sha256"],
        "verification": report,
        "files": {str(p.relative_to(root)): sha256_file(p) for p in files},
        "bytes": sum(p.stat().st_size for p in files),
        "build_s": t_build,
        "verify_s": t_verify,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: record[k] for k in ("kept", "frames", "bytes", "build_s", "verify_s")}))
    print(json.dumps({k: report[k] for k in report if k not in ("features",)}, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
