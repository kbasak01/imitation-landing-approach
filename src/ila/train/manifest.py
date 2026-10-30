"""Run directories, manifests and status files for background training jobs (Phase 4).

Layout (``artifacts/`` is gitignored; per-run manifests without weights are copied to
``results/phase4/`` when a gate needs them):

* core runs: ``artifacts/runs/<arm>/s<seed>/``;
* DART sigma trials (tuning, P3-D1 §4): ``artifacts/runs/bc_dart_tune/sigma_<sigma>/s0/``;
* timing runs (P3-D1 §5, wall clock only, never selected): ``artifacts/runs/<arm>_timing/s<seed>/``;
* smoke runs: ``artifacts/runs/smoke/<arm>/``.

Each run directory holds ``manifest.json`` (written at start, updated at every checkpoint and at
the end), ``status.json`` (rewritten at every log interval), ``train_log.jsonl``,
``checkpoints/`` and, after selection, ``selection.json``. Both JSON files are replaced
atomically (write to a temporary file, then ``os.replace``).

The manifest records: config file SHA-256 against the P3-D1 frozen value, the data store
SHA-256 against ``results/demos_summary.json``, the P8 and ``ila`` commits with the dirty flag,
host, CPU and GPU, torch numerics, every checkpoint's SHA-256, wall clock and GPU-hours (one GPU:
GPU-hours = wall-clock hours of the training process).
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ila.policies.registry import REPO_ROOT
from ila.train.config import sha256_file

__all__ = [
    "DEMOS_SUMMARY",
    "FROZEN_CONFIG_SHA256",
    "RUNS_ROOT",
    "StatusFile",
    "check_config_frozen",
    "git_state",
    "host_info",
    "read_json",
    "run_dir",
    "store_record",
    "write_json_atomic",
]

#: Where background runs live (gitignored).
RUNS_ROOT: Final[Path] = REPO_ROOT / "artifacts" / "runs"

#: Committed demo-store summary with every store's SHA-256 (P3-D1 §2).
DEMOS_SUMMARY: Final[Path] = REPO_ROOT / "results" / "demos_summary.json"

#: SHA-256 of every training config frozen by P3-D1 §3 (the protocol prints them truncated:
#: ``tests/test_train_loop.py`` checks each against the truncated form in ``docs/protocol.md``),
#: plus ``configs/data/demos.yaml`` (P3-D1 §2, printed in full), which ``bc_dagger`` reads.
FROZEN_CONFIG_SHA256: Final[Mapping[str, str]] = {
    "configs/train/bc.yaml": "3d3386bb38718f5f79b9a61b62f5044052bdb603cb8b33ffd3b7664de6a01299",
    "configs/train/bc_state.yaml": (
        "36b65a7d6732288cad57fd7b6469e8b13bfe3b02fb6936a599821315ea86eb0f"
    ),
    "configs/train/bc_matched.yaml": (
        "f956e182d1d9e72f1c75010dce743428ee629639de13abdca75374c08ced8de5"
    ),
    "configs/train/bc_dart.yaml": (
        "5dff64bf8b951dfcd3d5c740fe68a1e85a219bfc3c286a3f2424d418b5a60c02"
    ),
    "configs/train/bc_dagger.yaml": (
        "dbc7c0d22438abe96e5e80de95f8f46d64b81216147496dc15bef55adc989da7"
    ),
    "configs/train/dp.yaml": "c8f90c3565d86b54f712dc5b0f624127c537c4b496ba9e414c7b8b6a952b6913",
    "configs/train/modular.yaml": (
        "84153e6dc96288d1512f7387b08ace6632aed195dcfeb2d4358c1c977500eadc"
    ),
    "configs/train/ablations/bc_mix.yaml": (
        "14f9efabd4374fc6e7401186ddb55d2cf854ce22ceecc2ca922159a6082af42c"
    ),
    "configs/train/ablations/dp_mix.yaml": (
        "9a94d0ae88af728d0e0b6c96b6849adaf5294806d038234a8925682f0fe529f9"
    ),
    "configs/train/ablations/bc_pid.yaml": (
        "7a0f91046ff857695478e8e2caf03d172672dad6620ed426d19cbb8dd1083804"
    ),
    "configs/train/ablations/dp_pid.yaml": (
        "3638e62c4060338ebfab36ca4e3a87424cb648008ec4deb6960afa8c5ca9feb8"
    ),
    "configs/train/ablations/bc_n100.yaml": (
        "b600329130d155fcd28e1ce2be10798d33fb52dbb28c9c7ce369ee618dc337c4"
    ),
    "configs/train/ablations/bc_n300.yaml": (
        "ed345e7f6b94afbc6af9e060b98c2a6b648788d35183f0cdc4d7a74ed9bea84b"
    ),
    "configs/train/ablations/dp_n100.yaml": (
        "b620251aec57de4a85723fe0a1beb72d96b4dcfca6b3054cdba25edba222ccc4"
    ),
    "configs/train/ablations/dp_n300.yaml": (
        "bab824113fa1433872bfe3483b5e1846ac54422b0bdd41b87b3364e9f3fdac26"
    ),
    "configs/train/ablations/bc_minus_last_action.yaml": (
        "a59757dd428d3bd7e7d018bed68add0f6280979ec1e659dacee77b8c6c4fc414"
    ),
    "configs/train/ablations/dp_minus_last_action.yaml": (
        "3f5609cda9999ba99b70f890014a74cdffd0c32e766a099613dc8edebb519e40"
    ),
    "configs/train/ablations/bc_plus_clock.yaml": (
        "505f3bef192125bf3882b84ac8a5387f96d58afdf67cbacde2998e91f11c9caa"
    ),
    # Frozen by P3-D1 §2 (printed in full there); read by bc_dagger.yaml's
    # ``dagger: {from: configs/data/demos.yaml#dagger}``, so it is part of that run's config.
    "configs/data/demos.yaml": ("1fd413e2823a1cefd4b38e6ddd00c7097ee182e45f2564e0f398ce19f5f63f11"),
}


def run_dir(
    arm: str,
    seed: int,
    *,
    root: Path = RUNS_ROOT,
    sigma: float | None = None,
    tuning: bool = False,
    timing: bool = False,
) -> Path:
    """Return a run's directory (module docstring layout).

    Args:
        arm: Arm name.
        seed: Run seed.
        root: Runs root.
        sigma: DART sigma (``bc_dart`` only).
        tuning: A DART sigma trial (seed 0 only).
        timing: A P3-D1 §5 timing run.

    Returns:
        The directory (not created).

    Raises:
        ValueError: On a tuning run that is not ``bc_dart`` seed 0 with a sigma, or a timing
            run that is also a tuning run.
    """
    if tuning:
        if arm != "bc_dart" or seed != 0 or sigma is None or timing:
            raise ValueError("DART trials are bc_dart seed 0 at one sigma (P3-D1 §4)")
        return root / "bc_dart_tune" / f"sigma_{sigma:.2f}" / f"s{seed}"
    name = f"{arm}_timing" if timing else arm
    return root / name / f"s{seed}"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def check_config_frozen(files_sha256: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    """Compare each config file's SHA-256 with P3-D1's frozen value.

    Args:
        files_sha256: ``(path, sha256)`` pairs (the named config first, then its base).

    Returns:
        ``{"files": [{path, sha256, frozen_sha256, matches}], "all_frozen": bool}``.
    """
    files = []
    for path, sha in files_sha256:
        rel = _rel(Path(path))
        frozen = FROZEN_CONFIG_SHA256.get(rel)
        files.append(
            {"path": rel, "sha256": sha, "frozen_sha256": frozen, "matches": frozen == sha}
        )
    return {"files": files, "all_frozen": all(f["matches"] for f in files)}


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    """Return the ``ila`` commit, its dirty flag and paths, and the P8 submodule commit.

    Returns:
        ``ila_commit``, ``ila_dirty``, ``ila_dirty_paths`` (first 50), ``p8_commit``.
    """
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln]
    from ila import p8_adapter  # lazy: imports rld

    return {
        "ila_commit": _git("rev-parse", "HEAD"),
        "ila_dirty": bool(dirty),
        "ila_dirty_paths": dirty[:50],
        "p8_commit": p8_adapter.p8_commit(),
    }


def host_info(device: str) -> dict[str, Any]:
    """Return host, CPU, torch and GPU facts for the manifest."""
    import torch

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": device,
        "cuda": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        info["gpu"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(idx).total_memory
    return info


def store_record(collection: str, path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    """Return a store's SHA-256 and the value committed in ``results/demos_summary.json``.

    Args:
        collection: The collection name (``ppo_2000``, ``dart_0.10``, ...).
        path: The store file.
        hash_file: Hash the file (about 1 s per 240 MB); tests may skip it.

    Returns:
        ``{collection, path, sha256, expected_sha256, matches}``.
    """
    expected = None
    if DEMOS_SUMMARY.is_file():
        summary = json.loads(DEMOS_SUMMARY.read_text(encoding="utf-8"))
        entry = summary.get("collections", {}).get(collection, {})
        expected = entry.get("summary", {}).get("file_sha256")
    sha = sha256_file(path) if hash_file else None
    return {
        "collection": collection,
        "path": str(path),
        "sha256": sha,
        "expected_sha256": expected,
        "matches": sha is not None and sha == expected,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write sorted, indented JSON with a trailing newline, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


class StatusFile:
    """``status.json`` of a background job: state, progress, ETA, PID.

    States: ``starting``, ``running``, ``done``, ``failed``, ``diverged``.

    Attributes:
        path: The file.
        fields: The current content.
    """

    def __init__(self, path: Path, *, total_steps: int, **fields: Any) -> None:
        """Start a status file (state ``starting``)."""
        self.path = path
        self.fields: dict[str, Any] = {
            "state": "starting",
            "step": 0,
            "total_steps": int(total_steps),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_unix": time.time(),
            "last_select_success": None,
            **fields,
        }
        self.write()

    def update(self, **fields: Any) -> None:
        """Merge fields, stamp ``updated_unix`` and rewrite."""
        self.fields.update(fields)
        self.write()

    def progress(self, step: int, steps_per_s: float | None, **fields: Any) -> None:
        """Record progress and the ETA (seconds) at the current rate."""
        remaining = max(0, self.fields["total_steps"] - step)
        eta = None if not steps_per_s else remaining / steps_per_s
        self.update(state="running", step=step, steps_per_s=steps_per_s, eta_s=eta, **fields)

    def write(self) -> None:
        """Rewrite the file atomically."""
        self.fields["updated_unix"] = time.time()
        write_json_atomic(self.path, self.fields)
