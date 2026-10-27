"""The only module in ``ila`` that imports P8 (``rld``); a Phase 0 stub, completed in Phase 1.

P8 (``external/rl-land-on-moving-deck``) is a pinned, read-only submodule. Everything this
project takes from it -- configs, frozen episode lists, the environment, teachers and
statistics -- passes through here, unchanged. Lengths are metres and times seconds, model
scale (Froude lambda = 1/25), unless stated.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from rld.config import REPO_ROOT as P8_ROOT
from rld.envs.landing_env import DeckLandingAviary
from rld.eval.envs import EvalConfigs, load_eval_configs, motion_for
from rld.eval.envs import make_env as _p8_make_env
from rld.eval.episodes import ListedEpisode, read_list

__all__ = [
    "P8_ROOT",
    "REPO_ROOT",
    "DeckLandingAviary",
    "EvalConfigs",
    "ListedEpisode",
    "list_path",
    "load_eval_configs",
    "load_paths",
    "make_env",
    "manifest_path",
    "manifest_sha256",
    "p8_commit",
    "p8_submodule_commits",
    "read_list",
    "sha256_file",
    "success_yaml_path",
    "success_yaml_sha256",
]

#: This repository's root, resolved from this file so nothing depends on the working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"
_VAR = re.compile(r"\$\{(\w+)\}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paths(path: Path = _PATHS_FILE) -> dict[str, Path]:
    """Load ``configs/paths.local.yaml``, expanding ``${key}`` references to earlier keys.

    Args:
        path: The local paths file (gitignored; copy of ``paths.local.example.yaml``).

    Returns:
        Key -> absolute path; relative entries resolve against this repository's root.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a ``${key}`` reference names an undefined key.
    """
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = _VAR.sub(lambda m: resolved[m.group(1)], str(value))
    return {k: (REPO_ROOT / v).resolve() for k, v in resolved.items()}


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout
    return out.strip()


def p8_commit() -> str:
    """Return the P8 submodule's checked-out commit SHA (40 hex characters)."""
    return _git("rev-parse", "HEAD", cwd=P8_ROOT)


def p8_submodule_commits() -> dict[str, str]:
    """Return P8's nested submodules (``third_party/*``) as path -> checked-out commit SHA."""
    commits: dict[str, str] = {}
    for line in _git("submodule", "status", cwd=P8_ROOT).splitlines():
        # Each line is "<status char><sha> <path> [(<describe>)]"; the status char is one of
        # " +-U", and the leading space of the first line is lost to strip().
        sha, sub_path = line.split()[:2]
        commits[sub_path] = sha.lstrip("+-U")
    return commits


def success_yaml_path() -> Path:
    """Return the path of P8's frozen ``configs/env/success.yaml``."""
    return P8_ROOT / "configs" / "env" / "success.yaml"


def success_yaml_sha256() -> str:
    """Return the SHA-256 of P8's frozen success criteria file."""
    return sha256_file(success_yaml_path())


def manifest_path() -> Path:
    """Return the path of P8's frozen episode-list ``results/episodes/MANIFEST.csv``."""
    return P8_ROOT / "results" / "episodes" / "MANIFEST.csv"


def manifest_sha256() -> str:
    """Return the SHA-256 of P8's frozen episode-list manifest."""
    return sha256_file(manifest_path())


def list_path(name: str) -> Path:
    """Return the path of one of P8's frozen episode lists, e.g. ``"id"`` or ``"static"``."""
    return P8_ROOT / "results" / "episodes" / f"{name}.parquet"


def make_env(cfgs: EvalConfigs, listed: ListedEpisode) -> DeckLandingAviary:
    """Build P8's landing environment for one listed episode, exactly as P8's runner does.

    Args:
        cfgs: P8's committed configs (:func:`load_eval_configs`).
        listed: One row of a frozen list (:func:`read_list`).

    Returns:
        P8's ``DeckLandingAviary`` (240 Hz physics, 30 Hz control). The caller resets it with
        ``seed=listed.episode_seed`` and closes it.
    """
    motion = motion_for(
        cfgs,
        listed.vessel,
        listed.ss,
        listed.heading_deg,
        listed.speed_kn,
        listed.realization_seed,
        episode_seed=listed.episode_seed,
    )
    return _p8_make_env(cfgs, motion, listed.vessel, listed.pad, listed.episode_seed)
