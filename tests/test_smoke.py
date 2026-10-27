"""Phase 0 smoke: P8 and ila import; one P8 env built through the adapter resets cleanly."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import ila
from ila import p8_adapter

SRC = Path(ila.__file__).resolve().parent


def test_imports() -> None:
    import rld

    assert rld.__name__ == "rld"
    assert p8_adapter.P8_ROOT.name == "rl-land-on-moving-deck"


@pytest.mark.pybullet
def test_one_env_reset() -> None:
    cfgs = p8_adapter.load_eval_configs()
    listed = p8_adapter.read_list(p8_adapter.list_path("id"))[0]
    env = p8_adapter.make_env(cfgs, listed)
    try:
        obs, _ = env.reset(seed=listed.episode_seed)
    finally:
        env.close()
    obs = np.asarray(obs)
    assert obs.shape == (25,)
    assert np.all(np.isfinite(obs))


def test_only_adapter_imports_rld() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "p8_adapter.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "rld" or n.startswith("rld.") for n in names):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == []


def test_frozen_file_hashes() -> None:
    from ila import provenance

    assert p8_adapter.manifest_sha256() == provenance.EXPECTED_MANIFEST_SHA256
    assert p8_adapter.success_yaml_sha256() == provenance.EXPECTED_SUCCESS_SHA256


@pytest.mark.checkpoints
def test_provenance_all_match() -> None:
    from ila import provenance

    record = provenance.collect(p8_adapter.load_paths())
    assert record["all_match"], [c for c in record["checks"] if not c["match"]]
