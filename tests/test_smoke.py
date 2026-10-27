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


def _is_rld(name: str) -> bool:
    return name == "rld" or name.startswith("rld.")


def _rld_imports(source: str) -> list[int]:
    """Return the line numbers that import ``rld``.

    Covers import statements, and ``importlib.import_module("rld…")``,
    ``import_module("rld…")`` and ``__import__("rld…")`` with a literal module name.
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hit = any(_is_rld(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            hit = node.level == 0 and _is_rld(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0] if node.args else None
            hit = (
                name in {"import_module", "__import__"}
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _is_rld(first.value)
            )
        else:
            continue
        if hit:
            lines.append(node.lineno)
    return lines


def test_rld_import_scanner_patterns() -> None:
    bad = [
        "import rld",
        "import rld.eval.stats as s",
        "from rld.envs import observation",
        'importlib.import_module("rld.eval")',
        'import_module("rld")',
        '__import__("rld.envs")',
    ]
    for line in bad:
        assert _rld_imports(line) == [1], line
    for line in ("import rldx", "from ila import p8_adapter", 'import_module("ila.rld")'):
        assert _rld_imports(line) == [], line


def test_only_adapter_imports_rld() -> None:
    files = sorted(SRC.rglob("*.py")) + sorted((SRC.parents[1] / "scripts").rglob("*.py"))
    assert any(p.parent.name == "scripts" for p in files)
    offenders = [
        f"{path}:{line}"
        for path in files
        if path != SRC / "p8_adapter.py"
        for line in _rld_imports(path.read_text(encoding="utf-8"))
    ]
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
