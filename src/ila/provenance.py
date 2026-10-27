"""P8 provenance (Phase 0): pinned commits, frozen-file hashes and the PPO seed-0 teacher's hashes.

Every expected value below is P8's published one: the commit is the pin in ``.gitmodules`` /
``REPO_SETUP.md``, the two frozen-file hashes are P8 ``docs/protocol.md`` (P3-D1), and the
teacher's checkpoint hashes are cross-checked against P8's committed
``results/latency/run_info.json`` (seed-0 ``ppo`` graph entry). Nothing here imports ``rld``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ila import p8_adapter

__all__ = [
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_P8_COMMIT",
    "EXPECTED_P8_SUBMODULES",
    "EXPECTED_SUCCESS_SHA256",
    "Check",
    "collect",
    "write",
]

#: P8 release merge (PR #9, "Gate 9 passed").
EXPECTED_P8_COMMIT = "7237b7065c8af8172b3848ac6e9d3ceb2cf82075"
#: P8's own pins of its two submodules.
EXPECTED_P8_SUBMODULES = {
    "third_party/deck-motion-forecast": "e9fa15cc35312a5c1ecf0c42668a690b5061c090",
    "third_party/gym-pybullet-drones": "7ebad1ecabd28a7000add2d05f888aa2e837c2cc",
}
#: ``results/episodes/MANIFEST.csv`` (plan: ``e6f30e55…``; full value from P8 protocol P3-D1).
EXPECTED_MANIFEST_SHA256 = "e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2"
#: ``configs/env/success.yaml`` (plan: ``c7fbdcc4…``; full value from P8 protocol P3-D1).
EXPECTED_SUCCESS_SHA256 = "c7fbdcc4df48bd4a2cfe21811613e5e50e3e4868822fe6dddd13745d96692118"

_RUN_INFO = Path("results") / "latency" / "run_info.json"


@dataclass(frozen=True)
class Check:
    """One provenance comparison.

    Attributes:
        name: What is compared.
        expected: The published value.
        actual: The value found on this machine.
        match: ``expected == actual``.
    """

    name: str
    expected: str
    actual: str
    match: bool


def _check(name: str, expected: str, actual: str) -> Check:
    return Check(name=name, expected=expected, actual=actual, match=expected == actual)


def _ppo_seed0_entry(p8_root: Path) -> dict[str, Any]:
    info = json.loads((p8_root / _RUN_INFO).read_text(encoding="utf-8"))
    for graph in info["graphs"]:
        if graph["method"] == "ppo" and graph["seed"] == 0:
            entry: dict[str, Any] = graph
            return entry
    raise KeyError(f"{_RUN_INFO}: no ppo seed-0 graph entry")


def collect(paths: dict[str, Path]) -> dict[str, Any]:
    """Compute every Phase 0 provenance check.

    Args:
        paths: :func:`ila.p8_adapter.load_paths` output; uses ``p8_ppo_seed0_run`` (the run
            directory holding ``final/``) and ``p8_ppo_seed0_onnx``.

    Returns:
        A JSON-ready dict: ``checks`` (list of :class:`Check` dicts), ``all_match`` and the
        file paths that were hashed.
    """
    p8_root = p8_adapter.P8_ROOT
    final = paths["p8_ppo_seed0_run"] / "final"
    published = _ppo_seed0_entry(p8_root)
    submodules = p8_adapter.p8_submodule_commits()
    checks = [
        _check("p8_commit", EXPECTED_P8_COMMIT, p8_adapter.p8_commit()),
        *(
            _check(f"p8_submodule:{sub}", sha, submodules.get(sub, "<missing>"))
            for sub, sha in EXPECTED_P8_SUBMODULES.items()
        ),
        _check("manifest_sha256", EXPECTED_MANIFEST_SHA256, p8_adapter.manifest_sha256()),
        _check("success_yaml_sha256", EXPECTED_SUCCESS_SHA256, p8_adapter.success_yaml_sha256()),
        _check(
            "ppo_s0_model_zip_sha256",
            str(published["model_sha256"]),
            p8_adapter.sha256_file(final / "model.zip"),
        ),
        _check(
            "ppo_s0_vecnormalize_sha256",
            str(published["vecnormalize_sha256"]),
            p8_adapter.sha256_file(final / "vecnormalize.pkl"),
        ),
        _check(
            "ppo_s0_onnx_sha256",
            str(published["onnx_sha256"]),
            p8_adapter.sha256_file(paths["p8_ppo_seed0_onnx"]),
        ),
    ]
    return {
        "all_match": all(c.match for c in checks),
        "checks": [asdict(c) for c in checks],
        "published_reference": f"{p8_root.name}/{_RUN_INFO.as_posix()} graphs[ppo, seed 0]",
        "ppo_s0_checkpoint_json_sha256": p8_adapter.sha256_file(final / "checkpoint.json"),
        "ppo_s0_checkpoint": "final",
    }


def write(record: dict[str, Any], out: Path) -> None:
    """Write the provenance record as sorted, indented JSON with a trailing newline."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
