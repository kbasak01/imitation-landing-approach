"""Tests for ``ila.release`` on a synthetic repository (no checkpoints, no artifacts needed)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ila import release


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake repo: two .pt selections, one R1 ONNX selection, two graphs, pins, graphs.json."""
    root = tmp_path / "repo"
    pins: dict[str, str] = {}
    for arm, seed in (("bc", 0), ("dp", 1)):
        ckpt = root / "artifacts" / "runs" / arm / f"s{seed}" / "checkpoints" / "s0_step010000.pt"
        sha = _write(ckpt, f"{arm}{seed}".encode())
        sel = {"selected": {"candidate": {"file": str(ckpt), "sha256": sha}}}
        pins[f"{arm}/s{seed}"] = _write(
            root / "results" / "phase4" / "runs" / arm / f"s{seed}" / "selection.json",
            json.dumps(sel).encode(),
        )
    r1 = root / "artifacts" / "runs" / "R1_lerobot_dp" / "s0" / "onnx" / "step100000"
    enc = _write(r1 / "encoder.onnx", b"enc")
    den = _write(r1 / "denoiser.onnx", b"den")
    meta = _write(
        r1 / "meta.json",
        json.dumps({"onnx_sha256": {"encoder.onnx": enc, "denoiser.onnx": den}}).encode(),
    )
    sel = {"selected": {"candidate": {"file": str(r1), "sha256": meta}}}
    pins["R1_lerobot_dp/s0"] = _write(
        root / "results" / "phase6" / "runs" / "R1_lerobot_dp" / "s0" / "selection.json",
        json.dumps(sel).encode(),
    )
    _write(root / release.PINS, json.dumps({"pins": pins}).encode())
    gdir = root / release.GRAPH_DIR
    graphs = {
        n: {"sha256": _write(gdir / n, n.encode())} for n in ("bc_dagger.onnx", "dp_K10.onnx")
    }
    gmeta = _write(gdir / "meta.json", b"{}")
    _write(
        root / release.GRAPHS_JSON, json.dumps({"graphs": graphs, "meta_sha256": gmeta}).encode()
    )
    return root


def test_assets_cover_every_pin_and_graph(repo: Path) -> None:
    names = [a.name for a in release.all_assets(repo)]
    assert names == sorted(names)
    assert set(names) == {
        "R1_lerobot_dp_s0_step100000_denoiser.onnx",
        "R1_lerobot_dp_s0_step100000_encoder.onnx",
        "R1_lerobot_dp_s0_step100000_meta.json",
        "bc_dagger.onnx",
        "bc_s0_s0_step010000.pt",
        "dp_K10.onnx",
        "dp_s1_s0_step010000.pt",
        "graphs_meta.json",
    }


def test_verify_and_sha256sums_round_trip(repo: Path, tmp_path: Path) -> None:
    assets = release.all_assets(repo)
    sizes = release.verify(assets, repo)
    assert sizes["bc_s0_s0_step010000.pt"] == len(b"bc0")
    sums = release.sha256sums(assets)
    assert sums.endswith("\n") and "\r" not in sums
    assert all(len(line.split("  ")[0]) == 64 for line in sums.splitlines())
    out = tmp_path / "stage"
    release.stage(assets, out, repo)
    assert (out / "SHA256SUMS.txt").read_text() == sums
    res = subprocess.run(
        ["sha256sum", "-c", "--quiet", "SHA256SUMS.txt"], cwd=out, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout + res.stderr
    man = json.loads(release.manifest(assets, sizes, repo))
    assert man["n_assets"] == len(assets)
    assert man["total_bytes"] == sum(sizes.values())
    assert release.manifest(assets, sizes, repo) == release.manifest(assets, sizes, repo)


def test_verify_refuses_a_changed_file(repo: Path) -> None:
    assets = release.all_assets(repo)
    (repo / "artifacts/runs/bc/s0/checkpoints/s0_step010000.pt").write_bytes(b"tampered")
    with pytest.raises(release.ReleaseError, match="bc_s0_s0_step010000.pt"):
        release.verify(assets, repo)


def test_selection_must_hash_to_its_pin(repo: Path) -> None:
    sel = repo / "results/phase4/runs/bc/s0/selection.json"
    sel.write_bytes(sel.read_bytes() + b" ")
    with pytest.raises(release.ReleaseError, match="does not hash to its pin"):
        release.all_assets(repo)


def test_missing_selection_is_refused(repo: Path) -> None:
    (repo / "results/phase4/runs/dp/s1/selection.json").unlink()
    with pytest.raises(release.ReleaseError, match="no committed selection"):
        release.all_assets(repo)
