"""Release assets: the pinned student checkpoints and the Phase 9 ONNX graphs, with SHA-256 sums.

Simulation only. Every asset is a file this project trained or exported; nothing is re-trained or
re-exported here. Each asset's SHA-256 comes from a **committed** record and is re-checked against
the file's bytes before anything is staged:

* the 71 pinned selections (``results/e08/selection_pins.json``) -> each run's committed
  ``results/phase*/runs/<arm>/s<n>/selection.json`` (its own SHA-256 must equal the pin) -> the
  selected checkpoint and its ``sha256``. A ``.pt`` checkpoint ships as stored (model, EMA and
  optimiser state), so the committed SHA applies to the released bytes unchanged. R1's selection
  is an ONNX export directory: its ``meta.json`` carries the selected SHA, and its two graphs
  carry the SHAs recorded in that ``meta.json`` (``onnx_sha256``);
* ``results/latency/graphs.json`` -> the 8 Phase 9 graphs and their ``meta.json``
  (``meta_sha256``), shipped as ``graphs_meta.json``.

Staging hardlinks (or copies) each asset under a flat, unique name into a directory and writes
``SHA256SUMS.txt`` in ``sha256sum`` format (two spaces, sorted by name, LF) so
``sha256sum -c SHA256SUMS.txt`` verifies the upload. ``manifest.json`` maps every asset name to
its source, size, committed SHA and the record that commits it.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ila.p8_adapter import REPO_ROOT, sha256_file

PINS = Path("results/e08/selection_pins.json")
GRAPHS_JSON = Path("results/latency/graphs.json")
GRAPH_DIR = Path("artifacts/deploy/graphs")
SELECTION_GLOB = "results/phase*/runs/*/s*/selection.json"


class ReleaseError(RuntimeError):
    """A committed record and a file on disk disagree, or an asset name is not unique."""


@dataclass(frozen=True)
class Asset:
    """One release asset.

    Attributes:
        name: Flat asset name on the release (unique).
        source: File on disk (absolute, or relative to the repository root).
        sha256: SHA-256 hex digest taken from a committed record.
        kind: ``checkpoint``, ``r1_onnx``, ``r1_meta``, ``graph`` or ``graph_meta``.
        record: Committed file (repository-relative) that records ``sha256``.
        arm: Student arm, or ``""`` for graph metadata shared by both exported arms.
        seed: Run seed, or ``None`` for the Phase 9 graphs (seed 0 by construction, P9-D1).
    """

    name: str
    source: str
    sha256: str
    kind: str
    record: str
    arm: str
    seed: int | None


def _load(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def selection_records(root: Path = REPO_ROOT) -> dict[str, Path]:
    """Map every pin key ``<arm>/s<n>`` to its committed ``selection.json``, checked by SHA.

    Raises:
        ReleaseError: A pin has no committed selection, or the selection's bytes do not hash to
            the pin.
    """
    pins: dict[str, str] = _load(root / PINS)["pins"]
    found: dict[str, Path] = {}
    for path in sorted(root.glob(SELECTION_GLOB)):
        key = f"{path.parent.parent.name}/{path.parent.name}"
        if key in pins:
            found[key] = path
    out: dict[str, Path] = {}
    for key, pin in sorted(pins.items()):
        if key not in found:
            raise ReleaseError(f"pin {key}: no committed selection.json under {SELECTION_GLOB}")
        if sha256_file(found[key]) != pin:
            raise ReleaseError(f"pin {key}: {found[key]} does not hash to its pin")
        out[key] = found[key]
    return out


def checkpoint_assets(root: Path = REPO_ROOT) -> list[Asset]:
    """The selected checkpoint of every pinned run, as recorded in its ``selection.json``."""
    assets: list[Asset] = []
    for key, sel_path in selection_records(root).items():
        arm, seed_dir = key.split("/")
        seed = int(seed_dir.removeprefix("s"))
        cand = _load(sel_path)["selected"]["candidate"]
        src = Path(cand["file"])
        record = _rel(sel_path, root)
        if src.suffix == ".pt":
            assets.append(
                Asset(
                    f"{arm}_s{seed}_{src.name}",
                    str(src),
                    cand["sha256"],
                    "checkpoint",
                    record,
                    arm,
                    seed,
                )
            )
            continue
        # R1: an ONNX export directory; meta.json is the selected file, its graphs are in it.
        meta_path = src / "meta.json"
        tag = f"{arm}_s{seed}_{src.name}"
        assets.append(
            Asset(f"{tag}_meta.json", str(meta_path), cand["sha256"], "r1_meta", record, arm, seed)
        )
        if meta_path.exists():
            onnx_shas: dict[str, str] = _load(meta_path)["onnx_sha256"]
        else:
            raise ReleaseError(f"{key}: {meta_path} missing")
        for fname, sha in sorted(onnx_shas.items()):
            assets.append(
                Asset(
                    f"{tag}_{fname}",
                    str(src / fname),
                    sha,
                    "r1_onnx",
                    f"{record} -> meta.json onnx_sha256",
                    arm,
                    seed,
                )
            )
    return assets


def graph_assets(root: Path = REPO_ROOT, graph_dir: Path = GRAPH_DIR) -> list[Asset]:
    """The Phase 9 graphs and their ``meta.json``, from ``results/latency/graphs.json``."""
    rec = _load(root / GRAPHS_JSON)
    gdir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    record = GRAPHS_JSON.as_posix()
    assets = [
        Asset(
            name,
            str(gdir / name),
            g["sha256"],
            "graph",
            record,
            "bc_dagger" if name.startswith("bc_dagger") else "dp",
            0,
        )
        for name, g in sorted(rec["graphs"].items())
    ]
    assets.append(
        Asset(
            "graphs_meta.json",
            str(gdir / "meta.json"),
            rec["meta_sha256"],
            "graph_meta",
            f"{record} meta_sha256",
            "",
            None,
        )
    )
    return assets


def all_assets(root: Path = REPO_ROOT, graph_dir: Path = GRAPH_DIR) -> list[Asset]:
    """Every release asset, sorted by name.

    Raises:
        ReleaseError: Two assets share a name.
    """
    assets = sorted(checkpoint_assets(root) + graph_assets(root, graph_dir), key=lambda a: a.name)
    names = [a.name for a in assets]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ReleaseError(f"duplicate asset names: {dup}")
    return assets


def verify(assets: Iterable[Asset], root: Path = REPO_ROOT) -> dict[str, int]:
    """Re-hash every asset file against its committed SHA-256.

    Returns:
        Asset name -> size in bytes.

    Raises:
        ReleaseError: Listing every missing file and every SHA mismatch.
    """
    sizes: dict[str, int] = {}
    problems: list[str] = []
    for a in assets:
        src = Path(a.source) if Path(a.source).is_absolute() else root / a.source
        if not src.is_file():
            problems.append(f"{a.name}: missing {src}")
            continue
        got = sha256_file(src)
        if got != a.sha256:
            problems.append(f"{a.name}: {src} hashes to {got}, record says {a.sha256}")
            continue
        sizes[a.name] = src.stat().st_size
    if problems:
        raise ReleaseError("; ".join(problems))
    return sizes


def sha256sums(assets: Sequence[Asset]) -> str:
    """``sha256sum``-format text: ``<hex>  <name>`` per asset, sorted by name, LF endings."""
    return "".join(f"{a.sha256}  {a.name}\n" for a in sorted(assets, key=lambda a: a.name))


def manifest(assets: Sequence[Asset], sizes: dict[str, int], root: Path = REPO_ROOT) -> str:
    """Deterministic JSON (sorted keys, indent 2, LF) describing every asset."""
    rows = []
    for a in sorted(assets, key=lambda a: a.name):
        d = asdict(a)
        d["source"] = _rel(Path(a.source), root)
        d["size_bytes"] = sizes[a.name]
        rows.append(d)
    body = {
        "what": "release assets: pinned student checkpoints as stored, R1 ONNX exports, Phase 9 "
        "graphs; every SHA-256 from a committed record, re-hashed before staging (simulation only)",
        "n_assets": len(rows),
        "total_bytes": sum(sizes.values()),
        "verify": "sha256sum -c SHA256SUMS.txt in the directory holding the downloaded assets",
        "assets": rows,
    }
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


def stage(assets: Sequence[Asset], out_dir: Path, root: Path = REPO_ROOT) -> None:
    """Hardlink (copy across filesystems) every asset into ``out_dir`` under its asset name.

    An existing file of the same name is replaced only if its bytes differ in SHA-256.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in assets:
        src = Path(a.source) if Path(a.source).is_absolute() else root / a.source
        dst = out_dir / a.name
        if dst.exists():
            if sha256_file(dst) == a.sha256:
                continue
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    (out_dir / "SHA256SUMS.txt").write_text(sha256sums(assets), encoding="utf-8", newline="\n")
