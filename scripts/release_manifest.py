"""Build and verify the release assets (logic: ila.release). Simulation only.

Re-hashes every pinned student checkpoint and every Phase 9 graph against its committed SHA-256,
writes ``results/release/{SHA256SUMS.txt,manifest.json}`` and, with ``--stage``, hardlinks the
assets into ``artifacts/release/`` (gitignored) with the same ``SHA256SUMS.txt``:

    python scripts/release_manifest.py --stage
    (cd artifacts/release && sha256sum -c SHA256SUMS.txt)

``--check`` re-hashes and compares with the committed files without writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila import release
from ila.p8_adapter import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "release")
    parser.add_argument("--stage", action="store_true", help="hardlink assets into --stage-dir")
    parser.add_argument("--stage-dir", type=Path, default=REPO_ROOT / "artifacts" / "release")
    parser.add_argument("--check", action="store_true", help="compare with --out; write nothing")
    args = parser.parse_args()

    assets = release.all_assets()
    sizes = release.verify(assets)
    sums = release.sha256sums(assets)
    man = release.manifest(assets, sizes)
    print(f"{len(assets)} assets, {sum(sizes.values()) / 2**30:.2f} GiB, all SHA-256 verified")
    if args.check:
        ok = True
        for name, text in (("SHA256SUMS.txt", sums), ("manifest.json", man)):
            path = args.out / name
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            print(f"{path}: {'identical' if same else 'DIFFERS'}")
            ok = ok and same
        return 0 if ok else 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    (args.out / "manifest.json").write_text(man, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}/SHA256SUMS.txt, manifest.json")
    if args.stage:
        release.stage(assets, args.stage_dir)
        print(f"staged {len(assets)} assets + SHA256SUMS.txt in {args.stage_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
