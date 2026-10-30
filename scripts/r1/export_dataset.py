"""Export ``dp``'s training episodes to R1's neutral intermediate (Phase 6, ``.venv`` only).

Step 1 of 2 for the reference arm ``R1_lerobot_dp`` (P3-D1 §3: "same 1 000"). Reads
``configs/train/dp.yaml``'s data (``ppo_2000``, first 1 000 by stream index, success filter)
through ``ila.data.export_lerobot`` -- the ``dp`` training path's own episode choice and the
student read path only -- and writes ``${data_root}/lerobot/r1_ppo_2000_first1000/``
(``ep_<k>.npz`` + ``manifest.json``), then re-checks every episode against the store byte for
byte. Step 2 (``.venv-lerobot`` only) is ``scripts/r1/build_lerobot_dataset.py``. Example::

    .venv/bin/python scripts/r1/export_dataset.py
    .venv/bin/python scripts/r1/export_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ila import p8_adapter
from ila.data import export_lerobot as ex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: data_root/lerobot/...")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="skip the byte re-check")
    args = parser.parse_args()

    data_root = p8_adapter.load_paths()["data_root"] if args.data_root is None else args.data_root
    t0 = time.perf_counter()
    result = ex.export_r1(data_root, args.out, overwrite=args.overwrite)
    m = result.manifest
    print(
        f"wrote {result.out_dir}: kept {m['kept']} / attempted {m['attempted']}, "
        f"{m['frames']} frames, outcomes {m['selection']['outcomes']}, "
        f"store sha matches committed: {m['store']['matches']} "
        f"({time.perf_counter() - t0:.1f} s)"
    )
    if not args.skip_verify:
        t1 = time.perf_counter()
        report = ex.verify_export(result.out_dir, data_root)
        print(f"verified: {json.dumps(report)} ({time.perf_counter() - t1:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
