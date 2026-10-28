"""Write the select list and the demo lists to splits/, with MANIFEST.csv.

Every row is resolved through P8's own ``resolve_draws`` (one real reset per episode) and
checked against the pools, the five frozen lists, SS6 and 90 deg before it is written
(``ila.data.episodes``). With no option this regenerates every committed split file byte for
byte: ``select.parquet`` and the demo-stream prefixes the collections in
``configs/data/demos.yaml`` fly (``demos_0-999.parquet``, ``demos_0-1999.parquet``). Examples::

    python scripts/make_splits.py --workers 8          # select + demo lists
    python scripts/make_splits.py --no-select --no-demos --train 30 --stream scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ila.data import episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=episodes.SPLITS_DIR)
    parser.add_argument("--config", type=Path, default=episodes.DEFAULT_CONFIG)
    parser.add_argument("--no-select", action="store_true", help="skip the select list")
    parser.add_argument("--no-demos", action="store_true", help="skip the demo lists")
    parser.add_argument("--train", type=int, default=0, help="extra train episodes (0: none)")
    parser.add_argument("--start", type=int, default=0, help="first global train index")
    parser.add_argument("--stream", default=episodes.TRAIN_STREAM, help="extra train stream label")
    parser.add_argument(
        "--workers", type=int, default=1, help="reset processes (no effect on rows)"
    )
    args = parser.parse_args()

    cfg = episodes.load_splits_config(args.config)
    api = episodes.p8_episode_api(workers=args.workers)
    written: list[episodes.SplitManifestRow] = []
    if not args.no_select:
        rows = episodes.select_rows(api, cfg)
        written.append(
            episodes.write_split(
                episodes.SELECT_NAME,
                rows,
                args.out,
                pool=cfg.select_pool,
                generator_seed=cfg.generator_seed,
                draw=episodes.SELECT_DRAW_RULE,
            )
        )
    if not args.no_demos:
        design = episodes.load_stream_design(args.config)
        for name, rows in episodes.demo_rows(api, cfg, episodes.demo_lists(design)).items():
            written.append(
                episodes.write_split(
                    name,
                    rows,
                    args.out,
                    pool=cfg.train_pool,
                    generator_seed=cfg.generator_seed,
                    draw=episodes.DEMO_DRAW_RULE,
                )
            )
    if args.train > 0:
        rows = episodes.train_rows(api, cfg, args.train, start=args.start, stream=args.stream)
        written.append(
            episodes.write_split(
                episodes.train_name(args.train, start=args.start, stream=args.stream),
                rows,
                args.out,
                pool=cfg.train_pool,
                generator_seed=cfg.generator_seed,
                draw=episodes.TRAIN_DRAW_RULE,
            )
        )
    for row in written:
        print(
            f"{row.file}: {row.rows} rows, cells {row.cells} = {row.n_per_cell}, "
            f"{row.n_realizations} realizations, file {row.file_sha256}, "
            f"content {row.content_sha256}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
