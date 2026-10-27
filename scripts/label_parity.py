"""Teacher label parity: SB3 vs ONNX/ORT CPU on 10 000 train-pool PPO observations.

Writes ``results/label_parity.json``; the observations go to
``artifacts/runs/label_parity/observations.npy`` (gitignored). Exit 0 if the 1e-6 criterion
is met, 1 otherwise (the SB3 path is then the label path).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import label_parity as lp
from ila.eval.runner import init_worker_threads, set_single_thread_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=lp.N_OBS)
    parser.add_argument("--out", type=Path, default=Path("results/label_parity.json"))
    parser.add_argument(
        "--obs-out", type=Path, default=Path("artifacts/runs/label_parity/observations.npy")
    )
    args = parser.parse_args()
    threads = set_single_thread_env()
    init_worker_threads()
    started = time.perf_counter()
    obs, where, flown = lp.record_observations(args.n_obs)
    args.obs_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.obs_out, obs)
    result = lp.compare(obs, where)
    record = {
        "item": "plan Phase 2 item 3 (teacher label parity)",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **result,
        "observations": {
            "n": int(obs.shape[0]),
            "dim": int(obs.shape[1]),
            "sha256": lp.obs_sha256(obs),
            "sha256_rule": "sha256(str(shape) + C-order little-endian float32 bytes)",
            "source": "ppo seed 0 rollouts (render off, P8 loop), first n in flight order",
            "pool": "P8 dev_pool train",
            "stream": lp.STREAM,
            "generator_seed": lp.SEED,
            "episodes_flown": len(flown),
            "episodes_content_sha256": data_episodes.content_sha256(flown),
            "sea_states": sorted({e.ss for e in flown}),
            "file": args.obs_out.as_posix(),
        },
        "threads": {**threads, "torch_num_threads": 1, "ort_threads": 1},
        "p8_commit": p8_adapter.p8_commit(),
        "cpu_count": os.cpu_count(),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    primary = record["primary_sb3_reference_vs_ort_cpu_single"]
    print(
        f"max_abs {primary['max_abs']:.3g}, p99 {primary['p99_abs']:.3g}, "
        f"argmax {primary.get('argmax_episode')} comp {primary['argmax_component']}; "
        f"passed={record['passed']} -> {record['label_path']}; wrote {args.out}"
    )
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
