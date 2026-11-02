"""R1 loader-alone throughput (CPU, .venv-lerobot), DataLoader built exactly as train_r1.py builds it.

Modes:
  cycle      -- lerobot.utils.utils.cycle(dataloader): a fresh iter() per epoch (what training does)
  continuous -- the identical EpisodeAwareSampler, its epochs chained inside one sampler iterator
                (one DataLoader iterator, no per-epoch pipeline restart). Same index stream; the
                partial batch at each epoch end is merged into the next (drop_last False).
Discard the first `--discard` batches, time the next `--timed`; per-batch inter-arrival times kept.
The consumer does nothing but the uint8 -> float /255 the training loop does (no preprocessor, no
model), so this is the loader's ceiling. Writes nothing in the repo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO = Path("/home/kbasak/projects_02/imitation-landing-approach")


class ChainedEpochs:
    def __init__(self, sampler, n_epochs: int = 1000) -> None:
        self.sampler, self.n_epochs = sampler, n_epochs

    def __iter__(self):
        for _ in range(self.n_epochs):
            yield from iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler) * self.n_epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None, help="e.g. 0-7; default all 999")
    ap.add_argument("--mode", choices=["cycle", "continuous"], default="cycle")
    ap.add_argument("--workers", type=int, default=None, help="default: config's (12)")
    ap.add_argument("--discard", type=int, default=100)
    ap.add_argument("--timed", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("train_r1", REPO / "scripts/r1/train_r1.py")
    tr = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(tr)
    import torch
    import yaml
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.utils.utils import cycle

    cfg = yaml.safe_load(tr.CONFIG.read_text(encoding="utf-8"))
    data_root = tr.data_root_from_paths()
    ds_root = data_root / "lerobot" / cfg["data"]["dataset"]
    scratch_run = Path(args.out).with_suffix(".rundir")
    train_cfg = tr.build_configs(
        cfg, dataset_root=ds_root, episodes=tr.parse_episodes(args.episodes), seed=0,
        device="cpu", steps=int(cfg["train"]["steps"]), save_freq=int(cfg["train"]["save_freq"]),
        run_dir=scratch_run, num_workers=args.workers,
    )
    t0 = time.perf_counter()
    dataset, _ = make_train_eval_datasets(train_cfg)
    t_ds = time.perf_counter() - t0
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(train_cfg.policy, "drop_n_last_frames", 0),
        shuffle=True,
        seed=train_cfg.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    workers = train_cfg.num_workers > 0
    loader_sampler = sampler if args.mode == "cycle" else ChainedEpochs(sampler)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_cfg.num_workers,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        sampler=loader_sampler,
        pin_memory=False,  # train_r1: device.type == "cuda" (True there); CPU-only here
        drop_last=False,
        collate_fn=None,
        prefetch_factor=train_cfg.prefetch_factor if workers else None,
        persistent_workers=train_cfg.persistent_workers and workers,
        multiprocessing_context=train_cfg.dataloader_multiprocessing_context if workers else None,
    )
    it = cycle(dataloader) if args.mode == "cycle" else iter(dataloader)
    cam_keys = dataset.meta.camera_keys
    stamps = []
    sizes = []
    t_start = time.perf_counter()
    for _ in range(args.discard + args.timed):
        batch = next(it)
        for k in cam_keys:
            if k in batch and batch[k].dtype == torch.uint8:
                batch[k] = batch[k].to(dtype=torch.float32) / 255.0
        stamps.append(time.perf_counter())
        sizes.append(int(batch["action"].shape[0]))
    import numpy as np

    st = np.asarray(stamps)
    timed = st[args.discard:]
    dt = np.diff(np.concatenate([[st[args.discard - 1]], timed]))
    rate = args.timed / (timed[-1] - st[args.discard - 1])
    rec = {
        "episodes": args.episodes or "all",
        "mode": args.mode,
        "num_workers": train_cfg.num_workers,
        "prefetch_factor": train_cfg.prefetch_factor,
        "persistent_workers": train_cfg.persistent_workers,
        "mp_context": train_cfg.dataloader_multiprocessing_context,
        "batch_size": train_cfg.batch_size,
        "dataset_frames": dataset.num_frames,
        "dataset_episodes": dataset.num_episodes,
        "sampler_len": len(sampler),
        "batches_per_epoch": -(-len(sampler) // train_cfg.batch_size),
        "dataset_build_s": t_ds,
        "first_batch_s": float(st[0] - t_start),
        "discard": args.discard,
        "timed": args.timed,
        "steady_batches_per_s": rate,
        "first100_batches_per_s": args.discard / (st[args.discard - 1] - st[0]) if args.discard > 1 else None,
        "dt_median_s": float(np.median(dt)),
        "dt_p90_s": float(np.quantile(dt, 0.9)),
        "dt_max_s": float(dt.max()),
        "partial_batches_in_timed": int(sum(s < train_cfg.batch_size for s in sizes[args.discard:])),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "loadavg_end": os.getloadavg(),
    }
    print(json.dumps(rec, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    del it, dataloader


if __name__ == "__main__":
    main()
