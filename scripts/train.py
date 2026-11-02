"""Train one arm and seed from its frozen config (Phase 4). A background job: never foreground.

``make train-bg CFG=configs/train/<arm>.yaml SEED=<s>`` wraps this. The run directory
(``artifacts/runs/<arm>/s<seed>/``; DART trials ``artifacts/runs/bc_dart_tune/sigma_<s>/s0/``;
timing runs ``artifacts/runs/<arm>_timing/s<seed>/``) gets ``manifest.json``, ``status.json``,
``train_log.jsonl`` and ``checkpoints/``. An existing run directory is **resumed** from its last
checkpoint with identical RNG state (P3-D1 §4) and the resume is logged. Examples::

    python scripts/train.py configs/train/bc.yaml --seed 0
    python scripts/train.py configs/train/bc.yaml --seed 0 --max-steps 2000      # P3-D1 §5 timing
    python scripts/train.py configs/train/bc_dart.yaml --seed 0 --sigma 0.10 --tuning
    python scripts/train.py configs/train/bc_dart.yaml --seed 3                   # at sigma*
    python scripts/train.py configs/train/bc_matched.yaml --seed 2                # init: bc s2
    python scripts/train.py configs/train/bc_dagger.yaml --seed 2 --rollout-workers 24
    python scripts/train.py configs/train/modular.yaml --seed 0                  # Phase 7
    python scripts/train.py configs/train/ablations/bc_plus_clock.yaml --seed 1  # 14-d proprio
    python scripts/train.py configs/train/ablations/dp_mix.yaml --seed 0         # T_mix, dp

``modular`` trains the estimator only (Huber on normalised ``priv``; the frozen ``T_ppo`` is not
part of training). An ablation config (``configs/train/ablations/<arm>.yaml``, ``base:`` the
core config) trains under its ``obs_variant`` and lands in ``artifacts/runs/<arm>/s<seed>/``
with ``tag: ablation`` in its manifest.

``bc_dagger`` (``make dagger-bg SEED=<s>``) runs ``bc_matched``'s iteration driver; each
iteration first flies its 200 on-policy rollouts on the CPU (``--rollout-workers`` processes,
rows independent of the count) from the previous iteration's last-step EMA, audits them, then
trains on the aggregate. Rollouts land in ``${data_root}/dagger/bc_dagger/s<seed>/`` and are
reused on resume.

``--max-steps`` exists only for the timing runs: such a run is labelled ``timing`` in its
manifest, writes no checkpoint and can never be selected. A ``train``/``tuning``/``timing`` run
refuses a config whose SHA-256 differs from P3-D1's and a dirty tree (``--allow-dirty``
overrides the latter and is recorded).
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import torch

from ila import p8_adapter
from ila.policies.registry import REPO_ROOT
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import DART_SIGMAS, TrainConfig

SIGMA_STAR = mf.RUNS_ROOT / "bc_dart_tune" / "sigma_star.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="configs/train/<arm>.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="P3-D1 §5 timing runs only")
    parser.add_argument("--sigma", type=float, default=None, help="bc_dart: DART sigma")
    parser.add_argument("--tuning", action="store_true", help="bc_dart sigma trial (seed 0)")
    parser.add_argument("--init-from", type=Path, default=None, help="iterated arms: bc ckpt")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=mf.RUNS_ROOT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--rollout-workers", type=int, default=24, help="bc_dagger: CPU rollout processes"
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config.resolve())
    if args.seed not in cfg.seeds and not args.tuning:
        parser.error(f"seed {args.seed} is not one of {cfg.arm}'s frozen seeds {cfg.seeds}")
    timing = args.max_steps is not None
    purpose: loop.Purpose = "timing" if timing else ("tuning" if args.tuning else "train")
    sigma = args.sigma
    if cfg.collection == "dart" and not args.tuning and not timing:
        star = mf.read_json(SIGMA_STAR)["sigma_star"] if SIGMA_STAR.is_file() else None
        if star is None:
            parser.error(f"bc_dart core runs need sigma* first ({SIGMA_STAR} missing)")
        if sigma is not None and not math.isclose(sigma, float(star)):
            parser.error(f"--sigma {sigma} differs from sigma* {star}")
        sigma = float(star)
    if args.tuning and (sigma is None or round(sigma, 6) not in DART_SIGMAS):
        parser.error(f"a DART trial needs --sigma in {DART_SIGMAS}")
    init_from = args.init_from
    if cfg.iterations is not None and init_from is None:
        init_from = mf.run_dir("bc", args.seed, root=args.run_root) / "checkpoints"
        init_from = init_from / f"s0_step{_bc_steps():06d}.pt"
    run_dir = mf.run_dir(
        cfg.arm, args.seed, root=args.run_root, sigma=sigma, tuning=args.tuning, timing=timing
    )
    data_root = p8_adapter.load_paths()["data_root"] if args.data_root is None else args.data_root
    spec = loop.RunSpec(
        cfg=cfg,
        run_seed=args.seed,
        run_dir=run_dir,
        data_root=data_root,
        device=args.device,
        purpose=purpose,
        sigma=sigma,
        max_steps=args.max_steps,
        heldout=None if timing else loop.HELDOUT,
        init_from=init_from,
        allow_dirty=args.allow_dirty,
        rollout_workers=args.rollout_workers,
    )
    print(f"run dir: {run_dir}  purpose: {purpose}  device: {args.device}", flush=True)
    try:
        result = loop.train_run(spec, loop.build_stages(spec))
    except BaseException:
        traceback.print_exc()
        status = run_dir / "status.json"
        if status.is_file():
            st = mf.read_json(status)
            st["state"] = "failed"
            mf.write_json_atomic(status, st)
        return 1
    print(
        f"{result.state}: {result.global_step} steps, wall {result.wall_s:.0f} s, "
        f"{result.steps_timed / max(result.loop_s, 1e-9):.2f} steps/s (loop)",
        flush=True,
    )
    return 0 if result.state == "done" else 2


def _bc_steps() -> int:
    return TrainConfig.load(REPO_ROOT / "configs" / "train" / "bc.yaml").steps


if __name__ == "__main__":
    sys.exit(main())
