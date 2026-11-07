"""Tiny Phase 9 fixtures: small real-architecture networks, settings, inputs (CPU, no PyBullet).

``tiny_arms()`` builds ``bc_dagger`` and ``dp`` networks with the shared ResNet-18 encoder (its
architecture is fixed) but a small head and a two-level U-Net (``down_dims`` 8, 16; step
embedding 8), random init, eval mode, wrapped as :class:`ila.deploy.export.LoadedArm` with a
synthetic checkpoint record. ``tiny_settings(tmp)`` is ``configs/deploy/bench.yaml`` with its
output roots moved under ``tmp``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ila.deploy.export import LoadedArm
from ila.deploy.inputs import SelectInputs
from ila.deploy.settings import BENCH_YAML, BenchSettings, from_mapping
from ila.eval import arms as ev_arms
from ila.policies.registry import build_net_from_config, load_train_config


def _ckpt(arm: str) -> ev_arms.Checkpoint:
    return ev_arms.Checkpoint(
        arm=arm,
        seed=0,
        kind="checkpoint",
        path=Path("/nonexistent") / f"{arm}.pt",
        recorded_path=f"/nonexistent/{arm}.pt",
        sha256="0" * 64,
        teacher="T_ppo",
        selection_json=f"results/test/{arm}/selection.json",
        selection_json_sha256="1" * 64,
        global_step=0,
        select_success=0,
        select_n=150,
        select_mean_steps=60.0,
        selection_wall_clock_s=1.0,
        selection_workers=1,
        selection_steps=1.0,
        obs_variant="default",
    )


def tiny_arms(seed: int = 0) -> dict[str, LoadedArm]:
    """Small ``bc_dagger`` and ``dp`` networks (module docstring)."""
    bc_cfg = copy.deepcopy(load_train_config("bc_dagger"))
    bc_cfg["head"]["hidden"] = [16]
    dp_cfg = copy.deepcopy(load_train_config("dp"))
    dp_cfg["unet"] = {
        "down_dims": [8, 16],
        "kernel_size": 3,
        "diffusion_step_embed_dim": 8,
        "cond": "film",
    }
    out = {}
    for arm, kind, cfg in (("bc_dagger", "bc", bc_cfg), ("dp", "dp", dp_cfg)):
        net = build_net_from_config(kind, cfg, seed)  # type: ignore[arg-type]
        net.eval()
        out[arm] = LoadedArm(arm, seed, _ckpt(arm), net, net.encoder.cfg, "default")  # type: ignore[arg-type]
    return out


def tiny_settings(tmp: Path, **overrides: Any) -> BenchSettings:
    """``bench.yaml`` with ``out_dir`` and ``results_dir`` under ``tmp`` (and any overrides)."""
    raw = yaml.safe_load(BENCH_YAML.read_text("utf-8"))
    raw["export"]["out_dir"] = str(tmp / "artifacts")
    raw["results_dir"] = str(tmp / "results")
    for dotted, value in overrides.items():
        node = raw
        *head, last = dotted.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    return from_mapping(raw)


def tiny_inputs(n: int = 4, seed: int = 0) -> SelectInputs:
    """Random windows with the contract's shapes (not rendered frames)."""
    g = np.random.default_rng(seed)
    return SelectInputs(
        images=g.integers(0, 256, (n, 2, 96, 96, 3), dtype=np.uint8),
        proprio=g.standard_normal((n, 2, 13)).astype(np.float32),
        episode=np.arange(n, dtype=np.int64),
        step=np.zeros(n, np.int64),
        episode_seed=np.arange(n, dtype=np.int64),
    )
