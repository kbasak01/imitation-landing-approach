"""Frozen training configs: ``configs/train/*.yaml`` -> dataclasses (P3-D1 §3).

The YAML files are frozen by SHA-256 (P3-D1 §3); this module only reads them. A config with a
``base:`` key (the ablations, ``bc_dagger``) is the base file's mapping with this file's top-level
keys replacing the base's (one level, as the ablation headers state). The SHA-256 recorded for a
run is the SHA-256 of the file named on the command line (and of its base, if any).

Steps are optimiser steps; learning rates are per step; nothing here carries physical units.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import yaml

__all__ = [
    "DART_SIGMAS",
    "EMAConfig",
    "IterationSchedule",
    "OptimConfig",
    "TrainConfig",
    "load_train_yaml",
    "sha256_file",
]

#: The DART sigma grid (normalised action units; ``configs/data/demos.yaml``, P3-D1 §4).
DART_SIGMAS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)

#: AdamW betas / eps: the configs do not set them, so torch's defaults apply (recorded).
ADAMW_BETAS: Final[tuple[float, float]] = (0.9, 0.999)
ADAMW_EPS: Final[float] = 1e-8


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_train_yaml(path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Parse a train YAML, resolving one ``base:`` level.

    Args:
        path: ``configs/train/<arm>.yaml`` (or an ablation file).

    Returns:
        ``(mapping, files)``: the resolved mapping (without ``base``) and the files read, the
        named file first.

    Raises:
        ValueError: If a base itself has a base.
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = [path]
    base = raw.pop("base", None)
    if base is None:
        return raw, files
    base_path = (path.parent / str(base)).resolve()
    parent: dict[str, Any] = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if "base" in parent:
        raise ValueError(f"{base_path}: nested base configs are not supported")
    return {**parent, **raw}, [path, base_path]


@dataclass(frozen=True)
class EMAConfig:
    """EMA of the weights (evaluated weights; P3-D1 §4).

    ``constant``: ``ema = d ema + (1 - d) w`` with fixed ``d`` (BC family: 0.999). ``power``:
    diffusion_policy's ``EMAModel`` warm-up, ``d_n = clamp(1 - (1 + (n - 1) / inv_gamma)^-power,
    min_value, max_value)`` for the ``n``-th update (``d = 0`` for ``n <= 1``, i.e. a copy),
    ``update_after_step = 0``, ``inv_gamma = 1`` (``dp``: power 0.75, max 0.9999).

    Attributes:
        kind: ``"constant"`` or ``"power"``.
        decay: The constant decay.
        power: Warm-up power.
        max_value: Upper clamp.
        min_value: Lower clamp.
        inv_gamma: Warm-up scale, updates.
    """

    kind: Literal["constant", "power"]
    decay: float = 0.0
    power: float = 0.0
    max_value: float = 1.0
    min_value: float = 0.0
    inv_gamma: float = 1.0

    @classmethod
    def from_value(cls, value: Any) -> EMAConfig:
        """Build from the YAML ``ema`` value (a float, or ``{power, max_value}``)."""
        if isinstance(value, Mapping):
            power, top = float(value["power"]), float(value["max_value"])
            return cls(kind="power", power=power, max_value=top)
        d = float(value)
        if not 0.0 <= d < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {d}")
        return cls(kind="constant", decay=d)

    def decay_at(self, n_update: int) -> float:
        """Return the decay used by the ``n_update``-th EMA update (1-based).

        Args:
            n_update: How many updates have been made including this one (>= 1).

        Returns:
            The decay in ``[0, 1)``.
        """
        if self.kind == "constant":
            return self.decay
        # diffusion_policy's EMAModel.step() calls get_decay(optimization_step) with
        # optimization_step = n_update - 1, and step = max(0, optimization_step - 0 - 1).
        step = max(0, n_update - 2)
        if step <= 0:
            return 0.0
        value = 1.0 - float((1.0 + step / self.inv_gamma) ** -self.power)
        return max(self.min_value, min(value, self.max_value))


@dataclass(frozen=True)
class OptimConfig:
    """AdamW with cosine decay after linear warm-up (``optim`` key).

    Attributes:
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay.
        warmup_steps: Linear warm-up steps (from 0).
        schedule: ``"cosine"`` only.
    """

    lr: float
    weight_decay: float
    warmup_steps: int
    schedule: str = "cosine"

    @classmethod
    def from_mapping(cls, optim: Mapping[str, Any]) -> OptimConfig:
        """Build from the YAML ``optim`` mapping.

        Raises:
            ValueError: If the optimiser is not AdamW or the schedule not cosine.
        """
        if optim["name"] != "adamw" or optim["schedule"] != "cosine":
            raise ValueError(f"unsupported optimiser block {dict(optim)}")
        return cls(
            lr=float(optim["lr"]),
            weight_decay=float(optim["weight_decay"]),
            warmup_steps=int(optim["warmup_steps"]),
        )


@dataclass(frozen=True)
class IterationSchedule:
    """The shared ``bc_dagger`` / ``bc_matched`` loop (P3-D1 §3, skeptic B2).

    Iteration 0 is the same-seed ``bc`` run (its last-step weights and EMA). Iteration ``i`` in
    ``1 .. n`` adds ``add_per_iteration`` episodes starting at stream index
    ``first_added + add_per_iteration (i - 1)`` and trains ``steps_per_iteration`` warm-started
    steps (weights and EMA copied, AdamW reset, cosine with ``warmup_steps``), sampling frames
    uniformly from the aggregate.

    Attributes:
        n: Iterations after iteration 0 (5).
        add_per_iteration: Episodes added per iteration (200).
        first_added: Stream index of iteration 1's first episode (1000).
        steps_per_iteration: Optimiser steps per iteration (10 000).
        warmup_steps: Warm-up per iteration (500).
        labelled_episodes_total: Asserted labelled-episode count after the last iteration.
    """

    n: int
    add_per_iteration: int
    first_added: int
    steps_per_iteration: int
    warmup_steps: int
    labelled_episodes_total: int

    def added_range(self, iteration: int) -> range:
        """Return the stream indices iteration ``iteration`` (1-based) adds.

        Raises:
            ValueError: Outside ``1 .. n``.
        """
        if not 1 <= iteration <= self.n:
            raise ValueError(f"no iteration {iteration} (1 .. {self.n})")
        first = self.first_added + self.add_per_iteration * (iteration - 1)
        return range(first, first + self.add_per_iteration)


@dataclass(frozen=True)
class TrainConfig:
    """One arm's frozen training config.

    Attributes:
        arm: Arm name.
        kind: ``"bc"``, ``"dp"`` or ``"state"``.
        path: The YAML named on the command line.
        files_sha256: SHA-256 of every file read (the named file first, then its base).
        raw: The resolved mapping (read-only), stored in every checkpoint.
        collection: Demo collection (``ppo_2000``; ``dart`` is resolved with sigma).
        first_n: Episodes taken by stream index **before** the success filter.
        success_filter: Train only on ``success`` episodes.
        obs_variant: Observation-contract variant.
        steps: Optimiser steps (iteration 0 for the iterated arms).
        batch_size: Samples per step.
        precision: ``"bf16"`` (autocast) or ``"fp32"``.
        ema: EMA config.
        optim: Optimiser config.
        checkpoint_every: Checkpoint cadence, steps.
        seeds: The frozen seeds.
        iterations: The iterated schedule (``bc_matched``), else ``None``.
        dart_sigma_rule: ``"selected_on_select"`` for ``bc_dart``, else ``None``.
    """

    arm: str
    kind: Literal["bc", "dp", "state"]
    path: Path
    files_sha256: tuple[tuple[str, str], ...]
    raw: Mapping[str, Any]
    collection: str
    first_n: int
    success_filter: bool
    obs_variant: str
    steps: int
    batch_size: int
    precision: Literal["bf16", "fp32"]
    ema: EMAConfig
    optim: OptimConfig
    checkpoint_every: int
    seeds: tuple[int, ...]
    iterations: IterationSchedule | None = None
    dart_sigma_rule: str | None = None

    @property
    def sha256(self) -> str:
        """SHA-256 of the named config file."""
        return self.files_sha256[0][1]

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Load and validate a train config.

        Args:
            path: ``configs/train/<arm>.yaml``.

        Returns:
            The config.

        Raises:
            ValueError: On an unsupported key value.
        """
        raw, files = load_train_yaml(path)
        data = raw["data"]
        if "horizons" in raw:
            kind: Literal["bc", "dp", "state"] = "dp"
        elif data.get("input") == "teacher_obs":
            kind = "state"
        else:
            kind = "bc"
        if (kind == "state") != (str(raw["arm"]) == "bc_state"):
            raise ValueError(f"{path}: only bc_state reads teacher_obs (kind {kind!r})")
        precision = str(raw["precision"])
        if precision not in ("bf16", "fp32"):
            raise ValueError(f"unsupported precision {precision!r}")
        sf = raw["success_filter"]
        if not isinstance(sf, bool):
            raise ValueError(f"{path}: success_filter {sf!r} (per-source filters are Phase 5)")
        iterations = None
        sched = raw.get("schedule")
        if sched is not None:
            it = sched["iterations"]
            per = sched["loop"]["schedule_per_iteration"]
            warm = sched["loop"]["warm_start"]
            if sched["iteration_0"] != "reuse_bc_seed_run" or not it["warm_start"]:
                raise ValueError(f"{path}: unsupported schedule {dict(sched)}")
            if warm != {"weights": True, "ema": True, "optimizer_state": False}:
                raise ValueError(f"{path}: warm start must copy weights and EMA, reset AdamW")
            if sched["loop"]["sampling"] != "uniform_over_aggregate_frames":
                raise ValueError(f"{path}: unsupported sampling {sched['loop']['sampling']!r}")
            if int(per["steps"]) != int(it["steps_per_iteration"]) or per["name"] != "cosine":
                raise ValueError(f"{path}: inconsistent per-iteration schedule")
            iterations = IterationSchedule(
                n=int(it["n"]),
                add_per_iteration=int(it["add_episodes_per_iteration"]),
                first_added=int(it["first_added"]),
                steps_per_iteration=int(it["steps_per_iteration"]),
                warmup_steps=int(per["warmup_steps"]),
                labelled_episodes_total=int(data["first_n"]),
            )
        sigma = data.get("sigma")
        return cls(
            arm=str(raw["arm"]),
            kind=kind,
            path=path,
            files_sha256=tuple((f.as_posix(), sha256_file(f)) for f in files),
            raw=MappingProxyType(raw),
            collection=str(data["collection"]),
            first_n=int(data["first_n"]) if iterations is None else iterations.first_added,
            success_filter=sf,
            obs_variant=str(raw.get("obs_variant", "default")),
            steps=int(raw["steps"]),
            batch_size=int(raw["batch_size"]),
            precision="bf16" if precision == "bf16" else "fp32",
            ema=EMAConfig.from_value(raw["ema"]),
            optim=OptimConfig.from_mapping(raw["optim"]),
            checkpoint_every=int(raw["checkpoint_every"]),
            seeds=tuple(int(s) for s in raw["seeds"]),
            iterations=iterations,
            dart_sigma_rule=None if sigma is None else str(sigma),
        )

    def collection_for(self, sigma: float | None) -> str:
        """Return the store name: ``dart_<sigma>`` for ``bc_dart``, else ``collection``.

        Raises:
            ValueError: If a DART config gets no or an off-grid sigma, or another arm gets one.
        """
        if self.collection != "dart":
            if sigma is not None:
                raise ValueError(f"{self.arm} takes no DART sigma")
            return self.collection
        if sigma is None or round(float(sigma), 6) not in DART_SIGMAS:
            raise ValueError(f"{self.arm}: sigma must be one of {DART_SIGMAS}, got {sigma}")
        return f"dart_{float(sigma):.2f}"
