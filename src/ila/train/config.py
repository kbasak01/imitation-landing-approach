"""Frozen training configs: ``configs/train/*.yaml`` -> dataclasses (P3-D1 §3).

The YAML files are frozen by SHA-256 (P3-D1 §3); this module only reads them. A config with a
``base:`` key (the ablations, ``bc_dagger``) is the base file's mapping with this file's top-level
keys replacing the base's (one level, as the ablation headers state). The SHA-256 recorded for a
run is the SHA-256 of the file named on the command line, of its base, if any, and of any file it
references (``bc_dagger``'s ``dagger: {from: configs/data/demos.yaml#dagger}``).

``bc_matched`` and ``bc_dagger`` parse to one :class:`IterationSchedule` (P3-D1 §3, skeptic B2;
``tests/test_train_dagger.py`` asserts equality); ``bc_dagger`` adds a :class:`DaggerConfig`.

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
    "DaggerConfig",
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


#: The per-iteration loop keys both iterated configs pin (P3-D1 §3, skeptic B2).
_WARM_START: Final[dict[str, bool]] = {"weights": True, "ema": True, "optimizer_state": False}
_SAMPLING: Final[str] = "uniform_over_aggregate_frames"


def _check_loop(path: Path, loop: Mapping[str, Any], steps_per_iteration: int) -> int:
    """Check the shared ``loop`` keys; return the per-iteration warm-up steps.

    Raises:
        ValueError: On a warm start other than "weights and EMA copied, AdamW reset", a
            sampling rule other than uniform over the aggregate, or a per-iteration schedule
            that is not cosine over ``steps_per_iteration`` steps.
    """
    if dict(loop["warm_start"]) != _WARM_START:
        raise ValueError(f"{path}: warm start must copy weights and EMA, reset AdamW")
    if loop["sampling"] != _SAMPLING:
        raise ValueError(f"{path}: unsupported sampling {loop['sampling']!r}")
    per = loop["schedule_per_iteration"]
    if int(per["steps"]) != steps_per_iteration or per["name"] != "cosine":
        raise ValueError(f"{path}: inconsistent per-iteration schedule")
    return int(per["warmup_steps"])


def resolve_reference(path: Path, ref: str) -> tuple[Mapping[str, Any], Path]:
    """Resolve a ``<file>#<key>`` reference of a train config (``bc_dagger.yaml``'s ``dagger``).

    ``<file>`` is relative to the repository root, i.e. two levels above ``configs/train/``.

    Args:
        path: The train config holding the reference.
        ref: ``"configs/data/demos.yaml#dagger"``.

    Returns:
        ``(block, file)``: the referenced mapping and the file read.

    Raises:
        ValueError: On a malformed reference or a missing key.
    """
    file, sep, key = ref.partition("#")
    if not sep or not file or not key:
        raise ValueError(f"{path}: reference {ref!r} is not <file>#<key>")
    target = (path.resolve().parents[2] / file).resolve()
    data: dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8"))
    if key not in data or not isinstance(data[key], Mapping):
        raise ValueError(f"{target}: no mapping {key!r}")
    return data[key], target


@dataclass(frozen=True)
class DaggerConfig:
    """The DAgger-only part of ``bc_dagger.yaml`` (plan §1.6; P3-D1 §2, §3).

    The schedule shared with ``bc_matched`` is :class:`IterationSchedule`; this holds what
    only an on-policy arm has. Iteration ``i`` in ``1 .. n`` flies the EMA weights at the last
    step of iteration ``i - 1`` (iteration 0: the same-seed ``bc`` run's last step) on stream
    indices :meth:`IterationSchedule.added_range` ``(i)`` and executes the teacher's action
    with probability ``beta[i - 1]`` per step.

    Attributes:
        beta: Per-iteration teacher-execution probabilities, iterations ``1 .. n``.
        stream: The ``train`` stream the rollouts fly (``demos``).
        demos_success_filter: Iteration 0's demos are success-filtered (``True``).
        rollouts_success_filter: Rollouts are never filtered (``False``; P3-D1 §2).
        rollout_weights: ``ema_last_step_of_previous_iteration``.
        beta_coin: The coin's generator rule (recorded; implemented in ``ila.data.dagger``).
        executed_action: ``float32_before_step`` (recorded; ``ila.data.collect.executable``).
        inference_path: ``cpu_fp32_single_sample``.
        source: The resolved ``dagger.from`` reference.
    """

    beta: tuple[float, ...]
    stream: str
    demos_success_filter: bool
    rollouts_success_filter: bool
    rollout_weights: str
    beta_coin: str
    executed_action: str
    inference_path: str
    source: str

    def beta_for(self, iteration: int) -> float:
        """Return beta of iteration ``iteration`` (1-based).

        Raises:
            ValueError: Outside ``1 .. len(beta)``.
        """
        if not 1 <= iteration <= len(self.beta):
            raise ValueError(f"no iteration {iteration} (1 .. {len(self.beta)})")
        return self.beta[iteration - 1]


def _bc_matched_schedule(path: Path, raw: Mapping[str, Any]) -> IterationSchedule:
    sched = raw["schedule"]
    it = sched["iterations"]
    if sched["iteration_0"] != "reuse_bc_seed_run" or not it["warm_start"]:
        raise ValueError(f"{path}: unsupported schedule {dict(sched)}")
    steps = int(it["steps_per_iteration"])
    warmup = _check_loop(path, sched["loop"], steps)
    return IterationSchedule(
        n=int(it["n"]),
        add_per_iteration=int(it["add_episodes_per_iteration"]),
        first_added=int(it["first_added"]),
        steps_per_iteration=steps,
        warmup_steps=warmup,
        labelled_episodes_total=int(raw["data"]["first_n"]),
    )


def _dagger_schedule(
    path: Path, raw: Mapping[str, Any]
) -> tuple[IterationSchedule, DaggerConfig, Path]:
    """Parse ``bc_dagger.yaml``'s ``dagger``, ``loop`` and filter keys (module docstring).

    Raises:
        ValueError: On any key outside what P3-D1 §2-§3 pin.
    """
    ref = str(raw["dagger"]["from"])
    dg, source = resolve_reference(path, ref)
    loop = raw["loop"]
    eps = loop["episodes"]
    if raw["iteration_0"] != "reuse_bc_seed_run" or not dg["warm_start"]:
        raise ValueError(f"{path}: iteration 0 must be the same-seed bc run, warm-started")
    if (str(eps["stream"]), int(eps["first_index"]), int(eps["per_iteration"])) != (
        str(dg["stream"]),
        int(dg["first_index"]),
        int(dg["episodes_per_iteration"]),
    ):
        raise ValueError(f"{path}: loop.episodes {dict(eps)} disagree with {ref}")
    steps = int(dg["steps_per_iteration"])
    warmup = _check_loop(path, loop, steps)
    n = int(dg["iterations"])
    beta = tuple(float(b) for b in dg["beta"])
    if len(beta) != n or not all(0.0 <= b <= 1.0 for b in beta):
        raise ValueError(f"{path}: beta {beta} for {n} iterations")
    sf = raw["success_filter"]
    if not isinstance(sf, Mapping) or set(sf) != {"demos", "rollouts"}:
        raise ValueError(f"{path}: success_filter must be {{demos, rollouts}}, got {sf!r}")
    if sf["rollouts"] is not False or not isinstance(sf["demos"], bool):
        raise ValueError(f"{path}: DAgger rollouts are never filtered (P3-D1 §2)")
    if loop["rollout_weights"] != "ema_last_step_of_previous_iteration":
        raise ValueError(f"{path}: unsupported rollout weights {loop['rollout_weights']!r}")
    if loop["inference_path"] != "cpu_fp32_single_sample":
        raise ValueError(f"{path}: rollouts fly the CPU fp32 single-sample path")
    schedule = IterationSchedule(
        n=n,
        add_per_iteration=int(dg["episodes_per_iteration"]),
        first_added=int(dg["first_index"]),
        steps_per_iteration=steps,
        warmup_steps=warmup,
        labelled_episodes_total=int(raw["labelled_episodes_total"]),
    )
    implied = schedule.first_added + schedule.n * schedule.add_per_iteration
    if implied != schedule.labelled_episodes_total:
        raise ValueError(f"{path}: {implied} labelled episodes, config says "
                         f"{schedule.labelled_episodes_total}")  # fmt: skip
    if int(raw["data"]["first_n"]) != schedule.first_added:
        raise ValueError(f"{path}: iteration 0 holds {raw['data']['first_n']} demos, "
                         f"rollouts start at {schedule.first_added}")  # fmt: skip
    dagger = DaggerConfig(
        beta=beta,
        stream=str(dg["stream"]),
        demos_success_filter=bool(sf["demos"]),
        rollouts_success_filter=False,
        rollout_weights=str(loop["rollout_weights"]),
        beta_coin=str(loop["beta_coin"]),
        executed_action=str(loop["executed_action"]),
        inference_path=str(loop["inference_path"]),
        source=ref,
    )
    return schedule, dagger, source


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
        iterations: The iterated schedule (``bc_matched``, ``bc_dagger``: one schedule,
            P3-D1 §3), else ``None``.
        dart_sigma_rule: ``"selected_on_select"`` for ``bc_dart``, else ``None``.
        dagger: ``bc_dagger``'s on-policy settings, else ``None``. ``success_filter`` is then
            the demos' filter (``success_filter.demos``); rollouts are never filtered.
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
    dagger: DaggerConfig | None = None

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
        iterations = None
        dagger = None
        if "dagger" in raw:
            iterations, dagger, source = _dagger_schedule(path, raw)
            files.append(source)
            sf = dagger.demos_success_filter
        else:
            sf = raw["success_filter"]
            if not isinstance(sf, bool):
                raise ValueError(f"{path}: success_filter {sf!r} (per-source: bc_dagger only)")
            if raw.get("schedule") is not None:
                iterations = _bc_matched_schedule(path, raw)
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
            dagger=dagger,
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
