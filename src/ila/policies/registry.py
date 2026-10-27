"""Student registry: arm name -> config and builder implementing P8's ``EvalPolicy`` shape.

Phase 2 builds the **real architectures with random initialisation** (user decision
2026-10-27) for the throughput and determinism studies; Phases 4 and 6 add trained-checkpoint
loading on the same modules. Every builder:

* reads the arm's ``configs/train/<arm>.yaml``, the camera config (input 96 px, crop 88 px) and
  the observation contract (proprio width and ``To``, selected by P8 field name in
  :mod:`ila.obs.split`);
* initialises the weights on the CPU under ``torch.manual_seed(seed)`` inside a forked RNG
  (the caller's global RNG is untouched), then moves them to ``device``. The same seed gives
  bit-identical weights on any device.

:func:`build_dummy` is a module-level function, so ``functools.partial(build_dummy, "dp", 0,
"cuda")`` pickles into a worker or server process.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import torch
import yaml
from torch import nn

from ila.policies.bc import BCHeadConfig, BCNet, BCStudent
from ila.policies.dp.policy import DiffusionPolicyNet, DPConfig, DPStudent
from ila.policies.encoder import EncoderConfig

__all__ = [
    "CAMERA_CONFIG",
    "REGISTRY",
    "REPO_ROOT",
    "StudentEntry",
    "build_dummy",
    "build_net",
    "encoder_config",
    "load_train_config",
]

#: Repository root (``src/ila/policies/registry.py`` -> three levels up from ``src``).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Camera config: frame size and crop.
CAMERA_CONFIG: Final[Path] = REPO_ROOT / "configs" / "render" / "camera.yaml"


@dataclass(frozen=True)
class StudentEntry:
    """One registered student.

    Attributes:
        name: Arm name.
        config: Its ``configs/train/<arm>.yaml``.
        kind: Architecture family (``"bc"`` or ``"dp"``).
        obs_variant: Observation-contract variant (``ila.obs.split.VARIANTS``).
    """

    name: str
    config: Path
    kind: Literal["bc", "dp"]
    obs_variant: str = "default"


def _entry(name: str, kind: Literal["bc", "dp"]) -> StudentEntry:
    return StudentEntry(name, REPO_ROOT / "configs" / "train" / f"{name}.yaml", kind)


#: Registered vision students (more arms are added in Phases 4-7).
REGISTRY: Final[Mapping[str, StudentEntry]] = MappingProxyType(
    {"bc": _entry("bc", "bc"), "dp": _entry("dp", "dp")}
)


def load_train_config(arm: str) -> dict[str, Any]:
    """Parse an arm's train YAML.

    Args:
        arm: A key of :data:`REGISTRY`.

    Returns:
        The parsed mapping.

    Raises:
        KeyError: If the arm is not registered.
        ValueError: If the YAML's ``arm`` field differs from ``arm``.
    """
    entry = REGISTRY[arm]
    cfg: dict[str, Any] = yaml.safe_load(entry.config.read_text(encoding="utf-8"))
    if cfg.get("arm") != arm:
        raise ValueError(f"{entry.config} declares arm {cfg.get('arm')!r}, expected {arm!r}")
    return cfg


def encoder_config(train_cfg: Mapping[str, Any], obs_variant: str = "default") -> EncoderConfig:
    """Return the shared encoder config for an arm.

    Args:
        train_cfg: The arm's parsed train YAML.
        obs_variant: Observation-contract variant.

    Returns:
        The encoder config (proprio width and ``To`` from the contract).
    """
    from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

    contract = load_contract(variant=obs_variant)
    camera: Mapping[str, Any] = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))
    return EncoderConfig.from_mappings(
        train_cfg, camera, proprio_dim=contract.proprio_dim, n_obs_steps=contract.n_obs_steps
    )


def _seeded_cpu_init[M: nn.Module](seed: int, make: Callable[[], M]) -> M:
    """Construct ``make()`` on the CPU under ``manual_seed(seed)``, caller's RNG untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return make()


def build_net(arm: str, seed: int) -> BCNet | DiffusionPolicyNet:
    """Build an arm's network with seeded random initialisation, on the CPU.

    Args:
        arm: A key of :data:`REGISTRY`.
        seed: Initialisation seed.

    Returns:
        The network (training mode, CPU).
    """
    entry = REGISTRY[arm]
    cfg = load_train_config(arm)
    enc = encoder_config(cfg, entry.obs_variant)
    if entry.kind == "bc":
        head = BCHeadConfig.from_mapping(cfg)
        return _seeded_cpu_init(seed, lambda: BCNet(enc, head))
    dp_cfg = DPConfig.from_mapping(cfg)
    return _seeded_cpu_init(seed, lambda: DiffusionPolicyNet(enc, dp_cfg))


def build_dummy(
    arm: str,
    seed: int,
    device: str | torch.device = "cpu",
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
) -> BCStudent | DPStudent:
    """Build a randomly initialised student (Phase 2 throughput and determinism studies).

    Args:
        arm: ``"bc"`` or ``"dp"``.
        seed: Initialisation seed; for ``dp`` also the ``run_seed`` of every noise key.
        device: Torch device for inference.
        n_action_steps: ``dp`` only: ``Ta`` (config default 8 when ``None``).
        num_inference_steps: ``dp`` only: ``K`` (config default 10 when ``None``).

    Returns:
        A student in eval mode on ``device``.

    Raises:
        ValueError: If DP-only arguments are given for a BC arm.
    """
    net = build_net(arm, seed)
    if isinstance(net, BCNet):
        if n_action_steps is not None or num_inference_steps is not None:
            raise ValueError(f"{arm!r} is not a diffusion arm; Ta/K do not apply")
        return BCStudent(net, device)
    return DPStudent(
        net,
        run_seed=seed,
        device=device,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
    )
