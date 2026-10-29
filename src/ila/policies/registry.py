"""Student registry: arm name -> config and builder implementing P8's ``EvalPolicy`` shape.

Phase 2 builds the **real architectures with random initialisation** (user decision
2026-10-27) for the throughput and determinism studies; Phase 4 adds ``bc_state``
(:mod:`ila.policies.state_mlp`, P8's 25-d observation, privileged by design and never a vision
row), the ``bc_matched`` / ``bc_dart`` entries (the ``bc`` architecture) and trained-checkpoint
loading (:func:`load_student`) on the same modules. Every builder:

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
from ila.policies.state_mlp import StateMLP, StateMLPConfig, StateMLPStudent

__all__ = [
    "CAMERA_CONFIG",
    "CHECKPOINT_FORMAT",
    "REGISTRY",
    "REPO_ROOT",
    "Kind",
    "Net",
    "Student",
    "StudentEntry",
    "build_dummy",
    "build_net",
    "build_net_from_config",
    "encoder_config",
    "load_student",
    "load_train_config",
    "read_checkpoint",
]

#: Repository root (``src/ila/policies/registry.py`` -> three levels up from ``src``).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Camera config: frame size and crop.
CAMERA_CONFIG: Final[Path] = REPO_ROOT / "configs" / "render" / "camera.yaml"

#: Format tag of every training checkpoint (written by ``ila.train.loop``).
CHECKPOINT_FORMAT: Final[str] = "ila-ckpt/1"

#: Architecture families: vision BC, Diffusion Policy, and the 25-d state MLP (``bc_state``).
type Kind = Literal["bc", "dp", "state"]

#: Any network an arm builds.
type Net = BCNet | DiffusionPolicyNet | StateMLP

#: Any student wrapper (P8 ``EvalPolicy`` shape).
type Student = BCStudent | DPStudent | StateMLPStudent


@dataclass(frozen=True)
class StudentEntry:
    """One registered student.

    Attributes:
        name: Arm name.
        config: Its ``configs/train/<arm>.yaml``.
        kind: Architecture family (``"bc"``, ``"dp"`` or ``"state"``).
        obs_variant: Observation-contract variant (``ila.obs.split.VARIANTS``).
    """

    name: str
    config: Path
    kind: Kind
    obs_variant: str = "default"

    @property
    def is_vision(self) -> bool:
        """Whether the arm is a vision student (``act_vision``); ``bc_state`` is not."""
        return self.kind != "state"


def _entry(name: str, kind: Kind) -> StudentEntry:
    return StudentEntry(name, REPO_ROOT / "configs" / "train" / f"{name}.yaml", kind)


#: Registered students (more arms are added in Phases 5-7). ``bc_state`` is the only
#: non-vision entry (P8's 25-d observation, privileged by design; P3-D1 §3).
REGISTRY: Final[Mapping[str, StudentEntry]] = MappingProxyType(
    {
        "bc": _entry("bc", "bc"),
        "bc_matched": _entry("bc_matched", "bc"),
        "bc_dart": _entry("bc_dart", "bc"),
        "bc_state": _entry("bc_state", "state"),
        "dp": _entry("dp", "dp"),
    }
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


def build_net_from_config(
    kind: Kind, train_cfg: Mapping[str, Any], seed: int, obs_variant: str = "default"
) -> Net:
    """Build a network from a parsed train config with seeded random init, on the CPU.

    Args:
        kind: Architecture family.
        train_cfg: The arm's parsed (and ``base``-resolved) train YAML.
        seed: Initialisation seed.
        obs_variant: Observation-contract variant (vision arms: proprio width).

    Returns:
        The network (training mode, CPU).
    """
    if kind == "state":
        from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

        state_cfg = StateMLPConfig.from_mapping(train_cfg, load_contract().obs_dim)
        return _seeded_cpu_init(seed, lambda: StateMLP(state_cfg))
    enc = encoder_config(train_cfg, obs_variant)
    if kind == "bc":
        head = BCHeadConfig.from_mapping(train_cfg)
        return _seeded_cpu_init(seed, lambda: BCNet(enc, head))
    dp_cfg = DPConfig.from_mapping(train_cfg)
    return _seeded_cpu_init(seed, lambda: DiffusionPolicyNet(enc, dp_cfg))


def build_net(arm: str, seed: int) -> Net:
    """Build an arm's network with seeded random initialisation, on the CPU.

    Args:
        arm: A key of :data:`REGISTRY`.
        seed: Initialisation seed.

    Returns:
        The network (training mode, CPU).
    """
    entry = REGISTRY[arm]
    return build_net_from_config(entry.kind, load_train_config(arm), seed, entry.obs_variant)


def _wrap(
    net: Net,
    run_seed: int,
    device: str | torch.device,
    n_action_steps: int | None,
    num_inference_steps: int | None,
    arm: str,
) -> Student:
    if isinstance(net, DiffusionPolicyNet):
        return DPStudent(
            net,
            run_seed=run_seed,
            device=device,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
        )
    if n_action_steps is not None or num_inference_steps is not None:
        raise ValueError(f"{arm!r} is not a diffusion arm; Ta/K do not apply")
    if isinstance(net, StateMLP):
        return StateMLPStudent(net, device)
    return BCStudent(net, device)


def build_dummy(
    arm: str,
    seed: int,
    device: str | torch.device = "cpu",
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
) -> BCStudent | DPStudent:
    """Build a randomly initialised vision student (Phase 2 throughput and determinism studies).

    Args:
        arm: A vision key of :data:`REGISTRY` (``bc_state`` is refused: it is not a vision
            student; build it with :func:`build_net` and ``StateMLPStudent``).
        seed: Initialisation seed; for ``dp`` also the ``run_seed`` of every noise key.
        device: Torch device for inference.
        n_action_steps: ``dp`` only: ``Ta`` (config default 8 when ``None``).
        num_inference_steps: ``dp`` only: ``K`` (config default 10 when ``None``).

    Returns:
        A student in eval mode on ``device``.

    Raises:
        ValueError: If DP-only arguments are given for a non-diffusion arm, or the arm is not
            a vision student.
    """
    if not REGISTRY[arm].is_vision:
        raise ValueError(f"{arm!r} is not a vision student")
    student = _wrap(build_net(arm, seed), seed, device, n_action_steps, num_inference_steps, arm)
    assert isinstance(student, BCStudent | DPStudent)
    return student


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Load a training checkpoint onto the CPU and check its format tag.

    Args:
        path: A ``.pt`` file written by ``ila.train.loop``.

    Returns:
        The checkpoint mapping (keys ``format``, ``arm``, ``kind``, ``obs_variant``,
        ``config``, ``run_seed``, ``step``, ``model`` and/or ``ema``, ...).

    Raises:
        ValueError: If the format tag is not :data:`CHECKPOINT_FORMAT`.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not an {CHECKPOINT_FORMAT} checkpoint")
    return ckpt


def load_student(
    path: Path | str,
    device: str | torch.device = "cpu",
    *,
    weights: Literal["ema", "raw"] = "ema",
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
) -> Student:
    """Build a trained student from a checkpoint (EMA weights by default: always evaluated).

    The network is rebuilt from the config stored in the checkpoint (not the file on disk),
    so a checkpoint flies the architecture it was trained with.

    Args:
        path: The checkpoint.
        device: Torch device for inference (selection: ``"cpu"``, the fp32 single-sample path).
        weights: ``"ema"`` (default; P3-D1 §4) or ``"raw"`` (diagnostics only).
        n_action_steps: ``dp`` only: ``Ta``.
        num_inference_steps: ``dp`` only: ``K``.

    Returns:
        The student in eval mode on ``device``.

    Raises:
        KeyError: If the checkpoint lacks the requested weights.
    """
    ckpt = read_checkpoint(Path(path))
    if weights not in ckpt or ckpt[weights] is None:
        raise KeyError(f"{path}: no {weights!r} weights")
    net = build_net_from_config(
        ckpt["kind"], ckpt["config"], int(ckpt["run_seed"]), str(ckpt["obs_variant"])
    )
    net.load_state_dict(ckpt[weights])
    return _wrap(
        net, int(ckpt["run_seed"]), device, n_action_steps, num_inference_steps, str(ckpt["arm"])
    )
