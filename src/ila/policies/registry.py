"""Student registry: arm name -> config and builder implementing P8's ``EvalPolicy`` shape.

Phase 2 builds the **real architectures with random initialisation** (user decision
2026-10-27) for the throughput and determinism studies; Phase 4 adds ``bc_state``
(:mod:`ila.policies.state_mlp`, P8's 25-d observation, privileged by design and never a vision
row), the ``bc_matched`` / ``bc_dart`` entries (the ``bc`` architecture; Phase 5 adds
``bc_dagger``, ``base: bc.yaml``) and trained-checkpoint
loading (:func:`load_student`) on the same modules. Phase 6 adds ``R1_lerobot_dp`` (kind
``onnx_dp``, tag ``reference_external``): LeRobot's DiffusionPolicy trained in
``.venv-lerobot``; ``ila`` has no network for it and flies its ONNX export through
:class:`ila.policies.onnx_dp.OnnxDPStudent` (:func:`ila.policies.onnx_dp.load_onnx_student`).
Every builder of an ``ila`` network:

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

#: Architecture families: vision BC, Diffusion Policy, the 25-d state MLP (``bc_state``), and an
#: external diffusion policy flown from its ONNX export (``R1_lerobot_dp``; no ``ila`` network).
type Kind = Literal["bc", "dp", "state", "onnx_dp"]

#: Any network an arm builds.
type Net = BCNet | DiffusionPolicyNet | StateMLP

#: Any student wrapper built from an ``ila`` checkpoint (P8 ``EvalPolicy`` shape). An ``onnx_dp``
#: arm's student is :class:`ila.policies.onnx_dp.OnnxDPStudent`, loaded from its export.
type Student = BCStudent | DPStudent | StateMLPStudent


@dataclass(frozen=True)
class StudentEntry:
    """One registered student.

    Attributes:
        name: Arm name.
        config: Its ``configs/train/<arm>.yaml``.
        kind: Architecture family (``"bc"``, ``"dp"``, ``"state"`` or ``"onnx_dp"``).
        obs_variant: Observation-contract variant (``ila.obs.split.VARIANTS``).
        tag: Row tag (``core``; ``reference_external`` for R1, P3-D1 §3, never mixed into core
            tables).
    """

    name: str
    config: Path
    kind: Kind
    obs_variant: str = "default"
    tag: str = "core"

    @property
    def is_vision(self) -> bool:
        """Whether the arm is a vision student (``act_vision``); ``bc_state`` is not."""
        return self.kind != "state"

    @property
    def has_ila_network(self) -> bool:
        """Whether ``ila`` builds the network (:func:`build_net`); ``onnx_dp`` arms do not."""
        return self.kind != "onnx_dp"


def _entry(name: str, kind: Kind) -> StudentEntry:
    return StudentEntry(name, REPO_ROOT / "configs" / "train" / f"{name}.yaml", kind)


#: Registered students (more arms are added in Phases 5-7). ``bc_state`` is the only
#: non-vision entry (P8's 25-d observation, privileged by design; P3-D1 §3).
REGISTRY: Final[Mapping[str, StudentEntry]] = MappingProxyType(
    {
        "bc": _entry("bc", "bc"),
        "bc_matched": _entry("bc_matched", "bc"),
        "bc_dagger": _entry("bc_dagger", "bc"),
        "bc_dart": _entry("bc_dart", "bc"),
        "bc_state": _entry("bc_state", "state"),
        "dp": _entry("dp", "dp"),
        # P3-D1 §3 reference row (P6-D3): LeRobot 0.6.1 DiffusionPolicy, flown from ONNX.
        "R1_lerobot_dp": StudentEntry(
            "R1_lerobot_dp",
            REPO_ROOT / "configs" / "train" / "r1_lerobot_dp.yaml",
            "onnx_dp",
            tag="reference_external",
        ),
    }
)


def load_train_config(arm: str) -> dict[str, Any]:
    """Parse an arm's train YAML.

    Args:
        arm: A key of :data:`REGISTRY`.

    Returns:
        The parsed mapping, one ``base:`` level resolved.

    Raises:
        KeyError: If the arm is not registered.
        ValueError: If the YAML's ``arm`` field differs from ``arm``.
    """
    entry = REGISTRY[arm]
    cfg: dict[str, Any] = yaml.safe_load(entry.config.read_text(encoding="utf-8"))
    if cfg.get("arm") != arm:
        raise ValueError(f"{entry.config} declares arm {cfg.get('arm')!r}, expected {arm!r}")
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    # One level, this file's top-level keys replacing the base's (``ila.train.config``'s rule;
    # ``bc_dagger.yaml``: ``base: bc.yaml``, the identical architecture).
    parent: dict[str, Any] = yaml.safe_load(
        (entry.config.parent / str(base)).read_text(encoding="utf-8")
    )
    if "base" in parent:
        raise ValueError(f"{entry.config}: nested base configs are not supported")
    return {**parent, **cfg}


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

    Raises:
        ValueError: For ``onnx_dp`` (no ``ila`` network: load the export instead).
    """
    if kind == "onnx_dp":
        raise ValueError("an onnx_dp arm has no ila network; use onnx_dp.load_onnx_student")
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
    if not entry.has_ila_network:
        raise ValueError(f"{arm!r} has no ila network; use onnx_dp.load_onnx_student")
    return build_net_from_config(entry.kind, load_train_config(arm), seed, entry.obs_variant)


def _wrap(
    net: Net,
    run_seed: int,
    device: str | torch.device,
    n_action_steps: int | None,
    num_inference_steps: int | None,
    arm: str,
    temporal_ensembling: bool = False,
) -> Student:
    if isinstance(net, DiffusionPolicyNet):
        return DPStudent(
            net,
            run_seed=run_seed,
            device=device,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
            temporal_ensembling=temporal_ensembling,
        )
    if n_action_steps is not None or num_inference_steps is not None or temporal_ensembling:
        raise ValueError(f"{arm!r} is not a diffusion arm; Ta/K/ensembling do not apply")
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
    temporal_ensembling: bool = False,
) -> BCStudent | DPStudent:
    """Build a randomly initialised vision student (Phase 2 throughput and determinism studies).

    Args:
        arm: A vision key of :data:`REGISTRY` (``bc_state`` is refused: it is not a vision
            student; build it with :func:`build_net` and ``StateMLPStudent``).
        seed: Initialisation seed; for ``dp`` also the ``run_seed`` of every noise key.
        device: Torch device for inference.
        n_action_steps: ``dp`` only: ``Ta`` (config default 8 when ``None``).
        num_inference_steps: ``dp`` only: ``K`` (config default 10 when ``None``).
        temporal_ensembling: ``dp`` only: replan every step and average (P6-D1).

    Returns:
        A student in eval mode on ``device``.

    Raises:
        ValueError: If DP-only arguments are given for a non-diffusion arm, or the arm is not
            a vision student with an ``ila`` network.
    """
    if not REGISTRY[arm].is_vision:
        raise ValueError(f"{arm!r} is not a vision student")
    if not REGISTRY[arm].has_ila_network:
        raise ValueError(f"{arm!r} has no ila network to initialise; load its ONNX export")
    student = _wrap(
        build_net(arm, seed),
        seed,
        device,
        n_action_steps,
        num_inference_steps,
        arm,
        temporal_ensembling,
    )
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
    temporal_ensembling: bool = False,
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
        temporal_ensembling: ``dp`` only: replan every step and average (P6-D1).

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
        net,
        int(ckpt["run_seed"]),
        device,
        n_action_steps,
        num_inference_steps,
        str(ckpt["arm"]),
        temporal_ensembling,
    )
