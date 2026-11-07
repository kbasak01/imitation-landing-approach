"""Modular student: image -> privileged-block estimate -> frozen teacher (Phase 7; P3-D1 §3).

``modular`` = :class:`ModularEstimator` (the shared vision encoder of every vision arm,
:class:`ila.policies.encoder.VisionEncoder`, with the same ``EncoderConfig`` as ``bc`` -- ResNet-18
GroupNorm trunk, 32-keypoint spatial softmax, proprio MLP 2 x 64, ``To = 2``, crop 96 -> 88 px
random at train / centre at eval -- then an MLP head 2 x 256 (ReLU) -> 11) followed by P8's frozen
``T_ppo`` (seed 0, ``final/``) MLP with P8's frozen ``VecNormalize`` statistics
(``training=False``), deterministic, built through :func:`ila.p8_adapter.teacher`.

Estimate. The head regresses P8's privileged block in contract order
(``configs/obs/student.yaml`` ``privileged``; 11-d):

* ``rel_position`` (3) m model scale, drone-yaw frame (full scale x 25);
* ``rel_velocity`` (3) m/s model scale, drone-yaw frame (full scale x 5: lengths x 25 over
  times x 5);
* ``deck_normal`` (3) unit vector, drone-yaw frame (dimensionless);
* ``rel_tilt`` (1) rad;
* ``height`` (1) m model scale along the deck normal (full scale x 25).

The network output is **normalised**: ``z = (priv - target_mean) / target_std`` with the
per-dimension mean and std of ``priv`` over the training frames (the success-filtered first
1 000 ``ppo_2000`` episodes), held in the buffers ``target_mean`` / ``target_std`` and therefore
saved in every checkpoint's ``model`` and ``ema`` state dicts. Training minimises the Huber loss
(delta = 1.0, in normalised units) between ``z`` and the normalised target (:func:`estimator_loss`).

Clock path (P3-D1 §3, leak RISK 2; declared clock exception 2). The estimator's ``forward``
takes pixels and the 13-d default proprio only; it has **no** clock argument (a test inspects the
signature). P8's ``time_fraction`` reaches the frozen teacher only: :meth:`ModularStudent.
act_modular` receives a :class:`ila.obs.split.ModularInput` (built only in ``ila/obs/split.py``),
gives the estimator ``mi.student_input.images`` and ``mi.student_input.proprio``, de-normalises
the estimate, and assembles the teacher's 25-d observation with
:func:`ila.obs.split.assemble_teacher_obs` from the proprio and ``mi.time_fraction`` at window
position ``To - 1`` (step t) and the estimate, in P8's field order by name. ``act_vision`` (no
clock) and ``act`` with anything but a ``ModularInput`` raise.

Inference is fp32 (CPU single-sample on every selection and smoke path). The class is a student
(never privileged, never given a context or a motion feed); the batched GPU path refuses it
(``ila.eval.batched``: a ``SlotRequest`` carries no clock).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional

from ila.policies.encoder import EncoderConfig, VisionEncoder, VisionStudentBase, mlp

__all__ = [
    "PRIV_FIELDS",
    "ModularHeadConfig",
    "ModularEstimator",
    "ModularStudent",
    "TeacherPolicy",
    "build_frozen_teacher",
    "estimator_loss",
]

#: The privileged fields the head regresses, in contract order (checked against the contract).
PRIV_FIELDS: Final[tuple[str, ...]] = (
    "rel_position",
    "rel_velocity",
    "deck_normal",
    "rel_tilt",
    "height",
)


@dataclass(frozen=True)
class ModularHeadConfig:
    """``modular.yaml`` keys ``estimator_head`` and ``loss``.

    Attributes:
        hidden: Hidden widths, ReLU after each ((256, 256)).
        out_fields: Regressed P8 fields, in contract order (:data:`PRIV_FIELDS`).
        out_dim: Width of the estimate (11: the fields' sizes from P8's layout).
        loss_delta: Huber delta, normalised target units (1.0).
    """

    hidden: tuple[int, ...] = (256, 256)
    out_fields: tuple[str, ...] = PRIV_FIELDS
    out_dim: int = 11
    loss_delta: float = 1.0

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On bad widths, a non-positive delta or fields other than the privileged
                block in contract order.
        """
        if self.out_dim < 1 or (self.hidden and min(self.hidden) < 1):
            raise ValueError("widths must be positive")
        if not self.loss_delta > 0.0:
            raise ValueError(f"Huber delta must be positive, got {self.loss_delta}")
        if self.out_fields != PRIV_FIELDS:
            raise ValueError(f"estimator fields {self.out_fields} != {PRIV_FIELDS}")

    @classmethod
    def from_mapping(cls, train_cfg: Mapping[str, Any]) -> ModularHeadConfig:
        """Build from a parsed ``configs/train/modular.yaml``.

        The output width and field order come from the observation contract
        (:func:`ila.obs.split.load_contract`, P8's layout by field name); the config's
        ``out_fields`` must equal the contract's privileged block.

        Args:
            train_cfg: The parsed YAML (keys ``estimator_head``, ``loss``).

        Returns:
            The config.

        Raises:
            ValueError: If the loss is not Huber or the fields differ from the contract's.
        """
        from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

        head = train_cfg["estimator_head"]
        loss = train_cfg["loss"]
        if not isinstance(loss, Mapping) or loss.get("name") != "huber":
            raise ValueError(f"modular loss must be {{name: huber, delta}}, got {loss!r}")
        contract = load_contract()
        fields = tuple(str(f) for f in head["out_fields"])
        if fields != contract.privileged:
            raise ValueError(f"out_fields {fields} != contract privileged {contract.privileged}")
        return cls(
            hidden=tuple(int(w) for w in head["hidden"]),
            out_fields=fields,
            out_dim=contract.priv_dim,
            loss_delta=float(loss["delta"]),
        )


class ModularEstimator(nn.Module):
    """``(images, proprio) -> normalised privileged-block estimate``; no clock input.

    Buffers ``target_mean`` / ``target_std`` (``(11,)``, P8 units per field: module docstring)
    hold the training-set normalisation of the target; identity (0 / 1) until fitted
    (``ila.train.loop``). The encoder's ``proprio_mean`` / ``proprio_std`` likewise.

    Attributes:
        encoder: The shared vision encoder (``bc``'s architecture).
        head: MLP ``feature_dim -> 256 -> 256 -> 11``.
        head_cfg: Its config.
    """

    target_mean: torch.Tensor
    target_std: torch.Tensor

    def __init__(self, encoder_cfg: EncoderConfig, head_cfg: ModularHeadConfig) -> None:
        """Build the encoder and head.

        Args:
            encoder_cfg: Shared encoder architecture (default contract: 13-d proprio).
            head_cfg: Head architecture and loss.
        """
        super().__init__()
        self.encoder = VisionEncoder(encoder_cfg)
        self.head_cfg = head_cfg
        self.head = mlp(
            (encoder_cfg.feature_dim, *head_cfg.hidden, head_cfg.out_dim), final_activation=False
        )
        self.register_buffer("target_mean", torch.zeros(head_cfg.out_dim))
        self.register_buffer("target_std", torch.ones(head_cfg.out_dim))

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        crop_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Estimate the normalised privileged block. Pixels and allowlisted proprio only.

        Args:
            images: ``uint8[B, To, 96, 96, 3]``, oldest first.
            proprio: ``float32[B, To, 13]``, default-contract proprio (P8 units).
            crop_generator: Training-mode crop generator (see ``VisionEncoder``).

        Returns:
            ``float[B, 11]``, normalised by ``target_mean`` / ``target_std``.
        """
        out: torch.Tensor = self.head(self.encoder(images, proprio, crop_generator))
        return out

    def normalise(self, priv: torch.Tensor) -> torch.Tensor:
        """Return ``(priv - target_mean) / target_std`` (float32)."""
        return (priv.to(torch.float32) - self.target_mean) / self.target_std

    def denormalise(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``z * target_std + target_mean`` (float32; P8 units per field)."""
        return z.to(torch.float32) * self.target_std + self.target_mean


def estimator_loss(
    net: ModularEstimator,
    images: torch.Tensor,
    proprio: torch.Tensor,
    target: torch.Tensor,
    crop_generator: torch.Generator | None,
) -> torch.Tensor:
    """Per-sample Huber loss on normalised targets (P3-D1 §3: delta = 1.0).

    Args:
        net: The estimator.
        images: ``uint8[B, To, 96, 96, 3]``.
        proprio: ``float32[B, To, 13]``.
        target: ``float32[B, 11]``, the stored ``priv[t]`` in P8 units (normalised here with
            the network's fixed buffers).
        crop_generator: Training-mode crop generator.

    Returns:
        ``float32[B]``: the Huber loss averaged over the 11 dimensions.
    """
    z = net(images, proprio, crop_generator).float()
    per = functional.huber_loss(
        z, net.normalise(target), reduction="none", delta=net.head_cfg.loss_delta
    )
    return per.mean(dim=-1)


class TeacherPolicy(Protocol):
    """The frozen teacher as ``modular`` uses it: P8's ``EvalPolicy`` shape."""

    def reset(self, seed: int, context: Any = None) -> None:
        """Start an episode (the PPO MLP keeps no state)."""
        ...

    def act(self, obs: Any) -> Any:
        """Return a ``(3,)`` normalised action from P8's 25-d observation."""
        ...


def build_frozen_teacher() -> TeacherPolicy:
    """Return P8's ``T_ppo`` seed 0 (``final/``) with its frozen ``VecNormalize``.

    Built by :func:`ila.p8_adapter.teacher` exactly as P8's evaluation builds it (SB3 on the
    CPU, ``deterministic=True``, normaliser ``training=False``, ``norm_reward=False``).

    Returns:
        The policy.

    Raises:
        RuntimeError: If the policy has no normaliser or its normaliser is in training mode.
    """
    from ila import p8_adapter  # lazy: p8_adapter imports rld and pybullet

    policy = p8_adapter.teacher("ppo", 0)
    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None or bool(getattr(normalizer, "training", True)):
        raise RuntimeError("T_ppo must carry P8's frozen VecNormalize with training=False")
    return policy


class ModularStudent(VisionStudentBase):
    """P8 ``EvalPolicy`` shape around a :class:`ModularEstimator` and the frozen teacher.

    Flown by the runner through :meth:`act_modular` (``ila.eval.runner.is_modular``).

    Attributes:
        net: The estimator, eval mode.
        cfg: Its encoder config.
        device: Where the estimator runs (``"cpu"`` on every selection path).
        teacher: The frozen teacher (P8 ``T_ppo`` seed 0 unless injected, e.g. by a test).
        last_estimate: The de-normalised estimate behind the last action (``float32[11]``,
            P8 units), or ``None`` before the first; for the estimation-error study only.
        privileged: False (a student).
        needs_motion_feed: False.
    """

    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(
        self,
        net: ModularEstimator,
        device: torch.device | str = "cpu",
        teacher: TeacherPolicy | None = None,
    ) -> None:
        """Wrap a trained estimator; build the frozen teacher unless one is given.

        Args:
            net: The estimator (moved to ``device``, eval mode).
            device: Torch device for the estimator (the teacher runs on P8's CPU path).
            teacher: A stand-in teacher (tests); ``None`` builds :func:`build_frozen_teacher`.

        Raises:
            ValueError: If the estimator was not built for the default 13-d contract.
        """
        from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cfg = net.encoder.cfg
        self.contract = load_contract()
        if self.cfg.proprio_dim != self.contract.proprio_dim:
            raise ValueError("modular uses the default contract's 13-d proprio")
        self.teacher: TeacherPolicy = build_frozen_teacher() if teacher is None else teacher
        self.last_estimate: NDArray[np.float32] | None = None

    def _reset_episode(self, seed: int) -> None:
        """Reset the teacher with the episode seed (P8's ``reset``; PPO keeps no state)."""
        self.last_estimate = None
        self.teacher.reset(seed, None)

    def act(self, obs: object) -> NDArray[np.float32]:
        """Act on a ``ModularInput``; refuse anything else.

        Args:
            obs: An ``ila.obs.split.ModularInput``.

        Returns:
            ``float32[3]`` in ``[-1, 1]``.

        Raises:
            TypeError: For a ``StudentInput`` (no clock for the teacher) or a raw observation.
        """
        from ila.obs.split import ModularInput  # lazy

        if not isinstance(obs, ModularInput):
            raise TypeError(
                "modular takes a ModularInput (act_modular); got "
                f"{type(obs).__name__} (a raw P8 observation is privileged; a StudentInput "
                "carries no clock for the teacher)"
            )
        return self.act_modular(obs)

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Refuse: the teacher needs ``time_fraction``, which only a ``ModularInput`` carries.

        Raises:
            TypeError: Always.
        """
        del image_stack, proprio
        raise TypeError("modular needs a ModularInput (act_modular): the teacher reads the clock")

    def estimate(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """De-normalised privileged-block estimate for one window (no clock).

        Args:
            image_stack: ``uint8[To, 96, 96, 3]``, oldest first.
            proprio: ``float32[To, 13]``, default contract, P8 units.

        Returns:
            ``float32[11]`` in contract order and P8 units (module docstring).
        """
        images, prop = self._single_to_batch(image_stack, proprio)
        with torch.inference_mode():
            est = self.net.denormalise(self.net(images, prop))
        out: NDArray[np.float32] = est[0].cpu().numpy().astype(np.float32)
        return out

    def estimate_batch(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Batched :meth:`estimate` (open-loop study).

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, 13]``.

        Returns:
            ``float32[B, 11]``, P8 units.
        """
        img_t, prop_t = self._to_device(np.asarray(images), np.asarray(proprio, np.float32))
        with torch.inference_mode():
            est = self.net.denormalise(self.net(img_t, prop_t))
        out: NDArray[np.float32] = est.cpu().numpy().astype(np.float32)
        return out

    def act_modular(self, mi: Any) -> NDArray[np.float32]:
        """Estimate, assemble P8's observation with the true clock, query the frozen teacher.

        Args:
            mi: An ``ila.obs.split.ModularInput`` (images and 13-d proprio at
                ``t-To+1 ... t``; ``time_fraction`` at the same steps, dimensionless).

        Returns:
            ``float32[3]`` in ``[-1, 1]``: the teacher's normalised world-frame velocity
            setpoint (x v_max = 1.5 m/s model scale, norm-capped by the env).

        Raises:
            TypeError: If ``mi`` is not a ``ModularInput``.
        """
        from ila.obs.split import ModularInput, assemble_teacher_obs  # lazy

        if not isinstance(mi, ModularInput):
            raise TypeError(f"act_modular takes a ModularInput, got {type(mi).__name__}")
        si = mi.student_input
        est = self.estimate(si.images, si.proprio)  # the estimator sees no clock
        self.inference_calls += 1  # the estimator call (the frozen teacher's MLP not counted)
        # Time axis, not a field index: window position To - 1 is step t (oldest first,
        # ila.policies.dp.chunking.obs_window_steps); the fields are placed by name.
        newest = mi.n_obs_steps - 1
        teacher_obs = assemble_teacher_obs(
            self.contract, si.proprio[newest], est, mi.time_fraction[newest]
        )
        action = np.asarray(self.teacher.act(teacher_obs), dtype=np.float32).reshape(-1)
        self.last_estimate = est
        return action
