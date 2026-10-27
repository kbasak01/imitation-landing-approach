"""Behaviour-cloning vision student: shared encoder + MLP head.

``bc``, ``bc_matched``, ``bc_dart`` and ``bc_dagger`` share this architecture (plan §1.6):
:class:`ila.policies.encoder.VisionEncoder` (256-d feature for ``To = 2``) -> MLP 2 x 256
(ReLU) -> 3 -> ``tanh``. The output is the normalised world-frame velocity setpoint in
``[-1, 1]^3``; the env scales it by v_max = 1.5 m/s model scale (lambda = 1/25) and applies its
own norm cap. Inference is fp32; training (Phase 4) is not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ila.policies.encoder import EncoderConfig, VisionEncoder, VisionStudentBase, mlp

__all__ = ["BCHeadConfig", "BCNet", "BCStudent"]


@dataclass(frozen=True)
class BCHeadConfig:
    """BC head architecture (``configs/train/bc.yaml`` key ``head``).

    Attributes:
        hidden: Hidden widths, ReLU after each ((256, 256)).
        out: Action width (3).
        activation: Output activation; only ``"tanh"``.
    """

    hidden: tuple[int, ...] = (256, 256)
    out: int = 3
    activation: str = "tanh"

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On an unsupported activation or bad widths.
        """
        if self.activation != "tanh":
            raise ValueError(f"only a tanh output is implemented, got {self.activation!r}")
        if self.out < 1 or (self.hidden and min(self.hidden) < 1):
            raise ValueError("widths must be positive")

    @classmethod
    def from_mapping(cls, train_cfg: Mapping[str, Any]) -> BCHeadConfig:
        """Build from a parsed ``configs/train/bc*.yaml``.

        Args:
            train_cfg: The parsed YAML (key ``head``).

        Returns:
            The config.
        """
        head = train_cfg["head"]
        return cls(
            hidden=tuple(int(w) for w in head["hidden"]),
            out=int(head["out"]),
            activation=str(head["activation"]),
        )


class BCNet(nn.Module):
    """``(images, proprio) -> action`` in ``[-1, 1]``."""

    def __init__(self, encoder_cfg: EncoderConfig, head_cfg: BCHeadConfig) -> None:
        """Build the encoder and head.

        Args:
            encoder_cfg: Shared encoder architecture.
            head_cfg: Head architecture.
        """
        super().__init__()
        self.encoder = VisionEncoder(encoder_cfg)
        self.head = mlp(
            (encoder_cfg.feature_dim, *head_cfg.hidden, head_cfg.out), final_activation=False
        )

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        crop_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict actions.

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, D]``.
            crop_generator: Training-mode crop generator (see ``VisionEncoder``).

        Returns:
            ``float32[B, 3]`` in ``[-1, 1]`` (``tanh``).
        """
        return torch.tanh(self.head(self.encoder(images, proprio, crop_generator)))


class BCStudent(VisionStudentBase):
    """P8 ``EvalPolicy`` shape around a :class:`BCNet` (stateless across steps).

    Attributes:
        net: The network, in eval mode.
        cfg: Its encoder config.
        device: Where inference runs.
    """

    def __init__(self, net: BCNet, device: torch.device | str = "cpu") -> None:
        """Wrap a network for closed-loop evaluation.

        Args:
            net: The network (moved to ``device`` and put in eval mode).
            device: Torch device.
        """
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cfg = net.encoder.cfg

    def _reset_episode(self, seed: int) -> None:
        """BC keeps no per-episode state."""
        del seed

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return the action for one window.

        Args:
            image_stack: ``uint8[To, 96, 96, 3]``, oldest first.
            proprio: ``float32[To, D]``, contract fields, P8 units.

        Returns:
            ``float32[3]`` in ``[-1, 1]``.
        """
        images, prop = self._single_to_batch(image_stack, proprio)
        with torch.inference_mode():
            out = self.net(images, prop)
        action: NDArray[np.float32] = out[0].cpu().numpy().astype(np.float32)
        return action

    def act_batch(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Batched entry point for a GPU server (one row per lockstep slot).

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, D]``.

        Returns:
            ``float32[B, 3]`` in ``[-1, 1]``.
        """
        img_t, prop_t = self._to_device(np.asarray(images), np.asarray(proprio, np.float32))
        with torch.inference_mode():
            out = self.net(img_t, prop_t)
        actions: NDArray[np.float32] = out.cpu().numpy().astype(np.float32)
        return actions
