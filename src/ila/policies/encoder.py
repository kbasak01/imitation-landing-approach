"""Shared ResNet-18 (GroupNorm) + spatial-softmax image encoder, and the vision-student contract.

One encoder for every vision arm (``bc*``, ``dp``, ``modular``), so architecture is never a
confound between arms (plan §1.6):

* per frame: ResNet-18 trunk (torchvision, ``weights=None``), every BatchNorm replaced by
  ``GroupNorm(16, C)``, then a 1 x 1 conv to 32 keypoint maps and a spatial softmax giving the
  expected ``(x, y)`` of each map in normalised image coordinates [-1, 1] -> 64-d;
* ``To`` frames concatenated (oldest first);
* proprio (13-d by default, P8 units: rad, rad/s, m/s **model scale**, normalised action,
  {0, 1}) standardised by fixed buffers, then an MLP (64, 64) applied per step with shared
  weights, ``To`` outputs concatenated;
* feature = ``[frames ‖ proprio]``, ``To * (64 + 64)`` = 256-d for ``To = 2``.

Crop convention (P1-D1, ``configs/render/camera.yaml``): the input is the 96 px frame; at
evaluation the encoder takes the **centre** 88 px crop (offset 4 px); in training mode it takes
a random 88 px crop per image, drawn from an explicit ``torch.Generator`` (never a global RNG).

Images enter as ``uint8`` RGB in ``[0, 255]`` and are scaled to ``[0, 1]``; no ImageNet
statistics (the trunk is not pretrained).

Nothing here imports ``rld`` or reads render metadata: a student sees pixels and the proprio
allowlist only (il-conventions).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from torch import nn

__all__ = [
    "EncoderConfig",
    "SpatialSoftmax",
    "VisionEncoder",
    "VisionStudentBase",
    "centre_crop",
    "count_parameters",
    "mlp",
    "random_crop",
]


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture of the shared vision encoder.

    Attributes:
        trunk: Only ``"resnet18"``.
        pretrained: Must be ``False`` (random init).
        norm: Only ``"groupnorm"``.
        groups: GroupNorm groups in the trunk (16).
        keypoints: Spatial-softmax keypoints per frame (32 -> 64-d).
        proprio_dim: Student proprio width (13 default contract).
        proprio_mlp: Hidden/output widths of the per-step proprio MLP ((64, 64)).
        n_obs_steps: ``To``, frames and proprio steps per input (2).
        input_px: Frame side the student receives, px (96).
        crop_px: Crop side fed to the trunk, px (88); centre at eval, random at train.
    """

    trunk: str = "resnet18"
    pretrained: bool = False
    norm: str = "groupnorm"
    groups: int = 16
    keypoints: int = 32
    proprio_dim: int = 13
    proprio_mlp: tuple[int, ...] = (64, 64)
    n_obs_steps: int = 2
    input_px: int = 96
    crop_px: int = 88

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On an unsupported trunk/norm, pretrained weights, or bad sizes.
        """
        if self.trunk != "resnet18":
            raise ValueError(f"only trunk 'resnet18' is implemented, got {self.trunk!r}")
        if self.pretrained:
            raise ValueError("the encoder is never pretrained (plan §1.6)")
        if self.norm != "groupnorm":
            raise ValueError(f"only norm 'groupnorm' is implemented, got {self.norm!r}")
        if self.groups < 1 or 64 % self.groups:
            raise ValueError(f"groups must divide 64 (the trunk's narrowest width): {self.groups}")
        if self.keypoints < 1 or self.proprio_dim < 1 or self.n_obs_steps < 1:
            raise ValueError("keypoints, proprio_dim and n_obs_steps must be >= 1")
        if not self.proprio_mlp or min(self.proprio_mlp) < 1:
            raise ValueError(f"proprio_mlp must be non-empty positive widths: {self.proprio_mlp}")
        if not 0 < self.crop_px <= self.input_px:
            raise ValueError(f"need 0 < crop_px <= input_px, got {self.crop_px}, {self.input_px}")

    @classmethod
    def from_mappings(
        cls,
        train_cfg: Mapping[str, Any],
        camera_cfg: Mapping[str, Any],
        *,
        proprio_dim: int,
        n_obs_steps: int,
    ) -> EncoderConfig:
        """Build from a ``configs/train/<arm>.yaml`` and ``configs/render/camera.yaml``.

        Args:
            train_cfg: Parsed train config (keys ``encoder``, ``proprio_mlp``).
            camera_cfg: Parsed camera config (``camera.width_px``/``height_px``,
                ``train_augmentation.random_crop_px``).
            proprio_dim: From the observation contract.
            n_obs_steps: From the observation contract (``To``).

        Returns:
            The config.

        Raises:
            ValueError: If the camera frame is not square.
        """
        enc = train_cfg["encoder"]
        cam = camera_cfg["camera"]
        if int(cam["width_px"]) != int(cam["height_px"]):
            raise ValueError("the encoder assumes a square frame")
        return cls(
            trunk=str(enc["trunk"]),
            pretrained=bool(enc["pretrained"]),
            norm=str(enc["norm"]),
            groups=int(enc["groups"]),
            keypoints=int(enc["spatial_softmax_keypoints"]),
            proprio_dim=proprio_dim,
            proprio_mlp=tuple(int(w) for w in train_cfg["proprio_mlp"]),
            n_obs_steps=n_obs_steps,
            input_px=int(cam["width_px"]),
            crop_px=int(camera_cfg["train_augmentation"]["random_crop_px"]),
        )

    @property
    def frame_feature_dim(self) -> int:
        """Per-frame feature width: ``2 * keypoints`` (64)."""
        return 2 * self.keypoints

    @property
    def proprio_feature_dim(self) -> int:
        """Per-step proprio feature width: the proprio MLP's last width (64)."""
        *_, last = self.proprio_mlp
        return last

    @property
    def feature_dim(self) -> int:
        """Encoder output width: ``To * (2 * keypoints + proprio_feature_dim)`` (256)."""
        return self.n_obs_steps * (self.frame_feature_dim + self.proprio_feature_dim)


def centre_crop(images: torch.Tensor, size: int) -> torch.Tensor:
    """Take the centre ``size x size`` crop of ``(N, C, H, W)`` images.

    Args:
        images: ``(N, C, H, W)``.
        size: Crop side, px.

    Returns:
        ``(N, C, size, size)``; offset ``(H - size) // 2`` rows, ``(W - size) // 2`` columns
        (4 px for 88 of 96).
    """
    h, w = images.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return images[..., top : top + size, left : left + size]


def random_crop(images: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
    """Take an independent random ``size x size`` crop of each ``(N, C, H, W)`` image.

    Args:
        images: ``(N, C, H, W)``.
        size: Crop side, px.
        generator: CPU generator for the offsets (explicit; never the global RNG).

    Returns:
        ``(N, C, size, size)``; offsets uniform in ``[0, H - size]`` and ``[0, W - size]``.
    """
    n = images.shape[0]
    h, w = images.shape[-2:]
    tops = torch.randint(0, h - size + 1, (n,), generator=generator).tolist()
    lefts = torch.randint(0, w - size + 1, (n,), generator=generator).tolist()
    return torch.stack(
        [
            images[i, :, t : t + size, s : s + size]
            for i, (t, s) in enumerate(zip(tops, lefts, strict=True))
        ]
    )


class SpatialSoftmax(nn.Module):
    """1 x 1 conv to ``K`` maps, softmax over positions, expected ``(x, y)`` per map.

    Coordinates are normalised to ``[-1, 1]`` over the feature map (x = column, left -> right;
    y = row, top -> bottom), temperature fixed at 1 (robomimic's default). Output ordering is
    ``[x_0, y_0, x_1, y_1, ...]``.
    """

    def __init__(self, in_channels: int, keypoints: int) -> None:
        """Create the keypoint projection.

        Args:
            in_channels: Feature-map channels (512 for ResNet-18).
            keypoints: ``K``.
        """
        super().__init__()
        self.keypoints = keypoints
        self.proj = nn.Conv2d(in_channels, keypoints, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(N, C, h, w)`` features to ``(N, 2K)`` keypoint coordinates.

        Args:
            features: Trunk output.

        Returns:
            Expected coordinates in ``[-1, 1]``.
        """
        maps = self.proj(features)
        n, k, h, w = maps.shape
        attn = torch.softmax(maps.reshape(n, k, h * w), dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=maps.device, dtype=maps.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=maps.device, dtype=maps.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ex = (attn * grid_x.reshape(1, 1, h * w)).sum(-1)
        ey = (attn * grid_y.reshape(1, 1, h * w)).sum(-1)
        return torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)


def _group_norm(groups: int, channels: int) -> nn.GroupNorm:
    """Return ``GroupNorm(groups, channels)`` (torchvision ``norm_layer`` factory)."""
    return nn.GroupNorm(groups, channels)


def _resnet18_trunk(groups: int) -> nn.Sequential:
    """Return torchvision's ResNet-18 up to ``layer4`` (no pooling, no fc), GroupNorm'd.

    Args:
        groups: GroupNorm groups replacing every BatchNorm.

    Returns:
        ``(N, 3, H, W) -> (N, 512, ceil(H/32), ceil(W/32))`` (3 x 3 at 88 px).
    """
    net = torchvision.models.resnet18(
        weights=None, norm_layer=functools.partial(_group_norm, groups)
    )
    return nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
    )


def mlp(widths: Sequence[int], *, final_activation: bool) -> nn.Sequential:
    """Return ``Linear-ReLU-...-Linear`` over ``widths`` (input first).

    Args:
        widths: Layer widths, input first.
        final_activation: Whether a ReLU follows the last Linear.

    Returns:
        The MLP.
    """
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VisionEncoder(nn.Module):
    """``(images, proprio) -> feature``, shared by every vision arm.

    Buffers ``proprio_mean`` / ``proprio_std`` (shape ``(D,)``, P8 units per field) standardise
    the proprio. They are identity (0 / 1) until fitted on the training split (Phase 4).
    """

    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor

    def __init__(self, cfg: EncoderConfig) -> None:
        """Build the trunk, keypoint head and proprio MLP.

        Args:
            cfg: Architecture.
        """
        super().__init__()
        self.cfg = cfg
        self.trunk = _resnet18_trunk(cfg.groups)
        self.keypoints = SpatialSoftmax(512, cfg.keypoints)
        self.proprio_mlp = mlp((cfg.proprio_dim, *cfg.proprio_mlp), final_activation=False)
        self.register_buffer("proprio_mean", torch.zeros(cfg.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(cfg.proprio_dim))

    @property
    def feature_dim(self) -> int:
        """Output width (``cfg.feature_dim``)."""
        return self.cfg.feature_dim

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        crop_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode a batch of observation windows.

        Args:
            images: ``uint8[B, To, H, W, 3]`` RGB, ``H = W = input_px`` (96), oldest first.
            proprio: ``float32[B, To, D]``, the contract's proprio fields at the same steps.
            crop_generator: Required in training mode (random crop offsets); ignored in eval
                mode (centre crop).

        Returns:
            ``float32[B, feature_dim]``.

        Raises:
            ValueError: On a wrong dtype or shape, or training mode without a generator.
        """
        c = self.cfg
        if images.dtype != torch.uint8:
            raise ValueError(f"images must be uint8, got {images.dtype}")
        b = images.shape[0]
        if images.shape[1:] != (c.n_obs_steps, c.input_px, c.input_px, 3):
            raise ValueError(
                f"images must be [B, {c.n_obs_steps}, {c.input_px}, {c.input_px}, 3], "
                f"got {tuple(images.shape)}"
            )
        if proprio.shape != (b, c.n_obs_steps, c.proprio_dim):
            raise ValueError(
                f"proprio must be [{b}, {c.n_obs_steps}, {c.proprio_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        x = images.reshape(b * c.n_obs_steps, c.input_px, c.input_px, 3).permute(0, 3, 1, 2)
        x = x.to(torch.float32) / 255.0
        if self.training:
            if crop_generator is None:
                raise ValueError("training-mode crop needs an explicit crop_generator")
            x = random_crop(x, c.crop_px, crop_generator)
        else:
            x = centre_crop(x, c.crop_px)
        kp = self.keypoints(self.trunk(x)).reshape(b, c.n_obs_steps * c.frame_feature_dim)
        p = (proprio.to(torch.float32) - self.proprio_mean) / self.proprio_std
        pf = self.proprio_mlp(p).reshape(b, c.n_obs_steps * c.proprio_feature_dim)
        return torch.cat([kp, pf], dim=-1)


class VisionStudentBase:
    """The vision-student side of P8's ``EvalPolicy`` shape, shared by ``bc`` and ``dp``.

    * ``reset(seed, context=None, motion_feed=None)``: a vision student refuses a privileged
      context or a motion feed (both would be privileged inputs).
    * ``act(obs)``: accepts only an ``ila.obs.split.StudentInput``; a raw P8 observation (the
      25-d vector) raises ``TypeError``.
    * ``act_vision(image_stack, proprio)``: ``uint8[To, 96, 96, 3]`` and ``float32[To, D]``
      (the contract's proprio, P8 units) -> ``float32[3]`` normalised world-frame velocity
      setpoint in ``[-1, 1]`` (the env scales by v_max = 1.5 m/s model scale, norm-capped).
    * ``inference_calls``: network invocations on the closed-loop path since the last
      ``reset`` (``bc``: one per ``act_vision``; diffusion: one per chunk sampled; ``modular``:
      one estimator call per ``act_modular``). A diagnostic counter only: it is never read by
      the student and never changes an action. The batched GPU entry points (``act_batch``,
      ``sample_batch``) do not count; their server keeps its own tally.
    """

    cfg: EncoderConfig
    device: torch.device
    #: Network invocations since the last ``reset`` (closed-loop single-sample path).
    inference_calls: int = 0

    def reset(
        self, seed: int, context: object | None = None, motion_feed: object | None = None
    ) -> None:
        """Clear per-episode state.

        Args:
            seed: The episode seed.
            context: Must be ``None`` (privileged deck trajectory).
            motion_feed: Must be ``None``.

        Raises:
            ValueError: If a context or motion feed is passed.
        """
        if context is not None or motion_feed is not None:
            raise ValueError("a vision student never receives a privileged context or feed")
        self.inference_calls = 0
        self._reset_episode(int(seed))

    def _reset_episode(self, seed: int) -> None:
        raise NotImplementedError

    def act(self, obs: object) -> NDArray[np.float32]:
        """Act on a ``StudentInput``; refuse anything else (in particular P8's 25-d obs).

        Args:
            obs: An ``ila.obs.split.StudentInput``.

        Returns:
            ``float32[3]`` in ``[-1, 1]``.

        Raises:
            TypeError: If ``obs`` is not a ``StudentInput`` (e.g. a raw observation vector).
        """
        from ila.obs.split import StudentInput  # lazy: keeps this module free of rld

        if not isinstance(obs, StudentInput):
            raise TypeError(
                "a vision student takes a StudentInput (act) or act_vision(image_stack, "
                f"proprio); got {type(obs).__name__} (a raw P8 observation is privileged)"
            )
        return self.act_vision(obs.images, obs.proprio)

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return the action for one observation window (see class docstring)."""
        raise NotImplementedError

    def _check_window(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Validate one observation window.

        Returns:
            The window as numpy arrays.

        Raises:
            ValueError: On a wrong dtype or shape (a 25-d proprio fails the width check).
        """
        c = self.cfg
        img = np.asarray(image_stack)
        prop = np.asarray(proprio)
        if img.dtype != np.uint8 or img.shape != (c.n_obs_steps, c.input_px, c.input_px, 3):
            raise ValueError(
                f"image_stack must be uint8[{c.n_obs_steps}, {c.input_px}, {c.input_px}, 3], "
                f"got {img.dtype} {img.shape}"
            )
        if prop.dtype != np.float32 or prop.shape != (c.n_obs_steps, c.proprio_dim):
            raise ValueError(
                f"proprio must be float32[{c.n_obs_steps}, {c.proprio_dim}], "
                f"got {prop.dtype} {prop.shape}"
            )
        return img, prop

    def _single_to_batch(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate one window and return ``(1, To, ...)`` tensors on the device."""
        img, prop = self._check_window(image_stack, proprio)
        return self._to_device(img[None], prop[None])

    def _to_device(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy stacked numpy windows to the device (read-only inputs are copied first)."""
        img = np.array(
            images, dtype=np.uint8, order="C", copy=True if not images.flags.writeable else None
        )
        prop = np.array(
            proprio, dtype=np.float32, order="C", copy=True if not proprio.flags.writeable else None
        )
        return torch.from_numpy(img).to(self.device), torch.from_numpy(prop).to(self.device)


def count_parameters(module: nn.Module) -> int:
    """Return the number of parameters (trainable or not; buffers excluded)."""
    return int(sum(p.numel() for p in module.parameters()))
