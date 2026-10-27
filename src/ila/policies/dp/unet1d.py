# Adapted from real-stanford/diffusion_policy (https://github.com/real-stanford/diffusion_policy),
# files diffusion_policy/model/diffusion/conditional_unet1d.py, conv1d_components.py and
# positional_embedding.py. Changes: type hints and docstrings; global conditioning only (the
# local-conditioning path is dropped); FiLM (``cond_predict_scale``) fixed on; einops replaced
# by ``transpose``/``unsqueeze``; a check that the horizon is divisible by the down-sampling
# factor. Original licence:
#
# MIT License
#
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Conditional 1D U-Net with FiLM conditioning (Chi et al., RSS 2023).

Input: a noisy action sequence ``(B, Tp, A)`` (``A = 3`` normalised actions, ``Tp = 16``
control steps of 1/30 s model scale), the diffusion step ``k`` (integer in
``[0, num_train_timesteps)``) and a global conditioning vector (the encoder feature). Output:
the predicted noise, same shape as the input.

Structure (``down_dims = [128, 256, 512]``, kernel 5, GroupNorm 8 groups, Mish):

* diffusion-step encoder: sinusoidal(128) -> Linear(512) -> Mish -> Linear(128);
* FiLM: each residual block maps ``[step embedding ‖ obs feature]`` to a per-channel scale and
  bias applied after its first conv block;
* down path: two residual blocks per level, stride-2 down-sampling between levels
  (16 -> 8 -> 4); two mid blocks; up path with skip connections back to ``Tp``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

__all__ = ["ConditionalResidualBlock1D", "ConditionalUnet1D", "SinusoidalPosEmb", "UNetConfig"]


@dataclass(frozen=True)
class UNetConfig:
    """U-Net architecture (``configs/train/dp.yaml`` key ``unet``).

    Attributes:
        down_dims: Channel widths per level ((128, 256, 512)).
        kernel_size: Conv1d kernel (5).
        diffusion_step_embed_dim: Sinusoidal step-embedding width (128).
        n_groups: GroupNorm groups in every conv block (8, the reference's value; not in the
            config).
        cond: Conditioning type; only ``"film"`` (scale and bias).
    """

    down_dims: tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    diffusion_step_embed_dim: int = 128
    n_groups: int = 8
    cond: str = "film"

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On unsupported conditioning or widths GroupNorm cannot split.
        """
        if self.cond != "film":
            raise ValueError(f"only FiLM conditioning is implemented, got {self.cond!r}")
        if not self.down_dims or any(d % self.n_groups for d in self.down_dims):
            raise ValueError(f"every down_dim must be divisible by n_groups={self.n_groups}")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (same-length padding)")
        if self.diffusion_step_embed_dim % 2 or self.diffusion_step_embed_dim < 4:
            raise ValueError("diffusion_step_embed_dim must be even and >= 4")

    @classmethod
    def from_mapping(cls, unet: Mapping[str, Any]) -> UNetConfig:
        """Build from the parsed ``unet`` block of ``configs/train/dp.yaml``.

        Args:
            unet: Keys ``down_dims``, ``kernel_size``, ``diffusion_step_embed_dim``,
                ``cond`` and optionally ``n_groups``.

        Returns:
            The config.
        """
        return cls(
            down_dims=tuple(int(d) for d in unet["down_dims"]),
            kernel_size=int(unet["kernel_size"]),
            diffusion_step_embed_dim=int(unet["diffusion_step_embed_dim"]),
            n_groups=int(unet.get("n_groups", 8)),
            cond=str(unet["cond"]),
        )

    @property
    def downsample_factor(self) -> int:
        """``2 ** (levels - 1)``: the horizon must be a multiple of this (4)."""
        return int(2 ** (len(self.down_dims) - 1))


class SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding of the (integer) diffusion step."""

    def __init__(self, dim: int) -> None:
        """Create the embedding.

        Args:
            dim: Output width (even).
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``(B,)`` steps as ``(B, dim)`` ``[sin ‖ cos]``."""
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -scale)
        arg = x.to(torch.float32)[:, None] * freqs[None, :]
        return torch.cat((arg.sin(), arg.cos()), dim=-1)


def _conv_block(c_in: int, c_out: int, kernel_size: int, n_groups: int) -> nn.Sequential:
    """Conv1d (same length) -> GroupNorm -> Mish."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
        nn.GroupNorm(n_groups, c_out),
        nn.Mish(),
    )


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM (scale, bias) from the conditioning after the first."""

    def __init__(
        self, c_in: int, c_out: int, cond_dim: int, kernel_size: int, n_groups: int
    ) -> None:
        """Create the block.

        Args:
            c_in: Input channels.
            c_out: Output channels.
            cond_dim: Conditioning width (step embedding + obs feature).
            kernel_size: Conv kernel.
            n_groups: GroupNorm groups.
        """
        super().__init__()
        self.out_channels = c_out
        self.block1 = _conv_block(c_in, c_out, kernel_size, n_groups)
        self.block2 = _conv_block(c_out, c_out, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * c_out))
        self.residual_conv: nn.Module = (
            nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(B, c_in, L)``.
            cond: ``(B, cond_dim)``.

        Returns:
            ``(B, c_out, L)``.
        """
        out = self.block1(x)
        film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        out = film[:, 0] * out + film[:, 1]
        out = self.block2(out)
        res: torch.Tensor = out + self.residual_conv(x)
        return res


class ConditionalUnet1D(nn.Module):
    """Noise-prediction U-Net over the action sequence, globally conditioned."""

    def __init__(self, action_dim: int, global_cond_dim: int, cfg: UNetConfig) -> None:
        """Build the network.

        Args:
            action_dim: Action channels ``A`` (3).
            global_cond_dim: Obs-feature width (the encoder's ``feature_dim``, 256).
            cfg: Architecture.
        """
        super().__init__()
        self.cfg = cfg
        dsed = cfg.diffusion_step_embed_dim
        k, g = cfg.kernel_size, cfg.n_groups
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim
        all_dims: Sequence[int] = (action_dim, *cfg.down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        mid = all_dims[-1]

        self.down_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(d_in, d_out, cond_dim, k, g),
                        ConditionalResidualBlock1D(d_out, d_out, cond_dim, k, g),
                        nn.Identity() if last else nn.Conv1d(d_out, d_out, 3, 2, 1),
                    ]
                )
            )
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid, mid, cond_dim, k, g),
                ConditionalResidualBlock1D(mid, mid, cond_dim, k, g),
            ]
        )
        self.up_modules = nn.ModuleList()
        for d_in, d_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(2 * d_out, d_in, cond_dim, k, g),
                        ConditionalResidualBlock1D(d_in, d_in, cond_dim, k, g),
                        nn.ConvTranspose1d(d_in, d_in, 4, 2, 1),
                    ]
                )
            )
        start = cfg.down_dims[0]
        self.final_conv = nn.Sequential(
            _conv_block(start, start, k, g), nn.Conv1d(start, action_dim, 1)
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise in ``sample``.

        Args:
            sample: ``(B, Tp, A)`` noisy actions.
            timestep: ``(B,)`` or scalar integer diffusion step.
            global_cond: ``(B, global_cond_dim)``.

        Returns:
            ``(B, Tp, A)`` predicted noise (epsilon).

        Raises:
            ValueError: If ``Tp`` is not a multiple of the down-sampling factor.
        """
        b, horizon = sample.shape[0], sample.shape[1]
        if horizon % self.cfg.downsample_factor:
            raise ValueError(
                f"horizon {horizon} must be a multiple of {self.cfg.downsample_factor}"
            )
        steps = timestep.reshape(-1).to(sample.device)
        if steps.numel() == 1:
            steps = steps.expand(b)
        cond = torch.cat([self.diffusion_step_encoder(steps), global_cond], dim=-1)
        x = sample.transpose(1, 2)
        skips: list[torch.Tensor] = []
        for level in self.down_modules:
            res1, res2, down = cast(nn.ModuleList, level)
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for block in self.mid_modules:
            x = block(x, cond)
        for level in self.up_modules:
            res1, res2, up = cast(nn.ModuleList, level)
            x = torch.cat((x, skips.pop()), dim=1)
            x = up(res2(res1(x, cond), cond))
        out: torch.Tensor = self.final_conv(x).transpose(1, 2)
        return out
