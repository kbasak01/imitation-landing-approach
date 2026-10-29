"""``bc_state``: MLP student on P8's full 25-d observation (the imitation-gap reference).

**Privileged by design** (P3-D1 §3, declared exception 3 of il-conventions "Randomness"): the
input is P8's whole observation vector, ``time_fraction`` and the five privileged fields
included. ``bc_state`` is never a vision row and never shares a code path with a vision
student's input: it has no ``act_vision``, so the runner flies it on P8's observation path
(``TeacherActor``), exactly like the teacher it imitates.

Model (``configs/train/bc_state.yaml``; "the same head on the 25-d obs", P3-D1 §3):
``obs`` -> standardise by fixed per-dimension buffers -> MLP 2 x 256 (ReLU) -> 3 -> ``tanh``.
The output is the normalised world-frame velocity setpoint in ``[-1, 1]^3``; the env scales it
by v_max = 1.5 m/s **model scale** (lambda = 1/25) and applies its norm cap.

Normaliser (``input_norm: train_mean_std``): the per-dimension mean and standard deviation of
``teacher_obs`` over the run's own training frames, computed once at train start by
:func:`fit_normaliser`, stored as buffers (so they are in every checkpoint) and never updated.
Units per field are P8's (rad, rad/s, m and m/s model scale, unit vectors, normalised action,
{0, 1}, the dimensionless clock in [0, 1]). Trains in fp32 (vision arms train in bf16; the
precision confound is stated with H4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ila.policies.encoder import mlp

__all__ = [
    "MIN_STD",
    "StateMLP",
    "StateMLPConfig",
    "StateMLPStudent",
    "fit_normaliser",
]

#: A dimension whose training-frame standard deviation is below this is not scaled (std := 1).
MIN_STD: Final[float] = 1e-6


@dataclass(frozen=True)
class StateMLPConfig:
    """``bc_state`` architecture (``configs/train/bc_state.yaml`` key ``model``).

    Attributes:
        obs_dim: P8's observation width (25 under the eval config).
        hidden: Hidden widths, ReLU after each ((256, 256)).
        out: Action width (3).
        activation: Output activation; only ``"tanh"``.
    """

    obs_dim: int = 25
    hidden: tuple[int, ...] = (256, 256)
    out: int = 3
    activation: str = "tanh"

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On an unsupported activation or non-positive widths.
        """
        if self.activation != "tanh":
            raise ValueError(f"only a tanh output is implemented, got {self.activation!r}")
        if self.obs_dim < 1 or self.out < 1 or (self.hidden and min(self.hidden) < 1):
            raise ValueError("widths must be positive")

    @classmethod
    def from_mapping(cls, train_cfg: Mapping[str, Any], obs_dim: int) -> StateMLPConfig:
        """Build from a parsed ``configs/train/bc_state.yaml``.

        Args:
            train_cfg: The parsed YAML (key ``model``; ``input_norm`` must be
                ``train_mean_std``).
            obs_dim: P8's observation width (from the contract).

        Returns:
            The config.

        Raises:
            ValueError: If the input normalisation is not ``train_mean_std``.
        """
        if train_cfg.get("input_norm") != "train_mean_std":
            raise ValueError(f"unsupported input_norm {train_cfg.get('input_norm')!r}")
        model = train_cfg["model"]
        return cls(
            obs_dim=int(obs_dim),
            hidden=tuple(int(w) for w in model["hidden"]),
            out=int(model["out"]),
            activation=str(model["activation"]),
        )


def fit_normaliser(frames: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the per-dimension mean and std of the training frames (population std, float64 sums).

    Args:
        frames: ``float32[N, obs_dim]``, every training frame of the run (P8 units per field).

    Returns:
        ``(mean, std)``, each ``float32[obs_dim]``; a dimension with ``std < MIN_STD`` gets
        ``std = 1`` (it is centred, not scaled).

    Raises:
        ValueError: On an empty or non-2-D input or non-finite values.
    """
    x = np.asarray(frames, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"need float[N >= 1, D] frames, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite training frames")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < MIN_STD, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class StateMLP(nn.Module):
    """``obs (25-d, P8 order) -> action`` in ``[-1, 1]``, standardised by fixed buffers.

    Buffers ``obs_mean`` / ``obs_std`` (``(obs_dim,)``, P8 units) are identity (0 / 1) until
    :meth:`set_normaliser` writes the training-frame statistics at train start.
    """

    obs_mean: torch.Tensor
    obs_std: torch.Tensor

    def __init__(self, cfg: StateMLPConfig) -> None:
        """Build the MLP.

        Args:
            cfg: Architecture.
        """
        super().__init__()
        self.cfg = cfg
        self.mlp = mlp((cfg.obs_dim, *cfg.hidden, cfg.out), final_activation=False)
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))

    def set_normaliser(self, mean: NDArray[np.float32], std: NDArray[np.float32]) -> None:
        """Write the fixed normaliser (called once, at train start).

        Args:
            mean: ``float32[obs_dim]``.
            std: ``float32[obs_dim]``, every entry > 0.

        Raises:
            ValueError: On a wrong shape or a non-positive std.
        """
        m = torch.as_tensor(np.asarray(mean, np.float32))
        s = torch.as_tensor(np.asarray(std, np.float32))
        if m.shape != self.obs_mean.shape or s.shape != self.obs_std.shape:
            raise ValueError(f"normaliser shapes {m.shape}, {s.shape} != {self.obs_mean.shape}")
        if not torch.all(s > 0):
            raise ValueError("std must be positive")
        with torch.no_grad():
            self.obs_mean.copy_(m)
            self.obs_std.copy_(s)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict actions.

        Args:
            obs: ``float32[B, obs_dim]``, P8's observation in P8's field order.

        Returns:
            ``float32[B, 3]`` in ``[-1, 1]`` (``tanh``).

        Raises:
            ValueError: On a wrong shape.
        """
        if obs.ndim != 2 or obs.shape[1] != self.cfg.obs_dim:
            raise ValueError(f"obs must be [B, {self.cfg.obs_dim}], got {tuple(obs.shape)}")
        x = (obs.to(torch.float32) - self.obs_mean) / self.obs_std
        return torch.tanh(self.mlp(x))


class StateMLPStudent:
    """P8 ``EvalPolicy`` around a :class:`StateMLP`: ``reset(seed, context)``, ``act(obs)``.

    It deliberately has **no** ``act_vision``: the runner flies it on P8's observation path,
    never as a vision student. It never receives a privileged context or motion feed (it is
    not ``privileged`` in P8's sense: its input is the observation, as for the PPO teacher).

    Attributes:
        net: The network, in eval mode.
        device: Where inference runs.
        privileged: False (P8's spec flag: no ``PrivilegedContext``).
        needs_motion_feed: False.
    """

    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(self, net: StateMLP, device: torch.device | str = "cpu") -> None:
        """Wrap a network for closed-loop evaluation.

        Args:
            net: The network (moved to ``device`` and put in eval mode).
            device: Torch device.
        """
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    def reset(
        self, seed: int, context: object | None = None, motion_feed: object | None = None
    ) -> None:
        """Start an episode (the MLP keeps no state).

        Args:
            seed: The episode seed (unused).
            context: Must be ``None``.
            motion_feed: Must be ``None``.

        Raises:
            ValueError: If a context or feed is passed.
        """
        del seed
        if context is not None or motion_feed is not None:
            raise ValueError("bc_state never receives a privileged context or motion feed")

    def act(self, obs: NDArray[Any]) -> NDArray[np.float32]:
        """Return the action for one P8 observation.

        Args:
            obs: ``float[obs_dim]`` (25-d), P8's observation.

        Returns:
            ``float32[3]`` in ``[-1, 1]``.

        Raises:
            ValueError: On a wrong shape.
        """
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action: NDArray[np.float32] = self.act_batch(x)[0]
        return action

    def act_batch(self, obs: NDArray[Any]) -> NDArray[np.float32]:
        """Return actions for ``float[B, obs_dim]`` observations.

        Args:
            obs: ``float[B, obs_dim]``.

        Returns:
            ``float32[B, 3]`` in ``[-1, 1]``.
        """
        x = torch.from_numpy(np.array(obs, dtype=np.float32, order="C")).to(self.device)
        with torch.inference_mode():
            out = self.net(x)
        actions: NDArray[np.float32] = out.cpu().numpy().astype(np.float32)
        return actions
