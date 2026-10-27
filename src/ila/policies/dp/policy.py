"""Diffusion Policy student: conditioning, DDIM sampling, chunk execution.

Model (``configs/train/dp.yaml``, plan §1.6): the shared
:class:`ila.policies.encoder.VisionEncoder` gives a 256-d observation feature (``To = 2``),
which globally conditions a :class:`ila.policies.dp.unet1d.ConditionalUnet1D` (FiLM) that
predicts epsilon over an action chunk ``(Tp, 3) = (16, 3)``. Actions are normalised
world-frame velocity setpoints in ``[-1, 1]`` (identity normalisation); the env scales by
v_max = 1.5 m/s model scale and applies its norm cap. One control step = 1/30 s model scale
(lambda = 1/25), so ``Ta = 8`` is 0.267 s model (1.33 s full scale) flown open loop.

Sampling: diffusers' ``DDIMScheduler`` with the training schedule's betas
(``squaredcos_cap_v2``, 100 train steps, epsilon prediction, ``clip_sample``), eta = 0, ``K``
inference steps chosen at inference time (``1 <= K <= 100``). The final sample is clipped to
``[-1, 1]``.

Randomness: with eta = 0 the only randomness is the initial noise ``x_K ~ N(0, I)``. It is an
explicit input of :meth:`DiffusionPolicyNet.sample` (so a graph export can take it as an input),
drawn by :func:`initial_noise` on the CPU from one ``torch.Generator`` **per slot**, seeded by
:func:`noise_seed` ``(run_seed, episode_seed, step)``. A slot's noise therefore depends only on
its own seed tuple, never on its batch position, the batch size, the device or a global RNG.

Chunk indices come only from :mod:`ila.policies.dp.chunking`. No training code lives here.
"""

from __future__ import annotations

import hashlib
import struct
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import torch
from diffusers import DDIMScheduler as _DiffusersDDIM
from numpy.typing import NDArray
from torch import nn

from ila.policies.dp import chunking
from ila.policies.dp.unet1d import ConditionalUnet1D, UNetConfig
from ila.policies.encoder import EncoderConfig, VisionEncoder, VisionStudentBase

__all__ = [
    "DPConfig",
    "DPStudent",
    "DiffusionPolicyNet",
    "SeedTuple",
    "initial_noise",
    "noise_seed",
]

#: ``(run_seed, episode_seed, step)``: the key of one replanning call's noise.
SeedTuple = tuple[int, int, int]

_SEED_STRUCT: Final[struct.Struct] = struct.Struct("<qqq")

#: diffusers' DDIM scheduler (diffusers is only partially typed, so it is held as ``Any``).
_DDIM: Any = _DiffusersDDIM


def noise_seed(run_seed: int, episode_seed: int, step: int) -> int:
    """Return the 64-bit generator seed for one replanning call (``hash64`` in il-conventions).

    BLAKE2b (8-byte digest) of the three values packed as little-endian signed int64, read as
    an unsigned little-endian integer. Stable across processes, platforms and Python versions
    (unlike ``hash()``).

    Args:
        run_seed: The training run's seed.
        episode_seed: P8's episode seed.
        step: The control step at which the chunk is planned (>= 0).

    Returns:
        An integer in ``[0, 2**64)`` for ``torch.Generator.manual_seed``.

    Raises:
        ValueError: If ``step < 0``.
        struct.error: If a value does not fit in a signed int64.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    digest = hashlib.blake2b(
        _SEED_STRUCT.pack(int(run_seed), int(episode_seed), int(step)), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def initial_noise(seeds: Sequence[SeedTuple], horizon: int, action_dim: int) -> torch.Tensor:
    """Draw the DDIM initial noise for each slot from its own CPU generator.

    Args:
        seeds: One ``(run_seed, episode_seed, step)`` per slot.
        horizon: ``Tp``.
        action_dim: ``A``.

    Returns:
        ``float32[B, Tp, A]`` on the CPU; row ``i`` is a function of ``seeds[i]`` only.
    """
    rows = []
    for run_seed, episode_seed, step in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(noise_seed(run_seed, episode_seed, step))
        rows.append(torch.randn((horizon, action_dim), generator=g, dtype=torch.float32))
    if not rows:
        return torch.zeros((0, horizon, action_dim), dtype=torch.float32)
    return torch.stack(rows)


@dataclass(frozen=True)
class DPConfig:
    """Diffusion Policy architecture and sampler (``configs/train/dp.yaml``).

    Attributes:
        n_obs_steps: ``To`` (2).
        horizon: ``Tp`` (16), control steps.
        n_action_steps: Default ``Ta`` (8); an inference-time choice.
        action_dim: 3.
        unet: U-Net architecture.
        num_train_timesteps: DDPM training steps (100).
        beta_schedule: ``"squaredcos_cap_v2"``.
        prediction_type: ``"epsilon"``.
        clip_sample: Clip the predicted ``x_0`` to ``[-1, 1]`` inside each DDIM step.
        num_inference_steps: Default ``K`` (10); an inference-time choice.
        timestep_spacing: diffusers DDIM spacing. Not set by the config; ``"leading"`` is
            diffusers' default and the reference's behaviour (see the module notes in the
            hand-off: with it, ``K = 1`` denoises only at ``k = 0``).
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    action_dim: int = 3
    unet: UNetConfig = field(default_factory=UNetConfig)
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    num_inference_steps: int = 10
    timestep_spacing: str = "leading"

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: On an inadmissible ``(To, Tp, Ta)``, ``K`` out of range, a horizon
                the U-Net cannot halve, or an unsupported prediction type.
        """
        chunking.ChunkSpec(self.n_obs_steps, self.horizon, self.n_action_steps)
        if self.horizon % self.unet.downsample_factor:
            raise ValueError(
                f"Tp={self.horizon} must be a multiple of {self.unet.downsample_factor}"
            )
        if self.prediction_type != "epsilon":
            raise ValueError(f"only epsilon prediction is implemented: {self.prediction_type}")
        _check_k(self.num_inference_steps, self.num_train_timesteps)

    @classmethod
    def from_mapping(cls, train_cfg: Mapping[str, Any]) -> DPConfig:
        """Build from a parsed ``configs/train/dp*.yaml``.

        Args:
            train_cfg: Keys ``horizons``, ``unet``, ``noise_scheduler``, ``inference``.

        Returns:
            The config.

        Raises:
            ValueError: If the sampler is not DDIM or the noise is not per-episode.
        """
        hz = train_cfg["horizons"]
        ns = train_cfg["noise_scheduler"]
        inf = train_cfg["inference"]
        if inf["sampler"] != "ddim" or inf["noise"] != "per_episode_generator":
            raise ValueError(f"unsupported inference block: {dict(inf)}")
        if ns["train"] != "ddpm":
            raise ValueError(f"unsupported training scheduler: {ns['train']!r}")
        return cls(
            n_obs_steps=int(hz["n_obs_steps"]),
            horizon=int(hz["horizon"]),
            n_action_steps=int(hz["n_action_steps"]),
            unet=UNetConfig.from_mapping(train_cfg["unet"]),
            num_train_timesteps=int(ns["num_train_timesteps"]),
            beta_schedule=str(ns["beta_schedule"]),
            prediction_type=str(ns["prediction_type"]),
            clip_sample=bool(ns["clip_sample"]),
            num_inference_steps=int(inf["num_inference_steps"]),
            timestep_spacing=str(inf.get("timestep_spacing", "leading")),
        )

    @property
    def chunk(self) -> chunking.ChunkSpec:
        """The default ``(To, Tp, Ta)``."""
        return chunking.ChunkSpec(self.n_obs_steps, self.horizon, self.n_action_steps)


def _check_k(k: int, num_train_timesteps: int) -> None:
    if not 1 <= k <= num_train_timesteps:
        raise ValueError(f"K must be in [1, {num_train_timesteps}] (train steps), got {k}")


class DiffusionPolicyNet(nn.Module):
    """Encoder + noise-prediction U-Net, with a DDIM sampler taking explicit noise."""

    def __init__(self, encoder_cfg: EncoderConfig, cfg: DPConfig) -> None:
        """Build the networks.

        Args:
            encoder_cfg: Shared encoder architecture (its ``n_obs_steps`` must equal ``To``).
            cfg: Diffusion Policy config.

        Raises:
            ValueError: If the encoder's ``To`` differs from the policy's.
        """
        super().__init__()
        if encoder_cfg.n_obs_steps != cfg.n_obs_steps:
            raise ValueError(
                f"encoder To={encoder_cfg.n_obs_steps} differs from DP To={cfg.n_obs_steps}"
            )
        self.cfg = cfg
        self.encoder = VisionEncoder(encoder_cfg)
        self.unet = ConditionalUnet1D(cfg.action_dim, encoder_cfg.feature_dim, cfg.unet)
        self._ddim: dict[int, Any] = {}

    def ddim(self, num_inference_steps: int) -> Any:
        """Return a DDIM scheduler with ``set_timesteps(K)`` applied (cached per ``K``).

        Args:
            num_inference_steps: ``K``.

        Returns:
            The scheduler (betas from the training schedule).
        """
        k = int(num_inference_steps)
        _check_k(k, self.cfg.num_train_timesteps)
        if k not in self._ddim:
            sched = _DDIM(
                num_train_timesteps=self.cfg.num_train_timesteps,
                beta_schedule=self.cfg.beta_schedule,
                clip_sample=self.cfg.clip_sample,
                prediction_type=self.cfg.prediction_type,
                set_alpha_to_one=True,
                steps_offset=0,
                timestep_spacing=self.cfg.timestep_spacing,
            )
            sched.set_timesteps(k)
            self._ddim[k] = sched
        return self._ddim[k]

    def predict_noise(
        self, sample: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Return epsilon-hat for ``(B, Tp, A)`` noisy actions at diffusion step(s) ``k``."""
        eps: torch.Tensor = self.unet(sample, timestep, cond)
        return eps

    def sample(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        noise: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Denoise ``noise`` into an action chunk (DDIM, eta = 0, deterministic given noise).

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, D]``.
            noise: ``float32[B, Tp, A]`` initial noise (:func:`initial_noise`).
            num_inference_steps: ``K``.

        Returns:
            ``float32[B, Tp, A]`` chunk, chunk index ``j`` = step ``t-To+1+j``, clipped to
            ``[-1, 1]``.

        Raises:
            ValueError: If ``noise`` has the wrong shape.
        """
        b = images.shape[0]
        if noise.shape != (b, self.cfg.horizon, self.cfg.action_dim):
            raise ValueError(
                f"noise must be [{b}, {self.cfg.horizon}, {self.cfg.action_dim}], "
                f"got {tuple(noise.shape)}"
            )
        sched = self.ddim(num_inference_steps)
        cond = self.encoder(images, proprio)
        x = noise.to(device=cond.device, dtype=torch.float32)
        for k in sched.timesteps:
            kk = torch.full((b,), int(k), dtype=torch.long, device=cond.device)
            eps = self.predict_noise(x, kk, cond)
            x = sched.step(eps, int(k), x, eta=0.0).prev_sample
        return x.clamp(-1.0, 1.0)


class DPStudent(VisionStudentBase):
    """P8 ``EvalPolicy`` shape around a :class:`DiffusionPolicyNet`, executing chunks.

    Per episode (after ``reset(seed)``): at step ``t`` (the number of ``act_vision`` calls since
    the reset), if :func:`chunking.is_replan_step` ``(t, Ta)``, a chunk is sampled with noise
    keyed ``(run_seed, seed, t)`` and its executed indices (:func:`chunking.executed_slice`,
    ``To-1 ... To-2+Ta``) are queued; each step pops one action.

    Attributes:
        net: The network (eval mode).
        cfg: Encoder config.
        dp_cfg: Diffusion Policy config.
        run_seed: The run's seed (first element of every noise key).
        chunk_spec: ``(To, Tp, Ta)`` in use (``Ta`` chosen at construction).
        num_inference_steps: ``K`` in use.
        device: Where inference runs.
    """

    def __init__(
        self,
        net: DiffusionPolicyNet,
        run_seed: int,
        device: torch.device | str = "cpu",
        *,
        n_action_steps: int | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        """Wrap a network for closed-loop evaluation.

        Args:
            net: The network (moved to ``device``, eval mode).
            run_seed: The run's seed.
            device: Torch device.
            n_action_steps: ``Ta``; the config's default (8) when ``None``.
            num_inference_steps: ``K``; the config's default (10) when ``None``.
        """
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cfg = net.encoder.cfg
        self.dp_cfg = net.cfg
        self.run_seed = int(run_seed)
        ta = self.dp_cfg.n_action_steps if n_action_steps is None else int(n_action_steps)
        self.chunk_spec = self.dp_cfg.chunk.with_action_steps(ta)
        k = self.dp_cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        _check_k(int(k), self.dp_cfg.num_train_timesteps)
        self.num_inference_steps = int(k)
        self._episode_seed: int | None = None
        self._t = 0
        self._queue: deque[NDArray[np.float32]] = deque()

    def _reset_episode(self, seed: int) -> None:
        """Start an episode: step counter 0, empty chunk queue."""
        self._episode_seed = seed
        self._t = 0
        self._queue.clear()

    def sample_batch(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        seeds: Sequence[SeedTuple],
        num_inference_steps: int | None = None,
    ) -> NDArray[np.float32]:
        """Batched entry point for a GPU server: one full chunk per slot.

        Slot ``i``'s noise is drawn from its own generator keyed ``seeds[i]``; the caller
        executes ``chunk[:, chunking.executed_slice(To, Tp, Ta)]``.

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, D]``.
            seeds: ``B`` tuples ``(run_seed, episode_seed, step)``.
            num_inference_steps: ``K``; this student's ``K`` when ``None``.

        Returns:
            ``float32[B, Tp, A]`` in ``[-1, 1]``.

        Raises:
            ValueError: If ``len(seeds) != B``.
        """
        img = np.asarray(images)
        if len(seeds) != img.shape[0]:
            raise ValueError(f"{len(seeds)} seed tuples for a batch of {img.shape[0]}")
        k = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        noise = initial_noise(seeds, self.dp_cfg.horizon, self.dp_cfg.action_dim)
        img_t, prop_t = self._to_device(img, np.asarray(proprio, np.float32))
        with torch.inference_mode():
            out = self.net.sample(img_t, prop_t, noise, k)
        chunks: NDArray[np.float32] = out.cpu().numpy().astype(np.float32)
        return chunks

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return the action for step ``t`` of the current episode (CPU single-sample path).

        Args:
            image_stack: ``uint8[To, 96, 96, 3]``, steps ``t-To+1 ... t`` oldest first.
            proprio: ``float32[To, D]``.

        Returns:
            ``float32[3]`` in ``[-1, 1]``.

        Raises:
            RuntimeError: If ``reset`` has not been called.
        """
        if self._episode_seed is None:
            raise RuntimeError("call reset(seed) before act_vision")
        img, prop = self._check_window(image_stack, proprio)
        if chunking.is_replan_step(self._t, self.chunk_spec.n_action_steps):
            chunk = self.sample_batch(
                img[None], prop[None], [(self.run_seed, self._episode_seed, self._t)]
            )[0]
            self._queue.clear()
            self._queue.extend(chunk[self.chunk_spec.executed])
        self._t += 1
        return self._queue.popleft().copy()
