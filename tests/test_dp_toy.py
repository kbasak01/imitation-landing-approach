"""The 1-D two-Gaussian toy (plan Phase 6 item 2; diffusion-policy-recipe pitfall 5).

Actions are 1-D, in normalised units, drawn from an equal mixture of N(-0.6, 0.08^2) and
N(+0.6, 0.08^2) under one constant observation (so the conditioning is dropped). A tiny MLP
epsilon-network is trained with the **same scheduler config as ``dp``**: built from the frozen
``configs/train/dp.yaml`` through :class:`DPConfig` (100 DDPM steps, ``squaredcos_cap_v2``,
epsilon prediction, ``clip_sample``; DDIM with ``trailing`` spacing) by
:func:`ddpm_scheduler`/:func:`ddim_scheduler`, sampled by the same :func:`ddim_sample` loop the
policy runs. Asserted:

(a) DDIM at ``K = 100`` (= the train steps) matches DDPM ancestral sampling
    (:func:`ddpm_sample`, explicit per-step noise) in distribution: mode fractions and
    per-mode means agree within tolerance. A sampler with timesteps in the wrong order or K
    above the train steps fails here (pitfall 5);
(b) Diffusion sampling recovers both modes (each holds 30-70 % of samples; few in between),
    at the deployed ``K = 10`` and at ``K = 100``;
(c) MSE regression (BC) on the same data collapses to the mean, which lies in the gap
    between the modes where the diffusion samples almost never land: H2's mechanism in
    miniature.

Plus: :func:`ddpm_sample` is diffusers' ``DDPMScheduler.step`` loop with its noise supplied
instead of drawn. Every random draw comes from an explicit ``torch.Generator``; the global RNG
is only used (forked) to initialise the MLPs. CPU, a few seconds.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from ila.policies import registry
from ila.policies.dp.policy import (
    DPConfig,
    ddim_sample,
    ddim_scheduler,
    ddpm_sample,
    ddpm_scheduler,
)

MODES = (-0.6, 0.6)
SIGMA = 0.08
N_DATA = 4096
N_SAMPLES = 2000
#: |a| below this is "between the modes" (more than 3.75 sigma from either mode).
GAP = 0.3


def _cfg() -> DPConfig:
    cfg = DPConfig.from_mapping(registry.load_train_config("dp"))
    # The frozen sampler config (P3-D1 §3), pinned so the toy tests what dp runs.
    assert (cfg.num_train_timesteps, cfg.beta_schedule, cfg.prediction_type) == (
        100,
        "squaredcos_cap_v2",
        "epsilon",
    )
    assert cfg.clip_sample and cfg.timestep_spacing == "trailing"
    return cfg


def _data(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mode = torch.randint(0, 2, (n, 1), generator=g)
    centre = torch.where(mode == 0, MODES[0], MODES[1])
    return (centre + SIGMA * torch.randn((n, 1), generator=g)).clamp(-1.0, 1.0)


class _EpsMLP(nn.Module):
    """epsilon_hat(x_k, k) for 1-D x: sinusoidal k embedding (16) || x -> 3 x 128 Mish -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.freqs = torch.exp(-math.log(1000.0) * torch.arange(8) / 8)
        self.net = nn.Sequential(
            nn.Linear(17, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        arg = k.float()[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        out: torch.Tensor = self.net(torch.cat([x, emb], dim=-1))
        return out


@pytest.fixture(scope="module")
def trained() -> tuple[DPConfig, _EpsMLP, torch.Tensor]:
    return train_toy()


def train_toy() -> tuple[DPConfig, _EpsMLP, torch.Tensor]:
    """Train the epsilon MLP: 3 000 Adam steps, batch 256, k ~ U{0..99}, eps ~ N(0, 1)."""
    cfg = _cfg()
    data = _data(N_DATA, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = _EpsMLP()
    ddpm = ddpm_scheduler(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps = 3000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        idx = torch.randint(0, N_DATA, (256,), generator=g)
        a = data[idx]
        eps = torch.randn(a.shape, generator=g)
        k = torch.randint(0, cfg.num_train_timesteps, (256,), generator=g)
        loss = ((net(ddpm.add_noise(a, eps, k), k) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return cfg, net, data


def _modes(x: torch.Tensor) -> dict[str, float]:
    v = x.flatten()
    lo, hi = v[v < -GAP], v[v > GAP]
    return {
        "frac_lo": lo.numel() / v.numel(),
        "frac_hi": hi.numel() / v.numel(),
        "frac_gap": float((v.abs() <= GAP).float().mean()),
        "mean_lo": float(lo.mean()),
        "mean_hi": float(hi.mean()),
        "std_lo": float(lo.std()),
        "std_hi": float(hi.std()),
    }


def _ddim(cfg: DPConfig, net: _EpsMLP, k: int, seed: int) -> torch.Tensor:
    noise = torch.randn((N_SAMPLES, 1), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return ddim_sample(net, noise, ddim_scheduler(cfg, k))


def _ddpm(cfg: DPConfig, net: _EpsMLP, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((N_SAMPLES, 1), generator=g)
    step_noise = torch.randn((cfg.num_train_timesteps, N_SAMPLES, 1), generator=g)
    with torch.no_grad():
        return ddpm_sample(net, noise, step_noise, ddpm_scheduler(cfg))


def test_ddim_k100_matches_ddpm_in_distribution(
    trained: tuple[DPConfig, _EpsMLP, torch.Tensor],
) -> None:
    cfg, net, _ = trained
    ddim = _modes(_ddim(cfg, net, 100, 2))
    ddpm = _modes(_ddpm(cfg, net, 3))
    # Mode fractions: binomial sd at n = 2000 is 0.011 per sampler; 0.06 is > 3.8 sd of
    # the difference. Mode means within 0.05 (0.6 sigma of a mode).
    assert abs(ddim["frac_hi"] - ddpm["frac_hi"]) < 0.06, (ddim, ddpm)
    assert abs(ddim["frac_lo"] - ddpm["frac_lo"]) < 0.06, (ddim, ddpm)
    assert abs(ddim["mean_hi"] - ddpm["mean_hi"]) < 0.05, (ddim, ddpm)
    assert abs(ddim["mean_lo"] - ddpm["mean_lo"]) < 0.05, (ddim, ddpm)
    assert ddim["frac_gap"] < 0.05 and ddpm["frac_gap"] < 0.05, (ddim, ddpm)
    # Per-mode spread: both near the data's 0.08, within 0.03 of each other.
    for side in ("lo", "hi"):
        assert abs(ddim[f"std_{side}"] - ddpm[f"std_{side}"]) < 0.03, (ddim, ddpm)
        assert 0.04 < ddim[f"std_{side}"] < 0.13 and 0.04 < ddpm[f"std_{side}"] < 0.13


@pytest.mark.parametrize("k", [10, 100])
def test_diffusion_recovers_both_modes(
    trained: tuple[DPConfig, _EpsMLP, torch.Tensor], k: int
) -> None:
    cfg, net, _ = trained
    m = _modes(_ddim(cfg, net, k, 4))
    assert 0.3 <= m["frac_lo"] <= 0.7 and 0.3 <= m["frac_hi"] <= 0.7, m
    assert m["frac_gap"] < 0.05, m
    assert abs(m["mean_lo"] - MODES[0]) < 0.1 and abs(m["mean_hi"] - MODES[1]) < 0.1, m


def test_mse_bc_collapses_to_the_mean(trained: tuple[DPConfig, _EpsMLP, torch.Tensor]) -> None:
    cfg, net, data = trained
    obs = torch.ones((N_DATA, 4))  # the one constant observation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        bc = nn.Sequential(nn.Linear(4, 64), nn.Mish(), nn.Linear(64, 1), nn.Tanh())
    opt = torch.optim.Adam(bc.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(5)
    for _ in range(500):
        idx = torch.randint(0, N_DATA, (256,), generator=g)
        loss = ((bc(obs[idx]) - data[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = float(bc(obs[:1]))
    assert abs(pred - float(data.mean())) < 0.05, (pred, float(data.mean()))
    assert abs(pred) < GAP  # the BC action lies between the modes ...
    m = _modes(_ddim(cfg, net, 10, 6))
    assert m["frac_gap"] < 0.05  # ... where diffusion puts almost no samples


def test_ddpm_sample_is_diffusers_step_with_supplied_noise(
    trained: tuple[DPConfig, _EpsMLP, torch.Tensor],
) -> None:
    cfg, net, _ = trained
    n = 64
    noise = torch.randn((n, 1), generator=torch.Generator().manual_seed(7))
    # diffusers draws z_k with randn(shape, generator) at every k > 0, in order k = 99 ... 1.
    g = torch.Generator().manual_seed(8)
    sched = ddpm_scheduler(cfg)
    x = noise.clone()
    with torch.no_grad():
        for k in range(cfg.num_train_timesteps - 1, -1, -1):
            eps = net(x, torch.full((n,), k, dtype=torch.long))
            x = sched.step(eps, k, x, generator=g).prev_sample
        reference = x.clamp(-1.0, 1.0)
        g = torch.Generator().manual_seed(8)
        rows = [torch.randn((n, 1), generator=g) for _ in range(cfg.num_train_timesteps - 1)]
        step_noise = torch.stack([*rows, torch.zeros((n, 1))])
        ours = ddpm_sample(net, noise, step_noise, ddpm_scheduler(cfg))
    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="step_noise"):
        ddpm_sample(net, noise, step_noise[:-1], ddpm_scheduler(cfg))


def test_ddim_trailing_k100_runs_every_train_step() -> None:
    cfg = _cfg()
    assert ddim_scheduler(cfg, 100).timesteps.tolist() == list(range(99, -1, -1))
    assert ddim_scheduler(cfg, 10).timesteps.tolist() == list(range(99, 0, -10))
    with pytest.raises(ValueError, match="K must be"):
        ddim_scheduler(cfg, 101)
