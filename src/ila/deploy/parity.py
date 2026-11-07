"""Open-loop numeric parity of the exported graphs, per provider (P3-D1 §6 H5(a), §7; P9-D1).

Inputs: the 1 000 fixed ``select`` windows (:mod:`ila.deploy.inputs`). TF32 off everywhere.

**Reference** (:func:`reference`): torch fp32 on the CPU, batch 1, one intra-op thread, the
pinned EMA networks (:func:`ila.deploy.export.load_arm`):

* ``bc_dagger``: the action of each input;
* ``dp``: per input ``i``, initial noise :func:`ila.policies.dp.policy.initial_noise` keyed
  ``(run_seed, noise_episode_key, i)`` (``noise_episode_key`` = -100: never a P8 episode seed),
  then ``ila``'s sampler (:func:`ila.policies.dp.policy.ddim_sample`, DDIM trailing, K = 10,
  ``k = 99, 89, ..., 9``), recording at each step ``s`` the sample ``x_s`` the U-Net received and
  its epsilon-hat; and the final chunk.

**Per provider, at batch 1** (:func:`provider_parity`):

* *judged* ``bc_dagger``: ``max_i |action_p - action_ref|`` (all 3 axes) ``<= 1e-4``
  (P3-D1 §7);
* *judged* ``dp``: per denoising step ``s``, ``max_i |eps_p(x_s^ref, k_s, cond_p) - eps_ref|``
  ``<= 1e-4`` at every one of the K = 10 steps, end to end: ``cond_p`` is the provider's own
  encoder output, ``eps_p`` its own denoise-step network, fed the reference path's ``x_s``
  (P3-D1 §6 H5(a), as R1's check, P6-D3);
* *reported, not judged*: the final-chunk max-abs of the provider's unrolled K = 10 network
  (``dp_K10.onnx`` / :class:`ila.deploy.export.UnrolledDDIM`) and of ``ila``'s loop around its
  denoise step, beside 1e-3 (the first trailing step's gain from epsilon to ``x_0`` is about
  2 029, P3-D1 §8); ``global_cond`` max-abs; epsilon-hat with the reference ``global_cond``
  (isolating the denoise-step network); the count of inputs whose chunk gap exceeds 1e-3.

fp16 providers are judged by the same tests (P3-D1 §6: "fp16 passes only under the same
tests"). Graph files are re-hashed against ``meta.json`` before any session is built.

Outputs: ``results/latency/parity.csv`` (one row per arm x provider), ``parity_steps.csv``
(per-step epsilon max-abs), ``parity.json`` (everything, with hashes and provenance). Actions
are normalised world-frame velocity setpoints (x v_max = 1.5 m/s model scale). Simulation only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila.deploy.backends import Backend
from ila.deploy.export import BCGraph, EncoderGraph, LoadedArm
from ila.deploy.inputs import SelectInputs
from ila.deploy.settings import BenchSettings
from ila.policies.bc import BCNet
from ila.policies.dp.policy import DiffusionPolicyNet, ddim_sample, initial_noise

__all__ = [
    "PARITY_COLUMNS",
    "STEP_COLUMNS",
    "Reference",
    "parity_csv",
    "provider_parity",
    "reference",
    "steps_csv",
]

#: Columns of ``parity.csv``.
PARITY_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "provider",
    "scored",
    "precision",
    "n_inputs",
    "status",
    "judged_quantity",
    "judged_max_abs",
    "tolerance",
    "open_loop_pass",
    "chunk_unrolled_max_abs",
    "chunk_loop_max_abs",
    "chunk_reported_bound",
    "n_chunk_over_bound",
    "global_cond_max_abs",
    "eps_reference_cond_max_abs",
    "realized",
)

#: Columns of ``parity_steps.csv``.
STEP_COLUMNS: Final[tuple[str, ...]] = (
    "provider",
    "step",
    "timestep",
    "eps_max_abs",
    "eps_p99_abs",
)


class Reference:
    """Torch fp32 CPU batch-1 reference outputs (module docstring).

    Attributes:
        action: ``float32[n, 3]`` (``bc_dagger``).
        cond: ``float32[n, 256]`` (``dp`` ``global_cond``).
        noise: ``float32[n, Tp, 3]`` initial noise.
        x: ``float32[K, n, Tp, 3]`` sample fed to the U-Net at each step.
        eps: ``float32[K, n, Tp, 3]`` epsilon-hat at each step.
        timesteps: ``int64[K]``.
        chunk: ``float32[n, Tp, 3]``.
    """

    def __init__(self, arrays: Mapping[str, NDArray[Any]]) -> None:
        """Hold the arrays (keys as the attributes)."""
        self.action = np.asarray(arrays["action"], np.float32)
        self.cond = np.asarray(arrays["cond"], np.float32)
        self.noise = np.asarray(arrays["noise"], np.float32)
        self.x = np.asarray(arrays["x"], np.float32)
        self.eps = np.asarray(arrays["eps"], np.float32)
        self.timesteps = np.asarray(arrays["timesteps"], np.int64)
        self.chunk = np.asarray(arrays["chunk"], np.float32)

    def arrays(self) -> dict[str, NDArray[Any]]:
        """The arrays, for ``np.savez``."""
        return {
            "action": self.action,
            "cond": self.cond,
            "noise": self.noise,
            "x": self.x,
            "eps": self.eps,
            "timesteps": self.timesteps,
            "chunk": self.chunk,
        }

    @property
    def n(self) -> int:
        """Number of inputs."""
        return int(self.cond.shape[0])


def parity_noise(run_seed: int, key: int, n: int, horizon: int, action_dim: int) -> torch.Tensor:
    """Initial noise of input ``i``, keyed ``(run_seed, key, i)`` (P2-D3 item 4's generator)."""
    return initial_noise([(run_seed, key, i) for i in range(n)], horizon, action_dim)


def reference(
    arms: Mapping[str, LoadedArm],
    inputs: SelectInputs,
    settings: BenchSettings,
    *,
    log: Callable[[str], None] | None = None,
) -> Reference:
    """Compute the torch fp32 CPU batch-1 reference on every input (module docstring)."""
    torch.set_num_threads(1)
    bc, dp = arms["bc_dagger"].net, arms["dp"].net
    assert isinstance(bc, BCNet) and isinstance(dp, DiffusionPolicyNet)
    bc_g = BCGraph(bc.cpu()).eval()
    enc_g = EncoderGraph(dp.cpu()).eval()
    n = inputs.n
    tp, a = dp.cfg.horizon, dp.cfg.action_dim
    noise = parity_noise(arms["dp"].seed, settings.noise_episode_key, n, tp, a)
    sched = dp.ddim(settings.dp_k)
    k = len(sched.timesteps)
    action = np.zeros((n, a), np.float32)
    cond = np.zeros((n, dp.encoder.cfg.feature_dim), np.float32)
    xs = np.zeros((k, n, tp, a), np.float32)
    eps = np.zeros_like(xs)
    chunk = np.zeros((n, tp, a), np.float32)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(n):
            img = torch.from_numpy(inputs.images[i : i + 1].copy())
            prop = torch.from_numpy(inputs.proprio[i : i + 1].copy())
            action[i] = bc_g(img, prop)[0].numpy()
            c = enc_g(img, prop)
            cond[i] = c[0].numpy()
            calls = iter(range(k))

            def eps_fn(
                x: torch.Tensor, kk: torch.Tensor, i: int = i, c: torch.Tensor = c, it: Any = calls
            ) -> torch.Tensor:
                s = next(it)
                e = dp.predict_noise(x, kk, c)
                xs[s, i] = x[0].numpy()
                eps[s, i] = e[0].numpy()
                return e

            chunk[i] = ddim_sample(eps_fn, noise[i : i + 1], sched)[0].numpy()
            if log is not None and (i + 1) % 100 == 0:
                log(f"reference {i + 1}/{n} ({time.perf_counter() - t0:.0f} s)")
    return Reference(
        {
            "action": action,
            "cond": cond,
            "noise": noise.numpy(),
            "x": xs,
            "eps": eps,
            "timesteps": np.array([int(t) for t in sched.timesteps], np.int64),
            "chunk": chunk,
        }
    )


def _maxabs(a: NDArray[Any], b: NDArray[Any]) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def provider_parity(
    backend: Backend,
    inputs: SelectInputs,
    ref: Reference,
    settings: BenchSettings,
    *,
    arms: Sequence[str] = ("bc_dagger", "dp"),
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compare one provider with the reference at batch 1 (module docstring).

    Returns:
        ``{"bc_dagger": {...}, "dp": {...}, "provider", "realized", "wall_s"}``; each arm's
        ``open_loop_pass`` is its verdict.
    """
    t0 = time.perf_counter()
    n = ref.n
    out: dict[str, Any] = {"provider": backend.provider}
    if "bc_dagger" in arms:
        act = np.stack(
            [backend.bc(inputs.images[i : i + 1], inputs.proprio[i : i + 1])[0] for i in range(n)]
        )
        gap = _maxabs(act, ref.action)
        out["bc_dagger"] = {
            "judged_quantity": "action max-abs",
            "judged_max_abs": gap,
            "tolerance": settings.action_max_abs,
            "open_loop_pass": bool(gap <= settings.action_max_abs),
            "per_axis_max_abs": [_maxabs(act[:, j], ref.action[:, j]) for j in range(act.shape[1])],
            "p99_abs": float(np.quantile(np.abs(act.astype(np.float64) - ref.action), 0.99)),
        }
        if log is not None:
            log(f"{backend.provider} bc_dagger action max-abs {gap:.3g}")
    if "dp" in arms:
        k = ref.timesteps.shape[0]
        cond = np.zeros_like(ref.cond)
        eps = np.zeros_like(ref.eps)
        eps_rc = np.zeros_like(ref.eps)
        chunk_u = np.zeros_like(ref.chunk)
        chunk_l = np.zeros_like(ref.chunk)
        for i in range(n):
            c = backend.encode(inputs.images[i : i + 1], inputs.proprio[i : i + 1])
            cond[i] = c[0]
            for s in range(k):
                tt = np.full((1,), int(ref.timesteps[s]), np.int64)
                x = ref.x[s, i : i + 1]
                eps[s, i] = backend.denoise(x, tt, c)[0]
                eps_rc[s, i] = backend.denoise(x, tt, ref.cond[i : i + 1])[0]
            chunk_u[i] = backend.unrolled(c, ref.noise[i : i + 1])[0]
            chunk_l[i] = backend.loop(c, ref.noise[i : i + 1], settings.dp_k)[0]
            if log is not None and (i + 1) % 250 == 0:
                log(f"{backend.provider} dp {i + 1}/{n} ({time.perf_counter() - t0:.0f} s)")
        per_step = [_maxabs(eps[s], ref.eps[s]) for s in range(k)]
        per_step_p99 = [
            float(np.quantile(np.abs(eps[s].astype(np.float64) - ref.eps[s]), 0.99))
            for s in range(k)
        ]
        gap = max(per_step)
        chunk_gap = np.abs(chunk_u.astype(np.float64) - ref.chunk).reshape(n, -1).max(axis=1)
        out["dp"] = {
            "judged_quantity": "per-step eps-hat max-abs (K = 10 steps, own encoder)",
            "judged_max_abs": gap,
            "tolerance": settings.eps_max_abs,
            "open_loop_pass": bool(gap <= settings.eps_max_abs),
            "eps_per_step_max_abs": per_step,
            "eps_per_step_p99_abs": per_step_p99,
            "timesteps": [int(t) for t in ref.timesteps],
            "eps_reference_cond_max_abs": max(_maxabs(eps_rc[s], ref.eps[s]) for s in range(k)),
            "global_cond_max_abs": _maxabs(cond, ref.cond),
            "chunk_unrolled_max_abs": _maxabs(chunk_u, ref.chunk),
            "chunk_loop_max_abs": _maxabs(chunk_l, ref.chunk),
            "chunk_reported_bound": settings.chunk_reported,
            "n_chunk_over_bound": int(np.sum(chunk_gap > settings.chunk_reported)),
            "chunk_unrolled_p50_p99_abs": [
                float(np.quantile(chunk_gap, 0.5)),
                float(np.quantile(chunk_gap, 0.99)),
            ],
        }
        if log is not None:
            log(
                f"{backend.provider} dp eps max-abs {gap:.3g}; "
                f"chunk {out['dp']['chunk_unrolled_max_abs']:.3g}"
            )
    out["realized"] = backend.realized()
    out["wall_s"] = time.perf_counter() - t0
    return out


def _fmt(x: Any) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def parity_csv(records: Mapping[str, Mapping[str, Any]], settings: BenchSettings) -> str:
    """Render ``parity.csv`` from ``{provider: record}`` (refused / failed rows have no number).

    A record may be ``{"status": "failed: ..."}`` (the provider could not be built or run).
    """
    from ila.deploy.settings import precision_of

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(PARITY_COLUMNS)
    for provider in settings.providers:
        if provider == settings.reference or provider not in records:
            continue
        rec = records[provider]
        for arm in ("bc_dagger", "dp"):
            a = rec.get(arm)
            status = "measured" if a is not None else str(rec.get("status", "missing"))
            realized = ";".join(
                f"{g}={'|'.join(p)}" for g, p in sorted(rec.get("realized", {}).items())
            )
            if a is None:
                w.writerow(
                    [
                        arm,
                        provider,
                        settings.is_scored(provider),
                        precision_of(provider),
                        settings.n_inputs,
                        status,
                        "",
                        "",
                        "",
                        False,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        realized,
                    ]
                )
                continue
            w.writerow(
                [
                    arm,
                    provider,
                    settings.is_scored(provider),
                    precision_of(provider),
                    settings.n_inputs,
                    status,
                    a.get("judged_quantity", ""),
                    _fmt(a.get("judged_max_abs", "")),
                    _fmt(a.get("tolerance", "")),
                    bool(a.get("open_loop_pass", False)),
                    _fmt(a.get("chunk_unrolled_max_abs", "")),
                    _fmt(a.get("chunk_loop_max_abs", "")),
                    _fmt(a.get("chunk_reported_bound", "")),
                    _fmt(a.get("n_chunk_over_bound", "")),
                    _fmt(a.get("global_cond_max_abs", "")),
                    _fmt(a.get("eps_reference_cond_max_abs", "")),
                    realized,
                ]
            )
    return buf.getvalue()


def steps_csv(records: Mapping[str, Mapping[str, Any]], settings: BenchSettings) -> str:
    """Render ``parity_steps.csv`` (``dp`` per-step epsilon-hat max-abs and p99)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(STEP_COLUMNS)
    for provider in settings.providers:
        dp = records.get(provider, {}).get("dp")
        if dp is None or "eps_per_step_max_abs" not in dp:
            continue
        for s, (t, m, p) in enumerate(
            zip(
                dp["timesteps"], dp["eps_per_step_max_abs"], dp["eps_per_step_p99_abs"], strict=True
            )
        ):
            w.writerow([provider, s, t, _fmt(m), _fmt(p)])
    return buf.getvalue()


def array_sha256(arrays: Mapping[str, NDArray[Any]]) -> str:
    """SHA-256 over the arrays' names, dtypes, shapes and bytes (sorted by name)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()
