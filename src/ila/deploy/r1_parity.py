"""R1 ONNX graphs: loading, ila-side sampling, and numeric parity (Phase 6; ``.venv`` only).

``scripts/r1/export_onnx.py`` (``.venv-lerobot``) writes, per R1 checkpoint at step ``N``,
``<run>/onnx/step<N>/{encoder.onnx, denoiser.onnx, meta.json, reference.npz}`` (format
:data:`FORMAT`). This module reads them with ONNX Runtime on the CPU and is the ``.venv`` side
of the contract:

* ``encoder.onnx``: ``image uint8[B, To, 96, 96, 3]`` (HWC, oldest frame first) and ``state
  float32[B, To, 13]`` (raw student proprio) -> ``global_cond float32[B, C]``; LeRobot's
  ``/255``, MEAN_STD / MIN_MAX normalisation and eval centre crop are inside the graph.
* ``denoiser.onnx``: ``sample float32[B, Tp, 3]`` (normalised actions), ``timestep int64[B]``,
  ``global_cond`` -> ``eps float32[B, Tp, 3]``.
* Sampling stays outside the graphs: :func:`ddim_from_meta` builds ``ila``'s own
  :func:`ila.policies.dp.policy.ddim_scheduler` (DDIM, ``trailing``, ``set_alpha_to_one``,
  eta = 0) from ``meta.json``, and :func:`sample_chunk` runs ``ila``'s
  :func:`~ila.policies.dp.policy.ddim_sample` (final clip to [-1, 1]) with the ORT denoiser.
  The initial noise is an input (``ila``'s per-episode generator); no graph holds an RNG op.
* :func:`unnormalise` maps a normalised chunk to the env's action units with LeRobot's
  MIN_MAX inverse, in float32, exactly as the checkpoint's postprocessor does
  (``(x + 1) / 2 * d + min``, ``d = max - min``, ``eps`` where ``d == 0``). Unlike ``dp``'s
  identity normalisation, R1's executable range per axis is the training labels' [min, max].

:func:`run_parity` compares ORT CPU with the torch fp32 CPU batch-1 reference in
``reference.npz`` and writes ``parity.json`` beside the graphs. Judged (all must hold):

* both graphs' SHA-256 equal ``meta.json``'s; opset 17 only; IR <= 10; no RNG op;
* **per-step epsilon max-abs <= 1e-4** (P3-D1 §6 H5(a)'s open-loop criterion), end to end
  (ORT encoder -> ORT denoiser) on the torch path's own ``x_k``: at every one of the
  ``K = 10`` DDIM-trailing steps and at the fixed probe timesteps {99, 89, 49, 9, 0};
* ``ila``'s sampler replaying the stored epsilons reproduces the reference chunk
  (<= 1e-6; proves the export script's sampler reimplementation equals ``ila``'s), and
  :func:`unnormalise` reproduces LeRobot's postprocessor (<= 1e-6);
* the export's LeRobot-path check (LeRobot's own ``select_action`` vs the reference chunk's
  executed slice) passed its tolerance.

Reported, not judged: the free-running chunk max-abs (the first trailing step's gain from
epsilon to ``x_0`` is about 2 029, P3-D1 §8, so ~1e-6 epsilon gaps become ~1e-3 chunk gaps on
a trained net), the ORT executed slice vs LeRobot's own ``select_action``, ``global_cond``
max-abs, and batch-size consistency (one ORT call on all inputs vs batch 1).

Actions: world-frame velocity setpoints (normalised by the env's v_max = 1.5 m/s model
scale); one control step = 1/30 s model scale (lambda = 1/25). Simulation only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import torch
from numpy.typing import NDArray

from ila.policies.dp.policy import DPConfig, ddim_sample, ddim_scheduler

__all__ = [
    "EPS_TOL",
    "FORMAT",
    "GRAPHS",
    "MAX_IR_VERSION",
    "OPSET",
    "REPLAY_TOL",
    "RNG_OPS",
    "R1ExportError",
    "cpu_session",
    "ddim_from_meta",
    "denoise",
    "encode",
    "graph_checks",
    "load_meta",
    "rng_ops",
    "run_parity",
    "sample_chunk",
    "unnormalise",
    "verify_files",
]

#: ``meta.json`` format written by ``scripts/r1/export_onnx.py``.
FORMAT: Final[str] = "ila-r1-onnx/1"
#: Per-step epsilon max-abs criterion (P3-D1 §6 H5(a), open loop).
EPS_TOL: Final[float] = 1e-4
#: Sampler / unnormalisation replay tolerance (same arithmetic on both sides).
REPLAY_TOL: Final[float] = 1e-6
#: Opset the graphs must use (Phase 9 rule: opset 17).
OPSET: Final[int] = 17
#: Highest IR version accepted (ORT 1.29 and onnx 1.22 in ``.venv`` both load it).
MAX_IR_VERSION: Final[int] = 10
#: ONNX op types that draw random numbers (or may, in training mode). None may appear.
RNG_OPS: Final[frozenset[str]] = frozenset(
    {
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
        "Bernoulli",
        "Dropout",
    }
)
GRAPHS: Final[tuple[str, str]] = ("encoder.onnx", "denoiser.onnx")


class R1ExportError(RuntimeError):
    """An export directory is incomplete, of another format, or does not match its hashes."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_meta(step_dir: Path) -> dict[str, Any]:
    """Read ``meta.json`` of one export.

    Args:
        step_dir: ``<run>/onnx/step<N>``.

    Returns:
        The parsed record.

    Raises:
        R1ExportError: If it is missing or of another format.
    """
    path = Path(step_dir) / "meta.json"
    if not path.is_file():
        raise R1ExportError(f"{path} does not exist")
    meta: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if meta.get("format") != FORMAT:
        raise R1ExportError(f"{path}: format {meta.get('format')!r} != {FORMAT!r}")
    return meta


def verify_files(step_dir: Path, meta: dict[str, Any], *, reference: bool = True) -> None:
    """Check both graphs (and ``reference.npz``) against the SHA-256 in ``meta.json``.

    Args:
        step_dir: The export directory.
        meta: Its :func:`load_meta` record.
        reference: Also check ``reference.npz``.

    Raises:
        R1ExportError: On a missing file or a hash mismatch.
    """
    want = dict(meta["onnx_sha256"])
    if reference:
        want["reference.npz"] = meta["reference_sha256"]
    for name, sha in want.items():
        path = Path(step_dir) / name
        if not path.is_file():
            raise R1ExportError(f"{path} does not exist")
        got = _sha256(path)
        if got != sha:
            raise R1ExportError(f"{path}: SHA-256 {got[:12]}... != meta.json {sha[:12]}...")


def rng_ops(model: Any) -> list[str]:
    """Return the RNG op types (:data:`RNG_OPS`) anywhere in an ONNX model.

    Scans the main graph, every subgraph attribute (``If``/``Loop``/``Scan`` bodies) and every
    local function.

    Args:
        model: An ``onnx.ModelProto``.

    Returns:
        Sorted ``"<domain>::<op>"`` strings; empty when the graph cannot draw random numbers.
    """
    found: set[str] = set()

    def scan(nodes: Any) -> None:
        for n in nodes:
            if n.op_type in RNG_OPS:
                found.add(f"{n.domain or 'ai.onnx'}::{n.op_type}")
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    scan(a.g.node)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for g in a.graphs:
                        scan(g.node)

    scan(model.graph.node)
    for f in model.functions:
        scan(f.node)
    return sorted(found)


def graph_checks(step_dir: Path) -> dict[str, Any]:
    """Opset, IR version and RNG-op scan of both graphs.

    Args:
        step_dir: The export directory.

    Returns:
        ``{graph: {"opsets", "ir_version", "rng_ops", "ok"}}`` plus ``"ok"`` overall.
    """
    out: dict[str, Any] = {}
    ok = True
    for name in GRAPHS:
        model = onnx.load(str(Path(step_dir) / name))
        opsets = {o.domain or "ai.onnx": int(o.version) for o in model.opset_import}
        ir_version, rng, n_fn = int(model.ir_version), rng_ops(model), len(model.functions)
        graph_ok = opsets == {"ai.onnx": OPSET} and ir_version <= MAX_IR_VERSION and not rng
        rec = {
            "opsets": opsets,
            "ir_version": ir_version,
            "rng_ops": rng,
            "n_functions": n_fn,
            "ok": graph_ok,
        }
        ok = ok and graph_ok
        out[name] = rec
    out["ok"] = ok
    return out


def cpu_session(path: Path, threads: int = 1) -> Any:
    """Return an ONNX Runtime ``InferenceSession`` on ``CPUExecutionProvider``.

    Args:
        path: An ``.onnx`` file.
        threads: Intra-op threads (inter-op is 1; sequential execution).

    Returns:
        The session (default graph optimisation level, ``ORT_ENABLE_ALL``).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def encode(
    session: Any, image: NDArray[np.uint8], state: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Run ``encoder.onnx``.

    Args:
        session: Its session.
        image: ``uint8[B, To, 96, 96, 3]``, oldest frame first.
        state: ``float32[B, To, 13]`` raw student proprio.

    Returns:
        ``float32[B, C]`` global conditioning.
    """
    out = session.run(
        ["global_cond"],
        {
            "image": np.ascontiguousarray(image, dtype=np.uint8),
            "state": np.ascontiguousarray(state, dtype=np.float32),
        },
    )[0]
    return np.asarray(out, dtype=np.float32)


def denoise(
    session: Any,
    sample: NDArray[np.float32],
    timestep: NDArray[np.int64],
    global_cond: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Run ``denoiser.onnx``.

    Args:
        session: Its session.
        sample: ``float32[B, Tp, 3]`` noisy normalised actions ``x_k``.
        timestep: ``int64[B]`` diffusion indices ``k``.
        global_cond: ``float32[B, C]``.

    Returns:
        ``float32[B, Tp, 3]`` epsilon-hat.
    """
    out = session.run(
        ["eps"],
        {
            "sample": np.ascontiguousarray(sample, dtype=np.float32),
            "timestep": np.ascontiguousarray(timestep, dtype=np.int64),
            "global_cond": np.ascontiguousarray(global_cond, dtype=np.float32),
        },
    )[0]
    return np.asarray(out, dtype=np.float32)


def ddim_from_meta(meta: dict[str, Any], num_inference_steps: int) -> Any:
    """Return ``ila``'s DDIM scheduler for an R1 export (:func:`ddim_scheduler`).

    Args:
        meta: The export's ``meta.json`` record.
        num_inference_steps: ``K``.

    Returns:
        diffusers ``DDIMScheduler`` with ``set_timesteps(K)`` applied.

    Raises:
        R1ExportError: If ``meta.json``'s inference block is not what ``ila`` implements.
    """
    inf = meta["scheduler"]["ila_inference"]
    if not (
        inf["sampler"] == "ddim"
        and inf["set_alpha_to_one"] is True
        and int(inf["steps_offset"]) == 0
        and float(inf["eta"]) == 0.0
    ):
        raise R1ExportError(f"unsupported inference block: {inf}")
    cfg = DPConfig(
        n_obs_steps=int(meta["n_obs_steps"]),
        horizon=int(meta["horizon"]),
        n_action_steps=int(meta["n_action_steps"]),
        action_dim=int(meta["action_dim"]),
        num_train_timesteps=int(inf["num_train_timesteps"]),
        beta_schedule=str(inf["beta_schedule"]),
        prediction_type=str(inf["prediction_type"]),
        clip_sample=bool(inf["clip_sample"]),
        num_inference_steps=int(num_inference_steps),
        timestep_spacing=str(inf["timestep_spacing"]),
    )
    return ddim_scheduler(cfg, int(num_inference_steps))


def sample_chunk(
    denoiser: Any,
    global_cond: NDArray[np.float32],
    noise: NDArray[np.float32],
    scheduler: Any,
) -> NDArray[np.float32]:
    """DDIM-sample a normalised chunk with ``ila``'s :func:`ddim_sample` and the ORT denoiser.

    Args:
        denoiser: ``denoiser.onnx`` session.
        global_cond: ``float32[B, C]``.
        noise: ``float32[B, Tp, 3]`` initial noise (``ila``'s per-episode generator).
        scheduler: From :func:`ddim_from_meta`.

    Returns:
        ``float32[B, Tp, 3]`` normalised chunk clipped to [-1, 1]; chunk index ``j`` = step
        ``t - To + 1 + j``.
    """

    def eps_fn(x: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(denoise(denoiser, x.numpy(), kk.numpy(), global_cond))

    out = ddim_sample(eps_fn, torch.from_numpy(np.asarray(noise, dtype=np.float32)), scheduler)
    return np.asarray(out.numpy(), dtype=np.float32)


def unnormalise(meta: dict[str, Any], x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Map normalised actions to env action units (LeRobot MIN_MAX inverse, float32, no clip).

    Args:
        meta: The export's ``meta.json`` record.
        x: ``float32[..., A]`` normalised actions.

    Returns:
        ``float32[..., A]``: ``(x + 1) / 2 * d + min`` with ``d = max - min`` (``eps`` where
        ``d == 0``), i.e. within the training labels' per-axis [min, max] for ``x`` in [-1, 1].

    Raises:
        R1ExportError: If the mode is not MIN_MAX.
    """
    u = meta["action_unnormalisation"]
    if u["mode"] != "MIN_MAX":
        raise R1ExportError(f"unsupported action unnormalisation {u['mode']!r}")
    lo = np.asarray(u["min"], dtype=np.float32)
    hi = np.asarray(u["max"], dtype=np.float32)
    d = hi - lo
    d = np.where(d == 0, np.float32(u["eps"]), d).astype(np.float32)
    xf = np.asarray(x, dtype=np.float32)
    out: NDArray[np.float32] = ((xf + np.float32(1)) / np.float32(2) * d + lo).astype(np.float32)
    return out


def _maxabs(a: NDArray[Any], b: NDArray[Any]) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_parity(step_dir: Path, *, threads: int = 1, write: bool = True) -> dict[str, Any]:
    """ORT CPU vs the torch reference of one export (module docstring); writes ``parity.json``.

    Args:
        step_dir: ``<run>/onnx/step<N>``.
        threads: ORT intra-op threads.
        write: Write ``parity.json`` into ``step_dir``.

    Returns:
        The record; ``record["pass"]`` is the verdict.

    Raises:
        R1ExportError: If the export is incomplete or its files do not match ``meta.json``.
    """
    import onnxruntime as ort

    t_start = time.perf_counter()
    step_dir = Path(step_dir)
    meta = load_meta(step_dir)
    verify_files(step_dir, meta)
    graphs = graph_checks(step_dir)
    enc = cpu_session(step_dir / "encoder.onnx", threads)
    den = cpu_session(step_dir / "denoiser.onnx", threads)
    with np.load(step_dir / "reference.npz", allow_pickle=False) as z:
        ref = {k: np.asarray(z[k]) for k in z.files}
    n = int(ref["image"].shape[0])
    to, ta = int(meta["n_obs_steps"]), int(meta["n_action_steps"])
    k = int(ref["ddim_k"])
    probe_t = ref["eps_timesteps"].astype(np.int64)
    ddim_t = ref["ddim_timesteps"].astype(np.int64)

    # Sampler identity: ila's scheduler (this venv's diffusers) vs the export's.
    sched = ddim_from_meta(meta, k)
    inf = meta["scheduler"]["ila_inference"]
    sched_t = [int(v) for v in sched.timesteps]
    alphas = [float(v) for v in sched.alphas_cumprod.numpy()]
    scheduler_match = {
        "timesteps": sched_t == [int(v) for v in ddim_t],
        "alphas_cumprod_bit_identical": alphas == [float(v) for v in inf["alphas_cumprod"]],
    }

    # Batch 1, end to end (ORT encoder -> ORT denoiser) and denoiser-only (reference cond).
    gc = np.concatenate([encode(enc, ref["image"][[i]], ref["state"][[i]]) for i in range(n)])
    probe_e2e = np.zeros((len(probe_t), n, *ref["eps"].shape[2:]), np.float32)
    probe_den = np.zeros_like(probe_e2e)
    step_e2e = np.zeros((k, n, *ref["ddim_eps"].shape[2:]), np.float32)
    step_den = np.zeros_like(step_e2e)
    chunk_norm = np.zeros_like(ref["chunk_norm"])
    for i in range(n):
        for j, t in enumerate(probe_t):
            x = ref["eps_sample"][j, i : i + 1]
            tt = np.full((1,), t, np.int64)
            probe_e2e[j, i] = denoise(den, x, tt, gc[i : i + 1])[0]
            probe_den[j, i] = denoise(den, x, tt, ref["global_cond"][i : i + 1])[0]
        for s, t in enumerate(ddim_t):
            x = ref["ddim_x"][s, i : i + 1]
            tt = np.full((1,), t, np.int64)
            step_e2e[s, i] = denoise(den, x, tt, gc[i : i + 1])[0]
            step_den[s, i] = denoise(den, x, tt, ref["global_cond"][i : i + 1])[0]
        chunk_norm[i] = sample_chunk(den, gc[i : i + 1], ref["noise"][i : i + 1], sched)[0]
    ref_eps_probe = ref["eps"]  # [T, n, Tp, A], the layout of eps_sample
    per_step_e2e = [_maxabs(step_e2e[s], ref["ddim_eps"][s]) for s in range(k)]
    per_step_den = [_maxabs(step_den[s], ref["ddim_eps"][s]) for s in range(k)]
    probe_e2e_m = {int(t): _maxabs(probe_e2e[j], ref_eps_probe[j]) for j, t in enumerate(probe_t)}
    probe_den_m = {int(t): _maxabs(probe_den[j], ref_eps_probe[j]) for j, t in enumerate(probe_t)}
    chunk = unnormalise(meta, chunk_norm)
    executed = chunk[:, to - 1 : to - 1 + ta]

    # Replays: ila's sampler on the stored epsilons (checking, at every step, that it asks at
    # the stored timestep with the stored x_k); unnormalise on the stored chunk.
    replay = []
    replay_x_gap = 0.0
    replay_t_ok = True
    for i in range(n):
        calls = iter(range(k))

        def eps_replay(
            x: torch.Tensor, kk: torch.Tensor, i: int = i, calls: Any = calls
        ) -> torch.Tensor:
            nonlocal replay_x_gap, replay_t_ok
            s = next(calls)
            replay_t_ok = replay_t_ok and bool(np.all(kk.numpy() == ddim_t[s]))
            replay_x_gap = max(replay_x_gap, _maxabs(x.numpy(), ref["ddim_x"][s, i : i + 1]))
            return torch.from_numpy(ref["ddim_eps"][s, i : i + 1].copy())

        replay.append(
            ddim_sample(eps_replay, torch.from_numpy(ref["noise"][i : i + 1].copy()), sched).numpy()
        )
    replay_chunk = np.concatenate(replay)
    unnorm_replay = unnormalise(meta, ref["chunk_norm"])

    # Batch-size consistency: one ORT call on all n inputs.
    gc_b = encode(enc, ref["image"], ref["state"])
    step_b = [
        denoise(den, ref["ddim_x"][s], np.full((n,), int(t), np.int64), gc_b)
        for s, t in enumerate(ddim_t)
    ]

    lr = meta["lerobot_path_check"]
    judged = {
        "files_match_meta": True,
        "graphs_opset17_no_rng": bool(graphs["ok"]),
        "per_step_eps_le_tol": max(per_step_e2e) <= EPS_TOL,
        "probe_eps_le_tol": max(probe_e2e_m.values()) <= EPS_TOL,
        "sampler_replay_le_tol": _maxabs(replay_chunk, ref["chunk_norm"]) <= REPLAY_TOL
        and replay_x_gap <= REPLAY_TOL
        and replay_t_ok,
        "scheduler_match": all(scheduler_match.values()),
        "unnormalise_replay_le_tol": _maxabs(unnorm_replay, ref["chunk"]) <= REPLAY_TOL,
        "lerobot_path_check": float(lr["max_abs"]) <= float(lr["tolerance"]),
    }
    record: dict[str, Any] = {
        "pass": all(judged.values()),
        "judged": judged,
        "criterion": (
            f"per-step eps max-abs <= {EPS_TOL:g} end to end (ORT encoder -> ORT denoiser at the "
            f"torch path's x_k), at each of the K={k} DDIM-trailing steps and at probe timesteps "
            f"{[int(t) for t in probe_t]} (P3-D1 §6 H5(a), open loop); sampler and "
            f"unnormalisation replays <= {REPLAY_TOL:g}; graphs opset {OPSET}, IR <= "
            f"{MAX_IR_VERSION}, no RNG op; SHA-256 match. Final-chunk max-abs reported, not "
            "judged (x2 029 first-step gain, P3-D1 §8)."
        ),
        "n_inputs": n,
        "step": int(meta["step"]),
        "eps_per_step_max_abs": per_step_e2e,
        "eps_per_step_max_abs_reference_cond": per_step_den,
        "eps_probe_max_abs": {str(t): v for t, v in probe_e2e_m.items()},
        "eps_probe_max_abs_reference_cond": {str(t): v for t, v in probe_den_m.items()},
        "ddim_timesteps": [int(t) for t in ddim_t],
        "global_cond_max_abs": _maxabs(gc, ref["global_cond"]),
        "global_cond_max_abs_rel": _maxabs(gc, ref["global_cond"])
        / max(float(np.max(np.abs(ref["global_cond"]))), 1e-30),
        "chunk_norm_max_abs": _maxabs(chunk_norm, ref["chunk_norm"]),
        "chunk_max_abs": _maxabs(chunk, ref["chunk"]),
        "chunk_executed_max_abs": _maxabs(executed, ref["chunk"][:, to - 1 : to - 1 + ta]),
        "chunk_executed_vs_lerobot_select_action_max_abs": _maxabs(
            executed, ref["lerobot_actions"]
        ),
        "chunk_within_1e-3": _maxabs(chunk_norm, ref["chunk_norm"]) <= 1e-3,
        "sampler_replay_max_abs": _maxabs(replay_chunk, ref["chunk_norm"]),
        "sampler_replay_bit_identical": bool(np.array_equal(replay_chunk, ref["chunk_norm"])),
        "sampler_replay_x_k_max_abs": replay_x_gap,
        "sampler_replay_timesteps_match": replay_t_ok,
        "unnormalise_replay_max_abs": _maxabs(unnorm_replay, ref["chunk"]),
        "unnormalise_replay_bit_identical": bool(np.array_equal(unnorm_replay, ref["chunk"])),
        "scheduler_match": scheduler_match,
        "batch_consistency": {
            "batch": n,
            "global_cond_max_abs_vs_b1": _maxabs(gc_b, gc),
            "eps_per_step_max_abs_vs_b1": None,  # filled below
        },
        "lerobot_path_check": lr,
        "graphs": graphs,
        "meta_sha256": _sha256(step_dir / "meta.json"),
        "onnx_sha256": meta["onnx_sha256"],
        "source_checkpoint_sha256": meta["source"]["checkpoint_sha256"],
        "provider": "CPUExecutionProvider",
        "onnxruntime": str(ort.__version__),
        "ort_threads": int(threads),
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "torch_reference": "torch fp32 CPU batch 1 (" + meta["versions"]["torch"] + ")",
        "machine": {"cpu": _cpu_model(), "platform": platform.platform()},
        "wall_s": 0.0,
    }
    # The batched denoiser is compared with batch-1 runs on the same (batched-encoder) cond,
    # so the figure isolates the denoiser's own batch-size effect.
    step_b1_same = [
        np.concatenate(
            [
                denoise(
                    den,
                    ref["ddim_x"][s, i : i + 1],
                    np.full((1,), int(t), np.int64),
                    gc_b[i : i + 1],
                )
                for i in range(n)
            ]
        )
        for s, t in enumerate(ddim_t)
    ]
    record["batch_consistency"]["eps_per_step_max_abs_vs_b1"] = max(
        _maxabs(step_b[s], step_b1_same[s]) for s in range(k)
    )
    record["wall_s"] = time.perf_counter() - t_start
    if write:
        (step_dir / "parity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
