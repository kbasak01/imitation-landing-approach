"""Teacher label parity (plan Phase 2 item 3): P8's PPO teacher, SB3 path vs ONNX on ORT CPU.

Observations: the first :data:`N_OBS` observations ``ppo`` seed 0 is actually given along its
own rollouts on P8 ``train``-pool realizations (``ila.data.episodes.train_rows``, stream
:data:`STREAM`, generator seed 20261023). The rollouts go through the runner's P8 loop
(:func:`ila.eval.runner.fly_chunk`: the start check on every reset, render off; obs do not
depend on the render). No frozen list and no ``tune`` realization is used.

Comparisons (all actions normalised, dimensionless; observations raw P8 units, model scale):

* **primary**: P8's ``reference_actions`` (SB3 ``predict(deterministic=True)`` on the frozen
  ``VecNormalize``, batched, CPU) vs the ONNX label path, ``attach_onnx(..., threads=1)``'s
  ``act(obs)`` on ORT ``CPUExecutionProvider``, one observation per call, clipped as P8 clips.
  Criterion: max |diff| <= :data:`CRITERION` (1e-6).
* the SB3 label path itself (``LearnedPolicy.act``, batch 1) vs ``reference_actions``;
* ORT batched (one call on all rows) vs ORT per row;
* P8's own ``check_ort`` record (unclipped graph output vs the reference, P8's 1e-4 scaled
  criterion), for the record.
"""

from __future__ import annotations

import functools
import hashlib
import platform
import socket
from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval.runner import Actor, RowTags, TeacherActor, VisionTask, fly_chunk
from ila.obs.split import ObsContract

__all__ = ["CRITERION", "N_OBS", "SEED", "STREAM", "compare", "diff_stats", "record_observations"]

#: Observations compared.
N_OBS: Final[int] = 10_000

#: Draw stream of the rollout realizations (independent of every demo stream).
STREAM: Final[str] = "label_parity"

#: The project's generator seed (configs/data/demos.yaml).
SEED: Final[int] = 20261023

#: Max-abs criterion on the primary comparison, normalised action units.
CRITERION: Final[float] = 1e-6


class _Recorder:
    """A teacher actor that keeps a copy of every observation it is asked to act on."""

    needs_frames: bool = False

    def __init__(self, inner: TeacherActor, sink: list[NDArray[np.float32]], where: list[Any]):
        self.inner = inner
        self.sink = sink
        self.where = where
        self._episode: tuple[str, int] = ("", -1)

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        self._episode = (listed.ss, int(listed.index))
        self.inner.begin(listed, context)

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        self.sink.append(np.array(obs, dtype=np.float32, copy=True))
        self.where.append((*self._episode, step))
        return self.inner.act(step, obs, frame)


def record_observations(
    n_obs: int = N_OBS, n_episodes: int = 240, chunk: int = 24, workers: int = 8
) -> tuple[NDArray[np.float32], list[tuple[str, int, int]], list[p8_adapter.ListedEpisode]]:
    """Fly ``ppo`` seed 0 on train-pool episodes until ``n_obs`` observations are recorded.

    Args:
        n_obs: Observations to keep (the first ``n_obs`` in flight order).
        n_episodes: Train episodes drawn and resolved (more than needed; flown in order).
        chunk: Episodes flown per environment.
        workers: Processes for P8's draw resolution (rows do not depend on it).

    Returns:
        ``(obs float32[n_obs, 25], (ss, index, step) per row, the episodes flown)``.

    Raises:
        RuntimeError: If the drawn episodes run out first, or the seed is not 20261023.
    """
    cfgs = p8_adapter.load_eval_configs()
    split_cfg = data_episodes.load_splits_config()
    if split_cfg.generator_seed != SEED:
        raise RuntimeError(f"generator seed {split_cfg.generator_seed} != {SEED}")
    api = data_episodes.p8_episode_api(cfgs, workers=workers)
    rows = data_episodes.train_rows(api, split_cfg, n_episodes, stream=STREAM)
    policy = p8_adapter.teacher("ppo", 0, cfgs)
    spec = p8_adapter.callable_spec(
        "ppo", functools.partial(p8_adapter.teacher, "ppo", 0), run_seed=0
    )
    sink: list[NDArray[np.float32]] = []
    where: list[Any] = []

    def make(c: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Actor, object]:
        del c, contract
        return _Recorder(TeacherActor(policy), sink, where), policy

    flown: list[p8_adapter.ListedEpisode] = []
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        fly_chunk(VisionTask(part, spec, cfgs, RowTags("ppo"), render="off"), make)
        flown += part
        if len(sink) >= n_obs:
            break
    if len(sink) < n_obs:
        raise RuntimeError(f"only {len(sink)} observations from {len(rows)} episodes")
    used = {(ss, idx) for ss, idx, _ in where[:n_obs]}
    flown = [e for e in flown if (e.ss, int(e.index)) in used]
    return np.stack(sink[:n_obs]), [tuple(w) for w in where[:n_obs]], flown


def diff_stats(
    a: NDArray[Any], b: NDArray[Any], where: Sequence[tuple[str, int, int]] | None = None
) -> dict[str, Any]:
    """Return max/p99/mean |a - b|, the argmax location and the bit-equal fraction."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    err = np.abs(x - y)
    flat = int(np.argmax(err))
    row, comp = divmod(flat, err.shape[1])
    out: dict[str, Any] = {
        "max_abs": float(err.max()),
        "p99_abs": float(np.quantile(err, 0.99)),
        "mean_abs": float(err.mean()),
        "argmax_row": row,
        "argmax_component": comp,
        "a_at_argmax": float(x[row, comp]),
        "b_at_argmax": float(y[row, comp]),
        "fraction_bit_equal": float(np.mean(np.asarray(a) == np.asarray(b))),
        "n_values": int(err.size),
    }
    if where is not None:
        ss, index, step = where[row]
        out["argmax_episode"] = {"ss": ss, "index": int(index), "step": int(step)}
    return out


def compare(obs: NDArray[np.float32], where: Sequence[tuple[str, int, int]]) -> dict[str, Any]:
    """Run every comparison on recorded observations (see module docstring)."""
    import onnxruntime as ort

    cfgs = p8_adapter.load_eval_configs()
    paths = p8_adapter.load_paths()
    onnx_path = paths["p8_ppo_seed0_onnx"]
    sb3 = p8_adapter.teacher("ppo", 0, cfgs)
    x = np.ascontiguousarray(obs, dtype=np.float32)
    reference = np.asarray(p8_adapter.reference_actions(sb3, x))  # type: ignore[arg-type]
    sb3_single = np.stack([np.asarray(sb3.act(o), dtype=np.float32) for o in x])
    onnx_policy = p8_adapter.attach_onnx(p8_adapter.teacher("ppo", 0, cfgs), onnx_path, threads=1)  # type: ignore[arg-type]
    ort_single = np.stack([np.asarray(onnx_policy.act(o), dtype=np.float32) for o in x])
    session, provider = p8_adapter.open_session(onnx_path, "CPUExecutionProvider", 1)
    (raw,) = session.run([], {session.get_inputs()[0].name: x})
    ort_batch = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
    p8_record = p8_adapter.check_ort(session, cast(Any, x), reference, provider, seed=SEED)
    primary = diff_stats(reference, ort_single, where)
    passed = primary["max_abs"] <= CRITERION
    return {
        "criterion": f"max |SB3 reference_actions - ORT CPU 1-thread act| <= {CRITERION:g}",
        "passed": passed,
        "label_path": "onnx_cpu permitted (parity met)" if passed else "sb3 (parity not met)",
        "primary_sb3_reference_vs_ort_cpu_single": primary,
        "sb3_reference_vs_sb3_act_single": diff_stats(reference, sb3_single, where),
        "ort_single_vs_ort_batch": diff_stats(ort_single, ort_batch, where),
        "sb3_act_single_vs_ort_cpu_single": diff_stats(sb3_single, ort_single, where),
        "p8_check_ort": {
            k: (float(v) if isinstance(v, np.floating) else v) for k, v in vars(p8_record).items()
        },
        "onnx": {
            "path": "p8_ppo_seed0_onnx (configs/paths.local.yaml)",
            "sha256": p8_adapter.sha256_file(onnx_path),
            "provider": provider,
            "threads": 1,
            "onnxruntime": str(ort.__version__),
        },
        "sb3_model_sha256": p8_adapter.sha256_file(
            paths["p8_ppo_seed0_run"] / "final" / "model.zip"
        ),
        "action_abs_max": float(np.abs(reference).max()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }


def obs_sha256(obs: NDArray[np.float32]) -> str:
    """SHA-256 of the observations: C-order little-endian float32 bytes, shape prefixed."""
    arr = np.ascontiguousarray(obs, dtype="<f4")
    digest = hashlib.sha256(f"{arr.shape}".encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()
