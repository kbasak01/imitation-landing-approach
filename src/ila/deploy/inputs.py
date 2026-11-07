"""The 1 000 fixed open-loop parity inputs, from ``select`` only (P3-D1 §6 H5(a); P9-D1).

P8's ``ppo`` s0 teacher (the teacher every core arm was distilled from; P8's frozen
``VecNormalize``) flies the 150 ``splits/select.parquet`` episodes through
:func:`ila.eval.runner.fly_chunk` with render ``full`` and its CPU path. At every control step
the student input that a vision student would receive is built from the same physics state
through :class:`ila.eval.runner.StudentWindow` (``ObsContract.student_input``: the images and
the proprio allowlist, selected by P8 field name; no privileged field and no clock) and kept.
The teacher's own action drives the flight; nothing a student computes enters it.

From the pooled windows, ``n`` (1 000) are drawn without replacement by
``numpy.random.default_rng(draw_seed)`` and stored, sorted by (episode, step), in
``artifacts/deploy/inputs/select_inputs.npz`` (gitignored):

* ``images uint8[n, To, 96, 96, 3]`` (RGB, oldest frame first);
* ``proprio float32[n, To, 13]`` (P8 units: rad, rad/s, m/s model scale, normalised action,
  {0, 1});
* ``episode int64[n]`` (position in ``select``), ``step int64[n]`` (control step, 1/30 s model
  scale), ``episode_seed int64[n]``.

Its SHA-256 and the teacher flight's outcome counts go to ``inputs.json``. Windows are input
data only: nothing here is selected, tuned or scored, and no frozen list is read.
"""

from __future__ import annotations

import functools
import hashlib
import multiprocessing as mp
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import runner
from ila.obs.split import ObsContract

__all__ = [
    "INPUTS_FILE",
    "SelectInputs",
    "draw",
    "load_inputs",
    "record_select_inputs",
]

INPUTS_FILE: Final[str] = "select_inputs.npz"


@dataclass(frozen=True)
class SelectInputs:
    """The open-loop inputs (module docstring)."""

    images: NDArray[np.uint8]
    proprio: NDArray[np.float32]
    episode: NDArray[np.int64]
    step: NDArray[np.int64]
    episode_seed: NDArray[np.int64]

    @property
    def n(self) -> int:
        """Number of inputs."""
        return int(self.images.shape[0])


class _RecordingTeacherActor:
    """The teacher acts on P8's observation; the student window of the same step is kept."""

    needs_frames: bool = True

    def __init__(self, policy: Any, contract: ObsContract) -> None:
        self.policy = policy
        self.window = runner.StudentWindow(contract)
        self.episodes: list[dict[str, list[Any]]] = []

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        self.window.clear()
        self.policy.reset(listed.episode_seed, context)
        self.episodes.append({"images": [], "proprio": [], "seed": [listed.episode_seed]})

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        si = self.window.push(step, obs, frame)
        ep = self.episodes[-1]
        ep["images"].append(np.array(si.images, copy=True))
        ep["proprio"].append(np.array(si.proprio, copy=True))
        return np.asarray(self.policy.act(obs), dtype=np.float64)


def _fly(task: runner.VisionTask) -> tuple[list[dict[str, Any]], list[dict[str, list[Any]]]]:
    actors: list[_RecordingTeacherActor] = []

    def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Any, object]:
        policy = task.spec.build(cfgs)
        actor = _RecordingTeacherActor(policy, contract)
        actors.append(actor)
        return actor, policy

    rows = runner.fly_chunk(task, make)
    return rows, (actors[0].episodes if actors else [])


def draw(n_total: int, n: int, seed: int) -> NDArray[np.int64]:
    """``n`` sorted indices of ``range(n_total)`` without replacement, ``default_rng(seed)``."""
    if n > n_total:
        raise ValueError(f"cannot draw {n} of {n_total} windows")
    rng = np.random.default_rng(int(seed))
    out: NDArray[np.int64] = np.sort(rng.choice(n_total, size=n, replace=False)).astype(np.int64)
    return out


def record_select_inputs(
    out_dir: Path,
    *,
    n: int,
    draw_seed: int,
    policy: str = "ppo",
    workers: int = 24,
    chunk: int = 10,
    episodes: Sequence[p8_adapter.ListedEpisode] | None = None,
) -> dict[str, Any]:
    """Fly the teacher on ``select``, keep every student window, draw ``n`` and store them.

    Args:
        out_dir: Directory of :data:`INPUTS_FILE`.
        n: Inputs kept (1 000).
        draw_seed: Draw seed.
        policy: P8 teacher (``ppo``).
        workers: Processes (windows do not depend on it).
        chunk: Episodes per task.
        episodes: Override of the ``select`` list (tests only).

    Returns:
        The ``inputs.json`` record.
    """
    eps = list(data_episodes.read_split("select") if episodes is None else episodes)
    spec = p8_adapter.teacher_spec(policy)
    arm = runner.VisionArm(spec=spec, episodes=eps, tags=runner.RowTags(arm=policy), render="full")
    tasks = runner.chunk_tasks([arm], p8_adapter.load_eval_configs(), chunk)
    runner.set_single_thread_env()
    if workers == 1 or len(tasks) <= 1:
        runner.init_worker_threads()
        results = [_fly(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=runner.init_worker_threads) as pool:
            results = list(pool.imap(_fly, tasks, chunksize=1))
    rows = [r for rs, _ in results for r in rs]
    recs = [e for _, es in results for e in es]
    index = [(e, s) for e, rec in enumerate(recs) for s in range(len(rec["images"]))]
    pick = draw(len(index), n, draw_seed)
    chosen = [index[i] for i in pick]
    images = np.stack([recs[e]["images"][s] for e, s in chosen]).astype(np.uint8)
    proprio = np.stack([recs[e]["proprio"][s] for e, s in chosen]).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / INPUTS_FILE
    np.savez(
        path,
        images=images,
        proprio=proprio,
        episode=np.array([e for e, _ in chosen], np.int64),
        step=np.array([s for _, s in chosen], np.int64),
        episode_seed=np.array([recs[e]["seed"][0] for e, _ in chosen], np.int64),
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": str(path),
        "sha256": sha,
        "n": int(n),
        "draw_seed": int(draw_seed),
        "draw": "numpy.random.default_rng(draw_seed).choice(N, n, replace=False), sorted",
        "windows_total": len(index),
        "list": "select",
        "list_sha256": p8_adapter.sha256_file(data_episodes.SPLITS_DIR / "select.parquet")
        if episodes is None
        else None,
        "episodes": len(eps),
        "policy": f"{policy} s0 (P8), CPU path, render full",
        "teacher_outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items())),
        "images_shape": list(images.shape),
        "proprio_shape": list(proprio.shape),
    }


@functools.cache
def _load(path: str, sha: str | None) -> SelectInputs:
    p = Path(path)
    if sha is not None:
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            raise ValueError(f"{p}: SHA-256 {got[:12]}... != recorded {sha[:12]}...")
    with np.load(p, allow_pickle=False) as z:
        return SelectInputs(
            images=np.asarray(z["images"], np.uint8),
            proprio=np.asarray(z["proprio"], np.float32),
            episode=np.asarray(z["episode"], np.int64),
            step=np.asarray(z["step"], np.int64),
            episode_seed=np.asarray(z["episode_seed"], np.int64),
        )


def load_inputs(path: Path, sha256: str | None = None) -> SelectInputs:
    """Read :data:`INPUTS_FILE` (its SHA-256 checked when given)."""
    return _load(str(path), sha256)
