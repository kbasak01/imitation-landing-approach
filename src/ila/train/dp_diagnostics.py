"""Diffusion Policy diagnostics of a run's selected checkpoint (Phase 6; P6-D1).

Three measurements, each of the **EMA** weights in eval mode (centre crop), fp32, on the CPU,
at the selection setting ``Ta = 8``, ``K = 10`` (trailing DDIM), no ensembling. Actions are
normalised world-frame velocity setpoints (x v_max = 1.5 m/s model scale, lambda = 1/25); one
control step is 1/30 s model scale.

1. **Sample spread.** For each state, 16 chunks are sampled; sample ``j`` of state ``i`` takes
   its initial noise from :func:`ila.policies.dp.policy.initial_noise` keyed
   ``(run_seed, SPREAD_EPISODE_KEY, 16 i + j)`` (P2-D3 item 4's generator; never a global RNG;
   the episode-key slot is negative, so it never collides with a flight's key, whose episode
   seed is P8's, >= 0). Per state: the per-element standard deviation over the 16 samples
   (``ddof = 1``), averaged over the executed slice (chunk indices ``To-1 ... To+6``, 8 steps x
   3 axes) -> ``spread_executed``, and over index ``To-1`` only (the action at step ``t``) ->
   ``spread_first``. Normalised action units (x 1.5 m/s model scale). The teacher (SB3 PPO,
   deterministic) gives one label per state, so its per-state spread is 0: any spread is the
   student's sampling distribution, not imitated teacher multimodality. On held-out states the
   RMS distance of the 16-sample mean from the label (``bias_executed``) is reported beside it.
2. **Clip fraction.** The fraction of sampled chunk elements at the ``[-1, 1]`` bound,
   ``|a| >= 1 - 1e-6``. With ``clip_sample`` and ``set_alpha_to_one`` the last DDIM step returns
   the clipped ``x0`` estimate itself, so a saturated element is exactly +-1. Reported for the
   whole chunk and the executed slice, beside the labels' own fraction at the bound (held-out)
   and the executed actions' (closed loop).
3. **Chunk-boundary jerk.** Over an executed action sequence ``a_t`` (replanning at
   :func:`chunking.is_replan_step` ``(t, 8)``): the first difference ``|a_t - a_{t-1}|``
   (Euclidean over 3 axes; normalised action per control step, x 45 m/s^2 model scale) is a
   **boundary** difference when ``t`` is a replanning step (``a_t`` from the new chunk,
   ``a_{t-1}`` from the old), else **within** a chunk; the second difference
   ``|a_t - 2 a_{t-1} + a_{t-2}|`` (the jerk of the velocity setpoint; x 1350 m/s^3 model
   scale) is a boundary difference when ``t`` or ``t-1`` is a replanning step. Reported: mean,
   median, p95 and count of each class, and the ratio of means boundary / within.

State sets:

* **Held-out** (:func:`ila.train.diagnostics.heldout_states`, P4-D5's set): ``ppo_2000``
  stream indices 1950-1999, every second episode (25), read through the student path only
  under the checkpoint's ``obs_variant`` (an ablation's proprio is 10-d or 14-d, not 13-d).
  Spread and clip on 512 frames drawn by ``default_rng(0)`` (P4-D5's draw). Jerk **open loop**:
  each episode's recorded observation windows fed to the student in order (a teacher-forced
  replay), noise keyed ``(run_seed, OPEN_LOOP_KEY_BASE - e, t)`` for held-out episode ``e``; the
  teacher's labels at the same steps, classified by the same replanning steps, are the
  reference (labels have no chunks, so their ratio is about 1).
* **select, closed loop** (optional; :func:`closed_loop_record`): the selected checkpoint
  re-flown on the 150 ``select`` episodes through the runner (CPU fp32 single-sample path,
  render ``full``, the selection's tags) with an actor that records each executed action and the
  student input at each replanning step. Its rows must equal the stored selection rows (the
  recorder changes nothing); its per-episode mean first difference must equal the rows'
  ``action_jerk_mean`` (P8's column, mean ``|a_t - a_{t-1}|``). Jerk on the executed actions;
  spread and clip on up to 512 recorded replanning-step inputs drawn by ``default_rng(0)``.

Per-episode rows do not hold per-step actions, which is why ``select`` is re-flown rather than
read from the stored rows. No frozen list is read.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila import p8_adapter
from ila.eval import runner
from ila.obs.split import ObsContract
from ila.policies.dp import chunking
from ila.policies.dp.policy import DiffusionPolicyNet, DPStudent, ddim_sample, initial_noise
from ila.train.dataset import DPChunkDataset
from ila.train.diagnostics import COPYCAT_SAMPLE, COPYCAT_SEED, sample_indices

__all__ = [
    "CLIP_TOL",
    "DIAG_K",
    "DIAG_TA",
    "OPEN_LOOP_KEY_BASE",
    "SPREAD_EPISODE_KEY",
    "SPREAD_SAMPLES",
    "SPREAD_STATES",
    "ClosedLoopRecord",
    "chunk_jerk",
    "closed_loop_record",
    "jerk_summary",
    "open_loop_actions",
    "draw_states",
    "quantiles",
    "run_diagnostics",
    "spread_and_clip",
    "spread_record",
]

#: Samples per state for the spread.
SPREAD_SAMPLES: Final[int] = 16

#: States drawn for spread and clip (P4-D5's 512, ``default_rng(0)``).
SPREAD_STATES: Final[int] = COPYCAT_SAMPLE

#: Episode-key slot of the spread's noise keys (negative: never a P8 episode seed).
SPREAD_EPISODE_KEY: Final[int] = -1

#: Open-loop replay of held-out episode ``e`` uses episode key ``OPEN_LOOP_KEY_BASE - e``.
OPEN_LOOP_KEY_BASE: Final[int] = -2

#: An element is at the clip bound when ``|a| >= 1 - CLIP_TOL``.
CLIP_TOL: Final[float] = 1e-6

#: The setting measured (the selection setting, P3-D1 §4).
DIAG_TA: Final[int] = 8
DIAG_K: Final[int] = 10

#: States per sampler call in :func:`spread_and_clip` (memory only).
_STATE_BATCH: Final[int] = 16


def quantiles(x: Sequence[float] | NDArray[np.float64]) -> dict[str, float | int | None]:
    """``n``, ``mean``, ``median``, ``p90``, ``p95``, ``p99``, ``max`` (``None`` when empty)."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return dict.fromkeys(("mean", "median", "p90", "p95", "p99", "max"), None) | {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


@torch.inference_mode()
def spread_and_clip(
    net: DiffusionPolicyNet,
    run_seed: int,
    images: NDArray[np.uint8],
    proprio: NDArray[np.float32],
    targets: NDArray[np.float32] | None = None,
    *,
    n_samples: int = SPREAD_SAMPLES,
    n_action_steps: int = DIAG_TA,
    num_inference_steps: int = DIAG_K,
) -> dict[str, Any]:
    """Per-state spread of ``n_samples`` chunks and the fraction of elements at the bound.

    Args:
        net: The network (EMA weights, eval mode, CPU).
        run_seed: The run's seed (first element of every noise key).
        images: ``uint8[N, To, 96, 96, 3]`` observation windows.
        proprio: ``float32[N, To, D]``.
        targets: ``float32[N, Tp, 3]`` label chunks, or ``None`` (closed-loop states).
        n_samples: Samples per state.
        n_action_steps: ``Ta`` whose executed slice is measured.
        num_inference_steps: ``K``.

    Returns:
        ``spread_executed``, ``spread_first`` (per state, normalised action units),
        ``clip_chunk``, ``clip_executed`` (fractions), and with targets ``bias_executed``
        (per state, RMS of sample mean - label over the executed slice) and
        ``label_clip_executed``.
    """
    cfg = net.cfg
    ex = chunking.executed_slice(cfg.n_obs_steps, cfg.horizon, n_action_steps)
    first = ex.start
    sched = net.ddim(num_inference_steps)
    spread_ex, spread_first, bias = [], [], []
    clip_chunk = clip_ex = 0
    n_chunk = n_ex = 0
    for start in range(0, images.shape[0], _STATE_BATCH):
        stop = min(images.shape[0], start + _STATE_BATCH)
        cond = net.encoder(
            torch.from_numpy(np.ascontiguousarray(images[start:stop])),
            torch.from_numpy(np.ascontiguousarray(proprio[start:stop])),
        )
        cond_r = cond.repeat_interleave(n_samples, dim=0)
        keys = [
            (int(run_seed), SPREAD_EPISODE_KEY, i * n_samples + j)
            for i in range(start, stop)
            for j in range(n_samples)
        ]
        noise = initial_noise(keys, cfg.horizon, cfg.action_dim)

        def eps_fn(x: torch.Tensor, k: torch.Tensor, c: torch.Tensor = cond_r) -> torch.Tensor:
            return net.predict_noise(x, k, c)

        out = ddim_sample(eps_fn, noise, sched).numpy().astype(np.float64)
        out = out.reshape(stop - start, n_samples, cfg.horizon, cfg.action_dim)
        at_bound = np.abs(out) >= 1.0 - CLIP_TOL
        clip_chunk += int(at_bound.sum())
        n_chunk += at_bound.size
        clip_ex += int(at_bound[:, :, ex].sum())
        n_ex += at_bound[:, :, ex].size
        sd = out.std(axis=1, ddof=1)  # [B, Tp, 3]
        spread_ex += list(sd[:, ex].mean(axis=(1, 2)))
        spread_first += list(sd[:, first].mean(axis=1))
        if targets is not None:
            mean = out.mean(axis=1)[:, ex]
            tgt = np.asarray(targets[start:stop], dtype=np.float64)[:, ex]
            bias += list(np.sqrt(((mean - tgt) ** 2).mean(axis=(1, 2))))
    rec: dict[str, Any] = {
        "spread_executed": spread_ex,
        "spread_first": spread_first,
        "clip_chunk": clip_chunk / n_chunk if n_chunk else None,
        "clip_executed": clip_ex / n_ex if n_ex else None,
    }
    if targets is not None:
        lab = np.abs(np.asarray(targets, dtype=np.float64)[:, ex]) >= 1.0 - CLIP_TOL
        rec["bias_executed"] = bias
        rec["label_clip_executed"] = float(lab.mean())
    return rec


def chunk_jerk(actions: NDArray[np.float64], n_action_steps: int) -> dict[str, list[float]]:
    """Classify an executed sequence's differences as chunk-boundary or within-chunk.

    Args:
        actions: ``float[T, 3]`` executed actions of one episode, step 0 first.
        n_action_steps: ``Ta``; step ``t`` starts a chunk iff :func:`chunking.is_replan_step`.

    Returns:
        ``d1_boundary``, ``d1_within`` (``|a_t - a_{t-1}|``, normalised action per control
        step), ``d2_boundary``, ``d2_within`` (``|a_t - 2 a_{t-1} + a_{t-2}|``, per control
        step squared); module docstring item 3.
    """
    a = np.asarray(actions, dtype=np.float64)
    out: dict[str, list[float]] = {"d1_boundary": [], "d1_within": [], "d2_boundary": [],
                                   "d2_within": []}  # fmt: skip
    replan = [chunking.is_replan_step(t, n_action_steps) for t in range(a.shape[0])]
    for t in range(1, a.shape[0]):
        d1 = float(np.linalg.norm(a[t] - a[t - 1]))
        out["d1_boundary" if replan[t] else "d1_within"].append(d1)
        if t >= 2:
            d2 = float(np.linalg.norm(a[t] - 2.0 * a[t - 1] + a[t - 2]))
            out["d2_boundary" if (replan[t] or replan[t - 1]) else "d2_within"].append(d2)
    return out


def jerk_summary(per_episode: Sequence[Mapping[str, list[float]]]) -> dict[str, Any]:
    """Pool :func:`chunk_jerk` over episodes; ratio of means boundary / within."""
    pooled = {k: [v for ep in per_episode for v in ep[k]] for k in per_episode[0]}
    out: dict[str, Any] = {k: quantiles(v) for k, v in pooled.items()}
    for d in ("d1", "d2"):
        b, w = out[f"{d}_boundary"]["mean"], out[f"{d}_within"]["mean"]
        out[f"{d}_ratio_boundary_over_within"] = b / w if b is not None and w else None
    out["units"] = (
        "d1: normalised action per control step (x 45 m/s^2 model scale); d2: per control "
        "step squared (x 1350 m/s^3 model scale)"
    )
    return out


def open_loop_actions(
    student: DPStudent, data: DPChunkDataset
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Teacher-forced replay: each episode's windows fed to the student in order.

    Episode ``e`` is reset with key ``OPEN_LOOP_KEY_BASE - e`` (module docstring), so the noise
    of the replan at step ``t`` is keyed ``(run_seed, OPEN_LOOP_KEY_BASE - e, t)``.

    Returns:
        Per episode ``(executed float64[T, 3], labels float64[T, 3])``.
    """
    table = data.table
    out = []
    for e in range(table.n_episodes):
        start, t_len = int(table.episode_start[e]), int(table.episode_len[e])
        student.reset(OPEN_LOOP_KEY_BASE - e)
        acts = []
        for t in range(t_len):
            sample = data[start + t]  # one window, no batch axis
            acts.append(
                student.act_vision(
                    np.asarray(sample["images"], dtype=np.uint8),
                    np.asarray(sample["proprio"], dtype=np.float32),
                )
            )
        labels = np.asarray(table.label[start : start + t_len], dtype=np.float64)
        out.append((np.asarray(acts, dtype=np.float64), labels))
    return out


# --------------------------------------------------------------------------- closed loop


class _RecordingStudentActor(runner.StudentActor):
    """:class:`runner.StudentActor` that keeps each executed action and replan-step input."""

    def __init__(self, policy: Any, contract: ObsContract, every: int) -> None:
        super().__init__(policy, contract)
        self.every = every
        self.episodes: list[dict[str, list[NDArray[Any]]]] = []

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        super().begin(listed, context)
        self.episodes.append({"actions": [], "images": [], "proprio": []})

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        si = self.window.push(step, obs, frame)
        action = np.asarray(self.policy.act_vision(si.images, si.proprio), dtype=np.float64)
        ep = self.episodes[-1]
        ep["actions"].append(action.copy())
        if chunking.is_replan_step(step, self.every):
            ep["images"].append(np.array(si.images, copy=True))
            ep["proprio"].append(np.array(si.proprio, copy=True))
        return action


def _fly_recorded(
    job: tuple[runner.VisionTask, int],
) -> tuple[list[dict[str, Any]], list[dict[str, list[NDArray[Any]]]]]:
    """Fly one chunk with a recording student actor (picklable pool entry point)."""
    task, every = job
    actors: list[_RecordingStudentActor] = []

    def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Any, object]:
        policy = task.spec.build(cfgs)
        actor = _RecordingStudentActor(policy, contract, every)
        actors.append(actor)
        return actor, policy

    rows = runner.fly_chunk(task, make)
    return rows, (actors[0].episodes if actors else [])


@dataclass(frozen=True)
class ClosedLoopRecord:
    """Rows, executed actions and replan-step inputs of one closed-loop flight.

    Attributes:
        rows: Runner rows, episode order.
        actions: Per episode ``float64[T, 3]`` executed actions (normalised).
        images: ``uint8[M, To, 96, 96, 3]`` inputs at every replanning step, episode order.
        proprio: ``float32[M, To, D]``.
    """

    rows: list[dict[str, Any]]
    actions: list[NDArray[np.float64]]
    images: NDArray[np.uint8]
    proprio: NDArray[np.float32]


def closed_loop_record(
    ckpt: Path,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    arm: str,
    run_seed: int,
    sha256: str,
    workers: int,
    chunk: int = 10,
    every: int = DIAG_TA,
) -> ClosedLoopRecord:
    """Fly a checkpoint exactly as selection does, recording actions and replan inputs.

    Same spec, tags, render and contract as :func:`ila.train.select.fly_checkpoint` (so the
    rows must equal the selection rows); spawn pool, one torch thread per worker.

    Args:
        ckpt: The checkpoint (EMA weights; config defaults ``Ta = 8``, ``K = 10``).
        episodes: Listed episodes.
        arm: Arm label.
        run_seed: The run seed.
        sha256: Checkpoint SHA-256.
        workers: Processes (rows do not depend on it).
        chunk: Episodes per task.
        every: Record the input at steps where :func:`chunking.is_replan_step` ``(t, every)``.

    Returns:
        The record.
    """
    from ila.policies.registry import read_checkpoint
    from ila.train.select import checkpoint_policy

    spec = p8_adapter.callable_spec(
        arm, functools.partial(checkpoint_policy, str(ckpt)), run_seed=run_seed
    )
    tags = runner.RowTags(arm=arm, teacher="T_ppo", student_seed=run_seed, ckpt_sha256=sha256)
    vision_arm = runner.VisionArm(
        spec=spec,
        episodes=episodes,
        tags=tags,
        render="full",
        contract_variant=str(read_checkpoint(ckpt)["obs_variant"]),
    )
    tasks = runner.chunk_tasks([vision_arm], p8_adapter.load_eval_configs(), chunk)
    runner.set_single_thread_env()
    jobs = [(t, every) for t in tasks]
    results: list[tuple[list[dict[str, Any]], list[dict[str, list[NDArray[Any]]]]]] = []
    if workers == 1 or len(jobs) <= 1:
        runner.init_worker_threads()
        results = [_fly_recorded(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(workers, len(jobs)), initializer=runner.init_worker_threads
        ) as pool:
            results = list(pool.imap(_fly_recorded, jobs, chunksize=1))
    rows = [r for rs, _ in results for r in rs]
    eps = [e for _, es in results for e in es]
    actions = [np.asarray(e["actions"], dtype=np.float64) for e in eps]
    images = np.asarray([im for e in eps for im in e["images"]], dtype=np.uint8)
    proprio = np.asarray([p for e in eps for p in e["proprio"]], dtype=np.float32)
    return ClosedLoopRecord(rows, actions, images, proprio)


def spread_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise :func:`spread_and_clip` output (quantiles per state; fractions as is)."""
    out: dict[str, Any] = {
        "spread_executed": quantiles(rec["spread_executed"]),
        "spread_first": quantiles(rec["spread_first"]),
        "clip_chunk": rec["clip_chunk"],
        "clip_executed": rec["clip_executed"],
    }
    if "bias_executed" in rec:
        out["bias_executed"] = quantiles(rec["bias_executed"])
        out["label_clip_executed"] = rec["label_clip_executed"]
    return out


def draw_states(n: int, k: int = SPREAD_STATES) -> NDArray[np.int64]:
    """``min(k, n)`` state indices by ``default_rng(0)`` (P4-D5's draw)."""
    return sample_indices(n, COPYCAT_SEED, k)


def _default_out_root() -> Path:
    """Where :func:`run_diagnostics` writes (``<arm>/s<n>.json``)."""
    from ila.policies.registry import REPO_ROOT

    return REPO_ROOT / "results" / "phase6" / "diagnostics"


def run_diagnostics(
    run_dir: Path,
    *,
    data_root: Path | None = None,
    workers: int = 24,
    closed_loop: bool = True,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Measure a selected ``dp`` run's checkpoint and write ``<out_root>/<arm>/s<n>.json``.

    Args:
        run_dir: A selected run (``manifest.json``, ``selection.json``, ``select/``).
        data_root: ``${data_root}`` (default from ``configs/paths.local.yaml``).
        workers: Closed-loop processes (rows do not depend on it).
        closed_loop: Re-fly ``select`` with the recorder (module docstring).
        out_root: Output root (default ``results/phase6/diagnostics``).

    Returns:
        The record (also written).

    Raises:
        ValueError: If the checkpoint's bytes differ from the selection's SHA-256, or it is not
            a diffusion checkpoint.
        AssertionError: If the recorded flight's rows differ from the selection rows, or its
            mean first differences differ from the rows' ``action_jerk_mean``.
    """
    import csv
    import io
    import time

    from ila.data import episodes as data_episodes
    from ila.policies.registry import load_student, read_checkpoint
    from ila.train import manifest as mf
    from ila.train.config import sha256_file
    from ila.train.diagnostics import heldout_states
    from ila.train.dp_sweeps import compare_to_selection
    from ila.train.select import SELECT_SPLIT, write_rows_csv

    t0 = time.perf_counter()
    man = mf.read_json(run_dir / "manifest.json")
    sel = mf.read_json(run_dir / "selection.json")
    cand = sel["selected"]["candidate"]
    ckpt = Path(cand["file"])
    sha = str(cand["sha256"])
    if sha256_file(ckpt) != sha:
        raise ValueError(f"{ckpt}: SHA-256 differs from selection.json")
    student = load_student(ckpt, "cpu", weights="ema")
    if not isinstance(student, DPStudent):
        raise ValueError(f"{ckpt}: not a diffusion checkpoint")
    if (student.chunk_spec.n_action_steps, student.num_inference_steps) != (DIAG_TA, DIAG_K):
        raise ValueError("diagnostics measure the selection setting Ta = 8, K = 10")
    arm, seed = str(man["arm"]), int(man["run_seed"])
    net = student.net
    root = p8_adapter.load_paths()["data_root"] if data_root is None else data_root

    # Held-out states (P4-D5's set), student path only, under the checkpoint's contract.
    held = heldout_states(Path(root), obs_variant=str(read_checkpoint(ckpt)["obs_variant"]))
    data = DPChunkDataset(held.table, held.n_obs_steps, net.cfg.horizon)
    idx = draw_states(len(data))
    b = data.batch(idx)
    held_spread = spread_and_clip(
        net, seed, np.asarray(b["images"]), np.asarray(b["proprio"]), np.asarray(b["target"])
    )
    replay = open_loop_actions(student, data)
    record: dict[str, Any] = {
        "arm": arm,
        "run_seed": seed,
        "run_dir": str(run_dir),
        "checkpoint": {"file": str(ckpt), "sha256": sha, "global_step": cand["global_step"]},
        "config": man["config"],
        "train_git": man["git"],
        "diagnostics_git": mf.git_state(),
        "torch_threads": torch.get_num_threads(),
        "settings": {
            "Ta": DIAG_TA,
            "K": DIAG_K,
            "ensemble": False,
            "weights": "ema",
            "path": "cpu fp32 eval mode (centre crop)",
            "samples_per_state": SPREAD_SAMPLES,
            "states": SPREAD_STATES,
            "state_draw": "numpy default_rng(0), P4-D5",
            "spread_noise_key": f"(run_seed, {SPREAD_EPISODE_KEY}, 16 i + j)",
            "open_loop_noise_key": f"(run_seed, {OPEN_LOOP_KEY_BASE} - e, t)",
            "clip_bound": f"|a| >= 1 - {CLIP_TOL}",
        },
        "definitions": "ila.train.dp_diagnostics module docstring; actions normalised "
        "(x v_max = 1.5 m/s model scale), one control step = 1/30 s model scale",
        "heldout": {
            "states": "ppo_2000 stream indices 1950-1999, every second episode (25)",
            "episodes": data.table.n_episodes,
            "frames": len(data),
            "spread_states": len(idx),
            **spread_record(held_spread),
            "open_loop_jerk": jerk_summary([chunk_jerk(a, DIAG_TA) for a, _ in replay]),
            "teacher_label_jerk": jerk_summary([chunk_jerk(lab, DIAG_TA) for _, lab in replay]),
        },
    }
    if closed_loop:
        episodes = data_episodes.read_split(SELECT_SPLIT)
        flight = closed_loop_record(
            ckpt, episodes, arm=arm, run_seed=seed, sha256=sha, workers=workers
        )
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        rows_file = diag_dir / "select_recorded_rows.csv"
        write_rows_csv(rows_file, flight.rows)
        with rows_file.open(newline="", encoding="utf-8") as f:
            ours = list(csv.DictReader(f))
        sel_rows_file = run_dir / "select" / sel["selected"]["rows_file"]
        with sel_rows_file.open(newline="", encoding="utf-8") as f:
            theirs = list(csv.DictReader(io.StringIO(f.read())))
        diffs = compare_to_selection(ours, theirs)
        if diffs:
            raise AssertionError(f"recorded flight differs from {sel_rows_file}: {diffs[:3]}")
        mean_d1 = [
            float(np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1))) if len(a) > 1 else np.nan
            for a in flight.actions
        ]
        row_jerk = [float(r["action_jerk_mean"]) for r in ours]
        if not np.allclose(mean_d1, row_jerk, rtol=1e-12, atol=0.0, equal_nan=True):
            raise AssertionError("recorded actions disagree with the rows' action_jerk_mean")
        sidx = draw_states(int(flight.images.shape[0]))
        cl_spread = spread_and_clip(net, seed, flight.images[sidx], flight.proprio[sidx])
        executed = np.concatenate(flight.actions) if flight.actions else np.zeros((0, 3))
        record["select"] = {
            "list": SELECT_SPLIT,
            "episodes": len(flight.rows),
            "rows_match_selection": True,
            "rows_file": str(rows_file),
            "action_jerk_mean_matches_rows": True,
            "replan_states": int(flight.images.shape[0]),
            "spread_states": len(sidx),
            **spread_record(cl_spread),
            "executed_at_bound": float((np.abs(executed) >= 1.0 - CLIP_TOL).mean())
            if executed.size
            else None,
            "closed_loop_jerk": jerk_summary([chunk_jerk(a, DIAG_TA) for a in flight.actions]),
            "action_jerk_mean_rows": quantiles(row_jerk),
            "workers": workers,
        }
    record["wall_clock_s"] = time.perf_counter() - t0
    out = (_default_out_root() if out_root is None else out_root) / arm / f"s{seed}.json"
    mf.write_json_atomic(out, record)
    return record
