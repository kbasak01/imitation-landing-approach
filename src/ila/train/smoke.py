"""``make train-smoke ARM=<arm>``: overfit 8 episodes, then land them closed loop (Phase 4).

The gating pre-flight before any multi-seed run (plan Phase 4 item 2): train the arm's frozen
architecture, optimiser, batch size, precision and EMA on the first 8 ``ppo_2000`` ``train``
episodes (stream indices 0-7, success-filtered as the config says), for a smoke-only step
count, and require

0. the il-conventions checklist item 1: stored steps 0-3 of the first episode (pixels, proprio,
   labels) byte-identical to a fresh teacher rollout of it (:func:`replay_check`);
1. action MSE < 1e-4 over every frame of those episodes, EMA weights, eval mode (centre crop),
   fp32 on the CPU -- the weights and path that fly;
2. at least 7 of the 8 episodes ``success`` closed loop, flown on the **CPU fp32 single-sample
   path** through :func:`ila.eval.runner.run_list_vision` from the saved ``ema_last.pt``, with
   the outcome classes printed.

``dp`` (Phase 6; ``results/phase6/smoke_dp.json``; P6-D1) replaces criterion 1, since a
diffusion student has no deterministic action, by the **sampled executed-chunk MSE**: at every
frame ``t`` of the 8 episodes, one chunk sampled from the EMA weights on the CPU in fp32 (eval
mode, centre crop) with DDIM ``K = 10`` (trailing) and the flight's noise key
``(run_seed = 0, episode_seed, t)``; its executed indices at ``Ta = 8``
(:func:`ila.policies.dp.chunking.executed_slice`: ``To-1 ... To+6``, steps ``t ... t+7``)
against ``label`` at those steps (edge-repeated past the end, :class:`DPChunkDataset`); the
mean over frames, steps and the 3 axes must be below :data:`DP_MSE_THRESHOLD` (normalised
action units squared). Criterion 2 flies at the selection setting ``Ta = 8``, ``K = 10``,
no ensembling. Criterion 0 is unchanged.

A student that cannot land the episodes it memorised has an alignment bug, not a capacity
problem (il-conventions, "can't land" checklist). Only the step count differs from the real
runs; it is recorded as ``steps_override`` with purpose ``smoke``, which can never be selected.
The run lives in ``artifacts/runs/smoke/<arm>/`` (replaced each time); the summary goes to
``results/phase4/smoke_<arm>.json``.

``bc_dagger`` (Phase 5, :func:`run_dagger_smoke`; ``results/phase5/smoke_bc_dagger.json``) has
the ``bc`` architecture, whose smoke passed in Phase 4; what is new is the loop and the
on-policy data. Its smoke therefore checks:

A. **the loop**: 2 iterations x 4 rollout episodes from the real ``bc`` s0 ``s0_step030000.pt``
   (purpose ``smoke``, a smoke-only step count): rollouts flown by the right checkpoints
   (iteration 1: the ``bc`` file; iteration 2: this run's iteration-1 last step), audits clean,
   1 000 + 4 i labelled episodes, held-out overlap 0; then a **resume between stages** (the
   iteration-2 checkpoints removed, the run restarted) must reuse iteration 2's rollouts;
B. **the on-policy data's alignment**: the ``bc`` architecture overfit on those 8 rollout
   episodes (labels = teacher, actions executed by the beta-mixture) to action MSE < 1e-4, then
   flown closed loop from their 8 starting conditions: >= 7/8 ``success``. A student that cannot
   land states it memorised the teacher's labels on points at the rollout store's alignment.

Rollouts go to a scratch data root (``artifacts/runs/smoke/bc_dagger/data``, whose ``demos``
links to the real ``${data_root}/demos``), never to the real ``${data_root}/dagger/``.

``modular`` (Phase 7, P7-D1; ``results/phase7/smoke_modular.json``) trains the estimator on the
same 8 episodes and replaces criterion 1, since its output is not an action, by the
**normalised estimator MSE**: the EMA estimator's output (CPU fp32, eval mode, centre crop)
against ``(priv[t] - mean) / std`` (the run's fixed target normaliser) over every frame, mean
over frames and the 11 dimensions, below :data:`MODULAR_MSE_THRESHOLD`; the per-field RMSE in
P8 units is reported beside it. Criterion 0 adds the target side: stored ``priv`` of steps 0-3
byte-identical to the privileged fields of a fresh teacher rollout
(:func:`ila.train.modular_targets.replay_priv_check`). Criterion 2 flies the estimator with
P8's frozen ``T_ppo`` through ``act_modular`` (the true ``time_fraction`` to the teacher only),
CPU fp32 single-sample path, and needs >= 7/8 ``success``.

``R1_lerobot_dp`` (Phase 6, P6-D3, :func:`run_r1_smoke`;
``results/phase6/smoke_R1_lerobot_dp.json``)
checks R1's whole pipeline end to end: LeRobot's training on the LeRobotDataset's episodes 0-7
(``scripts/r1/train_r1.py --purpose smoke`` in ``.venv-lerobot``, 10 000 steps, every other
setting the R1 config's), the ONNX export (``scripts/r1/export_onnx.py``, ``.venv-lerobot``),
parity (:func:`ila.deploy.r1_parity.run_parity`), and the flight through
:class:`ila.policies.onnx_dp.OnnxDPStudent` at ``Ta = 8``, ``K = 10`` on the CPU path. It passes
iff (0) the replay check, (1) LeRobot episodes 0-7 are the store's first 8 kept episodes (the
``dp`` smoke's), (2) the export's parity passed, and (3) >= 7/8 ``success`` closed loop. The
sampled executed-chunk MSE (``dp``'s definition, through the ONNX student) is reported over all
frames and over the frames LeRobot samples as anchors (``drop_n_last_frames = 7``), not judged.
"""

from __future__ import annotations

import functools
import shutil
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.data.store import DemoStore
from ila.obs.split import load_contract
from ila.policies.dp.policy import DPStudent
from ila.policies.modular import ModularStudent
from ila.policies.registry import REGISTRY, REPO_ROOT, load_student, read_checkpoint
from ila.policies.state_mlp import StateMLPStudent
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset, DPChunkDataset, FrameTable
from ila.train.modular_targets import EstimatorDataset, replay_priv_check
from ila.train.sampling import choose_episodes
from ila.train.select import OUTCOME_ORDER, fly_checkpoint, summarise_rows
from ila.train.state_dataset import StateFrames

__all__ = [
    "R1_ARM",
    "R1_SMOKE_STEPS",
    "run_r1_smoke",
    "DAGGER_SMOKE_ITERATIONS",
    "DAGGER_SMOKE_ROLLOUTS",
    "DAGGER_SMOKE_STEPS",
    "DP_MSE_THRESHOLD",
    "DP_SMOKE_K",
    "DP_SMOKE_TA",
    "MIN_LANDED",
    "MODULAR_MSE_THRESHOLD",
    "MSE_THRESHOLD",
    "REPLAY_STEPS",
    "SMOKE_EPISODES",
    "SMOKE_STEPS",
    "dp_sampled_mse",
    "replay_check",
    "run_dagger_smoke",
    "run_smoke",
]

#: Stored steps compared with a fresh teacher rollout (il-conventions checklist item 1).
REPLAY_STEPS: Final[int] = 4

#: Episodes overfitted and flown.
SMOKE_EPISODES: Final[int] = 8

#: Action MSE the overfit must reach (normalised action units squared).
MSE_THRESHOLD: Final[float] = 1e-4

#: Closed-loop successes required of the 8.
MIN_LANDED: Final[int] = 7

#: Default smoke step counts (smoke-only; never a real run's). The constant-decay EMA (0.999)
#: starts at the random init and is not bias-corrected, so after ``n`` steps the init still
#: weighs ``0.999^n``: 1.8 % at 4 000 steps (measured: EMA action MSE 5.1e-5 and actions 3 %
#: short, against 3.1e-6 for the raw weights), 3.4e-4 at 8 000, 1e-13 at the real 30 000. The
#: smoke runs long enough for that to vanish, so it tests the weights that will fly.
#: ``dp`` (P6-D1): 10 000 steps at the frozen batch 128 is about 2 600 passes over the ~490
#: frames of 8 episodes (13 min at the timing run's 12.79 steps/s). Its power-law EMA (0.75)
#: has no init bias to wait out: after 10 000 updates its decay is 0.999, a window of about
#: 1 000 steps at the tail of the cosine schedule, where the weights barely move.
#: ``modular`` (P7-D1): ``bc``'s 8 000, for the same reason (same encoder, batch, constant EMA).
SMOKE_STEPS: Final[dict[str, int]] = {"bc": 8000, "bc_state": 10000, "dp": 10000, "modular": 8000}

#: ``modular`` overfit criterion (P7-D1): estimator MSE in normalised target units (each
#: dimension divided by its training-set std), mean over frames and the 11 dimensions. 1e-3 is
#: an RMS of 0.032 std per dimension. The binding criterion is the closed-loop one.
MODULAR_MSE_THRESHOLD: Final[float] = 1e-3

#: ``dp`` overfit criterion (P6-D1): sampled executed-chunk MSE, normalised action units
#: squared (RMS 0.032 = 0.047 m/s model scale per axis at v_max = 1.5 m/s). Ten times the BC
#: threshold, because a K = 10 DDIM sample adds the sampler's error (the last step's x0 estimate
#: carries sqrt(1 - abar_9) = 0.15 times the epsilon error) to the fit, and it is scored over 8
#: steps of open-loop prediction, not one.
DP_MSE_THRESHOLD: Final[float] = 1e-3

#: ``dp`` smoke inference setting: the selection setting (P3-D1 §4).
DP_SMOKE_TA: Final[int] = 8
DP_SMOKE_K: Final[int] = 10

#: ``R1_lerobot_dp`` smoke (P6-D3): the arm, its smoke steps (``dp``'s, P6-D1) and LeRobot's
#: ``drop_n_last_frames`` (frames never sampled as anchors, reported in the MSE split).
R1_ARM: Final[str] = "R1_lerobot_dp"
R1_SMOKE_STEPS: Final[int] = 10000
R1_DROP_LAST: Final[int] = 7

#: The demo split whose rows are the smoke episodes' listed starting conditions.
SMOKE_SPLIT: Final[str] = "demos_0-1999"

#: ``bc_dagger`` smoke: iterations, rollout episodes per iteration, steps per iteration.
DAGGER_SMOKE_ITERATIONS: Final[int] = 2
DAGGER_SMOKE_ROLLOUTS: Final[int] = 4
DAGGER_SMOKE_STEPS: Final[int] = 200


class _TeacherRecorder:
    """Fly a teacher (reset at t = 0, called once per step, in order) and record steps 0-3.

    Records, per step, the frame's pixels, the contract's proprio of the observation and the
    teacher's action (the label, float32 as collection stores it). The observation itself is
    used only to label (the teacher's input) and to select the proprio fields by name.
    """

    needs_frames: bool = True

    def __init__(self, policy: Any, contract: Any, n_steps: int) -> None:
        self.policy = policy
        self.contract = contract
        self.n_steps = n_steps
        self.images: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

    def begin(self, listed: p8_adapter.ListedEpisode, context: object) -> None:
        if context is not None:
            raise ValueError("the teacher here is non-privileged; no context")
        self.policy.reset(listed.episode_seed, None)

    def act(self, step: int, obs: np.ndarray, frame: Any) -> np.ndarray:
        action = np.asarray(self.policy.act(obs), dtype=np.float64)
        if step < self.n_steps:
            if frame is None:
                raise ValueError("replay needs frames (render 'full')")
            self.images.append(np.array(frame.rgb, dtype=np.uint8, copy=True))
            self.proprio.append(self.contract.proprio(np.asarray(obs, np.float32)))
            self.labels.append(action.astype(np.float32))
        return action


def replay_check(
    store: DemoStore,
    name: str,
    listed: p8_adapter.ListedEpisode,
    teacher_name: str,
    n_steps: int = REPLAY_STEPS,
) -> dict[str, Any]:
    """Stored steps ``0 .. n-1`` of one episode vs a fresh teacher rollout of it.

    il-conventions "can't land" checklist item 1: ``image[t]``, ``proprio[t]`` and
    ``label[t]`` from the store (student read path and labels) must equal, byte for byte, the
    frame, the contract proprio and the teacher's action of a fresh rollout of the same listed
    episode flown by the same teacher through the runner (render ``full``).

    Args:
        store: The demo store.
        name: Episode group name.
        listed: Its listed episode (starting conditions).
        teacher_name: The P8 teacher that flew and labelled it (``ppo`` for ``ppo_2000``).
        n_steps: Steps compared.

    Returns:
        ``{episode, teacher, steps, images_identical, proprio_identical, labels_identical,
        max_abs_label_diff, passed}``.
    """
    from ila.eval import runner  # lazy: runner imports pybullet

    inputs = store.student_inputs(name)
    labels = store.labels(name)
    n = min(n_steps, int(labels.shape[0]))
    task = runner.VisionTask(
        episodes=[listed],
        spec=p8_adapter.callable_spec(
            "smoke:replay", functools.partial(p8_adapter.teacher, teacher_name, 0), run_seed=0
        ),
        cfgs=p8_adapter.load_eval_configs(),
        tags=runner.RowTags(arm="smoke:replay"),
        render="full",
    )
    recorders: list[_TeacherRecorder] = []

    def make(cfgs: p8_adapter.EvalConfigs, contract: Any) -> tuple[Any, object]:
        policy = p8_adapter.teacher(teacher_name, 0, cfgs)
        rec = _TeacherRecorder(policy, contract, n)
        recorders.append(rec)
        return rec, policy

    runner.fly_chunk(task, make)
    rec = recorders[0]
    images = [bool(np.array_equal(rec.images[t], inputs.images[t])) for t in range(n)]
    proprio = [bool(rec.proprio[t].tobytes() == inputs.proprio[t].tobytes()) for t in range(n)]
    lab = [bool(rec.labels[t].tobytes() == labels[t].tobytes()) for t in range(n)]
    diff = float(np.max(np.abs(np.stack(rec.labels[:n]) - labels[:n])))
    return {
        "episode": name,
        "teacher": teacher_name,
        "steps": n,
        "images_identical": sum(images),
        "proprio_identical": sum(proprio),
        "labels_identical": sum(lab),
        "max_abs_label_diff": diff,
        "passed": all(images) and all(proprio) and all(lab),
    }


def dp_sampled_mse(
    student: Any,
    data: DPChunkDataset,
    episode_seeds: list[int],
    *,
    n_action_steps: int = DP_SMOKE_TA,
    num_inference_steps: int = DP_SMOKE_K,
    batch: int = 64,
) -> dict[str, Any]:
    """Sampled executed-chunk MSE of a ``DPStudent`` over every frame of a chunk dataset.

    Frame ``t`` of episode ``e`` is sampled with the flight's noise key
    ``(student.run_seed, episode_seeds[e], t)`` (:func:`ila.policies.dp.policy.initial_noise`),
    so the chunk is the one the policy would plan if it replanned at ``t``. Batched on the
    student's device (CPU fp32 here); a slot's noise does not depend on the batch.

    Args:
        student: A ``DPStudent`` (EMA weights, eval mode).
        data: The episodes' samples (targets ``label`` at :func:`chunking.target_steps`).
        episode_seeds: P8 episode seed of each episode of ``data.table``, in order.
        n_action_steps: ``Ta`` whose executed slice is scored.
        num_inference_steps: ``K``.
        batch: Samples per call (memory only).

    Returns:
        ``value`` (mean over frames, executed steps and axes; normalised action units
        squared), ``first_step`` (index ``To-1`` only, the action at ``t``), ``per_episode``,
        ``per_frame`` (each frame's mean over its executed steps and axes, frame order of
        ``data``), ``frames``, ``Ta``, ``K``.
    """
    from ila.policies.dp import chunking

    table = data.table
    if len(episode_seeds) != table.n_episodes:
        raise ValueError(f"{len(episode_seeds)} seeds for {table.n_episodes} episodes")
    keys: list[tuple[int, int, int]] = []
    for e in range(table.n_episodes):
        keys += [(student.run_seed, int(episode_seeds[e]), t) for t in range(table.episode_len[e])]
    ex = chunking.executed_slice(data.n_obs_steps, data.horizon, n_action_steps)
    errs = []
    for start in range(0, len(data), batch):
        idx = np.arange(start, min(len(data), start + batch), dtype=np.int64)
        b = data.batch(idx)
        chunk = student.sample_batch(
            b["images"], b["proprio"], keys[start : start + len(idx)], num_inference_steps
        )
        diff = chunk[:, ex].astype(np.float64) - np.asarray(b["target"])[:, ex].astype(np.float64)
        errs.append(diff**2)
    err = np.concatenate(errs)  # [N, Ta, 3]
    per_episode = []
    start = 0
    for t_len in table.episode_len:
        per_episode.append(float(err[start : start + int(t_len)].mean()))
        start += int(t_len)
    return {
        "value": float(err.mean()),
        "first_step": float(err[:, 0].mean()),
        "per_episode": per_episode,
        "per_frame": [float(v) for v in err.mean(axis=(1, 2))],
        "frames": len(data),
        "Ta": n_action_steps,
        "K": num_inference_steps,
        "noise_key": "(run_seed, episode_seed, t)",
    }


def run_smoke(
    arm: str,
    *,
    steps: int | None = None,
    device: str = "cuda",
    workers: int = 8,
    run_root: Path = mf.RUNS_ROOT,
    out_json: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Run the smoke for one arm and write its summary JSON.

    Args:
        arm: ``bc``, ``bc_state``, ``dp`` (Phase 6: Ta = 8, K = 10), ``bc_dagger``,
            ``modular`` (Phase 7) or an ablation (its base's criteria, its own data/variant).
        steps: Smoke steps (default :data:`SMOKE_STEPS`).
        device: Training device.
        workers: Closed-loop rollout processes (rows do not depend on it).
        run_root: Runs root (the run goes to ``<run_root>/smoke/<arm>``).
        out_json: Summary path (default ``results/phase4/smoke_<arm>.json``; ``dp``:
            ``results/phase6/smoke_dp.json``).
        data_root: ``${data_root}`` (default from ``configs/paths.local.yaml``).

    Returns:
        The summary (``passed`` true iff both criteria hold).

    Raises:
        ValueError: For an arm without a smoke here.
    """
    entry = REGISTRY[arm]
    if entry.kind == "onnx_dp":
        return run_r1_smoke(
            steps=steps,
            device=device,
            workers=workers,
            run_root=run_root,
            out_json=out_json,
            data_root=data_root,
        )
    if arm == "bc_dagger":
        return run_dagger_smoke(
            steps=steps,
            device=device,
            workers=workers,
            run_root=run_root,
            out_json=out_json,
            data_root=data_root,
        )
    cfg = TrainConfig.load(entry.config)
    if cfg.iterations is not None:
        raise ValueError(f"{arm} is the bc architecture on bc's iteration 0; its smoke is bc's")
    root = p8_adapter.load_paths()["data_root"] if data_root is None else data_root
    run_dir = run_root / "smoke" / arm
    shutil.rmtree(run_dir, ignore_errors=True)
    default_steps = SMOKE_STEPS.get(arm, SMOKE_STEPS.get(entry.kind, SMOKE_STEPS["bc"]))
    n_steps = default_steps if steps is None else steps
    spec = loop.RunSpec(
        cfg=cfg,
        run_seed=0,
        run_dir=run_dir,
        data_root=root,
        device=device,
        purpose="smoke",
        steps_override=n_steps,
        checkpoint_every_override=n_steps,
        first_n_override=SMOKE_EPISODES,
        allow_dirty=True,
    )
    t0 = time.perf_counter()
    result = loop.train_run(spec, loop.build_stages(spec))
    train_s = time.perf_counter() - t0
    ckpt = run_dir / "ema_last.pt"

    store_file = loop.store_path(root, cfg.collection)
    with DemoStore(store_file) as store:
        choice = choose_episodes(store, SMOKE_EPISODES, cfg.success_filter)
        rows_all = data_episodes.read_split(SMOKE_SPLIT)
        first = choice.names[0]
        replay = replay_check(
            store,
            first,
            rows_all[choice.demo_indices[0]],
            store.meta(first).episode_teacher,  # reporting/selection use of meta only
        )
        n_obs = load_contract(variant=cfg.obs_variant).n_obs_steps
        replay_priv = None
        if entry.kind == "state":
            data: Any = StateFrames(store, choice.names)
        elif entry.kind == "modular":
            data = EstimatorDataset.from_store(store, choice.names, n_obs)
            replay_priv = replay_priv_check(
                store, first, rows_all[choice.demo_indices[0]], store.meta(first).episode_teacher
            )
        elif entry.kind == "dp":
            data = DPChunkDataset(
                FrameTable.from_store(store, choice.names, cfg.obs_variant),
                n_obs,
                int(cfg.raw["horizons"]["horizon"]),
            )
        else:
            data = BCWindowDataset(
                FrameTable.from_store(store, choice.names, cfg.obs_variant), n_obs
            )
    student = load_student(ckpt, "cpu", weights="ema")
    import torch

    torch.set_num_threads(1)
    listed = [rows_all[i] for i in choice.demo_indices]
    inference: dict[str, Any] | None = None
    estimator: dict[str, Any] | None = None
    if isinstance(student, ModularStudent):
        assert isinstance(data, EstimatorDataset)
        estimator = modular_overfit(student, data)
        mse, per_episode_mse = estimator["value"], estimator["per_episode"]
        threshold = MODULAR_MSE_THRESHOLD
    elif isinstance(student, DPStudent):
        dp_mse = dp_sampled_mse(student, data, [e.episode_seed for e in listed])
        mse, per_episode_mse = dp_mse["value"], dp_mse["per_episode"]
        threshold = DP_MSE_THRESHOLD
        # The flight's student (select.checkpoint_policy) is built at the config defaults.
        inference = {
            "Ta": student.chunk_spec.n_action_steps,
            "K": student.num_inference_steps,
            "ensemble": student.temporal_ensembling,
            "timestep_spacing": student.dp_cfg.timestep_spacing,
        }
        if (inference["Ta"], inference["K"], inference["ensemble"]) != (
            DP_SMOKE_TA,
            DP_SMOKE_K,
            False,
        ):
            raise ValueError(f"dp smoke must fly Ta = 8, K = 10, no ensembling: {inference}")
    else:
        if isinstance(student, StateMLPStudent):
            pred = student.act_batch(data.obs)
        else:
            pred = loop.numpy_actions(student.net, data, torch.device("cpu"))
        target = data.batch(np.arange(len(data), dtype=np.int64))["target"]
        err = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
        mse = float(err.mean())
        threshold = MSE_THRESHOLD
        per_episode_mse = []
        start = 0
        lengths = data.episode_len if entry.kind == "state" else data.table.episode_len
        for t_len in lengths:
            per_episode_mse.append(float(err[start : start + int(t_len)].mean()))
            start += int(t_len)

    t1 = time.perf_counter()
    rows = fly_checkpoint(
        ckpt,
        listed,
        arm=f"smoke:{arm}",
        run_seed=0,
        sha256=sha256_file(ckpt),
        workers=workers,
        chunk=1,
    )
    fly_s = time.perf_counter() - t1
    summary = summarise_rows(rows)
    man = mf.read_json(run_dir / "manifest.json")
    record: dict[str, Any] = {
        "arm": arm,
        "what": f"overfit the first 8 {cfg.collection} train episodes "
        f"(contract {cfg.obs_variant!r}), then fly them closed loop "
        "(CPU fp32 single-sample path, EMA weights from ema_last.pt)",
        "config": man["config"],
        "frozen_hyperparameters_unchanged": True,
        "overrides": {"steps": n_steps, "first_n": SMOKE_EPISODES, "purpose": "smoke"},
        "episodes": choice.summary(),
        "train": {
            "state": result.state,
            "steps": result.global_step,
            "wall_s": train_s,
            "loop_s": result.loop_s,
            "steps_per_s": result.steps_timed / result.loop_s if result.loop_s else None,
            "device": device,
            "precision": cfg.precision,
            "batch_size": cfg.batch_size,
            "gpu": man["host"].get("gpu"),
            "gpu_mem_max_bytes": man.get("gpu_mem_max_bytes"),
        },
        "mse": {
            "value": mse,
            "threshold": threshold,
            "passed": mse < threshold,
            "per_episode": per_episode_mse,
            "frames": len(data),
            "weights": "ema",
            "path": "cpu fp32 eval mode (centre crop)",
            **(
                {
                    "what": "sampled executed-chunk MSE (one DDIM sample per frame, flight "
                    "noise key; executed indices To-1 ... To+6 vs label; P6-D1)",
                    "first_step": dp_mse["first_step"],
                    "Ta": dp_mse["Ta"],
                    "K": dp_mse["K"],
                    "noise_key": dp_mse["noise_key"],
                }
                if inference is not None
                else {}
            ),
            **(
                {
                    "what": "EMA estimator MSE in normalised target units (priv minus the "
                    "run's training mean, over its std), mean over frames and 11 dims (P7-D1)",
                    "rmse_by_field": estimator["rmse_by_field"],
                }
                if estimator is not None
                else {}
            ),
        },
        "inference": inference,
        "closed_loop": {
            **summary,
            "outcome_order": list(OUTCOME_ORDER),
            "required_success": MIN_LANDED,
            "passed": summary["success"] >= MIN_LANDED,
            "wall_s": fly_s,
            "workers": workers,
            "per_episode": [
                {
                    "demo_index": i,
                    "ss": r["ss"],
                    "outcome": r["outcome"],
                    "steps": r["steps"],
                    "closing_speed_normal_m_s": r["closing_speed_normal_m_s"],
                }
                for i, r in zip(choice.demo_indices, rows, strict=True)
            ],
        },
        "replay": replay,
        **({} if replay_priv is None else {"replay_priv": replay_priv}),
        "checkpoint": {"file": _repo_relative(ckpt), "sha256": sha256_file(ckpt)},
        "git": man["git"],
    }
    record["passed"] = (
        record["mse"]["passed"]
        and record["closed_loop"]["passed"]
        and replay["passed"]
        and (replay_priv is None or bool(replay_priv["passed"]))
    )
    phase = {"dp": "phase6", "modular": "phase7"}.get(entry.kind, "phase4")
    if entry.tag == "ablation":
        phase = "phase7"
    out = Path("results") / phase / f"smoke_{arm}.json" if out_json is None else out_json
    mf.write_json_atomic(out if out.is_absolute() else REPO_ROOT / out, record)
    return record


def modular_overfit(student: ModularStudent, data: EstimatorDataset) -> dict[str, Any]:
    """The ``modular`` smoke's overfit measure (module docstring) on the CPU, fp32, eval mode.

    Args:
        student: The EMA estimator wrapped as a student (its ``net`` holds the normaliser).
        data: The episodes' samples.

    Returns:
        ``value`` (normalised MSE, mean over frames and dims), ``per_episode``, ``frames`` and
        ``rmse_by_field`` (P8 units, model scale: m, m/s, unit vector, rad, m).
    """
    import torch

    est = []
    for start in range(0, len(data), 256):
        idx = np.arange(start, min(len(data), start + 256), dtype=np.int64)
        b = data.batch(idx)
        images = np.asarray(b["images"], dtype=np.uint8)
        est.append(student.estimate_batch(images, np.asarray(b["proprio"], dtype=np.float32)))
    pred = np.concatenate(est).astype(np.float64)
    target = data.targets.astype(np.float64)
    std = student.net.target_std.cpu().to(torch.float64).numpy()
    err = ((pred - target) / std) ** 2
    per_episode = []
    start = 0
    for t_len in data.table.episode_len:
        per_episode.append(float(err[start : start + int(t_len)].mean()))
        start += int(t_len)
    contract = load_contract()
    rmse: dict[str, float] = {}
    col = 0
    for name in contract.privileged:
        size = contract.layout[name][1]
        rmse[name] = float(np.sqrt(np.mean((pred - target)[:, col : col + size] ** 2)))
        col += size
    return {
        "value": float(err.mean()),
        "per_episode": per_episode,
        "frames": len(data),
        "rmse_by_field": rmse,
    }


def _repo_relative(path: Path) -> str:
    """Return ``path`` relative to the repository root (absolute only if outside it)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# --------------------------------------------------------------------------- bc_dagger


def _scratch_data_root(root: Path, real: Path) -> Path:
    """A data root whose ``demos`` is the real one (read only) and whose ``dagger/`` is new."""
    root.mkdir(parents=True, exist_ok=True)
    link = root / "demos"
    if not link.exists():
        link.symlink_to((real / "demos").resolve(), target_is_directory=True)
    return root


def _rollout_bundle(
    cfg: TrainConfig, stores: list[tuple[Path, int, int]], hash_store: bool
) -> loop.DataBundle:
    """The 8 smoke rollout episodes, unfiltered, as one bc training set (part B)."""
    del hash_store
    tables = []
    choices = []
    for path, start, n in stores:
        with DemoStore(path) as store:
            choice = choose_episodes(store, n, False, start=start)
            tables.append(FrameTable.from_store(store, choice.names, cfg.obs_variant))
            choices.append(choice)
    table = FrameTable.concat(tables)
    train = BCWindowDataset(table, load_contract(variant=cfg.obs_variant).n_obs_steps)
    return loop.DataBundle(
        train=train,
        heldout=None,
        choices=choices,
        labelled_episodes=sum(c.attempted for c in choices),
        stores=[{"path": str(p), "start": a, "n": n} for p, a, n in stores],
    )


def run_dagger_smoke(
    *,
    steps: int | None = None,
    overfit_steps: int | None = None,
    device: str = "cuda",
    workers: int = 8,
    run_root: Path = mf.RUNS_ROOT,
    out_json: Path | None = None,
    data_root: Path | None = None,
    init_from: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """The ``bc_dagger`` smoke (module docstring, parts A and B); writes its summary JSON.

    Args:
        steps: Steps per DAgger iteration (default :data:`DAGGER_SMOKE_STEPS`).
        overfit_steps: Part B's steps (default ``SMOKE_STEPS["bc"]``).
        device: Training device.
        workers: Rollout and closed-loop processes (rows do not depend on it).
        run_root: Runs root (the smoke lives in ``<run_root>/smoke/bc_dagger``).
        out_json: Summary path (default ``results/phase5/smoke_bc_dagger.json``).
        data_root: The real ``${data_root}`` (only its ``demos/`` is read).
        init_from: Iteration 0 (default: ``bc`` s0's ``s0_step030000.pt`` under ``run_root``).
        batch_size: Tests only: a smaller batch than the frozen 256 (recorded).

    Returns:
        The summary (``passed`` iff every check of A and B holds).
    """
    import dataclasses

    import torch

    cfg = TrainConfig.load(REGISTRY["bc_dagger"].config)
    if batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=batch_size)
    real_root = p8_adapter.load_paths()["data_root"] if data_root is None else data_root
    base = run_root / "smoke" / "bc_dagger"
    shutil.rmtree(base, ignore_errors=True)
    scratch = _scratch_data_root(base / "data", Path(real_root))
    bc_steps = TrainConfig.load(REGISTRY["bc"].config).steps
    if init_from is None:
        init_from = mf.run_dir("bc", 0, root=run_root) / "checkpoints" / f"s0_step{bc_steps:06d}.pt"
    n_steps = DAGGER_SMOKE_STEPS if steps is None else steps
    run_dir = base / "run"
    spec = loop.RunSpec(
        cfg=cfg,
        run_seed=0,
        run_dir=run_dir,
        data_root=scratch,
        device=device,
        purpose="smoke",
        steps_override=n_steps,
        checkpoint_every_override=n_steps,
        rollout_limit=DAGGER_SMOKE_ROLLOUTS,
        iterations_override=DAGGER_SMOKE_ITERATIONS,
        rollout_workers=workers,
        init_from=init_from,
        allow_dirty=True,
    )
    checks: dict[str, bool] = {}
    t0 = time.perf_counter()
    result = loop.train_run(spec, loop.build_stages(spec))
    loop_s = time.perf_counter() - t0
    man = mf.read_json(run_dir / "manifest.json")
    recs = {i: man["data"][str(i)]["dagger"] for i in range(1, DAGGER_SMOKE_ITERATIONS + 1)}
    last1 = run_dir / "checkpoints" / f"s1_step{n_steps:06d}.pt"
    checks["loop_done"] = result.state == "done" and result.global_step == 2 * n_steps
    checks["iteration1_flies_bc_last_step"] = recs[1]["rollout_student"]["sha256"] == sha256_file(
        init_from
    )
    checks["iteration2_flies_iteration1_last_step"] = recs[2]["rollout_student"][
        "sha256"
    ] == sha256_file(last1)
    checks["labelled_episodes_1000_plus_4i"] = all(
        recs[i]["labelled_episodes"] == cfg.first_n + DAGGER_SMOKE_ROLLOUTS * i for i in recs
    )
    checks["audits_clean"] = all(recs[i]["audit"]["clean"] is True for i in recs)
    checks["heldout_overlap_0"] = all(recs[i]["heldout_overlap_episodes"] == 0 for i in recs)
    checks["normaliser_inherited"] = "normaliser" not in man

    # A, resume between stages: drop iteration 2's checkpoints, rerun; rollouts must be reused.
    final = run_dir / "checkpoints" / f"s2_step{n_steps:06d}.pt"
    before = read_checkpoint(final)
    final.unlink()
    (run_dir / "ema_last.pt").unlink()
    loop.train_run(spec, loop.build_stages(spec))
    after = read_checkpoint(final)
    man = mf.read_json(run_dir / "manifest.json")
    resumed = man["data"]["2"]["dagger"]
    checks["resume_reuses_rollouts"] = resumed["rollout"]["reused"] is True and (
        resumed["rollout"]["store_sha256"] == recs[2]["rollout"]["store_sha256"]
    )
    ema_identical = all(torch.equal(before["ema"][k], after["ema"][k]) for k in before["ema"])

    # B, overfit the 8 rollout episodes and land them.
    from ila.data import dagger

    stores = [
        (
            dagger.rollout_store_path(scratch, 0, i),
            cfg.iterations.added_range(i).start if cfg.iterations else 0,
            DAGGER_SMOKE_ROLLOUTS,
        )
        for i in range(1, DAGGER_SMOKE_ITERATIONS + 1)
    ]
    bc_cfg = TrainConfig.load(REGISTRY["bc"].config)
    if batch_size is not None:
        bc_cfg = dataclasses.replace(bc_cfg, batch_size=batch_size)
    o_steps = SMOKE_STEPS["bc"] if overfit_steps is None else overfit_steps
    o_dir = base / "overfit"
    o_spec = loop.RunSpec(
        cfg=bc_cfg,
        run_seed=0,
        run_dir=o_dir,
        data_root=scratch,
        device=device,
        purpose="smoke",
        steps_override=o_steps,
        checkpoint_every_override=o_steps,
        heldout=None,
        allow_dirty=True,
    )
    stage = loop.Stage(
        index=0,
        steps=o_steps,
        warmup=bc_cfg.optim.warmup_steps,
        checkpoint_every=o_steps,
        data=functools.partial(_rollout_bundle, bc_cfg, stores),
    )
    t1 = time.perf_counter()
    loop.train_run(o_spec, [stage])
    overfit_s = time.perf_counter() - t1
    ckpt = o_dir / "ema_last.pt"
    data = _rollout_bundle(bc_cfg, stores, False).train
    assert isinstance(data, BCWindowDataset)
    student = load_student(ckpt, "cpu", weights="ema")
    torch.set_num_threads(1)
    assert not isinstance(student, StateMLPStudent)
    pred = loop.numpy_actions(student.net, data, torch.device("cpu"))
    target = data.batch(np.arange(len(data), dtype=np.int64))["target"]
    mse = float(((pred.astype(np.float64) - target.astype(np.float64)) ** 2).mean())
    rows_all = data_episodes.read_split(SMOKE_SPLIT)
    indices = [start + k for _, start, n in stores for k in range(n)]
    t2 = time.perf_counter()
    rows = fly_checkpoint(
        ckpt,
        [rows_all[i] for i in indices],
        arm="smoke:bc_dagger",
        run_seed=0,
        sha256=sha256_file(ckpt),
        workers=workers,
        chunk=1,
    )
    fly_s = time.perf_counter() - t2
    summary = summarise_rows(rows)
    checks["overfit_mse_below_threshold"] = mse < MSE_THRESHOLD
    checks["overfit_lands_7_of_8"] = summary["success"] >= MIN_LANDED
    record: dict[str, Any] = {
        "arm": "bc_dagger",
        "what": "A: 2 DAgger iterations x 4 rollout episodes from bc s0's last step, then a "
        "resume between stages; B: the bc architecture overfit on those 8 rollout episodes, "
        "flown closed loop from their starting conditions (CPU fp32 single-sample path, EMA)",
        "config": man["config"],
        "frozen_hyperparameters_unchanged": batch_size is None,
        "overrides": {
            "steps_per_iteration": n_steps,
            "iterations": DAGGER_SMOKE_ITERATIONS,
            "rollout_limit": DAGGER_SMOKE_ROLLOUTS,
            "overfit_steps": o_steps,
            "batch_size": batch_size,
            "purpose": "smoke",
        },
        "init_from": {"file": _repo_relative(init_from), "sha256": sha256_file(init_from)},
        "loop": {
            "state": result.state,
            "steps": result.global_step,
            "wall_s": loop_s,
            "device": device,
            "iterations": {
                str(i): {
                    "beta": recs[i]["beta"],
                    "rollout_student": recs[i]["rollout_student"],
                    "rollout_outcomes": recs[i]["rollout"]["outcomes"],
                    "labelled_episodes": recs[i]["labelled_episodes"],
                    "labelled_frames_rollouts": recs[i]["labelled_frames_rollouts"],
                    "aggregate_frames": recs[i]["aggregate_frames"],
                    "audit_clean": recs[i]["audit"]["clean"],
                    "heldout_shared_starting_conditions": recs[i][
                        "heldout_shared_starting_conditions"
                    ],
                }
                for i in recs
            },
            "resume": {
                "rollouts_reused": resumed["rollout"]["reused"],
                "final_ema_identical_after_resume": ema_identical,
                "note": "identity is informational: GPU training is not guaranteed "
                "bit-reproducible (P4-D4 correction 1)",
            },
        },
        "overfit": {
            "episodes": [{"store": str(p), "start": a, "n": n} for p, a, n in stores],
            "mse": {"value": mse, "threshold": MSE_THRESHOLD, "frames": len(data)},
            "wall_s": overfit_s,
        },
        "closed_loop": {
            **summary,
            "outcome_order": list(OUTCOME_ORDER),
            "required_success": MIN_LANDED,
            "wall_s": fly_s,
            "workers": workers,
            "per_episode": [
                {
                    "demo_index": i,
                    "ss": r["ss"],
                    "outcome": r["outcome"],
                    "steps": r["steps"],
                    "closing_speed_normal_m_s": r["closing_speed_normal_m_s"],
                }
                for i, r in zip(indices, rows, strict=True)
            ],
        },
        "checkpoint": {"file": _repo_relative(ckpt), "sha256": sha256_file(ckpt)},
        "checks": checks,
        "git": man["git"],
    }
    record["passed"] = all(checks.values())
    out = Path("results") / "phase5" / "smoke_bc_dagger.json" if out_json is None else out_json
    mf.write_json_atomic(out if out.is_absolute() else REPO_ROOT / out, record)
    return record


# --------------------------------------------------------------------------- R1_lerobot_dp


def _venv_lerobot(*args: str) -> None:
    """Run a ``scripts/r1`` script in ``.venv-lerobot`` (raises on a non-zero exit)."""
    import subprocess

    py = REPO_ROOT / ".venv-lerobot" / "bin" / "python"
    subprocess.run([str(py), *args], cwd=REPO_ROOT, check=True)


def run_r1_smoke(
    *,
    steps: int | None = None,
    device: str = "cuda",
    workers: int = 8,
    run_root: Path = mf.RUNS_ROOT,
    out_json: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """The ``R1_lerobot_dp`` smoke (module docstring); writes its summary JSON.

    Args:
        steps: LeRobot training steps (default :data:`R1_SMOKE_STEPS`); also the checkpoint
            cadence, so exactly one checkpoint is exported.
        device: Training device (``.venv-lerobot``).
        workers: Closed-loop rollout processes (rows do not depend on it).
        run_root: Runs root (the run goes to ``<run_root>/smoke/R1_lerobot_dp``, replaced).
        out_json: Summary path (default ``results/phase6/smoke_R1_lerobot_dp.json``).
        data_root: ``${data_root}`` (default from ``configs/paths.local.yaml``).

    Returns:
        The summary (``passed`` iff checks 0-3 hold).
    """
    import json

    import torch

    from ila.deploy import r1_parity
    from ila.policies.onnx_dp import bundle_sha256, load_onnx_student, parity_passed

    root = p8_adapter.load_paths()["data_root"] if data_root is None else data_root
    n_steps = R1_SMOKE_STEPS if steps is None else int(steps)
    run_dir = run_root / "smoke" / R1_ARM
    shutil.rmtree(run_dir, ignore_errors=True)
    checks: dict[str, bool] = {}

    # Episodes: LeRobot episodes 0-7 -> store groups and demo indices (the export manifest),
    # which must be the dp smoke's (the store's first 8 kept episodes).
    inter = json.loads(
        (Path(root) / "lerobot" / "r1_ppo_2000_first1000" / "manifest.json").read_text("utf-8")
    )
    by_index = {int(e["episode_index"]): e for e in inter["episodes"]}
    lr_eps = [by_index[i] for i in range(SMOKE_EPISODES)]
    cfg = TrainConfig.load(REGISTRY["dp"].config)
    store_file = loop.store_path(root, cfg.collection)
    with DemoStore(store_file) as store:
        choice = choose_episodes(store, SMOKE_EPISODES, cfg.success_filter)
        checks["lerobot_episodes_are_dp_smoke_episodes"] = [
            e["store_group"] for e in lr_eps
        ] == list(choice.names) and [int(e["demo_index"]) for e in lr_eps] == list(
            choice.demo_indices
        )
        rows_all = data_episodes.read_split(SMOKE_SPLIT)
        replay = replay_check(
            store,
            choice.names[0],
            rows_all[choice.demo_indices[0]],
            store.meta(choice.names[0]).episode_teacher,  # reporting use of meta only
        )
        n_obs = load_contract().n_obs_steps
        data = DPChunkDataset(
            FrameTable.from_store(store, list(choice.names), "default"),
            n_obs,
            int(cfg.raw["horizons"]["horizon"]),
        )
    checks["replay"] = bool(replay["passed"])

    t0 = time.perf_counter()
    _venv_lerobot(
        "scripts/r1/train_r1.py",
        "--purpose", "smoke",
        "--seed", "0",
        "--episodes", f"0-{SMOKE_EPISODES - 1}",
        "--steps", str(n_steps),
        "--save-freq", str(n_steps),
        "--device", device,
        "--run-dir", str(run_dir),
    )  # fmt: skip
    train_s = time.perf_counter() - t0
    man = mf.read_json(run_dir / "manifest.json")
    t1 = time.perf_counter()
    _venv_lerobot("scripts/r1/export_onnx.py", "export", "--run", str(run_dir))
    export_s = time.perf_counter() - t1
    bundle = run_dir / "onnx" / f"step{n_steps}"
    parity = r1_parity.run_parity(bundle, threads=1)
    checks["parity_passed"] = parity_passed(bundle)[0]

    torch.set_num_threads(1)
    student = load_onnx_student(bundle, 0)
    listed = [rows_all[i] for i in choice.demo_indices]
    mse = dp_sampled_mse(student, data, [e.episode_seed for e in listed])
    # Frames LeRobot samples as anchors: t <= T - 1 - drop_n_last_frames, per episode.
    anchored = []
    start = 0
    for t_len in data.table.episode_len:
        anchored += [start + t for t in range(int(t_len) - R1_DROP_LAST)]
        start += int(t_len)
    mse_anchored = float(np.mean([mse["per_frame"][i] for i in anchored]))
    sha = bundle_sha256(bundle)
    t2 = time.perf_counter()
    rows = fly_checkpoint(
        bundle, listed, arm=f"smoke:{R1_ARM}", run_seed=0, sha256=sha, workers=workers, chunk=1
    )
    fly_s = time.perf_counter() - t2
    summary = summarise_rows(rows)
    checks["closed_loop_7_of_8"] = summary["success"] >= MIN_LANDED
    seg = man.get("segments", [{}])
    record: dict[str, Any] = {
        "arm": R1_ARM,
        "what": "LeRobot trains R1 on LeRobotDataset episodes 0-7 (= the dp smoke's 8 episodes), "
        "export to ONNX, parity, then fly them closed loop through OnnxDPStudent (ORT CPU, one "
        "thread, Ta = 8, K = 10, DDIM trailing, noise key (0, episode_seed, t); raw weights: "
        "lerobot 0.6.1 has no EMA)",
        "config": man["config"],
        "frozen_hyperparameters_unchanged": True,
        "overrides": {
            "steps": n_steps,
            "save_freq": n_steps,
            "episodes": "0-7",
            "purpose": "smoke",
        },
        "episodes": {
            "lerobot_episode_index": [int(e["episode_index"]) for e in lr_eps],
            "store_groups": [e["store_group"] for e in lr_eps],
            "demo_indices": [int(e["demo_index"]) for e in lr_eps],
            **choice.summary(),
        },
        "train": {
            "state": man["state"],
            "steps": man["global_step"],
            "wall_s": train_s,
            "steps_per_s": man.get("steps_per_s_loop"),
            "steps_per_s_after_warmup": seg[-1].get("steps_per_s_after_warmup"),
            "device": device,
            "precision": man["precision"],
            "batch_size": man["batch_size"],
            "gpu": man["host"].get("gpu"),
            "run_dir": _repo_relative(run_dir),
        },
        "export": {"wall_s": export_s, "bundle": _repo_relative(bundle), "meta_sha256": sha},
        "parity": {
            "pass": parity["pass"],
            "judged": parity["judged"],
            "wall_s": parity["wall_s"],
        },
        "mse": {
            "value": mse["value"],
            "first_step": mse["first_step"],
            "per_episode": mse["per_episode"],
            "frames": mse["frames"],
            "anchored_frames_value": mse_anchored,
            "anchored_frames": len(anchored),
            "reference_threshold_dp": DP_MSE_THRESHOLD,
            "judged": False,
            "what": "sampled executed-chunk MSE through OnnxDPStudent (dp's P6-D1 definition, "
            "env action units); anchored = frames with t <= T - 8 (drop_n_last_frames = 7)",
            "Ta": mse["Ta"],
            "K": mse["K"],
        },
        "inference": {
            "Ta": student.chunk_spec.n_action_steps,
            "K": student.num_inference_steps,
            "ensemble": student.temporal_ensembling,
            "timestep_spacing": "trailing",
            "path": "ort_cpu_fp32_single_sample",
        },
        "closed_loop": {
            **summary,
            "outcome_order": list(OUTCOME_ORDER),
            "required_success": MIN_LANDED,
            "passed": checks["closed_loop_7_of_8"],
            "wall_s": fly_s,
            "workers": workers,
            "per_episode": [
                {
                    "demo_index": i,
                    "ss": r["ss"],
                    "outcome": r["outcome"],
                    "steps": r["steps"],
                    "closing_speed_normal_m_s": r["closing_speed_normal_m_s"],
                }
                for i, r in zip(choice.demo_indices, rows, strict=True)
            ],
        },
        "replay": replay,
        "checks": checks,
        "git": man["git"],
    }
    record["passed"] = all(checks.values())
    out = Path("results") / "phase6" / f"smoke_{R1_ARM}.json" if out_json is None else out_json
    mf.write_json_atomic(out if out.is_absolute() else REPO_ROOT / out, record)
    return record
