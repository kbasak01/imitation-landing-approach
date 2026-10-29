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

A student that cannot land the episodes it memorised has an alignment bug, not a capacity
problem (il-conventions, "can't land" checklist). Only the step count differs from the real
runs; it is recorded as ``steps_override`` with purpose ``smoke``, which can never be selected.
The run lives in ``artifacts/runs/smoke/<arm>/`` (replaced each time); the summary goes to
``results/phase4/smoke_<arm>.json``.
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
from ila.policies.registry import REGISTRY, REPO_ROOT, load_student
from ila.policies.state_mlp import StateMLPStudent
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset, FrameTable
from ila.train.sampling import choose_episodes
from ila.train.select import OUTCOME_ORDER, fly_checkpoint, summarise_rows
from ila.train.state_dataset import StateFrames

__all__ = [
    "MIN_LANDED",
    "MSE_THRESHOLD",
    "REPLAY_STEPS",
    "SMOKE_EPISODES",
    "SMOKE_STEPS",
    "replay_check",
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
SMOKE_STEPS: Final[dict[str, int]] = {"bc": 8000, "bc_state": 10000}

#: The demo split whose rows are the smoke episodes' listed starting conditions.
SMOKE_SPLIT: Final[str] = "demos_0-1999"


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
        arm: ``bc`` or ``bc_state`` (``dp``'s smoke is Phase 6: Ta = 8, K = 10).
        steps: Smoke steps (default :data:`SMOKE_STEPS`).
        device: Training device.
        workers: Closed-loop rollout processes (rows do not depend on it).
        run_root: Runs root (the run goes to ``<run_root>/smoke/<arm>``).
        out_json: Summary path (default ``results/phase4/smoke_<arm>.json``).
        data_root: ``${data_root}`` (default from ``configs/paths.local.yaml``).

    Returns:
        The summary (``passed`` true iff both criteria hold).

    Raises:
        ValueError: For an arm without a smoke here.
    """
    entry = REGISTRY[arm]
    if entry.kind == "dp":
        raise ValueError("the dp smoke (closed loop at Ta = 8, K = 10) is Phase 6's")
    cfg = TrainConfig.load(entry.config)
    root = p8_adapter.load_paths()["data_root"] if data_root is None else data_root
    run_dir = run_root / "smoke" / arm
    shutil.rmtree(run_dir, ignore_errors=True)
    n_steps = SMOKE_STEPS.get(arm, 8000) if steps is None else steps
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
        if entry.kind == "state":
            data: Any = StateFrames(store, choice.names)
        else:
            data = BCWindowDataset(
                FrameTable.from_store(store, choice.names, cfg.obs_variant),
                load_contract(variant=cfg.obs_variant).n_obs_steps,
            )
    student = load_student(ckpt, "cpu", weights="ema")
    import torch

    torch.set_num_threads(1)
    if isinstance(student, StateMLPStudent):
        pred = student.act_batch(data.obs)
    else:
        pred = loop.numpy_actions(student.net, data, torch.device("cpu"))
    target = data.batch(np.arange(len(data), dtype=np.int64))["target"]
    err = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    mse = float(err.mean())
    per_episode_mse = []
    start = 0
    lengths = data.episode_len if entry.kind == "state" else data.table.episode_len
    for t_len in lengths:
        per_episode_mse.append(float(err[start : start + int(t_len)].mean()))
        start += int(t_len)

    listed = [rows_all[i] for i in choice.demo_indices]
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
        "what": "overfit the first 8 ppo_2000 train episodes, then fly them closed loop "
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
            "threshold": MSE_THRESHOLD,
            "passed": mse < MSE_THRESHOLD,
            "per_episode": per_episode_mse,
            "frames": len(data),
            "weights": "ema",
            "path": "cpu fp32 eval mode (centre crop)",
        },
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
        "checkpoint": {"file": _repo_relative(ckpt), "sha256": sha256_file(ckpt)},
        "git": man["git"],
    }
    record["passed"] = (
        record["mse"]["passed"] and record["closed_loop"]["passed"] and replay["passed"]
    )
    out = Path("results") / "phase4" / f"smoke_{arm}.json" if out_json is None else out_json
    mf.write_json_atomic(out if out.is_absolute() else REPO_ROOT / out, record)
    return record


def _repo_relative(path: Path) -> str:
    """Return ``path`` relative to the repository root (absolute only if outside it)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()
