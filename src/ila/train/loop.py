"""Training loop: AdamW, cosine with warm-up, bf16 autocast, EMA, checkpoints, resume (Phase 4).

One loop serves every arm and every stage structure:

* a **single-stage** run (``bc``, ``bc_state``, ``bc_dart``, ``dp``, the ablations) is stage 0;
* an **iterated** run (``bc_matched``, ``bc_dagger``: one driver, P4-D5 item 1) is stages
  ``1 .. n``, warm-started from the same-seed ``bc`` run's last step (weights **and** EMA
  copied, AdamW reset, its own cosine schedule with warm-up per stage, frames sampled uniformly
  from the aggregate; P3-D1 §3). A stage builds its data lazily, when it starts, so a DAgger
  stage rolls out the previous stage's last-step EMA first (:mod:`ila.train.dagger_stage`).

Per step (``s`` = optimiser steps already taken in the stage): ``lr = lr_peak * m(s)`` with
diffusers' ``cosine`` multiplier ``m(s) = s / W`` for ``s < W``, else
``0.5 (1 + cos(pi (s - W) / (S - W)))``; forward under ``torch.autocast(bfloat16)`` (fp32
for ``bc_state``); MSE (BC: on the ``tanh`` output against ``label[t]``; DP: epsilon-prediction
DDPM, :func:`ila.policies.dp.policy.ddpm_loss`); AdamW (torch's default betas and eps, recorded);
the gradient norm is measured, never clipped (no config sets clipping); EMA update.

Data randomness is a pure function of ``(run_seed, stage, step)``
(:mod:`ila.train.sampling`), so a resumed run is bit-identical on the CPU (tested). Checkpoints
every ``checkpoint_every`` steps of a stage hold the raw weights, the EMA, AdamW's state and the
RNG states; the last step of the last stage also writes ``ema_last.pt`` (EMA only), which is
iteration 0 of ``bc_matched``/``bc_dagger`` for a ``bc`` run (P3-D1 §3). The EMA weights are
the evaluated weights (P3-D1 §4).

Held-out loss: every ``eval_every`` steps and at each checkpoint, raw and EMA weights, centre
crop, on :data:`HELDOUT` -- ``ppo_2000`` stream indices 1950-1999, ``train``-pool demos outside
every single-stage arm's first 1 000 (never ``tune``, ``select`` or a frozen list). The
iterated ``bc_matched`` trains on them from iteration 5 on; the overlap is counted and logged.
``bc_dagger`` never trains on them (overlap 0), but its iteration 5 flies the same 50 starting
conditions with the student; that count is logged as ``heldout_shared_starting_conditions``.

Units: steps are optimiser steps; times are wall-clock seconds; actions are normalised
world-frame velocity setpoints (x v_max = 1.5 m/s model scale, lambda = 1/25).
"""

from __future__ import annotations

import copy
import functools
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ila.data.store import DemoStore
from ila.policies.bc import BCNet
from ila.policies.dp.policy import DiffusionPolicyNet, ddpm_loss
from ila.policies.registry import CHECKPOINT_FORMAT, Net, build_net_from_config, read_checkpoint
from ila.policies.state_mlp import StateMLP, fit_normaliser
from ila.train import manifest as mf
from ila.train.config import ADAMW_BETAS, ADAMW_EPS, EMAConfig, TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset, DPChunkDataset, FrameTable
from ila.train.sampling import (
    STREAM_CROP,
    STREAM_HELDOUT,
    STREAM_NOISE,
    EpisodeChoice,
    EpochSampler,
    choose_episodes,
    loader_seed,
)
from ila.train.state_dataset import StateFrames

__all__ = [
    "EMA",
    "HELDOUT",
    "PURPOSES",
    "DataBundle",
    "HeldoutSpec",
    "Purpose",
    "RunResult",
    "RunSpec",
    "Stage",
    "build_stages",
    "configure_numerics",
    "find_resume_checkpoint",
    "lr_multiplier",
    "store_path",
    "train_run",
]

#: Why a run exists. Only ``train`` and ``tuning`` runs may ever be selected.
type Purpose = Literal["train", "tuning", "timing", "smoke", "test"]

#: Every :data:`Purpose`.
PURPOSES: Final[tuple[str, ...]] = ("train", "tuning", "timing", "smoke", "test")

#: Samples per held-out evaluation batch.
_EVAL_BATCH: Final[int] = 512


@dataclass(frozen=True)
class HeldoutSpec:
    """The held-out slice of ``train``-pool demos (module docstring).

    Attributes:
        collection: Store name.
        start: First stream index.
        n: Episodes (success-filtered like a clean teacher set).
    """

    collection: str = "ppo_2000"
    start: int = 1950
    n: int = 50

    def indices(self) -> range:
        """Stream indices of the slice."""
        return range(self.start, self.start + self.n)


#: The held-out slice every Phase 4 run logs against.
HELDOUT: Final[HeldoutSpec] = HeldoutSpec()


def store_path(data_root: Path, collection: str) -> Path:
    """Return ``<data_root>/demos/<collection>.h5`` (``old_float64/`` is never used)."""
    return data_root / "demos" / f"{collection}.h5"


def lr_multiplier(step: int, total: int, warmup: int) -> float:
    """Return diffusers' ``cosine`` schedule multiplier at stage step ``step`` (0-based).

    Args:
        step: Optimiser steps already taken in the stage.
        total: Stage steps ``S``.
        warmup: Warm-up steps ``W``.

    Returns:
        ``step / W`` while ``step < W``, then ``0.5 (1 + cos(pi (step - W) / (S - W)))``.
    """
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def configure_numerics(device: torch.device) -> dict[str, Any]:
    """Pin torch numerics for training and return them for the manifest.

    TF32 off for matmul and cuDNN (P2-D3's GPU settings; bf16 autocast governs the vision arms'
    convolutions and matmuls anyway), ``float32_matmul_precision = "highest"``, cuDNN autotuning
    off so kernel choice does not depend on timing.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "device": str(device),
    }


# --------------------------------------------------------------------------- EMA


class EMA:
    """Exponential moving average of a network's parameters; buffers copied.

    Attributes:
        cfg: Decay rule.
        model: The averaged copy (eval mode, no gradients).
        updates: Updates made so far.
    """

    def __init__(self, net: Net, cfg: EMAConfig) -> None:
        """Start the average at ``net``'s current weights."""
        self.cfg = cfg
        self.model: Net = copy.deepcopy(net).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, net: nn.Module) -> float:
        """Fold ``net``'s parameters in; return the decay used.

        ``ema <- ema + (1 - d) (w - ema)`` (= ``d ema + (1 - d) w``), ``d`` from
        :meth:`EMAConfig.decay_at` of the update count; buffers (the fixed normaliser) copied.
        """
        self.updates += 1
        d = self.cfg.decay_at(self.updates)
        ema_p: list[torch.Tensor] = list(self.model.parameters())
        src_p: list[torch.Tensor] = [p.detach() for p in net.parameters()]
        if d == 0.0:
            for e, p in zip(ema_p, src_p, strict=True):
                e.copy_(p)
        else:
            torch._foreach_lerp_(ema_p, src_p, 1.0 - d)
        for e_b, b in zip(self.model.buffers(), net.buffers(), strict=True):
            e_b.copy_(b)
        return d

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The averaged weights (a network state dict)."""
        return self.model.state_dict()

    def load(self, state: Mapping[str, torch.Tensor], updates: int) -> None:
        """Restore weights and the update count."""
        self.model.load_state_dict(dict(state))
        self.updates = int(updates)


# --------------------------------------------------------------------------- data


@dataclass
class DataBundle:
    """A stage's training data, its held-out slice and the accounting.

    Attributes:
        train: Training dataset (``batch(indices)`` and ``len``).
        heldout: Held-out dataset, or ``None``.
        choices: Every episode choice in the aggregate (accounting).
        labelled_episodes: Episodes attempted in the aggregate (P3-D1 §2's labelled count).
        heldout_overlap: Held-out episodes that are also training episodes.
        stores: Store records (path, SHA-256 vs the committed summary).
        extra: Arm-specific stage records merged into :meth:`summary` (``bc_dagger``: the
            ``"dagger"`` record of :mod:`ila.train.dagger_stage`); empty otherwise.
    """

    train: BCWindowDataset | StateFrames
    heldout: BCWindowDataset | StateFrames | None
    choices: list[EpisodeChoice]
    labelled_episodes: int
    heldout_overlap: int = 0
    stores: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-ready accounting."""
        return {
            "choices": [c.summary() for c in self.choices],
            "labelled_episodes": self.labelled_episodes,
            "kept_episodes": sum(c.kept for c in self.choices),
            "train_frames": len(self.train),
            "heldout_frames": None if self.heldout is None else len(self.heldout),
            "heldout_overlap_episodes": self.heldout_overlap,
            "stores": self.stores,
            **self.extra,
        }


def _dataset(kind: str, cfg: TrainConfig, store: DemoStore, names: Sequence[str]) -> Any:
    """Build the arm's dataset over ``names`` (vision: student path only)."""
    if kind == "state":
        return StateFrames(store, names)
    table = FrameTable.from_store(store, names, cfg.obs_variant)
    return _wrap_table(kind, cfg, table)


def _wrap_table(kind: str, cfg: TrainConfig, table: FrameTable) -> BCWindowDataset:
    from ila.obs.split import load_contract  # lazy: P8's field layout

    to = load_contract(variant=cfg.obs_variant).n_obs_steps
    if kind == "dp":
        horizon = int(cfg.raw["horizons"]["horizon"])
        return DPChunkDataset(table, int(cfg.raw["horizons"]["n_obs_steps"]), horizon)
    return BCWindowDataset(table, to)


def _heldout(
    spec: RunSpec, choices: Sequence[EpisodeChoice]
) -> tuple[BCWindowDataset | StateFrames | None, int]:
    h = spec.heldout
    if h is None:
        return None, 0
    path = store_path(spec.data_root, h.collection)
    with DemoStore(path) as store:
        choice = choose_episodes(store, h.n, True, start=h.start)
        ds = _dataset(spec.cfg.kind, spec.cfg, store, choice.names)
    overlap = sum(len(set(c.names) & set(choice.names)) for c in choices if Path(c.store) == path)
    return ds, overlap


def load_stage_data(
    spec: RunSpec, ranges: Sequence[tuple[int, int]], *, hash_store: bool = True
) -> DataBundle:
    """Choose and decompress the episodes of ``ranges`` (``(start, n)`` stream slices).

    Args:
        spec: The run.
        ranges: Stream slices of the run's collection, in aggregate order.
        hash_store: Hash the store file for the manifest.

    Returns:
        The bundle (held-out slice included when the spec has one).
    """
    collection = spec.cfg.collection_for(spec.sigma)
    path = store_path(spec.data_root, collection)
    choices: list[EpisodeChoice] = []
    tables: list[Any] = []
    with DemoStore(path) as store:
        for start, n in ranges:
            choice = choose_episodes(store, n, spec.cfg.success_filter, start=start)
            choices.append(choice)
            tables.append(_dataset(spec.cfg.kind, spec.cfg, store, choice.names))
    if len(tables) == 1:
        train = tables[0]
    elif spec.cfg.kind == "state":
        raise ValueError("bc_state is single-stage")
    else:
        train = _wrap_table(spec.cfg.kind, spec.cfg, FrameTable.concat([t.table for t in tables]))
    heldout, overlap = _heldout(spec, choices)
    return DataBundle(
        train=train,
        heldout=heldout,
        choices=choices,
        labelled_episodes=sum(c.attempted for c in choices),
        heldout_overlap=overlap,
        stores=[mf.store_record(collection, path, hash_file=hash_store)],
    )


# --------------------------------------------------------------------------- stages


@dataclass(frozen=True)
class Stage:
    """One schedule segment: a fresh AdamW and cosine schedule over ``steps``.

    Attributes:
        index: 0 (single-stage) or the iteration ``i``.
        steps: Optimiser steps.
        warmup: Warm-up steps.
        checkpoint_every: Checkpoint cadence within the stage.
        data: Builds the stage's data when the stage starts (``hash_store`` flag).
        labelled_episodes_before: Labelled episodes of earlier stages not in ``data``'s count
            (``bc_matched``: iteration 0's 1 000 come from the ``bc`` run and are re-read).
    """

    index: int
    steps: int
    warmup: int
    checkpoint_every: int
    data: Callable[[bool], DataBundle]
    labelled_episodes_before: int = 0


@dataclass(frozen=True)
class RunSpec:
    """Everything a run needs besides its stages.

    Attributes:
        cfg: The frozen config.
        run_seed: Seed (initialisation, data order, crops, DDPM noise).
        run_dir: Output directory.
        data_root: ``${data_root}`` (stores under ``demos/``).
        device: Torch device string.
        purpose: :data:`Purpose`.
        sigma: DART sigma (``bc_dart``).
        max_steps: Timing runs only (P3-D1 §5): stop after this many global steps.
        steps_override: Smoke/test runs only: replaces the config's step count.
        checkpoint_every_override: Smoke/test runs only.
        first_n_override: Smoke/test runs only: episodes taken (before the filter).
        heldout: Held-out slice, or ``None`` (tests).
        log_every: Log interval, steps.
        eval_every: Held-out interval, steps.
        prefetch: Batches prepared ahead by the loader thread (0: inline).
        init_from: Iterated arms: the same-seed ``bc`` run's final full checkpoint.
        hash_store: Hash the store file for the manifest.
        allow_dirty: Allow a dirty ``ila`` tree (recorded; never for ``train``/``tuning``).
        rollout_workers: ``bc_dagger``: rollout processes (rows do not depend on it).
        rollout_limit: Smoke/test runs only: ``bc_dagger`` flies the first ``rollout_limit``
            episodes of each iteration's 200.
        iterations_override: Smoke/test runs only: run iterations ``1 .. iterations_override``.
    """

    cfg: TrainConfig
    run_seed: int
    run_dir: Path
    data_root: Path
    device: str = "cuda"
    purpose: Purpose = "train"
    sigma: float | None = None
    max_steps: int | None = None
    steps_override: int | None = None
    checkpoint_every_override: int | None = None
    first_n_override: int | None = None
    heldout: HeldoutSpec | None = HELDOUT
    log_every: int = 100
    eval_every: int = 1000
    prefetch: int = 4
    init_from: Path | None = None
    hash_store: bool = True
    allow_dirty: bool = False
    rollout_workers: int = 24
    rollout_limit: int | None = None
    iterations_override: int | None = None

    def __post_init__(self) -> None:
        """Validate the purpose-gated overrides.

        Raises:
            ValueError: If ``max_steps`` is set outside a timing run, a step override outside
                smoke/test, or a timing run lacks ``max_steps``.
        """
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {self.purpose!r}")
        if (self.max_steps is not None) != (self.purpose == "timing"):
            raise ValueError("--max-steps is for P3-D1 §5 timing runs only, and they need it")
        overrides = (
            self.steps_override is not None
            or self.checkpoint_every_override is not None
            or self.first_n_override is not None
            or self.rollout_limit is not None
            or self.iterations_override is not None
        )
        if overrides and self.purpose not in ("smoke", "test"):
            raise ValueError("step/cadence overrides are for smoke and test runs only")
        if self.purpose == "tuning" and (self.cfg.arm != "bc_dart" or self.run_seed != 0):
            raise ValueError("the only tuning runs are bc_dart seed 0 per sigma (P3-D1 §4)")
        if self.cfg.dagger is not None and self.purpose == "timing":
            raise ValueError("bc_dagger has no timing run: iteration i needs i - 1's checkpoint")
        if self.rollout_workers < 1:
            raise ValueError(f"rollout_workers must be >= 1, got {self.rollout_workers}")


def _iteration_data(spec: RunSpec, ranges: list[tuple[int, int]], hash_store: bool) -> DataBundle:
    return load_stage_data(spec, ranges, hash_store=hash_store)


def build_stages(spec: RunSpec) -> list[Stage]:
    """Return the run's stages from its config.

    Single-stage arms: stage 0 over the first ``first_n`` episodes. Iterated arms
    (``cfg.iterations``): stages ``1 .. n``, stage ``i`` training on stream slices
    ``[0, first_added)`` plus every slice added up to ``i`` (P3-D1 §3) -- the teacher's demos
    of those slices for ``bc_matched``; for ``bc_dagger`` the demos ``[0, first_added)`` and
    the on-policy rollouts of iterations ``1 .. i``, flown when stage ``i`` starts
    (:func:`ila.train.dagger_stage.stage_data`). Both arms get the same stages (indices,
    steps, warm-up, cadence).

    Raises:
        ValueError: If an iterated arm has no ``init_from``.
    """
    cfg = spec.cfg
    steps = cfg.steps if spec.steps_override is None else spec.steps_override
    every = cfg.checkpoint_every
    if spec.checkpoint_every_override is not None:
        every = spec.checkpoint_every_override
    it = cfg.iterations
    first_n = cfg.first_n if spec.first_n_override is None else spec.first_n_override
    if it is None:
        return [
            Stage(
                index=0,
                steps=steps,
                warmup=cfg.optim.warmup_steps,
                checkpoint_every=every,
                data=functools.partial(_iteration_data, spec, [(0, first_n)]),
            )
        ]
    if spec.init_from is None:
        raise ValueError(f"{cfg.arm}: iteration 0 is the same-seed bc run (--init-from)")
    it_steps = it.steps_per_iteration if spec.steps_override is None else spec.steps_override
    n_iter = it.n if spec.iterations_override is None else min(it.n, spec.iterations_override)

    def ranges(i: int) -> list[tuple[int, int]]:
        added = [(it.added_range(j).start, it.add_per_iteration) for j in range(1, i + 1)]
        return [(0, first_n), *added]

    def data(i: int) -> Callable[[bool], DataBundle]:
        if cfg.dagger is None:
            return functools.partial(_iteration_data, spec, ranges(i))
        from ila.train import dagger_stage  # lazy: it imports this module

        return functools.partial(dagger_stage.stage_data, spec, i, it_steps)

    return [
        Stage(
            index=i,
            steps=it_steps,
            warmup=it.warmup_steps,
            checkpoint_every=every,
            data=data(i),
        )
        for i in range(1, n_iter + 1)
    ]


# --------------------------------------------------------------------------- steps


@dataclass
class _StepInput:
    batch: dict[str, torch.Tensor]
    crop: torch.Generator
    noise: torch.Tensor | None
    timestep: torch.Tensor | None


def _make_input(
    data: BCWindowDataset | StateFrames,
    sampler: EpochSampler,
    step: int,
    dp: tuple[int, int, int, int] | None,
    pin: bool,
) -> _StepInput:
    """Build one step's input on the CPU: a pure function of ``(run_seed, stage, step)``.

    ``dp`` is ``(B, Tp, A, num_train_timesteps)`` for a diffusion arm: epsilon and ``k`` are
    drawn here, from the step's ``STREAM_NOISE`` generator (P2-D3 item 4).
    """
    arrays = data.batch(sampler.indices(step))
    batch = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
    if pin:
        batch = {k: v.pin_memory() for k, v in batch.items()}
    crop = torch.Generator(device="cpu")
    crop.manual_seed(sampler.seed(step, STREAM_CROP))
    noise = timestep = None
    if dp is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(sampler.seed(step, STREAM_NOISE))
        b, horizon, a, n_k = dp
        noise = torch.randn((b, horizon, a), generator=g, dtype=torch.float32)
        timestep = torch.randint(0, n_k, (b,), generator=g)
    return _StepInput(batch, crop, noise, timestep)


def _step_input(
    data: BCWindowDataset | StateFrames,
    sampler: EpochSampler,
    dp: tuple[int, int, int, int] | None,
    pin: bool,
    step: int,
) -> _StepInput:
    return _make_input(data, sampler, step, dp, pin)


class _Prefetch:
    """Prepare step inputs ahead in threads (outputs are a pure function of the step)."""

    def __init__(
        self, make: Callable[[int], _StepInput], start: int, stop: int, depth: int
    ) -> None:
        self._make = make
        self._next = start
        self._stop = stop
        self._depth = max(0, depth)
        workers = min(2, self._depth)
        self._pool = ThreadPoolExecutor(max_workers=workers) if self._depth else None
        self._queue: list[tuple[int, Future[_StepInput]]] = []
        self.wait_s = 0.0

    def _fill(self) -> None:
        assert self._pool is not None
        while len(self._queue) < self._depth and self._next < self._stop:
            self._queue.append((self._next, self._pool.submit(self._make, self._next)))
            self._next += 1

    def __iter__(self) -> Iterator[tuple[int, _StepInput]]:
        if self._pool is None:
            for step in range(self._next, self._stop):
                yield step, self._make(step)
            return
        try:
            self._fill()
            while self._queue:
                step, fut = self._queue.pop(0)
                t0 = time.perf_counter()
                item = fut.result()
                self.wait_s += time.perf_counter() - t0
                self._fill()
                yield step, item
        finally:
            for _, fut in self._queue:
                fut.cancel()
            self._pool.shutdown(wait=True)


def _forward_loss(
    net: Net, kind: str, inp: _StepInput, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(per-sample loss float32[B], timesteps or None)`` for one input."""
    b = {k: v.to(device, non_blocking=True) for k, v in inp.batch.items()}
    if kind == "state":
        assert isinstance(net, StateMLP)
        out = net(b["obs"])
        return ((out.float() - b["target"]) ** 2).mean(dim=-1), None
    if kind == "bc":
        assert isinstance(net, BCNet)
        out = net(b["images"], b["proprio"], inp.crop)
        return ((out.float() - b["target"]) ** 2).mean(dim=-1), None
    assert isinstance(net, DiffusionPolicyNet) and inp.noise is not None
    assert inp.timestep is not None
    k = inp.timestep.to(device)
    per = ddpm_loss(net, b["images"], b["proprio"], b["target"], inp.noise.to(device), k, inp.crop)
    return per, k


def _autocast(device: torch.device, precision: str) -> Any:
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
    )


@torch.no_grad()
def heldout_loss(
    net: Net,
    data: BCWindowDataset | StateFrames,
    kind: str,
    device: torch.device,
    precision: str,
    run_seed: int,
    num_train_timesteps: int = 100,
) -> dict[str, float]:
    """Mean loss over a dataset in eval mode (centre crop); DP also by diffusion-step bucket.

    DP noise and ``k`` come from a fixed generator (``loader_seed`` stream ``HELDOUT``), so the
    held-out loss of two checkpoints is computed on the same noise.

    Returns:
        ``{"loss": ...}`` plus, for DP, ``"loss_k<lo>_<hi>"`` per quarter of the k range.
    """
    was_training = net.training
    net.eval()
    n = len(data)
    total = 0.0
    buckets = np.zeros(4)
    counts = np.zeros(4)
    try:
        for start in range(0, n, _EVAL_BATCH):
            idx = np.arange(start, min(n, start + _EVAL_BATCH), dtype=np.int64)
            arrays = data.batch(idx)
            batch = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
            noise = timestep = None
            if kind == "dp":
                g = torch.Generator(device="cpu")
                g.manual_seed(loader_seed(run_seed, -1, 0, 0, start, STREAM_HELDOUT))
                target = batch["target"]
                noise = torch.randn(tuple(target.shape), generator=g, dtype=torch.float32)
                timestep = torch.randint(0, num_train_timesteps, (len(idx),), generator=g)
            inp = _StepInput(batch, torch.Generator(), noise, timestep)
            with _autocast(device, precision):
                per, k = _forward_loss(net, kind, inp, device)
            per_np = per.float().cpu().numpy().astype(np.float64)
            total += float(per_np.sum())
            if k is not None:
                q = np.minimum(k.cpu().numpy() * 4 // num_train_timesteps, 3)
                np.add.at(buckets, q, per_np)
                np.add.at(counts, q, 1)
    finally:
        net.train(was_training)
    out = {"loss": total / n}
    if kind == "dp":
        width = num_train_timesteps // 4
        for i in range(4):
            name = f"loss_k{i * width}_{(i + 1) * width - 1}"
            out[name] = float(buckets[i] / counts[i]) if counts[i] else float("nan")
    return out


# --------------------------------------------------------------------------- checkpoints


def _ckpt_name(stage: int, stage_step: int) -> str:
    return f"s{stage}_step{stage_step:06d}.pt"


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    """Return the latest full checkpoint of a run (highest ``(stage, step)``), or ``None``."""
    ckpts = sorted((run_dir / "checkpoints").glob("s*_step*.pt"))
    if not ckpts:
        return None

    def key(p: Path) -> tuple[int, int]:
        stage, step = p.stem.split("_step")
        return int(stage[1:]), int(step)

    return max(ckpts, key=key)


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _finite(net: nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in net.parameters())


@dataclass
class RunResult:
    """What :func:`train_run` returns.

    Attributes:
        state: Final status state (``done``, ``diverged``, ``stopped``).
        global_step: Optimiser steps taken in total.
        net: The raw network (on the training device).
        ema: The EMA.
        checkpoints: Manifest entries of every checkpoint written.
        wall_s: Wall clock of the whole call, seconds.
        loop_s: Wall clock of the training loops only (data loading excluded), seconds.
        steps_timed: Steps covered by ``loop_s``.
    """

    state: str
    global_step: int
    net: Net
    ema: EMA
    checkpoints: list[dict[str, Any]]
    wall_s: float
    loop_s: float
    steps_timed: int


def _log(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=float) + "\n")


def _init_from(path: Path, cfg: TrainConfig, run_seed: int, purpose: str) -> dict[str, Any]:
    """Read and check iteration 0: the same-seed ``bc`` run's final full checkpoint.

    A ``train`` iterated run needs a finished ``train`` ``bc`` run; a test/smoke one accepts a
    finished test/smoke ``bc`` run (with its effective ``first_n``).
    """
    ckpt = read_checkpoint(path)
    run = path.parent.parent
    man = mf.read_json(run / "manifest.json")
    problems = []
    if ckpt["arm"] != "bc" or int(ckpt["run_seed"]) != run_seed:
        problems.append(f"arm/seed {ckpt['arm']}/{ckpt['run_seed']} != bc/{run_seed}")
    allowed = ("train",) if purpose in ("train", "tuning") else ("train", "test", "smoke")
    if man.get("purpose") not in allowed or man.get("state") != "done":
        problems.append(f"bc run purpose/state {man.get('purpose')}/{man.get('state')}")
    if int(ckpt["global_step"]) != int(man.get("total_steps", -1)):
        problems.append("not the bc run's last step")
    bc_cfg = ckpt["config"]
    bc_first_n = man.get("first_n_override") or bc_cfg["data"]["first_n"]
    bc_data = (bc_cfg["data"]["collection"], bc_first_n)
    if bc_data != (cfg.collection, cfg.first_n) or set(bc_cfg["data"]) != {"collection", "first_n"}:
        problems.append(f"bc data {bc_cfg['data']} (first_n {bc_first_n}) != iteration 0")
    if bc_cfg["success_filter"] != cfg.success_filter:
        problems.append("bc success filter differs")
    if problems:
        raise ValueError(f"{path}: not a valid iteration 0: {'; '.join(problems)}")
    return ckpt


def train_run(
    spec: RunSpec, stages: Sequence[Stage], *, stop_after: int | None = None
) -> RunResult:
    """Train (or resume) a run through its stages; write manifest, status, log, checkpoints.

    Args:
        spec: The run.
        stages: From :func:`build_stages`.
        stop_after: Tests only: stop after this many global steps as if interrupted.

    Returns:
        The :class:`RunResult`.

    Raises:
        RuntimeError: On a dirty tree for a ``train``/``tuning``/``timing`` run without
            ``allow_dirty``, an unfrozen config for one, or a resume checkpoint from a
            different config or seed.
    """
    t_start = time.perf_counter()
    cfg = spec.cfg
    device = torch.device(spec.device)
    run_dir = spec.run_dir
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    total_steps = sum(s.steps for s in stages)
    stage_offset = {s.index: sum(x.steps for x in stages[:k]) for k, s in enumerate(stages)}
    budget = total_steps if spec.max_steps is None else min(total_steps, spec.max_steps)

    numerics = configure_numerics(device)
    frozen = mf.check_config_frozen(cfg.files_sha256)
    git = mf.git_state()
    serious = spec.purpose in ("train", "tuning", "timing")  # timing feeds the P3-D1 §5 rule
    if serious and not frozen["all_frozen"]:
        raise RuntimeError(f"config differs from P3-D1's frozen SHA-256: {frozen}")
    if serious and git["ila_dirty"] and not spec.allow_dirty:
        raise RuntimeError("the ila tree is dirty; commit first (or --allow-dirty, recorded)")

    man_file = run_dir / "manifest.json"
    manifest: dict[str, Any] = (
        mf.read_json(man_file)
        if man_file.is_file()
        else {
            "arm": cfg.arm,
            "kind": cfg.kind,
            "run_seed": spec.run_seed,
            "purpose": spec.purpose,
            "selectable": spec.purpose in ("train", "tuning"),
            "sigma": spec.sigma,
            "config": frozen,
            "config_resolved": dict(cfg.raw),
            "max_steps": spec.max_steps,
            "steps_override": spec.steps_override,
            "first_n_override": spec.first_n_override,
            "checkpoint_every_override": spec.checkpoint_every_override,
            "total_steps": total_steps,
            "stages": [
                {
                    "index": s.index,
                    "steps": s.steps,
                    "warmup": s.warmup,
                    "checkpoint_every": s.checkpoint_every,
                }
                for s in stages
            ],
            "optimizer": {
                "name": "adamw",
                "lr": cfg.optim.lr,
                "weight_decay": cfg.optim.weight_decay,
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "schedule": "cosine (diffusers), linear warm-up from 0",
                "grad_clip": None,
            },
            "precision": cfg.precision,
            "ema": vars(cfg.ema),
            "batch_size": cfg.batch_size,
            "heldout": None if spec.heldout is None else vars(spec.heldout),
            "init_from": None if spec.init_from is None else str(spec.init_from),
            **(
                {}
                if cfg.dagger is None
                else {
                    "dagger_run": {
                        "rollout_workers": spec.rollout_workers,
                        "rollout_limit": spec.rollout_limit,
                        "iterations_override": spec.iterations_override,
                        "beta": list(cfg.dagger.beta),
                        "per_stage_records": "data.<i>.dagger and dagger/iter<i>.json",
                        "wall_clock_note": "wall_clock_s and gpu_hours include the CPU "
                        "rollouts and audits (data.<i>.dagger.timing_s); subtract them for "
                        "GPU training time",
                    }
                }
            ),
            "checkpoints": [],
            "resumes": [],
            "segments": [],
        }
    )
    host = mf.host_info(spec.device)
    invocation = {"git": git, "host": host, "numerics": numerics, "unix": time.time()}
    manifest.setdefault("invocations", []).append(invocation)
    # The first invocation's provenance is the run's; a resume on other code is flagged.
    manifest.setdefault("git", git)
    manifest.setdefault("host", host)
    manifest.setdefault("numerics", numerics)
    first_git = manifest["git"]
    if (first_git["ila_commit"], first_git["ila_dirty_paths"]) != (
        git["ila_commit"],
        git["ila_dirty_paths"],
    ):
        manifest["code_changed_since_start"] = True
    manifest["state"] = "running"
    status = mf.StatusFile(
        run_dir / "status.json",
        total_steps=budget,
        arm=cfg.arm,
        run_seed=spec.run_seed,
        purpose=spec.purpose,
    )
    mf.write_json_atomic(man_file, manifest)

    n_k = int(cfg.raw["noise_scheduler"]["num_train_timesteps"]) if cfg.kind == "dp" else 0

    net = build_net_from_config(cfg.kind, cfg.raw, spec.run_seed, cfg.obs_variant)
    ema = EMA(net, cfg.ema)
    resume = find_resume_checkpoint(run_dir)
    ckpt: dict[str, Any] | None = None
    start_stage_pos, start_step = 0, 0
    if resume is not None:
        ckpt = read_checkpoint(resume)
        if ckpt["config_sha256"] != cfg.sha256 or int(ckpt["run_seed"]) != spec.run_seed:
            raise RuntimeError(f"{resume}: written by another config or seed")
        net.load_state_dict(ckpt["model"])
        ema.load(ckpt["ema"], int(ckpt["ema_updates"]))
        pos = [s.index for s in stages].index(int(ckpt["stage"]))
        start_stage_pos, start_step = pos, int(ckpt["stage_step"])
        if start_step >= stages[pos].steps:
            start_stage_pos, start_step = pos + 1, 0
        manifest["resumes"].append(
            {"from": resume.name, "global_step": int(ckpt["global_step"]), "unix": time.time()}
        )
        _log(log_path, {"event": "resume", "from": resume.name, "step": ckpt["global_step"]})
    elif spec.init_from is not None:
        init = _init_from(spec.init_from, cfg, spec.run_seed, spec.purpose)
        net.load_state_dict(init["model"])
        ema.load(init["ema"], int(init["ema_updates"]))
        manifest["init_from_sha256"] = sha256_file(spec.init_from)
        manifest["iteration0"] = {
            "file": str(spec.init_from),
            "labelled_episodes": int(init["labelled_episodes"]),
            "global_step": int(init["global_step"]),
        }
    net.to(device)
    ema.model.to(device)
    if ckpt is not None:
        _restore_rng(ckpt["rng"], device)

    iteration0 = manifest.get("iteration0")
    checkpoints: list[dict[str, Any]] = list(manifest["checkpoints"])
    loop_s = 0.0
    steps_timed = 0
    state = "done"
    if start_stage_pos < len(stages):
        global_step = stage_offset[stages[start_stage_pos].index] + start_step
    else:
        global_step = total_steps
    pin = device.type == "cuda"
    params = [p for p in net.parameters() if p.requires_grad]

    for pos in range(start_stage_pos, len(stages)):
        stage = stages[pos]
        offset = stage_offset[stage.index]
        if spec.max_steps is not None and offset >= spec.max_steps:
            break
        t_data = time.perf_counter()
        data = stage.data(spec.hash_store)
        labelled = data.labelled_episodes  # the aggregate re-reads iteration 0's episodes
        if iteration0 is not None and data.choices[0].attempted != iteration0["labelled_episodes"]:
            raise RuntimeError("iteration 0's episodes differ from the bc run's")
        stage_summary = {
            "stage": stage.index,
            "data_load_s": time.perf_counter() - t_data,
            "labelled_episodes_total": labelled,
            **data.summary(),
        }
        it = cfg.iterations
        if it is not None and stage.index == it.n and labelled != it.labelled_episodes_total:
            raise RuntimeError(
                f"{cfg.arm}: {labelled} labelled episodes after iteration {stage.index}, "
                f"expected {it.labelled_episodes_total} (P3-D1 §3)"
            )
        manifest.setdefault("data", {})[str(stage.index)] = stage_summary
        if pos == 0 and spec.init_from is None and resume is None:
            _fit_normaliser(net, ema, data.train, device)
            manifest["normaliser"] = _normaliser_record(net)
        mf.write_json_atomic(man_file, manifest)
        _log(log_path, {"event": "stage", **stage_summary})

        opt = torch.optim.AdamW(
            params,
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
        )
        first = start_step if pos == start_stage_pos else 0
        if ckpt is not None and pos == start_stage_pos and first > 0:
            opt.load_state_dict(ckpt["optimizer"])
        sampler = EpochSampler(len(data.train), cfg.batch_size, spec.run_seed, stage.index)
        dp = None
        if cfg.kind == "dp":
            dp = (cfg.batch_size, int(cfg.raw["horizons"]["horizon"]), 3, n_k)
        stop = stage.steps
        if spec.max_steps is not None:
            stop = min(stop, spec.max_steps - offset)
        if stop_after is not None:
            stop = min(stop, max(first, stop_after - offset))
        prefetch = _Prefetch(
            functools.partial(_step_input, data.train, sampler, dp, pin),
            first,
            stop,
            spec.prefetch,
        )
        net.train()
        acc_loss = torch.zeros((), device=device)
        acc_gn = torch.zeros((), device=device)
        max_gn = torch.zeros((), device=device)
        n_acc = 0
        t_int = time.perf_counter()
        t_loop = time.perf_counter()
        steps_in_loop = 0
        diverged = False
        for step, inp in prefetch:
            lr = cfg.optim.lr * lr_multiplier(step, stage.steps, stage.warmup)
            for group in opt.param_groups:
                group["lr"] = lr
            with _autocast(device, cfg.precision):
                per, _ = _forward_loss(net, cfg.kind, inp, device)
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            grads = [p.grad for p in params if p.grad is not None]
            gn = torch.nn.utils.get_total_norm(grads)
            opt.step()
            ema.update(net)
            acc_loss += loss.detach()
            acc_gn += gn
            max_gn = torch.maximum(max_gn, gn)
            n_acc += 1
            steps_in_loop += 1
            done = step + 1
            global_step = offset + done
            at_log = done % spec.log_every == 0 or done == stop
            if at_log:
                mean_loss = float(acc_loss) / n_acc
                elapsed = time.perf_counter() - t_int
                rate = n_acc / elapsed if elapsed > 0 else None
                record = {
                    "event": "train",
                    "step": global_step,
                    "stage": stage.index,
                    "stage_step": done,
                    "loss": mean_loss,
                    "loss_last": float(loss.detach()),
                    "grad_norm_mean": float(acc_gn) / n_acc,
                    "grad_norm_max": float(max_gn),
                    "lr": lr,
                    "ema_decay": cfg.ema.decay_at(ema.updates),
                    "epoch": sampler.position(step)[0],
                    "steps_per_s": rate,
                    "loader_wait_s": prefetch.wait_s,
                    "wall_s": time.perf_counter() - t_start,
                }
                if device.type == "cuda":
                    record["gpu_mem_max_bytes"] = torch.cuda.max_memory_allocated(device)
                _log(log_path, record)
                status.progress(global_step, rate, loss=mean_loss, stage=stage.index)
                acc_loss.zero_()
                acc_gn.zero_()
                max_gn.zero_()
                n_acc = 0
                t_int = time.perf_counter()
                if not math.isfinite(mean_loss):
                    diverged = True
                    break
            at_ckpt = done % stage.checkpoint_every == 0 or done == stage.steps
            at_eval = done % spec.eval_every == 0 or at_ckpt
            if at_eval and data.heldout is not None and spec.purpose != "timing":
                held = data.heldout
                held_raw = heldout_loss(
                    net, held, cfg.kind, device, cfg.precision, spec.run_seed, n_k or 100
                )
                held_ema = heldout_loss(
                    ema.model, held, cfg.kind, device, cfg.precision, spec.run_seed, n_k or 100
                )
                _log(
                    log_path,
                    {
                        "event": "heldout",
                        "step": global_step,
                        "stage": stage.index,
                        "raw": held_raw,
                        "ema": held_ema,
                        "overlap_episodes": data.heldout_overlap,
                    },
                )
            if at_ckpt and spec.purpose != "timing":
                if not (_finite(net) and _finite(ema.model)):
                    diverged = True
                    break
                entry = _save_checkpoint(
                    spec, net, ema, opt, stage.index, done, global_step, labelled, device
                )
                checkpoints.append(entry)
                if done == stage.steps and pos == len(stages) - 1:
                    checkpoints.append(
                        _save_ema_last(spec, ema, stage.index, done, global_step, labelled)
                    )
                manifest["checkpoints"] = checkpoints
                mf.write_json_atomic(man_file, manifest)
                _log(log_path, {"event": "checkpoint", **entry})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seg = time.perf_counter() - t_loop
        loop_s += seg
        steps_timed += steps_in_loop
        manifest["segments"].append(
            {
                "stage": stage.index,
                "first_step": first,
                "steps": steps_in_loop,
                "loop_s": seg,
                "loader_wait_s": prefetch.wait_s,
            }
        )
        if diverged:
            state = "diverged"
            break
        if steps_in_loop and first + steps_in_loop < stage.steps:
            state = "stopped" if spec.max_steps is None else "done"
            break
        # Release this stage's aggregate before the next stage builds (and, for bc_dagger,
        # flies) its own: two aggregates never coexist.
        del data, prefetch, sampler
    wall = time.perf_counter() - t_start
    prior = float(manifest.get("wall_clock_s", 0.0))
    manifest.update(
        {
            "state": state,
            "global_step": global_step,
            "wall_clock_s": prior + wall,
            "gpu_hours": (prior + wall) / 3600.0 if device.type == "cuda" else 0.0,
            "loop_s_total": float(manifest.get("loop_s_total", 0.0)) + loop_s,
            "steps_timed_total": int(manifest.get("steps_timed_total", 0)) + steps_timed,
            "finished_unix": time.time(),
        }
    )
    if device.type == "cuda":
        manifest["gpu_mem_max_bytes"] = torch.cuda.max_memory_allocated(device)
    if steps_timed:
        manifest["steps_per_s_loop"] = manifest["steps_timed_total"] / manifest["loop_s_total"]
    mf.write_json_atomic(man_file, manifest)
    status.update(state=state, step=global_step, eta_s=0.0 if state == "done" else None)
    return RunResult(state, global_step, net, ema, checkpoints, wall, loop_s, steps_timed)


def _fit_normaliser(
    net: Net, ema: EMA, data: BCWindowDataset | StateFrames, device: torch.device
) -> None:
    """Fix the input normaliser from the training frames (state: obs; vision: proprio)."""
    if isinstance(net, StateMLP):
        assert isinstance(data, StateFrames)
        mean, std = fit_normaliser(data.obs)
        net.set_normaliser(mean, std)
    else:
        assert isinstance(data, BCWindowDataset)
        mean, std = data.table.proprio_stats()
        enc = net.encoder
        with torch.no_grad():
            enc.proprio_mean.copy_(torch.from_numpy(mean).to(enc.proprio_mean.device))
            enc.proprio_std.copy_(torch.from_numpy(std).to(enc.proprio_std.device))
    with torch.no_grad():
        for e_b, b in zip(ema.model.buffers(), net.buffers(), strict=True):
            e_b.copy_(b.to(e_b.device))
    del device


def _normaliser_record(net: Net) -> dict[str, list[float]]:
    if isinstance(net, StateMLP):
        return {"obs_mean": net.obs_mean.tolist(), "obs_std": net.obs_std.tolist()}
    return {
        "proprio_mean": net.encoder.proprio_mean.tolist(),
        "proprio_std": net.encoder.proprio_std.tolist(),
    }


def _common_payload(spec: RunSpec, stage: int, stage_step: int, global_step: int) -> dict[str, Any]:
    cfg = spec.cfg
    return {
        "format": CHECKPOINT_FORMAT,
        "arm": cfg.arm,
        "kind": cfg.kind,
        "obs_variant": cfg.obs_variant,
        "config": dict(cfg.raw),
        "config_sha256": cfg.sha256,
        "run_seed": spec.run_seed,
        "purpose": spec.purpose,
        "sigma": spec.sigma,
        "stage": stage,
        "stage_step": stage_step,
        "global_step": global_step,
    }


def _entry(path: Path, kind: str, stage: int, stage_step: int, global_step: int) -> dict[str, Any]:
    return {
        "file": path.name,
        "kind": kind,
        "stage": stage,
        "stage_step": stage_step,
        "global_step": global_step,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _save_checkpoint(
    spec: RunSpec,
    net: Net,
    ema: EMA,
    opt: torch.optim.Optimizer,
    stage: int,
    stage_step: int,
    global_step: int,
    labelled: int,
    device: torch.device,
) -> dict[str, Any]:
    payload = _common_payload(spec, stage, stage_step, global_step)
    payload.update(
        {
            "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "ema_updates": ema.updates,
            "optimizer": opt.state_dict(),
            "rng": _rng_state(device),
            "labelled_episodes": labelled,
        }
    )
    path = spec.run_dir / "checkpoints" / _ckpt_name(stage, stage_step)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return _entry(path, "full", stage, stage_step, global_step)


def _save_ema_last(
    spec: RunSpec, ema: EMA, stage: int, stage_step: int, global_step: int, labelled: int
) -> dict[str, Any]:
    payload = _common_payload(spec, stage, stage_step, global_step)
    payload.update(
        {
            "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "ema_updates": ema.updates,
            "labelled_episodes": labelled,
        }
    )
    path = spec.run_dir / "ema_last.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return _entry(path, "ema_last", stage, stage_step, global_step)


def batch_mse(
    net: Net, data: BCWindowDataset | StateFrames, kind: str, device: torch.device
) -> float:
    """Return the fp32 eval-mode (centre-crop) action MSE of a BC/state net over a dataset.

    Raises:
        ValueError: For a diffusion network (its loss is the denoising loss).
    """
    if kind == "dp":
        raise ValueError("action MSE is for BC and bc_state networks")
    return heldout_loss(net, data, kind, device, "fp32", 0)["loss"]


def numpy_actions(
    net: Net, data: BCWindowDataset | StateFrames, device: torch.device
) -> NDArray[np.float32]:
    """Return eval-mode fp32 actions ``float32[N, 3]`` of a BC/state net over a dataset."""
    out = []
    was = net.training
    net.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(data), _EVAL_BATCH):
                idx = np.arange(start, min(len(data), start + _EVAL_BATCH), dtype=np.int64)
                arrays = data.batch(idx)
                if isinstance(net, StateMLP):
                    y = net(torch.from_numpy(arrays["obs"]).to(device))
                else:
                    assert isinstance(net, BCNet)
                    y = net(
                        torch.from_numpy(arrays["images"]).to(device),
                        torch.from_numpy(arrays["proprio"]).to(device),
                    )
                out.append(y.float().cpu().numpy())
    finally:
        net.train(was)
    return np.concatenate(out).astype(np.float32)
