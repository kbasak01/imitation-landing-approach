"""Lockstep rendering workers and one batching inference process.

Layout
------
* **Workers** (spawned processes, one per *slot*) each pull contiguous chunks of listed
  episodes (the runner's cut: ``chunk`` episodes, independent of the worker count) from a
  queue and fly them with :func:`ila.eval.runner.fly_chunk`: P8's loop, rendering after
  ``reset`` and after every ``env.step()``, the start check on every reset, the timing
  assertion. Instead of a local policy, a worker's actor (:class:`RemoteSlotActor`) builds the
  contract's ``StudentInput`` and sends ``(slot, episode_seed, step, images, proprio)`` to the
  server over a pipe, then blocks for the action.
* **The server** is the calling process: the one inference process (it owns the policy and,
  later, the GPU). Every round it takes exactly one message from every live worker (lockstep),
  orders the requests by slot, and runs the policy on batches of at most ``batch_cap``
  requests (``B`` in :data:`BATCH_CAPS`; batch = live slots, capped). Rows are reassembled in
  list order whatever slot flew them.

Randomness
----------
Every request carries its :class:`NoiseKey` ``(run_seed, episode_seed, step)``: a stochastic
policy (Diffusion Policy) seeds a per-episode ``torch.Generator`` from it (il-conventions:
``hash64(run_seed, episode_seed, t)``), never from a global stream, so its samples do not
depend on which slot, worker or batch position served the episode.

Determinism contract (what is guaranteed here; the GPU numbers are measured in P2-D3)
-------------------------------------------------------------------------------------
* **CPU single-sample path** (``device="cpu"``, a :class:`BatchPolicy` that evaluates each
  request alone, e.g. :class:`PerSlotPolicy`): rows are bit-identical across worker counts and
  equal to the in-worker runner's rows. ``tests/test_eval_batched.py`` checks it.
* **Batched GPU path:** batch composition changes kernel choice and reduction order, so rows
  are **not** claimed bit-identical across ``B`` or worker counts. They are judged by the
  outcome-flip rate against P8's one-ulp floor (P2-D3).

Units: control steps at 30 Hz model scale (lambda = 1/25); images ``uint8`` RGB; proprio per
``configs/obs/student.yaml``; actions normalised (x v_max = 1.5 m/s model scale).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Final, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.eval.runner import (
    DEFAULT_TIMING_EVERY,
    Actor,
    RowTags,
    StudentWindow,
    VisionTask,
    check_privilege,
    fly_chunk,
    init_worker_threads,
    is_modular,
    is_student,
    set_single_thread_env,
)
from ila.obs.split import ObsContract
from ila.policies.dp import chunking

__all__ = [
    "BATCH_CAPS",
    "BatchPolicy",
    "BatchedArm",
    "NoiseKey",
    "PerSlotPolicy",
    "RemoteSlotActor",
    "SharedBCPolicy",
    "SharedDPPolicy",
    "NUMERICS",
    "SlotRequest",
    "TimedPolicy",
    "WorkerError",
    "build_dummy_student",
    "configure_torch_numerics",
    "per_slot_factory",
    "run_batched",
    "shared_net_factory",
]

#: The batch caps ``B`` the throughput and determinism runs use.
BATCH_CAPS: Final[tuple[int, ...]] = (1, 8, 24)

#: Server numerics modes (:func:`configure_torch_numerics`).
NUMERICS: Final[tuple[str, ...]] = ("strict", "torch_default")


class WorkerError(RuntimeError):
    """A rendering worker raised; the message carries its traceback."""


class NoiseKey(NamedTuple):
    """What a stochastic policy seeds its per-episode generator from (il-conventions).

    Attributes:
        run_seed: Training seed of the policy (P8's ``PolicySpec.run_seed``).
        episode_seed: The listed episode seed.
        step: Control step t, 0 after reset (30 Hz model scale).
    """

    run_seed: int
    episode_seed: int
    step: int


@dataclass(frozen=True)
class SlotRequest:
    """One worker's request for an action at one control step.

    Attributes:
        slot: The worker's slot (0 ... workers-1); per-slot policy state lives under it.
        key: The request's :class:`NoiseKey`.
        images: ``uint8[To, H, W, 3]`` from the contract-built ``StudentInput``.
        proprio: ``float32[To, D]`` from the same ``StudentInput``.
    """

    slot: int
    key: NoiseKey
    images: NDArray[np.uint8]
    proprio: NDArray[np.float32]


class BatchPolicy(Protocol):
    """What the server runs: per-slot episode state, batched inference.

    A Diffusion Policy implements this natively (one forward over the batch, a per-request
    ``torch.Generator`` from each key, per-slot action queues for chunk execution);
    :class:`PerSlotPolicy` wraps any single-episode student.
    """

    #: The policy's training seed; must equal the spec's ``run_seed``.
    run_seed: int

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        """Clear one slot's per-episode state before its step-0 request."""
        ...

    def act_batch(self, requests: Sequence[SlotRequest]) -> NDArray[np.float64]:
        """Return ``float64[len(requests), 3]`` actions in normalised units, request order."""
        ...


class PerSlotPolicy:
    """The CPU single-sample path: one student instance per slot, each request alone.

    Each slot's student is built by the spec's factory and reset with the episode seed, and
    ``act_batch`` calls ``act_vision(images, proprio)`` request by request, so a slot's result
    cannot depend on what else is in the batch.

    Attributes:
        run_seed: The spec's run seed.
        students: Slot -> student.
    """

    def __init__(self, spec: p8_adapter.PolicySpec, cfgs: p8_adapter.EvalConfigs, slots: int):
        """Build ``slots`` students from ``spec.build``.

        Raises:
            TypeError: If the spec does not build a student (no ``act_vision``), or builds a
                ``modular`` student (``act_modular``): a :class:`SlotRequest` does not carry
                the teacher's clock, so that arm flies through the in-worker runner only.
            ValueError: If a student's flags differ from the spec's.
        """
        self.run_seed = int(spec.run_seed)
        self.students: dict[int, Any] = {}
        for slot in range(slots):
            student = spec.build(cfgs)
            if is_modular(student):
                raise TypeError(
                    f"{spec.method}: a modular student (act_modular) flies through "
                    "ila.eval.runner only; the batched path does not carry time_fraction"
                )
            if not is_student(student):
                raise TypeError(f"{spec.method}: the batched path flies students only")
            check_privilege(spec, student)
            self.students[slot] = student

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        """Reset the slot's student with the episode seed (never a context)."""
        self.students[slot].reset(episode_seed, None)

    def act_batch(self, requests: Sequence[SlotRequest]) -> NDArray[np.float64]:
        """Evaluate every request on its own slot's student, one at a time."""
        out = np.empty((len(requests), p8_adapter.ACTION_DIM), dtype=np.float64)
        for i, req in enumerate(requests):
            out[i] = np.asarray(
                self.students[req.slot].act_vision(req.images, req.proprio), dtype=np.float64
            ).reshape(p8_adapter.ACTION_DIM)
        return out


def per_slot_factory(
    spec: p8_adapter.PolicySpec,
) -> Callable[[p8_adapter.EvalConfigs, int], BatchPolicy]:
    """Return a server factory wrapping a single-episode student spec in :class:`PerSlotPolicy`."""

    def build(cfgs: p8_adapter.EvalConfigs, slots: int) -> BatchPolicy:
        return PerSlotPolicy(spec, cfgs, slots)

    return build


class _ChunkSampler(Protocol):
    """What :class:`SharedDPPolicy` needs from ``ila.policies.dp.policy.DPStudent``."""

    run_seed: int
    num_inference_steps: int

    @property
    def chunk_spec(self) -> Any:
        """``(To, Tp, Ta)`` as a ``chunking.ChunkSpec``."""
        ...

    def sample_batch(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        seeds: Sequence[tuple[int, int, int]],
        num_inference_steps: int | None = None,
    ) -> NDArray[np.float32]:
        """``float32[B, Tp, 3]`` chunks, slot i's noise keyed ``seeds[i]``."""
        ...


class _BatchActor(Protocol):
    """What :class:`SharedBCPolicy` needs from ``ila.policies.bc.BCStudent``."""

    def act_batch(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``float32[B, 3]`` actions."""
        ...


def _stack(requests: Sequence[SlotRequest]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Stack requests into ``uint8[B, To, H, W, 3]`` and ``float32[B, To, D]``."""
    return np.stack([r.images for r in requests]), np.stack([r.proprio for r in requests])


class _SlotClock:
    """Per-slot step bookkeeping: every slot's steps must run 0, 1, 2, ... after its reset."""

    def __init__(self) -> None:
        self.next_step: dict[int, int] = {}

    def reset(self, slot: int) -> None:
        self.next_step[slot] = 0

    def advance(self, req: SlotRequest) -> None:
        expected = self.next_step.get(req.slot)
        if expected != req.key.step:
            raise ValueError(f"slot {req.slot}: step {req.key.step}, expected {expected}")
        self.next_step[req.slot] = req.key.step + 1


class SharedDPPolicy:
    """The GPU path for Diffusion Policy: one shared network, per-slot chunk queues.

    Per slot, exactly as ``DPStudent.act_vision`` does it for one episode: at a step where
    ``chunking.is_replan_step(t, Ta)``, the slot's chunk is sampled with noise keyed by its
    :class:`NoiseKey`; the executed part ``chunk[chunking.executed_slice(To, Tp, Ta)]`` is
    queued and one action is popped per step. All slots that replan in one server round are
    sampled in one ``sample_batch`` call (``B`` = those slots, within the server's cap).

    Attributes:
        student: The shared ``DPStudent`` (built once in the server, on its device).
        run_seed: The student's run seed.
        numerics: Torch numerics flags in force (:func:`configure_torch_numerics`).
        sample_calls: Batch size of every ``sample_batch`` call (diagnostics).
    """

    def __init__(self, student: _ChunkSampler, numerics: dict[str, Any] | None = None) -> None:
        """Hold the student; queues start empty."""
        self.student = student
        self.run_seed = int(student.run_seed)
        self.numerics = {} if numerics is None else dict(numerics)
        spec = student.chunk_spec
        self._to, self._tp, self._ta = spec.n_obs_steps, spec.horizon, spec.n_action_steps
        self._executed = chunking.executed_slice(self._to, self._tp, self._ta)
        self._queues: dict[int, deque[NDArray[np.float32]]] = {}
        self._clock = _SlotClock()
        self.sample_calls: list[int] = []

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        """Empty the slot's queue; its next request must be step 0."""
        del episode_seed
        self._queues[slot] = deque()
        self._clock.reset(slot)

    def act_batch(self, requests: Sequence[SlotRequest]) -> NDArray[np.float64]:
        """Replan the slots that are due, then pop one action per request."""
        for req in requests:
            self._clock.advance(req)
        due = [r for r in requests if chunking.is_replan_step(r.key.step, self._ta)]
        if due:
            images, proprio = _stack(due)
            chunks = self.student.sample_batch(
                images,
                proprio,
                [(r.key.run_seed, r.key.episode_seed, r.key.step) for r in due],
                self.student.num_inference_steps,
            )
            self.sample_calls.append(len(due))
            for req, chunk in zip(due, chunks, strict=True):
                queue = self._queues[req.slot]
                queue.clear()
                queue.extend(np.asarray(chunk, dtype=np.float32)[self._executed])
        out = np.empty((len(requests), p8_adapter.ACTION_DIM), dtype=np.float64)
        for i, req in enumerate(requests):
            out[i] = self._queues[req.slot].popleft()
        return out


class SharedBCPolicy:
    """The GPU path for BC: one shared network, one ``act_batch`` per server batch.

    Attributes:
        student: The shared ``BCStudent``.
        run_seed: The arm's run seed (BC draws no noise; carried for the key check).
        numerics: Torch numerics flags in force.
    """

    def __init__(
        self, student: _BatchActor, run_seed: int, numerics: dict[str, Any] | None = None
    ) -> None:
        """Hold the student."""
        self.student = student
        self.run_seed = int(run_seed)
        self.numerics = {} if numerics is None else dict(numerics)
        self._clock = _SlotClock()

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        """BC keeps no per-episode state; only the step bookkeeping restarts."""
        del episode_seed
        self._clock.reset(slot)

    def act_batch(self, requests: Sequence[SlotRequest]) -> NDArray[np.float64]:
        """One forward over the batch."""
        for req in requests:
            self._clock.advance(req)
        images, proprio = _stack(requests)
        return np.asarray(self.student.act_batch(images, proprio), dtype=np.float64)


def configure_torch_numerics(device: str, numerics: str = "strict") -> dict[str, Any]:
    """Pin torch numerics for the inference server and return them for ``run_info.json``.

    ``"strict"`` (the default, every determinism-contract run): TF32 off for cuDNN and matmul
    (policy-trainer measured a dp-chunk CPU-vs-GPU gap of 2.8e-3 with cuDNN's default TF32,
    5e-6 without), matmul precision ``"highest"``. ``"torch_default"``: torch's own defaults
    (cuDNN TF32 **on**, matmul TF32 off, precision ``"highest"``), the labelled control. cuDNN
    autotuning is off in both, so the kernel choice does not depend on timing. Called by
    :func:`shared_net_factory` in the server before the network is built.

    Args:
        device: The torch device string the server runs on.
        numerics: ``"strict"`` or ``"torch_default"``.

    Returns:
        The flags in force, plus the device name and torch version.

    Raises:
        ValueError: On an unknown ``numerics``.
    """
    import torch

    if numerics not in NUMERICS:
        raise ValueError(f"unknown numerics {numerics!r}; expected {NUMERICS}")
    torch.backends.cudnn.allow_tf32 = numerics == "torch_default"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return {
        "numerics": numerics,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(torch.device(device)) if cuda else "cpu",
        "torch": str(torch.__version__),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": int(torch.get_num_threads()),
    }


def shared_net_factory(
    arm: str,
    seed: int,
    device: str = "cuda",
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
    numerics: str = "strict",
) -> Callable[[p8_adapter.EvalConfigs, int], BatchPolicy]:
    """Return a server factory for one shared dummy ``bc``/``dp`` network on ``device``.

    The network is ``ila.policies.registry.build_dummy(arm, seed, device, n_action_steps=,
    num_inference_steps=)`` (random init, Phase 2 throughput and determinism). Trained
    checkpoints plug in the same way once they exist.

    Args:
        arm: ``"bc"`` or ``"dp"``.
        seed: Initialisation seed = the run seed of every noise key.
        device: Torch device of the server.
        n_action_steps: ``Ta`` (dp only).
        num_inference_steps: ``K`` (dp only).
        numerics: ``"strict"`` (TF32 off) or ``"torch_default"`` (the labelled control).

    Returns:
        ``(cfgs, slots) -> BatchPolicy``.
    """

    def build(cfgs: p8_adapter.EvalConfigs, slots: int) -> BatchPolicy:
        del cfgs, slots
        from ila.policies.dp.policy import DPStudent
        from ila.policies.registry import build_dummy

        flags = configure_torch_numerics(device, numerics)
        student = build_dummy(
            arm,
            seed,
            device,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
        )
        if isinstance(student, DPStudent):
            return SharedDPPolicy(student, flags)
        return SharedBCPolicy(student, seed, flags)

    return build


def build_dummy_student(
    arm: str,
    seed: int,
    device: str,
    n_action_steps: int | None,
    num_inference_steps: int | None,
    cfgs: p8_adapter.EvalConfigs,
) -> Any:
    """Picklable ``PolicySpec.build`` for a random-init student (``functools.partial`` of it).

    ``ila.policies.registry.build_dummy(arm, seed, device, n_action_steps=,
    num_inference_steps=)``; ``cfgs`` is accepted and unused (P8's ``build(cfgs)`` shape).
    """
    del cfgs
    from ila.policies.registry import build_dummy

    if arm == "bc":
        return build_dummy(arm, seed, device)
    return build_dummy(
        arm, seed, device, n_action_steps=n_action_steps, num_inference_steps=num_inference_steps
    )


class TimedPolicy:
    """Wrap a :class:`BatchPolicy` and record every ``act_batch`` call's wall time and size.

    Attributes:
        inner: The wrapped policy.
        run_seed: ``inner.run_seed``.
        calls: ``(batch size, seconds)`` per ``act_batch`` call, in order.
        first_start: ``perf_counter`` at the first call's start, or ``None``.
        last_end: ``perf_counter`` at the last call's end, or ``None``.
    """

    def __init__(self, inner: BatchPolicy) -> None:
        """Wrap ``inner``."""
        self.inner = inner
        self.run_seed = int(inner.run_seed)
        self.calls: list[tuple[int, float]] = []
        self.first_start: float | None = None
        self.last_end: float | None = None

    def reset_slot(self, slot: int, episode_seed: int) -> None:
        """Forward."""
        self.inner.reset_slot(slot, episode_seed)

    def act_batch(self, requests: Sequence[SlotRequest]) -> NDArray[np.float64]:
        """Forward and time (wall clock, includes host-device copies)."""
        import time

        start = time.perf_counter()
        out = self.inner.act_batch(requests)
        end = time.perf_counter()
        if self.first_start is None:
            self.first_start = start
        self.last_end = end
        self.calls.append((len(requests), end - start))
        return out


@dataclass(frozen=True)
class BatchedArm:
    """One student on one ordered list, flown through the batching server.

    Attributes:
        spec: P8's ``PolicySpec`` (method, run seed; never privileged, never a feed). Its
            ``build`` must be picklable; workers only call it to read the declared flags.
        episodes: Listed episodes, in committed order.
        tags: The P9 columns.
        server_build: ``(cfgs, slots) -> BatchPolicy``, called once in the server; ``None``
            wraps ``spec.build`` in :class:`PerSlotPolicy` (the CPU single-sample path).
        batch_cap: ``B``, the largest batch the server runs.
        appearance_shift: Render variant, or ``None``.
        contract_variant: Student contract variant.
        timing_every: Stride of the timing assertion, control steps.
    """

    spec: p8_adapter.PolicySpec
    episodes: Sequence[p8_adapter.ListedEpisode]
    tags: RowTags
    server_build: Callable[[p8_adapter.EvalConfigs, int], BatchPolicy] | None = None
    batch_cap: int = 1
    appearance_shift: str | None = None
    contract_variant: str = "default"
    timing_every: int = DEFAULT_TIMING_EVERY


class _Flags:
    """The declared flags of the server's policy, checked against the spec in the worker."""

    privileged: bool = False
    needs_motion_feed: bool = False


class RemoteSlotActor:
    """A worker's actor: builds the contract input and asks the server for the action.

    Attributes:
        conn: The worker end of its pipe.
        slot: The worker's slot.
        run_seed: The spec's run seed (first field of every :class:`NoiseKey`).
        window: The frame/observation history.
        needs_frames: True.
    """

    needs_frames: bool = True

    def __init__(self, conn: Connection, slot: int, run_seed: int, contract: ObsContract):
        """Hold the pipe and a fresh window."""
        self.conn = conn
        self.slot = slot
        self.run_seed = run_seed
        self.window = StudentWindow(contract)
        self._episode_seed = 0

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        """Start an episode; the server resets the slot on the step-0 request.

        Raises:
            ValueError: If a context is passed (a student is never privileged).
        """
        if context is not None:
            raise ValueError("a vision student is never privileged")
        self.window.clear()
        self._episode_seed = int(listed.episode_seed)

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        """Send this step's contract-built input; block for the server's action."""
        si = self.window.push(step, obs, frame)
        key = NoiseKey(self.run_seed, self._episode_seed, step)
        self.conn.send(("act", SlotRequest(self.slot, key, si.images, si.proprio)))
        action = self.conn.recv()
        return np.asarray(action, dtype=np.float64)


def _worker(
    slot: int,
    conn: Connection,
    tasks: Any,
    chunks: Sequence[VisionTask],
) -> None:
    """Worker entry point: fly chunks from the queue until it yields ``None``."""
    try:
        init_worker_threads()
        for index in iter(tasks.get, None):
            task = chunks[index]

            def make(
                cfgs: p8_adapter.EvalConfigs, contract: ObsContract, task: VisionTask = task
            ) -> tuple[Actor, object]:
                del cfgs
                return RemoteSlotActor(conn, slot, int(task.spec.run_seed), contract), _Flags()

            rows = fly_chunk(task, make)
            conn.send(("rows", index, rows))
        conn.send(("done", slot))
    except BaseException:  # report everything to the server, then stop
        conn.send(("error", slot, traceback.format_exc()))
    finally:
        conn.close()


def _serve(
    conns: Sequence[Connection],
    policy: BatchPolicy,
    batch_cap: int,
) -> dict[int, list[dict[str, Any]]]:
    """Run the lockstep server loop until every worker is done; return rows by chunk index.

    Raises:
        WorkerError: If any worker raised.
    """
    live = list(range(len(conns)))
    rows: dict[int, list[dict[str, Any]]] = {}
    while live:
        requests: list[SlotRequest] = []
        for slot in list(live):
            msg = conns[slot].recv()
            while msg[0] == "rows":
                rows[int(msg[1])] = msg[2]
                msg = conns[slot].recv()
            if msg[0] == "done":
                live.remove(slot)
            elif msg[0] == "error":
                raise WorkerError(f"worker {msg[1]} failed:\n{msg[2]}")
            elif msg[0] == "act":
                req = msg[1]
                req.images.setflags(write=False)  # as read-only as the in-worker StudentInput
                req.proprio.setflags(write=False)
                requests.append(req)
            else:
                raise RuntimeError(f"unknown worker message {msg[0]!r}")
        requests.sort(key=lambda r: r.slot)
        for i in range(0, len(requests), batch_cap):
            batch = requests[i : i + batch_cap]
            for req in batch:
                if req.key.step == 0:
                    policy.reset_slot(req.slot, req.key.episode_seed)
                if req.key.run_seed != policy.run_seed:
                    raise ValueError(f"run seed {req.key.run_seed} != policy {policy.run_seed}")
            actions = np.asarray(policy.act_batch(batch), dtype=np.float64)
            if actions.shape != (len(batch), p8_adapter.ACTION_DIM):
                raise ValueError(f"act_batch returned shape {actions.shape}")
            for req, action in zip(batch, actions, strict=True):
                conns[req.slot].send(action)
    return rows


def run_batched(
    arm: BatchedArm,
    cfgs: p8_adapter.EvalConfigs,
    *,
    workers: int = p8_adapter.DEFAULT_WORKERS,
    chunk: int = p8_adapter.DEFAULT_CHUNK,
) -> list[dict[str, Any]]:
    """Fly a student through ``workers`` lockstep rendering workers and this process's server.

    Args:
        arm: The student, its list and batching.
        cfgs: P8's configs.
        workers: Rendering worker processes = slots (capped at the number of chunks).
        chunk: Episodes per chunk (the runner's cut; rows do not depend on it).

    Returns:
        One row per episode (``VISION_EPISODE_COLUMNS``), in list order.

    Raises:
        ValueError: On a non-positive ``workers``/``chunk``/``batch_cap``, a privileged or
            feed-consuming spec, or a policy whose run seed differs from the spec's.
        WorkerError: If a worker raised.
    """
    if workers < 1 or chunk < 1 or arm.batch_cap < 1:
        raise ValueError(f"workers, chunk, batch_cap must be positive: {workers}, {chunk}")
    if arm.spec.privileged or arm.spec.needs_motion_feed:
        raise ValueError(f"{arm.spec.method}: a batched student is never privileged or fed")
    chunks = [
        VisionTask(
            episodes=arm.episodes[i : i + chunk],
            spec=arm.spec,
            cfgs=cfgs,
            tags=arm.tags,
            render="full",
            appearance_shift=arm.appearance_shift,
            contract_variant=arm.contract_variant,
            timing_every=arm.timing_every,
        )
        for i in range(0, len(arm.episodes), chunk)
    ]
    if not chunks:
        return []
    n = min(workers, len(chunks))
    set_single_thread_env()
    init_worker_threads()
    build = per_slot_factory(arm.spec) if arm.server_build is None else arm.server_build
    policy = build(cfgs, n)
    if int(policy.run_seed) != int(arm.spec.run_seed):
        raise ValueError(f"policy run seed {policy.run_seed} != spec {arm.spec.run_seed}")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for index in range(len(chunks)):
        queue.put(index)
    for _ in range(n):
        queue.put(None)
    pairs = [ctx.Pipe(duplex=True) for _ in range(n)]
    procs = [
        ctx.Process(target=_worker, args=(slot, pairs[slot][1], queue, chunks), daemon=True)
        for slot in range(n)
    ]
    for proc in procs:
        proc.start()
    for _, child in pairs:
        child.close()
    try:
        by_chunk = _serve([parent for parent, _ in pairs], policy, arm.batch_cap)
    except BaseException:
        for proc in procs:
            proc.terminate()
        raise
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
        for parent, _ in pairs:
            parent.close()
    missing = sorted(set(range(len(chunks))) - set(by_chunk))
    if missing:
        raise WorkerError(f"chunks never returned: {missing}")
    return [row for index in range(len(chunks)) for row in by_chunk[index]]
