"""Latency-in-the-loop arm: the measured p99 as a delay of d control steps (P3-D1 §7).

Simulated delay, in control steps (30 Hz, 33.333 ms model scale, 166.7 ms full scale). PyBullet
does not advance while a network computes, so the result of a request is computed when the
request is issued, on the arm's own CPU fp32 single-sample path, and held back for ``d`` steps.
``d`` is read from ``results/latency/delay.json`` (:func:`ila.deploy.latency.score`;
:func:`read_delays`), one per arm, the same for every seed.

Rules, as P3-D1 §7 states them:

* **One worker; requests never overlap.** A request is issued with observation ``t`` only when
  none is in flight; its result is available at step ``t + d``.
* **BC** (:class:`DelayedBCStudent`): the returned action is held from ``t + d`` until the next
  result arrives; the next request is issued at ``t + d``.
* **dp** (:class:`DelayedDiffusionStudent`): a request is issued when the executing chunk has
  ``d`` steps left (immediately if fewer, and at the first step of an episode, when no chunk is
  executing). The returned chunk executes from index ``(To - 1) + d``, aligned in time, for
  ``Ta`` steps (:func:`ila.policies.dp.chunking.delayed_chunk_index`); past ``Tp - 1`` the last
  chunk action is held (from ``d > 7`` at ``To = 2``, ``Tp = 16``, ``Ta = 8``). If the
  executing chunk runs out before the next result arrives, its last executed action is held.
  The request's noise key is ``(run_seed, episode_seed, t)`` with ``t`` the step it is issued
  at, exactly the synchronous student's key at its replanning step.
* **Start of an episode:** the action is zero until the first result arrives.
* **d = 0:** the arm equals the synchronous arm row for row (tested).

Order within step ``t``: (1) a result due at ``t`` lands; (2) if nothing is in flight and the
request rule holds, a request is issued with observation ``t`` (with ``d = 0`` it lands at once);
(3) the action is chosen. The step counter and episode seed are read only for this bookkeeping
and the noise key (P2-D3 item 4); they never enter a network.

Interface for ``eval-auditor`` (``ila.eval.arms`` / ``ila.eval.matrix``):
:func:`delayed_checkpoint_policy` is a picklable ``PolicySpec.build`` with the shape of
:func:`ila.train.select.checkpoint_policy` plus ``delay``. Build it with
``functools.partial(delayed_checkpoint_policy, str(ckpt.path), d, n_action_steps=8,
num_inference_steps=10)`` (BC: no ``Ta``/``K``), wrap it in
``p8_adapter.callable_spec(arm, build, run_seed=seed)`` and tag rows with
``RowTags(..., Ta=8, K=10, ensemble=False, latency_steps=d)`` (BC: ``latency_steps=d`` only),
tag ``latency_arm``. Rows are then flown by the unchanged runner on the CPU path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila import p8_adapter
from ila.policies.bc import BCStudent
from ila.policies.dp import chunking
from ila.policies.dp.policy import ChunkedDiffusionStudent
from ila.policies.encoder import VisionStudentBase

__all__ = [
    "DELAY_FILE",
    "DelayedBCStudent",
    "DelayedDiffusionStudent",
    "delayed_checkpoint_policy",
    "read_delays",
    "wrap_delayed",
]

#: ``results/latency/delay.json`` (written by ``ila.deploy.latency.score``).
DELAY_FILE: Final[Path] = p8_adapter.REPO_ROOT / "results" / "latency" / "delay.json"


def _check_delay(delay: int) -> int:
    d = int(delay)
    if d < 0:
        raise ValueError(f"delay must be >= 0 control steps, got {delay}")
    return d


class DelayedBCStudent(VisionStudentBase):
    """A BC student whose action takes effect ``d`` control steps after its observation.

    Attributes:
        inner: The synchronous student (stateless across steps).
        delay: ``d``, control steps.
    """

    def __init__(self, inner: VisionStudentBase, delay: int) -> None:
        """Wrap ``inner`` (its window checks, config and device are reused)."""
        self.inner = inner
        self.delay = _check_delay(delay)
        self.cfg = inner.cfg
        self.device = getattr(inner, "device", torch.device("cpu"))
        self._t = 0
        self._held = np.zeros(3, np.float32)
        self._pending: tuple[int, NDArray[np.float32]] | None = None

    def _reset_episode(self, seed: int) -> None:
        self.inner.reset(seed)
        self._t = 0
        self._held = np.zeros(3, np.float32)
        self._pending = None

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """The held action at this step (module docstring); ``float32[3]`` in [-1, 1]."""
        t = self._t
        if self._pending is not None and self._pending[0] == t:
            self._held = self._pending[1]
            self._pending = None
        if self._pending is None:
            action = np.asarray(self.inner.act_vision(image_stack, proprio), np.float32)
            self.inference_calls += 1
            if self.delay == 0:
                self._held = action
            else:
                self._pending = (t + self.delay, action)
        self._t += 1
        return self._held.copy()


class DelayedDiffusionStudent(VisionStudentBase):
    """A chunked diffusion student under P3-D1 §7's delayed, non-overlapping requests.

    Attributes:
        inner: The synchronous student; only its ``sample_batch`` and ``chunk_spec`` are used.
        delay: ``d``, control steps.
    """

    def __init__(self, inner: ChunkedDiffusionStudent, delay: int) -> None:
        """Wrap ``inner``.

        Raises:
            ValueError: If ``inner`` uses temporal ensembling (not part of the latency arm).
        """
        if inner.temporal_ensembling:
            raise ValueError("the latency arm flies dp without temporal ensembling (P3-D1 §7)")
        self.inner = inner
        self.delay = _check_delay(delay)
        self.cfg = inner.cfg
        self.device = getattr(inner, "device", torch.device("cpu"))
        self.chunk_spec = inner.chunk_spec
        self._seed: int | None = None
        self._clear()

    def _clear(self) -> None:
        self._t = 0
        self._chunk: NDArray[np.float32] | None = None
        self._start = 0
        self._pending: tuple[int, NDArray[np.float32]] | None = None
        self._last = np.zeros(3, np.float32)

    def _reset_episode(self, seed: int) -> None:
        self._seed = seed
        self._clear()

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """The action at this step (module docstring); ``float32[3]`` in [-1, 1].

        Raises:
            RuntimeError: If ``reset`` has not been called.
        """
        if self._seed is None:
            raise RuntimeError("call reset(seed) before act_vision")
        img, prop = self.inner._check_window(image_stack, proprio)
        spec = self.chunk_spec
        ta, d, t = spec.n_action_steps, self.delay, self._t
        if self._pending is not None and self._pending[0] == t:
            self._chunk, self._start = self._pending[1], t
            self._pending = None
        left = self._start + ta - t if self._chunk is not None else 0
        if self._pending is None and left <= d:
            # The clock and episode seed: noise key and schedule only (P2-D3 item 4).
            key = (self.inner.run_seed, self._seed, t)
            chunk = self.inner.sample_batch(img[None], prop[None], [key])[0]
            self.inference_calls += 1
            if d == 0:
                self._chunk, self._start = chunk, t
            else:
                self._pending = (t + d, chunk)
        if self._chunk is not None and t < self._start + ta:
            j = chunking.delayed_chunk_index(spec.n_obs_steps, spec.horizon, ta, d, t - self._start)
            self._last = np.asarray(self._chunk[j], np.float32)
        self._t += 1
        return self._last.copy()


def wrap_delayed(student: Any, delay: int) -> DelayedBCStudent | DelayedDiffusionStudent:
    """Wrap a synchronous BC or chunked diffusion student.

    Raises:
        TypeError: For any other student (``modular``, ``bc_state``).
    """
    if isinstance(student, ChunkedDiffusionStudent):
        return DelayedDiffusionStudent(student, delay)
    if isinstance(student, BCStudent) or (
        isinstance(student, VisionStudentBase) and not hasattr(student, "chunk_spec")
    ):
        return DelayedBCStudent(student, delay)
    raise TypeError(f"no latency-arm wrapper for {type(student).__name__}")


def delayed_checkpoint_policy(
    path: str,
    delay: int,
    cfgs: p8_adapter.EvalConfigs,
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
) -> Any:
    """Picklable ``PolicySpec.build``: a checkpoint's EMA student (CPU fp32), delayed by ``d``.

    Args:
        path: The checkpoint.
        delay: ``d``, control steps (:func:`read_delays`).
        cfgs: P8's configs (unused).
        n_action_steps: ``dp`` only: ``Ta`` (8).
        num_inference_steps: ``dp`` only: ``K`` (10).

    Returns:
        A :class:`DelayedBCStudent` or :class:`DelayedDiffusionStudent`.
    """
    from ila.policies.registry import load_student

    del cfgs
    inner = load_student(
        Path(path),
        "cpu",
        weights="ema",
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
    )
    return wrap_delayed(inner, delay)


def read_delays(path: Path = DELAY_FILE) -> Mapping[str, int]:
    """Arm -> d from ``delay.json``.

    Raises:
        ValueError: If an arm has no d (its fallback row was not timed).
    """
    rec = json.loads(Path(path).read_text("utf-8"))
    out: dict[str, int] = {}
    for arm, a in rec["arms"].items():
        if a.get("d") is None:
            raise ValueError(f"{path}: {arm} has no delay ({a.get('status')})")
        out[arm] = int(a["d"])
    return out
