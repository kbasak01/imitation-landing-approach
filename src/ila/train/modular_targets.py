"""``modular``'s estimator targets and truth logging: the one ``ila`` file reading true ``priv``.

**Privileged by design, as a target, never as an input** (P3-D1 §3 ``modular``; §8 leak RISK 1).
This module is the only file under ``ila/train``, ``ila/policies``, ``ila/deploy`` and
``ila/eval`` that references the store accessor ``estimator_targets`` and the contract accessor
``audit_field``; its ``AUDIT_ALLOWLIST`` entry in ``tests/test_render_access.py`` names exactly
those two, pending the ``privileged-leak-auditor`` review that P3-D1 §8 requires before any
``modular`` run. ``tests/test_render_access.py`` also pins which files may name this module
(``train/modular_targets.py``, ``train/loop.py``, ``train/smoke.py``,
``train/modular_eval.py``). What it reads, and where each goes:

* :class:`EstimatorDataset` -- inputs through the student path only (:class:`FrameTable`,
  ``DemoStore.student_inputs(name, "default")``: pixels and the 13-d proprio), and the
  **target** ``priv[t]`` (``DemoStore.estimator_targets``, ``float32[T, 11]``) of the same step
  (il-conventions alignment: ``image[t]``, ``proprio[t]`` and ``priv[t]`` share one
  ``stepSimulation`` count). Batches hold exactly ``images``, ``proprio`` and ``target``; the
  target enters the loss only (:func:`ila.policies.modular.estimator_loss`), and the target's
  training-set mean/std become the network's fixed output normaliser. ``time_fraction`` is
  never read here.
* :func:`true_priv` -- the 11 privileged fields of one env observation, selected by name
  (``ObsContract.audit_field``), for the estimation-error study's **logging** of the closed-loop
  truth (:mod:`ila.train.modular_eval`) and the smoke's target replay check. Its result is
  written to a log; it is never passed to a policy.

Units (P8's, model scale lambda = 1/25): ``rel_position`` m and ``rel_velocity`` m/s in the
drone-yaw frame, ``deck_normal`` unit drone-yaw, ``rel_tilt`` rad, ``height`` m along the deck
normal. Full scale: lengths x 25, velocities x 5, times x 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ila.data.store import DemoStore
from ila.obs.split import ObsContract
from ila.train.dataset import BCWindowDataset, FrameTable

__all__ = ["EstimatorDataset", "replay_priv_check", "true_priv"]


class EstimatorDataset(BCWindowDataset):
    """``modular`` samples: an observation window and ``priv[t]``, one sample per stored frame.

    The window is :class:`BCWindowDataset`'s (``chunking.obs_window_steps``; images and the
    default 13-d proprio from the student read path); only the target differs.

    Attributes:
        table: The frames (default contract).
        n_obs_steps: ``To``.
        targets: ``float32[N, 11]``, ``priv`` of every frame, aligned with ``table``.
    """

    def __init__(self, table: FrameTable, targets: NDArray[np.float32], n_obs_steps: int) -> None:
        """Index the windows and hold the targets.

        Args:
            table: The frames (variant must be ``default``).
            targets: ``float32[N, 11]`` aligned with ``table``'s frames.
            n_obs_steps: ``To`` (2).

        Raises:
            ValueError: On a non-default table, misaligned or non-finite targets.
        """
        if table.variant != "default":
            raise ValueError("modular uses the default contract")
        if targets.ndim != 2 or targets.shape[0] != table.n_frames:
            raise ValueError(f"targets {targets.shape} for {table.n_frames} frames")
        if not np.all(np.isfinite(targets)):
            raise ValueError("non-finite estimator targets")
        super().__init__(table, n_obs_steps)
        self.targets: NDArray[np.float32] = np.ascontiguousarray(targets, dtype=np.float32)

    @classmethod
    def from_store(
        cls, store: DemoStore, names: Sequence[str], n_obs_steps: int
    ) -> EstimatorDataset:
        """Read episodes (chosen by :func:`ila.train.sampling.choose_episodes`).

        Args:
            store: The demo store.
            names: Episode group names.
            n_obs_steps: ``To``.

        Returns:
            The dataset.

        Raises:
            ValueError: If an episode's ``priv`` and its student frames differ in length.
        """
        table = FrameTable.from_store(store, names, "default")
        targets = []
        for e, name in enumerate(table.names):
            priv = store.estimator_targets(name)
            if priv.shape[0] != int(table.episode_len[e]):
                raise ValueError(f"{name}: priv and student frames differ in length")
            targets.append(priv)
        return cls(table, np.concatenate(targets).astype(np.float32), n_obs_steps)

    def target_rows(self, indices: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return ``priv[t]`` of samples ``indices``, ``float32[B, 11]`` (P8 units)."""
        out: NDArray[np.float32] = self.targets[indices]
        return out

    def target_stats(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Per-dimension mean and std of the targets over every frame (the output normaliser).

        Returns:
            ``(mean, std)`` ``float32[11]``; a dimension with std < 1e-6 gets std 1.
        """
        x = self.targets.astype(np.float64)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return x.mean(axis=0).astype(np.float32), std.astype(np.float32)


def true_priv(contract: ObsContract, obs: ArrayLike) -> NDArray[np.float32]:
    """The privileged block of P8 observation(s), selected by field name, for LOGGING only.

    Args:
        contract: The default contract.
        obs: P8 observation(s) ``(..., 25)``.

    Returns:
        ``float32[..., 11]`` in contract order (``contract.privileged``), P8 units.
    """
    return np.concatenate([contract.audit_field(obs, f) for f in contract.privileged], axis=-1)


class _PrivRecorder:
    """Fly a teacher (reset at t = 0, once per step, in order) and log ``priv`` of steps 0..n-1."""

    needs_frames: bool = False

    def __init__(self, policy: Any, contract: ObsContract, n_steps: int) -> None:
        self.policy = policy
        self.contract = contract
        self.n_steps = n_steps
        self.priv: list[NDArray[np.float32]] = []

    def begin(self, listed: Any, context: object) -> None:
        if context is not None:
            raise ValueError("the teacher here is non-privileged; no context")
        self.policy.reset(listed.episode_seed, None)

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        del frame
        if step < self.n_steps:
            self.priv.append(true_priv(self.contract, np.asarray(obs, np.float32)))
        return np.asarray(self.policy.act(obs), dtype=np.float64)


def replay_priv_check(
    store: DemoStore, name: str, listed: Any, teacher_name: str, n_steps: int = 4
) -> dict[str, Any]:
    """Stored ``priv`` steps ``0 .. n-1`` vs a fresh teacher rollout's observation fields.

    The target side of il-conventions' "can't land" checklist item 1: ``priv[t]`` in the store
    must equal, byte for byte, the privileged fields of the observation a fresh rollout of the
    same listed episode by the same teacher produces at step ``t`` (render ``full``, as stored).

    Args:
        store: The demo store.
        name: Episode group name.
        listed: Its ``ListedEpisode``.
        teacher_name: The P8 teacher that flew it (``ppo`` for ``ppo_2000``).
        n_steps: Steps compared.

    Returns:
        ``{episode, steps, priv_identical, max_abs_diff, passed}``.
    """
    import functools

    from ila import p8_adapter
    from ila.eval import runner  # lazy: runner imports pybullet

    stored = store.estimator_targets(name)
    n = min(n_steps, int(stored.shape[0]))
    task = runner.VisionTask(
        episodes=[listed],
        spec=p8_adapter.callable_spec(
            "smoke:replay_priv",
            functools.partial(p8_adapter.teacher, teacher_name, 0),
            run_seed=0,
        ),
        cfgs=p8_adapter.load_eval_configs(),
        tags=runner.RowTags(arm="smoke:replay_priv"),
        render="full",
    )
    recorders: list[_PrivRecorder] = []

    def make(cfgs: Any, contract: ObsContract) -> tuple[Any, object]:
        policy = p8_adapter.teacher(teacher_name, 0, cfgs)
        rec = _PrivRecorder(policy, contract, n)
        recorders.append(rec)
        return rec, policy

    runner.fly_chunk(task, make)
    fresh = np.stack(recorders[0].priv[:n])
    same = [bool(fresh[t].tobytes() == stored[t].tobytes()) for t in range(n)]
    return {
        "episode": name,
        "teacher": teacher_name,
        "steps": n,
        "priv_identical": sum(same),
        "max_abs_diff": float(np.max(np.abs(fresh.astype(np.float64) - stored[:n]))),
        "passed": all(same),
    }
