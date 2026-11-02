"""The only place ``To``/``Tp``/``Ta`` indices are computed.

Notation (il-conventions, "Chunks"). Control step ``t`` is an integer, one step = 1/30 s
**model scale** (lambda = 1/25: 1/6 s full scale).

* ``To`` (``n_obs_steps``): the policy sees images and proprio at steps ``t-To+1 ... t``;
  steps < 0 repeat step 0.
* ``Tp`` (``horizon``): at replanning step ``t`` the model predicts the chunk
  ``a_hat[t-To+1 ... t-To+Tp]``, so chunk index ``j`` is step ``t-To+1+j``.
* ``Ta`` (``n_action_steps``): executed chunk indices ``To-1 ... To-2+Ta``, i.e. steps
  ``t ... t+Ta-1``. Hence **``Ta <= Tp - To + 1``** (``To = 2``, ``Tp = 16`` gives ``Ta <= 15``).
  Executing from index 0 instead of ``To-1`` gives a policy one step late.
* Without ensembling, replanning happens at ``t = 0, Ta, 2 Ta, ...``.
* Training targets for a sample anchored at ``t``: ``label[t-To+1 ... t-To+Tp]``; steps < 0
  repeat ``label[0]``, steps >= ``T`` repeat ``label[T-1]``.
* Temporal ensembling: replan every step; for step ``t`` average every prediction made for
  ``t`` by earlier replans with weights ``w_i = exp(-m i)``, ``i = 0`` the oldest, normalised
  (``m = 0.01``). **Which** replans count (P6-D1's reading of P3-D1 §3): those whose executed
  window covers ``t``, i.e. replans ``s`` in ``[max(0, t-Ta+1), t]``, each read at chunk index
  ``To-1+(t-s)`` (:func:`ensemble_sources`). ``Ta`` is then the ensemble window, a chunk
  contributes to exactly the steps it would have executed without ensembling, and ``Ta = 1``
  with ensembling is the same policy as ``Ta = 1`` without.

Every function here is pure integer arithmetic (and :func:`ensemble_weights`, numpy), so the
runner, the dataset and the policy all take their indices from one table-tested place
(``tests/test_dp_chunking.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Temporal-ensembling decay per prediction, ``m`` (dimensionless; il-conventions and
#: ``configs/train/dp.yaml`` ``inference_sweeps.temporal_ensembling.m``).
ENSEMBLE_DECAY: Final[float] = 0.01

__all__ = [
    "ENSEMBLE_DECAY",
    "ChunkSpec",
    "chunk_steps",
    "ensemble_sources",
    "ensemble_weights",
    "executed_indices",
    "executed_slice",
    "is_replan_step",
    "max_action_steps",
    "obs_window_steps",
    "replan_step",
    "target_steps",
]


def max_action_steps(n_obs_steps: int, horizon: int) -> int:
    """Return the largest admissible ``Ta``: ``Tp - To + 1``.

    Args:
        n_obs_steps: ``To`` (>= 1).
        horizon: ``Tp`` (>= ``To``).

    Returns:
        ``Tp - To + 1`` control steps.

    Raises:
        ValueError: If ``To < 1`` or ``Tp < To``.
    """
    if n_obs_steps < 1:
        raise ValueError(f"To must be >= 1, got {n_obs_steps}")
    if horizon < n_obs_steps:
        raise ValueError(f"Tp must be >= To, got Tp={horizon}, To={n_obs_steps}")
    return horizon - n_obs_steps + 1


@dataclass(frozen=True)
class ChunkSpec:
    """A validated ``(To, Tp, Ta)`` triple.

    Attributes:
        n_obs_steps: ``To``, observation steps per input.
        horizon: ``Tp``, predicted chunk length (control steps).
        n_action_steps: ``Ta``, executed steps per chunk, ``1 <= Ta <= Tp - To + 1``.
    """

    n_obs_steps: int
    horizon: int
    n_action_steps: int

    def __post_init__(self) -> None:
        """Validate the triple.

        Raises:
            ValueError: If ``To < 1``, ``Tp < To`` or ``Ta`` is outside
                ``[1, Tp - To + 1]``.
        """
        bound = max_action_steps(self.n_obs_steps, self.horizon)
        if not 1 <= self.n_action_steps <= bound:
            raise ValueError(
                f"Ta must satisfy 1 <= Ta <= Tp - To + 1 = {bound} "
                f"(To={self.n_obs_steps}, Tp={self.horizon}), got Ta={self.n_action_steps}"
            )

    def with_action_steps(self, n_action_steps: int) -> ChunkSpec:
        """Return the same ``To``/``Tp`` with another (validated) ``Ta``.

        Args:
            n_action_steps: The new ``Ta``.

        Returns:
            A new :class:`ChunkSpec`.
        """
        return ChunkSpec(self.n_obs_steps, self.horizon, n_action_steps)

    @property
    def executed(self) -> slice:
        """Chunk-index slice executed after a replan (:func:`executed_slice`)."""
        return executed_slice(self.n_obs_steps, self.horizon, self.n_action_steps)


def _check_step(t: int) -> None:
    if t < 0:
        raise ValueError(f"control step must be >= 0, got {t}")


def obs_window_steps(t: int, n_obs_steps: int) -> tuple[int, ...]:
    """Return the steps whose image and proprio the policy sees at step ``t``.

    Args:
        t: Current control step (>= 0).
        n_obs_steps: ``To``.

    Returns:
        ``To`` steps, oldest first: ``max(0, t-To+1+i)`` for ``i = 0 ... To-1``.
    """
    _check_step(t)
    if n_obs_steps < 1:
        raise ValueError(f"To must be >= 1, got {n_obs_steps}")
    return tuple(max(0, t - n_obs_steps + 1 + i) for i in range(n_obs_steps))


def chunk_steps(t: int, n_obs_steps: int, horizon: int) -> tuple[int, ...]:
    """Return the control step each chunk index refers to, for a chunk planned at ``t``.

    Args:
        t: Replanning step (>= 0).
        n_obs_steps: ``To``.
        horizon: ``Tp``.

    Returns:
        ``Tp`` steps, ``t-To+1+j`` for ``j = 0 ... Tp-1``. Entries may be negative (the
        chunk's past part, never executed).
    """
    _check_step(t)
    max_action_steps(n_obs_steps, horizon)
    return tuple(t - n_obs_steps + 1 + j for j in range(horizon))


def executed_slice(n_obs_steps: int, horizon: int, n_action_steps: int) -> slice:
    """Return the slice of chunk indices executed after a replan.

    Args:
        n_obs_steps: ``To``.
        horizon: ``Tp``.
        n_action_steps: ``Ta``.

    Returns:
        ``slice(To-1, To-1+Ta)``: chunk indices ``To-1 ... To-2+Ta`` (steps ``t ... t+Ta-1``).

    Raises:
        ValueError: If ``Ta`` is outside ``[1, Tp - To + 1]``.
    """
    ChunkSpec(n_obs_steps, horizon, n_action_steps)
    return slice(n_obs_steps - 1, n_obs_steps - 1 + n_action_steps)


def executed_indices(n_obs_steps: int, horizon: int, n_action_steps: int) -> tuple[int, ...]:
    """Return the executed chunk indices as a tuple (see :func:`executed_slice`).

    Args:
        n_obs_steps: ``To``.
        horizon: ``Tp``.
        n_action_steps: ``Ta``.

    Returns:
        ``(To-1, ..., To-2+Ta)``.
    """
    s = executed_slice(n_obs_steps, horizon, n_action_steps)
    return tuple(range(s.start, s.stop))


def is_replan_step(t: int, n_action_steps: int) -> bool:
    """Return whether a new chunk is planned at step ``t`` (no ensembling).

    Args:
        t: Control step (>= 0).
        n_action_steps: ``Ta`` (>= 1).

    Returns:
        ``t % Ta == 0``.
    """
    _check_step(t)
    if n_action_steps < 1:
        raise ValueError(f"Ta must be >= 1, got {n_action_steps}")
    return t % n_action_steps == 0


def replan_step(t: int, n_action_steps: int) -> int:
    """Return the replanning step whose chunk is executing at step ``t`` (no ensembling).

    Args:
        t: Control step (>= 0).
        n_action_steps: ``Ta`` (>= 1).

    Returns:
        ``t - t % Ta``; the action at ``t`` is chunk index ``To-1 + t % Ta`` of that plan.
    """
    _check_step(t)
    if n_action_steps < 1:
        raise ValueError(f"Ta must be >= 1, got {n_action_steps}")
    return t - t % n_action_steps


def target_steps(t: int, n_obs_steps: int, horizon: int, episode_len: int) -> tuple[int, ...]:
    """Return the label steps forming the training target of a sample anchored at ``t``.

    Args:
        t: Anchor step, ``0 <= t < T``.
        n_obs_steps: ``To``.
        horizon: ``Tp``.
        episode_len: ``T``, the number of labelled steps in the episode (>= 1).

    Returns:
        ``Tp`` steps ``clip(t-To+1+j, 0, T-1)``: before the start repeats ``label[0]``, past
        the end repeats ``label[T-1]``.

    Raises:
        ValueError: If ``t`` is outside ``[0, T)``.
    """
    if episode_len < 1 or not 0 <= t < episode_len:
        raise ValueError(f"anchor t={t} outside [0, {episode_len})")
    return tuple(min(max(s, 0), episode_len - 1) for s in chunk_steps(t, n_obs_steps, horizon))


def ensemble_sources(
    t: int, n_obs_steps: int, horizon: int, n_action_steps: int
) -> tuple[tuple[int, int], ...]:
    """Return the predictions temporal ensembling averages for step ``t``, oldest first.

    With ensembling the policy replans at every step. Step ``t`` averages the chunks planned
    at ``s = max(0, t-Ta+1) ... t``, chunk planned at ``s`` read at index ``To-1+(t-s)``
    (which :func:`chunk_steps` maps back to step ``t``). These are exactly the chunks whose
    executed window (:func:`executed_slice`) covers ``t``; at the start of an episode there
    are fewer (``t+1`` when ``t < Ta-1``). Weight ``i`` of :func:`ensemble_weights` goes to
    entry ``i`` (``i = 0`` the oldest). ``Ta = 1`` gives the single pair ``(t, To-1)``: the
    same action as without ensembling.

    Args:
        t: Control step (>= 0; 1/30 s model scale).
        n_obs_steps: ``To``.
        horizon: ``Tp``.
        n_action_steps: ``Ta``, the ensemble window (``1 <= Ta <= Tp - To + 1``).

    Returns:
        ``((s, j), ...)`` pairs: replanning step ``s`` and chunk index ``j``, ``s`` ascending.

    Raises:
        ValueError: If ``t < 0`` or ``(To, Tp, Ta)`` is inadmissible.
    """
    _check_step(t)
    ChunkSpec(n_obs_steps, horizon, n_action_steps)
    first = max(0, t - n_action_steps + 1)
    return tuple((s, n_obs_steps - 1 + (t - s)) for s in range(first, t + 1))


def ensemble_weights(n: int, m: float = ENSEMBLE_DECAY) -> NDArray[np.float64]:
    """Return normalised temporal-ensembling weights for ``n`` predictions of one step.

    Args:
        n: Number of predictions for the step (>= 1).
        m: Decay rate per prediction (dimensionless; 0.01 by the convention).

    Returns:
        ``w_i = exp(-m i) / sum_k exp(-m k)``, ``i = 0`` the **oldest** prediction.
    """
    if n < 1:
        raise ValueError(f"need at least one prediction, got {n}")
    w = np.exp(-m * np.arange(n, dtype=np.float64))
    out: NDArray[np.float64] = w / w.sum()
    return out
