"""``bc_state``'s dataset: P8's full 25-d observation and the teacher label (Phase 4).

**Privileged by design** (P3-D1 §3; il-conventions "Randomness", declared exception 3): this
dataset reads ``teacher_obs`` -- every P8 field, the five privileged fields and
``time_fraction`` included -- through ``DemoStore.privileged_observations``. It is the only file
under ``ila/train`` with an ``AUDIT_ALLOWLIST`` entry in ``tests/test_render_access.py`` (that
name, nothing else), and it is kept apart from :mod:`ila.train.dataset` so that no vision-student
dataset shares a file with a privileged read. Its arrays feed only :class:`ila.policies.state_mlp
.StateMLP`; ``bc_state`` is never a vision row.

Alignment (il-conventions): sample ``t`` is ``(teacher_obs[t], label[t])``, the observation the
teacher saw and the action it chose there. No window (``bc_state`` sees one step). Units are
P8's per field; labels are normalised world-frame velocity setpoints (x v_max = 1.5 m/s model
scale, lambda = 1/25).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ila.data.store import DemoStore

__all__ = ["StateFrames"]


class StateFrames:
    """Chosen episodes' ``teacher_obs`` and ``label`` rows, decompressed into RAM.

    Attributes:
        names: Episode group names, in the order given.
        obs: ``float32[N, 25]``, P8's observation in P8's field order.
        label: ``float32[N, 3]``.
        episode_len: ``int64[E]`` (from the arrays).
    """

    def __init__(self, store: DemoStore, names: Sequence[str]) -> None:
        """Read the episodes (chosen by :func:`ila.train.sampling.choose_episodes`).

        Args:
            store: The demo store.
            names: Episode group names.

        Raises:
            ValueError: On no episodes, mismatched lengths or non-finite values.
        """
        if not names:
            raise ValueError("no episodes")
        obs, labels, lengths = [], [], []
        for name in names:
            o = store.privileged_observations(name)
            y = store.labels(name)
            if o.shape[0] != y.shape[0]:
                raise ValueError(f"{name}: teacher_obs and label lengths differ")
            obs.append(o)
            labels.append(y)
            lengths.append(int(y.shape[0]))
        self.names = tuple(names)
        self.obs: NDArray[np.float32] = np.ascontiguousarray(np.concatenate(obs), np.float32)
        self.label: NDArray[np.float32] = np.ascontiguousarray(np.concatenate(labels), np.float32)
        self.episode_len = np.asarray(lengths, dtype=np.int64)
        if not (np.all(np.isfinite(self.obs)) and np.all(np.isfinite(self.label))):
            raise ValueError("non-finite teacher_obs or label")

    def __len__(self) -> int:
        """Samples (= frames)."""
        return int(self.obs.shape[0])

    @property
    def n_frames(self) -> int:
        """Frames held."""
        return len(self)

    def batch(self, indices: NDArray[np.int64]) -> dict[str, NDArray[np.float32]]:
        """Gather ``{"obs": float32[B, 25], "target": float32[B, 3]}``."""
        idx = np.asarray(indices, dtype=np.int64)
        return {"obs": self.obs[idx], "target": self.label[idx]}
