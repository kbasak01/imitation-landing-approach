"""Vision-student datasets over the demo store, decompressed into RAM (Phase 4).

Read path (P1-D2 items 3, 4, 6; P3-D1 §8 obligations; ``tests/test_data_store.py``):

* every array comes from ``DemoStore.student_inputs(name, variant)`` (``images`` and the
  variant's ``proprio``, nothing else) and ``DemoStore.labels(name)`` (the clean teacher label);
* episode lengths and window indices come from those arrays, never from ``meta()``; which
  episodes to read (the success filter, "first N") is decided before, by
  :func:`ila.train.sampling.choose_episodes`, the only training code that reads ``meta()``;
* no store-audit accessor is referenced in this file (``tests/test_render_access.py``).

The stores are lzf-compressed HDF5 (which cannot be memory-mapped), so :class:`FrameTable`
decompresses the chosen episodes once into contiguous RAM arrays (``ppo_2000`` first 1 000:
about 61.6 k frames, 1.7 GB of uint8 pixels).

Windows and targets (il-conventions "Alignment" and "Chunks"; indices only from
:mod:`ila.policies.dp.chunking`). A sample is anchored at control step ``t`` of one episode
(1/30 s model scale per step, lambda = 1/25):

* ``images``: ``uint8[To, 96, 96, 3]`` at :func:`chunking.obs_window_steps` ``(t, To)`` (oldest
  first; indices < 0 repeat step 0, i.e. first-frame padding). The 96 -> 88 px crop is applied
  later, by the encoder: random at train (explicit generator), centre at eval (P3-D1 §3);
* ``proprio``: ``float32[To, D]`` at the same steps (P8 units; ``D`` = 13 default, 10
  ``minus_last_action``, 14 ``plus_clock``);
* ``target``: BC ``label[t]`` (``float32[3]``); Diffusion Policy ``label`` at
  :func:`chunking.target_steps` ``(t, To, Tp, T)`` (``float32[Tp, 3]``, steps < 0 repeat
  ``label[0]``, steps >= T repeat ``label[T-1]``). Normalised world-frame velocity setpoints
  (x v_max = 1.5 m/s model scale).

Every class here is registered in ``STUDENT_DATASET_CLASSES`` (``tests/test_data_store.py``):
the windowed contract, the exact-columns and the poison tests run over all of them.
``bc_state``'s dataset (P8's 25-d observation, privileged by design) is deliberately **not**
here: it lives in :mod:`ila.train.state_dataset`, the one training file with a reviewed
``AUDIT_ALLOWLIST`` entry, so this file never needs one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ila.data.store import DemoStore
from ila.policies.dp import chunking

__all__ = ["SAMPLE_KEYS", "BCWindowDataset", "DPChunkDataset", "FrameTable"]

#: The keys of every sample and batch: nothing else, under any name.
SAMPLE_KEYS: Final[tuple[str, ...]] = ("images", "proprio", "target")


class FrameTable:
    """Chosen episodes, decompressed into contiguous RAM arrays (student path only).

    Attributes:
        names: Episode group names, in the order given.
        variant: Observation-contract variant of ``proprio``.
        images: ``uint8[N, 96, 96, 3]``, every frame of every episode, episode after episode.
        proprio: ``float32[N, D]``.
        label: ``float32[N, 3]``.
        episode_start: ``int64[E]``, first frame of each episode in the arrays.
        episode_len: ``int64[E]``, ``T`` of each episode (from the arrays).
        episode_of: ``int64[N]``, episode position of each frame.
        step_of: ``int64[N]``, control step ``t`` of each frame within its episode.
    """

    def __init__(
        self,
        names: Sequence[str],
        variant: str,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        label: NDArray[np.float32],
        episode_len: NDArray[np.int64],
    ) -> None:
        """Assemble from already-concatenated arrays (use :meth:`from_store`).

        Raises:
            ValueError: On inconsistent lengths, an empty table or non-finite floats.
        """
        n = int(np.sum(episode_len))
        if not names or len(names) != len(episode_len) or n < 1:
            raise ValueError("a frame table needs >= 1 episode and one length per name")
        if not (images.shape[0] == proprio.shape[0] == label.shape[0] == n):
            raise ValueError("images, proprio and label must hold sum(episode_len) frames")
        if not (np.all(np.isfinite(proprio)) and np.all(np.isfinite(label))):
            raise ValueError("non-finite proprio or label")
        self.names = tuple(names)
        self.variant = variant
        self.images = images
        self.proprio = proprio
        self.label = label
        self.episode_len = np.asarray(episode_len, dtype=np.int64)
        self.episode_start = np.concatenate([[0], np.cumsum(self.episode_len)[:-1]]).astype(
            np.int64
        )
        self.episode_of = np.repeat(np.arange(len(self.names), dtype=np.int64), self.episode_len)
        self.step_of = np.arange(n, dtype=np.int64) - self.episode_start[self.episode_of]

    @classmethod
    def from_store(
        cls, store: DemoStore, names: Sequence[str], variant: str = "default"
    ) -> FrameTable:
        """Decompress episodes through the student read path.

        Args:
            store: The demo store.
            names: Episode group names (chosen by :func:`ila.train.sampling.choose_episodes`).
            variant: Observation-contract variant.

        Returns:
            The table.

        Raises:
            ValueError: If an episode's images, proprio and labels differ in length.
        """
        images, proprio, labels, lengths = [], [], [], []
        for name in names:
            inputs = store.student_inputs(name, variant)
            label = store.labels(name)
            t_len = int(label.shape[0])
            if inputs.images.shape[0] != t_len or inputs.proprio.shape[0] != t_len:
                raise ValueError(f"{name}: images/proprio/label lengths differ")
            images.append(inputs.images)
            proprio.append(inputs.proprio)
            labels.append(label)
            lengths.append(t_len)
        return cls(
            names,
            variant,
            np.ascontiguousarray(np.concatenate(images), dtype=np.uint8),
            np.ascontiguousarray(np.concatenate(proprio), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(labels), dtype=np.float32),
            np.asarray(lengths, dtype=np.int64),
        )

    @classmethod
    def concat(cls, tables: Sequence[FrameTable]) -> FrameTable:
        """Concatenate tables (the iterated arms' aggregate), in the order given.

        Raises:
            ValueError: If the variants differ or an episode name repeats.
        """
        if len({t.variant for t in tables}) != 1:
            raise ValueError("cannot aggregate tables of different contract variants")
        names = [n for t in tables for n in t.names]
        if len(set(names)) != len(names):
            raise ValueError("an episode appears twice in the aggregate")
        return cls(
            names,
            tables[0].variant,
            np.ascontiguousarray(np.concatenate([t.images for t in tables])),
            np.ascontiguousarray(np.concatenate([t.proprio for t in tables])),
            np.ascontiguousarray(np.concatenate([t.label for t in tables])),
            np.concatenate([t.episode_len for t in tables]),
        )

    @property
    def n_frames(self) -> int:
        """``N``, frames in the table."""
        return int(self.images.shape[0])

    @property
    def n_episodes(self) -> int:
        """``E``, episodes in the table."""
        return len(self.names)

    def frames(self, name: str) -> dict[str, NDArray[np.generic]]:
        """Return one episode's frames ``{"images", "proprio", "target"}`` (target = label).

        Args:
            name: An episode in the table.

        Returns:
            ``images`` ``uint8[T, 96, 96, 3]``, ``proprio`` ``float32[T, D]``, ``target``
            ``float32[T, 3]``.
        """
        e = self.names.index(name)
        s = slice(int(self.episode_start[e]), int(self.episode_start[e] + self.episode_len[e]))
        return {"images": self.images[s], "proprio": self.proprio[s], "target": self.label[s]}

    def proprio_stats(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Per-dimension mean and std of ``proprio`` over every frame (encoder buffers).

        Returns:
            ``(mean, std)`` ``float32[D]``; a dimension with std < 1e-6 gets std 1.
        """
        x = self.proprio.astype(np.float64)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return x.mean(axis=0).astype(np.float32), std.astype(np.float32)


def _window_index(table: FrameTable, n_obs_steps: int) -> NDArray[np.int64]:
    """Global frame index of every sample's observation window, ``int64[N, To]``."""
    out = np.empty((table.n_frames, n_obs_steps), dtype=np.int64)
    for e in range(table.n_episodes):
        start, t_len = int(table.episode_start[e]), int(table.episode_len[e])
        for t in range(t_len):
            out[start + t] = [start + s for s in chunking.obs_window_steps(t, n_obs_steps)]
    return out


class BCWindowDataset:
    """BC samples: an observation window and ``label[t]``, one sample per stored frame.

    Attributes:
        table: The frames.
        n_obs_steps: ``To``.
        window_index: ``int64[N, To]``, global frame index of each sample's window.
    """

    def __init__(self, table: FrameTable, n_obs_steps: int) -> None:
        """Index the windows (through :func:`chunking.obs_window_steps`).

        Args:
            table: The frames.
            n_obs_steps: ``To`` (the contract's, 2).
        """
        self.table = table
        self.n_obs_steps = int(n_obs_steps)
        self.window_index = _window_index(table, self.n_obs_steps)

    def __len__(self) -> int:
        """Samples (= frames)."""
        return self.table.n_frames

    def target_rows(self, indices: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return the targets of samples ``indices``: ``label[t]``, ``float32[B, 3]``."""
        out: NDArray[np.float32] = self.table.label[indices]
        return out

    def batch(self, indices: NDArray[np.int64]) -> dict[str, NDArray[np.generic]]:
        """Gather samples into one batch.

        Args:
            indices: ``int64[B]`` sample indices.

        Returns:
            ``images`` ``uint8[B, To, 96, 96, 3]``, ``proprio`` ``float32[B, To, D]``,
            ``target`` per :meth:`target_rows`.
        """
        idx = np.asarray(indices, dtype=np.int64)
        win = self.window_index[idx]
        return {
            "images": self.table.images[win],
            "proprio": self.table.proprio[win],
            "target": self.target_rows(idx),
        }

    def __getitem__(self, i: int) -> dict[str, NDArray[np.generic]]:
        """Return sample ``i`` (the batch of one, without the leading axis)."""
        b = self.batch(np.asarray([i], dtype=np.int64))
        return {k: v[0] for k, v in b.items()}

    @classmethod
    def episode_samples(
        cls, store: DemoStore, name: str, variant: str = "default", n_obs_steps: int = 2
    ) -> dict[str, NDArray[np.generic]]:
        """Every sample of one episode, stacked (the registered adapter of the store tests)."""
        ds = cls(FrameTable.from_store(store, [name], variant), n_obs_steps)
        return ds.batch(np.arange(len(ds), dtype=np.int64))


class DPChunkDataset(BCWindowDataset):
    """Diffusion Policy samples: an observation window and an edge-repeated label chunk.

    Attributes:
        horizon: ``Tp``.
        target_index: ``int64[N, Tp]``, global frame index of each sample's target steps.
    """

    def __init__(self, table: FrameTable, n_obs_steps: int, horizon: int) -> None:
        """Index windows and targets (through :mod:`ila.policies.dp.chunking`).

        Args:
            table: The frames.
            n_obs_steps: ``To`` (2).
            horizon: ``Tp`` (16).
        """
        super().__init__(table, n_obs_steps)
        self.horizon = int(horizon)
        chunking.max_action_steps(self.n_obs_steps, self.horizon)
        out = np.empty((table.n_frames, self.horizon), dtype=np.int64)
        for e in range(table.n_episodes):
            start, t_len = int(table.episode_start[e]), int(table.episode_len[e])
            for t in range(t_len):
                steps = chunking.target_steps(t, self.n_obs_steps, self.horizon, t_len)
                out[start + t] = [start + s for s in steps]
        self.target_index = out

    def target_rows(self, indices: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return target chunks ``float32[B, Tp, 3]``: ``label`` at the chunk's steps."""
        out: NDArray[np.float32] = self.table.label[self.target_index[indices]]
        return out

    @classmethod
    def episode_samples(
        cls,
        store: DemoStore,
        name: str,
        variant: str = "default",
        n_obs_steps: int = 2,
        horizon: int = 16,
    ) -> dict[str, NDArray[np.generic]]:
        """Every sample of one episode, stacked (the registered adapter of the store tests)."""
        ds = cls(FrameTable.from_store(store, [name], variant), n_obs_steps, horizon)
        return ds.batch(np.arange(len(ds), dtype=np.int64))
