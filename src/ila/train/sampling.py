"""Episode choice (the success filter) and deterministic batch sampling.

Two things live here, outside the per-episode batchers of :mod:`ila.train.dataset`:

* :func:`choose_episodes` -- the only reader of ``DemoStore.meta()`` in training. It takes the
  first ``N`` episodes **by stream index, before filtering** (P3-D1 §2: a set's labelled-episode
  count is the ``N`` attempted episodes), then keeps the ``success`` ones if the config asks, and
  reports the kept fraction. Nothing it reads ever reaches a batch: it returns episode names.
* :class:`EpochSampler` and :func:`loader_seed` -- the data-loading randomness (il-conventions
  "Randomness": worker seeds from ``(run_seed, epoch, worker_id)``). Every random draw of the
  loader -- the epoch permutation, the per-batch crop offsets and the DDPM training noise -- comes
  from a generator seeded by :func:`loader_seed` over ``(run_seed, stage, epoch, worker_id,
  batch, stream)``. Batches are assembled in-process with ``worker_id = 0`` (an optional
  prefetch thread computes the same pure function of the step), so a batch is a pure function of
  ``(run_seed, stage, global step)``: identical for any number of prefetch threads, and a resumed
  run draws exactly the batches it would have drawn.

``stage`` is 0 for a single-stage run and the iteration index ``i`` for the iterated arms.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ila.data.store import DemoStore

__all__ = [
    "STREAM_CROP",
    "STREAM_HELDOUT",
    "STREAM_NOISE",
    "STREAM_PERM",
    "EpisodeChoice",
    "EpochSampler",
    "choose_episodes",
    "loader_seed",
]

#: Stream tags of :func:`loader_seed` (one independent generator per purpose).
STREAM_PERM: Final[int] = 0
STREAM_CROP: Final[int] = 1
STREAM_NOISE: Final[int] = 2
STREAM_HELDOUT: Final[int] = 3

_DOMAIN: Final[bytes] = b"ila-loader/1"


def loader_seed(
    run_seed: int, stage: int, epoch: int, worker_id: int, batch: int, stream: int
) -> int:
    """Return a 64-bit generator seed for one data-loading draw.

    BLAKE2b (8-byte digest, domain-separated from ``ila.policies.dp.policy.noise_seed``) of the
    six values as little-endian int64. Stable across processes and platforms.

    Args:
        run_seed: The training run's seed.
        stage: 0, or the iteration index of an iterated arm.
        epoch: Epoch index within the stage.
        worker_id: Loader worker (0: batches are built in-process).
        batch: Batch index within the epoch (-1 for the epoch permutation itself).
        stream: One of the ``STREAM_*`` tags.

    Returns:
        An integer in ``[0, 2**64)``.
    """
    packed = struct.pack("<qqqqqq", run_seed, stage, epoch, worker_id, batch, stream)
    digest = hashlib.blake2b(packed, digest_size=8, person=_DOMAIN[:16]).digest()
    return int.from_bytes(digest, "little")


class EpochSampler:
    """Uniform sampling without replacement within an epoch, deterministic in the step.

    Step ``s`` of a stage (0-based) uses batch ``b = s mod E`` of epoch ``e = s div E``, where
    ``E = n_samples div batch_size`` (the last partial batch of each epoch is dropped). Epoch
    ``e``'s order is a permutation drawn from :func:`loader_seed` ``(run_seed, stage, e, 0, -1,
    STREAM_PERM)``.

    Attributes:
        n_samples: Samples (frames) in the dataset.
        batch_size: Samples per batch.
        run_seed: Run seed.
        stage: Stage index.
        batches_per_epoch: ``E``.
    """

    def __init__(self, n_samples: int, batch_size: int, run_seed: int, stage: int = 0) -> None:
        """Create the sampler.

        Raises:
            ValueError: Unless ``1 <= batch_size <= n_samples`` (a batch never spans epochs).
        """
        if batch_size < 1 or n_samples < batch_size:
            raise ValueError(f"need 1 <= batch_size <= n_samples, got {batch_size}, {n_samples}")
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.run_seed = int(run_seed)
        self.stage = int(stage)
        self.batches_per_epoch = self.n_samples // self.batch_size
        self._perms: dict[int, NDArray[np.int64]] = {}
        self._lock = threading.Lock()

    def position(self, step: int) -> tuple[int, int]:
        """Return ``(epoch, batch)`` of a stage step (0-based)."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        return divmod(int(step), self.batches_per_epoch)

    def permutation(self, epoch: int) -> NDArray[np.int64]:
        """Return epoch ``epoch``'s sample order (cached; thread-safe)."""
        with self._lock:
            perm = self._perms.get(epoch)
            if perm is None:
                seed = loader_seed(self.run_seed, self.stage, epoch, 0, -1, STREAM_PERM)
                perm = np.random.Generator(np.random.PCG64(seed)).permutation(self.n_samples)
                perm = perm.astype(np.int64)
                self._perms = {k: v for k, v in self._perms.items() if k >= epoch - 2}
                self._perms[epoch] = perm
            return perm

    def indices(self, step: int) -> NDArray[np.int64]:
        """Return the sample indices of a stage step's batch."""
        epoch, b = self.position(step)
        perm = self.permutation(epoch)
        return perm[b * self.batch_size : (b + 1) * self.batch_size]

    def seed(self, step: int, stream: int) -> int:
        """Return the :func:`loader_seed` of a step's batch for ``stream`` (crop, noise)."""
        epoch, b = self.position(step)
        return loader_seed(self.run_seed, self.stage, epoch, 0, b, stream)


@dataclass(frozen=True)
class EpisodeChoice:
    """Which episodes of one store a run trains on, and the accounting.

    Attributes:
        store: The store file (as a string).
        start: First stream index taken.
        attempted: Episodes taken by stream index **before** filtering (the labelled count).
        names: Episode group names kept, in stream order.
        demo_indices: Their stream indices.
        success_filter: Whether only ``success`` episodes were kept.
        outcomes: Outcome class -> count over the attempted episodes.
    """

    store: str
    start: int
    attempted: int
    names: tuple[str, ...]
    demo_indices: tuple[int, ...]
    success_filter: bool
    outcomes: tuple[tuple[str, int], ...]

    @property
    def kept(self) -> int:
        """Episodes kept."""
        return len(self.names)

    @property
    def kept_fraction(self) -> float:
        """``kept / attempted``."""
        return self.kept / self.attempted

    def summary(self) -> dict[str, object]:
        """JSON-ready accounting for the manifest."""
        return {
            "store": self.store,
            "start": self.start,
            "attempted": self.attempted,
            "kept": self.kept,
            "kept_fraction": self.kept_fraction,
            "success_filter": self.success_filter,
            "outcomes": dict(self.outcomes),
            "first_kept": self.demo_indices[0] if self.demo_indices else None,
            "last_kept": self.demo_indices[-1] if self.demo_indices else None,
        }


def choose_episodes(
    store: DemoStore, first_n: int, success_filter: bool, *, start: int = 0
) -> EpisodeChoice:
    """Take stream indices ``start .. start + first_n - 1``, then optionally keep successes.

    Uses ``store.meta()`` only (stream index and outcome); returns names, never arrays.

    Args:
        store: The demo store.
        first_n: Episodes attempted (taken before filtering).
        success_filter: Keep only ``success`` episodes.
        start: First stream index.

    Returns:
        The :class:`EpisodeChoice`.

    Raises:
        ValueError: If the store does not hold every index of the range exactly once.
    """
    if first_n < 1 or start < 0:
        raise ValueError(f"need first_n >= 1 and start >= 0, got {first_n}, {start}")
    wanted = range(start, start + first_n)
    by_index: dict[int, tuple[str, bool, str]] = {}
    for name in store.episodes():
        meta = store.meta(name)
        if meta.demo_index in wanted:
            if meta.demo_index in by_index:
                raise ValueError(f"{store.path}: stream index {meta.demo_index} stored twice")
            by_index[meta.demo_index] = (name, meta.outcome == "success", meta.outcome)
    missing = [i for i in wanted if i not in by_index]
    if missing:
        raise ValueError(f"{store.path}: {len(missing)} of {wanted} missing ({missing[:5]})")
    outcomes: dict[str, int] = {}
    names: list[str] = []
    indices: list[int] = []
    for i in wanted:
        name, ok, outcome = by_index[i]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if ok or not success_filter:
            names.append(name)
            indices.append(i)
    return EpisodeChoice(
        store=str(store.path),
        start=start,
        attempted=first_n,
        names=tuple(names),
        demo_indices=tuple(indices),
        success_filter=success_filter,
        outcomes=tuple(sorted(outcomes.items())),
    )
