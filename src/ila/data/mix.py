"""``T_mix``: the balanced per-cell ppo/gated assignment (plan §1.4; P3-D1 decision, 2026-10-27).

``T_mix`` flies each episode with P8 ``ppo`` or P8 ``gated``. The choice is not an independent
coin: within each cell (a sea state of the demo list) exactly half the episodes go to each
teacher, by a seeded permutation.

The rule (:func:`balanced_assignment`), for cells given in order, each an ordered list of
episode keys ``(realization key, episode seed)``:

1. Cell entropy: ``[seed, H("T_mix"), H(canonical JSON of the cell's ordered keys)]``, where
   ``H`` is the SHA-256 label entropy (``ila.data.episodes.label_entropy``, never ``hash()``).
2. ``perm = default_rng(SeedSequence(entropy)).permutation(n)``.
3. ``n_ppo = n // 2``, plus one if ``n`` is odd and this is the 1st, 3rd, 5th ... odd cell in
   the given order (the 2nd, 4th ... odd cells give their extra episode to ``gated``). Over
   all cells the two counts therefore differ by at most one, and are equal when the total is
   even.
4. Episodes at cell positions ``perm[:n_ppo]`` fly ``ppo``; the rest fly ``gated``.

It is a pure function of the seed and the ordered keys (and, for odd cells only, of the cell
order), so a re-flown episode gets the same teacher, and Phase 8 can build H2's ``T_mix``
reference on P8's frozen ``id`` SS4 rows with the same function
(:func:`assign_rows` on that list, one cell).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np

from ila.data.episodes import label_entropy

__all__ = ["MIX_LABEL", "MIX_POLICIES", "assign_rows", "balanced_assignment", "cell_entropy"]

#: The label mixed into every ``T_mix`` cell's entropy.
MIX_LABEL: Final[str] = "T_mix"

#: The two P8 policies of the mixture, in rule order (the first gets the odd surplus first).
MIX_POLICIES: Final[tuple[str, str]] = ("ppo", "gated")

#: An episode's key: ``(realization key, episode seed)``.
type EpisodeKey = tuple[tuple[Any, ...], int]


def _canonical(keys: Sequence[EpisodeKey]) -> str:
    return json.dumps(
        [[list(key), int(seed)] for key, seed in keys], separators=(",", ":"), allow_nan=False
    )


def cell_entropy(seed: int, keys: Sequence[EpisodeKey]) -> list[int]:
    """Return the entropy words of one cell's permutation (rule step 1)."""
    return [int(seed), label_entropy(MIX_LABEL), label_entropy(_canonical(keys))]


def balanced_assignment(seed: int, cells: Sequence[Sequence[EpisodeKey]]) -> list[list[str]]:
    """Assign ``ppo``/``gated`` to every episode, balanced within each cell (module rule).

    Args:
        seed: The ``T_mix`` seed (``configs/data/demos.yaml``, 20261023).
        cells: Cells in order, each the ordered episode keys of that cell.

    Returns:
        Per cell, the policy of each episode, in the cell's order.

    Raises:
        ValueError: If a cell is empty or repeats an episode key.
    """
    out: list[list[str]] = []
    odd_seen = 0
    for keys in cells:
        n = len(keys)
        if n == 0 or len(set(keys)) != n:
            raise ValueError("a T_mix cell must be non-empty without repeated episodes")
        n_first = n // 2
        if n % 2:
            n_first += 1 if odd_seen % 2 == 0 else 0
            odd_seen += 1
        perm = np.random.default_rng(np.random.SeedSequence(cell_entropy(seed, keys))).permutation(
            n
        )
        policy = [MIX_POLICIES[1]] * n
        for position in perm[:n_first]:
            policy[int(position)] = MIX_POLICIES[0]
        out.append(policy)
    return out


def assign_rows(
    seed: int,
    rows: Sequence[Any],
    *,
    cell_of: Callable[[Any], str] = lambda r: str(r.ss),
) -> list[str]:
    """Assign ``ppo``/``gated`` to listed episodes (P8 ``ListedEpisode`` rows), balanced per cell.

    Cells are formed by ``cell_of`` (sea state by default) in order of first appearance; each
    cell's keys are its rows' ``(realization key, episode seed)`` in list order.

    Args:
        seed: The ``T_mix`` seed.
        rows: Listed episodes (with ``key`` and ``episode_seed``), in list order.
        cell_of: Row -> cell label.

    Returns:
        The policy of each row, in ``rows`` order.
    """
    cells: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        cells.setdefault(cell_of(row), []).append(i)
    keys = [
        [(tuple(rows[i].key), int(rows[i].episode_seed)) for i in idx] for idx in cells.values()
    ]
    out = [""] * len(rows)
    for idx, policies in zip(cells.values(), balanced_assignment(seed, keys), strict=True):
        for i, policy in zip(idx, policies, strict=True):
            out[i] = policy
    return out
