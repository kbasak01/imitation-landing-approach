"""``T_mix`` balanced per-sea-state assignment (P3-D1 decision 2026-10-27; ``ila.data.mix``).

Covers: exact 500/500 on ``demos_0-999``; exact halves per sea state with the odd-cell rule
(SS3 334 -> 167/167, SS4 333 -> 167 ppo / 166 gated, SS5 333 -> 166 / 167); determinism and
dependence on the seed and keys; reuse on any ``ListedEpisode`` list (Phase 8 builds H2's
``T_mix`` reference on P8's frozen ``id`` SS4 rows with the same function -- read here only to
check the function applies; nothing is flown, trained, tuned or selected).
"""

from __future__ import annotations

from collections import Counter

import pytest

from ila import p8_adapter
from ila.data import episodes as ep
from ila.data import mix

SEED = 20261023


@pytest.fixture(scope="module")
def rows() -> list[p8_adapter.ListedEpisode]:
    return ep.read_split("demos_0-999")


def _keys(n: int, offset: int = 0) -> list[mix.EpisodeKey]:
    return [(("SS4", 135.0, 6.0, "frigate", i % 7), 1000 + offset + i) for i in range(n)]


def test_demos_exact_halves(rows: list[p8_adapter.ListedEpisode]) -> None:
    policies = mix.assign_rows(SEED, rows)
    assert Counter(policies) == {"ppo": 500, "gated": 500}
    per_ss = Counter((r.ss, p) for r, p in zip(rows, policies, strict=True))
    assert dict(per_ss) == {
        ("SS3", "ppo"): 167,
        ("SS3", "gated"): 167,
        ("SS4", "ppo"): 167,
        ("SS4", "gated"): 166,
        ("SS5", "ppo"): 166,
        ("SS5", "gated"): 167,
    }


def test_deterministic_and_seeded(rows: list[p8_adapter.ListedEpisode]) -> None:
    first = mix.assign_rows(SEED, rows)
    assert first == mix.assign_rows(SEED, list(rows))
    assert first != mix.assign_rows(SEED + 1, rows)
    # A prefix is a different cell content, hence a different (still balanced) permutation.
    short = mix.assign_rows(SEED, rows[:300])
    assert Counter(short) == {"ppo": 150, "gated": 150}


def test_pure_function_of_cell_keys() -> None:
    keys = _keys(10)
    (a,) = mix.balanced_assignment(SEED, [keys])
    (b,) = mix.balanced_assignment(SEED, [list(keys)])
    assert a == b and Counter(a) == {"ppo": 5, "gated": 5}
    (c,) = mix.balanced_assignment(SEED, [_keys(10, offset=1)])
    assert Counter(c) == {"ppo": 5, "gated": 5}
    assert mix.cell_entropy(SEED, keys) != mix.cell_entropy(SEED, _keys(10, offset=1))
    assert mix.cell_entropy(SEED, keys) != mix.cell_entropy(SEED, keys[::-1])
    # The same cell, alone or after another even cell, gets the same assignment.
    assert mix.balanced_assignment(SEED, [_keys(4, 50), keys])[1] == a


def test_odd_cells_alternate_the_surplus() -> None:
    cells = [_keys(5, 0), _keys(4, 100), _keys(7, 200), _keys(3, 300)]
    out = mix.balanced_assignment(SEED, cells)
    # Odd cells (5, 7, 3) give their extra episode to ppo, gated, ppo in turn.
    assert [Counter(c)["ppo"] for c in out] == [3, 2, 3, 2]
    assert sum(Counter(c)["ppo"] for c in out) == 10  # 19 episodes: totals differ by one
    assert sum(Counter(c)["gated"] for c in out) == 9
    even = mix.balanced_assignment(SEED, [_keys(5, 0), _keys(7, 200)])
    assert [Counter(c)["ppo"] for c in even] == [3, 3]  # 12 episodes: exactly 6 / 6
    with pytest.raises(ValueError):
        mix.balanced_assignment(SEED, [[]])
    with pytest.raises(ValueError):
        mix.balanced_assignment(SEED, [_keys(2) + _keys(1)])


def test_reusable_on_frozen_id_ss4_rows() -> None:
    """Phase 8: one cell = the listed ``id`` SS4 episodes, balanced ppo/gated."""
    id_ss4 = [r for r in p8_adapter.read_list(p8_adapter.list_path("id")) if r.ss == "SS4"]
    policies = mix.assign_rows(SEED, id_ss4)
    n = len(id_ss4)
    assert Counter(policies) == {"ppo": n - n // 2, "gated": n // 2}
    assert policies == mix.assign_rows(SEED, id_ss4)
    (direct,) = mix.balanced_assignment(SEED, [[(r.key, r.episode_seed) for r in id_ss4]])
    assert policies == direct
