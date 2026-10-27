"""Train and select episode lists from P8's ``train`` and ``tune`` pools (plan §1.5, Phase 3).

What an episode is
------------------
An episode is ``(realization key, episode seed)``. The realization key is dmf's
``(ss, heading_deg, speed_kn, vessel, seed_ordinal)``. The environment draws the start offset
``t0`` and the initial state from the episode seed exactly as P8 does: every draw is resolved
by **P8's own** ``rld.eval.episodes.resolve_draws`` (a real ``DeckLandingAviary`` reset per
draw), so every row is a P8 ``ListedEpisode`` with P8's list columns and P8's runner, start
check and ``read_list`` accept it unchanged.

Pools (P8 P3-D2, inherited)
---------------------------
``rld.deck.splits.dev_pool(sim_cfg) -> (train, tune)``: frigate, SS3-SS5, headings 45/135/180
deg, speeds 0/6/12 kn; seed ordinals 0-26 (``train``, 729 realizations) and 27-31 (``tune``,
135). Neither shares a realization with any frozen list. SS6 and 90 deg are absent from both
and :func:`check_rows` refuses any row that carries them (CLAUDE.md non-negotiable 4).

* ``select`` (checkpoint selection and the DART sigma search only): 150 ``tune`` episodes,
  50 per SS3/SS4/SS5, ``splits/select.parquet``.
* ``train`` (demos, DART noise, DAgger rollouts): any number of ``train`` episodes, dealt
  round-robin and stratified by sea state.

The draw (P8's balanced round-robin, generator seed 20261023)
-------------------------------------------------------------
Per (list, sea state) cell the pool's ``R`` realizations at that sea state, in P8's
``sorted_specs`` order, are shuffled once by a generator seeded from
``(generator_seed, sha256(label), sha256(ss))``; cell episode ``i`` flies
``shuffled[i mod R]`` with an episode seed drawn from a generator seeded by
``(generator_seed, sha256(label), sha256(ss), i)``. This is P8's ``cell_order`` /
``_episode_rng`` rule, re-implemented here (P8 keeps ``_episode_rng`` private) and tested equal
to P8's ``draw_cell`` on P8's own cells. ``label`` is the list name (``"select"``, or the train
stream, ``"train"`` by default). Labels enter through SHA-256, never ``hash()``.

* ``select``: ``R`` = 45 per sea state, ``N`` = 50: five realizations get two episodes, forty
  get one.
* ``train``: the global episode ``j`` is cell episode ``j // 3`` of sea state
  ``(SS3, SS4, SS5)[j mod 3]``. Any prefix of ``n`` episodes therefore has per-sea-state
  counts within 1 of ``n / 3`` and every per-realization count within a cell within 1 of the
  others, and the first ``k`` episodes of a longer list are exactly the ``k``-episode list
  (so ``bc``'s 1 000 are a prefix of ``bc_matched``'s 2 000).

Streams (:class:`StreamDesign`, ``configs/data/demos.yaml``)
-----------------------------------------------------------
* ``demos``: every demo collection. ``ppo_2000`` flies global indices 0-1999
  (``splits/demos_0-1999.parquet``); ``pid_1000``, ``mix_1000`` and each DART sigma set fly
  0-999 (``splits/demos_0-999.parquet``), the same episodes as ``bc``'s 1 000, so teacher,
  mixture and noise comparisons are paired by episode.
* ``dagger_s{seed}_i{iteration}``: DAgger rollouts of student seed ``seed`` in iteration
  ``iteration`` (1-5), 200 each, drawn in Phase 5. Distinct labels give independent episode
  seeds; no ``(realization key, episode seed)`` is shared with ``demos`` (tested).
* ``label_parity``: the Phase 2 label-parity rollouts (no stored data).

Hashes and manifest (P8 ``results/episodes/MANIFEST.csv`` style)
-----------------------------------------------------------------
``splits/MANIFEST.csv`` records per file the SHA-256 of the parquet **bytes** and a
**content** SHA-256 over P8's canonical CSV (fixed column order, ``%.17g`` floats,
``true``/``false``, newline line endings). The file hash depends on the pyarrow version; the
content hash does not. The parquet schema is P8's list schema, so P8's ``read_list`` reads it.

P8 access
---------
This module does not import ``rld`` (only ``ila.p8_adapter`` does). The four P8 names the
draw needs -- ``dev_pool``, ``realization_key``, ``EpisodeDraw``, ``resolve_draws`` -- are
passed in through :func:`make_api`; :func:`p8_episode_api` wires them from the adapter.

Units: ``t0_model_s`` seconds **model** scale (lambda = 1/25: one model second is five
full-scale seconds); ``init_*_m`` metres model scale, world frame; ``heading_deg`` degrees;
``speed_kn`` knots full scale; seeds dimensionless.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ila import p8_adapter
from ila.p8_adapter import REPO_ROOT, ListedEpisode

__all__ = [
    "ADAPTER_NAMES_NEEDED",
    "DEFAULT_CONFIG",
    "DEMO_DRAW_RULE",
    "FORBIDDEN_HEADINGS_DEG",
    "FORBIDDEN_SEA_STATES",
    "LIST_COLUMNS",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "PAD",
    "SELECT_DRAW_RULE",
    "SELECT_NAME",
    "SPLITS_DIR",
    "TRAIN_DRAW_RULE",
    "TRAIN_STREAM",
    "DemoList",
    "Draw",
    "P8EpisodeApi",
    "RealizationKey",
    "RealizationSpecLike",
    "SplitManifestRow",
    "SplitsConfig",
    "StreamDesign",
    "canonical_csv",
    "cell_order",
    "check_rows",
    "content_sha256",
    "deal_cell",
    "demo_lists",
    "demo_rows",
    "draw_select",
    "draw_train",
    "episode_rng",
    "file_sha256",
    "frozen_keys",
    "label_entropy",
    "load_splits_config",
    "load_stream_design",
    "make_api",
    "manifest_row",
    "p8_episode_api",
    "pool_keys",
    "read_manifest",
    "read_split",
    "row_key",
    "select_rows",
    "train_name",
    "train_rows",
    "write_split",
]

#: dmf's canonical realization key ``(ss, heading_deg, speed_kn, vessel, seed_ordinal)``.
type RealizationKey = tuple[str, float, float, str, int]

#: Where the split files and their manifest live.
SPLITS_DIR: Final[Path] = REPO_ROOT / "splits"

#: The manifest's file name inside :data:`SPLITS_DIR`.
MANIFEST_NAME: Final[str] = "MANIFEST.csv"

#: The demo-data config that fixes the generator seed and the select-list shape.
DEFAULT_CONFIG: Final[Path] = REPO_ROOT / "configs" / "data" / "demos.yaml"

#: The selection list's name: its file stem, its rows' ``regime`` and its draw label.
SELECT_NAME: Final[str] = "select"

#: The default train stream label (rows' ``regime`` and the draw label).
TRAIN_STREAM: Final[str] = "train"

#: The pad every row carries: P8's primary-arm pad (``rld.eval.episodes.DEFAULT_PAD``); every
#: frozen list carries it too (tested).
PAD: Final[str] = "aft"

#: Never trained, tuned or selected on (CLAUDE.md non-negotiable 4).
FORBIDDEN_SEA_STATES: Final[tuple[str, ...]] = ("SS6",)

#: Never trained, tuned or selected on, degrees (CLAUDE.md non-negotiable 4).
FORBIDDEN_HEADINGS_DEG: Final[tuple[float, ...]] = (90.0,)

#: Draw rules recorded in the manifest. ``balanced_round_robin`` is P8's ``DRAW_RULE``.
SELECT_DRAW_RULE: Final[str] = "balanced_round_robin"
TRAIN_DRAW_RULE: Final[str] = "balanced_round_robin+ss_interleaved"

#: Draw rule of the demo lists: the train rule (a demo list is a train-stream prefix).
DEMO_DRAW_RULE: Final[str] = TRAIN_DRAW_RULE

#: Column order of every list file: P8's ``ListedEpisode`` field order (P8's ``LIST_COLUMNS``).
LIST_COLUMNS: Final[tuple[str, ...]] = tuple(f.name for f in fields(ListedEpisode))

#: Arrow schema of every list file: P8's list schema (tested equal to P8's frozen files').
_LIST_SCHEMA: Final = pa.schema(
    [
        ("regime", pa.string()),
        ("ss", pa.string()),
        ("index", pa.int64()),
        ("pad", pa.string()),
        ("vessel", pa.string()),
        ("heading_deg", pa.float64()),
        ("speed_kn", pa.float64()),
        ("realization_seed", pa.int64()),
        ("episode_seed", pa.int64()),
        ("t0_model_s", pa.float64()),
        ("init_x_m", pa.float64()),
        ("init_y_m", pa.float64()),
        ("init_z_m", pa.float64()),
        ("in_training_distribution", pa.bool_()),
    ]
)

#: The ``ila.p8_adapter`` names :func:`p8_episode_api` needs, as P8 defines them:
#: ``rld.deck.splits.dev_pool``, ``dmf.data.splits.realization_key`` (P8 re-imports it),
#: ``rld.eval.episodes.EpisodeDraw`` and ``rld.eval.episodes.resolve_draws``.
ADAPTER_NAMES_NEEDED: Final[tuple[str, ...]] = (
    "dev_pool",
    "realization_key",
    "EpisodeDraw",
    "resolve_draws",
)


# --------------------------------------------------------------------------- types


class RealizationSpecLike(Protocol):
    """Structural type of dmf's ``RealizationSpec`` (heading degrees, speed knots)."""

    @property
    def seed(self) -> int:
        """Seed ordinal within the grid cell, dimensionless."""
        ...

    @property
    def sea_state(self) -> str:
        """Sea state label, e.g. ``"SS4"``."""
        ...

    @property
    def heading_deg(self) -> float:
        """Encounter heading, degrees."""
        ...

    @property
    def speed_kn(self) -> float:
        """Forward speed, knots full scale."""
        ...

    @property
    def vessel(self) -> str:
        """Vessel config stem (dmf), e.g. ``"frigate"``."""
        ...


@dataclass(frozen=True)
class Draw:
    """One episode as drawn, before a reset has said where it starts.

    Field for field P8's ``rld.eval.episodes.EpisodeDraw`` (tested), so the wiring converts
    with ``EpisodeDraw(**asdict(draw))``.

    Attributes:
        regime: List label (``"select"`` or the train stream); P8 resets it as JONSWAP.
        ss: Sea state label.
        index: Position in the (list, sea state) cell's draw order.
        pad: Pad name.
        vessel: dmf vessel config stem.
        heading_deg: Encounter heading, degrees.
        speed_kn: Forward speed, knots full scale.
        realization_seed: dmf realization seed ordinal within its grid cell.
        episode_seed: The environment's episode seed; fixes ``t0``, the initial state and
            P8's noise stream.
        in_training_distribution: Whether the realization's grid cell is a dev-pool cell
            (always true here).
    """

    regime: str
    ss: str
    index: int
    pad: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int
    in_training_distribution: bool


@dataclass(frozen=True)
class P8EpisodeApi:
    """The P8 functions the draw needs, already bound to P8's committed configs.

    Attributes:
        dev_pool: ``() -> (train, tune)``, P8's ``dev_pool(cfgs.sim)``: realization specs in
            P8's ``sorted_specs`` order.
        realization_key: dmf's canonical key builder ``(ss, heading, speed, vessel, seed)``.
        resolve: ``draws -> rows``: P8's ``resolve_draws`` on P8's ``EpisodeDraw`` copies of
            the draws, one real reset each, rows in input order.
    """

    dev_pool: Callable[[], tuple[Sequence[RealizationSpecLike], Sequence[RealizationSpecLike]]]
    realization_key: Callable[[str, float, float, str, int], RealizationKey]
    resolve: Callable[[Sequence[Draw]], list[ListedEpisode]]


def make_api(
    *,
    dev_pool: Callable[[Any], tuple[Sequence[RealizationSpecLike], Sequence[RealizationSpecLike]]],
    realization_key: Callable[[str, float, float, str, int], RealizationKey],
    episode_draw: Callable[..., Any],
    resolve_draws: Callable[..., Sequence[ListedEpisode]],
    cfgs: Any,
    workers: int = 1,
    chunk: int = 25,
) -> P8EpisodeApi:
    """Bind P8's four episode functions to one config into a :class:`P8EpisodeApi`.

    Args:
        dev_pool: P8's ``rld.deck.splits.dev_pool(sim_cfg)``.
        realization_key: dmf's ``realization_key``.
        episode_draw: P8's ``rld.eval.episodes.EpisodeDraw`` class.
        resolve_draws: P8's ``rld.eval.episodes.resolve_draws(draws, cfgs, *, workers,
            chunk)``.
        cfgs: P8's ``EvalConfigs`` (``load_eval_configs()``); its ``sim`` defines the pools.
        workers: Reset processes; P8's result does not depend on it.
        chunk: Draws per reset task.

    Returns:
        The bound API.
    """

    def resolve(draws: Sequence[Draw]) -> list[ListedEpisode]:
        p8_draws = [episode_draw(**asdict(d)) for d in draws]
        return list(resolve_draws(p8_draws, cfgs, workers=workers, chunk=chunk))

    return P8EpisodeApi(
        dev_pool=lambda: dev_pool(cfgs.sim),
        realization_key=realization_key,
        resolve=resolve,
    )


def p8_episode_api(cfgs: Any = None, *, workers: int = 1, chunk: int = 25) -> P8EpisodeApi:
    """Return the episode API wired through ``ila.p8_adapter``.

    Args:
        cfgs: P8's ``EvalConfigs``; the committed ones when ``None``.
        workers: Reset processes; the rows do not depend on it.
        chunk: Draws per reset task.

    Returns:
        :func:`make_api` on the adapter's re-exports of P8's names.

    Raises:
        NotImplementedError: While the adapter does not yet re-export every name in
            :data:`ADAPTER_NAMES_NEEDED`.
    """
    missing = [name for name in ADAPTER_NAMES_NEEDED if not hasattr(p8_adapter, name)]
    if missing:
        raise NotImplementedError(
            f"ila.p8_adapter does not re-export {missing} yet; make_api() takes them as "
            "arguments until it does"
        )
    adapter: Any = p8_adapter  # typed access once the adapter exports ADAPTER_NAMES_NEEDED
    return make_api(
        dev_pool=adapter.dev_pool,
        realization_key=adapter.realization_key,
        episode_draw=adapter.EpisodeDraw,
        resolve_draws=adapter.resolve_draws,
        cfgs=p8_adapter.load_eval_configs() if cfgs is None else cfgs,
        workers=workers,
        chunk=chunk,
    )


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SplitsConfig:
    """The split-relevant part of ``configs/data/demos.yaml``.

    Attributes:
        generator_seed: The draw seed (20261023).
        train_pool: The P8 pool demos come from (``"train"``).
        stratify_by: Stratification axes (``("ss",)``).
        select_pool: The P8 pool the select list comes from (``"tune"``).
        select_per_ss: Select episodes per sea state (50).
        sea_states: Sea states of both lists, in file order (SS3, SS4, SS5).
    """

    generator_seed: int
    train_pool: str
    stratify_by: tuple[str, ...]
    select_pool: str
    select_per_ss: int
    sea_states: tuple[str, ...]


def load_splits_config(path: Path = DEFAULT_CONFIG) -> SplitsConfig:
    """Load and validate the split settings of the demo-data config.

    Args:
        path: ``configs/data/demos.yaml``.

    Returns:
        The :class:`SplitsConfig`.

    Raises:
        ValueError: If the pools are not ``train``/``tune``, stratification is not by sea
            state, or a forbidden sea state is listed.
    """
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    select = raw["select_list"]
    cfg = SplitsConfig(
        generator_seed=int(raw["generator_seed"]),
        train_pool=str(raw["pool"]),
        stratify_by=tuple(str(v) for v in raw["stratify_by"]),
        select_pool=str(select["pool"]),
        select_per_ss=int(select["per_ss"]),
        sea_states=tuple(str(v) for v in select["sea_states"]),
    )
    if (cfg.train_pool, cfg.select_pool) != ("train", "tune"):
        raise ValueError(f"pools must be train/tune, got {cfg.train_pool}/{cfg.select_pool}")
    if cfg.stratify_by != ("ss",):
        raise ValueError(f"stratification must be by sea state, got {cfg.stratify_by}")
    if bad := sorted(set(cfg.sea_states) & set(FORBIDDEN_SEA_STATES)):
        raise ValueError(f"forbidden sea states in config: {bad}")
    if cfg.select_per_ss < 1:
        raise ValueError(f"select_per_ss must be positive, got {cfg.select_per_ss}")
    return cfg


# --------------------------------------------------------------------------- the draw


def label_entropy(label: str) -> int:
    """Return a stable 64-bit entropy value for a label (P8's ``label_entropy``).

    Args:
        label: A list label or sea-state name.

    Returns:
        The first eight bytes of the label's UTF-8 SHA-256, big-endian, in ``[0, 2**64)``.
    """
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def episode_rng(generator_seed: int, label: str, ss: str, index: int) -> np.random.Generator:
    """Return one episode's generator, a pure function of its coordinates (P8's rule).

    Args:
        generator_seed: The draw seed.
        label: The list label.
        ss: The cell's sea state.
        index: The episode's index in its cell.

    Returns:
        ``default_rng(SeedSequence([seed, H(label), H(ss), index]))``.
    """
    entropy = [int(generator_seed), label_entropy(label), label_entropy(ss), int(index)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def cell_order(generator_seed: int, label: str, ss: str, n_realizations: int) -> list[int]:
    """Return a cell's shuffled realization order (P8's ``cell_order``).

    Args:
        generator_seed: The draw seed.
        label: The list label.
        ss: The cell's sea state.
        n_realizations: ``R``, the number of candidates.

    Returns:
        A permutation of ``range(R)``; cell episode ``i`` flies candidate ``order[i % R]``.
    """
    entropy = [int(generator_seed), label_entropy(label), label_entropy(ss)]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return [int(v) for v in rng.permutation(n_realizations)]


def deal_cell(
    candidates: Sequence[RealizationSpecLike],
    *,
    generator_seed: int,
    label: str,
    ss: str,
    indices: Iterable[int],
    pad: str = PAD,
) -> list[Draw]:
    """Deal episodes of one (list, sea state) cell by P8's balanced round-robin.

    Args:
        candidates: The cell's realizations in P8's ``sorted_specs`` order, all at ``ss``.
        generator_seed: The draw seed.
        label: The list label (rows' ``regime``).
        ss: The cell's sea state.
        indices: Cell episode indices to draw; episode ``i`` depends only on ``i`` and the
            cell.
        pad: Pad name.

    Returns:
        The draws in the order of ``indices``.

    Raises:
        ValueError: If there are no candidates or one is not at ``ss``.
    """
    if not candidates:
        raise ValueError(f"no candidate realization for {label!r} at {ss}")
    if off := [c for c in candidates if c.sea_state != ss]:
        raise ValueError(f"{len(off)} candidates of {label!r}/{ss} are at another sea state")
    order = cell_order(generator_seed, label, ss, len(candidates))
    draws: list[Draw] = []
    for index in indices:
        spec = candidates[order[int(index) % len(order)]]
        rng = episode_rng(generator_seed, label, ss, int(index))
        draws.append(
            Draw(
                regime=label,
                ss=ss,
                index=int(index),
                pad=pad,
                vessel=str(spec.vessel),
                heading_deg=float(spec.heading_deg),
                speed_kn=float(spec.speed_kn),
                realization_seed=int(spec.seed),
                episode_seed=int(rng.integers(0, 2**31 - 1)),
                in_training_distribution=True,
            )
        )
    return draws


def _by_ss(
    specs: Sequence[RealizationSpecLike], sea_states: Sequence[str]
) -> dict[str, list[RealizationSpecLike]]:
    """Split pool specs by sea state, keeping P8's order; refuse forbidden cells."""
    if bad := sorted({s.sea_state for s in specs} & set(FORBIDDEN_SEA_STATES)):
        raise ValueError(f"pool holds forbidden sea states {bad}")
    if any(_is_forbidden_heading(s.heading_deg) for s in specs):
        raise ValueError(f"pool holds a forbidden heading {FORBIDDEN_HEADINGS_DEG}")
    return {ss: [s for s in specs if s.sea_state == ss] for ss in sea_states}


def draw_select(api: P8EpisodeApi, cfg: SplitsConfig) -> list[Draw]:
    """Draw the selection list: ``select_per_ss`` ``tune`` episodes per sea state.

    Args:
        api: The P8 episode API.
        cfg: The split settings.

    Returns:
        Draws ordered by sea state (config order) then cell index.
    """
    _, tune = api.dev_pool()
    cells = _by_ss(tune, cfg.sea_states)
    draws: list[Draw] = []
    for ss in cfg.sea_states:
        draws += deal_cell(
            cells[ss],
            generator_seed=cfg.generator_seed,
            label=SELECT_NAME,
            ss=ss,
            indices=range(cfg.select_per_ss),
        )
    return draws


def draw_train(
    api: P8EpisodeApi,
    cfg: SplitsConfig,
    n: int,
    *,
    start: int = 0,
    stream: str = TRAIN_STREAM,
) -> list[Draw]:
    """Draw ``train``-pool episodes ``start .. start + n - 1`` of one stream.

    Global episode ``j`` is cell episode ``j // S`` of sea state ``sea_states[j % S]``
    (``S`` = 3), dealt by :func:`deal_cell`; so a prefix is stratified to within one episode
    and a longer list extends a shorter one.

    Args:
        api: The P8 episode API.
        cfg: The split settings.
        n: Episodes to draw.
        start: The first global index.
        stream: Stream label. Streams with different labels are independent draws.

    Returns:
        The draws in global order.

    Raises:
        ValueError: If ``n`` or ``start`` is negative, or the stream label is the select
            list's.
    """
    if n < 0 or start < 0:
        raise ValueError(f"n and start must be non-negative, got {n}, {start}")
    if stream == SELECT_NAME:
        raise ValueError(f"stream label {SELECT_NAME!r} is reserved for the select list")
    train, _ = api.dev_pool()
    cells = _by_ss(train, cfg.sea_states)
    n_ss = len(cfg.sea_states)
    draws: list[Draw] = []
    for j in range(start, start + n):
        ss = cfg.sea_states[j % n_ss]
        draws += deal_cell(
            cells[ss], generator_seed=cfg.generator_seed, label=stream, ss=ss, indices=[j // n_ss]
        )
    return draws


# --------------------------------------------------------------------------- keys and checks


def row_key(api: P8EpisodeApi, row: ListedEpisode | Draw) -> RealizationKey:
    """Return dmf's canonical realization key of one row or draw."""
    return api.realization_key(
        row.ss, row.heading_deg, row.speed_kn, row.vessel, row.realization_seed
    )


def pool_keys(api: P8EpisodeApi) -> tuple[frozenset[RealizationKey], frozenset[RealizationKey]]:
    """Return the realization keys of P8's ``(train, tune)`` pools."""
    train, tune = api.dev_pool()

    def keys(specs: Sequence[RealizationSpecLike]) -> frozenset[RealizationKey]:
        return frozenset(
            api.realization_key(s.sea_state, s.heading_deg, s.speed_kn, s.vessel, s.seed)
            for s in specs
        )

    return keys(train), keys(tune)


def frozen_keys() -> dict[str, frozenset[RealizationKey]]:
    """Return every frozen P8 list's realization keys, by list name (all five lists)."""
    return {
        name: frozenset(row.key for row in p8_adapter.read_list(path))
        for name, path in p8_adapter.list_paths().items()
    }


def _is_forbidden_heading(heading_deg: float) -> bool:
    return any(abs(float(heading_deg) - h) < 1e-9 for h in FORBIDDEN_HEADINGS_DEG)


def check_rows(api: P8EpisodeApi, rows: Sequence[ListedEpisode], pool: str) -> None:
    """Refuse a list that leaves its pool or touches anything held out.

    Args:
        api: The P8 episode API.
        rows: The resolved rows.
        pool: ``"train"`` or ``"tune"``: the pool every row must come from.

    Raises:
        ValueError: If a row is SS6 or 90 deg, outside ``pool``, in the other pool, in any
            frozen list, not flagged in-distribution, or if ``(key, episode seed)`` repeats.
    """
    train, tune = pool_keys(api)
    own, other = {"train": (train, tune), "tune": (tune, train)}[pool]
    keys = [row_key(api, r) for r in rows]
    problems: list[str] = []
    if any(r.ss in FORBIDDEN_SEA_STATES for r in rows):
        problems.append("forbidden sea state")
    if any(_is_forbidden_heading(r.heading_deg) for r in rows):
        problems.append("forbidden heading")
    if not set(keys) <= own:
        problems.append(f"realization outside the {pool} pool")
    if set(keys) & other:
        problems.append("realization in the other pool")
    for name, frozen in frozen_keys().items():
        if set(keys) & frozen:
            problems.append(f"realization in frozen list {name!r}")
    if not all(r.in_training_distribution for r in rows):
        problems.append("row not flagged in_training_distribution")
    episodes = [(k, r.episode_seed) for k, r in zip(keys, rows, strict=True)]
    if len(set(episodes)) != len(episodes):
        problems.append("repeated (realization key, episode seed)")
    if problems:
        raise ValueError(f"{pool} list refused: {problems}")


def select_rows(api: P8EpisodeApi, cfg: SplitsConfig) -> list[ListedEpisode]:
    """Draw, resolve through P8 and check the selection list."""
    rows = api.resolve(draw_select(api, cfg))
    check_rows(api, rows, cfg.select_pool)
    return rows


def train_rows(
    api: P8EpisodeApi, cfg: SplitsConfig, n: int, *, start: int = 0, stream: str = TRAIN_STREAM
) -> list[ListedEpisode]:
    """Draw, resolve through P8 and check ``n`` train episodes of one stream."""
    rows = api.resolve(draw_train(api, cfg, n, start=start, stream=stream))
    check_rows(api, rows, cfg.train_pool)
    return rows


def train_name(n: int, *, start: int = 0, stream: str = TRAIN_STREAM) -> str:
    """Return a train list's file stem, e.g. ``train_0-1999`` or ``dagger1_0-199``."""
    return f"{stream}_{start}-{start + n - 1}"


# --------------------------------------------------------------------------- stream design


@dataclass(frozen=True)
class StreamDesign:
    """Which ``train`` stream each kind of data comes from (``configs/data/demos.yaml``).

    Every demo collection flies a prefix of the one demo stream: ``ppo_2000`` its global
    indices 0-1999, ``pid_1000``, ``mix_1000`` and each DART set indices 0-999 (the ``bc``
    1 000). DAgger iteration ``i`` of student seed ``s`` draws its rollout episodes from its
    own stream :meth:`dagger_stream` ``(s, i)``, never from the demo stream.

    Attributes:
        demo_stream: The demo stream's label (``"demos"``).
        collection_sizes: Collection key in the YAML -> episodes (the ``dart`` key once for
            all of its sigma values).
        dagger_pattern: ``str.format`` pattern of a DAgger stream label, fields ``seed`` and
            ``iteration``.
        dagger_seeds: Student training seeds that run DAgger.
        dagger_iterations: DAgger iterations (numbered 1 .. ``dagger_iterations``).
        dagger_per_iteration: Rollout episodes per iteration.
    """

    demo_stream: str
    collection_sizes: Mapping[str, int]
    dagger_pattern: str
    dagger_seeds: tuple[int, ...]
    dagger_iterations: int
    dagger_per_iteration: int

    def dagger_stream(self, seed: int, iteration: int) -> str:
        """Return the stream label of DAgger iteration ``iteration`` (1-based) of ``seed``.

        Raises:
            ValueError: If the seed or iteration is outside the design.
        """
        if seed not in self.dagger_seeds or not 1 <= iteration <= self.dagger_iterations:
            raise ValueError(f"no DAgger stream for seed {seed}, iteration {iteration}")
        return self.dagger_pattern.format(seed=seed, iteration=iteration)

    def dagger_streams(self) -> tuple[str, ...]:
        """Return every DAgger stream label, seed-major."""
        return tuple(
            self.dagger_stream(s, i)
            for s in self.dagger_seeds
            for i in range(1, self.dagger_iterations + 1)
        )


@dataclass(frozen=True)
class DemoList:
    """One demo split file: global indices ``start .. start + n - 1`` of one stream.

    Attributes:
        stream: Stream label.
        start: First global index.
        n: Episodes.
    """

    stream: str
    start: int
    n: int

    @property
    def name(self) -> str:
        """The file stem, e.g. ``demos_0-1999``."""
        return train_name(self.n, start=self.start, stream=self.stream)


def load_stream_design(path: Path = DEFAULT_CONFIG) -> StreamDesign:
    """Load and validate the stream design of the demo-data config.

    Args:
        path: ``configs/data/demos.yaml``.

    Returns:
        The :class:`StreamDesign`.

    Raises:
        ValueError: If a stream label is reserved, repeated, or a DAgger label collides with
            the demo stream; or a size is not positive.
    """
    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    dagger = raw["dagger"]
    design = StreamDesign(
        demo_stream=str(raw["demo_stream"]),
        collection_sizes=MappingProxyType(
            {str(k): int(v["n"]) for k, v in raw["collections"].items()}
        ),
        dagger_pattern=str(dagger["stream"]),
        dagger_seeds=tuple(int(s) for s in dagger["seeds"]),
        dagger_iterations=int(dagger["iterations"]),
        dagger_per_iteration=int(dagger["episodes_per_iteration"]),
    )
    labels = [design.demo_stream, *design.dagger_streams()]
    if len(set(labels)) != len(labels):
        raise ValueError(f"stream labels repeat: {labels}")
    if reserved := sorted({SELECT_NAME, TRAIN_STREAM} & set(labels)):
        raise ValueError(f"reserved stream labels used: {reserved}")
    if min(design.collection_sizes.values()) < 1 or design.dagger_per_iteration < 1:
        raise ValueError("collection and DAgger sizes must be positive")
    return design


def demo_lists(design: StreamDesign) -> tuple[DemoList, ...]:
    """Return the demo split files the collections need: one prefix per distinct size.

    Args:
        design: The stream design.

    Returns:
        ``DemoList(demo_stream, 0, n)`` for each distinct collection size, ascending.
    """
    sizes = sorted(set(design.collection_sizes.values()))
    return tuple(DemoList(design.demo_stream, 0, n) for n in sizes)


def demo_rows(
    api: P8EpisodeApi, cfg: SplitsConfig, lists: Sequence[DemoList]
) -> dict[str, list[ListedEpisode]]:
    """Draw, resolve and check every demo list, resolving each stream only once.

    A row depends only on its draw (one reset each), so the longest list of a stream is
    resolved and the others are its slices (tested equal to resolving them directly).

    Args:
        api: The P8 episode API.
        cfg: The split settings.
        lists: The demo lists.

    Returns:
        List name -> checked rows, in global-index order.
    """
    out: dict[str, list[ListedEpisode]] = {}
    for stream in dict.fromkeys(d.stream for d in lists):
        mine = [d for d in lists if d.stream == stream]
        end = max(d.start + d.n for d in mine)
        rows = train_rows(api, cfg, end, stream=stream)
        for d in mine:
            out[d.name] = rows[d.start : d.start + d.n]
    return out


def read_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[ListedEpisode]:
    """Read one committed split file after checking it against ``MANIFEST.csv``.

    Args:
        name: File stem, e.g. ``demos_0-999`` or ``select``.
        splits_dir: The splits directory.

    Returns:
        The rows, in file order.

    Raises:
        ValueError: If the file has no manifest row, or its bytes or content differ from it.
    """
    path = splits_dir / f"{name}.parquet"
    records = [r for r in read_manifest(splits_dir / MANIFEST_NAME) if r["file"] == path.name]
    if len(records) != 1:
        raise ValueError(f"{path.name}: {len(records)} manifest rows, expected 1")
    if file_sha256(path) != records[0]["file_sha256"]:
        raise ValueError(f"{path.name}: file SHA-256 differs from MANIFEST.csv")
    rows = p8_adapter.read_list(path)
    if content_sha256(rows) != records[0]["content_sha256"]:
        raise ValueError(f"{path.name}: content SHA-256 differs from MANIFEST.csv")
    return rows


# --------------------------------------------------------------------------- files


def canonical_csv(rows: Sequence[ListedEpisode]) -> str:
    """Serialise rows canonically for the content hash (P8's ``canonical_csv``).

    Args:
        rows: The rows, in file order.

    Returns:
        CSV text: header :data:`LIST_COLUMNS`, ``%.17g`` floats, ``true``/``false``,
        decimal integers, newline line endings.

    Raises:
        ValueError: If a string contains a comma, quote or newline.
    """
    lines = [",".join(LIST_COLUMNS)]
    for row in rows:
        cells: list[str] = []
        for value in astuple(row):
            if isinstance(value, bool):
                cells.append("true" if value else "false")
            elif isinstance(value, int):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append("%.17g" % value)  # noqa: UP031 -- P8's format is the contract
            else:
                text = str(value)
                if any(ch in text for ch in ',"\n\r'):
                    raise ValueError(f"unescapable string in canonical CSV: {text!r}")
                cells.append(text)
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def content_sha256(rows: Sequence[ListedEpisode]) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_csv` of ``rows``."""
    return hashlib.sha256(canonical_csv(rows).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return p8_adapter.sha256_file(path)


def _to_table(rows: Sequence[ListedEpisode]) -> pa.Table:
    columns: dict[str, list[Any]] = {name: [] for name in LIST_COLUMNS}
    for row in rows:
        for name, value in zip(LIST_COLUMNS, astuple(row), strict=True):
            columns[name].append(value)
    return pa.Table.from_pydict(columns, schema=_LIST_SCHEMA)


@dataclass(frozen=True)
class SplitManifestRow:
    """One row of ``splits/MANIFEST.csv`` (P8's manifest columns plus pool and provenance).

    Attributes:
        file: File name inside ``splits/``.
        regime: The rows' list label.
        pool: The P8 pool the rows come from (``train`` or ``tune``).
        rows: Episodes in the file.
        n_cells: Sea-state cells.
        cells: The cells' sea states, ``|``-joined in file order.
        n_per_cell: Episodes per cell, ``|``-joined in cell order (a train prefix may differ
            by one between cells, so this is not a single number as in P8's manifest).
        n_realizations: Distinct realizations covered.
        generator_seed: The draw seed.
        draw: The draw rule.
        pad: The pad every row carries.
        frac_in_training_distribution: Fraction of rows with the flag set (1.0).
        p8_commit: P8's checked-out commit when the file was written.
        file_sha256: SHA-256 of the parquet bytes.
        content_sha256: SHA-256 of :func:`canonical_csv` of the rows.
    """

    file: str
    regime: str
    pool: str
    rows: int
    n_cells: int
    cells: str
    n_per_cell: str
    n_realizations: int
    generator_seed: int
    draw: str
    pad: str
    frac_in_training_distribution: float
    p8_commit: str
    file_sha256: str
    content_sha256: str


#: Column order of ``splits/MANIFEST.csv``.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = tuple(f.name for f in fields(SplitManifestRow))


def manifest_row(
    path: Path, rows: Sequence[ListedEpisode], *, pool: str, generator_seed: int, draw: str
) -> SplitManifestRow:
    """Summarise one written list file for the manifest.

    Raises:
        ValueError: If the rows are empty, carry more than one label or pad.
    """
    if not rows:
        raise ValueError(f"{path.name}: no rows")
    labels = {r.regime for r in rows}
    pads = {r.pad for r in rows}
    if len(labels) != 1 or len(pads) != 1:
        raise ValueError(f"{path.name}: mixed labels {sorted(labels)} or pads {sorted(pads)}")
    ss_order = list(dict.fromkeys(r.ss for r in rows))
    return SplitManifestRow(
        file=path.name,
        regime=next(iter(labels)),
        pool=pool,
        rows=len(rows),
        n_cells=len(ss_order),
        cells="|".join(ss_order),
        n_per_cell="|".join(str(sum(r.ss == ss for r in rows)) for ss in ss_order),
        n_realizations=len({r.key for r in rows}),
        generator_seed=generator_seed,
        draw=draw,
        pad=next(iter(pads)),
        frac_in_training_distribution=sum(r.in_training_distribution for r in rows) / len(rows),
        p8_commit=p8_adapter.p8_commit(),
        file_sha256=file_sha256(path),
        content_sha256=content_sha256(rows),
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read ``splits/MANIFEST.csv`` as raw string records (empty if the file is absent).

    Raises:
        ValueError: If the header is not :data:`MANIFEST_COLUMNS`.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: header {reader.fieldnames} != {list(MANIFEST_COLUMNS)}")
        return list(reader)


def _write_manifest(path: Path, records: Sequence[Mapping[str, str]]) -> None:
    ordered = sorted(records, key=lambda r: (r["file"] != f"{SELECT_NAME}.parquet", r["file"]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(ordered)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def write_split(
    name: str,
    rows: Sequence[ListedEpisode],
    out_dir: Path,
    *,
    pool: str,
    generator_seed: int,
    draw: str,
) -> SplitManifestRow:
    """Write one list as ``<name>.parquet`` and upsert its row in ``MANIFEST.csv``.

    The parquet uses P8's list schema and snappy compression, as P8 writes its lists. Other
    files' manifest rows are kept; rows are ordered ``select`` first, then by file name.

    Args:
        name: File stem.
        rows: Checked rows (:func:`select_rows` / :func:`train_rows`).
        out_dir: ``splits/``, created if missing.
        pool: ``"train"`` or ``"tune"``.
        generator_seed: The draw seed.
        draw: The draw rule.

    Returns:
        The file's manifest row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    pq.write_table(_to_table(rows), path, compression="snappy")
    row = manifest_row(path, rows, pool=pool, generator_seed=generator_seed, draw=draw)
    record = {
        k: (repr(v) if isinstance(v, float) else str(v))
        for k, v in zip(MANIFEST_COLUMNS, astuple(row), strict=True)
    }
    manifest = out_dir / MANIFEST_NAME
    kept = [r for r in read_manifest(manifest) if r["file"] != path.name]
    _write_manifest(manifest, [*kept, record])
    return row
