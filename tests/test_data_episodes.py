"""Phase 3 step 1: train/select episode lists from P8's pools (plan §1.5, non-negotiable 4).

P8's episode functions are injected from ``rld`` here (tests may import it; ``src/`` and
``scripts/`` may not) until ``ila.p8_adapter`` re-exports them;
``test_adapter_wiring_matches_direct`` takes over automatically once it does.
"""

from __future__ import annotations

import dataclasses
import typing
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import rld.eval.episodes as p8_episodes
from dmf.data.splits import realization_key as dmf_realization_key
from rld.deck.splits import build_all_splits, dev_pool, part_specs

from ila import p8_adapter
from ila.data import episodes as ep

SELECT = ep.SPLITS_DIR / "select.parquet"
MANIFEST = ep.SPLITS_DIR / ep.MANIFEST_NAME
FROZEN = tuple(p8_adapter.list_paths())
N_TRAIN = 2000


@pytest.fixture(scope="module")
def cfg() -> ep.SplitsConfig:
    return ep.load_splits_config()


@pytest.fixture(scope="module")
def api() -> ep.P8EpisodeApi:
    return ep.make_api(
        dev_pool=dev_pool,
        realization_key=dmf_realization_key,
        episode_draw=p8_episodes.EpisodeDraw,
        resolve_draws=p8_episodes.resolve_draws,
        cfgs=p8_adapter.load_eval_configs(),
        workers=1,
    )


@pytest.fixture(scope="module")
def pools(api: ep.P8EpisodeApi) -> tuple[frozenset[ep.RealizationKey], ...]:
    return ep.pool_keys(api)


@pytest.fixture(scope="module")
def frozen() -> dict[str, frozenset[ep.RealizationKey]]:
    return ep.frozen_keys()


@pytest.fixture(scope="module")
def select() -> list[p8_adapter.ListedEpisode]:
    return p8_adapter.read_list(SELECT)


@pytest.fixture(scope="module")
def train_draws(api: ep.P8EpisodeApi, cfg: ep.SplitsConfig) -> list[ep.Draw]:
    return ep.draw_train(api, cfg, N_TRAIN)


# --------------------------------------------------------------------------- config / P8 parity


def test_config(cfg: ep.SplitsConfig) -> None:
    assert cfg.generator_seed == 20261023
    assert (cfg.train_pool, cfg.select_pool, cfg.stratify_by) == ("train", "tune", ("ss",))
    assert cfg.select_per_ss == 50
    assert cfg.sea_states == ("SS3", "SS4", "SS5")


def test_draw_mirrors_p8_episode_draw() -> None:
    assert [f.name for f in dataclasses.fields(ep.Draw)] == [
        f.name for f in dataclasses.fields(p8_episodes.EpisodeDraw)
    ]
    assert typing.get_type_hints(ep.Draw) == typing.get_type_hints(p8_episodes.EpisodeDraw)
    assert ep.LIST_COLUMNS == p8_episodes.LIST_COLUMNS


def test_pad_is_p8_default_and_on_every_frozen_row() -> None:
    assert ep.PAD == p8_episodes.DEFAULT_PAD
    for path in p8_adapter.list_paths().values():
        assert {r.pad for r in p8_adapter.read_list(path)} == {ep.PAD}


@pytest.mark.parametrize("label", ["select", "train", "id", "SS4"])
def test_entropy_and_order_equal_p8(label: str) -> None:
    assert ep.label_entropy(label) == p8_episodes.label_entropy(label)
    for n in (1, 45, 243):
        assert ep.cell_order(20261023, label, "SS4", n) == p8_episodes.cell_order(
            20261023, label, "SS4", n
        )


@pytest.mark.parametrize(
    ("regime", "ss"), [("id", "SS3"), ("id", "SS5"), ("unseen_heading", "SS4")]
)
def test_dealer_reproduces_p8_frozen_cell(regime: str, ss: str) -> None:
    """Our dealer on P8's own cell reproduces P8's committed frozen rows (draw fields)."""
    cfgs = p8_adapter.load_eval_configs()
    candidates = [
        s for s in part_specs(build_all_splits(cfgs.sim)[regime], "test") if s.sea_state == ss
    ]
    ours = ep.deal_cell(
        candidates,
        generator_seed=p8_episodes.DEFAULT_GENERATOR_SEED,
        label=regime,
        ss=ss,
        indices=range(p8_episodes.EPISODES_PER_CELL),
    )
    committed = [r for r in p8_adapter.read_list(p8_adapter.list_path(regime)) if r.ss == ss]
    drop = {"in_training_distribution"}  # P8 flags by pool cell; ours are pool rows only
    for d, r in zip(ours, committed, strict=True):
        ours_fields = {k: v for k, v in dataclasses.asdict(d).items() if k not in drop}
        assert ours_fields == {k: getattr(r, k) for k in ours_fields}


def test_content_hash_equals_p8_manifest() -> None:
    manifest = {r["file"]: r for r in p8_episodes.read_manifest(p8_adapter.manifest_path())}
    for name, path in p8_adapter.list_paths().items():
        rows = p8_adapter.read_list(path)
        assert ep.canonical_csv(rows) == p8_episodes.canonical_csv(rows)
        assert ep.content_sha256(rows) == manifest[f"{name}.parquet"]["content_sha256"]


def test_pools(pools: tuple[frozenset[ep.RealizationKey], ...]) -> None:
    train, tune = pools
    assert (len(train), len(tune)) == (729, 135)
    assert not train & tune
    assert {k[4] for k in train} == set(range(27))
    assert {k[4] for k in tune} == set(range(27, 32))


def test_adapter_wiring_matches_direct(api: ep.P8EpisodeApi, cfg: ep.SplitsConfig) -> None:
    missing = [n for n in ep.ADAPTER_NAMES_NEEDED if not hasattr(p8_adapter, n)]
    if missing:
        with pytest.raises(NotImplementedError):
            ep.p8_episode_api()
        pytest.skip(f"ila.p8_adapter does not re-export {missing} yet")
    assert p8_adapter.dev_pool is dev_pool
    assert p8_adapter.realization_key is dmf_realization_key
    assert p8_adapter.EpisodeDraw is p8_episodes.EpisodeDraw
    assert p8_adapter.resolve_draws is p8_episodes.resolve_draws
    wired = ep.p8_episode_api()
    assert ep.draw_select(wired, cfg) == ep.draw_select(api, cfg)


# --------------------------------------------------------------------------- select


def test_select_shape(select: list[p8_adapter.ListedEpisode]) -> None:
    assert len(select) == 150
    assert Counter(r.ss for r in select) == {"SS3": 50, "SS4": 50, "SS5": 50}
    assert [r.ss for r in select] == ["SS3"] * 50 + ["SS4"] * 50 + ["SS5"] * 50
    assert [r.index for r in select] == list(range(50)) * 3
    assert {r.regime for r in select} == {"select"}
    assert all(r.in_training_distribution for r in select)
    # Balanced round-robin: 45 tune realizations per SS, so every one is flown, 1 or 2 times.
    for ss in ("SS3", "SS4", "SS5"):
        per = Counter(r.key for r in select if r.ss == ss)
        assert len(per) == 45
        assert sorted(Counter(per.values()).items()) == [(1, 40), (2, 5)]


def test_select_is_tune_only(
    select: list[p8_adapter.ListedEpisode], pools: tuple[frozenset[ep.RealizationKey], ...]
) -> None:
    train, tune = pools
    keys = {r.key for r in select}
    assert keys <= tune
    assert not keys & train


@pytest.mark.parametrize("name", FROZEN)
def test_select_disjoint_from_frozen(
    select: list[p8_adapter.ListedEpisode],
    frozen: dict[str, frozenset[ep.RealizationKey]],
    name: str,
) -> None:
    assert frozen[name]
    assert not {r.key for r in select} & frozen[name]


def test_select_no_ss6_no_beam(select: list[p8_adapter.ListedEpisode]) -> None:
    assert not {r.ss for r in select} & {"SS6"}
    assert all(r.heading_deg != 90.0 for r in select)
    assert {r.heading_deg for r in select} == {45.0, 135.0, 180.0}
    assert {r.vessel for r in select} == {"frigate"}


def test_select_file_format_is_p8s(select: list[p8_adapter.ListedEpisode]) -> None:
    ours = pq.read_schema(SELECT)
    p8 = pq.read_schema(p8_adapter.list_path("id"))
    assert ours.equals(p8, check_metadata=True)
    assert len({(r.key, r.episode_seed) for r in select}) == 150


def test_select_manifest(select: list[p8_adapter.ListedEpisode]) -> None:
    (row,) = [r for r in ep.read_manifest(MANIFEST) if r["file"] == "select.parquet"]
    assert row["file_sha256"] == ep.file_sha256(SELECT)
    assert row["content_sha256"] == ep.content_sha256(select)
    assert (row["pool"], row["rows"], row["n_per_cell"]) == ("tune", "150", "50|50|50")
    assert (row["generator_seed"], row["draw"]) == ("20261023", ep.SELECT_DRAW_RULE)
    assert row["p8_commit"] == p8_adapter.p8_commit()


def test_select_draw_deterministic(api: ep.P8EpisodeApi, cfg: ep.SplitsConfig) -> None:
    first = ep.draw_select(api, cfg)
    assert first == ep.draw_select(api, cfg)
    assert [d.episode_seed for d in first] == [r.episode_seed for r in p8_adapter.read_list(SELECT)]


@pytest.mark.pybullet
def test_select_regenerates_byte_identical(
    api: ep.P8EpisodeApi, cfg: ep.SplitsConfig, tmp_path: Path
) -> None:
    """A fresh draw + 150 real resets reproduces the committed file and manifest row."""
    rows = ep.select_rows(api, cfg)
    row = ep.write_split(
        ep.SELECT_NAME,
        rows,
        tmp_path,
        pool=cfg.select_pool,
        generator_seed=cfg.generator_seed,
        draw=ep.SELECT_DRAW_RULE,
    )
    assert (tmp_path / "select.parquet").read_bytes() == SELECT.read_bytes()
    (fresh,) = ep.read_manifest(tmp_path / ep.MANIFEST_NAME)
    (committed,) = [r for r in ep.read_manifest(MANIFEST) if r["file"] == "select.parquet"]
    assert fresh == committed
    assert row.file_sha256 == ep.file_sha256(SELECT)
    assert rows == p8_adapter.read_list(SELECT)


# --------------------------------------------------------------------------- train


def test_train_is_train_only(
    api: ep.P8EpisodeApi,
    train_draws: list[ep.Draw],
    pools: tuple[frozenset[ep.RealizationKey], ...],
) -> None:
    train, tune = pools
    keys = {ep.row_key(api, d) for d in train_draws}
    assert keys == train  # 2000 > 729: every realization is flown
    assert not keys & tune


@pytest.mark.parametrize("name", FROZEN)
def test_train_disjoint_from_frozen(
    api: ep.P8EpisodeApi,
    train_draws: list[ep.Draw],
    frozen: dict[str, frozenset[ep.RealizationKey]],
    name: str,
) -> None:
    assert not {ep.row_key(api, d) for d in train_draws} & frozen[name]


def test_train_no_ss6_no_beam(train_draws: list[ep.Draw]) -> None:
    assert {d.ss for d in train_draws} == {"SS3", "SS4", "SS5"}
    assert all(d.heading_deg != 90.0 for d in train_draws)
    assert {d.vessel for d in train_draws} == {"frigate"}


def test_train_stratified_and_balanced(api: ep.P8EpisodeApi, train_draws: list[ep.Draw]) -> None:
    assert Counter(d.ss for d in train_draws) == {"SS3": 667, "SS4": 667, "SS5": 666}
    for n in (1, 2, 3, 100, 1000, 1001):
        counts = Counter(d.ss for d in train_draws[:n])
        per_ss = [counts.get(ss, 0) for ss in ("SS3", "SS4", "SS5")]
        assert max(per_ss) - min(per_ss) <= 1
    for ss in ("SS3", "SS4", "SS5"):
        per = Counter(ep.row_key(api, d) for d in train_draws if d.ss == ss)
        assert len(per) == 243
        assert max(per.values()) - min(per.values()) <= 1
    assert len({(ep.row_key(api, d), d.episode_seed) for d in train_draws}) == N_TRAIN


def test_train_prefix_and_slices(
    api: ep.P8EpisodeApi, cfg: ep.SplitsConfig, train_draws: list[ep.Draw]
) -> None:
    assert ep.draw_train(api, cfg, 1000) == train_draws[:1000]
    assert ep.draw_train(api, cfg, 300, start=700) == train_draws[700:1000]
    other = ep.draw_train(api, cfg, 30, stream="dagger1")
    assert {d.episode_seed for d in other}.isdisjoint(d.episode_seed for d in train_draws[:30])
    with pytest.raises(ValueError, match="reserved"):
        ep.draw_train(api, cfg, 1, stream=ep.SELECT_NAME)


@pytest.mark.pybullet
def test_train_resolves_and_writes(
    api: ep.P8EpisodeApi, cfg: ep.SplitsConfig, tmp_path: Path
) -> None:
    rows = ep.train_rows(api, cfg, 30)
    assert [(r.ss, r.index, r.episode_seed) for r in rows] == [
        (d.ss, d.index, d.episode_seed) for d in ep.draw_train(api, cfg, 30)
    ]
    name = ep.train_name(30)
    assert name == "train_0-29"
    row = ep.write_split(
        name,
        rows,
        tmp_path,
        pool=cfg.train_pool,
        generator_seed=cfg.generator_seed,
        draw=ep.TRAIN_DRAW_RULE,
    )
    assert (row.pool, row.n_per_cell) == ("train", "10|10|10")
    assert p8_adapter.read_list(tmp_path / f"{name}.parquet") == rows


# --------------------------------------------------------------------------- the guard refuses


def _first(path: Path, ss: str | None = None) -> p8_adapter.ListedEpisode:
    return next(r for r in p8_adapter.read_list(path) if ss is None or r.ss == ss)


@pytest.mark.parametrize("name", FROZEN)
def test_guard_refuses_frozen_rows(
    api: ep.P8EpisodeApi, select: list[p8_adapter.ListedEpisode], name: str
) -> None:
    leaked = dataclasses.replace(_first(p8_adapter.list_path(name)), in_training_distribution=True)
    with pytest.raises(ValueError, match="refused"):
        ep.check_rows(api, [*select, leaked], "tune")


def test_guard_refuses_wrong_pool_and_held_out(
    api: ep.P8EpisodeApi, select: list[p8_adapter.ListedEpisode]
) -> None:
    ep.check_rows(api, select, "tune")
    with pytest.raises(ValueError, match="outside the train pool"):
        ep.check_rows(api, select, "train")
    for change in ({"ss": "SS6"}, {"heading_deg": 90.0}):
        bad = dataclasses.replace(select[0], **change)
        with pytest.raises(ValueError, match="forbidden"):
            ep.check_rows(api, [bad, *select[1:]], "tune")
    with pytest.raises(ValueError, match="repeated"):
        ep.check_rows(api, [*select, select[0]], "tune")


# --------------------------------------------------------------------------- demo stream design

DEMO_FILES = ("demos_0-999", "demos_0-1999")


@pytest.fixture(scope="module")
def design() -> ep.StreamDesign:
    return ep.load_stream_design()


@pytest.fixture(scope="module")
def demos() -> dict[str, list[p8_adapter.ListedEpisode]]:
    return {name: ep.read_split(name) for name in DEMO_FILES}


def test_stream_design(design: ep.StreamDesign) -> None:
    assert design.demo_stream == "demos"
    assert dict(design.collection_sizes) == {
        "ppo_2000": 2000,
        "pid_1000": 1000,
        "mix_1000": 1000,
        "dart": 1000,
    }
    assert [d.name for d in ep.demo_lists(design)] == list(DEMO_FILES)
    assert design.dagger_indices(1) == range(1000, 1200)
    assert design.dagger_indices(5) == range(1800, 2000)
    assert design.dagger_split == "demos_0-1999"
    for bad in (0, 6):
        with pytest.raises(ValueError):
            design.dagger_indices(bad)


def test_demo_files_match_manifest_and_draw(
    api: ep.P8EpisodeApi,
    cfg: ep.SplitsConfig,
    demos: dict[str, list[p8_adapter.ListedEpisode]],
) -> None:
    long, short = demos["demos_0-1999"], demos["demos_0-999"]
    assert (len(short), len(long)) == (1000, 2000)
    assert short == long[:1000]  # bc's 1 000 are a prefix of bc_matched's 2 000
    draws = ep.draw_train(api, cfg, 2000, stream="demos")
    assert [(r.ss, r.index, r.episode_seed, r.realization_seed) for r in long] == [
        (d.ss, d.index, d.episode_seed, d.realization_seed) for d in draws
    ]
    assert {r.regime for r in long} == {"demos"}
    for name in DEMO_FILES:
        (row,) = [r for r in ep.read_manifest(MANIFEST) if r["file"] == f"{name}.parquet"]
        assert (row["pool"], row["generator_seed"], row["draw"]) == (
            "train",
            "20261023",
            ep.DEMO_DRAW_RULE,
        )
        assert pq.read_schema(ep.SPLITS_DIR / f"{name}.parquet").equals(
            pq.read_schema(p8_adapter.list_path("id")), check_metadata=True
        )


def test_demo_files_train_only(
    api: ep.P8EpisodeApi,
    demos: dict[str, list[p8_adapter.ListedEpisode]],
    pools: tuple[frozenset[ep.RealizationKey], ...],
    frozen: dict[str, frozenset[ep.RealizationKey]],
) -> None:
    train, tune = pools
    rows = demos["demos_0-1999"]
    keys = {r.key for r in rows}
    assert keys <= train and not keys & tune
    for name, frozen_set in frozen.items():
        assert not keys & frozen_set, name
    assert not {r.ss for r in rows} & {"SS6"}
    assert all(r.heading_deg != 90.0 for r in rows)
    assert len({(r.key, r.episode_seed) for r in rows}) == 2000
    ep.check_rows(api, rows, "train")


def test_read_split_refuses_tampered_file(tmp_path: Path) -> None:
    for name in ("select", "demos_0-999"):
        (tmp_path / f"{name}.parquet").write_bytes((ep.SPLITS_DIR / f"{name}.parquet").read_bytes())
    (tmp_path / ep.MANIFEST_NAME).write_text(MANIFEST.read_text(encoding="utf-8"))
    ep.read_split("demos_0-999", tmp_path)
    rows = p8_adapter.read_list(tmp_path / "demos_0-999.parquet")
    ep.write_split("demos_0-999", rows[1:], tmp_path / "other", pool="train",
                   generator_seed=20261023, draw=ep.DEMO_DRAW_RULE)  # fmt: skip
    (tmp_path / "demos_0-999.parquet").write_bytes(
        (tmp_path / "other" / "demos_0-999.parquet").read_bytes()
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ep.read_split("demos_0-999", tmp_path)
    with pytest.raises(ValueError, match="manifest rows"):
        ep.read_split("train_0-29", tmp_path)


def test_dagger_slices_tile_the_second_thousand(
    design: ep.StreamDesign, demos: dict[str, list[p8_adapter.ListedEpisode]]
) -> None:
    """DAgger iterations 1-5 fly demos 1000-1999 exactly (every seed), disjoint from 0-999."""
    long = demos[design.dagger_split]
    slices = [design.dagger_indices(i) for i in range(1, design.dagger_iterations + 1)]
    flat = [j for r in slices for j in r]
    assert flat == list(range(1000, 2000))  # tile exactly, in order, no gap or overlap
    assert all(len(r) == design.dagger_per_iteration == 200 for r in slices)
    dagger_rows = [long[j] for j in flat]
    assert dagger_rows == long[1000:2000]
    first = {(r.key, r.episode_seed) for r in demos["demos_0-999"]}
    assert not {(r.key, r.episode_seed) for r in dagger_rows} & first
    assert not set(flat) & set(range(1000))
    # bc_dagger's starting conditions (bc 1 000 + 5 x 200) are bc_matched's 2 000.
    assert demos["demos_0-999"] + dagger_rows == long


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  first_index: 1000\n", "  first_index: 800\n"),  # overlaps the bc 1 000
        ("  episodes_per_iteration: 200\n", "  episodes_per_iteration: 150\n"),  # gap
        ("  stream: demos\n", "  stream: dagger\n"),  # not the demo stream
    ],
)
def test_stream_design_refuses_bad_dagger_slices(tmp_path: Path, old: str, new: str) -> None:
    text = ep.DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "demos.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        ep.load_stream_design(path)


@pytest.mark.pybullet
@pytest.mark.slow
def test_demo_files_regenerate_byte_identical(cfg: ep.SplitsConfig, tmp_path: Path) -> None:
    """``make_splits``' demo path (2 000 real resets) reproduces both files and manifest rows."""
    api8 = ep.make_api(
        dev_pool=dev_pool,
        realization_key=dmf_realization_key,
        episode_draw=p8_episodes.EpisodeDraw,
        resolve_draws=p8_episodes.resolve_draws,
        cfgs=p8_adapter.load_eval_configs(),
        workers=8,
    )
    design = ep.load_stream_design()
    for name, rows in ep.demo_rows(api8, cfg, ep.demo_lists(design)).items():
        ep.write_split(name, rows, tmp_path, pool="train", generator_seed=cfg.generator_seed,
                       draw=ep.DEMO_DRAW_RULE)  # fmt: skip
        assert (tmp_path / f"{name}.parquet").read_bytes() == (
            ep.SPLITS_DIR / f"{name}.parquet"
        ).read_bytes()
    fresh = {r["file"]: r for r in ep.read_manifest(tmp_path / ep.MANIFEST_NAME)}
    committed = {r["file"]: r for r in ep.read_manifest(MANIFEST)}
    for name in DEMO_FILES:
        assert fresh[f"{name}.parquet"] == committed[f"{name}.parquet"]
