"""Phase 3: the demo store (plan §1.5) -- schema, alignment checks, the student read path.

P1-D2 obligations covered here:

* (3) poison test: the privileged fields and ``time_fraction`` of every stored ``teacher_obs``
  (selected by name), the whole ``priv`` block and ``t_model_s`` are set to NaN in the file, and
  the episode attributes a dataset could index by (``t0_model_s``, ``steps``, ``ss``,
  ``heading_deg``, ``speed_kn``, ``outcome``) are scrambled; every student batch built from the
  poisoned store must be finite and identical to the batch from the clean store.
* (4)/(6) exact student columns: the student read path returns exactly ``images`` and the
  variant's ``proprio``. ``default`` reads only ``images`` and ``proprio``; the two ablation
  variants read ``images`` and ``teacher_obs`` and select their fields by name inside
  ``ila/data``. An extra dataset under any name makes the store refuse the episode.

``STUDENT_BATCHERS`` is the list every student dataset/batch builder must join; both obligations
are parametrised over it. Every class defined in ``ila.train.dataset`` (Phase 4) must be registered
in ``STUDENT_DATASET_CLASSES``, except the one named in ``BC_STATE_DATASET_CLASS`` (``bc_state``
takes P8's 25-d observation by design and is covered by its own allowlist entry in
``tests/test_render_access.py``). ``test_every_train_dataset_class_is_registered`` enforces this;
it is an expected failure (xfail, strict=False, reason "Phase 4 must register ...") until the
module defines a class, and a hard failure from then on.

The windowed contract Phase 4's dataset must satisfy (each registered adapter returns, for one
episode, its samples stacked along a leading axis):

* sample keys are exactly ``{"images", "proprio", "target"}`` -- nothing else, under any name;
* for a sample anchored at step ``t`` with ``To`` observation steps, the window steps are
  ``ila.policies.dp.chunking.obs_window_steps(t, To)`` (``t-To+1 .. t``, indices < 0 repeat 0);
  ``images[i]`` is the store's ``images[window[i]]`` (uint8, pixels from the store only, crop
  applied afterwards), and ``proprio[i]`` equals ``ObsContract.proprio(teacher_obs[window[i]])``
  of the variant's contract -- i.e. the stored ``proprio`` row for the default contract;
* a dataset gets variant proprio (``minus_last_action``: 10 columns, ``plus_clock``: 14) ONLY
  through ``DemoStore.student_inputs(name, variant)``, which builds it inside ``ila/data`` by
  field name and returns images + that proprio. No vision-student dataset ever gets a
  ``privileged_observations`` (or any other store-audit) allowlist entry in
  ``tests/test_render_access.py``; only ``bc_state``'s dataset may, after leak-auditor review;
* ``target`` is the store's ``label`` at the chunk indices of il-conventions (``label[t]`` for
  BC; ``label[t-To+1 .. t-To+Tp]`` with edge repetition for Diffusion Policy);
* episode lengths and window indices come from the arrays themselves, never from ``meta()``;
  ``meta()`` may only select episodes (the success filter), outside the per-episode batcher.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from ila.data import store as st
from ila.eval.runner import render_cfg_sha256
from ila.obs.split import CLOCK_FIELD, NEVER_STUDENT, load_contract

CONTRACT = load_contract()
CAMERA = render_cfg_sha256("full", None)


def _meta(demo_index: int, steps: int, **change: Any) -> st.EpisodeMeta:
    base = st.EpisodeMeta(
        collection="test",
        stream="demos",
        demo_index=demo_index,
        realization_key=json.dumps(["SS4", 135.0, 6.0, "frigate", 3]),
        regime="demos",
        ss="SS4",
        index=demo_index // 3,
        pad="aft",
        vessel="frigate",
        heading_deg=135.0,
        speed_kn=6.0,
        realization_seed=3,
        episode_seed=1234 + demo_index,
        t0_model_s=10.0,
        teacher="T_ppo",
        episode_teacher="ppo",
        beta=1.0,
        dart_sigma=0.0,
        noise_seed=-1,
        dagger_iteration=0,
        student_ckpt_sha256="",
        outcome="success",
        termination_reason="dwell_complete",
        success=True,
        steps=steps,
        p8_commit="p8",
        ila_commit="ila",
        camera_cfg_sha256=CAMERA,
        clear_frac=np.zeros(steps, np.float32),
        clear_frac_max=0.0,
        clear_frac_mean=0.0,
        clear_frames=0,
    )
    return dataclasses.replace(base, **change)


def _set(obs: np.ndarray, name: str, value: Any) -> None:
    offset, size = CONTRACT.layout[name]
    obs[..., offset : offset + size] = value


def synthetic_record(demo_index: int = 0, steps: int = 7, seed: int = 0) -> st.EpisodeRecord:
    """An episode that satisfies every alignment rule (random but consistent)."""
    rng = np.random.default_rng(seed)
    label = rng.uniform(-1.2, 1.2, (steps, 3)).astype(np.float32)
    executed = np.clip(label, -1.0, 1.0)
    obs = rng.normal(size=(steps, CONTRACT.obs_dim)).astype(np.float32)
    last = np.zeros((steps, 3), np.float32)
    last[1:] = executed[:-1]
    _set(obs, "last_action", last)
    images = rng.integers(0, 255, (steps, *st.IMAGE_SHAPE), dtype=np.uint8)
    return st.EpisodeRecord.build(
        images=images,
        teacher_obs=obs,
        action_exec=executed,
        label=label,
        t_model_s=10.0 + np.arange(steps) / 30.0,
        meta=_meta(demo_index, steps),
    )


def file_attrs() -> dict[str, Any]:
    return {
        "collection": "test",
        "teacher": "T_ppo",
        "dart_sigma": 0.0,
        "beta": 1.0,
        "split_file": "demos_0-999.parquet",
        "split_file_sha256": "x",
        "generator_seed": 20261023,
        "p8_commit": "p8",
        "ila_commit": "ila",
        "camera_cfg_sha256": CAMERA,
        **st.contract_field_attrs(),
    }


def write_store(path: Path, records: list[st.EpisodeRecord]) -> Path:
    with st.DemoWriter(path, file_attrs(), expected_camera_sha256=CAMERA) as w:
        for r in records:
            w.write(r)
    return path


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return write_store(tmp_path / "s.h5", [synthetic_record(j, 5 + j, j) for j in range(3)])


# --------------------------------------------------------------------------- student batchers

#: Every way a student batch is built from a store: ``(store, episode, variant) -> batch``.
#: Phase 4's dataset classes join this list (``STUDENT_DATASET_CLASSES``).
Batcher = Callable[[st.DemoStore, str, str], dict[str, np.ndarray]]

#: Every observation-contract variant (default and the two labelled ablations).
VARIANTS = ("default", "minus_last_action", "plus_clock")


def _store_path_batch(store: st.DemoStore, name: str, variant: str) -> dict[str, np.ndarray]:
    inputs = store.student_inputs(name, variant)
    return {"images": inputs.images, "proprio": inputs.proprio, "target": store.labels(name)}


def _expected_proprio(obs: np.ndarray, variant: str) -> np.ndarray:
    """The variant's proprio built field by field from P8 names (independent of the store)."""
    fields = load_contract(variant=variant).student_proprio
    return np.concatenate([CONTRACT.audit_field(obs, f) for f in fields], axis=-1)


#: Phase 4: every dataset/batch class of ``ila.train.dataset`` -> an adapter ``Batcher`` building
#: one episode's samples with it (windowed contract in the module docstring). Empty in Phase 3.
STUDENT_DATASET_CLASSES: dict[str, Batcher] = {}

#: Phase 4: the name of the ``bc_state`` dataset class (privileged 25-d input by design), the
#: only class of ``ila.train.dataset`` exempt from ``STUDENT_DATASET_CLASSES``.
BC_STATE_DATASET_CLASS: str | None = None

STUDENT_BATCHERS: list[Batcher] = [_store_path_batch, *STUDENT_DATASET_CLASSES.values()]


def _train_dataset_classes() -> set[str]:
    """Names of the classes defined in ``ila.train.dataset`` (empty if absent or a stub)."""
    try:
        module = importlib.import_module("ila.train.dataset")
    except ImportError:
        return set()
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    }


_TRAIN_DATASET_CLASSES = _train_dataset_classes()


@pytest.mark.xfail(
    not _TRAIN_DATASET_CLASSES,
    reason=(
        "Phase 4 must register every dataset/batch class of ila.train.dataset in "
        "STUDENT_DATASET_CLASSES (tests/test_data_store.py); the module defines none yet"
    ),
    strict=False,
)
def test_every_train_dataset_class_is_registered() -> None:
    assert _TRAIN_DATASET_CLASSES, "ila.train.dataset defines no dataset class yet"
    expected = _TRAIN_DATASET_CLASSES - {BC_STATE_DATASET_CLASS}
    assert set(STUDENT_DATASET_CLASSES) == expected
    if BC_STATE_DATASET_CLASS is not None:
        assert BC_STATE_DATASET_CLASS in _TRAIN_DATASET_CLASSES


# --------------------------------------------------------------------------- schema


def test_schema_is_plan_1_5() -> None:
    assert st.STORE_KEYS == (
        "images",
        "proprio",
        "priv",
        "teacher_obs",
        "action_exec",
        "label",
        "t_model_s",
    )
    s = st.schema()
    assert s["images"] == (np.dtype(np.uint8), (96, 96, 3))
    assert s["proprio"] == (np.dtype(np.float32), (13,))
    assert s["priv"] == (np.dtype(np.float32), (11,))
    assert s["teacher_obs"] == (np.dtype(np.float32), (25,))
    assert s["action_exec"] == s["label"] == (np.dtype(np.float32), (3,))
    assert s["t_model_s"] == (np.dtype(np.float64), ())


def test_meta_has_the_plan_attrs() -> None:
    names = {f.name for f in dataclasses.fields(st.EpisodeMeta)}
    plan = {
        "realization_key",
        "episode_seed",
        "teacher",
        "beta",
        "dart_sigma",
        "outcome",
        "p8_commit",
        "camera_cfg_sha256",
        "ila_commit",
        "clear_frac",
        "clear_frac_max",
    }
    assert plan <= names


def test_round_trip(store_path: Path) -> None:
    recs = [synthetic_record(j, 5 + j, j) for j in range(3)]
    with st.DemoStore(store_path) as s:
        assert s.episodes() == ["ep_00000", "ep_00001", "ep_00002"]
        for r, name in zip(recs, s.episodes(), strict=True):
            arrays = s.audit_arrays(name)
            for k in st.STORE_KEYS:
                assert arrays[k].tobytes() == r.arrays[k].tobytes(), k
            m = s.meta(name)
            assert m.steps == r.meta.steps and m.outcome == "success"
            np.testing.assert_array_equal(m.clear_frac, r.meta.clear_frac)
            s.validate_episode(name)
        with h5py.File(store_path, "r") as f:
            ds = f["ep_00000"]["images"]
            assert ds.compression == "lzf" and ds.chunks == (1, 96, 96, 3)


def test_bytes_are_deterministic(tmp_path: Path) -> None:
    recs = [synthetic_record(j, 4, j) for j in range(2)]
    a = write_store(tmp_path / "a.h5", recs)
    b = write_store(tmp_path / "b.h5", recs)
    assert a.read_bytes() == b.read_bytes()


def test_proprio_and_priv_are_by_name() -> None:
    r = synthetic_record()
    proprio, priv, _ = CONTRACT.split(r.arrays["teacher_obs"])
    np.testing.assert_array_equal(r.arrays["proprio"], proprio)
    np.testing.assert_array_equal(r.arrays["priv"], priv)
    for i, name in enumerate(("attitude", "body_rates", "velocity", "last_action", "in_contact")):
        np.testing.assert_array_equal(
            r.arrays["proprio"][:, i * 3 : i * 3 + CONTRACT.layout[name][1]],
            CONTRACT.audit_field(r.arrays["teacher_obs"], name),
        )


# --------------------------------------------------------------------------- refusals


def test_validation_refuses_misalignment() -> None:
    r = synthetic_record()
    bad = dict(r.arrays)
    bad["action_exec"] = np.roll(r.arrays["action_exec"], 1, axis=0)  # off by one step
    with pytest.raises(st.StoreSchemaError, match="last_action"):
        st.validate(bad, r.meta)
    shifted = dict(r.arrays)
    shifted["t_model_s"] = r.arrays["t_model_s"] + 1.0 / 30.0
    with pytest.raises(st.StoreSchemaError, match="t_model_s"):
        st.validate(shifted, r.meta)
    with pytest.raises(st.StoreSchemaError, match="datasets"):
        st.validate({**r.arrays, "extra": r.arrays["label"]}, r.meta)
    noisy = dict(r.arrays)
    noisy["label"] = r.arrays["label"] + np.float32(0.01)
    with pytest.raises(st.StoreSchemaError, match="clip"):
        st.validate(noisy, r.meta)
    st.validate(noisy, dataclasses.replace(r.meta, dart_sigma=0.1))  # DART: label != exec


def test_writer_refuses_non_committed_camera(tmp_path: Path) -> None:
    shifted = render_cfg_sha256("full", "bg_hue")
    with pytest.raises(st.StoreSchemaError, match="committed look"):
        st.DemoWriter(
            tmp_path / "x.h5",
            {**file_attrs(), "camera_cfg_sha256": shifted},
            expected_camera_sha256=CAMERA,
        )
    r = synthetic_record()
    bad = dataclasses.replace(r, meta=dataclasses.replace(r.meta, camera_cfg_sha256=shifted))
    with (
        st.DemoWriter(tmp_path / "y.h5", file_attrs(), expected_camera_sha256=CAMERA) as w,
        pytest.raises(st.StoreSchemaError, match="non-committed"),
    ):
        w.write(bad)


def test_writer_refuses_overwrite(store_path: Path) -> None:
    with pytest.raises(FileExistsError):
        st.DemoWriter(store_path, file_attrs(), expected_camera_sha256=CAMERA)


# --------------------------------------------------------------------------- student path


def test_student_episode_fields_are_exactly_images_and_proprio() -> None:
    assert [f.name for f in dataclasses.fields(st.StudentEpisode)] == ["images", "proprio"]
    assert st.STUDENT_INPUT_KEYS == ("images", "proprio")


@pytest.mark.parametrize(
    ("variant", "datasets"),
    [
        ("default", {"images", "proprio"}),
        ("minus_last_action", {"images", "teacher_obs"}),
        ("plus_clock", {"images", "teacher_obs"}),
    ],
)
def test_student_path_reads_only_what_it_needs(
    store_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, datasets: set[str]
) -> None:
    read: list[str] = []
    original = st.DemoStore._group

    class Spy:
        def __init__(self, group: h5py.Group) -> None:
            self.group = group

        def __getitem__(self, key: str) -> Any:
            read.append(key)
            return self.group[key]

    monkeypatch.setattr(st.DemoStore, "_group", lambda self, n: Spy(original(self, n)))
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            out = s.student_inputs(name, variant)
            assert [f.name for f in dataclasses.fields(out)] == ["images", "proprio"]
    assert set(read) == datasets


def test_variant_contracts_are_default_plus_minus_one_field() -> None:
    default = CONTRACT.student_proprio
    minus = st.variant_contract("minus_last_action")
    plus = st.variant_contract("plus_clock")
    assert minus.student_proprio == tuple(f for f in default if f != "last_action")
    assert plus.student_proprio == (*default, CLOCK_FIELD)
    assert (minus.proprio_dim, CONTRACT.proprio_dim, plus.proprio_dim) == (10, 13, 14)
    assert st.variant_contract("default").student_proprio == default
    with pytest.raises(ValueError):
        st.variant_contract("plus_rel_position")


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_proprio_columns(store_path: Path, variant: str) -> None:
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            obs = s.privileged_observations(name)
            default = s.student_inputs(name).proprio
            stored = s.audit_arrays(name)["proprio"]
            assert default.tobytes() == stored.tobytes()  # default = stored, byte for byte
            p = s.student_inputs(name, variant).proprio
            np.testing.assert_array_equal(p, _expected_proprio(obs, variant))
            if variant == "plus_clock":
                assert p.shape[1] == default.shape[1] + 1
                assert p[:, :-1].tobytes() == default.tobytes()
                assert p[:, -1:].tobytes() == CONTRACT.audit_field(obs, CLOCK_FIELD).tobytes()
            elif variant == "minus_last_action":
                assert p.shape[1] == default.shape[1] - 3 == 10  # last_action is 3 wide
                assert p.tobytes() == _expected_proprio(obs, "minus_last_action").tobytes()
            else:
                assert p.tobytes() == stored.tobytes()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_exact_student_columns(store_path: Path, batcher: Batcher, variant: str) -> None:
    """Every student batch carries pixels, the variant's proprio columns and the label."""
    width = load_contract(variant=variant).proprio_dim
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            batch = batcher(s, name, variant)
            assert set(batch) == {"images", "proprio", "target"}
            arrays = s.audit_arrays(name)
            t_len = arrays["label"].shape[0]
            assert batch["images"].shape == (t_len, 96, 96, 3)
            assert batch["images"].dtype == np.uint8
            assert batch["proprio"].shape == (t_len, width)
            np.testing.assert_array_equal(batch["images"], arrays["images"])
            np.testing.assert_array_equal(
                batch["proprio"], _expected_proprio(arrays["teacher_obs"], variant)
            )
            np.testing.assert_array_equal(batch["target"], arrays["label"])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_extra_column_under_any_name_is_refused(
    store_path: Path, tmp_path: Path, batcher: Batcher, variant: str
) -> None:
    for extra in ("innocent", "proprio_extra", "rel_position"):
        path = tmp_path / f"{extra}.h5"
        path.write_bytes(store_path.read_bytes())
        with h5py.File(path, "r+") as f:
            f["ep_00001"].create_dataset(extra, data=f["ep_00001"]["priv"][()])
        with st.DemoStore(path) as s:
            batcher(s, "ep_00000", variant)
            with pytest.raises(st.StoreSchemaError, match="datasets"):
                batcher(s, "ep_00001", variant)


#: Episode attributes a dataset could be tempted to index or condition by, and the scrambled
#: values written over them in the poison test.
SCRAMBLED_ATTRS: dict[str, Any] = {
    "t0_model_s": -12345.5,
    "steps": 9999,
    "ss": "SS9",
    "heading_deg": 999.0,
    "speed_kn": -1.0,
    "outcome": "crash",
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_poisoned_store_gives_finite_student_batches(
    store_path: Path, batcher: Batcher, variant: str
) -> None:
    """P1-D2 obligation 3: poisoned store -> finite student batches, identical to clean ones.

    NaN the five privileged fields always and ``time_fraction`` for every variant except
    ``plus_clock`` (where it is the labelled input); NaN ``priv`` and ``t_model_s``; scramble
    the episode attributes.
    """
    poison = sorted(NEVER_STUDENT | (set() if variant == "plus_clock" else {CLOCK_FIELD}))
    assert set(CONTRACT.privileged) <= set(poison)
    with st.DemoStore(store_path) as s:
        clean = {name: batcher(s, name, variant) for name in s.episodes()}
    with h5py.File(store_path, "r+") as f:
        for name in f:
            g = f[name]
            obs = g["teacher_obs"][()]
            for field in poison:
                _set(obs, field, np.nan)
            g["teacher_obs"][...] = obs
            g["priv"][...] = np.nan
            g["t_model_s"][...] = np.nan
            for key, value in SCRAMBLED_ATTRS.items():
                assert key in g.attrs
                g.attrs[key] = value
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            obs = s.privileged_observations(name)
            assert all(np.isnan(CONTRACT.audit_field(obs, f)).all() for f in poison)
            m = s.meta(name)
            assert {k: getattr(m, k) for k in SCRAMBLED_ATTRS} == SCRAMBLED_ATTRS
            batch = batcher(s, name, variant)
            assert set(batch) == set(clean[name])
            for key, value in batch.items():
                if value.dtype.kind == "f":
                    assert np.isfinite(value).all(), key
                assert value.dtype == clean[name][key].dtype
                assert value.tobytes() == clean[name][key].tobytes(), key


def test_clear_fraction() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    assert st.clear_fraction(rgb) == 0.0
    rgb[0, :] = 255
    rgb[1, 0] = (255, 255, 254)  # not the clear colour
    assert st.clear_fraction(rgb) == 4 / 16
