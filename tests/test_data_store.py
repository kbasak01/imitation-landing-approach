"""Phase 3: the demo store (plan §1.5) -- schema, alignment checks, the student read path.

P1-D2 obligations covered here:

* (3) poison test: the privileged fields and ``time_fraction`` of every stored ``teacher_obs``
  (selected by name), the whole ``priv`` block and ``t_model_s`` are set to NaN in the file; every
  student batch built from the poisoned store must be finite.
* (4)/(6) exact student columns: the student read path returns exactly ``images`` and
  ``proprio``, reads no other dataset, and an extra dataset under any name makes the store
  refuse the episode.

``STUDENT_BATCHERS`` is the list every student dataset/batch builder must join (Phase 4's
dataset class included); both obligations are parametrised over it.
"""

from __future__ import annotations

import dataclasses
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

#: Every way a student batch is built from a store. Phase 4's dataset class joins this list.
Batcher = Callable[[st.DemoStore, str], dict[str, np.ndarray]]


def _store_path_batch(store: st.DemoStore, name: str) -> dict[str, np.ndarray]:
    inputs = store.student_inputs(name)
    return {"images": inputs.images, "proprio": inputs.proprio, "target": store.labels(name)}


STUDENT_BATCHERS: list[Batcher] = [_store_path_batch]


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
            CONTRACT.field(r.arrays["teacher_obs"], name),
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


def test_student_path_reads_only_images_and_proprio(
    store_path: Path, monkeypatch: pytest.MonkeyPatch
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
            s.student_inputs(name)
    assert set(read) == {"images", "proprio"}


@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_exact_student_columns(store_path: Path, batcher: Batcher) -> None:
    """Every student batch carries pixels, the 13 allowlisted proprio columns and the label."""
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            batch = batcher(s, name)
            assert set(batch) == {"images", "proprio", "target"}
            arrays = s.audit_arrays(name)
            t_len = arrays["label"].shape[0]
            assert batch["images"].shape == (t_len, 96, 96, 3)
            assert batch["proprio"].shape == (t_len, CONTRACT.proprio_dim)
            np.testing.assert_array_equal(batch["images"], arrays["images"])
            np.testing.assert_array_equal(batch["proprio"], CONTRACT.proprio(arrays["teacher_obs"]))
            np.testing.assert_array_equal(batch["target"], arrays["label"])


@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_extra_column_under_any_name_is_refused(
    store_path: Path, tmp_path: Path, batcher: Batcher
) -> None:
    for extra in ("innocent", "proprio_extra", "rel_position"):
        path = tmp_path / f"{extra}.h5"
        path.write_bytes(store_path.read_bytes())
        with h5py.File(path, "r+") as f:
            f["ep_00001"].create_dataset(extra, data=f["ep_00001"]["priv"][()])
        with st.DemoStore(path) as s:
            batcher(s, "ep_00000")
            with pytest.raises(st.StoreSchemaError, match="datasets"):
                batcher(s, "ep_00001")


@pytest.mark.parametrize("batcher", STUDENT_BATCHERS)
def test_poisoned_store_gives_finite_student_batches(store_path: Path, batcher: Batcher) -> None:
    """P1-D2 obligation 3: NaN every privileged field and the clock; student batches stay finite."""
    poison = sorted(NEVER_STUDENT | {CLOCK_FIELD})
    assert set(CONTRACT.privileged) | {CLOCK_FIELD} == set(poison)
    with h5py.File(store_path, "r+") as f:
        for name in f:
            g = f[name]
            obs = g["teacher_obs"][()]
            for field in poison:
                _set(obs, field, np.nan)
            g["teacher_obs"][...] = obs
            g["priv"][...] = np.nan
            g["t_model_s"][...] = np.nan
    with st.DemoStore(store_path) as s:
        for name in s.episodes():
            obs = s.privileged_observations(name)
            assert all(np.isnan(CONTRACT.field(obs, f)).all() for f in poison)
            batch = batcher(s, name)
            for key, value in batch.items():
                if value.dtype.kind == "f":
                    assert np.isfinite(value).all(), key


def test_clear_fraction() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    assert st.clear_fraction(rgb) == 0.0
    rgb[0, :] = 255
    rgb[1, 0] = (255, 255, 254)  # not the clear colour
    assert st.clear_fraction(rgb) == 4 / 16
