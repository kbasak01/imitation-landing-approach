"""The closed-loop clock intervention (``ila.train.clock_diagnostics``), on tiny cases.

* the interventions' values (P8's clip into [0, 1], the clamp, the held mean, ``none`` bit-exact);
* only the ``time_fraction`` column of the ``plus_clock`` proprio changes, every other input
  (the other 13 columns of both ``To`` rows and the images) bit-identical, by hand and in
  closed loop (recorders on both sides of the wrapper);
* the teacher path changes only the observation's ``time_fraction`` field;
* refusals: a non-``plus_clock`` contract, checkpoint or selection;
* ``none`` reproduces the selection's rows of a tiny ``plus_clock`` checkpoint end to end
  (the study raises otherwise), and a tampered checkpoint is refused;
* the T_ppo ``none`` path equals the bare teacher (workstation: P8's checkpoint).

Positions of fields are looked up by name from the contract here (tests are not scanned).
Units: ``time_fraction`` dimensionless (``t / 12 s`` model scale, lambda = 1/25).
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ila import p8_adapter
from ila.obs.split import CLOCK_FIELD, ObsContract, load_contract
from ila.train import clock_diagnostics as cd

EPISODE_LEN_S = 12.0


@pytest.fixture(scope="module")
def plus() -> ObsContract:
    return load_contract(variant="plus_clock")


def _proprio_clock_col(contract: ObsContract) -> int:
    start = 0
    for name in contract.student_proprio:
        if name == CLOCK_FIELD:
            return start
        start += contract.layout[name][1]
    raise AssertionError("no clock column")


def _window(contract: ObsContract, rng: np.random.Generator, clock: tuple[float, float]) -> Any:
    obs = rng.normal(size=(contract.n_obs_steps, contract.obs_dim)).astype(np.float32)
    off = contract.layout[CLOCK_FIELD][0]
    obs[:, off] = np.asarray(clock, dtype=np.float32)
    images = rng.integers(0, 256, size=(contract.n_obs_steps, 96, 96, 3), dtype=np.uint8)
    return contract.student_input(images, obs)


class _Recorder:
    """A student recording what it is given; acts like a gentle descent."""

    privileged = False
    needs_motion_feed = False

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner
        self.seen: list[tuple[NDArray[np.uint8], NDArray[np.float32]]] = []

    def reset(self, seed: int, context: Any = None) -> None:
        if self.inner is not None:
            self.inner.reset(seed, context)

    def act_vision(self, images: NDArray[np.uint8], proprio: NDArray[np.float32]) -> Any:
        self.seen.append((np.array(images, copy=True), np.array(proprio, copy=True)))
        if self.inner is not None:
            return self.inner.act_vision(images, proprio)
        return np.array([0.0, 0.0, -0.3], dtype=np.float32)

    def act(self, obs: Any) -> Any:
        raise TypeError(f"students take act_vision, not {type(obs).__name__}")


# --------------------------------------------------------------------------- interventions


def test_intervention_values() -> None:
    c = np.array([[0.0], [0.05], [0.5], [0.95], [1.0]], dtype=np.float32)
    none = cd.make_intervention("none", episode_len_s=EPISODE_LEN_S, const_mean=0.3)
    assert none.apply(c).tobytes() == c.tobytes()
    shift = cd.make_intervention("shift_s", episode_len_s=EPISODE_LEN_S, const_mean=0.3)
    assert shift.value == pytest.approx(1.0 / 12.0)
    np.testing.assert_allclose(shift.apply(c)[:, 0], [1 / 12, 0.05 + 1 / 12, 0.5 + 1 / 12, 1, 1])
    assert shift.apply(c).max() == np.float32(1.0)  # clipped like P8
    const = cd.make_intervention("const_mean", episode_len_s=EPISODE_LEN_S, const_mean=0.3)
    assert np.all(const.apply(c) == np.float32(0.3))
    clamp = cd.make_intervention("clamp", episode_len_s=EPISODE_LEN_S, const_mean=0.3)
    out = clamp.apply(c)
    assert out[:2].tobytes() == c[:2].tobytes()  # below 1.6 s: untouched, bit for bit
    assert np.all(out[2:] == np.float32(1.6 / 12.0))
    with pytest.raises(ValueError, match="unknown"):
        cd.make_intervention("rewind", episode_len_s=EPISODE_LEN_S, const_mean=0.3)
    with pytest.raises(ValueError):
        cd.ClockIntervention("const_mean", 1.5)


@pytest.mark.parametrize("mode", cd.MODES)
def test_only_the_clock_column_changes(plus: ObsContract, mode: str) -> None:
    rng = np.random.default_rng(3)
    si = _window(plus, rng, (0.25, 0.25 + 1 / 30 / EPISODE_LEN_S))
    iv = cd.make_intervention(mode, episode_len_s=EPISODE_LEN_S, const_mean=0.0842)
    rec = _Recorder()
    wrapped = cd.ClockInterventionStudent(rec, plus, iv)
    wrapped.reset(7)
    wrapped.act_vision(si.images, si.proprio)
    images, proprio = rec.seen[0]
    col = _proprio_clock_col(plus)
    assert images.tobytes() == si.images.tobytes()
    others = [j for j in range(plus.proprio_dim) if j != col]
    assert proprio[:, others].tobytes() == si.proprio[:, others].tobytes()
    assert proprio.dtype == np.float32 and proprio.shape == si.proprio.shape
    assert proprio[:, col].tobytes() == iv.apply(si.proprio[:, [col]])[:, 0].tobytes()
    if mode == "none":
        assert proprio.tobytes() == si.proprio.tobytes()
    else:
        assert not np.array_equal(proprio[:, col], si.proprio[:, col])
    # the StudentInput entry point goes through the same path; the input is not modified
    wrapped.act(si)
    assert rec.seen[1][1].tobytes() == proprio.tobytes()
    with pytest.raises(TypeError):
        wrapped.act(np.zeros(25, dtype=np.float32))


def test_proprio_clock_reads_the_named_column(plus: ObsContract) -> None:
    vec = np.arange(plus.proprio_dim, dtype=np.float32) / 100.0
    assert plus.proprio_clock(vec).tolist() == [vec[_proprio_clock_col(plus)]]
    with pytest.raises(ValueError, match="width"):
        plus.proprio_clock(np.zeros(13, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plus.with_proprio_clock(vec, lambda c: c + 2.0)
    with pytest.raises(ValueError, match="shape"):
        plus.with_proprio_clock(vec, lambda c: np.zeros(c.size + 2, dtype=np.float32))


@pytest.mark.parametrize("mode", cd.MODES)
def test_teacher_path_changes_only_obs_time_fraction(mode: str) -> None:
    contract = load_contract()
    obs = np.random.default_rng(5).normal(size=contract.obs_dim).astype(np.float32)
    off = contract.layout[CLOCK_FIELD][0]
    obs[off] = np.float32(0.4)
    original = obs.copy()
    seen: list[NDArray[np.float32]] = []

    class Teacher:
        def reset(self, seed: int, context: Any = None) -> None:
            pass

        def act(self, o: NDArray[np.float32]) -> NDArray[np.float32]:
            seen.append(np.array(o, copy=True))
            return np.zeros(3, dtype=np.float32)

    iv = cd.make_intervention(mode, episode_len_s=EPISODE_LEN_S, const_mean=0.0842)
    t = cd.ClockInterventionTeacher(Teacher(), contract, iv)
    t.reset(1)
    t.act(obs)
    got = seen[0]
    assert obs.tobytes() == original.tobytes()  # the env's observation is not touched
    others = [j for j in range(contract.obs_dim) if j != off]
    assert got[others].tobytes() == obs[others].tobytes()
    assert got.dtype == np.float32 and got.shape == obs.shape
    assert got[off] == iv.apply(obs[[off]])[0]
    assert (got[off] == obs[off]) == (mode == "none")


# --------------------------------------------------------------------------- refusals


def test_refuses_non_plus_clock_students(tmp_path: Path) -> None:
    default = load_contract()
    iv = cd.ClockIntervention("none")
    with pytest.raises(ValueError, match="plus_clock"):
        cd.ClockInterventionStudent(_Recorder(), default, iv)
    with pytest.raises(ValueError, match="plus_clock"):
        default.with_proprio_clock(np.zeros((2, 13), np.float32), iv.apply)
    with pytest.raises(ValueError, match="plus_clock"):
        load_contract(variant="minus_last_action").proprio_clock(np.zeros(10, np.float32))
    with pytest.raises(TypeError):
        cd.ClockInterventionStudent(object(), load_contract(variant="plus_clock"), iv)
    path = _tiny_checkpoint(tmp_path, "bc", "default")
    with pytest.raises(ValueError, match="plus_clock"):
        cd.build_student(str(path), iv, p8_adapter.load_eval_configs())
    run = tmp_path / "bc" / "s0"
    (run / "select").mkdir(parents=True)
    (run / "selection.json").write_text(json.dumps({"obs_variant": "default"}))
    with pytest.raises(ValueError, match="plus_clock"):
        cd.selected_run(run)


# --------------------------------------------------------------------------- closed loop

CLOCK_MEAN = 0.0842


def _quiet(message: str) -> None:
    del message


def _tiny_checkpoint(root: Path, arm: str, variant: str) -> Path:
    """A randomly initialised (seeded) checkpoint in the training loop's format."""
    import torch

    from ila.policies.registry import CHECKPOINT_FORMAT, REGISTRY, build_net, load_train_config

    entry = REGISTRY[arm]
    assert entry.obs_variant == variant
    net: Any = build_net(arm, 0)
    if variant == "plus_clock":
        mean = net.encoder.proprio_mean.clone()
        col = _proprio_clock_col(load_contract(variant=variant))
        mean[col] = CLOCK_MEAN
        net.encoder.proprio_mean.copy_(mean)
    state = net.state_dict()
    path = root / f"{arm}_tiny.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "arm": arm,
            "kind": entry.kind,
            "obs_variant": variant,
            "config": load_train_config(arm),
            "run_seed": 0,
            "global_step": 5000,
            "model": state,
            "ema": state,
        },
        path,
    )
    return path


def _fake_selected_run(root: Path, n: int) -> Path:
    """A ``plus_clock`` run dir whose selection flew a tiny checkpoint on ``select[:n]``."""
    from ila.data import episodes as data_episodes
    from ila.train.config import sha256_file
    from ila.train.select import fly_checkpoint, write_rows_csv

    run = root / "bc_plus_clock" / "s0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "select").mkdir()
    path = _tiny_checkpoint(run / "checkpoints", "bc_plus_clock", "plus_clock")
    sha = sha256_file(path)
    listed = data_episodes.read_split("select")[:n]
    rows = fly_checkpoint(
        path, listed, arm="bc_plus_clock", run_seed=0, sha256=sha, workers=1, chunk=1
    )
    write_rows_csv(run / "select" / "tiny_rows.csv", rows)
    split = data_episodes.SPLITS_DIR / "select.parquet"
    sel = {
        "arm": "bc_plus_clock",
        "obs_variant": "plus_clock",
        "run_seed": 0,
        "teacher": "T_ppo",
        "weights": "ema",
        "inference": None,
        "inference_path": "cpu_fp32_single_sample",
        "list": {"name": "select", "file_sha256": data_episodes.file_sha256(split)},
        "selected": {
            "candidate": {"file": str(path), "sha256": sha, "global_step": 5000},
            "rows_file": "tiny_rows.csv",
        },
    }
    (run / "selection.json").write_text(json.dumps(sel))
    return run


@pytest.mark.pybullet
@pytest.mark.slow
def test_none_reproduces_selection_rows_and_tamper_is_refused(tmp_path: Path) -> None:
    run = _fake_selected_run(tmp_path / "runs", 1)
    out = tmp_path / "out"
    rec = cd.run_study([run], out_dir=out, teacher=False, workers=1, limit=1, log=_quiet)
    (subject,) = rec["subjects"]
    assert subject["none_reproduces"]["identical"] is True
    assert subject["training_clock_mean"] == pytest.approx(CLOCK_MEAN)
    assert subject["interventions"]["const_mean"]["value_time_fraction"] == pytest.approx(
        CLOCK_MEAN
    )
    assert [r["mode"] for r in rec["summary"]] == list(cd.MODES)
    assert rec["scope"].startswith("diagnostic on select; chooses nothing; not scored")
    with (out / "clock_intervention.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert [r["mode"] for r in summary] == list(cd.MODES)
    assert summary[0]["success_minus_none"] == "0"
    with gzip.open(out / "clock_intervention_episodes.csv.gz", "rt") as g:
        eps = list(csv.DictReader(g))
    assert len(eps) == len(cd.MODES) and eps[0]["mode"] == "none"
    # a tampered "none" reference is caught
    rows_csv = run / "select" / "tiny_rows.csv"
    lines = rows_csv.read_text().splitlines()
    head = lines[0].split(",")
    cells = lines[1].split(",")
    cells[head.index("steps")] = str(int(cells[head.index("steps")]) + 1)
    rows_csv.write_text("\n".join([lines[0], ",".join(cells)]) + "\n")
    with pytest.raises(cd.ReproductionError):
        cd.run_study(
            [run], out_dir=out, teacher=False, modes=("none",), workers=1, limit=1,
            log=_quiet,
        )  # fmt: skip
    # a checkpoint whose bytes changed is refused before flying
    ckpt = next((run / "checkpoints").glob("*.pt"))
    ckpt.write_bytes(ckpt.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="SHA-256"):
        cd.selected_run(run)


@pytest.mark.pybullet
def test_closed_loop_inputs_differ_only_in_the_clock(tmp_path: Path, plus: ObsContract) -> None:
    """Recorders outside and inside the wrapper, one ``select`` episode, ``shift_s``."""
    from ila.data import episodes as data_episodes
    from ila.eval import runner
    from ila.train.select import checkpoint_policy

    path = _tiny_checkpoint(tmp_path, "bc_plus_clock", "plus_clock")
    cfgs = p8_adapter.load_eval_configs()
    inner = _Recorder(checkpoint_policy(str(path), cfgs))
    iv = cd.make_intervention("shift_s", episode_len_s=EPISODE_LEN_S, const_mean=CLOCK_MEAN)
    outer = _Recorder(cd.ClockInterventionStudent(inner, plus, iv))

    def build(cfgs: Any) -> Any:
        del cfgs
        return outer

    spec = p8_adapter.callable_spec("bc_plus_clock", build, run_seed=0)
    listed = data_episodes.read_split("select")[:1]
    rows = runner.run_list_vision(
        listed, spec, cfgs, render="full", workers=1, contract_variant="plus_clock"
    )
    assert len(outer.seen) == len(inner.seen) == int(rows[0]["steps"]) > 1
    col = _proprio_clock_col(plus)
    others = [j for j in range(plus.proprio_dim) if j != col]
    for (img_o, p_o), (img_i, p_i) in zip(outer.seen, inner.seen, strict=True):
        assert img_i.tobytes() == img_o.tobytes()
        assert p_i[:, others].tobytes() == p_o[:, others].tobytes()
        expected = np.clip(p_o[:, col].astype(np.float64) + 1 / 12, 0, 1).astype(np.float32)
        assert p_i[:, col].tobytes() == expected.tobytes()


@pytest.mark.pybullet
@pytest.mark.checkpoints
def test_teacher_none_equals_bare_teacher() -> None:
    from ila.data import episodes as data_episodes
    from ila.eval import runner

    listed = data_episodes.read_split("select")[:1]
    bare = cd.fly_teacher(None, listed, workers=1)
    none = cd.fly_teacher(cd.ClockIntervention("none"), listed, workers=1)
    assert cd.rows_mismatch(none, bare, runner.VISION_EPISODE_COLUMNS) == []
    shifted = cd.fly_teacher(
        cd.make_intervention("shift_s", episode_len_s=EPISODE_LEN_S, const_mean=CLOCK_MEAN),
        listed,
        workers=1,
    )
    assert shifted[0]["episode_seed"] == bare[0]["episode_seed"]


def test_diagnostic_reads_no_frozen_list_and_names_no_audit_accessor() -> None:
    import test_render_access as ra
    from ila.policies.registry import REPO_ROOT

    for path in (
        REPO_ROOT / "src" / "ila" / "train" / "clock_diagnostics.py",
        REPO_ROOT / "scripts" / "clock_intervention.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert ra._references(text, ra.FROZEN_LIST_NAMES) == []
        assert ra._references(text, ra.AUDIT_NAMES) == []
        assert ra._references(text, frozenset({"DemoStore", "store"})) == []
    assert "clock_intervention.py" in ra.TRAIN_SCRIPTS
