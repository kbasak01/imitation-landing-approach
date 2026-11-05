"""Phase 5: DAgger rollouts (plan §1.6, P3-D1 §2-§3, il-conventions alignment).

Without physics (CPU CI):

* the beta schedule comes from ``configs/data/demos.yaml``; the coin is reproducible from
  (run_seed, iteration, episode seed), draws once per step whatever beta is, and its empirical
  rate matches beta;
* the recorder (fake teacher and student) resets the teacher, the student and the window on
  every episode, queries the teacher on EVERY step (shadowing), executes the teacher's label
  exactly when the coin says so and the student's proposal otherwise (float32, clipped), and
  builds records that validate with DAgger metadata (``beta``, ``dagger_iteration``, student SHA);
* ``aggregate_manifest`` asserts the labelled-episode count (1 000 + 200 i; 2 000 after
  iteration 5) and refuses gaps, wrong beta and stores outside the seed's directory;
* resume: a finished iteration is reused for the same checkpoint SHA and refused for another;
* ``limit`` is refused for ``purpose="train"``.

With P8's PPO checkpoint and the ``bc`` s0 checkpoint (``checkpoints`` + ``slow``): two episodes of
iteration 1 into a temporary data root, audited clean (re-label, replay, attributes, sidecar),
bit-identical for 1 and 2 workers, with the 2-worker run forced to complete in reverse order.
Provenance (``ila_commit``, ``+dirty``) is pinned for the whole test: it is in the store's
attributes by design, and a commit landing between two runs changed the store SHA-256 once
(2026-11-04, ``make test`` during the Phase 7 commits; reproduced exactly).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.data import collect, dagger
from ila.data import episodes as ep
from ila.data.store import DemoStore, EpisodeRecord, StoreSchemaError, validate
from ila.eval import runner
from ila.obs.split import load_contract
from ila.render.camera import Frame
from ila.train.select import OUTCOME_ORDER

BC_S0 = p8_adapter.REPO_ROOT / "artifacts" / "runs" / "bc" / "s0" / "checkpoints"
BC_S0_FINAL = BC_S0 / "s0_step030000.pt"


# --------------------------------------------------------------------------- schedule and paths


def test_beta_schedule_and_paths(tmp_path: Path) -> None:
    assert [dagger.beta_for(i) for i in range(1, 6)] == [0.5, 0.25, 0.125, 0.0, 0.0]
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            dagger.beta_for(bad)
    assert dagger.rollout_dir(tmp_path, 3) == tmp_path / "dagger" / "bc_dagger" / "s3"
    assert dagger.rollout_store_path(tmp_path, 3, 2) == tmp_path / "dagger/bc_dagger/s3/iter2.h5"
    assert dagger.rollout_store_path(tmp_path, 0, 1, arm="x") == tmp_path / "dagger/x/s0/iter1.h5"
    assert dagger.ARM == "bc_dagger"


def test_iteration_rows_are_the_frozen_slices() -> None:
    design = ep.load_stream_design()
    seen: set[int] = set()
    for i in range(1, 6):
        split, pairs = dagger._iteration_rows(i)
        assert split == "demos_0-1999"
        idx = [j for j, _ in pairs]
        assert idx == list(design.dagger_indices(i)) == list(range(800 + 200 * i, 1000 + 200 * i))
        seen |= set(idx)
    assert seen == set(range(1000, 2000))


# --------------------------------------------------------------------------- the coin


def test_coin_is_reproducible_and_keyed() -> None:
    assert dagger.coin_entropy(2, 3, 77)[0::2] == [2, 3]
    assert dagger.coin_entropy(2, 3, 77)[3] == 77
    assert dagger.coin_entropy(2, 3, 77)[1] == ep.label_entropy("dagger_beta")

    def seq(
        beta: float, run_seed: int, it: int, seed: int, n: int = 50
    ) -> list[tuple[float, bool]]:
        coin = dagger.BetaCoin(beta, run_seed, it, seed)
        return [coin.draw() for _ in range(n)]

    a = seq(0.5, 0, 1, 12345)
    assert a == seq(0.5, 0, 1, 12345)
    for other in (seq(0.5, 1, 1, 12345), seq(0.5, 0, 2, 12345), seq(0.5, 0, 1, 12346)):
        assert [u for u, _ in other] != [u for u, _ in a]
    # One draw per step whatever beta: the u sequence does not depend on beta.
    for beta in (0.0, 0.125, 0.25, 1.0):
        s = seq(beta, 0, 1, 12345)
        assert [u for u, _ in s] == [u for u, _ in a]
        assert [b for _, b in s] == [u < beta for u, _ in a]
    assert not any(b for _, b in seq(0.0, 0, 4, 5, 2000))
    assert all(b for _, b in seq(1.0, 0, 4, 5, 2000))
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            dagger.BetaCoin(bad, 0, 1, 1)


@pytest.mark.parametrize("beta", [0.5, 0.25, 0.125])
def test_coin_empirical_rate(beta: float) -> None:
    hits = n = 0
    for seed in range(200):  # 200 episodes x 100 steps, the iteration's shape
        coin = dagger.BetaCoin(beta, 0, 1, seed)
        hits += sum(coin.draw()[1] for _ in range(100))
        n += 100
    sd = (beta * (1 - beta) / n) ** 0.5
    assert abs(hits / n - beta) < 4 * sd


# --------------------------------------------------------------------------- the recorder


class _FakeTeacher:
    """Stateful double: counts resets and calls; returns a float32 action per step."""

    def __init__(self) -> None:
        self.resets: list[int] = []
        self.calls = 0
        self.calls_since_reset = 0

    def reset(self, seed: int, context: Any = None) -> None:
        assert context is None
        self.resets.append(int(seed))
        self.calls_since_reset = 0

    def act(self, obs: Any) -> np.ndarray:
        del obs
        self.calls += 1
        self.calls_since_reset += 1
        k = self.calls_since_reset
        return np.array([0.1 * k, -0.05 * k, 0.3], dtype=np.float32)


class _FakeStudent:
    """Vision double: records what it is shown; proposes a different action."""

    def __init__(self) -> None:
        self.resets: list[int] = []
        self.inputs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.calls = 0

    def reset(self, seed: int, context: Any = None) -> None:
        assert context is None
        self.resets.append(int(seed))

    def act_vision(self, image_stack: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.inputs.append((image_stack.shape, proprio.shape))
        return np.array([-0.2, 0.123456789, 1.5], dtype=np.float32)  # z outside [-1, 1]


def _rows(n: int = 2) -> list[tuple[int, p8_adapter.ListedEpisode]]:
    _, pairs = dagger._iteration_rows(1)
    return pairs[:n]


def _recorder(
    beta: float, teacher: _FakeTeacher, student: _FakeStudent, run_seed: int = 0
) -> dagger.DaggerRecorder:
    pairs = _rows()
    return dagger.DaggerRecorder(
        dagger._collection(run_seed, 1, dagger.ARM, 200),
        teacher,  # type: ignore[arg-type]
        student,
        load_contract(),
        {(r.ss, int(r.index), int(r.episode_seed)): j for j, r in pairs},
        run_seed=run_seed,
        iteration=1,
        beta=beta,
        student_sha256="ab" * 32,
    )


def _fly(rec: dagger.DaggerRecorder, steps: tuple[int, ...]) -> list[list[np.ndarray]]:
    """Drive the recorder like the runner: obs[t+1] reports action_exec[t] as last_action."""
    contract = load_contract()
    off, size = contract.layout["last_action"]
    out: list[list[np.ndarray]] = []
    for (_, row), t_len in zip(_rows(), steps, strict=True):
        rec.begin(row, None)
        executed: list[np.ndarray] = []
        last = np.zeros(3, np.float64)
        for t in range(t_len):
            obs = np.full(contract.obs_dim, 0.01 * t, dtype=np.float32)
            obs[off : off + size] = last
            frame = Frame(np.full((96, 96, 3), t % 255, np.uint8))
            a = rec.act(t, obs, frame)
            executed.append(a)
            last = np.clip(a, -1.0, 1.0)
        out.append(executed)
    return out


def _fake_row(row: p8_adapter.ListedEpisode, steps: int) -> dict[str, Any]:
    return {
        "ss": row.ss,
        "index": int(row.index),
        "episode_seed": int(row.episode_seed),
        "steps": steps,
        "frames_rendered": steps + 1,
        "outcome": "timeout",
        "termination_reason": "test",
        "render_cfg_sha256": runner.render_cfg_sha256("full", None),
    }


def test_recorder_shadows_teacher_and_selects_by_coin() -> None:
    teacher, student = _FakeTeacher(), _FakeStudent()
    rec = _recorder(0.5, teacher, student)
    steps = (37, 23)
    executed = _fly(rec, steps)
    seeds = [int(r.episode_seed) for _, r in _rows()]
    # reset once per episode, teacher called on every step (also where the student executes).
    assert teacher.resets == seeds and student.resets == seeds
    assert teacher.calls == student.calls == sum(steps)
    # the student only ever saw (To, 96, 96, 3) frames and (To, 13) proprio.
    assert set(student.inputs) == {((2, 96, 96, 3), (2, 13))}
    pairs = rec.finished + ([] if rec._buf is None else [(rec._buf, rec._side)])
    assert len(pairs) == 2
    n_teacher = 0
    for (buf, side), seed, ex in zip(pairs, seeds, executed, strict=True):
        assert side is not None
        coin = dagger.BetaCoin(0.5, 0, 1, seed)
        for t, a in enumerate(ex):
            u, use_teacher = coin.draw()
            assert side.u[t] == u and side.teacher_exec[t] == use_teacher
            label = buf.label[t].astype(np.float64)
            proposed = side.proposed[t].astype(np.float64)
            want = collect.executable(label if use_teacher else proposed)
            assert np.array_equal(a, want) and a.dtype == np.float64
            assert np.array_equal(a.astype(np.float32).astype(np.float64), a)  # float32 exact
            assert np.all(np.abs(a) <= 1.0)
            assert np.array_equal(buf.executed[t], a.astype(np.float32))
            n_teacher += int(use_teacher)
        # the label is the teacher's own value, step by step from its reset.
        assert np.array_equal(buf.label[1], np.array([0.2, -0.1, 0.3], np.float32))
    assert 0 < n_teacher < sum(steps)


def test_recorder_records_validate_with_dagger_metadata() -> None:
    teacher, student = _FakeTeacher(), _FakeStudent()
    rec = _recorder(0.25, teacher, student)
    steps = (12, 9)
    _fly(rec, steps)
    rows = [_fake_row(r, t) for (_, r), t in zip(_rows(), steps, strict=True)]
    prov = {"p8_commit": "p8", "ila_commit": "ila"}
    records, sides = rec.finish(rows, prov)
    assert [r.meta.demo_index for r in records] == [1000, 1001]
    for r, s, t_len in zip(records, sides, steps, strict=True):
        m = r.meta
        assert (m.beta, m.dagger_iteration, m.student_ckpt_sha256) == (0.25, 1, "ab" * 32)
        assert (m.teacher, m.episode_teacher, m.dart_sigma, m.noise_seed) == (
            "T_ppo",
            "ppo",
            0.0,
            -1,
        )
        assert m.collection == "bc_dagger_s0_iter1" and m.stream == "demos"
        validate(r.arrays, m)  # accepts action_exec != label under DAgger metadata
        assert not np.array_equal(r.arrays["action_exec"], np.clip(r.arrays["label"], -1, 1))
        assert s["proposed"].dtype == np.float32 and s["proposed"].shape == (t_len, 3)
        assert s["u"].dtype == np.float64 and s["teacher_exec"].dtype == bool
    # The same arrays under teacher-demo metadata are refused (action_exec != clip(label)).
    import dataclasses

    demo_meta = dataclasses.replace(records[0].meta, beta=1.0, dagger_iteration=0)
    with pytest.raises(StoreSchemaError, match="noise-free"):
        validate(records[0].arrays, demo_meta)
    # And a broken last_action chain is refused under DAgger metadata too.
    arrays = dict(records[0].arrays)
    arrays["action_exec"] = arrays["action_exec"].copy()
    arrays["action_exec"][0, 0] += 0.5
    with pytest.raises(StoreSchemaError, match="last_action"):
        EpisodeRecord.build(
            images=arrays["images"],
            teacher_obs=arrays["teacher_obs"],
            action_exec=arrays["action_exec"],
            label=arrays["label"],
            t_model_s=arrays["t_model_s"],
            meta=records[0].meta,
        )


def test_beta_zero_executes_only_the_student_and_beta_one_only_the_teacher() -> None:
    for beta, want_teacher in ((0.0, False), (1.0, True)):
        teacher, student = _FakeTeacher(), _FakeStudent()
        rec = _recorder(beta, teacher, student)
        _fly(rec, (15, 4))
        assert teacher.calls == student.calls == 19  # both queried every step regardless
        for _, side in rec.finished:
            assert side.teacher_exec == [want_teacher] * len(side.teacher_exec)


def test_recorder_refusals() -> None:
    teacher, student = _FakeTeacher(), _FakeStudent()
    rec = _recorder(0.5, teacher, student)
    row = _rows()[0][1]
    frame = Frame(np.zeros((96, 96, 3), np.uint8))
    with pytest.raises(RuntimeError):
        rec.act(0, np.zeros(25, np.float32), frame)
    with pytest.raises(ValueError, match="context"):
        rec.begin(row, object())  # type: ignore[arg-type]
    rec.begin(row, None)
    with pytest.raises(ValueError, match="frames"):
        rec.act(0, np.zeros(25, np.float32), None)
    with pytest.raises(RuntimeError, match="step"):
        rec.act(1, np.zeros(25, np.float32), frame)
    with pytest.raises(TypeError):
        dagger.DaggerRecorder(
            dagger._collection(0, 1, dagger.ARM, 200),
            teacher,  # type: ignore[arg-type]
            object(),
            load_contract(),
            {},
            run_seed=0,
            iteration=1,
            beta=0.5,
            student_sha256="",
        )

    class _F64Teacher(_FakeTeacher):
        def act(self, obs: Any) -> np.ndarray:
            del obs
            return np.array([0.1, 0.2, 0.3], dtype=np.float64)  # not float32-representable

    bad = _recorder(0.5, _F64Teacher(), _FakeStudent())
    bad.begin(row, None)
    with pytest.raises(ValueError, match="float32"):
        bad.act(0, np.zeros(25, np.float32), frame)


# --------------------------------------------------------------------------- aggregation


def _result(root: Path, it: int, *, run_seed: int = 0, n: int = 200, **kw: Any) -> Any:
    d = dagger.rollout_dir(root, run_seed)
    base: dict[str, Any] = {
        "iteration": it,
        "beta": dagger.beta_for(it),
        "store": d / f"iter{it}.h5",
        "manifest": d / f"iter{it}.manifest.json",
        "sidecar": d / f"iter{it}.sidecar.npz",
        "store_sha256": f"{it:064d}",
        "student_ckpt": Path(f"ckpt{it}.pt"),
        "student_ckpt_sha256": f"{it + 10:064d}",
        "labelled_episodes": n,
        "labelled_frames": 150 * n,
        "outcomes": dict.fromkeys(OUTCOME_ORDER, 0) | {"success": n},
        "reused": False,
        "run_seed": run_seed,
        "purpose": "train",
    }
    base.update(kw)
    return dagger.RolloutResult(**base)


DEMOS = {"labelled_episodes": 1000, "kept_episodes": 1000, "train_frames": 61_500}


def test_aggregate_manifest_counts(tmp_path: Path) -> None:
    for i in range(1, 6):
        rolls = [_result(tmp_path, k) for k in range(1, i + 1)]
        agg = dagger.aggregate_manifest(
            run_seed=0, iteration=i, data_root=tmp_path, demos=DEMOS, rollouts=rolls
        )
        assert agg["labelled_episodes"] == 1000 + 200 * i
        assert agg["betas"] == [0.5, 0.25, 0.125, 0.0, 0.0][:i]
        assert agg["student_ckpt_sha256"] == [f"{k + 10:064d}" for k in range(1, i + 1)]
        assert agg["labelled_frames"]["total"] == 61_500 + 30_000 * i
        on_disk = json.loads(dagger.aggregate_path(tmp_path, 0, i).read_text())
        assert on_disk == json.loads(json.dumps(agg))
    assert agg["labelled_episodes"] == agg["labelled_episodes_total_final"] == 2000

    two = [_result(tmp_path, 1), _result(tmp_path, 2)]
    kw: dict[str, Any] = {"run_seed": 0, "data_root": tmp_path, "write": False}
    with pytest.raises(ValueError, match="labelled episodes"):
        dagger.aggregate_manifest(iteration=2, demos={"labelled_episodes": 999}, rollouts=two, **kw)
    with pytest.raises(ValueError, match="labelled episodes"):
        dagger.aggregate_manifest(
            iteration=2, demos=DEMOS, rollouts=[two[0], _result(tmp_path, 2, n=199)], **kw
        )
    with pytest.raises(ValueError, match="needs rollouts"):
        dagger.aggregate_manifest(iteration=2, demos=DEMOS, rollouts=[two[1]], **kw)
    with pytest.raises(ValueError, match="needs rollouts"):
        dagger.aggregate_manifest(iteration=1, demos=DEMOS, rollouts=two, **kw)
    with pytest.raises(ValueError, match="beta"):
        dagger.aggregate_manifest(
            iteration=2, demos=DEMOS, rollouts=[two[0], _result(tmp_path, 2, beta=0.5)], **kw
        )
    with pytest.raises(ValueError, match="seed 1|not under"):
        dagger.aggregate_manifest(
            iteration=2, demos=DEMOS, rollouts=[two[0], _result(tmp_path, 2, run_seed=1)], **kw
        )
    elsewhere = _result(tmp_path, 2, store=tmp_path / "elsewhere" / "iter2.h5")
    with pytest.raises(ValueError, match="not under"):
        dagger.aggregate_manifest(iteration=2, demos=DEMOS, rollouts=[two[0], elsewhere], **kw)
    with pytest.raises(ValueError, match="train"):
        dagger.aggregate_manifest(
            iteration=1, demos=DEMOS, rollouts=[_result(tmp_path, 1, purpose="smoke")], **kw
        )
    # Smoke/test aggregates of limit rollouts: totals still add up, nothing frozen asserted.
    small = dagger.aggregate_manifest(
        iteration=2,
        demos={"labelled_episodes": 8},
        rollouts=[_result(tmp_path, 1, n=2, purpose="test"), _result(tmp_path, 2, n=2)],
        strict=False,
        **kw,
    )
    assert small["labelled_episodes"] == 12 and small["strict"] is False


# --------------------------------------------------------------------------- resume


def _plant(root: Path, ckpt: Path, *, limit: int | None = 2, purpose: str = "test") -> Path:
    """Write a finished-looking iteration 1 (fake bytes) flown by ``ckpt``."""
    d = dagger.rollout_dir(root, 0)
    d.mkdir(parents=True)
    store = dagger.rollout_store_path(root, 0, 1)
    side = dagger.sidecar_path(root, 0, 1)
    store.write_bytes(b"store")
    dagger.write_npz(side, {"u": np.zeros(3)})
    man = {
        "arm": dagger.ARM,
        "run_seed": 0,
        "iteration": 1,
        "beta": 0.5,
        "purpose": purpose,
        "limit": limit,
        "complete": True,
        "store": {"path": str(store), "sha256": p8_adapter.sha256_file(store)},
        "sidecar": {"path": str(side), "sha256": p8_adapter.sha256_file(side)},
        "student": {"ckpt": str(ckpt), "sha256": p8_adapter.sha256_file(ckpt)},
        "labelled_episodes": 2,
        "labelled_frames": 100,
        "outcomes": dict.fromkeys(OUTCOME_ORDER, 0) | {"success": 2},
    }
    path = dagger.manifest_path(root, 0, 1)
    path.write_text(json.dumps(man), encoding="utf-8")
    return path


def test_resume_reuses_same_student_and_refuses_another(tmp_path: Path) -> None:
    ck_a, ck_b = tmp_path / "a.pt", tmp_path / "b.pt"
    ck_a.write_bytes(b"checkpoint A")
    ck_b.write_bytes(b"checkpoint B")
    root = tmp_path / "root"
    _plant(root, ck_a)
    kw: dict[str, Any] = {"run_seed": 0, "iteration": 1, "data_root": root, "log": lambda _: None}
    got = dagger.run_iteration(student_ckpt=ck_a, limit=2, purpose="test", **kw)
    assert got.reused and got.labelled_episodes == 2 and got.outcomes["success"] == 2
    assert got.student_ckpt_sha256 == p8_adapter.sha256_file(ck_a)
    assert dagger.load_rollout(root, 0, 1).store_sha256 == got.store_sha256
    with pytest.raises(dagger.RolloutExistsError, match="refusing to overwrite"):
        dagger.run_iteration(student_ckpt=ck_b, limit=2, purpose="test", **kw)
    with pytest.raises(dagger.RolloutExistsError, match="limit"):
        dagger.run_iteration(student_ckpt=ck_a, limit=3, purpose="test", **kw)
    # Tampered store: the manifest's SHA no longer matches.
    dagger.rollout_store_path(root, 0, 1).write_bytes(b"other")
    with pytest.raises(dagger.RolloutExistsError, match="SHA-256"):
        dagger.load_rollout(root, 0, 1)
    # Final files without a manifest (interrupted after the rename): refused, not overwritten.
    dagger.manifest_path(root, 0, 1).unlink()
    with pytest.raises(dagger.RolloutExistsError, match="without a manifest"):
        dagger.run_iteration(student_ckpt=ck_a, limit=2, purpose="test", **kw)
    with pytest.raises(FileNotFoundError):
        dagger.load_rollout(root, 0, 1)


def test_limit_refused_for_train_and_bad_purpose(tmp_path: Path) -> None:
    kw: dict[str, Any] = {
        "run_seed": 0,
        "iteration": 1,
        "student_ckpt": tmp_path / "missing.pt",  # refused before the file is touched
        "data_root": tmp_path,
    }
    with pytest.raises(ValueError, match="limit is for smoke and test"):
        dagger.run_iteration(limit=2, purpose="train", **kw)
    with pytest.raises(ValueError, match="purpose"):
        dagger.run_iteration(purpose="tuning", **kw)
    with pytest.raises(ValueError, match="iteration"):
        dagger.run_iteration(**{**kw, "iteration": 6})
    with pytest.raises(ValueError, match="limit must be"):
        dagger.run_iteration(limit=201, purpose="test", **kw)
    assert not (tmp_path / "dagger").exists()


def test_write_npz_is_deterministic_and_loadable(tmp_path: Path) -> None:
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "a": np.arange(5, dtype=np.int64),
        "b": np.ones((2, 3), np.float32),
    }
    p1, p2 = tmp_path / "1.npz", tmp_path / "2.npz"
    dagger.write_npz(p1, arrays)
    time.sleep(1.1)  # zip timestamps have 2 s resolution; numpy.savez would differ here
    dagger.write_npz(p2, arrays)
    assert p1.read_bytes() == p2.read_bytes()
    with np.load(p1) as f:
        assert set(f.files) == {"a", "b"} and np.array_equal(f["b"], arrays["b"])


# --------------------------------------------------------------------------- end to end

_ORDER_DIR_ENV = "ILA_TEST_DAGGER_ORDER_DIR"


def _first_chunk_finishes_last(task: dagger.DaggerTask) -> dagger.ChunkResult:
    """Pool entry point forcing reversed completion (runs in a spawned worker).

    Flies the chunk with the real :func:`ila.data.dagger.dagger_chunk` (the worker imports the
    unpatched module), then the chunk holding demo 1000 waits until demo 1001's chunk has
    finished, so the parent's ``imap`` gets its results in reverse completion order. Each chunk
    stamps ``<demo>.done`` with ``time.time_ns()``.
    """
    out = Path(os.environ[_ORDER_DIR_ENV])
    result = dagger.dagger_chunk(task)
    first = task.episodes[0][0]
    if first == 1000:
        deadline = time.monotonic() + 900.0
        while not (out / "1001.done").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("the chunk of demo 1001 never finished")
            time.sleep(0.05)
    (out / f"{first}.done").write_text(str(time.time_ns()), encoding="utf-8")
    return result


@pytest.mark.pybullet
@pytest.mark.checkpoints
@pytest.mark.slow
def test_two_episodes_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Iteration 1, two episodes, bc s0's final EMA: clean audit, worker-count invariant."""
    if not BC_S0_FINAL.is_file():
        pytest.skip(f"{BC_S0_FINAL} not present")
    # Store bytes carry ila_commit (+dirty) by design (collect.provenance, read once per
    # run_iteration). Pin one snapshot so that a commit or an edit in the checkout between r1
    # and r2 cannot change the SHA under test; only workers, chunk and order differ below.
    prov = collect.provenance()
    monkeypatch.setattr(collect, "provenance", lambda: prov)
    quiet: dict[str, Any] = {"log": lambda _: None}
    t0 = time.perf_counter()
    r1 = dagger.run_iteration(
        run_seed=0,
        iteration=1,
        student_ckpt=BC_S0_FINAL,
        data_root=tmp_path / "w1",
        workers=1,
        limit=2,
        purpose="test",
        **quiet,
    )
    wall = time.perf_counter() - t0
    print(f"\n2-episode DAgger iteration (1 worker): {wall:.1f} s")
    assert not r1.reused and r1.labelled_episodes == 2 and r1.beta == 0.5
    assert sum(r1.outcomes.values()) == 2 and list(r1.outcomes) == list(OUTCOME_ORDER)
    man = json.loads(r1.manifest.read_text())
    assert man["complete"] and man["student"]["sha256"] == p8_adapter.sha256_file(BC_S0_FINAL)
    assert man["episodes"]["first_index"] == 1000 and man["episodes"]["last_index"] == 1001
    assert list(man["outcomes"]) == list(OUTCOME_ORDER)
    assert 0 < man["teacher_executed_frames"] < man["labelled_frames"] == r1.labelled_frames

    audit = dagger.audit_iteration(tmp_path / "w1", 0, 1)
    assert audit["clean"], audit
    assert audit["relabel"]["identical"] == audit["relabel"]["n_steps"] == r1.labelled_frames
    assert audit["replay"]["passed"]

    with DemoStore(r1.store) as s:
        metas = [s.meta(e) for e in s.episodes()]
        assert s.episodes() == ["ep_01000", "ep_01001"]
        assert Counter(m.dagger_iteration for m in metas) == {1: 2}
        assert {m.beta for m in metas} == {0.5} and s.file_attrs["beta"] == 0.5

    again = dagger.run_iteration(
        run_seed=0,
        iteration=1,
        student_ckpt=BC_S0_FINAL,
        data_root=tmp_path / "w1",
        limit=2,
        purpose="test",
        **quiet,
    )
    assert again.reused and again.store_sha256 == r1.store_sha256

    # Two workers, one episode each, demo 1001's chunk forced to complete first.
    order_dir = tmp_path / "order"
    order_dir.mkdir()
    monkeypatch.setenv(_ORDER_DIR_ENV, str(order_dir))
    monkeypatch.setattr(dagger, "dagger_chunk", _first_chunk_finishes_last)
    r2 = dagger.run_iteration(
        run_seed=0,
        iteration=1,
        student_ckpt=BC_S0_FINAL,
        data_root=tmp_path / "w2",
        workers=2,
        chunk=1,
        limit=2,
        purpose="test",
        **quiet,
    )
    done = {j: int((order_dir / f"{j}.done").read_text()) for j in (1000, 1001)}
    assert done[1001] < done[1000], done  # the adverse order really happened
    m2 = json.loads(r2.manifest.read_text())
    assert m2["ila_commit"] == man["ila_commit"] == prov["ila_commit"]
    assert r2.store_sha256 == r1.store_sha256
    assert m2["sidecar"]["sha256"] == man["sidecar"]["sha256"]
    assert m2["rows_csv"]["sha256"] == man["rows_csv"]["sha256"]
