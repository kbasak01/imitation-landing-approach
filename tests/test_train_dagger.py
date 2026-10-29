"""Phase 5 ``bc_dagger`` training side: config, stages, data, resume, selection, diagnostics.

CPU tests on synthetic stores with a fake ``ila.data.dagger`` flight (``run_iteration``,
``load_rollout`` and ``audit_iteration`` replaced; the real ``aggregate_manifest`` runs). One
workstation test (``slow``, ``checkpoints``, ``data``) flies two real iterations of 4 rollouts
from the real ``bc`` s0 checkpoint on the CPU with a tiny step override.

Covers: ``bc_dagger.yaml`` parses as-is to ``bc_matched``'s :class:`IterationSchedule` (skeptic
B2's "one schedule") plus a DAgger block, and its config SHA-256s stay frozen; both arms get the
same stages; stage ``i`` flies the right checkpoint (iteration 1: the ``bc`` run's last step;
``i > 1``: this run's ``s<i-1>`` last step), audits before training (an unclean audit trains
nothing), aggregates demos (success-filtered) with rollouts ``1 .. i`` (unfiltered) and counts
``first_n + n i`` labelled episodes (1 000 + 200 i at full scale); resume mid-stage reuses the
rollouts and is bit-identical on the CPU; candidates are iteration 5's checkpoints by config,
not by arm name; the copycat Jacobian; the report's agreement and row reuse.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from ila.data import dagger
from ila.data.store import DemoStore, episode_group
from ila.policies.registry import REPO_ROOT, build_net, load_train_config, read_checkpoint
from ila.train import dagger_report, dagger_stage, diagnostics, loop
from ila.train import manifest as mf
from ila.train.config import DaggerConfig, IterationSchedule, TrainConfig, sha256_file
from ila.train.select import FINAL_ITERATION, candidates
from test_data_store import synthetic_record, write_store

CONFIGS = REPO_ROOT / "configs" / "train"


def _load(arm: str) -> TrainConfig:
    return TrainConfig.load(CONFIGS / f"{arm}.yaml")


# --------------------------------------------------------------------------- config


def test_bc_dagger_parses_to_bc_matched_schedule() -> None:
    dagger_cfg, matched = _load("bc_dagger"), _load("bc_matched")
    assert (
        dagger_cfg.iterations
        == matched.iterations
        == IterationSchedule(5, 200, 1000, 10_000, 500, 2000)
    )
    assert dagger_cfg.dagger == DaggerConfig(
        beta=(0.5, 0.25, 0.125, 0.0, 0.0),
        stream="demos",
        demos_success_filter=True,
        rollouts_success_filter=False,
        rollout_weights="ema_last_step_of_previous_iteration",
        beta_coin="per_episode_generator(run_seed, iteration, episode_seed)",
        executed_action="float32_before_step",
        inference_path="cpu_fp32_single_sample",
        source="configs/data/demos.yaml#dagger",
    )
    assert matched.dagger is None
    # Iteration 0 is bc's: same collection, first_n, filter, steps, optimiser, batch, EMA.
    for f in ("kind", "collection", "first_n", "success_filter", "steps", "batch_size",
              "precision", "ema", "optim", "checkpoint_every", "seeds"):  # fmt: skip
        assert getattr(dagger_cfg, f) == getattr(matched, f) == getattr(_load("bc"), f), f
    assert [dagger.beta_for(i) for i in range(1, 6)] == list(dagger_cfg.dagger.beta)
    # The frozen-SHA check covers the named file, its base and the referenced demos.yaml.
    frozen = mf.check_config_frozen(dagger_cfg.files_sha256)
    assert frozen["all_frozen"]
    assert [f["path"] for f in frozen["files"]] == [
        "configs/train/bc_dagger.yaml",
        "configs/train/bc.yaml",
        "configs/data/demos.yaml",
    ]
    assert dagger_cfg.sha256.startswith("dbc7c0d2") and dagger_cfg.sha256.endswith("9da7")
    assert mf.check_config_frozen(matched.files_sha256)["all_frozen"]


def _tmp_configs(tmp_path: Path, change: dict[str, Any]) -> Path:
    """Copy bc.yaml, bc_dagger.yaml (with ``change``) and demos.yaml into a tmp repo layout."""
    (tmp_path / "configs" / "train").mkdir(parents=True)
    (tmp_path / "configs" / "data").mkdir(parents=True)
    shutil.copy(CONFIGS / "bc.yaml", tmp_path / "configs" / "train" / "bc.yaml")
    shutil.copy(REPO_ROOT / "configs" / "data" / "demos.yaml", tmp_path / "configs" / "data")
    raw = yaml.safe_load((CONFIGS / "bc_dagger.yaml").read_text())
    raw.update(change)
    path = tmp_path / "configs" / "train" / "bc_dagger.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"success_filter": {"demos": True, "rollouts": True}}, "never filtered"),
        ({"success_filter": True}, "success_filter"),
        ({"labelled_episodes_total": 2200}, "labelled episodes"),
        ({"iteration_0": "fresh"}, "iteration 0"),
    ],
)
def test_bc_dagger_config_refuses_off_protocol_values(
    tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    path = _tmp_configs(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        TrainConfig.load(path)


def test_bc_dagger_config_refuses_a_loop_that_disagrees_with_demos_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIGS / "bc_dagger.yaml").read_text())
    loop_cfg = raw["loop"]
    loop_cfg["episodes"] = {**loop_cfg["episodes"], "per_iteration": 100}
    with pytest.raises(ValueError, match=r"loop.episodes"):
        TrainConfig.load(_tmp_configs(tmp_path, {"loop": loop_cfg}))


def test_bc_dagger_is_registered_with_the_bc_architecture() -> None:
    cfg = load_train_config("bc_dagger")
    bc = load_train_config("bc")
    for key in ("encoder", "proprio_mlp", "augment", "head", "loss"):
        assert cfg[key] == bc[key], key
    a, b = build_net("bc_dagger", 0).state_dict(), build_net("bc", 0).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_both_iterated_arms_get_the_same_stages(tmp_path: Path) -> None:
    out = {}
    for arm in ("bc_matched", "bc_dagger"):
        spec = loop.RunSpec(_load(arm), 0, tmp_path / arm, tmp_path, init_from=tmp_path / "bc.pt")
        stages = loop.build_stages(spec)
        out[arm] = [(s.index, s.steps, s.warmup, s.checkpoint_every) for s in stages]
        assert 30_000 + sum(s.steps for s in stages) == 80_000  # P3-D1 §3
    assert out["bc_matched"] == out["bc_dagger"] == [(i, 10_000, 500, 5000) for i in range(1, 6)]
    spec = loop.RunSpec(_load("bc_dagger"), 0, tmp_path, tmp_path, init_from=tmp_path / "bc.pt")
    data = loop.build_stages(spec)[2].data
    assert data.func is dagger_stage.stage_data  # type: ignore[attr-defined]
    assert data.args[1:] == (3, 10_000)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="timing"):
        loop.RunSpec(_load("bc_dagger"), 0, tmp_path, tmp_path, purpose="timing", max_steps=10)
    with pytest.raises(ValueError, match="smoke and test"):
        loop.RunSpec(_load("bc_dagger"), 0, tmp_path, tmp_path, rollout_limit=4)


# --------------------------------------------------------------------------- fake flight

N_DEMOS = 8
BOUNCE = 2  # demo stream index of the one non-success demo


def _record(demo_index: int, steps: int, seed: int, outcome: str = "success") -> Any:
    r = synthetic_record(demo_index, steps, seed)
    meta = dataclasses.replace(r.meta, outcome=outcome, success=outcome == "success")
    return dataclasses.replace(r, meta=meta)


def _write_demos(root: Path) -> Path:
    records = [
        _record(j, 5 + j % 4, 100 + j, "bounce" if j == BOUNCE else "success")
        for j in range(N_DEMOS)
    ]
    write_store(root / "demos" / "ppo_2000.h5", records)
    return root


class FakeDagger:
    """``ila.data.dagger``'s flight, faked: synthetic rollout stores, idempotent by SHA.

    Iteration ``i`` writes episodes ``start(i) .. start(i) + n - 1`` (the first one a crash,
    which the aggregate must keep) and returns a real :class:`dagger.RolloutResult`.
    """

    def __init__(self, schedule: IterationSchedule, steps: int = 4, one_frame: bool = False):
        """Fake flights of ``schedule``'s iterations, ``steps`` frames per episode."""
        self.schedule = schedule
        self.steps = steps
        self.one_frame = one_frame
        self.results: dict[tuple[str, int, int], dagger.RolloutResult] = {}
        self.calls: list[tuple[int, str, bool]] = []
        self.clean = True
        self.audits: list[int] = []

    def run_iteration(
        self,
        *,
        run_seed: int,
        iteration: int,
        student_ckpt: Path,
        data_root: Path,
        limit: int | None = None,
        purpose: str = "train",
        **_: Any,
    ) -> dagger.RolloutResult:
        """Write (or reuse, same student SHA) iteration ``iteration``'s fake rollout store."""
        sha = sha256_file(student_ckpt)
        key = (str(data_root), run_seed, iteration)
        if key in self.results:
            done = self.results[key]
            if done.student_ckpt_sha256 != sha:
                raise dagger.RolloutExistsError("flown by another student")
            self.calls.append((iteration, Path(student_ckpt).name, True))
            return dataclasses.replace(done, reused=True)
        n = self.schedule.add_per_iteration if limit is None else limit
        start = self.schedule.added_range(iteration).start
        store = dagger.rollout_store_path(data_root, run_seed, iteration)
        store.parent.mkdir(parents=True, exist_ok=True)
        steps = 1 if self.one_frame else self.steps
        records = [
            _record(
                start + k,
                steps + k % 2 * (not self.one_frame),
                7 * iteration + k,
                "crash" if k == 0 else "success",
            )  # fmt: skip
            for k in range(n)
        ]
        write_store(store, records)
        outcomes = dict.fromkeys(("crash", "off_pad", "hard_landing", "bounce", "success",
                                  "timeout"), 0)  # fmt: skip
        outcomes.update({"crash": 1, "success": n - 1})
        result = dagger.RolloutResult(
            iteration=iteration,
            beta=dagger.beta_for(iteration),
            store=store,
            manifest=store.with_suffix(".manifest.json"),
            sidecar=store.with_suffix(".sidecar.npz"),
            store_sha256=sha256_file(store),
            student_ckpt=Path(student_ckpt),
            student_ckpt_sha256=sha,
            labelled_episodes=n,
            labelled_frames=sum(r.arrays["label"].shape[0] for r in records),
            outcomes=outcomes,
            reused=False,
            run_seed=run_seed,
            purpose=purpose,
            limit=limit,
        )
        self.results[key] = result
        self.calls.append((iteration, Path(student_ckpt).name, False))
        return result

    def load_rollout(
        self, data_root: Path, run_seed: int, iteration: int, **_: Any
    ) -> dagger.RolloutResult:
        """Read back a flown iteration."""
        return dataclasses.replace(self.results[(str(data_root), run_seed, iteration)], reused=True)

    def audit_iteration(self, *args: Any, n_steps: int = 1000, **_: Any) -> dict[str, Any]:
        """Return ``clean`` as set; record the iteration audited."""
        iteration = int(args[2])
        assert n_steps == dagger_stage.AUDIT_STEPS
        self.audits.append(iteration)
        return {"clean": self.clean, "iteration": iteration}


SCHEDULE = IterationSchedule(n=2, add_per_iteration=2, first_added=3, steps_per_iteration=3,
                             warmup_steps=1, labelled_episodes_total=7)  # fmt: skip


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDagger]:
    f = FakeDagger(SCHEDULE)
    for name in ("run_iteration", "load_rollout", "audit_iteration"):
        monkeypatch.setattr(dagger, name, getattr(f, name))
    yield f


def _bc_run(root: Path, data_root: Path) -> Path:
    cfg = dataclasses.replace(_load("bc"), batch_size=4, precision="fp32")
    spec = loop.RunSpec(cfg, 3, root, data_root, device="cpu", purpose="test",
                        first_n_override=3, steps_override=2, checkpoint_every_override=2,
                        heldout=None, log_every=1, prefetch=0, hash_store=False)  # fmt: skip
    loop.train_run(spec, loop.build_stages(spec))
    return root / "checkpoints" / "s0_step000002.pt"


def _dagger_spec(run_dir: Path, data_root: Path, init_from: Path, **kw: Any) -> loop.RunSpec:
    cfg = dataclasses.replace(
        _load("bc_dagger"), batch_size=4, precision="fp32", iterations=SCHEDULE, first_n=3,
        checkpoint_every=3,
    )  # fmt: skip
    base: dict[str, Any] = {
        "cfg": cfg,
        "run_seed": 3,
        "run_dir": run_dir,
        "data_root": data_root,
        "device": "cpu",
        "purpose": "test",
        "init_from": init_from,
        "heldout": loop.HeldoutSpec("ppo_2000", 4, 2),
        "log_every": 1,
        "eval_every": 1000,
        "prefetch": 0,
        "hash_store": False,
        "rollout_workers": 2,
    }
    base.update(kw)
    return loop.RunSpec(**base)


def test_stage_data_flies_the_right_student_and_aggregates(
    fake: FakeDagger, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    data_root = _write_demos(tmp_path / "data")
    bc_final = _bc_run(tmp_path / "bc", data_root)
    spec = _dagger_spec(tmp_path / "dagger", data_root, bc_final)
    result = loop.train_run(spec, loop.build_stages(spec))
    assert result.state == "done" and result.global_step == 6
    # (a) iteration 1 flies the bc run's last step; iteration 2 this run's s1 last step.
    assert fake.calls == [(1, "s0_step000002.pt", False), (2, "s1_step000003.pt", False)]
    assert fake.audits == [1, 2]
    man = mf.read_json(tmp_path / "dagger" / "manifest.json")
    rec2 = man["data"]["2"]["dagger"]
    assert rec2["rollout_student"]["sha256"] == sha256_file(
        tmp_path / "dagger" / "checkpoints" / "s1_step000003.pt"
    )
    assert man["data"]["1"]["dagger"]["rollout_student"]["sha256"] == sha256_file(bc_final)
    # (c) labelled first_n + n i; demos filtered (bounce dropped), rollouts not (crash kept).
    for i in (1, 2):
        d = man["data"][str(i)]
        assert d["labelled_episodes_total"] == 3 + 2 * i == d["dagger"]["labelled_episodes"]
        assert d["kept_episodes"] == 2 + 2 * i
        demo, *rolls = d["choices"]
        assert demo["success_filter"] is True and demo["kept"] == 2
        assert [r["success_filter"] for r in rolls] == [False] * i
        assert [r["outcomes"].get("crash") for r in rolls] == [1] * i
        assert [r["start"] for r in rolls] == [3, 5][:i]
        assert d["dagger"]["aggregate"]["labelled_episodes"] == 3 + 2 * i
        assert d["heldout_overlap_episodes"] == 0
        # Held-out = demo indices 4, 5; iteration 1 flies 3, 4 and iteration 2 flies 5, 6.
        assert d["dagger"]["heldout_shared_starting_conditions"] == i
    frames = man["data"]["2"]["train_frames"]
    demo_frames = sum(5 + j % 4 for j in (0, 1))
    roll_frames = sum(r.labelled_frames for r in fake.results.values())
    assert frames == demo_frames + roll_frames == rec2["aggregate_frames"]
    # (e) the record is in the manifest and next to it.
    on_disk = mf.read_json(dagger_stage.record_path(tmp_path / "dagger", 2))
    assert on_disk["aggregate"] == rec2["aggregate"] and on_disk["audit"]["clean"] is True
    assert dagger.aggregate_path(data_root, 3, 2).is_file()  # the real aggregate_manifest wrote it
    assert "normaliser" not in man  # inherited from the bc run
    assert (tmp_path / "dagger" / "ema_last.pt").is_file()


def test_an_unclean_audit_trains_nothing(fake: FakeDagger, tmp_path: Path) -> None:
    data_root = _write_demos(tmp_path / "data")
    bc_final = _bc_run(tmp_path / "bc", data_root)
    fake.clean = False
    spec = _dagger_spec(tmp_path / "dagger", data_root, bc_final)
    with pytest.raises(dagger_stage.DaggerAuditError, match="not clean"):
        loop.train_run(spec, loop.build_stages(spec))
    assert not list((tmp_path / "dagger" / "checkpoints").glob("*.pt"))


def test_resume_mid_stage_reuses_rollouts_and_is_bit_identical(
    fake: FakeDagger, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    finals = {}
    for label in ("straight", "resumed"):
        data_root = _write_demos(tmp_path / label / "data")
        bc_final = _bc_run(tmp_path / label / "bc", data_root)
        spec = _dagger_spec(tmp_path / label / "run", data_root, bc_final)
        spec = dataclasses.replace(spec, cfg=dataclasses.replace(spec.cfg, checkpoint_every=1))
        if label == "resumed":
            first = loop.train_run(spec, loop.build_stages(spec), stop_after=5)  # stage 2, step 2
            assert first.state == "stopped"
        result = loop.train_run(spec, loop.build_stages(spec))
        assert result.state == "done"
        finals[label] = read_checkpoint(
            tmp_path / label / "run" / "checkpoints" / "s2_step000003.pt"
        )
    resumed_calls = [c for c in fake.calls if c[2]]
    assert resumed_calls == [(2, "s1_step000003.pt", True)]  # iteration 2 reused, not re-flown
    for part in ("model", "ema"):
        a, b = finals["straight"][part], finals["resumed"][part]
        assert all(torch.equal(a[k], b[k]) for k in a), part


def test_rollout_student_refuses_a_checkpoint_the_manifest_did_not_record(
    fake: FakeDagger, tmp_path: Path
) -> None:
    data_root = _write_demos(tmp_path / "data")
    bc_final = _bc_run(tmp_path / "bc", data_root)
    spec = _dagger_spec(tmp_path / "run", data_root, bc_final)
    loop.train_run(spec, loop.build_stages(spec))
    path, sha = dagger_stage.rollout_student(spec, 2, 3)
    assert path.name == "s1_step000003.pt" and sha == sha256_file(path)
    assert dagger_stage.rollout_student(spec, 1, 3) == (bc_final, sha256_file(bc_final))
    with pytest.raises(FileNotFoundError):
        dagger_stage.rollout_student(spec, 2, 4)  # no s1_step000004.pt
    torch.save({"tampered": True}, path)
    with pytest.raises(RuntimeError, match="recorded"):
        dagger_stage.rollout_student(spec, 2, 3)


@pytest.mark.slow
def test_full_scale_counts_are_1000_plus_200_i(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stage data at the frozen scale: 1 000 demos + 200 per iteration (1-frame episodes)."""
    cfg = _load("bc_dagger")
    assert cfg.iterations is not None
    fake = FakeDagger(cfg.iterations, one_frame=True)
    for name in ("run_iteration", "load_rollout", "audit_iteration"):
        monkeypatch.setattr(dagger, name, getattr(fake, name))
    r0 = synthetic_record(0, 1, 0)
    demos = [
        dataclasses.replace(
            r0, meta=dataclasses.replace(r0.meta, demo_index=j, episode_seed=1234 + j)
        )  # fmt: skip
        for j in range(1000)
    ]
    data_root = tmp_path / "data"
    write_store(data_root / "demos" / "ppo_2000.h5", demos)
    student = tmp_path / "student.pt"
    student.write_bytes(b"ema")
    monkeypatch.setattr(dagger_stage, "rollout_student", lambda *_: (student, sha256_file(student)))
    spec = loop.RunSpec(cfg, 0, tmp_path / "run", data_root, purpose="test", heldout=None,
                        hash_store=False, init_from=student, device="cpu")  # fmt: skip
    for i in range(1, 6):
        bundle = dagger_stage.stage_data(spec, i, 10_000, False)
        rec = bundle.extra["dagger"]
        assert bundle.labelled_episodes == 1000 + 200 * i == rec["labelled_episodes"]
        assert rec["aggregate"]["labelled_episodes_expected"] == 1000 + 200 * i
        assert [c.start for c in bundle.choices] == [0, *range(1000, 1000 + 200 * i, 200)]
        assert len(bundle.train) == 1000 + 200 * i  # 1 frame per episode
    assert bundle.labelled_episodes == cfg.iterations.labelled_episodes_total == 2000


# --------------------------------------------------------------------------- selection


def test_candidates_are_the_final_iterations_by_config(tmp_path: Path) -> None:
    from test_train_select import _fake_run

    run = _fake_run(tmp_path / "dagger", stages=5)
    man = mf.read_json(run / "manifest.json")
    man["arm"] = "some_new_iterated_arm"  # the rule never reads the arm name
    man["config_resolved"] = {"selection": {"candidates": FINAL_ITERATION}}
    mf.write_json_atomic(run / "manifest.json", man)
    got = candidates(run)
    assert [(c.stage, c.stage_step) for c in got] == [(5, 5000), (5, 10000)]
    assert _load("bc_dagger").raw["selection"]["candidates"] == FINAL_ITERATION
    assert _load("bc_matched").raw["selection"]["candidates"] == FINAL_ITERATION
    man["config_resolved"] = {"selection": {"candidates": "every_checkpoint_ever"}}
    mf.write_json_atomic(run / "manifest.json", man)
    with pytest.raises(ValueError, match="candidate rule"):
        candidates(run)


# --------------------------------------------------------------------------- copycat


class _Tiny(torch.nn.Module):
    """``a = tanh(W p[t] + V p[t-1] + 0.01 mean(images))``: an analytic Jacobian."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(3, 13, generator=g) * 0.5)
        self.v = torch.nn.Parameter(torch.randn(3, 13, generator=g) * 0.5)

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        pix = images.float().mean(dim=(1, 2, 3, 4))[:, None] * 0.01
        return torch.tanh(proprio[:, -1] @ self.w.T + proprio[:, 0] @ self.v.T + pix)


def test_last_action_jacobian_on_a_tiny_net() -> None:
    net = _Tiny().eval()
    g = torch.Generator().manual_seed(1)
    images = torch.randint(0, 255, (10, 2, 4, 4, 3), generator=g, dtype=torch.uint8)
    proprio = torch.randn(10, 2, 13, generator=g)
    cols = diagnostics.proprio_field_slice()
    assert cols == slice(9, 12)
    with torch.no_grad():
        a = net(images, proprio)
        # d a_i / d p[t]_j = (1 - a_i^2) W[i, j]; the older row (V) and the pixels never enter.
        expected = torch.stack([((1 - a[:, i] ** 2)[:, None] * net.w[i, cols]).mean(dim=0)
                                for i in range(3)])  # fmt: skip
    jac = diagnostics.last_action_jacobian(net, images, proprio, cols, batch=3)
    np.testing.assert_allclose(jac, expected.double().numpy(), rtol=1e-5, atol=1e-6)
    fd = diagnostics.last_action_jacobian_fd(net, images, proprio, cols, eps=1e-3)
    np.testing.assert_allclose(jac, fd, atol=1e-4)
    # The batch size of the backward passes does not change the mean.
    np.testing.assert_allclose(
        diagnostics.last_action_jacobian(net, images, proprio, cols, batch=10), jac, atol=1e-7
    )
    with pytest.raises(ValueError, match="last_action"):
        diagnostics.proprio_field_slice(variant="minus_last_action")


def test_jacobian_matches_phase4_finite_differences_on_a_bc_net() -> None:
    torch.manual_seed(0)
    net = build_net("bc", 0).eval()
    g = torch.Generator().manual_seed(2)
    images = torch.randint(0, 255, (3, 2, 96, 96, 3), generator=g, dtype=torch.uint8)
    proprio = torch.randn(3, 2, 13, generator=g) * 0.3
    cols = diagnostics.proprio_field_slice()
    jac = diagnostics.last_action_jacobian(net, images, proprio, cols)
    fd = diagnostics.last_action_jacobian_fd(net, images, proprio, cols, eps=1e-3)
    np.testing.assert_allclose(jac, fd, atol=2e-3)
    assert (
        diagnostics.sample_indices(1000, 0).tolist()
        == np.random.default_rng(0).choice(1000, 512, replace=False).tolist()
    )  # P4-D5's draw
    assert len(diagnostics.sample_indices(10, [0, 1, 2])) == 10


# --------------------------------------------------------------------------- report


def test_agreement_from_the_sidecar(tmp_path: Path) -> None:
    store = write_store(tmp_path / "iter1.h5", [synthetic_record(1000, 3, 1),
                                                synthetic_record(1001, 4, 2)])  # fmt: skip
    with DemoStore(store) as s:
        label = np.concatenate([s.labels(episode_group(j)) for j in (1000, 1001)])
    offset = np.zeros_like(label)
    offset[:, 0] = [0.05, 0.2, 0.0, 0.1, 0.0, 0.3, 0.05]
    teacher_exec = np.array([True, False, False, True, False, False, True])
    side = tmp_path / "iter1.sidecar.npz"
    dagger.write_npz(side, {
        "proposed": (label + offset).astype(np.float32),
        "teacher_exec": teacher_exec,
        "offsets": np.array([0, 3, 7]),
        "demo_index": np.array([1000, 1001]),
    })  # fmt: skip
    got = dagger_report.agreement(store, side)
    d = np.abs((label + offset).astype(np.float32).astype(np.float64) - label)[:, 0]
    assert got["all"]["steps"] == 7 and got["student"]["steps"] == 4
    assert got["all"]["l2_mean"] == pytest.approx(d.mean(), abs=1e-7)
    assert got["all"]["linf_le_tol_frac"] == pytest.approx(np.mean(d <= 0.1))
    assert got["student"]["l2_mean"] == pytest.approx(d[~teacher_exec].mean(), abs=1e-7)
    bad = tmp_path / "bad.npz"
    arrays = {"proposed": np.zeros((7, 3), np.float32), "teacher_exec": teacher_exec,
              "offsets": np.array([0, 4, 7]), "demo_index": np.array([1000, 1001])}  # fmt: skip
    dagger.write_npz(bad, arrays)
    with pytest.raises(ValueError, match="length"):
        dagger_report.agreement(store, bad)


def test_select_rows_are_reused_only_for_the_same_checkpoint(tmp_path: Path) -> None:
    from ila.data import episodes as data_episodes

    episodes = data_episodes.read_split("select")
    run = tmp_path / "run"
    ckpt = run / "checkpoints" / "s5_step010000.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    (run / "select").mkdir()
    header = (
        "episode_seed,ckpt_sha256,outcome,closing_speed_normal_m_s,steps,tunnelled,"
        "detectors_disagree"
    )
    lines = [header] + [
        f"{e.episode_seed},abc,{'success' if k else 'bounce'},0.3,60,False,False"
        for k, e in enumerate(episodes)
    ]
    (run / "select" / "s5_step010000_rows.csv").write_text("\n".join(lines) + "\n")
    summary, source = dagger_report.select_summary(
        ckpt, "abc", arm="bc_dagger", run_seed=0, cache_dir=tmp_path / "cache", workers=1, fly=False
    )
    assert summary is not None and source.startswith("reused:")
    assert summary["success"] == len(episodes) - 1 and summary["outcomes"]["bounce"] == 1
    cells = dagger_report._select_cells(summary, source)
    assert cells["select_n_tunnelled"] == 0 and cells["select_n_detectors_disagree"] == 0
    other = dagger_report.select_summary(
        ckpt, "def", arm="bc_dagger", run_seed=0, cache_dir=tmp_path / "cache", workers=1, fly=False
    )
    assert other == (None, "not flown")


def test_report_csv_has_every_column_and_policy_flown(tmp_path: Path) -> None:
    assert dagger_report.policy_flown(0.5).startswith("beta-mixture")
    assert dagger_report.policy_flown(0.0) == "student EMA"
    rows = [{"seed": 0, "iteration": 0, "policy_flown": dagger_report.policy_flown(None)},
            {"seed": 0, "iteration": 1, "beta": 0.5, "agree_l2_mean": 0.1}]  # fmt: skip
    out = dagger_report.write_report({"rows": rows, "missing": []}, tmp_path / "d.csv")
    text = (tmp_path / "d.csv").read_text().splitlines()
    assert text[0].split(",") == list(dagger_report.COLUMNS) and len(text) == 3
    assert len(set(dagger_report.COLUMNS)) == len(dagger_report.COLUMNS)
    assert json.loads(out.read_text())["csv_sha256"] == sha256_file(tmp_path / "d.csv")


def test_dagger_report_script_reads_no_frozen_list() -> None:
    from test_render_access import FROZEN_LIST_NAMES, _references

    for path in (REPO_ROOT / "scripts" / "dagger_report.py",
                 REPO_ROOT / "src" / "ila" / "train" / "dagger_report.py",
                 REPO_ROOT / "src" / "ila" / "train" / "dagger_stage.py"):  # fmt: skip
        assert _references(path.read_text(encoding="utf-8"), FROZEN_LIST_NAMES) == [], path


# --------------------------------------------------------------------------- workstation


@pytest.mark.slow
@pytest.mark.checkpoints
@pytest.mark.data
def test_two_real_iterations_on_the_cpu(tmp_path: Path) -> None:
    """2 iterations x 4 real rollouts from bc s0's last step, 2 CPU steps each (purpose smoke)."""
    from ila import p8_adapter

    real = Path(p8_adapter.load_paths()["data_root"])
    init = mf.run_dir("bc", 0) / "checkpoints" / "s0_step030000.pt"
    if not init.is_file() or not (real / "demos" / "ppo_2000.h5").is_file():
        pytest.skip("needs bc s0 and the ppo_2000 store")
    scratch = tmp_path / "data"
    scratch.mkdir()
    (scratch / "demos").symlink_to(real / "demos", target_is_directory=True)
    cfg = dataclasses.replace(_load("bc_dagger"), batch_size=8, precision="fp32")
    spec = loop.RunSpec(cfg, 0, tmp_path / "run", scratch, device="cpu", purpose="smoke",
                        steps_override=2, checkpoint_every_override=2, rollout_limit=4,
                        iterations_override=2, rollout_workers=4, init_from=init,
                        allow_dirty=True, prefetch=0)  # fmt: skip
    result = loop.train_run(spec, loop.build_stages(spec))
    assert result.state == "done" and result.global_step == 4
    man = mf.read_json(tmp_path / "run" / "manifest.json")
    recs = [man["data"][str(i)]["dagger"] for i in (1, 2)]
    assert recs[0]["rollout_student"]["sha256"] == sha256_file(init)
    assert recs[1]["rollout_student"]["sha256"] == sha256_file(
        tmp_path / "run" / "checkpoints" / "s1_step000002.pt"
    )
    assert [r["labelled_episodes"] for r in recs] == [1004, 1008]
    assert all(r["audit"]["clean"] is True for r in recs)
    assert [r["rollout"]["beta"] for r in recs] == [0.5, 0.25]
    assert all(r["heldout_overlap_episodes"] == 0 for r in recs)
    assert man["data"]["1"]["choices"][0]["attempted"] == 1000
