"""Phase 8 matrix runner (``ila.eval.matrix``): rows files, resumability, checks, projection.

Synthetic rows only, except the ``checkpoints`` test, which flies two ``select`` episodes
(never a frozen list) with a ``T_mix`` checkpoint to show that the per-episode shadow split does
not change any row but ``teacher_agreement``. Frozen lists are read for keys only (P3-D1 §4).
Simulation only.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import runner

SHA = "ab" * 32

#: Synthetic latency-arm delays (control steps), for expansion only (never delay.json's).
DELAYS: dict[str, int] = {"bc_dagger": 2, "dp": 3}


def _job(**kw: Any) -> ev_arms.Job:
    base: dict[str, Any] = {
        "tag": "core",
        "arm": "bc",
        "seed": 0,
        "list_name": "id",
        "sea_states": ("SS3", "SS4", "SS5", "SS6"),
        "full_list": True,
    }
    base.update(kw)
    return ev_arms.Job(**base)


def _ckpt(teacher: str = "T_ppo") -> ev_arms.Checkpoint:
    return ev_arms.Checkpoint(
        arm="bc",
        seed=0,
        kind="checkpoint",
        path=Path("/nonexistent.pt"),
        recorded_path="/nonexistent.pt",
        sha256=SHA,
        teacher=teacher,
        selection_json="results/phase4/runs/bc/s0/selection.json",
        selection_json_sha256="cd" * 32,
        global_step=10_000,
        select_success=150,
        select_n=150,
        select_mean_steps=62.0,
        selection_wall_clock_s=440.0,
        selection_workers=16,
        selection_steps=6 * 150 * 62.0,
        obs_variant="default",
    )


def _rows(job: ev_arms.Job, episodes: list[p8_adapter.ListedEpisode]) -> list[dict[str, Any]]:
    inf = job.inference
    out = []
    for e in episodes:
        row: dict[str, Any] = dict.fromkeys(runner.EVAL_EPISODE_COLUMNS, 0.25)
        row.update(
            regime=e.regime,
            ss=e.ss,
            index=e.index,
            episode_seed=e.episode_seed,
            outcome="success",
            tunnelled=False,
            arm=job.arm,
            teacher="T_ppo",
            student_seed=job.seed,
            Ta=None if inf is None else inf.Ta,
            K=None if inf is None else inf.K,
            ensemble=None if inf is None else inf.ensemble,
            latency_steps=0,
            render_cfg_sha256=runner.render_cfg_sha256("full", job.appearance_shift),
            ckpt_sha256=SHA,
            inference_calls=61,
        )
        out.append(row)
    return out


def test_rows_csv_keeps_p8_columns_first_and_appends_tags() -> None:
    job = _job(appearance_shift="bg_hue", tag="appearance_shift", sea_states=("SS5",))
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS5"][:3]
    rows = _rows(job, eps)
    text = ev_matrix.rows_csv(rows, job)
    lines = text.splitlines()
    assert lines[0].split(",") == list(ev_matrix.ROW_COLUMNS)
    n_p8 = len(p8_adapter.EPISODE_COLUMNS)
    assert tuple(ev_matrix.ROW_COLUMNS[:n_p8]) == p8_adapter.EPISODE_COLUMNS
    assert ev_matrix.ROW_COLUMNS[-2:] == ("tag", "appearance_shift")
    runner_text = runner.vision_rows_csv(rows).splitlines()
    for ln, base in zip(lines[1:], runner_text[1:], strict=True):
        assert ln == base + ",appearance_shift,bg_hue"
    plain = ev_matrix.rows_csv(_rows(_job(), eps), _job())
    assert plain.splitlines()[1].endswith(",core,")
    with pytest.raises(ValueError, match="DIAGNOSTIC"):
        ev_matrix.rows_csv(
            [{k: v for k, v in r.items() if k != "teacher_agreement"} for r in rows], job
        )


def test_gzip_output_is_byte_reproducible(tmp_path: Path) -> None:
    job = _job(sea_states=("SS3",), full_list=False)
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS3"]
    rows = _rows(job, eps)
    a = ev_matrix.gzip_bytes(ev_matrix.rows_csv(rows, job))
    b = ev_matrix.gzip_bytes(ev_matrix.rows_csv(rows, job))
    assert a == b
    assert a[4:8] == b"\x00\x00\x00\x00"  # gzip MTIME field
    assert a[3] & 0x08 == 0  # no FNAME
    assert gzip.decompress(a).decode() == ev_matrix.rows_csv(rows, job)
    files = []
    for out in (tmp_path / "one", tmp_path / "two"):
        side = ev_matrix.write_job(
            out, job, _ckpt(), rows, plan=[], wall_clock_s=1.0, workers=1, provenance={}
        )
        files.append((out / job.rel_path).read_bytes())
        assert side["rows_gz_sha256"] == ev_matrix._sha(files[-1])
        assert side["n_rows"] == 200
    assert files[0] == files[1]


def test_resumability_matches_spec_checkpoint_and_bytes(tmp_path: Path) -> None:
    job = _job(sea_states=("SS4",), full_list=False)
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS4"]
    ck = _ckpt()
    assert not ev_matrix.job_done(tmp_path, job, ck)
    ev_matrix.write_job(
        tmp_path, job, ck, _rows(job, eps), plan=[], wall_clock_s=2.0, workers=3, provenance={}
    )
    assert ev_matrix.job_done(tmp_path, job, ck)
    other = ev_arms.Checkpoint(**{**ck.__dict__, "sha256": "ef" * 32})
    assert not ev_matrix.job_done(tmp_path, job, other)
    # another job with the same file name cannot exist (names are injective), but a changed spec
    # in the sidecar is refused
    assert not ev_matrix.job_done(tmp_path, ev_arms.Job(**{**job.__dict__, "tag": "ablation"}), ck)
    path = tmp_path / job.rel_path
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    assert not ev_matrix.job_done(tmp_path, job, ck)


def test_check_rows_requires_the_list_in_order() -> None:
    job = _job(tag="ablation", arm="bc_mix", sea_states=("SS4", "SS5", "SS6"), full_list=False)
    eps = ev_matrix.expected_episodes(job)
    assert len(eps) == 600
    ck = _ckpt()
    rows = _rows(job, eps)
    ev_matrix.check_rows(job, ck, rows, eps)
    with pytest.raises(ev_matrix.RowCheckError, match="rows for"):
        ev_matrix.check_rows(job, ck, rows[:-1], eps)
    with pytest.raises(ev_matrix.RowCheckError, match="list order"):
        ev_matrix.check_rows(job, ck, [rows[1], rows[0], *rows[2:]], eps)
    with pytest.raises(ev_matrix.RowCheckError, match="rows per cell"):
        ev_matrix.check_rows(job, ck, rows[:400], eps[:400])
    bad = [dict(r) for r in rows]
    bad[7]["ckpt_sha256"] = "00" * 32
    with pytest.raises(ev_matrix.RowCheckError, match="ckpt_sha256"):
        ev_matrix.check_rows(job, ck, bad, eps)
    bad = [dict(r) for r in rows]
    bad[0]["render_cfg_sha256"] = runner.render_cfg_sha256("full", "bg_hue")
    with pytest.raises(ev_matrix.RowCheckError, match="render_cfg_sha256"):
        ev_matrix.check_rows(job, ck, bad, eps)
    bad = [dict(r) for r in rows]
    bad[3]["inference_calls"] = None
    with pytest.raises(ev_matrix.RowCheckError, match="inference_calls"):
        ev_matrix.check_rows(job, ck, bad, eps)
    # rows read back from the CSV (text cells) pass the same check
    back = ev_matrix.text_rows(ev_matrix.rows_csv(rows, job))
    ev_matrix.check_rows(job, ck, back, eps)


def test_filter_and_order() -> None:
    m = ev_arms.load_matrix(delays=DELAYS)
    jobs = ev_matrix.filter_jobs(m.jobs, ["core", "dp", "s0"])
    assert {(j.tag, j.arm, j.seed) for j in jobs} == {("core", "dp", 0)} and len(jobs) == 5
    assert len(ev_matrix.filter_jobs(m.jobs, ["dp_K", "dp_Ta", "1"])) == 5 + 8
    with pytest.raises(ValueError, match="no tag"):
        ev_matrix.filter_jobs(m.jobs, ["latency"])
    lat = ev_matrix.filter_jobs(m.jobs, ["latency_arm"])
    assert len(lat) == 10 and {j.arm for j in lat} == {"bc_dagger", "dp"}
    assert len(ev_matrix.filter_jobs(m.jobs, ["latency_arm", "dp", "s0"])) == 1
    order = ["reference_external", *[t for t in ev_arms.TAGS if t != "reference_external"]]
    ordered = ev_matrix.order_jobs(m.jobs, order)
    assert ordered[0].tag == "reference_external" and ordered[-1].tag == "latency_arm"
    default = ev_matrix.order_jobs(m.jobs)
    assert [j.tag for j in default] == sorted(
        (j.tag for j in m.jobs), key=list(ev_arms.DEFAULT_ORDER).index
    )
    assert list(ev_arms.DEFAULT_ORDER) == [
        "core",
        "ablation",
        "dp_Ta",
        "dp_K",
        "appearance_shift",
        "reference_external",
        "latency_arm",
    ]
    with pytest.raises(ValueError, match="permutation"):
        ev_matrix.order_jobs(m.jobs, ["core"])
    smoke = ev_matrix.smoke_jobs(m.jobs, ("SS3",))
    assert all(j.list_name == "select" and j.seed == 0 for j in smoke)
    assert len({(j.tag, j.arm) for j in smoke}) == len(smoke) == 7 + 11 + 1 + 1 + 3 + 1 + 2
    assert {j.latency_steps for j in smoke if j.tag == "latency_arm"} == {2, 3}


def test_projection_known_answers() -> None:
    inf = ev_arms.Inference
    assert ev_matrix.diffusion_cost_ratio(inf(8, 10, False), 64.0) == 1.0
    # L = 64: base 64*11 + 8*(31 + 140) = 2072 ms; Ta = 1: 64*11 + 64*171 = 11648 ms.
    assert ev_matrix.diffusion_cost_ratio(inf(1, 10, False), 64.0) == pytest.approx(11648 / 2072)
    # ensembling calls every step, whatever Ta
    assert ev_matrix.diffusion_cost_ratio(inf(12, 10, True), 64.0) == pytest.approx(11648 / 2072)
    # K = 100 at Ta = 8: 704 + 8*(31 + 1400) = 12152
    assert ev_matrix.diffusion_cost_ratio(inf(8, 100, False), 64.0) == pytest.approx(12152 / 2072)
    ck = _ckpt()
    assert ck.cpu_s_per_step == pytest.approx(440.0 * 16 / (900 * 62.0))
    lengths = {("id", c): 60.0 for c in ("SS3", "SS4", "SS5")} | {("id", "SS6"): 90.0}
    job = _job(sea_states=("SS5", "SS6"), full_list=False)
    (p,) = ev_matrix.project([job], {("bc", 0): ck}, lengths)
    assert p.episodes == 400
    assert p.steps == pytest.approx(200 * 62.0 + 200 * 62.0 * 1.5)
    assert p.cpu_s == pytest.approx(p.steps * ck.cpu_s_per_step)


def test_latency_projection_known_answers() -> None:
    inf = ev_arms.Inference(8, 10, False)
    bc = _job(tag="latency_arm", arm="bc_dagger", sea_states=("SS5",), full_list=False)
    # BC: L = 60, sync 60*(11+31) = 2520 ms; d = 3: 60*11 + 20*31 = 1280 ms; d = 0, 1: sync.
    for d, want in ((0, 1.0), (1, 1.0), (3, 1280 / 2520)):
        j = ev_arms.Job(**{**bc.__dict__, "latency_steps": d})
        assert ev_matrix.latency_cost_ratio(j, 60.0) == pytest.approx(want)
    dp = _job(tag="latency_arm", arm="dp", inference=inf, sea_states=("SS5",), full_list=False)
    # dp: L = 64, calls every max(d, Ta) steps; d <= 8 costs as Ta = 8; d = 16: 4 calls.
    for d in (0, 3, 8):
        j = ev_arms.Job(**{**dp.__dict__, "latency_steps": d})
        assert ev_matrix.latency_cost_ratio(j, 64.0) == pytest.approx(1.0)
    j16 = ev_arms.Job(**{**dp.__dict__, "latency_steps": 16})
    assert ev_matrix.latency_cost_ratio(j16, 64.0) == pytest.approx((704 + 4 * 171) / 2072)
    with pytest.raises(ValueError, match="not a latency_arm job"):
        ev_matrix.latency_cost_ratio(_job(), 60.0)
    ck = _ckpt()
    lengths = {("id", c): 60.0 for c in ("SS3", "SS4", "SS5")} | {("id", "SS6"): 90.0}
    j3 = ev_arms.Job(**{**bc.__dict__, "arm": "bc", "latency_steps": 3})
    (p,) = ev_matrix.project([j3], {("bc", 0): ck}, lengths)
    ratio = ev_matrix.latency_cost_ratio(j3, 62.0)
    assert p.cpu_s == pytest.approx(200 * 62.0 * ck.cpu_s_per_step * ratio)


def test_check_rows_requires_the_latency_steps() -> None:
    job = _job(tag="latency_arm", arm="bc_dagger", sea_states=("SS4",), full_list=False)
    job = ev_arms.Job(**{**job.__dict__, "latency_steps": 2})
    eps = ev_matrix.expected_episodes(job)
    rows = [{**r, "latency_steps": 2} for r in _rows(job, eps)]
    ev_matrix.check_rows(job, _ckpt(), rows, eps)
    ev_matrix.check_rows(job, _ckpt(), ev_matrix.text_rows(ev_matrix.rows_csv(rows, job)), eps)
    with pytest.raises(ev_matrix.RowCheckError, match="latency_steps"):
        ev_matrix.check_rows(job, _ckpt(), _rows(job, eps), eps)  # rows say 0
    sync = _job(arm="bc_dagger", sea_states=("SS4",), full_list=False)
    with pytest.raises(ev_matrix.RowCheckError, match="latency_steps"):
        ev_matrix.check_rows(sync, _ckpt(), rows, eps)  # a synchronous job's rows must say 0


@pytest.mark.parametrize("arm", ["bc_dagger", "dp"])
def test_d0_latency_flight_is_the_synchronous_flight(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At d = 0, the runner call and the built student are the synchronous core job's.

    Both paths are captured at ``runner.run_list_vision`` (nothing is flown); the builds are
    called with ``load_student`` replaced by a recorder. Row-for-row equality of the delayed
    student at d = 0 is ``tests/test_deploy_latency_arm.py``'s.
    """
    from ila.deploy import latency_arm as la
    from ila.train import select as train_select

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    built: list[dict[str, Any]] = []

    def capture(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return []

    class Fake:
        def __init__(self, **kw: Any) -> None:
            built.append(kw)

    def fake_load(path: Any, device: Any, **kw: Any) -> Any:
        return Fake(path=str(path), device=device, **kw)

    monkeypatch.setattr(runner, "run_list_vision", capture)
    for mod in (train_select, ev_matrix):
        monkeypatch.setattr(mod, "read_checkpoint", lambda _p: {"obs_variant": "default"})
    import ila.policies.registry as registry

    monkeypatch.setattr(registry, "load_student", fake_load)
    monkeypatch.setattr(train_select, "load_student", fake_load)
    monkeypatch.setattr(la, "wrap_delayed", lambda s, d: ("delayed", d, s))

    m = ev_arms.load_matrix(delays={"bc_dagger": 0, "dp": 0})
    core = next(j for j in m.jobs if (j.tag, j.arm, j.seed, j.list_name) == ("core", arm, 0, "id"))
    lat = next(j for j in m.jobs if (j.tag, j.arm, j.seed) == ("latency_arm", arm, 0))
    assert lat.latency_steps == 0
    ck = ev_arms.Checkpoint(**{**_ckpt().__dict__, "arm": arm})
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS5"][:3]
    shadow = p8_adapter.teacher_spec("ppo")
    train_select.fly_checkpoint(
        ck.path,
        eps,
        arm=core.arm,
        run_seed=core.seed,
        sha256=ck.sha256,
        workers=2,
        chunk=5,
        inference=None if core.inference is None else core.inference.as_tuple(),
        teacher=ck.teacher,
        appearance_shift=core.appearance_shift,
        diagnostics=True,
        shadow_teacher=shadow,
    )
    ev_matrix._fly_latency(lat, ck, eps, workers=2, chunk=5, shadow_teacher=shadow)
    (sync_args, sync_kw), (lat_args, lat_kw) = calls
    assert sync_kw == lat_kw  # render, workers, chunk, shift, contract, diagnostics, shadow
    assert sync_args[0] == lat_args[0] and sync_args[2] == lat_args[2]  # episodes, cfgs
    assert sync_args[3] == lat_args[3]  # RowTags, latency_steps 0 on both
    assert sync_args[3].latency_steps == 0
    s_spec, l_spec = sync_args[1], lat_args[1]
    for field in ("method", "privileged", "run_seed", "needs_motion_feed"):
        assert getattr(s_spec, field) == getattr(l_spec, field), field
    sync_student = s_spec.build(None)
    delayed = l_spec.build(None)
    assert delayed[:2] == ("delayed", 0) and isinstance(delayed[2], Fake)
    assert isinstance(sync_student, Fake)
    sync_kw_built, lat_kw_built = built
    assert lat_kw_built.pop("temporal_ensembling", False) is False
    assert sync_kw_built.pop("temporal_ensembling") is False
    assert sync_kw_built == lat_kw_built  # path, cpu, ema, Ta/K (dp: 8, 10; bc: None)
    if arm == "dp":
        assert lat_kw_built["n_action_steps"] == 8 and lat_kw_built["num_inference_steps"] == 10
    else:
        assert lat_kw_built["n_action_steps"] is None


def test_latency_flight_refuses_other_jobs() -> None:
    ck = _ckpt()
    with pytest.raises(ValueError, match="not a latency_arm job"):
        ev_matrix.latency_flight(_job(), ck)
    ens = _job(tag="latency_arm", arm="dp", inference=ev_arms.Inference(8, 10, True))
    with pytest.raises(ValueError, match="Ta = 8, K = 10"):
        ev_matrix.latency_flight(ev_arms.Job(**{**ens.__dict__, "latency_steps": 1}), ck)
    onnx = ev_arms.Checkpoint(**{**ck.__dict__, "kind": "onnx_exports"})
    bc = ev_arms.Job(**{**_job(tag="latency_arm").__dict__, "latency_steps": 1})
    with pytest.raises(ValueError, match=".pt checkpoints only"):
        ev_matrix.latency_flight(bc, onnx)
    spec, tags = ev_matrix.latency_flight(bc, ck)
    assert tags.latency_steps == 1 and tags.Ta is None and spec.run_seed == 0


def test_dirty_tree_ignores_only_the_output_root(monkeypatch: pytest.MonkeyPatch) -> None:
    porcelain = " M src/ila/eval/arms.py\n?? results/e08/rows/bc/s0/id.csv.gz\n?? results/e08x/a\n"
    monkeypatch.setattr(ev_matrix, "_git", lambda *_: porcelain)
    out = p8_adapter.REPO_ROOT / "results" / "e08"
    assert ev_matrix.dirty_paths(out) == [" M src/ila/eval/arms.py", "?? results/e08x/a"]
    assert len(ev_matrix.dirty_paths()) == 3


@pytest.mark.checkpoints
@pytest.mark.slow
def test_tmix_split_changes_no_row_but_teacher_agreement(tmp_path: Path) -> None:
    """``bc_mix`` s0 on two ``select`` episodes: split flight == one joint flight, row for row."""
    from ila.data import episodes as data_episodes
    from ila.train import select

    m = ev_arms.load_matrix(delays=DELAYS)
    job = ev_matrix.smoke_jobs([j for j in m.jobs if j.arm == "bc_mix"], ("SS3",))[0]
    ck = ev_arms.resolve_checkpoint(m.arm("bc_mix"), 0)
    eps = data_episodes.read_split(data_episodes.SELECT_NAME)[:2]
    groups = ev_matrix.shadow_groups("T_mix", eps)
    assert sorted(len(i) for _, i in groups) == [1, 1]
    split, plan = ev_matrix.fly_job(job, ck, eps, workers=1)
    assert [p["shadow_teacher"] for p in plan] == ["ppo", "gated"]
    joint = select.fly_checkpoint(
        ck.path,
        eps,
        arm="bc_mix",
        run_seed=0,
        sha256=ck.sha256,
        workers=1,
        teacher=ck.teacher,
        diagnostics=True,
        shadow_teacher=p8_adapter.teacher_spec("ppo"),
    )

    def masked(rows: list[dict[str, Any]]) -> str:  # CSV text: NaN cells compare as "nan"
        return runner.vision_rows_csv([{**r, "teacher_agreement": 0.0} for r in rows])

    assert masked(split) == masked(joint)
    ppo_pos = dict(groups)["ppo"][0]
    assert split[ppo_pos]["teacher_agreement"] == joint[ppo_pos]["teacher_agreement"]
    ev_matrix.check_rows(job, ck, split, eps, frozen=False)


# --------------------------------------------------------------------------- RISK 1 / RISK 2


def _fake_ckpt(arm: str, seed: int, shas: dict[str, str]) -> ev_arms.Checkpoint:
    return ev_arms.Checkpoint(
        **{
            **_ckpt().__dict__,
            "arm": arm,
            "seed": seed,
            "selection_json_sha256": shas.get(f"{arm}/s{seed}", "11" * 32),
            "obs_variant": ev_arms.expected_obs_variant(arm),
        }
    )


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """``run_jobs`` with every flight, hash and P8 read replaced by recorders (nothing flown)."""
    state: dict[str, Any] = {"shas": {}, "verified": [], "flown": []}
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS3"][:2]

    def resolve(spec: ev_arms.ArmSpec, seed: int) -> ev_arms.Checkpoint:
        return _fake_ckpt(spec.arm, seed, state["shas"])

    def fly(job: ev_arms.Job, ck: Any, episodes: Any, **_: Any) -> Any:
        state["flown"].append(job.rel_path.as_posix())
        return _rows(job, list(episodes)), []

    monkeypatch.setattr(ev_matrix, "dirty_paths", lambda **_: [])
    monkeypatch.setattr(ev_matrix, "static_provenance", lambda _out: {"p8_commit": "p8"})
    monkeypatch.setattr(ev_matrix, "host_info", lambda: {})
    monkeypatch.setattr(ev_arms, "resolve_checkpoint", resolve)
    monkeypatch.setattr(ev_arms, "verify_checkpoint_bytes", lambda c: state["verified"].append(c))
    monkeypatch.setattr(ev_matrix, "fly_job", fly)
    monkeypatch.setattr(ev_matrix, "check_rows", lambda *_, **__: None)
    m = ev_arms.load_matrix(delays=DELAYS)
    jobs = [j for j in m.jobs if j.arm == "bc" and j.seed == 0][:2]
    state.update(matrix=m, jobs=jobs, out=tmp_path / "e08", status=tmp_path / "status")

    def run(frozen: bool = True) -> dict[str, Any]:
        return ev_matrix.run_jobs(
            state["jobs"],
            m,
            out_root=state["out"],
            status_dir=state["status"],
            workers=1,
            episodes_of=lambda _j: eps,
            frozen=frozen,
            log=lambda _line: None,
        )

    state["run"] = run
    return state


def test_checkpoint_verified_before_every_flown_job(fake_run: dict[str, Any]) -> None:
    fake_run["run"]()
    assert len(fake_run["flown"]) == 2
    assert [(c.arm, c.seed) for c in fake_run["verified"]] == [("bc", 0), ("bc", 0)]


def test_selection_pins_written_once_then_enforced(fake_run: dict[str, Any]) -> None:
    run, out = fake_run["run"], fake_run["out"]
    # a smoke-style invocation never writes pins
    run(frozen=False)
    assert not (out / ev_matrix.PINS_FILE).exists()
    fake_run["flown"].clear()
    for side in out.rglob("*.json"):
        side.unlink()
    # first non-smoke invocation pins every (arm, seed) of the whole matrix, not just its jobs
    info = run()
    pins = json.loads((out / ev_matrix.PINS_FILE).read_text(encoding="utf-8"))["pins"]
    assert len(pins) == 7 * 5 + 11 * 3 + 3 and set(pins) >= {"dp/s4", "R1_lerobot_dp/s2"}
    assert info["selection_pins"]["sha256"] == ev_matrix._sha(
        (out / ev_matrix.PINS_FILE).read_bytes()
    )
    assert info["invocations"][-1]["selection_pins_created"] is True
    # a later invocation with the same selections passes (and skips the finished jobs)
    fake_run["flown"].clear()
    run()
    assert fake_run["flown"] == []
    # any changed selection (even of an arm not in this invocation's jobs) refuses to fly
    fake_run["shas"]["bc_plus_clock/s2"] = "22" * 32
    for p in out.rglob("*.csv.gz"):
        p.unlink()
    with pytest.raises(ev_matrix.SelectionPinError, match="bc_plus_clock/s2"):
        run()
    assert fake_run["flown"] == []
    # the smoke path checks existing pins too
    with pytest.raises(ev_matrix.SelectionPinError):
        run(frozen=False)


# --------------------------------------------------------------------------- Phase 9 R1


def _write_delay(path: Path, delays: dict[str, int], *, extra: str = "") -> Path:
    arms = {a: {"d": d, "provider": "torch_cpu_fp32", "fallback": True} for a, d in delays.items()}
    rec = {"arms": arms, "h5a_verdict": "not supported", **({"note": extra} if extra else {})}
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def latency_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """``run_jobs`` on latency jobs with flights, hashes and P8 reads replaced (nothing flown)."""
    state: dict[str, Any] = {"flown": [], "on_fly": None}
    eps = [e for e in ev_arms.list_episodes("id") if e.ss == "SS4"][:2]

    def fly(job: ev_arms.Job, ck: Any, episodes: Any, **_: Any) -> Any:
        state["flown"].append(job.rel_path.as_posix())
        if state["on_fly"] is not None:
            state["on_fly"]()
        return _rows(job, list(episodes)), []

    monkeypatch.setattr(ev_matrix, "dirty_paths", lambda **_: [])
    monkeypatch.setattr(ev_matrix, "static_provenance", lambda _out: {"p8_commit": "p8"})
    monkeypatch.setattr(ev_matrix, "host_info", lambda: {})
    monkeypatch.setattr(ev_arms, "resolve_checkpoint", lambda spec, s: _fake_ckpt(spec.arm, s, {}))
    monkeypatch.setattr(ev_arms, "verify_checkpoint_bytes", lambda _c: None)
    monkeypatch.setattr(ev_matrix, "fly_job", fly)
    monkeypatch.setattr(ev_matrix, "check_rows", lambda *_, **__: None)
    delay = _write_delay(tmp_path / "delay.json", {"bc_dagger": 1, "dp": 6})
    out = tmp_path / "e08"

    def run(m: ev_arms.Matrix, jobs: list[ev_arms.Job], frozen: bool = True) -> dict[str, Any]:
        return ev_matrix.run_jobs(
            jobs,
            m,
            out_root=out,
            status_dir=tmp_path / "status",
            workers=1,
            episodes_of=lambda _j: eps,
            frozen=frozen,
            log=lambda _line: None,
        )

    def lat(m: ev_arms.Matrix, *arms: str) -> list[ev_arms.Job]:
        return [j for j in m.jobs if j.tag == "latency_arm" and j.seed == 0 and j.arm in arms]

    state.update(run=run, lat=lat, delay=delay, out=out)
    return state


def test_delay_json_is_pinned_and_recorded(latency_run: dict[str, Any]) -> None:
    run, lat, delay, out = (latency_run[k] for k in ("run", "lat", "delay", "out"))
    m = ev_arms.load_matrix(delay_file=delay)
    sha = p8_adapter.sha256_file(delay)
    assert m.delay_source == {
        "file": str(delay),
        "sha256": sha,
        "delays": {"bc_dagger": 1, "dp": 6},
    }
    info = run(m, lat(m, "bc_dagger", "dp"))
    assert len(latency_run["flown"]) == 2
    pin = json.loads((out / ev_matrix.LATENCY_PINS_FILE).read_text(encoding="utf-8"))
    assert pin["delay_json"] == str(delay) and pin["delay_json_sha256"] == sha
    assert pin["delays"] == {"bc_dagger": 1, "dp": 6}
    assert info["delay_json"] == {"file": str(delay), "sha256": sha, "delays": pin["delays"]}
    assert info["latency_pins"]["sha256"] == ev_matrix._sha(
        (out / ev_matrix.LATENCY_PINS_FILE).read_bytes()
    )
    assert info["invocations"][-1]["latency_pins_created"] is True
    job = lat(m, "dp")[0]
    side = out / job.rel_path.with_name(job.rel_path.name.removesuffix(".csv.gz") + ".json")
    rec = json.loads(side.read_text(encoding="utf-8"))
    assert (rec["latency_steps"], rec["delay_json_sha256"]) == (6, sha)
    assert rec["job"]["latency_steps"] == 6 and job.rel_path.name.endswith("__latency-d6.csv.gz")
    # A resumed invocation on the same delay.json passes the pin and skips finished jobs.
    latency_run["flown"].clear()
    info = run(ev_arms.load_matrix(delay_file=delay), lat(m, "bc_dagger", "dp"))
    assert latency_run["flown"] == [] and info["invocations"][-1]["latency_pins_created"] is False


@pytest.mark.parametrize("change", ["new_d", "same_d_new_bytes"])
def test_regenerated_delay_json_is_refused_on_resume(
    latency_run: dict[str, Any], change: str
) -> None:
    """R1: a regenerated delay.json never flies under a new stem; nothing is flown."""
    run, lat, delay, out = (latency_run[k] for k in ("run", "lat", "delay", "out"))
    m = ev_arms.load_matrix(delay_file=delay)
    run(m, lat(m, "bc_dagger"))  # first invocation pins d = 1 / 6
    assert len(latency_run["flown"]) == 1
    if change == "new_d":
        _write_delay(delay, {"bc_dagger": 1, "dp": 7})
    else:
        _write_delay(delay, {"bc_dagger": 1, "dp": 6}, extra="regenerated")
    m2 = ev_arms.load_matrix(delay_file=delay)
    with pytest.raises(ev_matrix.DelayPinError, match="differs from the pin"):
        run(m2, lat(m2, "dp"))
    assert len(latency_run["flown"]) == 1
    assert not list((out / "rows" / "dp").rglob("*latency-d7*"))
    # Even a core-only resume refuses while the matrix it expands disagrees with the pin.
    core = [j for j in m2.jobs if j.tag == "core" and j.arm == "bc" and j.seed == 0][:1]
    with pytest.raises(ev_matrix.DelayPinError):
        run(m2, core)


def test_delay_json_changed_mid_run_stops_the_next_latency_job(
    latency_run: dict[str, Any],
) -> None:
    run, lat, delay = latency_run["run"], latency_run["lat"], latency_run["delay"]
    m = ev_arms.load_matrix(delay_file=delay)
    latency_run["on_fly"] = lambda: _write_delay(delay, {"bc_dagger": 1, "dp": 6}, extra="x")
    with pytest.raises(ev_matrix.DelayPinError, match="changed mid-run"):
        run(m, lat(m, "bc_dagger", "dp"))
    assert len(latency_run["flown"]) == 1  # the second latency job was refused before flying


def test_latency_jobs_need_d_from_delay_json(latency_run: dict[str, Any]) -> None:
    run, lat, out = latency_run["run"], latency_run["lat"], latency_run["out"]
    m = ev_arms.load_matrix(delays={"bc_dagger": 1, "dp": 6})  # explicit, not from a file
    assert m.delay_source == {"file": None, "sha256": None, "delays": {"bc_dagger": 1, "dp": 6}}
    with pytest.raises(ev_matrix.DelayPinError, match="read from results/latency/delay.json"):
        run(m, lat(m, "dp"))
    assert latency_run["flown"] == [] and not (out / ev_matrix.LATENCY_PINS_FILE).exists()
    # A smoke-style run (select, frozen=False) may use them and writes no pin.
    run(m, lat(m, "dp"), frozen=False)
    assert len(latency_run["flown"]) == 1 and not (out / ev_matrix.LATENCY_PINS_FILE).exists()
    # A core-only invocation with explicit delays writes no delay pin either.
    core = [j for j in m.jobs if j.tag == "core" and j.arm == "bc" and j.seed == 0][:1]
    run(m, core)
    assert not (out / ev_matrix.LATENCY_PINS_FILE).exists()


def test_committed_delay_json_expands_and_is_recorded() -> None:
    """The committed results/latency/delay.json (Phase 9): d = 1 (bc_dagger), 6 (dp)."""
    path = p8_adapter.REPO_ROOT / "results" / "latency" / "delay.json"
    if not path.is_file():
        pytest.skip("results/latency/delay.json not committed on this branch")
    m = ev_arms.load_matrix()
    assert m.delay_source is not None
    assert m.delay_source["file"] == "results/latency/delay.json"
    assert m.delay_source["sha256"] == p8_adapter.sha256_file(path)
    flown = {j.arm: j.latency_steps for j in m.jobs if j.tag == "latency_arm"}
    assert flown == m.delay_source["delays"] == dict(ev_arms.read_delay_file(path))
    assert flown == {"bc_dagger": 1, "dp": 6}
    assert sum(j.n_episodes for j in m.jobs if j.tag == "latency_arm") == 6000
