"""Phase 8 matrix runner (``ila.eval.matrix``): rows files, resumability, checks, projection.

Synthetic rows only, except the ``checkpoints`` test, which flies two ``select`` episodes
(never a frozen list) with a ``T_mix`` checkpoint to show that the per-episode shadow split does
not change any row but ``teacher_agreement``. Frozen lists are read for keys only (P3-D1 §4).
Simulation only.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pytest

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import runner

SHA = "ab" * 32


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
    m = ev_arms.load_matrix()
    jobs = ev_matrix.filter_jobs(m.jobs, ["core", "dp", "s0"])
    assert {(j.tag, j.arm, j.seed) for j in jobs} == {("core", "dp", 0)} and len(jobs) == 5
    assert len(ev_matrix.filter_jobs(m.jobs, ["dp_K", "dp_Ta", "1"])) == 5 + 8
    with pytest.raises(ValueError, match="no tag"):
        ev_matrix.filter_jobs(m.jobs, ["latency_arm"])
    order = ["reference_external", *[t for t in ev_arms.TAGS if t != "reference_external"]]
    ordered = ev_matrix.order_jobs(m.jobs, order)
    assert ordered[0].tag == "reference_external" and ordered[-1].tag == "appearance_shift"
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
    ]
    with pytest.raises(ValueError, match="permutation"):
        ev_matrix.order_jobs(m.jobs, ["core"])
    smoke = ev_matrix.smoke_jobs(m.jobs, ("SS3",))
    assert all(j.list_name == "select" and j.seed == 0 for j in smoke)
    assert len({(j.tag, j.arm) for j in smoke}) == len(smoke) == 7 + 11 + 1 + 1 + 3 + 1


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

    m = ev_arms.load_matrix()
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
