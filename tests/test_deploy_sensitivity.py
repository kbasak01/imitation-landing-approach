"""Phase 10 dp-family closed-loop outcome sensitivity (``ila.deploy.sensitivity``; L4).

CPU only, no PyBullet, no network: flights are injected (rows built here), so these tests pin the
settings, the flip/lost/gained arithmetic, the record and CSV formats, the dirty-tree refusal,
resumability, and that ``sensitivity-check`` re-decides everything from the files byte for
byte. Arm resolution is checked for every ``dp_*`` arm against the committed selection records;
the checkpoint-bytes leg is marked ``checkpoints`` (workstation only).
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ila.deploy import closed_loop as cl
from ila.deploy import sensitivity as sens
from ila.deploy.pipeline import DirtyTreeError
from ila.deploy.settings import load_settings
from ila.eval import arms as ev_arms
from ila.eval import runner, stats

#: P8's frozen lists and their cells (as tests/test_eval_arms.py; keys and cells only).
LIST_CELLS: dict[str, tuple[str, ...]] = {
    "static": ("static",),
    "id": ("SS3", "SS4", "SS5", "SS6"),
    "unseen_seastate": ("SS6",),
    "unseen_heading": ("SS3", "SS4", "SS5", "SS6"),
    "unseen_vessel": ("SS3", "SS4", "SS5", "SS6"),
}

ABLATION_DP = ("dp_mix", "dp_pid", "dp_n100", "dp_n300", "dp_minus_last_action")

GIT_CLEAN: dict[str, Any] = {"ila_commit": "a" * 40, "dirty": False, "dirty_paths": []}
ENV: dict[str, Any] = {"cpu": "test-cpu", "gpu": "test-gpu", "torch": "test"}


def _quiet(msg: str) -> None:
    del msg


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(sens.SENSITIVITY_YAML.read_text("utf-8"))
    return raw


def _settings(tmp: Path, **overrides: Any) -> sens.SensitivitySettings:
    """The committed YAML with every output and the control's Phase 9 files under ``tmp``."""
    raw = copy.deepcopy(_raw())
    raw["results_dir"] = str(tmp / "out")
    raw["control"] = {
        "arm": "dp",
        "record": str(tmp / "p9" / "torch_cuda_fp32.json"),
        "rows": str(tmp / "p9" / "torch_cuda_fp32_rows.csv"),
        "reference_rows": str(tmp / "p9" / "torch_cpu_fp32_rows.csv"),
    }
    for dotted, value in overrides.items():
        node = raw
        *head, last = dotted.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    return sens.from_mapping(raw, sens.SENSITIVITY_YAML)


def _rows(outcomes: list[str], tag: str = "x") -> list[dict[str, Any]]:
    rows = []
    for i, o in enumerate(outcomes):
        r: dict[str, Any] = {c: f"{c}{i}{tag}" for c in runner.VISION_EPISODE_COLUMNS}
        r.update(regime="select", ss=f"SS{3 + i % 3}", index=i, episode_seed=1000 + i, outcome=o)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- settings


def test_committed_settings_load_and_agree_with_bench() -> None:
    s = sens.load_sensitivity_settings()
    b = load_settings(s.bench_yaml)
    assert s.arms == sens.DP_ARMS
    assert (s.seed, s.ta, s.k, s.temporal_ensembling) == (0, 8, 10, False)
    assert s.inference == {"Ta": 8, "K": 10, "temporal_ensembling": False}
    assert (s.list_name, s.n_episodes) == ("select", 150)
    assert (s.reference, s.perturbed) == ("torch_cpu_fp32", "torch_cuda_fp32")
    assert (s.workers_cpu, s.workers_gpu, s.chunk) == (b.workers_cpu, b.workers_gpu, b.chunk)
    assert (s.workers_cpu, s.workers_gpu) == (24, 6)
    assert s.results_dir.as_posix().endswith("results/phase10/dp_sensitivity")
    assert s.control_arm == "dp"
    # The control's Phase 9 files are committed, and say 32 flips on torch_cuda_fp32.
    p9 = json.loads(s.control_record.read_text("utf-8"))
    assert (p9["arm"], p9["provider"], p9["flips"], p9["n"]) == ("dp", "torch_cuda_fp32", 32, 150)
    assert s.control_rows.is_file() and s.control_reference_rows.is_file()


@pytest.mark.parametrize(
    ("dotted", "value", "match"),
    [
        ("arms", ["dp", "bc_dagger"], "Diffusion-Policy"),
        ("arms", ["dp_mix"], "control"),
        ("arms", ["dp", "dp"], "unique"),
        ("seed", 1, "seed 0"),
        ("inference.Ta", 4, "Ta = 8"),
        ("inference.K", 25, "Ta = 8"),
        ("inference.temporal_ensembling", True, "Ta = 8"),
        ("list", "id", "select"),
        ("n_episodes", 100, "select"),
        ("perturbed", "ort_cuda_fp32", "torch_cuda_fp32"),
        ("reference", "ort_cpu_fp32", "torch_cpu_fp32"),
        ("workers_gpu", 0, "positive"),
    ],
)
def test_settings_refuse_what_the_definition_does_not_allow(
    tmp_path: Path, dotted: str, value: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _settings(tmp_path, **{dotted: value})


def test_settings_must_equal_bench_where_definitions_meet(tmp_path: Path) -> None:
    b = load_settings()
    sens.check_against_bench(_settings(tmp_path), b)
    with pytest.raises(ValueError, match="workers_gpu"):
        sens.check_against_bench(_settings(tmp_path, workers_gpu=12), b)
    with pytest.raises(ValueError, match="chunk"):
        sens.check_against_bench(_settings(tmp_path, chunk=5), b)


# --------------------------------------------------------------------------- arithmetic


def test_summarise_counts_flips_lost_gained_and_classes() -> None:
    ref = _rows(["success", "success", "off_pad", "bounce", "crash", "success"])
    pert = _rows(["success", "off_pad", "success", "hard_landing", "crash", "success"], tag="y")
    s = sens.summarise(ref, pert)
    assert s["n"] == 6
    assert s["flips"] == 3
    assert s["flip_rate"] == pytest.approx(0.5)
    assert (s["lost"], s["gained"]) == (1, 1)  # a non-success flip is neither
    assert (s["success_ref"], s["success_perturbed"]) == (3, 3)
    assert s["rows_text_differ"] == 6  # every row's text differs (tag), only 3 outcomes do
    assert s["outcomes_ref"] == {
        "crash": 1,
        "off_pad": 1,
        "hard_landing": 0,
        "bounce": 1,
        "success": 3,
        "timeout": 0,
    }
    assert list(s["outcomes_perturbed"]) == ["crash", "off_pad", "hard_landing", "bounce"] + [
        "success",
        "timeout",
    ]
    assert s["transitions"] == {
        "bounce->hard_landing": 1,
        "off_pad->success": 1,
        "success->off_pad": 1,
    }
    assert [f["index"] for f in s["flipped"]] == ["1", "2", "3"]


def test_summarise_refuses_unknown_classes_and_other_episodes() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        sens.summarise(_rows(["success"]), _rows(["landed"]))
    other = _rows(["success", "success"])
    other[1]["episode_seed"] = 7
    with pytest.raises(ValueError, match="episode order"):
        sens.summarise(_rows(["success", "success"]), other)


def test_csv_format_is_fixed_lf_and_four_decimals() -> None:
    ref = _rows(["success"] * 2 + ["off_pad"])
    pert = _rows(["off_pad", "success", "success"], tag="y")
    rec = {
        "arm": "dp_mix",
        "seed": 0,
        "checkpoint_sha256": "c" * 64,
        "reference_equals_selection_rows": True,
        **sens.summarise(ref, pert),
    }
    other = {**rec, "arm": "dp", "reference_equals_selection_rows": False}
    text = sens.sensitivity_csv([rec, other])
    assert "\r" not in text and text.endswith("\n")
    lines = text.split("\n")[:-1]
    assert lines[0].split(",") == list(sens.SENSITIVITY_COLUMNS)
    assert lines[1] == (f"dp_mix,0,{'c' * 64},3,true,2,0.6667,1,1,2,2,0,1,0,0,2,0,0,1,0,0,2,0")
    assert lines[2].startswith("dp,0,") and ",false,2,0.6667," in lines[2]
    assert sens.sensitivity_csv([]) == ",".join(sens.SENSITIVITY_COLUMNS) + "\n"


# --------------------------------------------------------------------------- dirty tree


def test_dirty_tree_outside_phase10_is_refused() -> None:
    dirty = {**GIT_CLEAN, "dirty": True, "dirty_paths": [" M src/ila/deploy/sensitivity.py"]}
    with pytest.raises(DirtyTreeError, match="results/phase10/"):
        sens.git_checked(False, git=dirty)
    ok = sens.git_checked(True, git=dirty)
    assert ok["dirty_outside_results"] is True
    inside = {**GIT_CLEAN, "dirty": True, "dirty_paths": ["?? results/phase10/dp_sensitivity/x"]}
    assert sens.git_checked(False, git=inside)["dirty_outside_results"] is False
    assert sens.git_checked(False, git=GIT_CLEAN)["dirty_outside_results"] is False


def test_stage_refuses_a_dirty_tree_before_flying(tmp_path: Path) -> None:
    calls: list[sens.Flight] = []
    dirty = {**GIT_CLEAN, "dirty": True, "dirty_paths": [" M Makefile"]}
    with pytest.raises(DirtyTreeError):
        sens.stage_sensitivity(
            _settings(tmp_path),
            git=dirty,
            fly=lambda f: calls.append(f) or [],  # type: ignore[func-returns-value]
            resolve=lambda _s, _a: pytest.fail("resolved on a dirty tree"),
            episodes=[object()] * 150,  # type: ignore[list-item]
            env=ENV,
            log=_quiet,
        )
    assert calls == []
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- stage, check


#: Per arm: (reference outcomes, perturbed outcomes) over 150 episodes.
def _plan(arm: str) -> tuple[list[str], list[str]]:
    k = sens.DP_ARMS.index(arm)
    ref = ["success" if i % 2 else "off_pad" for i in range(150)]
    pert = list(ref)
    for i in range(k + 2):  # arm k flips k + 2 episodes
        pert[i * 3] = "success" if ref[i * 3] != "success" else "hard_landing"
    return ref, pert


class _World:
    """Injected flights, resolver and selection files for every arm (no physics)."""

    def __init__(self, tmp: Path, settings: sens.SensitivitySettings) -> None:
        self.tmp = tmp
        self.settings = settings
        self.calls: list[tuple[str, str, int]] = []
        self.sel: dict[str, Path] = {}
        for arm in sens.DP_ARMS:
            ref, _ = _plan(arm)
            sel = tmp / "sel" / arm / "rows.csv"
            sel.parent.mkdir(parents=True, exist_ok=True)
            sel.write_text(cl.rows_text(_rows(ref)), "utf-8")
            (sel.parent / "rows.csv.gz").write_bytes(gzip.compress(sel.read_bytes()))
            self.sel[arm] = sel
        # The control's Phase 9 files: the same flight as the injected dp flight.
        ref, pert = _plan("dp")
        p9 = settings.control_record.parent
        p9.mkdir(parents=True, exist_ok=True)
        settings.control_rows.write_text(cl.rows_text(_rows(pert, tag="g")), "utf-8")
        settings.control_reference_rows.write_text(cl.rows_text(_rows(ref)), "utf-8")
        fl = cl.count_flips(_rows(pert, tag="g"), _rows(ref))
        settings.control_record.write_text(
            json.dumps(
                {
                    "arm": "dp",
                    "provider": "torch_cuda_fp32",
                    "checkpoint_sha256": self.sha("dp"),
                    "n": 150,
                    **fl,
                }
            ),
            "utf-8",
        )

    @staticmethod
    def sha(arm: str) -> str:
        return f"{sens.DP_ARMS.index(arm):x}" * 64

    def fly(self, req: sens.Flight) -> list[dict[str, Any]]:
        assert len(req.episodes) == 150
        self.calls.append((req.arm, req.provider, req.workers))
        ref, pert = _plan(req.arm)
        if req.provider == "torch_cpu_fp32":
            return _rows(ref)
        return _rows(pert, tag="g")

    def resolve(self, settings: sens.SensitivitySettings, arm: str) -> sens.ArmInfo:
        ck = ev_arms.Checkpoint(
            arm=arm,
            seed=0,
            kind="checkpoint",
            path=Path("/nonexistent") / f"{arm}.pt",
            recorded_path=f"/nonexistent/{arm}.pt",
            sha256=self.sha(arm),
            teacher=ev_arms.expected_teacher(arm),
            selection_json=f"results/test/{arm}/selection.json",
            selection_json_sha256="1" * 64,
            global_step=1,
            select_success=75,
            select_n=150,
            select_mean_steps=60.0,
            selection_wall_clock_s=1.0,
            selection_workers=1,
            selection_steps=1.0,
            obs_variant=ev_arms.expected_obs_variant(arm),
        )
        return sens.ArmInfo(
            arm=arm,
            seed=0,
            tag="core" if arm == "dp" else "ablation",
            checkpoint=ck,
            selection_pin="1" * 64,
            inference=settings.inference,
            selection_rows=self.sel[arm],
            committed_selection_rows=self.sel[arm].with_name("rows.csv.gz"),
        )

    def stage(self, **kw: Any) -> dict[str, Any]:
        return sens.stage_sensitivity(
            self.settings,
            git=GIT_CLEAN,
            env=ENV,
            fly=self.fly,
            resolve=self.resolve,
            episodes=[object()] * 150,  # type: ignore[list-item]
            log=_quiet,
            **kw,
        )


def test_stage_writes_rows_records_csv_and_check_is_clean(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    w = _World(tmp_path, s)
    res = w.stage()
    assert res["flown"] == list(sens.DP_ARMS) and res["kept"] == []
    # Reference on the CPU workers, perturbation on the GPU workers, per arm, in order.
    assert w.calls == [
        (a, p, n) for a in sens.DP_ARMS for p, n in (("torch_cpu_fp32", 24), ("torch_cuda_fp32", 6))
    ]
    out = s.results_dir
    for arm in sens.DP_ARMS:
        assert (out / f"{arm}_s0_torch_cpu_fp32_rows.csv").is_file()
        assert (out / f"{arm}_s0_torch_cuda_fp32_rows.csv").is_file()
        rec = json.loads((out / f"{arm}_s0.json").read_text("utf-8"))
        k = sens.DP_ARMS.index(arm)
        assert rec["status"] == "flown"
        assert rec["flips"] == k + 2
        assert rec["reference_equals_selection_rows"] is True
        assert rec["reference_equals_committed_selection_rows"] is True
        assert rec["selection_rows_flips"] == 0
        assert rec["inference"] == {"Ta": 8, "K": 10, "temporal_ensembling": False}
        assert rec["reference"]["workers"] == 24 and rec["perturbed"]["workers"] == 6
        assert rec["git"]["ila_commit"] == "a" * 40 and rec["env"] == ENV
        assert rec["lost"] + rec["gained"] == rec["flips"]
        assert (rec["control"] is not None) == (arm == "dp")
    ctl = json.loads((out / "dp_s0.json").read_text("utf-8"))["control"]
    assert ctl["reproduced"] is True and ctl["phase9_flips"] == ctl["flips"] == 2
    assert ctl["same_flipped_episodes"] is True
    assert ctl["perturbed_rows_equal_phase9"] is True
    assert ctl["reference_rows_equal_phase9"] is True
    assert ctl["checkpoint_sha256_equal"] is True
    # The CRLF rows as selection writes them; the table LF.
    assert b"\r\n" in (out / "dp_s0_torch_cpu_fp32_rows.csv").read_bytes()
    table = (out / "sensitivity.csv").read_bytes()
    assert b"\r" not in table and len(table.splitlines()) == 1 + len(sens.DP_ARMS)
    assert sens.check(s) == []
    info = json.loads((out / "run_info.json").read_text("utf-8"))
    assert set(info["arms"]) == set(sens.DP_ARMS)
    assert info["invocations"][-1]["arms_flown"] == list(sens.DP_ARMS)
    assert info["invocations"][-1]["env"] == ENV
    assert len(info["sensitivity_yaml_sha256"]) == 64


def test_stage_is_resumable_per_arm_and_force_refiles(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    w = _World(tmp_path, s)
    w.stage(arms=["dp", "dp_pid"])
    assert [c[0] for c in w.calls] == ["dp", "dp", "dp_pid", "dp_pid"]
    assert len((s.results_dir / "sensitivity.csv").read_text("utf-8").splitlines()) == 3
    w.calls.clear()
    res = w.stage()
    assert res["kept"] == ["dp", "dp_pid"]
    assert [c[0] for c in w.calls][::2] == ["dp_mix", "dp_n100", "dp_n300", "dp_minus_last_action"]
    w.calls.clear()
    res = w.stage(arms=["dp_n100"], force=True)
    assert res["flown"] == ["dp_n100"] and w.calls[0][0] == "dp_n100" and len(w.calls) == 2
    assert sens.check(s) == []
    info = json.loads((s.results_dir / "run_info.json").read_text("utf-8"))
    assert len(info["invocations"]) == 3
    with pytest.raises(sens.SensitivityError, match="not in sensitivity.yaml"):
        w.stage(arms=["bc_dagger"])


def test_reference_not_equal_to_selection_is_recorded_not_hidden(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    w = _World(tmp_path, s)
    ref, _ = _plan("dp_mix")
    ref = ["crash"] + ref[1:]
    w.sel["dp_mix"].write_text(cl.rows_text(_rows(ref)), "utf-8")
    w.stage(arms=["dp", "dp_mix"])
    rec = json.loads((s.results_dir / "dp_mix_s0.json").read_text("utf-8"))
    assert rec["reference_equals_selection_rows"] is False
    assert rec["selection_rows_flips"] == 1
    row = (s.results_dir / "sensitivity.csv").read_text("utf-8").splitlines()[2]
    assert row.startswith("dp_mix,0,") and ",false," in row
    assert sens.check(s) == []  # the record says so; the files agree with it


def test_control_not_reproduced_is_recorded(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    w = _World(tmp_path, s)
    p9 = json.loads(s.control_record.read_text("utf-8"))
    s.control_record.write_text(json.dumps({**p9, "flips": 44}), "utf-8")
    msgs: list[str] = []
    sens.stage_sensitivity(
        s,
        arms=["dp"],
        git=GIT_CLEAN,
        env=ENV,
        fly=w.fly,
        resolve=w.resolve,
        episodes=[object()] * 150,  # type: ignore[list-item]
        log=msgs.append,
    )
    ctl = json.loads((s.results_dir / "dp_s0.json").read_text("utf-8"))["control"]
    assert ctl["reproduced"] is False and ctl["phase9_flips"] == 44
    assert any("NOT REPRODUCED" in m for m in msgs)


def test_check_catches_tampering(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    w = _World(tmp_path, s)
    w.stage(arms=["dp", "dp_n300"])
    out = s.results_dir
    csv_path = out / "sensitivity.csv"
    good = csv_path.read_bytes()
    # The table: one byte, or CRLF instead of LF.
    csv_path.write_bytes(good.replace(b"\n", b"\r\n"))
    assert sens.check(s) == ["sensitivity.csv does not re-render byte-identically"]
    csv_path.write_bytes(good)
    # A rows file: one outcome changed (hash and summary both disagree).
    rows = out / "dp_n300_s0_torch_cuda_fp32_rows.csv"
    data = rows.read_bytes()
    rows.write_bytes(data.replace(b",success,", b",crash,", 1))
    bad = sens.check(s)
    assert any("differs from the recorded SHA-256" in b for b in bad)
    assert "dp_n300: outcomes_perturbed disagrees with the rows files" in bad
    rows.write_bytes(data)
    assert sens.check(s) == []
    # A record edited by hand.
    rec_path = out / "dp_n300_s0.json"
    rec = json.loads(rec_path.read_text("utf-8"))
    rec_path.write_text(json.dumps({**rec, "lost": rec["lost"] + 1}), "utf-8")
    bad = sens.check(s)
    assert "dp_n300: lost disagrees with the rows files" in bad
    assert "sensitivity.csv does not re-render byte-identically" in bad


def test_bench_cli_sensitivity_check(tmp_path: Path) -> None:
    import importlib.util

    raw = copy.deepcopy(_raw())
    s = _settings(tmp_path)
    raw["results_dir"] = str(s.results_dir)
    raw["control"] = {
        "arm": "dp",
        "record": str(s.control_record),
        "rows": str(s.control_rows),
        "reference_rows": str(s.control_reference_rows),
    }
    cfg = tmp_path / "sensitivity.yaml"
    cfg.write_text(yaml.safe_dump(raw), "utf-8")
    w = _World(tmp_path, sens.from_mapping(raw, cfg))
    w.stage(arms=["dp"])
    path = Path(__file__).resolve().parents[1] / "scripts" / "deploy" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["sensitivity-check", "--sensitivity-settings", str(cfg)]) == 0
    (s.results_dir / "sensitivity.csv").write_text("tampered\n", "utf-8")
    assert mod.main(["sensitivity-check", "--sensitivity-settings", str(cfg)]) == 1


# --------------------------------------------------------------------------- arm resolution


@pytest.fixture(scope="module")
def matrix() -> ev_arms.Matrix:
    return ev_arms.expand(stats.read_matrix_yaml(), LIST_CELLS, latency_arm=False)


@pytest.mark.parametrize("arm", [*sens.DP_ARMS, "bc_dagger", "bc_pid", "modular", "bc_plus_clock"])
def test_matrix_arm_spec_equals_the_expanded_matrix(arm: str, matrix: ev_arms.Matrix) -> None:
    # ila.deploy reads no frozen list; its spec must still be the one Phase 8's expand builds.
    assert sens.matrix_arm_spec(arm) == matrix.arm(arm)


@pytest.mark.parametrize("arm", sens.DP_ARMS)
def test_every_dp_arm_resolves_from_committed_records(arm: str) -> None:
    s = sens.load_sensitivity_settings()
    spec = sens.arm_spec(s, arm)
    assert spec.tag == ("core" if arm == "dp" else "ablation")
    assert spec.inference == ev_arms.Inference(8, 10, False)
    ck, pin, inf = sens.pinned_selection(s, spec)
    assert pin == ck.selection_json_sha256
    assert inf == {"Ta": 8, "K": 10, "temporal_ensembling": False}
    assert ck.obs_variant == ev_arms.expected_obs_variant(arm)
    assert ck.teacher == ev_arms.expected_teacher(arm)
    # The selected candidate's rows are committed beside selection.json (Phase 7: .csv.gz).
    from ila import p8_adapter

    committed = sens._committed_selection_rows(p8_adapter.REPO_ROOT / ck.selection_json)
    assert committed is not None and committed.is_file()


def test_arm_spec_refuses_arms_outside_the_settings() -> None:
    s = sens.load_sensitivity_settings()
    with pytest.raises(sens.SensitivityError, match="not one of"):
        sens.arm_spec(s, "bc_dagger")
    with pytest.raises(sens.SensitivityError, match="not in configs/eval/matrix.yaml"):
        sens.matrix_arm_spec("dp_nonexistent")


def test_ablations_never_fly_another_arms_onnx_graphs() -> None:
    for arm in ABLATION_DP:
        with pytest.raises(ValueError, match="only torch providers"):
            cl.provider_policy(arm, "ort_cpu_fp32", "g", "p", "t", None, tag="ablation")  # type: ignore[arg-type]


@pytest.mark.checkpoints
@pytest.mark.slow
@pytest.mark.parametrize("arm", sens.DP_ARMS)
def test_every_dp_arm_resolves_with_checkpoint_bytes(arm: str) -> None:
    s = sens.load_sensitivity_settings()
    info = sens.resolve_arm(s, arm)
    assert info.tag == ("core" if arm == "dp" else "ablation")
    assert info.selection_rows.is_file()
    assert info.committed_selection_rows is not None
    assert sens._read_bytes(info.committed_selection_rows) == info.selection_rows.read_bytes()
