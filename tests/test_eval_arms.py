"""Phase 8 matrix expansion and arm resolution (``ila.eval.arms``).

The job count is checked against a hand count from P3-D1 §3 / §6 and ``configs/eval/matrix.yaml``
(arithmetic in :func:`test_job_count_equals_the_hand_count`). Frozen lists are read for their
keys and cells only (allowed, P3-D1 §4); nothing is flown here. Checkpoint bytes are read only in
tests marked ``checkpoints`` (workstation only).
"""

from __future__ import annotations

import ast
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ila import p8_adapter
from ila.data import mix
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import stats
from ila.policies.dp import chunking
from ila.render.config import APPEARANCE_SHIFTS

#: P8's frozen lists and their cells (P3-D1 §1), written out by hand.
LIST_CELLS: dict[str, tuple[str, ...]] = {
    "static": ("static",),
    "id": ("SS3", "SS4", "SS5", "SS6"),
    "unseen_seastate": ("SS6",),
    "unseen_heading": ("SS3", "SS4", "SS5", "SS6"),
    "unseen_vessel": ("SS3", "SS4", "SS5", "SS6"),
}

CORE = ("bc_state", "bc", "bc_matched", "bc_dart", "bc_dagger", "dp", "modular")
ABLATIONS = (
    "bc_pid",
    "dp_pid",
    "bc_mix",
    "dp_mix",
    "bc_n100",
    "bc_n300",
    "dp_n100",
    "dp_n300",
    "bc_minus_last_action",
    "dp_minus_last_action",
    "bc_plus_clock",
)


@pytest.fixture(scope="module")
def matrix() -> ev_arms.Matrix:
    return ev_arms.expand(stats.read_matrix_yaml(), LIST_CELLS)


def _counts(m: ev_arms.Matrix) -> tuple[Counter[str], Counter[str]]:
    jobs: Counter[str] = Counter(j.tag for j in m.jobs)
    eps: Counter[str] = Counter()
    for j in m.jobs:
        eps[j.tag] += j.n_episodes
    return jobs, eps


def test_frozen_list_cells_are_p3d1s() -> None:
    assert ev_arms.frozen_list_cells() == LIST_CELLS


def test_matrix_yaml_is_sha_checked(tmp_path: Path) -> None:
    assert p8_adapter.sha256_file(stats.MATRIX_YAML) == stats.MATRIX_SHA256
    m = ev_arms.load_matrix(list_cells=LIST_CELLS)
    assert m.lists_all == ("static", "id", "unseen_seastate", "unseen_heading", "unseen_vessel")
    assert m.always_printed == ("teacher", "pid_feedforward", "pid_track_descend", "oracle_gated")


def test_job_count_equals_the_hand_count(matrix: ev_arms.Matrix) -> None:
    # 200 episodes per cell (P3-D1 §1). One job = one (arm, seed, list, cells, inference, shift).
    # core: 7 arms x 5 seeds x 5 lists = 175 jobs; 7 x 5 x 2 800 = 98 000 episodes.
    # ablation: 11 arms x 3 seeds x (id SS4-SS6) = 33 jobs; 33 x 600 = 19 800 episodes.
    # dp_Ta: 5 Ta x 2 ensembling = 10 settings, minus (8, off) = core dp, minus (1, on) = alias
    #   of (1, off): 8 flown x 5 seeds = 40 jobs; 40 x 600 = 24 000 episodes.
    # dp_K: 6 K minus K = 10 (core) = 5 x 5 seeds = 25 jobs; 25 x 200 (id SS5) = 5 000.
    # appearance_shift: 3 arms x 5 variants x 5 seeds = 75 jobs; 75 x 400 (id SS5-SS6) = 30 000.
    # reference_external: R1 x 3 seeds = 3 jobs; 3 x 800 (id SS3-SS6) = 2 400.
    # Total: 351 jobs, 179 200 episodes. Shared cells: 5 (Ta 8 off) + 5 (Ta 1 on) + 5 (K 10).
    assert len(APPEARANCE_SHIFTS) == 5
    jobs, eps = _counts(matrix)
    assert jobs == {
        "core": 7 * 5 * 5,
        "ablation": 11 * 3,
        "dp_Ta": (5 * 2 - 2) * 5,
        "dp_K": (6 - 1) * 5,
        "appearance_shift": 3 * 5 * 5,
        "reference_external": 3,
    }
    assert eps == {
        "core": 7 * 5 * 2800,
        "ablation": 11 * 3 * 600,
        "dp_Ta": 8 * 5 * 600,
        "dp_K": 5 * 5 * 200,
        "appearance_shift": 3 * 5 * 5 * 400,
        "reference_external": 3 * 800,
    }
    assert len(matrix.jobs) == 351 and sum(eps.values()) == 179_200
    assert Counter((s.tag, s.inference.as_tuple()) for s in matrix.shared) == {
        ("dp_Ta", (8, 10, False)): 5,
        ("dp_Ta", (1, 10, True)): 5,
        ("dp_K", (8, 10, False)): 5,
    }
    # Flying (Ta = 1, on) instead adds 5 jobs and 3 000 episodes.
    flown = ev_arms.expand(stats.read_matrix_yaml(), LIST_CELLS, alias_ta1_ensemble=False)
    jobs2, eps2 = _counts(flown)
    assert jobs2["dp_Ta"] == 45 and eps2["dp_Ta"] == 27_000 and len(flown.shared) == 10


def test_arms_seeds_teachers_and_variants(matrix: ev_arms.Matrix) -> None:
    by = {a.arm: a for a in matrix.arms}
    assert set(by) == {*CORE, *ABLATIONS, "R1_lerobot_dp"}
    for a in CORE:
        assert by[a].tag == "core" and by[a].seeds == (0, 1, 2, 3, 4)
    for a in ABLATIONS:
        assert by[a].tag == "ablation" and by[a].seeds == (0, 1, 2)
    assert by["R1_lerobot_dp"].tag == "reference_external" and by["R1_lerobot_dp"].seeds == (
        0,
        1,
        2,
    )
    assert {a for a, s in by.items() if s.teacher == "T_pid"} == {"bc_pid", "dp_pid"}
    assert {a for a, s in by.items() if s.teacher == "T_mix"} == {"bc_mix", "dp_mix"}
    assert by["bc_minus_last_action"].obs_variant == "minus_last_action"
    assert by["dp_minus_last_action"].obs_variant == "minus_last_action"
    assert by["bc_plus_clock"].obs_variant == "plus_clock"
    assert {a for a, s in by.items() if s.obs_variant != "default"} == {
        "bc_minus_last_action",
        "dp_minus_last_action",
        "bc_plus_clock",
    }
    core_inf = ev_arms.Inference(8, 10, False)
    for name, spec in by.items():
        diffusion = name == "dp" or name.startswith("dp_") or name == "R1_lerobot_dp"
        assert spec.inference == (core_inf if diffusion else None), name
    for j in matrix.jobs:
        if j.tag in ("core", "ablation", "appearance_shift", "reference_external"):
            assert j.inference == by[j.arm].inference
    assert {j.list_name for j in matrix.jobs if j.tag == "ablation"} == {"id"}
    assert {j.sea_states for j in matrix.jobs if j.tag == "ablation"} == {("SS4", "SS5", "SS6")}
    assert {j.sea_states for j in matrix.jobs if j.tag == "appearance_shift"} == {("SS5", "SS6")}
    assert {j.arm for j in matrix.jobs if j.tag == "appearance_shift"} == {
        "bc_dagger",
        "dp",
        "modular",
    }
    assert {j.appearance_shift for j in matrix.jobs if j.tag == "appearance_shift"} == set(
        APPEARANCE_SHIFTS
    )
    assert all(j.appearance_shift is None for j in matrix.jobs if j.tag != "appearance_shift")
    ref = [j for j in matrix.jobs if j.tag == "reference_external"]
    assert {j.sea_states for j in ref} == {("SS3", "SS4", "SS5", "SS6")}
    assert all(j.full_list for j in ref)  # id SS3-SS6 is the whole id list


def test_latency_arm_is_excluded_and_stated(matrix: ev_arms.Matrix) -> None:
    raw = stats.read_matrix_yaml()
    assert "latency_arm" in raw and set(matrix.skipped) == {"latency_arm"}
    assert "Phase 9" in matrix.skipped["latency_arm"]
    assert all(j.tag in ev_arms.TAGS for j in matrix.jobs)
    assert "latency_arm" not in ev_arms.TAGS
    assert all(j.inference is None or j.inference.K > 0 for j in matrix.jobs)
    # Nothing is dropped silently: an unknown block or a missing block raises.
    with pytest.raises(ValueError, match="neither expanded nor skipped"):
        ev_arms.expand({**raw, "new_block": {}}, LIST_CELLS)
    with pytest.raises(ValueError, match="missing blocks"):
        ev_arms.expand({k: v for k, v in raw.items() if k != "latency_arm"}, LIST_CELLS)


def test_core_dp_cell_is_not_reflown_in_the_sweeps(matrix: ev_arms.Matrix) -> None:
    core_inf = (8, 10, False)
    sweeps = [j for j in matrix.jobs if j.tag in ("dp_Ta", "dp_K")]
    assert all(j.inference is not None and j.inference.as_tuple() != core_inf for j in sweeps)
    core_dp = {(j.seed, j.list_name): j for j in matrix.jobs if j.tag == "core" and j.arm == "dp"}
    for s in matrix.shared:
        if s.inference.as_tuple() == core_inf:
            assert s.source == core_dp[(s.seed, "id")]
        assert set(s.sea_states) <= set(s.source.sea_states)
        assert s.source in matrix.jobs
    # Settings per seed: Ta sweep 10 = 8 flown + 2 shared; K sweep 6 = 5 flown + 1 shared.
    for seed in range(5):
        ta = {j.inference.as_tuple() for j in sweeps if j.seed == seed and j.tag == "dp_Ta"}  # type: ignore[union-attr]
        ta |= {s.inference.as_tuple() for s in matrix.shared if s.seed == seed and s.tag == "dp_Ta"}
        assert ta == {(t, 10, e) for t in (1, 2, 4, 8, 12) for e in (False, True)}
        k = {j.inference.as_tuple() for j in sweeps if j.seed == seed and j.tag == "dp_K"}  # type: ignore[union-attr]
        k |= {s.inference.as_tuple() for s in matrix.shared if s.seed == seed and s.tag == "dp_K"}
        assert k == {(8, kk, False) for kk in (1, 2, 5, 10, 25, 100)}


def test_ta1_ensemble_alias_is_the_same_policy(matrix: ev_arms.Matrix) -> None:
    # With ensembling, Ta is the window: Ta = 1 gives one source (t, To-1) of weight 1.0, which
    # is the action Ta = 1 executes without ensembling (same noise key (run_seed, seed, t)).
    for t in range(40):
        assert chunking.ensemble_sources(t, 2, 16, 1) == ((t, 1),)
        assert chunking.executed_indices(2, 16, 1) == (1,)
    assert chunking.ensemble_weights(1).tolist() == [1.0]
    # Ta > 1 with ensembling is a different policy (more than one source after the start).
    assert len(chunking.ensemble_sources(10, 2, 16, 2)) == 2
    aliases = [s for s in matrix.shared if s.inference.as_tuple() == (1, 10, True)]
    assert len(aliases) == 5
    for s in aliases:
        assert s.source.inference is not None
        assert s.source.inference.as_tuple() == (1, 10, False) and s.source.seed == s.seed


def test_output_naming_is_injective_and_clean(matrix: ev_arms.Matrix) -> None:
    flown = ev_arms.expand(stats.read_matrix_yaml(), LIST_CELLS, alias_ta1_ensemble=False)
    for m in (matrix, flown):
        paths = [j.rel_path for j in m.jobs]
        assert len(set(paths)) == len(paths)
        keys = {
            (j.arm, j.seed, j.list_name, j.sea_states, j.inference, j.appearance_shift)
            for j in m.jobs
        }
        assert len(keys) == len(m.jobs)  # the file name is a function of these, and back
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    for j in matrix.jobs:
        assert set(j.stem) <= allowed, j.stem
        assert j.rel_path.parts[:3] == ("rows", j.arm, f"s{j.seed}")
    named = {j.rel_path.as_posix(): j for j in matrix.jobs}
    assert named["rows/bc/s0/id.csv.gz"].tag == "core"
    assert named["rows/dp/s0/id__Ta8_K10_ens0.csv.gz"].tag == "core"
    assert named["rows/dp/s3/id-SS4-SS5-SS6__Ta12_K10_ens1.csv.gz"].tag == "dp_Ta"
    assert named["rows/dp/s1/id-SS5__Ta8_K100_ens0.csv.gz"].tag == "dp_K"
    assert named["rows/modular/s2/id-SS5-SS6__shift-bg_hue.csv.gz"].tag == "appearance_shift"
    assert named["rows/dp/s0/id-SS5-SS6__shift-low_contrast__Ta8_K10_ens0.csv.gz"].arm == "dp"
    assert named["rows/bc_mix/s2/id-SS4-SS5-SS6.csv.gz"].tag == "ablation"
    assert named["rows/R1_lerobot_dp/s0/id__Ta8_K10_ens0.csv.gz"].tag == "reference_external"


def test_every_arm_seed_resolves_to_one_recorded_selection(matrix: ev_arms.Matrix) -> None:
    shas = []
    for spec in matrix.arms:
        for seed in spec.seeds:
            path = ev_arms.selection_path(spec.arm, seed)
            ck = ev_arms.resolve_checkpoint(spec, seed)
            assert ck.selection_json == path.relative_to(p8_adapter.REPO_ROOT).as_posix()
            rec = json.loads(path.read_text(encoding="utf-8"))
            assert ck.sha256 == rec["selected"]["candidate"]["sha256"]
            assert ck.recorded_path == rec["selected"]["candidate"]["file"]
            assert ck.teacher == spec.teacher
            assert ck.kind == ("onnx_exports" if spec.arm == "R1_lerobot_dp" else "checkpoint")
            assert ck.cpu_s_per_step > 0
            shas.append(ck.sha256)
    assert len(shas) == 7 * 5 + 11 * 3 + 3 == len(set(shas))
    # the DART sigma trials (purpose tuning, sigma_<s>/s0) are never an arm's selection
    with pytest.raises(ev_arms.ArmResolutionError):
        ev_arms.selection_path("bc_dart_tune", 0)


def _copy_run(tmp: Path, phase: str, arm: str, seed: int, to_phase: str | None = None) -> Path:
    src = ev_arms.RESULTS_ROOT / phase / "runs" / arm / f"s{seed}"
    dst = tmp / (to_phase or phase) / "runs" / arm / f"s{seed}"
    dst.mkdir(parents=True)
    for name in ("selection.json", "manifest.json"):
        shutil.copy(src / name, dst / name)
    return dst


def _edit(path: Path, **changes: Any) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.update(changes)
    path.write_text(json.dumps(rec), encoding="utf-8")


def test_resolution_refuses_inconsistent_records(matrix: ev_arms.Matrix, tmp_path: Path) -> None:
    spec = matrix.arm("bc_mix")
    run = _copy_run(tmp_path, "phase7", "bc_mix", 0)
    assert ev_arms.resolve_checkpoint(spec, 0, tmp_path).teacher == "T_mix"
    _edit(run / "selection.json", teacher="T_ppo")
    with pytest.raises(ev_arms.ArmResolutionError, match="teacher"):
        ev_arms.resolve_checkpoint(spec, 0, tmp_path)
    shutil.rmtree(run.parents[2])
    run = _copy_run(tmp_path, "phase7", "bc_mix", 0)
    _edit(run / "selection.json", obs_variant="plus_clock")
    with pytest.raises(ev_arms.ArmResolutionError, match="obs_variant"):
        ev_arms.resolve_checkpoint(spec, 0, tmp_path)
    # the same (arm, seed) under two phases is refused
    _copy_run(tmp_path, "phase7", "bc_mix", 0, to_phase="phase6")
    with pytest.raises(ev_arms.ArmResolutionError, match="2 selection.json"):
        ev_arms.resolve_checkpoint(spec, 0, tmp_path)
    # a diffusion arm selected at another inference setting is refused
    dp_run = _copy_run(tmp_path, "phase7", "dp_mix", 1)
    _edit(dp_run / "selection.json", inference={"Ta": 4, "K": 10, "temporal_ensembling": False})
    with pytest.raises(ev_arms.ArmResolutionError, match="inference"):
        ev_arms.resolve_checkpoint(matrix.arm("dp_mix"), 1, tmp_path)
    with pytest.raises(ev_arms.ArmResolutionError, match="seed 3"):
        ev_arms.resolve_checkpoint(spec, 3)


def test_tmix_shadow_split_covers_each_cell_once_and_matches_assign_rows(
    matrix: ev_arms.Matrix,
) -> None:
    job = next(j for j in matrix.jobs if j.arm == "bc_mix" and j.seed == 0)
    eps = ev_matrix.expected_episodes(job)
    assert [e.ss for e in eps] == ["SS4"] * 200 + ["SS5"] * 200 + ["SS6"] * 200
    groups = ev_matrix.shadow_groups("T_mix", eps)
    assert [g for g, _ in groups] == ["ppo", "gated"]
    covered = sorted(i for _, idx in groups for i in idx)
    assert covered == list(range(len(eps)))
    assigned = mix.assign_rows(20261023, eps)
    for name, idx in groups:
        assert idx == [i for i, a in enumerate(assigned) if a == name]
        assert Counter(eps[i].ss for i in idx) == {"SS4": 100, "SS5": 100, "SS6": 100}
    # The SS4 split is the H2 reference's: assign_rows(20261023, id SS4 rows).
    ss4 = [e for e in ev_arms.list_episodes("id") if e.ss == "SS4"]
    ss4_ppo = {
        (e.index, e.episode_seed)
        for e, a in zip(ss4, mix.assign_rows(20261023, ss4), strict=True)
        if a == "ppo"
    }
    ppo_idx = dict(groups)["ppo"]
    assert {(eps[i].index, eps[i].episode_seed) for i in ppo_idx if eps[i].ss == "SS4"} == ss4_ppo
    # Other teachers: one group, the whole job.
    assert ev_matrix.shadow_groups("T_ppo", eps) == [("ppo", list(range(600)))]
    assert ev_matrix.shadow_groups("T_pid", eps) == [("pid_feedforward", list(range(600)))]
    with pytest.raises(ValueError, match="unknown teacher"):
        ev_matrix.shadow_groups("T_gated", eps)


def _names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out |= {f"{node.module}.{a.name}" for a in node.names}
            out.add(str(node.module))
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def test_no_job_goes_to_the_batched_path() -> None:
    # modular (and every other arm) flies the CPU single-sample path through fly_checkpoint; the
    # batched path refuses modular anyway (tests/test_eval_runner.py::
    # test_modular_student_refused_off_default_and_on_batched_path).
    repo = p8_adapter.REPO_ROOT
    for path in (
        repo / "src" / "ila" / "eval" / "matrix.py",
        repo / "src" / "ila" / "eval" / "arms.py",
        repo / "scripts" / "eval_matrix.py",
    ):
        names = _names(path)
        assert not {n for n in names if "batched" in n or n in ("BatchedArm", "run_batched")}, path
    assert "fly_checkpoint" in _names(repo / "src" / "ila" / "eval" / "matrix.py")


def test_contract_variant_comes_from_the_checkpoint() -> None:
    # The driver never names a contract variant: fly_checkpoint reads obs_variant from the
    # checkpoint (tests/test_train_select.py::
    # test_fly_checkpoint_uses_the_checkpoints_contract_variant).
    import inspect

    from ila.train import select

    assert "contract_variant" not in inspect.signature(select.fly_checkpoint).parameters
    src = p8_adapter.REPO_ROOT / "src" / "ila" / "eval" / "matrix.py"
    assert "contract_variant" not in _names(src) and "obs_variant" not in _names(src)


@pytest.mark.checkpoints
@pytest.mark.slow
def test_checkpoint_bytes_match_the_recorded_sha256(matrix: ev_arms.Matrix) -> None:
    for spec in matrix.arms:
        for seed in spec.seeds:
            ev_arms.verify_checkpoint_bytes(ev_arms.resolve_checkpoint(spec, seed))


@pytest.mark.checkpoints
@pytest.mark.parametrize(
    "arm", ["bc_minus_last_action", "dp_minus_last_action", "bc_plus_clock", "bc_pid", "bc"]
)
def test_ablation_obs_variant_is_the_checkpoints(matrix: ev_arms.Matrix, arm: str) -> None:
    from ila.policies.registry import read_checkpoint

    spec = matrix.arm(arm)
    for seed in spec.seeds[:1]:
        ck = ev_arms.resolve_checkpoint(spec, seed)
        assert read_checkpoint(ck.path)["obs_variant"] == spec.obs_variant
