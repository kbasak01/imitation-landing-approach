"""Who may touch render metadata and PyBullet (privileged-leak re-audit, RISK B).

``RenderMeta`` (view/projection matrices = the drone's world pose, the physics step count =
the clock, the segmentation mask = a perfect deck mask) and raw PyBullet access are never student
inputs (P1-D1, il-conventions). This static scan, in the style of the ``rld`` import scanner,
restricts which packages may even reference them:

* ``ila/train``, ``ila/policies``, ``ila/deploy``: none of META_NAMES and no ``pybullet``.
* ``ila/data``: may build ``Scene`` via ``attach`` and call ``.render()`` (pixels only), but none
  of META_NAMES and no ``pybullet`` (all PyBullet access belongs to ``ila/render`` and the adapter).
* ``ila/eval``: restricted like the others, EXCEPT ``eval/runner.py`` (EXEMPT). The runner may
  call ``render_with_meta`` for its timing assertion (``meta.physics_step_count == 8 t``) only;
  it never returns, yields or passes on a ``RenderMeta`` (P1-D2). ``eval/batched.py`` was exempt
  until the Phase 2 leak audit (NOTE 2) and is scanned since: it has no forbidden reference.
  Metrics, reports, hypotheses and arms get nothing.
* ``ila/render``, ``ila/p8_adapter.py`` and ``ila/obs`` are the implementation and are not
  scanned here.

A reference is any of: ``import X`` / ``from … import X`` (module or alias name), a Name, an
Attribute (``x.render_with_meta``), or ``import_module("pybullet")`` / ``__import__(...)`` with a
literal module name, or ``sys.modules["pybullet…"]``. Dynamic ``getattr(obj, "render_with_meta")``
with a literal string is also flagged, as are ``hasattr``/``setattr``/``delattr``,
``operator.attrgetter`` and ``operator.methodcaller`` with a literal name. Any top-level module
whose name starts with ``pybullet`` (e.g. ``pybullet_utils.bullet_client``) counts as
``pybullet``, except ``pybullet_data`` (asset paths only).

Store audit accessors (leak audit, Phase 3, RISK 1)
--------------------------------------------------
``AUDIT_NAMES`` -- ``ObsContract.audit_field`` (any P8 field by name, privileged included) and
the demo store's non-student accessors ``privileged_observations`` (``teacher_obs``),
``model_times`` (``t_model_s``, the clock), ``audit_arrays`` (every dataset) and
``estimator_targets`` (``priv``) -- may not be referenced in ``ila/train``, ``ila/policies``,
``ila/deploy`` or ``ila/eval``, by the same reference rules (any Attribute on any object, a
Name, an import, a literal-string ``getattr``/``attrgetter``/...). ``ila/data`` (the store and
its audits) and ``ila/obs`` are where they live and are not scanned for them.

The only way in is a committed entry in ``AUDIT_ALLOWLIST`` (file relative to ``src/ila`` ->
the names that one file may reference), reviewed by ``privileged-leak-auditor``. Phase 4 adds
one, ``train/state_dataset.py`` -> ``privileged_observations`` (``bc_state``'s dataset, P8's
25-d observation by design; approved by ``privileged-leak-auditor`` 2026-10-28, conditional on
the ``StateFrames`` name scan: only ``train/state_dataset.py``, ``train/loop.py`` and
``train/smoke.py`` may name ``StateFrames`` or ``state_dataset``). Phase 7 adds the second,
``train/modular_targets.py`` -> ``estimator_targets`` (``modular``'s training target ``priv``)
and ``audit_field`` (the closed-loop truth of the estimation-error study, by field name, logged
only), reviewed 2026-11-02 by ``privileged-leak-auditor`` (no LEAK, no RISK; recorded in
P7-D1) before any ``modular`` run as P3-D1 §8 requires, conditional on the same kind of
name scan (``test_modular_targets_are_used_only_where_reviewed``: only
``train/modular_targets.py``, ``train/loop.py``, ``train/smoke.py`` and
``train/modular_eval.py`` may name the module, ``modular_eval`` or their names). No end-to-end
vision student's dataset ever gets one (``ila/train/dataset.py`` has none, and
``test_vision_dataset_has_no_audit_entry`` keeps it so).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ila

SRC = Path(ila.__file__).resolve().parent

META_NAMES = frozenset(
    {
        "render_with_meta",
        "RenderMeta",
        "render_pose",
        "scene_handles",
        "physics_step_count",
        "_plate_pose",
    }
)
PYBULLET = "pybullet"

#: Package (relative to src/ila) -> forbidden names.
RESTRICTED: dict[str, frozenset[str]] = {
    "train": META_NAMES | {PYBULLET},
    "policies": META_NAMES | {PYBULLET},
    "deploy": META_NAMES | {PYBULLET},
    "data": META_NAMES | {PYBULLET},
    "eval": META_NAMES | {PYBULLET},
}

#: Files (relative to src/ila) exempt from RESTRICTED: the runner's timing assertion only.
EXEMPT: frozenset[str] = frozenset({"eval/runner.py"})

#: Non-student store/contract accessors (module docstring, "Store audit accessors").
AUDIT_NAMES = frozenset(
    {
        "audit_field",
        "privileged_observations",
        "model_times",
        "audit_arrays",
        "estimator_targets",
    }
)

#: Packages (relative to src/ila) scanned for AUDIT_NAMES.
AUDIT_RESTRICTED: tuple[str, ...] = ("train", "policies", "deploy", "eval")

#: Reviewed exceptions: file (relative to src/ila) -> AUDIT_NAMES that file may reference. Each
#: entry needs a privileged-leak-auditor review recorded in docs/protocol.md.
AUDIT_ALLOWLIST: dict[str, frozenset[str]] = {
    # bc_state's dataset (Phase 4): P8's full 25-d teacher_obs, privileged by design (P3-D1 §3,
    # declared exception 3); feeds only ila.policies.state_mlp.StateMLP. Approved by
    # privileged-leak-auditor 2026-10-28 (Phase 4), conditional on the StateFrames name scan
    # (test_state_dataset_is_used_only_where_reviewed).
    "train/state_dataset.py": frozenset({"privileged_observations"}),
    # modular's estimator targets (Phase 7, P3-D1 §3/§8): ``priv`` as the training TARGET
    # (EstimatorDataset; inputs through the student path only) and the closed-loop truth of the
    # estimation-error study, selected by name and logged only (true_priv). Reviewed
    # 2026-11-02 by privileged-leak-auditor (no LEAK, no RISK; recorded in P7-D1), before any
    # modular run; conditional on the name scan (test_modular_targets_are_used_only_where_reviewed).
    "train/modular_targets.py": frozenset({"estimator_targets", "audit_field"}),
}

#: Vision-student dataset files that may never get an AUDIT_ALLOWLIST entry.
VISION_DATASET_FILES: frozenset[str] = frozenset({"train/dataset.py", "train/sampling.py"})

#: Call names whose literal-string argument names an attribute (aliased access).
_ATTR_BY_STRING_SECOND_ARG = frozenset({"getattr", "hasattr", "setattr", "delattr"})
_ATTR_BY_STRING_FIRST_ARG = frozenset({"attrgetter", "methodcaller"})


def _top(module: str) -> str:
    """Top-level module name; every ``pybullet*`` package except ``pybullet_data`` -> pybullet."""
    top = module.split(".")[0]
    if top.startswith(PYBULLET) and top != "pybullet_data":
        return PYBULLET
    return top


def _references(source: str, forbidden: frozenset[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                names += [_top(alias.name), alias.name.split(".")[-1]]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names += [_top(node.module), node.module.split(".")[-1]]
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Subscript):
            key = node.slice
            target = node.value
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "modules"
                and isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            ):
                names.append(_top(key.value))  # sys.modules["pybullet..."]
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if fname in {"import_module", "__import__"} and isinstance(first, ast.Constant):
                names.append(_top(str(first.value)))
            if fname in _ATTR_BY_STRING_SECOND_ARG and len(node.args) > 1:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    names.append(second.value)
            if fname in _ATTR_BY_STRING_FIRST_ARG:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        names += arg.value.split(".")  # attrgetter("a.b") reaches a and b
        found += [(getattr(node, "lineno", 0), n) for n in names if n in forbidden]
    return sorted(set(found))


PLANTED = [
    "import pybullet",
    "import pybullet as p",
    "from pybullet import getCameraImage",
    "from ila.render.camera import RenderMeta",
    "from ila.render.camera import render_pose as rp",
    "from ila.p8_adapter import physics_step_count",
    "frame, meta = scene.render_with_meta()",
    "h = p8_adapter.scene_handles(env)",
    "pose = sc._plate_pose()",
    'importlib.import_module("pybullet")',
    '__import__("pybullet.utils")',
    'fn = getattr(sc, "render_with_meta")',
    "x: RenderMeta",
    "import pybullet_utils.bullet_client as bc",
    "from pybullet_utils.bullet_client import BulletClient",
    'importlib.import_module("pybullet_envs")',
    'p = sys.modules["pybullet"]',
    'bc = sys.modules["pybullet_utils.bullet_client"]',
]
ALLOWED_IN_DATA = [
    "from ila.render import scene",
    "sc = scene.attach(env)",
    "frame = sc.render()",
    "images.append(frame.rgb)",
    "from ila.render.scene import Scene, attach",
    "import pybullet_data",
    'path = sys.modules["numpy"]',
]


@pytest.mark.parametrize("line", PLANTED)
def test_scanner_flags_planted_reference(line: str) -> None:
    assert _references(line, RESTRICTED["data"]) != []


@pytest.mark.parametrize("line", ALLOWED_IN_DATA)
def test_scanner_allows_pixels_only_use(line: str) -> None:
    assert _references(line, RESTRICTED["data"]) == []


def _offenders(package: str, sources: dict[str, str]) -> list[str]:
    """Forbidden references in ``sources`` (relative path under src/ila -> text), minus EXEMPT."""
    return [
        f"src/ila/{rel}:{line}:{name}"
        for rel, text in sorted(sources.items())
        if rel not in EXEMPT
        for line, name in _references(text, RESTRICTED[package])
    ]


def test_eval_exemption_is_only_runner() -> None:
    planted = "frame, meta = sc.render_with_meta()\nimport pybullet\n"
    sources = {
        "eval/runner.py": planted,
        "eval/batched.py": planted,
        "eval/metrics.py": planted,
        "eval/report.py": "from ila.render.camera import RenderMeta\n",
    }
    offenders = _offenders("eval", sources)
    assert {o.split(":")[0] for o in offenders} == {
        "src/ila/eval/batched.py",
        "src/ila/eval/metrics.py",
        "src/ila/eval/report.py",
    }


@pytest.mark.parametrize("package", sorted(RESTRICTED))
def test_restricted_packages_do_not_reference_render_meta_or_pybullet(package: str) -> None:
    root = SRC / package
    files = sorted(root.rglob("*.py"))
    assert files, f"src/ila/{package} has no modules to scan"
    sources = {p.relative_to(SRC).as_posix(): p.read_text(encoding="utf-8") for p in files}
    assert _offenders(package, sources) == []


# --------------------------------------------------------------------------- store audit names

PLANTED_AUDIT = [
    "obs = store.privileged_observations(name)",
    "t = ds.model_times(e)",
    "arrays = s.audit_arrays(e)",
    "priv = self.store.estimator_targets(e)",
    "x = contract.audit_field(obs, 'rel_position')",
    'f = getattr(store, "privileged_observations")',
    'ok = hasattr(store, "audit_arrays")',
    'get = operator.attrgetter("model_times")',
    'get = attrgetter("store.estimator_targets")',
    'call = methodcaller("audit_field", obs, "height")',
    "from ila.data.store import DemoStore\nread = DemoStore.privileged_observations",
    "fn = (lambda s: s.audit_arrays)(store)",
]
ALLOWED_AUDIT = [
    "inputs = store.student_inputs(name)",
    "y = store.labels(name)",
    "m = store.meta(name)",
    "p = contract.proprio(obs)",
    'f = getattr(store, "student_inputs")',
]


@pytest.mark.parametrize("line", PLANTED_AUDIT)
def test_audit_scanner_flags_planted_reference(line: str) -> None:
    assert _references(line, AUDIT_NAMES) != []


@pytest.mark.parametrize("line", ALLOWED_AUDIT)
def test_audit_scanner_allows_student_path(line: str) -> None:
    assert _references(line, AUDIT_NAMES) == []


def _audit_offenders(
    sources: dict[str, str], allowlist: dict[str, frozenset[str]] | None = None
) -> list[str]:
    """AUDIT_NAMES references in ``sources`` (path under src/ila -> text) not allowlisted."""
    allow = AUDIT_ALLOWLIST if allowlist is None else allowlist
    return [
        f"src/ila/{rel}:{line}:{name}"
        for rel, text in sorted(sources.items())
        for line, name in _references(text, AUDIT_NAMES - allow.get(rel, frozenset()))
    ]


def test_audit_allowlist_is_per_file_and_per_name() -> None:
    planted = "obs = store.privileged_observations(e)\npriv = store.estimator_targets(e)\n"
    sources = {"train/dataset.py": planted, "train/loop.py": planted}
    allow = {"train/dataset.py": frozenset({"privileged_observations"})}
    assert _audit_offenders(sources, allow) == [
        "src/ila/train/dataset.py:2:estimator_targets",
        "src/ila/train/loop.py:1:privileged_observations",
        "src/ila/train/loop.py:2:estimator_targets",
    ]
    # runner.py's RenderMeta exemption does not extend to the audit names.
    assert _audit_offenders({"eval/runner.py": planted}, {}) != []


def test_audit_allowlist_is_well_formed() -> None:
    for rel, names in AUDIT_ALLOWLIST.items():
        assert (SRC / rel).is_file(), rel
        assert rel.split("/")[0] in AUDIT_RESTRICTED, rel
        assert names and names <= AUDIT_NAMES, (rel, names)


@pytest.mark.parametrize("package", AUDIT_RESTRICTED)
def test_restricted_packages_do_not_reference_store_audit_accessors(package: str) -> None:
    files = sorted((SRC / package).rglob("*.py"))
    assert files, f"src/ila/{package} has no modules to scan"
    sources = {p.relative_to(SRC).as_posix(): p.read_text(encoding="utf-8") for p in files}
    assert _audit_offenders(sources) == []


def test_vision_dataset_has_no_audit_entry() -> None:
    assert not VISION_DATASET_FILES & set(AUDIT_ALLOWLIST)
    for rel in VISION_DATASET_FILES:
        assert (SRC / rel).is_file(), rel


#: P8 frozen-list accessors (``ila.p8_adapter``). Training and selection read ``splits/`` only.
FROZEN_LIST_NAMES = frozenset(
    {"read_list", "list_path", "list_paths", "list_names", "manifest_path"}
)


#: The training scripts (argparse wrappers over ``ila/train``; Phase 6 adds the dp sweeps and
#: diagnostics, Phase 7 the modular estimation study and the clock intervention).
TRAIN_SCRIPTS: tuple[str, ...] = (
    "train.py",
    "train_smoke.py",
    "select.py",
    "budget.py",
    "dp_sweeps.py",
    "dp_diagnostics.py",
    "modular_estimation.py",
    "clock_intervention.py",
)


#: R1's pathway (leak audit R3, Phase 6): the dataset export, every ``scripts/r1`` script
#: (dataset build, training, ONNX export, parity) and the ONNX student. Paths from the repo root.
R1_FILES: tuple[str, ...] = (
    "src/ila/data/export_lerobot.py",
    "src/ila/policies/onnx_dp.py",
    "src/ila/deploy/r1_parity.py",
)


def test_training_code_reads_no_frozen_list() -> None:
    """``ila/train``, its scripts and R1's pathway never read a P8 frozen list (P3-D1 §4)."""
    repo = SRC.parents[1]
    scripts = repo / "scripts"
    r1_scripts = sorted((scripts / "r1").glob("*.py"))
    assert {p.name for p in r1_scripts} >= {
        "export_dataset.py",
        "build_lerobot_dataset.py",
        "train_r1.py",
        "export_onnx.py",
    }
    files = [
        *sorted((SRC / "train").rglob("*.py")),
        *(scripts / n for n in TRAIN_SCRIPTS),
        *r1_scripts,
        *(repo / n for n in R1_FILES),
    ]
    assert all(p.is_file() for p in files)
    found = [
        f"{p.name}:{line}:{name}"
        for p in files
        for line, name in _references(p.read_text(encoding="utf-8"), FROZEN_LIST_NAMES)
    ]
    assert found == []
    assert _references("rows = p8_adapter.read_list(path)", FROZEN_LIST_NAMES) != []


#: ``bc_state``'s privileged dataset, by class and module name (leak audit Phase 4, R1).
STATE_DATASET_NAMES = frozenset({"StateFrames", "state_dataset"})

#: The only files (relative to src/ila) that may name it: the module, the loop, the smoke.
STATE_DATASET_USERS = frozenset({"train/state_dataset.py", "train/loop.py", "train/smoke.py"})


def _dynamic_imports(source: str, forbidden: frozenset[str]) -> list[tuple[int, str]]:
    """``import_module("a.b.c")`` / ``__import__(...)``: every dotted component, not just ``a``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if fname in {"import_module", "__import__"} and isinstance(first, ast.Constant):
                parts = str(first.value).split(".")
                found += [(node.lineno, n) for n in parts if n in forbidden]
    return found


def _state_dataset_offenders(sources: dict[str, str]) -> list[str]:
    return [
        f"src/ila/{rel}:{line}:{name}"
        for rel, text in sorted(sources.items())
        if rel not in STATE_DATASET_USERS
        for line, name in sorted(
            set(_references(text, STATE_DATASET_NAMES))
            | set(_dynamic_imports(text, STATE_DATASET_NAMES))
        )
    ]


def test_state_dataset_is_used_only_where_reviewed() -> None:
    sources = {
        p.relative_to(SRC).as_posix(): p.read_text(encoding="utf-8")
        for package in AUDIT_RESTRICTED
        for p in sorted((SRC / package).rglob("*.py"))
    }
    assert set(sources) >= STATE_DATASET_USERS
    assert _state_dataset_offenders(sources) == []


@pytest.mark.parametrize(
    "line",
    [
        "from ila.train.state_dataset import StateFrames",
        "from ila.train import state_dataset",
        "import ila.train.state_dataset as sd",
        "frames = sd.StateFrames(store, names)",
        'cls = getattr(module, "StateFrames")',
        'm = importlib.import_module("ila.train.state_dataset")',
    ],
)
def test_state_dataset_scan_flags_planted_use(line: str) -> None:
    for rel in ("train/dataset.py", "policies/bc.py", "eval/runner.py", "deploy/export.py"):
        assert _state_dataset_offenders({rel: line}) != [], (rel, line)
    assert _state_dataset_offenders({"train/loop.py": line}) == []


def test_audit_names_exist_where_they_live() -> None:
    """The scanned names are the real accessor names (a rename would silently void the scan)."""
    from ila.data.store import DemoStore
    from ila.obs.split import ObsContract

    assert callable(ObsContract.audit_field)
    for name in AUDIT_NAMES - {"audit_field"}:
        assert callable(getattr(DemoStore, name)), name


#: ``modular``'s target/truth module, by module and public names (Phase 7, mirrors bc_state's).
MODULAR_TARGET_NAMES = frozenset(
    {"modular_targets", "modular_eval", "EstimatorDataset", "true_priv", "replay_priv_check"}
)

#: The only files (relative to src/ila) that may name it: the module, the loop (training
#: targets), the smoke (overfit + target replay) and the estimation-error study (truth log).
MODULAR_TARGET_USERS = frozenset(
    {
        "train/modular_targets.py",
        "train/loop.py",
        "train/smoke.py",
        "train/modular_eval.py",
    }
)


def _modular_target_offenders(sources: dict[str, str]) -> list[str]:
    return [
        f"src/ila/{rel}:{line}:{name}"
        for rel, text in sorted(sources.items())
        if rel not in MODULAR_TARGET_USERS
        for line, name in sorted(
            set(_references(text, MODULAR_TARGET_NAMES))
            | set(_dynamic_imports(text, MODULAR_TARGET_NAMES))
        )
    ]


def test_modular_targets_are_used_only_where_reviewed() -> None:
    sources = {
        p.relative_to(SRC).as_posix(): p.read_text(encoding="utf-8")
        for package in AUDIT_RESTRICTED
        for p in sorted((SRC / package).rglob("*.py"))
    }
    assert set(sources) >= MODULAR_TARGET_USERS
    assert _modular_target_offenders(sources) == []


@pytest.mark.parametrize(
    "line",
    [
        "from ila.train.modular_targets import EstimatorDataset",
        "from ila.train import modular_targets",
        "x = mt.true_priv(contract, obs)",
        'm = importlib.import_module("ila.train.modular_targets")',
    ],
)
def test_modular_target_scan_flags_planted_use(line: str) -> None:
    for rel in ("train/dataset.py", "policies/modular.py", "eval/runner.py", "train/select.py"):
        assert _modular_target_offenders({rel: line}) != [], (rel, line)
    assert _modular_target_offenders({"train/modular_eval.py": line}) == []


def test_modular_policy_reads_no_truth() -> None:
    """``policies/modular.py`` names no store/contract audit accessor (truth stays out)."""
    text = (SRC / "policies" / "modular.py").read_text(encoding="utf-8")
    assert _references(text, AUDIT_NAMES) == []
    assert _references(text, frozenset({"modular_targets", "true_priv"})) == []
