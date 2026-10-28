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
the names that one file may reference), reviewed by ``privileged-leak-auditor``. It is empty
today. Expected later entries, each only by such a reviewed entry: ``bc_state``'s dataset may
read ``teacher_obs`` (``privileged_observations``), and the ``modular`` arm's estimator-target
code may read ``estimator_targets``. No end-to-end vision student's dataset ever gets one.
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
#: entry needs a privileged-leak-auditor review recorded in docs/protocol.md. Empty today.
#: Expected later: bc_state's dataset -> {"privileged_observations"}; the modular arm's
#: estimator-target code -> {"estimator_targets"}.
AUDIT_ALLOWLIST: dict[str, frozenset[str]] = {}

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


def test_audit_names_exist_where_they_live() -> None:
    """The scanned names are the real accessor names (a rename would silently void the scan)."""
    from ila.data.store import DemoStore
    from ila.obs.split import ObsContract

    assert callable(ObsContract.audit_field)
    for name in AUDIT_NAMES - {"audit_field"}:
        assert callable(getattr(DemoStore, name)), name
