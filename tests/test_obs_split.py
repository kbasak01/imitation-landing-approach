"""Phase 1: the observation contract (``ila.obs.split``) -- what a student is allowed to see.

The student's proprio vector is selected from P8's 25-d observation by field NAME
(``configs/obs/student.yaml``), with offsets taken from P8's ``obs_fields`` at import time.
These tests pin the split, prove no privileged field can reach a ``StudentInput``, and scan
the rest of ``src/ila`` for hand-written slices of an observation vector.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

import ila
from ila import p8_adapter
from ila.obs import split as obs_split

SRC = Path(ila.__file__).resolve().parent

PRIVILEGED = ("rel_position", "rel_velocity", "deck_normal", "rel_tilt", "height")
PROPRIO = ("attitude", "body_rates", "velocity", "last_action", "in_contact")

# Pinned layout of P8's 25-d evaluation observation: name -> (offset, size).
PINNED_LAYOUT = {
    "attitude": (0, 3),
    "body_rates": (3, 3),
    "velocity": (6, 3),
    "rel_position": (9, 3),
    "rel_velocity": (12, 3),
    "deck_normal": (15, 3),
    "rel_tilt": (18, 1),
    "height": (19, 1),
    "time_fraction": (20, 1),
    "last_action": (21, 3),
    "in_contact": (24, 1),
}


@pytest.fixture(scope="module")
def contract() -> obs_split.ObsContract:
    return obs_split.load_contract()


def _ramp(n: int = 25) -> np.ndarray:
    """An observation whose entry i equals i, so every selected index is visible."""
    return np.arange(n, dtype=np.float32)


# --------------------------------------------------------------------------- classification


def test_every_p8_field_classified_exactly_once(contract: obs_split.ObsContract) -> None:
    p8_names = [f.name for f in p8_adapter.obs_fields()]
    classified = [*contract.student_proprio, *contract.privileged, *contract.teacher_only]
    assert sorted(classified) == sorted(p8_names)
    assert len(classified) == len(set(classified))


def test_default_classification(contract: obs_split.ObsContract) -> None:
    assert contract.student_proprio == PROPRIO
    assert contract.privileged == PRIVILEGED
    assert contract.teacher_only == ("time_fraction",)


def test_yaml_names_are_classified_once() -> None:
    import yaml

    raw = yaml.safe_load(obs_split.CONTRACT_PATH.read_text(encoding="utf-8"))
    names = [*raw["student_proprio"], *raw["privileged"], *raw["teacher_only"]]
    assert len(names) == len(set(names))
    assert set(names) == {f.name for f in p8_adapter.obs_fields()}


# --------------------------------------------------------------------------- offsets


def test_offsets_and_sizes_equal_p8(contract: obs_split.ObsContract) -> None:
    offset = 0
    for field in p8_adapter.obs_fields():
        assert contract.layout[field.name] == (offset, field.size)
        offset += field.size
    assert contract.obs_dim == offset == 25


def test_pinned_layout(contract: obs_split.ObsContract) -> None:
    assert dict(contract.layout) == PINNED_LAYOUT
    assert (contract.proprio_dim, contract.priv_dim, contract.extras_dim) == (13, 11, 1)


def test_split_selects_by_name(contract: obs_split.ObsContract) -> None:
    proprio, priv, extras = contract.split(_ramp())
    assert proprio.dtype == priv.dtype == extras.dtype == np.float32
    assert proprio.tolist() == [*range(0, 9), *range(21, 25)]
    assert priv.tolist() == list(range(9, 20))
    assert extras.tolist() == [20]


def test_split_module_function_and_batches(contract: obs_split.ObsContract) -> None:
    batch = np.stack([_ramp(), _ramp() + 100.0])
    proprio, priv, extras = obs_split.split(batch)
    assert proprio.shape == (2, 13) and priv.shape == (2, 11) and extras.shape == (2, 1)
    assert proprio[1, 0] == 100.0 and priv[1, 0] == 109.0
    np.testing.assert_array_equal(proprio[0], contract.split(_ramp())[0])


def test_split_rejects_wrong_width(contract: obs_split.ObsContract) -> None:
    for bad in (_ramp(24), _ramp(13), _ramp(26)):
        with pytest.raises(ValueError, match="25"):
            contract.split(bad)


def test_split_on_real_observation(contract: obs_split.ObsContract) -> None:
    cfgs = p8_adapter.load_eval_configs()
    from ila.data.episodes import read_split

    listed = read_split("demos_0-1999")[0]  # a train-pool row (P1-D2 obligation 1)
    env = p8_adapter.make_env(cfgs, listed)
    try:
        obs, _ = env.reset(seed=listed.episode_seed)
    finally:
        env.close()
    proprio, priv, _ = contract.split(obs)
    np.testing.assert_array_equal(proprio[9:12], np.zeros(3, np.float32))  # last_action at t=0
    assert priv.shape == (11,)


# --------------------------------------------------------------------------- StudentInput


def test_student_input_has_no_field_for_privileged_data() -> None:
    names = {f.name for f in dataclasses.fields(obs_split.StudentInput)}
    assert names == {"images", "proprio", "proprio_fields", "variant"}
    for name in names:
        assert not any(p in name for p in ("priv", "obs", "state", "teacher", *PRIVILEGED))


def test_student_input_valid(contract: obs_split.ObsContract) -> None:
    images = np.zeros((2, 96, 96, 3), np.uint8)
    window = np.stack([_ramp(), _ramp() + 1.0])
    x = contract.student_input(images, window)
    assert x.proprio.shape == (2, 13) and x.proprio.dtype == np.float32
    assert x.images.shape == (2, 96, 96, 3) and x.images.dtype == np.uint8
    assert x.variant == "default" and x.proprio_fields == PROPRIO
    with pytest.raises(ValueError):
        x.proprio[0, 0] = 1.0  # read-only


def test_student_input_requires_field_names() -> None:
    """A hand-built array of the right width cannot pass without the contract's names."""
    images = np.zeros((2, 96, 96, 3), np.uint8)
    proprio = np.zeros((2, 13), np.float32)
    with pytest.raises(TypeError):
        obs_split.StudentInput(images=images, proprio=proprio)  # type: ignore[call-arg]
    # 13 columns made of privileged fields (rel_position 3 + rel_velocity 3 + deck_normal 3
    # + rel_tilt 1 + height 1 + 2 more) or simply misnamed: refused.
    for names in (
        ("attitude", "body_rates", "velocity", "rel_position", "in_contact"),
        ("attitude", "body_rates", "velocity", "in_contact", "last_action"),  # reordered
        ("attitude", "body_rates", "velocity", "last_action"),
        (),
    ):
        with pytest.raises(ValueError):
            obs_split.StudentInput(images=images, proprio=proprio, proprio_fields=names)
    ok = obs_split.StudentInput(images=images, proprio=proprio, proprio_fields=list(PROPRIO))
    assert ok.proprio_fields == PROPRIO  # normalised to a tuple


def test_student_input_rejects_full_observation() -> None:
    images = np.zeros((2, 96, 96, 3), np.uint8)
    with pytest.raises(ValueError, match="13"):
        obs_split.StudentInput(
            images=images, proprio=np.zeros((2, 25), np.float32), proprio_fields=PROPRIO
        )
    with pytest.raises(ValueError):
        obs_split.StudentInput(
            images=images, proprio=np.zeros((2, 11), np.float32), proprio_fields=PROPRIO
        )
    with pytest.raises(ValueError):
        obs_split.StudentInput(
            images=images, proprio=np.zeros(25, np.float32), proprio_fields=PROPRIO
        )
    with pytest.raises(TypeError):
        obs_split.StudentInput(  # type: ignore[call-arg]
            images=images,
            proprio=np.zeros((2, 13), np.float32),
            proprio_fields=PROPRIO,
            priv=np.zeros((2, 11)),
        )


def test_student_input_rejects_bad_images_and_dtypes() -> None:
    proprio = np.zeros((2, 13), np.float32)
    for images in (
        np.zeros((2, 96, 96, 3), np.float32),  # not uint8
        np.zeros((2, 96, 96), np.uint8),  # no channel axis
        np.zeros((3, 96, 96, 3), np.uint8),  # To mismatch with proprio
        np.zeros((2, 96, 96, 4), np.uint8),  # RGBA
    ):
        with pytest.raises(ValueError):
            obs_split.StudentInput(images=images, proprio=proprio, proprio_fields=PROPRIO)
    good = np.zeros((2, 96, 96, 3), np.uint8)
    with pytest.raises(ValueError):
        obs_split.StudentInput(
            images=good, proprio=proprio.astype(np.float64), proprio_fields=PROPRIO
        )
    with pytest.raises(ValueError):
        obs_split.StudentInput(
            images=good, proprio=proprio, proprio_fields=PROPRIO, variant="unknown"
        )


def test_student_input_from_window_requires_full_obs(contract: obs_split.ObsContract) -> None:
    images = np.zeros((2, 96, 96, 3), np.uint8)
    with pytest.raises(ValueError, match="25"):
        contract.student_input(images, np.zeros((2, 13), np.float32))
    with pytest.raises(ValueError):
        contract.student_input(images[:1], np.zeros((2, 25), np.float32))


# --------------------------------------------------------------------------- ablations


@pytest.mark.parametrize(
    ("variant", "field", "dim"),
    [("plus_clock", "time_fraction", 14), ("minus_last_action", "last_action", 10)],
)
def test_ablation_changes_exactly_one_field(
    contract: obs_split.ObsContract, variant: str, field: str, dim: int
) -> None:
    ablated = obs_split.load_contract(variant=variant)
    assert ablated.variant == variant
    changed = set(contract.student_proprio) ^ set(ablated.student_proprio)
    assert changed == {field}
    assert ablated.privileged == contract.privileged
    assert set(contract.teacher_only) ^ set(ablated.teacher_only) == {field}
    assert ablated.proprio_dim == dim
    # Every field still classified exactly once.
    names = [*ablated.student_proprio, *ablated.privileged, *ablated.teacher_only]
    assert sorted(names) == sorted(f.name for f in p8_adapter.obs_fields())
    # Order of the untouched fields is preserved.
    kept = [n for n in contract.student_proprio if n != field]
    assert [n for n in ablated.student_proprio if n != field] == kept


def test_plus_clock_appends_time_fraction() -> None:
    proprio, _, extras = obs_split.load_contract(variant="plus_clock").split(_ramp())
    assert proprio.tolist() == [*range(0, 9), *range(21, 25), 20]
    assert extras.shape == (0,)


def test_minus_last_action_drops_it() -> None:
    proprio, _, extras = obs_split.load_contract(variant="minus_last_action").split(_ramp())
    assert proprio.tolist() == [*range(0, 9), 24]
    assert sorted(extras.tolist()) == [20, 21, 22, 23]


def test_student_input_width_follows_variant() -> None:
    images = np.zeros((2, 96, 96, 3), np.uint8)
    clock = obs_split.load_contract(variant="plus_clock")
    window = np.stack([_ramp(), _ramp()])
    x = clock.student_input(images, window)
    assert x.proprio.shape == (2, 14) and x.proprio_fields == (*PROPRIO, "time_fraction")
    with pytest.raises(ValueError):  # clock columns under the default variant
        obs_split.StudentInput(
            images=images, proprio=np.zeros((2, 14), np.float32), proprio_fields=x.proprio_fields
        )


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="variant"):
        obs_split.load_contract(variant="plus_rel_position")


# --------------------------------------------------------------------------- guards


def _write_mutated(tmp_path: Path, move: str, src: str) -> Path:
    import yaml

    raw = yaml.safe_load(obs_split.CONTRACT_PATH.read_text(encoding="utf-8"))
    raw[src].remove(move)
    raw["student_proprio"].append(move)
    path = tmp_path / "student.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", PRIVILEGED)
def test_contract_refuses_privileged_in_proprio(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        obs_split.load_contract(_write_mutated(tmp_path, name, "privileged"))


def test_contract_refuses_clock_outside_plus_clock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="time_fraction"):
        obs_split.load_contract(_write_mutated(tmp_path, "time_fraction", "teacher_only"))


def test_contract_is_frozen(contract: obs_split.ObsContract) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.student_proprio = PRIVILEGED  # type: ignore[misc]


# ----------------------------------------------------------------------------- slice scanner
#
# AST heuristic for "no observation is indexed with literal positions outside ila/obs/split.py".
# Scanned: every module under src/ila (except obs/split.py) and under scripts/.
#
# An expression is OBS-NAMED when the thing being indexed resolves to a name matching OBS_NAME:
#   * a Name (`obs`, `obs0`, `ob`, `state`, `teacher_obs`, `proprio`, `priv`, ...);
#   * an Attribute's attr (`self.obs`), looking through `.iloc/.loc/.T/.data/.values/.mT`;
#     `.shape` is never an observation;
#   * a string key (`f["teacher_obs"]`, `batch["obs"]`), or the indexed value of a non-string
#     subscript (`obs_hist[t]` -> `obs_hist`);
#   * a call's function name (`env.last_obs()`), else its first argument (`np.asarray(obs)`).
# A LITERAL is an int constant (incl. negative) anywhere in: a plain index, slice bounds/step, a
# tuple or list index, `slice(...)`, `np.array/arange/torch.tensor/range(...)` arguments,
# `np.r_[...]`/`np.s_[...]`/`np.c_[...]`, or arithmetic (`obs[9 + k]`).
# FLAGGED: (1) an obs-named subscript whose index contains a literal; (2) a call to
# take/index_select/narrow/take_along_axis/gather/split/hsplit/array_split/tensor_split,
# method form on an obs-named object (`obs.narrow(-1, 9, 3)`) or function form whose first
# argument is obs-named (`np.take(obs, [9, 10])`), with a literal in any other positional
# argument or non-axis keyword.
# NO ESCAPE HATCH. A line pragma was tried and removed (leak re-audit, 2026-10-27): matched as a
# substring of the line it also silenced chained field slices (`obs_hist[0][9:12]`), multi-line
# subscripts, reason-less pragmas and text inside string literals. Write a literal time index
# under a name that is not obs-named, or index through ila.obs.split. If Phase 3 genuinely needs
# one, the escape comes back as a committed allowlist of (file, node) entries, scoped per node.
# NOT CAUGHT (by design of a syntactic check; the StudentInput name/width check, the auditor and
# the Phase 3 poison test cover these): an observation under an unrelated name (`x[9:12]`);
# index constants bound elsewhere (`IDX = (9, 10); obs[IDX]`); positions computed at run time
# from non-literal values; `getattr` / string-built access; column names in a DataFrame.
#
# TODO(Phase 3): poison test -- set the privileged and time_fraction columns of every stored
# teacher_obs to NaN (through ila.obs.split, by name), build every student dataset/batch from
# the poisoned store, and assert every student batch is finite. That catches what this
# syntactic scan cannot (any route by which a privileged column reaches a student input).
OBS_NAME = re.compile(
    r"(^|_)(obs\d*|ob|observations?|proprio|priv|privileged|teacher_obs|states?)(_|$)", re.I
)
_PASSTHROUGH_ATTRS = {"iloc", "loc", "T", "mT", "data", "values"}
_INDEX_BUILDERS = {"slice", "array", "asarray", "tensor", "as_tensor", "arange", "range"}
_INDEX_HELPERS = {"r_", "s_", "c_", "index_exp"}
_SELECT_CALLS = {
    "take", "index_select", "narrow", "take_along_axis", "gather",
    "split", "hsplit", "array_split", "tensor_split", "delete",
}  # fmt: skip
_AXIS_KEYWORDS = {"axis", "dim", "out", "mode"}


def _func_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _target_name(node: ast.expr) -> str | None:
    """The name an indexed expression resolves to (see the comment block above)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        if node.attr == "shape":
            return None
        if node.attr in _PASSTHROUGH_ATTRS:
            return _target_name(node.value)
        return node.attr
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        return _target_name(node.value)
    if isinstance(node, ast.Call):
        name = _func_name(node.func)
        if name is not None and OBS_NAME.search(name):
            return name
        if not node.args:
            return None
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value  # f.get("teacher_obs"), h5["obs"] via a getter
        return _target_name(first)
    return None


def _obs_named(node: ast.expr) -> bool:
    name = _target_name(node)
    return name is not None and OBS_NAME.search(name) is not None


def _has_int_literal(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.UnaryOp):
        return _has_int_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return _has_int_literal(node.left) or _has_int_literal(node.right)
    if isinstance(node, ast.Slice):
        return any(_has_int_literal(b) for b in (node.lower, node.upper, node.step))
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(_has_int_literal(e) for e in node.elts)
    if isinstance(node, ast.Starred):
        return _has_int_literal(node.value)
    if isinstance(node, ast.Call) and _func_name(node.func) in _INDEX_BUILDERS:
        return any(_has_int_literal(a) for a in node.args)
    if isinstance(node, ast.Subscript) and _func_name(node.value) in _INDEX_HELPERS:
        return _has_int_literal(node.slice)
    return False


def _select_call_flagged(node: ast.Call) -> bool:
    if _func_name(node.func) not in _SELECT_CALLS:
        return False
    keywords = [k.value for k in node.keywords if k.arg not in _AXIS_KEYWORDS]
    if isinstance(node.func, ast.Attribute) and _obs_named(node.func.value):
        return any(_has_int_literal(a) for a in [*node.args, *keywords])
    if node.args and _obs_named(node.args[0]):
        return any(_has_int_literal(a) for a in [*node.args[1:], *keywords])
    return False


def _obs_slices(source: str) -> list[int]:
    found: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript):
            hit = _obs_named(node.value) and _has_int_literal(node.slice)
        elif isinstance(node, ast.Call):
            hit = _select_call_flagged(node)
        else:
            continue
        if hit:
            found.add(node.lineno)
    return sorted(found)


SCANNER_BAD = [
    "obs[9:12]",
    "self.obs[..., 20]",
    "teacher_obs[:, 21]",
    "proprio[0]",
    "x.priv[-1]",
    'f["teacher_obs"][:, 9:12]',
    'batch["obs"][..., 9:20]',
    "obs[[9, 10, 11]]",
    "obs[slice(9, 12)]",
    "env.last_obs()[9:12]",
    "obs.narrow(-1, 9, 3)",
    "np.take(obs, [9, 10])",
    "np.take(obs, indices=[9, 10], axis=-1)",
    "obs[np.r_[9:12, 21]]",
    "obs[np.s_[9:12]]",
    "obs0[3:6]",
    "ob[1]",
    "state[0:3]",
    "states[:, 9]",
    "torch.index_select(obs, -1, torch.tensor([9, 10, 11]))",
    "obs.index_select(-1, idx)",
    "np.asarray(obs)[9:12]",
    "obs_hist[t][9:12]",
    "df_obs.iloc[:, 9:12]",
    "np.split(obs, [9, 20], axis=-1)",
    "obs[9 + k]",
    "privileged[..., 0:3]",
    "obs_hist[0][9:12]",
    'f.get("teacher_obs")[:, 9:12]',
    "np.delete(obs, [20], axis=-1)",
    "obs[0]  # obs-slice-ok: comments no longer silence the scanner",
    "obs[\n    9:12\n]",
]
SCANNER_OK = [
    "obs.shape[0]",
    "images[0:2]",
    "labels[0:3]",
    "frames[0]",
    "obs[idx]",
    "obs[:, contract.slc]",
    "obs[...]",
    "np.take(obs, idx, axis=-1)",
    "contract.split(obs)",
    "statement = [1][0]",
]
SCANNER_KNOWN_MISSES = ["x[9:12]", "IDX = (9, 10)\nobs[IDX]"]


@pytest.mark.parametrize("line", SCANNER_BAD)
def test_scanner_flags(line: str) -> None:
    assert _obs_slices(line) == [1]


@pytest.mark.parametrize("line", SCANNER_OK)
def test_scanner_passes(line: str) -> None:
    assert _obs_slices(line) == []


@pytest.mark.parametrize("source", SCANNER_KNOWN_MISSES)
def test_scanner_known_misses_are_documented(source: str) -> None:
    # Documented limits of a syntactic scan (comment block above); if one starts being caught,
    # move it to SCANNER_BAD.
    assert _obs_slices(source) == []


def _scan_roots() -> list[tuple[str, Path]]:
    scripts = p8_adapter.REPO_ROOT / "scripts"
    files = [(f"src/ila/{p.relative_to(SRC).as_posix()}", p) for p in sorted(SRC.rglob("*.py"))]
    files += [
        (f"scripts/{p.relative_to(scripts).as_posix()}", p) for p in sorted(scripts.rglob("*.py"))
    ]
    return [(rel, p) for rel, p in files if rel != "src/ila/obs/split.py"]


def test_no_literal_obs_slices_outside_split() -> None:
    roots = _scan_roots()
    assert any(rel.startswith("scripts/") for rel, _ in roots)
    offenders = [
        f"{rel}:{line}"
        for rel, path in roots
        for line in _obs_slices(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
