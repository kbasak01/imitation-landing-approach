"""The episode clock never reaches a network (Phase 2 leak audit, RISK 1).

``NoiseKey.step`` (control step t) and ``episode_seed`` travel with every ``SlotRequest`` so a
stochastic policy can seed its per-request generator (il-conventions, Randomness). They may
seed the generator, feed ``chunking`` and drive slot bookkeeping, and nothing else. Three guards:

1. **Signature allowlist.** Every network ``forward`` (and the dp ``sample``/``predict_noise``)
   takes only :data:`ALLOWED_PARAMS`, and the only integer the U-Net sees is a diffusion index
   from the scheduler's ``timesteps``.
2. **Invariance.** With the initial noise held fixed, the tensors entering ``net.encoder`` and
   ``net.unet`` (forward-pre-hooks) and the actions out are bit-identical across episode seeds,
   across replans at t = 0 and t = Ta on the same window, and across slots of the shared-net
   server that differ only in ``episode_seed``, for dp and for bc.
3. **P1-D2 obligation 1, explicit allowlist.** ``NoiseKey`` and ``SlotRequest`` have exactly
   their fields, and an AST scan pins every read of ``.key.step`` / ``.key.episode_seed`` in
   ``ila/eval/batched.py`` and of ``DPStudent._t`` / ``_episode_seed`` to the permitted sites.

CPU only, no PyBullet. Steps are 30 Hz control steps (model scale, lambda = 1/25).
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import ila.eval.batched as batched_mod
import ila.policies.dp.policy as dpp
from ila.eval import batched
from ila.obs.split import load_contract
from ila.policies.bc import BCNet
from ila.policies.dp.unet1d import ConditionalUnet1D
from ila.policies.encoder import VisionEncoder
from ila.policies.registry import build_dummy

#: The only parameter names a network entry point may take (besides ``self``).
ALLOWED_PARAMS = frozenset(
    {
        "images",
        "proprio",
        "crop_generator",
        "noise",
        "num_inference_steps",
        "sample",
        "timestep",
        "cond",
        "global_cond",
    }
)

TA, K = 2, 2


# --------------------------------------------------------------------------- 1. signatures


def _entry_points() -> list[Any]:
    points: list[Any] = [BCNet.forward, VisionEncoder.forward, ConditionalUnet1D.forward]
    for name in ("sample", "predict_noise", "forward"):
        fn = getattr(dpp.DiffusionPolicyNet, name, None)
        if fn is not None and fn is not torch.nn.Module.forward:
            points.append(fn)
    return points


@pytest.mark.parametrize("fn", _entry_points(), ids=lambda f: f.__qualname__)
def test_network_signatures_are_allowlisted(fn: Any) -> None:
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    assert set(params) <= ALLOWED_PARAMS, (fn.__qualname__, params)
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(fn).parameters.values()
    ), fn.__qualname__


def test_dp_entry_points_exist() -> None:
    names = {f.__qualname__ for f in _entry_points()}
    assert {"DiffusionPolicyNet.sample", "DiffusionPolicyNet.predict_noise"} <= names


def test_ddpm_loss_signature() -> None:
    """The DDPM training objective takes the batch, epsilon and the diffusion index only.

    ``noise`` and ``timestep`` come from the data-loading generator (P2-D3 item 4); the only
    integer is the diffusion index k. No clock, no episode seed, no ``**kwargs``.
    """
    params = list(inspect.signature(dpp.ddpm_loss).parameters)
    assert params == ["net", "images", "proprio", "target", "noise", "timestep", "crop_generator"]
    assert set(params) - {"net", "target"} <= ALLOWED_PARAMS


def test_state_mlp_is_the_declared_exception() -> None:
    """``bc_state``'s MLP takes P8's 25-d ``obs`` (declared exception 3) and nothing else.

    ``obs`` is never an allowed parameter of a vision network, and no vision network contains a
    ``StateMLP``.
    """
    from ila.policies.state_mlp import StateMLP

    params = [p for p in inspect.signature(StateMLP.forward).parameters if p != "self"]
    assert params == ["obs"]
    assert "obs" not in ALLOWED_PARAMS
    for arm in ("bc", "dp"):
        net = build_dummy(arm, 0).net
        assert not any(isinstance(m, StateMLP) for m in net.modules())


#: Every ``nn.Module`` subclass defined under ``ila.policies``, reviewed. ``entry``: its
#: ``forward`` is in :func:`_entry_points` (signature allowlisted). ``internal``: reached only
#: from an entry point's forward, with the argument named here, so it sees nothing the entry
#: point did not. ``state``: ``bc_state``'s MLP, P8's 25-d observation by design (P3-D1 §3,
#: declared exception 3), checked by :func:`test_state_mlp_is_the_declared_exception`. A new
#: network fails :func:`test_policy_networks_are_reviewed` until it is added here, and an
#: ``entry`` must also join ``_entry_points``.
REVIEWED_MODULES: dict[str, str] = {
    "ila.policies.bc.BCNet": "entry",
    "ila.policies.encoder.VisionEncoder": "entry",
    "ila.policies.dp.policy.DiffusionPolicyNet": "entry (sample, predict_noise)",
    "ila.policies.dp.unet1d.ConditionalUnet1D": "entry",
    "ila.policies.encoder.SpatialSoftmax": "internal: VisionEncoder.forward (trunk features)",
    "ila.policies.dp.unet1d.SinusoidalPosEmb": "internal: ConditionalUnet1D.forward (timestep)",
    "ila.policies.dp.unet1d.ConditionalResidualBlock1D": (
        "internal: ConditionalUnet1D.forward (sample, FiLM of timestep embedding + global_cond)"
    ),
    "ila.policies.state_mlp.StateMLP": "state: forward(obs), P8's 25-d observation (bc_state)",
}


def _policy_modules() -> set[str]:
    import importlib
    import pkgutil

    import ila.policies

    found = set()
    for info in pkgutil.walk_packages(ila.policies.__path__, "ila.policies."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, torch.nn.Module) and cls.__module__ == info.name:
                found.add(f"{cls.__module__}.{cls.__qualname__}")
    return found


def test_policy_networks_are_reviewed() -> None:
    assert _policy_modules() == set(REVIEWED_MODULES)
    entry_classes = {f.__qualname__.split(".")[0] for f in _entry_points()}
    for name, verdict in REVIEWED_MODULES.items():
        if verdict.startswith("entry"):
            assert name.rsplit(".", 1)[1] in entry_classes, name


# --------------------------------------------------------------------------- 2. invariance


class _Hooks:
    """Forward-pre-hooks that copy every tensor entering the hooked modules, in call order."""

    def __init__(self, *modules: torch.nn.Module) -> None:
        self.calls: list[tuple[str, tuple[torch.Tensor, ...]]] = []
        self._handles = [
            m.register_forward_pre_hook(self._hook(type(m).__name__), with_kwargs=True)
            for m in modules
        ]

    def _hook(self, name: str) -> Any:
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module
            tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
            self.calls.append((name, tuple(t.detach().clone() for t in tensors)))

        return hook

    def clear(self) -> None:
        self.calls.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def _same_calls(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all(
        na == nb
        and len(ta) == len(tb)
        and all(torch.equal(x, y) for x, y in zip(ta, tb, strict=True))
        for (na, ta), (nb, tb) in zip(a, b, strict=True)
    )


def _window(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    contract = load_contract()
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, (contract.n_obs_steps, 96, 96, 3), dtype=np.uint8)
    proprio = rng.standard_normal((contract.n_obs_steps, contract.proprio_dim)).astype(np.float32)
    return images, proprio


@pytest.fixture
def fixed_noise(monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """Hold ``initial_noise`` fixed: it is the only permitted route of a seed, and it is cut."""
    fixed = torch.randn((1, 16, 3), generator=torch.Generator().manual_seed(1))

    def noise(seeds: Any, horizon: int, action_dim: int) -> torch.Tensor:
        del horizon, action_dim
        return fixed.repeat(len(seeds), 1, 1)

    monkeypatch.setattr(dpp, "initial_noise", noise)
    return fixed


@pytest.fixture(scope="module")
def dp_student() -> Any:
    torch.set_num_threads(1)
    return build_dummy("dp", 0, "cpu", n_action_steps=TA, num_inference_steps=K)


def test_dp_inputs_and_actions_do_not_depend_on_episode_seed_or_step(
    dp_student: Any, fixed_noise: torch.Tensor
) -> None:
    del fixed_noise
    hooks = _Hooks(dp_student.net.encoder, dp_student.net.unet)
    images, proprio = _window()
    try:
        runs = {}
        for episode_seed in (11, 99_999):
            hooks.clear()
            dp_student.reset(episode_seed)
            actions = [dp_student.act_vision(images, proprio) for _ in range(2 * TA)]
            runs[episode_seed] = (actions, list(hooks.calls))
    finally:
        hooks.remove()
    (a11, c11), (a99, c99) = runs[11], runs[99_999]
    # Across episode seeds: same network inputs, same actions.
    assert all(np.array_equal(x, y) for x, y in zip(a11, a99, strict=True))
    assert _same_calls(c11, c99)
    # Two replans (t = 0 and t = Ta) on the same window: identical inputs and chunks.
    per_replan = len(c11) // 2
    assert per_replan == 1 + K  # one encoder call + K U-Net calls per replan
    assert _same_calls(c11[:per_replan], c11[per_replan:])
    assert all(np.array_equal(a11[i], a11[TA + i]) for i in range(TA))


def _registry_arms() -> list[str]:
    from ila.policies.registry import REGISTRY

    return sorted(a for a, e in REGISTRY.items() if e.is_vision)


@pytest.mark.parametrize("arm", _registry_arms())
def test_registry_arm_invariant_to_episode_seed_and_step(
    arm: str, fixed_noise: torch.Tensor
) -> None:
    """Every registry arm: same window -> same network inputs and actions, any seed, any step."""
    del fixed_noise
    from ila.policies.registry import REGISTRY

    torch.set_num_threads(1)
    kw = {"n_action_steps": TA, "num_inference_steps": K} if REGISTRY[arm].kind == "dp" else {}
    student = build_dummy(arm, 0, "cpu", **kw)
    net = student.net
    hooks = _Hooks(net, *net.children())
    images, proprio = _window()
    try:
        runs = {}
        for episode_seed in (11, 99_999):
            hooks.clear()
            student.reset(episode_seed)
            actions = [student.act_vision(images, proprio) for _ in range(2 * TA)]
            runs[episode_seed] = (actions, list(hooks.calls))
    finally:
        hooks.remove()
    (a11, c11), (a99, c99) = runs[11], runs[99_999]
    assert c11, f"{arm}: no hooked network call"
    assert all(np.array_equal(x, y) for x, y in zip(a11, a99, strict=True))
    assert _same_calls(c11, c99)
    # Steps t and t + Ta on the same window give the same action (dp: the same chunk index).
    assert all(np.array_equal(a11[i], a11[TA + i]) for i in range(TA))


def test_unet_timesteps_come_from_the_scheduler(dp_student: Any, fixed_noise: torch.Tensor) -> None:
    del fixed_noise
    seen: list[torch.Tensor] = []

    def record(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        del module
        seen.append(args[1].detach().clone())

    handle = dp_student.net.unet.register_forward_pre_hook(record)
    images, proprio = _window()
    try:
        dp_student.reset(5)
        for _ in range(2 * TA):
            dp_student.act_vision(images, proprio)
    finally:
        handle.remove()
    schedule = [int(k) for k in dp_student.net.ddim(K).timesteps]
    n_train = dp_student.dp_cfg.num_train_timesteps
    assert all(0 <= k < n_train for k in schedule)
    flat = [sorted({int(v) for v in t.reshape(-1).tolist()}) for t in seen]
    assert all(len(v) == 1 for v in flat)  # one diffusion index per call, the whole batch
    assert [v[0] for v in flat] == schedule * 2  # both replans walk the scheduler's timesteps


def _requests(
    seeds: tuple[int, int], step: int, images: np.ndarray, proprio: np.ndarray
) -> list[batched.SlotRequest]:
    return [
        batched.SlotRequest(slot, batched.NoiseKey(0, seed, step), images, proprio)
        for slot, seed in enumerate(seeds)
    ]


def test_shared_dp_slots_differing_only_in_episode_seed(
    dp_student: Any, fixed_noise: torch.Tensor
) -> None:
    del fixed_noise
    policy = batched.SharedDPPolicy(dp_student)
    hooks = _Hooks(dp_student.net.encoder, dp_student.net.unet)
    images, proprio = _window()
    seeds = (11, 99_999)
    try:
        for slot, seed in enumerate(seeds):
            policy.reset_slot(slot, seed)
        outs = []
        for t in range(2 * TA):
            outs.append(policy.act_batch(_requests(seeds, t, images, proprio)))
    finally:
        hooks.remove()
    assert all(np.array_equal(o[0], o[1]) for o in outs)  # the two slots act alike
    assert all(np.array_equal(outs[i], outs[TA + i]) for i in range(TA))  # replan t=0 vs t=Ta
    # Inside every hooked call the two batch rows are identical.
    for _, tensors in hooks.calls:
        for x in tensors:
            if x.ndim >= 1 and x.shape[0] == 2:
                assert torch.equal(x[0], x[1])


def test_shared_bc_slots_differing_only_in_episode_seed() -> None:
    torch.set_num_threads(1)
    student = build_dummy("bc", 0, "cpu")
    policy = batched.SharedBCPolicy(student, run_seed=0)
    hooks = _Hooks(student.net.encoder, student.net)
    images, proprio = _window()
    seeds = (11, 99_999)
    try:
        for slot, seed in enumerate(seeds):
            policy.reset_slot(slot, seed)
        outs = [policy.act_batch(_requests(seeds, t, images, proprio)) for t in range(3)]
    finally:
        hooks.remove()
    assert all(np.array_equal(o[0], o[1]) for o in outs)
    assert all(np.array_equal(outs[0], o) for o in outs)  # nor on the step
    per_step = len(hooks.calls) // 3
    assert _same_calls(hooks.calls[:per_step], hooks.calls[per_step : 2 * per_step])
    for _, tensors in hooks.calls:
        for x in tensors:
            if x.ndim >= 1 and x.shape[0] == 2:
                assert torch.equal(x[0], x[1])


# --------------------------------------------------------------------------- 3. obligation 1


def test_noise_key_and_slot_request_fields_are_exact() -> None:
    import dataclasses

    assert batched.NoiseKey._fields == ("run_seed", "episode_seed", "step")
    assert tuple(f.name for f in dataclasses.fields(batched.SlotRequest)) == (
        "slot",
        "key",
        "images",
        "proprio",
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names = []
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(cur.name)
        cur = parents.get(cur)
    return ".".join(reversed(names))


def _context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Classify how an attribute read is used: the call it feeds, a tuple, a comparison, ..."""
    parent = parents[node]
    if isinstance(parent, ast.Call) and node in parent.args:
        func = parent.func
        return "call:" + (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "?")
        )
    if isinstance(parent, ast.Tuple):
        return "tuple"
    return type(parent).__name__.lower()


def _reads(path: Path | str, match: Any) -> Iterator[tuple[str, str, str]]:
    source = path.read_text(encoding="utf-8") if isinstance(path, Path) else path
    tree = ast.parse(source)
    parents = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and match(node):
            yield _enclosing(node, parents), node.attr, _context(node, parents)


def _is_key_field(node: ast.Attribute) -> bool:
    return (
        node.attr in {"step", "episode_seed"}
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "key"
    )


def test_key_step_and_episode_seed_are_read_only_at_permitted_sites() -> None:
    sites = Counter(_reads(Path(batched_mod.__file__), _is_key_field))
    allowed_contexts = {
        # Slot bookkeeping: the step must run 0, 1, 2, ... (compare, message, next step).
        ("_SlotClock.advance", "step"): {"compare", "formattedvalue", "binop"},
        # chunking: whether this slot replans.
        ("SharedDPPolicy.act_batch", "step"): {"call:is_replan_step", "tuple"},
        # The generator seed tuple (run_seed, episode_seed, step) -> noise_seed.
        ("SharedDPPolicy.act_batch", "episode_seed"): {"tuple"},
        # The step-0 reset trigger.
        ("_serve", "step"): {"compare"},
        ("_serve", "episode_seed"): {"call:reset_slot"},
    }
    for (where, attr, context), n in sites.items():
        assert context in allowed_contexts.get((where, attr), set()), (where, attr, context, n)
    # Each permitted use is present exactly where expected (a move would show up here).
    assert sites[("SharedDPPolicy.act_batch", "step", "call:is_replan_step")] == 1
    assert sites[("SharedDPPolicy.act_batch", "step", "tuple")] == 1
    assert sites[("SharedDPPolicy.act_batch", "episode_seed", "tuple")] == 1
    assert sites[("_serve", "step", "compare")] == 1
    assert sites[("_serve", "episode_seed", "call:reset_slot")] == 1


def _seed_tuples(path: Path, cls: str) -> list[list[str]]:
    """Every tuple in class ``cls`` that contains a clock/seed attribute, as attribute names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = _parents(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and _enclosing(node, parents).startswith(cls):
            names = [e.attr for e in node.elts if isinstance(e, ast.Attribute)]
            if {"_t", "step", "_episode_seed", "episode_seed"} & set(names):
                out.append(names)
    return out


def test_seed_tuples_are_run_seed_episode_seed_step() -> None:
    assert _seed_tuples(Path(batched_mod.__file__), "SharedDPPolicy") == [
        ["run_seed", "episode_seed", "step"]
    ]
    assert _seed_tuples(Path(dpp.__file__), "DPStudent") == [["run_seed", "_episode_seed", "_t"]]


def test_dp_student_clock_is_read_only_at_permitted_sites() -> None:
    def match(node: ast.Attribute) -> bool:
        return (
            node.attr in {"_t", "_episode_seed"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Load)
        )

    sites = Counter(
        (where, attr, ctx)
        for where, attr, ctx in _reads(Path(dpp.__file__), match)
        if where.startswith("DPStudent")
    )
    allowed = {
        ("DPStudent.act_vision", "_t", "call:is_replan_step"),
        ("DPStudent.act_vision", "_t", "tuple"),
        ("DPStudent.act_vision", "_episode_seed", "tuple"),
        # ``if self._episode_seed is None: raise`` (reset-before-act check).
        ("DPStudent.act_vision", "_episode_seed", "compare"),
        # Temporal ensembling (P6-D1 item 1): chunking's ensembling indices, P2-D3 item 4 (b).
        ("DPStudent.act_vision", "_t", "call:ensemble_sources"),
    }
    assert set(sites) <= allowed, sorted(set(sites) - allowed)
    assert sites[("DPStudent.act_vision", "_t", "call:is_replan_step")] == 1
    assert sites[("DPStudent.act_vision", "_t", "call:ensemble_sources")] == 1
    assert sites[("DPStudent.act_vision", "_t", "tuple")] == 1
    # Writes (reset to 0, += 1) are bookkeeping; only these two methods may write the clock.
    tree = ast.parse(Path(dpp.__file__).read_text(encoding="utf-8"))
    parents = _parents(tree)
    writers = {
        _enclosing(n, parents)
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "_t" and isinstance(n.ctx, ast.Store)
    }
    assert writers == {"DPStudent.__init__", "DPStudent._reset_episode", "DPStudent.act_vision"}


# --------------------------------------------------------------------------- negative controls


def test_negative_control_seed_changes_actions_without_fixed_noise(dp_student: Any) -> None:
    """The invariance comparison has power: with the real noise key the seed does matter."""
    images, proprio = _window()
    actions = {}
    for episode_seed in (11, 99_999):
        dp_student.reset(episode_seed)
        actions[episode_seed] = dp_student.act_vision(images, proprio)
    assert not np.array_equal(actions[11], actions[99_999])


def test_negative_control_hooks_see_a_planted_clock_feature(dp_student: Any) -> None:
    """A forward that received t would differ between replans: the hooks would catch it."""
    hooks = _Hooks(dp_student.net.encoder)
    images, proprio = _window()
    try:
        for t in (0, TA):
            planted = proprio.copy()
            planted[:, 0] = float(t)  # what leaking the clock into the proprio would look like
            dp_student.net.encoder(*dp_student._to_device(images[None], planted[None]))
    finally:
        hooks.remove()
    assert not _same_calls(hooks.calls[:1], hooks.calls[1:])


PLANTED_SOURCE = """
class SharedDPPolicy:
    def act_batch(self, requests):
        feats = [r.key.step / 360.0 for r in requests]
        return self.student.net(feats, cond=requests[0].key.episode_seed)
"""


def test_negative_control_ast_scan_flags_planted_reads() -> None:
    sites = set(_reads(PLANTED_SOURCE, _is_key_field))
    assert ("SharedDPPolicy.act_batch", "step", "binop") in sites
    assert ("SharedDPPolicy.act_batch", "episode_seed", "keyword") in sites
