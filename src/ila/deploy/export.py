"""ONNX export of ``bc_dagger`` and ``dp`` (Phase 9; P3-D1 §6 H5, §7; P9-D1).

Source checkpoints: seed 0's pinned selection of each arm. :func:`resolve_pinned` resolves
(arm, seed) through :func:`ila.eval.arms.resolve_checkpoint` (the committed
``selection.json``), requires that file's SHA-256 to equal its pin in
``results/e08/selection_pins.json`` (the pins Phase 8 flew), and re-hashes the checkpoint bytes
(:func:`ila.eval.arms.verify_checkpoint_bytes`) **before** anything is exported. EMA weights,
eval mode (centre 88 px crop), fp32.

Graphs (opset 17, batch axis dynamic, no RNG op; inputs and outputs keep these names and dtypes
in both precisions):

* ``bc_dagger.onnx``: ``images uint8[B, To, 96, 96, 3]`` (RGB, oldest frame first),
  ``proprio float32[B, To, 13]`` (the contract's proprio, P8 units) -> ``action float32[B, 3]``
  (normalised world-frame velocity setpoint in ``[-1, 1]``, ``tanh`` inside the graph; the env
  scales by v_max = 1.5 m/s model scale). Encoder and head in one graph; the ``/255`` scaling,
  the centre crop and the proprio standardisation are inside it.
* ``dp_encoder.onnx``: ``images``, ``proprio`` -> ``global_cond float32[B, 256]``.
* ``dp_denoise_step.onnx``: ``sample float32[B, Tp, 3]`` (the noisy chunk ``x_k``, normalised
  actions), ``timestep int64[B]`` (the diffusion index ``k``, the only integer a network sees,
  P2-D3 item 4), ``global_cond`` -> ``eps float32[B, Tp, 3]``.
* ``dp_K10.onnx``: ``global_cond``, ``noise float32[B, Tp, 3]`` (the initial noise ``x_K``,
  drawn by the caller from the per-episode generator, so the graph is deterministic) ->
  ``chunk float32[B, Tp, 3]``: DDIM, ``trailing`` spacing, eta = 0, K = 10 steps
  (``k = 99, 89, ..., 9``), unrolled, each step diffusers' ``DDIMScheduler.step`` arithmetic
  with its coefficients as constants (``clip_sample`` to [-1, 1] inside each step), and the
  final clip to [-1, 1] inside the graph. Chunk index ``j`` is control step ``t - To + 1 + j``
  (1/30 s model scale each; ``Tp = 16`` is 0.533 s model, 2.67 s full scale).

``fp16`` graphs (``*_fp16.onnx``) are the fp32 graphs converted by ONNX Runtime's
``float16.convert_float_to_float16`` with ``keep_io_types=True``. Every graph carries ONNX
Runtime's symbolic shape inference (the TensorRT EP needs a shape on every subgraph input).

The unrolled module :class:`UnrolledDDIM` is checked against ``ila``'s own sampler
(:func:`ila.policies.dp.policy.ddim_sample` with the diffusers scheduler) in torch fp32 on the
CPU before export (:func:`unrolled_matches_sampler`); a mismatch refuses the export.

``meta.json`` (format :data:`META_FORMAT`) records every graph's SHA-256, I/O and opset, the
source checkpoints' and ``selection.json``'s SHA-256, the DDIM constants, versions and the
``ila`` commit. Simulation only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import torch
from numpy.typing import NDArray
from torch import nn

from ila import p8_adapter
from ila.deploy import r1_parity
from ila.deploy.settings import BenchSettings, Precision
from ila.eval import arms as ev_arms
from ila.policies.bc import BCNet
from ila.policies.dp.policy import DiffusionPolicyNet, DPConfig, ddim_sample, ddim_scheduler
from ila.policies.encoder import EncoderConfig
from ila.policies.registry import build_net_from_config, read_checkpoint

__all__ = [
    "GRAPH_FILES",
    "META_FILE",
    "META_FORMAT",
    "BCGraph",
    "DenoiseStepGraph",
    "EncoderGraph",
    "ExportError",
    "LoadedArm",
    "UnrolledDDIM",
    "dp_config_from_meta",
    "export_all",
    "export_graph",
    "git_state",
    "graph_path",
    "load_arm",
    "load_meta",
    "resolve_pinned",
    "sha256_file",
    "to_fp16",
    "unrolled_matches_sampler",
    "verify_graphs",
]

#: ``meta.json`` format tag.
META_FORMAT: Final[str] = "ila-deploy-onnx/1"
META_FILE: Final[str] = "meta.json"

#: Graph key -> fp32 file name (``*_fp16.onnx`` for fp16).
GRAPH_FILES: Final[Mapping[str, str]] = {
    "bc": "bc_dagger.onnx",
    "encoder": "dp_encoder.onnx",
    "step": "dp_denoise_step.onnx",
    "unrolled": "dp_K10.onnx",
}

#: Graph key -> (input names, output names).
GRAPH_IO: Final[Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "bc": (("images", "proprio"), ("action",)),
    "encoder": (("images", "proprio"), ("global_cond",)),
    "step": (("sample", "timestep", "global_cond"), ("eps",)),
    "unrolled": (("global_cond", "noise"), ("chunk",)),
}

#: Torch fp32 CPU tolerance of the unrolled module against ``ddim_sample`` (same arithmetic).
UNROLL_TOL: Final[float] = 1e-6


class ExportError(RuntimeError):
    """A pin, a checkpoint hash, a graph check or the unrolled-sampler check failed."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file (hex)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    """``ila`` commit, dirty flag and dirty paths of this checkout (``git status --porcelain``)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=p8_adapter.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout

    dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=all").splitlines() if ln]
    return {
        "ila_commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(dirty),
        "dirty_paths": dirty,
    }


# --------------------------------------------------------------------------- checkpoints


def resolve_pinned(
    arm: str, seed: int, pins_path: Path, *, tag: str = "core"
) -> ev_arms.Checkpoint:
    """Resolve (arm, seed) to its pinned, byte-verified selected checkpoint.

    Args:
        arm: ``bc_dagger`` or ``dp`` (Phase 9); any ``dp_*`` ablation for the Phase 10
            sensitivity stage (:mod:`ila.deploy.sensitivity`).
        seed: Training seed (0).
        pins_path: ``results/e08/selection_pins.json``.
        tag: The arm's matrix tag (``core``; ``ablation`` for the ablations), checked against
            the tag its ``selection.json`` records.

    Returns:
        The checkpoint record (path, SHA-256, ``selection.json`` and its SHA-256).

    Raises:
        ExportError: If the pin is missing or differs from the committed ``selection.json``.
        ila.eval.arms.ArmResolutionError: If the record or the checkpoint bytes do not match.
    """
    inference = ev_arms.Inference(8, 10, False) if ev_arms.is_diffusion(arm) else None
    spec = ev_arms.ArmSpec(
        arm=arm,
        tag=tag,
        seeds=(int(seed),),
        teacher=ev_arms.expected_teacher(arm),
        obs_variant=ev_arms.expected_obs_variant(arm),
        inference=inference,
    )
    ckpt = ev_arms.resolve_checkpoint(spec, int(seed))
    pins: Mapping[str, str] = json.loads(Path(pins_path).read_text(encoding="utf-8"))["pins"]
    key = f"{arm}/s{seed}"
    if pins.get(key) != ckpt.selection_json_sha256:
        raise ExportError(
            f"{key}: selection.json SHA-256 {ckpt.selection_json_sha256} != pin {pins.get(key)}"
        )
    ev_arms.verify_checkpoint_bytes(ckpt)
    return ckpt


@dataclass(frozen=True)
class LoadedArm:
    """A pinned checkpoint and its EMA network (CPU, eval mode, fp32).

    Attributes:
        arm: Arm.
        seed: Training seed (also ``run_seed`` of every noise key).
        ckpt: The resolved checkpoint.
        net: The network.
        encoder_cfg: Its encoder config (``To``, frame size, proprio width).
        obs_variant: The contract variant it flies (``default``).
    """

    arm: str
    seed: int
    ckpt: ev_arms.Checkpoint
    net: BCNet | DiffusionPolicyNet
    encoder_cfg: EncoderConfig
    obs_variant: str


def load_arm(arm: str, seed: int, pins_path: Path, *, tag: str = "core") -> LoadedArm:
    """:func:`resolve_pinned`, then build the EMA network from the checkpoint's own config.

    A diffusion arm (``dp``, ``dp_*``) must build a :class:`DiffusionPolicyNet`, any other a
    :class:`BCNet`.

    Raises:
        ExportError: If the checkpoint is not the arm's family.
    """
    ckpt = resolve_pinned(arm, seed, pins_path, tag=tag)
    raw = read_checkpoint(ckpt.path)
    net = build_net_from_config(
        raw["kind"], raw["config"], int(raw["run_seed"]), str(raw["obs_variant"])
    )
    net.load_state_dict(raw["ema"])
    net.eval()
    want = DiffusionPolicyNet if ev_arms.is_diffusion(arm) else BCNet
    if not isinstance(net, want):
        raise ExportError(f"{arm}: checkpoint builds {type(net).__name__}, not {want.__name__}")
    return LoadedArm(arm, int(seed), ckpt, net, net.encoder.cfg, str(raw["obs_variant"]))


# --------------------------------------------------------------------------- graph modules


class BCGraph(nn.Module):
    """``(images, proprio) -> action``: the BC encoder and head (eval mode)."""

    def __init__(self, net: BCNet) -> None:
        """Hold the network."""
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """``uint8[B, To, 96, 96, 3]``, ``float32[B, To, D]`` -> ``float32[B, 3]`` in [-1, 1]."""
        out: torch.Tensor = self.net(images, proprio)
        return out


class EncoderGraph(nn.Module):
    """``(images, proprio) -> global_cond``: the ``dp`` encoder (eval mode)."""

    def __init__(self, net: DiffusionPolicyNet) -> None:
        """Hold the network's encoder."""
        super().__init__()
        self.encoder = net.encoder

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """``uint8[B, To, 96, 96, 3]``, ``float32[B, To, D]`` -> ``float32[B, 256]``."""
        out: torch.Tensor = self.encoder(images, proprio)
        return out


class DenoiseStepGraph(nn.Module):
    """``(sample, timestep, global_cond) -> eps``: one U-Net call."""

    def __init__(self, net: DiffusionPolicyNet) -> None:
        """Hold the network's U-Net."""
        super().__init__()
        self.unet = net.unet

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """``x_k float32[B, Tp, 3]``, ``k int64[B]``, ``float32[B, 256]`` -> ``eps [B, Tp, 3]``."""
        out: torch.Tensor = self.unet(sample, timestep, global_cond)
        return out


class UnrolledDDIM(nn.Module):
    """``(global_cond, noise) -> chunk``: K DDIM steps unrolled, diffusers' arithmetic.

    Per step ``k`` with ``k_prev = k - 100 // K`` (``set_alpha_to_one``: ``abar_prev = 1`` when
    ``k_prev < 0``): ``x0 = clamp((x - sqrt(1 - abar_k) eps) / sqrt(abar_k), -1, 1)``;
    ``x <- sqrt(abar_prev) x0 + sqrt(1 - abar_prev - 0) eps``; finally ``clamp(x, -1, 1)``. The
    coefficients are the float32 0-d tensors diffusers computes, held as buffers.
    """

    def __init__(self, net: DiffusionPolicyNet, num_inference_steps: int) -> None:
        """Precompute the coefficients from ``ila``'s scheduler (:func:`ddim_scheduler`).

        Args:
            net: The policy network (its U-Net and config).
            num_inference_steps: ``K``.
        """
        super().__init__()
        self.unet = net.unet
        sched = ddim_scheduler(net.cfg, int(num_inference_steps))
        n_train = int(sched.config.num_train_timesteps)
        self.timesteps: tuple[int, ...] = tuple(int(t) for t in sched.timesteps)
        coeffs = []
        for t in self.timesteps:
            prev = t - n_train // int(sched.num_inference_steps)
            a_t = sched.alphas_cumprod[t]
            a_prev = sched.alphas_cumprod[prev] if prev >= 0 else sched.final_alpha_cumprod
            beta_t = 1 - a_t
            std = 0.0 * sched._get_variance(t, prev) ** 0.5
            coeffs.append(
                torch.stack(
                    [
                        beta_t**0.5,
                        a_t**0.5,
                        a_prev**0.5,
                        (1 - a_prev - std**2) ** 0.5,
                    ]
                ).to(torch.float32)
            )
        self.register_buffer("coeffs", torch.stack(coeffs))
        self.register_buffer("steps", torch.tensor(self.timesteps, dtype=torch.long))
        self.clip_range = float(sched.config.clip_sample_range)
        self.clip_sample = bool(sched.config.clip_sample)

    coeffs: torch.Tensor
    steps: torch.Tensor

    def forward(self, global_cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """``float32[B, 256]``, ``x_K float32[B, Tp, 3]`` -> chunk ``float32[B, Tp, 3]``."""
        x = noise.to(torch.float32)
        b = x.shape[0]
        for i in range(len(self.timesteps)):
            kk = self.steps[i : i + 1].expand(b)
            eps = self.unet(x, kk, global_cond)
            c = self.coeffs[i]
            x0 = (x - c[0] * eps) / c[1]
            if self.clip_sample:
                x0 = x0.clamp(-self.clip_range, self.clip_range)
            x = c[2] * x0 + c[3] * eps
        return x.clamp(-1.0, 1.0)


def unrolled_matches_sampler(
    net: DiffusionPolicyNet, k: int, cond: torch.Tensor, noise: torch.Tensor
) -> dict[str, Any]:
    """Compare :class:`UnrolledDDIM` with :func:`ddim_sample` (torch fp32, CPU).

    Returns:
        ``{"max_abs", "bit_identical", "ok"}``; ``ok`` iff max-abs <= :data:`UNROLL_TOL`.
    """
    with torch.inference_mode():
        a = UnrolledDDIM(net, k).eval()(cond, noise)

        def eps_fn(x: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
            return net.predict_noise(x, kk, cond)

        b = ddim_sample(eps_fn, noise, net.ddim(k))
    gap = float((a.double() - b.double()).abs().max())
    return {"max_abs": gap, "bit_identical": bool(torch.equal(a, b)), "ok": gap <= UNROLL_TOL}


# --------------------------------------------------------------------------- export


def graph_path(out_dir: Path, key: str, precision: Precision) -> Path:
    """``<out_dir>/<file>`` for a graph key and precision (``*_fp16.onnx`` for fp16)."""
    name = GRAPH_FILES[key]
    return Path(out_dir) / (name if precision == "fp32" else name.replace(".onnx", "_fp16.onnx"))


def _example_inputs(cfg: EncoderConfig, key: str, tp: int, a: int) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (1, cfg.n_obs_steps, cfg.input_px, cfg.input_px, 3), generator=g)
    proprio = torch.randn((1, cfg.n_obs_steps, cfg.proprio_dim), generator=g)
    cond = torch.randn((1, cfg.feature_dim), generator=g)
    sample = torch.randn((1, tp, a), generator=g)
    if key in ("bc", "encoder"):
        return images.to(torch.uint8), proprio
    if key == "step":
        return sample, torch.full((1,), 99, dtype=torch.long), cond
    return cond, sample


def export_graph(
    module: nn.Module, example: tuple[torch.Tensor, ...], key: str, path: Path, opset: int
) -> None:
    """Export one module with the batch axis dynamic (TorchScript exporter, constant folding).

    Raises:
        ExportError: If the written graph is not opset ``opset`` or holds an RNG op.
    """
    ins, outs = GRAPH_IO[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(False), torch.no_grad():
        torch.onnx.export(
            module.eval(),
            example,
            str(path),
            input_names=list(ins),
            output_names=list(outs),
            opset_version=opset,
            dynamic_axes={n: {0: "batch"} for n in (*ins, *outs)},
            do_constant_folding=True,
            dynamo=False,
        )
    _infer_shapes(path)
    _check_graph(path, opset)


def _infer_shapes(path: Path) -> None:
    """Store symbolic shapes on every tensor (ORT's tool; the TensorRT EP needs them)."""
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    model = SymbolicShapeInference.infer_shapes(onnx.load(str(path)), auto_merge=True)
    onnx.save(model, str(path))


def _check_graph(path: Path, opset: int) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opsets = {o.domain or "ai.onnx": int(o.version) for o in model.opset_import}
    rng = r1_parity.rng_ops(model)
    if opsets.get("ai.onnx") != opset or rng:
        raise ExportError(f"{path.name}: opsets {opsets}, RNG ops {rng}")
    return {"opsets": opsets, "ir_version": int(model.ir_version), "rng_ops": rng}


def to_fp16(src: Path, dst: Path, opset: int) -> None:
    """Convert an fp32 graph to fp16 (``keep_io_types=True``: I/O stay as in fp32)."""
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    model = onnx.load(str(src))
    half = OnnxModel(convert_float_to_float16(model, keep_io_types=True))
    # The converter appends its input casts after their consumers; ORT's own sort fixes it.
    half.topological_sort()
    onnx.save(half.model, str(dst))
    _infer_shapes(dst)
    _check_graph(dst, opset)


def _io_record(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))

    def one(v: Any) -> dict[str, Any]:
        t = v.type.tensor_type
        dims = [d.dim_param or int(d.dim_value) for d in t.shape.dim]
        return {
            "name": v.name,
            "elem_type": onnx.TensorProto.DataType.Name(t.elem_type),
            "shape": dims,
        }

    return {
        "inputs": [one(v) for v in model.graph.input],
        "outputs": [one(v) for v in model.graph.output],
    }


def _modules(arms: Mapping[str, LoadedArm], k: int) -> dict[str, nn.Module]:
    bc, dp = arms["bc_dagger"].net, arms["dp"].net
    assert isinstance(bc, BCNet) and isinstance(dp, DiffusionPolicyNet)
    return {
        "bc": BCGraph(bc),
        "encoder": EncoderGraph(dp),
        "step": DenoiseStepGraph(dp),
        "unrolled": UnrolledDDIM(dp, k),
    }


def export_all(settings: BenchSettings, out_dir: Path | None = None) -> dict[str, Any]:
    """Export every graph in every precision and write ``meta.json`` (module docstring).

    Args:
        settings: Bench settings.
        out_dir: Override of ``settings.out_dir`` (tests).

    Returns:
        The ``meta.json`` record.

    Raises:
        ExportError: On a pin / hash mismatch, a graph check or the unrolled-sampler check.
    """
    out = Path(out_dir) if out_dir is not None else settings.out_dir / "graphs"
    arms = {a: load_arm(a, s, settings.selection_pins) for a, s in settings.arm_seeds.items()}
    return export_loaded(arms, settings, out)


def export_loaded(
    arms: Mapping[str, LoadedArm], settings: BenchSettings, out: Path
) -> dict[str, Any]:
    """Export from already-loaded arms (:func:`export_all`; tests pass dummy networks)."""
    dp = arms["dp"].net
    assert isinstance(dp, DiffusionPolicyNet)
    tp, a = dp.cfg.horizon, dp.cfg.action_dim
    torch.set_num_threads(max(1, torch.get_num_threads()))
    g = torch.Generator().manual_seed(1)
    cond = torch.randn((4, dp.encoder.cfg.feature_dim), generator=g)
    noise = torch.randn((4, tp, a), generator=g)
    unroll = unrolled_matches_sampler(dp, settings.dp_k, cond, noise)
    if not unroll["ok"]:
        raise ExportError(f"unrolled DDIM differs from ddim_sample by {unroll['max_abs']:.3g}")
    modules = _modules(arms, settings.dp_k)
    graphs: dict[str, Any] = {}
    for key, module in modules.items():
        cfg = arms["bc_dagger" if key == "bc" else "dp"].encoder_cfg
        example = _example_inputs(cfg, key, tp, a)
        p32 = graph_path(out, key, "fp32")
        export_graph(module, example, key, p32, settings.opset)
        for prec in settings.precisions:
            p = graph_path(out, key, prec)
            if prec == "fp16":
                to_fp16(p32, p, settings.opset)
            graphs[p.name] = {
                "key": key,
                "precision": prec,
                "arm": "bc_dagger" if key == "bc" else "dp",
                "sha256": sha256_file(p),
                **_check_graph(p, settings.opset),
                **_io_record(p),
            }
    sched = ddim_scheduler(dp.cfg, settings.dp_k)
    meta: dict[str, Any] = {
        "format": META_FORMAT,
        "graphs": dict(sorted(graphs.items())),
        "sources": {
            name: {
                "arm": la.arm,
                "seed": la.seed,
                "checkpoint": ev_arms._rel(la.ckpt.path),
                "checkpoint_sha256": la.ckpt.sha256,
                "global_step": la.ckpt.global_step,
                "selection_json": la.ckpt.selection_json,
                "selection_json_sha256": la.ckpt.selection_json_sha256,
                "weights": "ema",
                "obs_variant": la.obs_variant,
            }
            for name, la in arms.items()
        },
        "selection_pins": {
            "file": ev_arms._rel(settings.selection_pins),
            "sha256": sha256_file(settings.selection_pins)
            if Path(settings.selection_pins).is_file()
            else None,
        },
        "dp": {
            "n_obs_steps": dp.cfg.n_obs_steps,
            "horizon": tp,
            "n_action_steps": settings.dp_ta,
            "action_dim": a,
            "cond_dim": dp.encoder.cfg.feature_dim,
            "K": settings.dp_k,
            "timestep_spacing": dp.cfg.timestep_spacing,
            "timesteps": [int(t) for t in sched.timesteps],
            "num_train_timesteps": dp.cfg.num_train_timesteps,
            "beta_schedule": dp.cfg.beta_schedule,
            "clip_sample": dp.cfg.clip_sample,
            "eta": 0.0,
            "unrolled_vs_ddim_sample_torch_cpu": unroll,
        },
        "opset": settings.opset,
        "fp16_conversion": "onnxruntime.transformers.float16.convert_float_to_float16"
        "(keep_io_types=True)",
        "exporter": "torch.onnx.export(dynamo=False, do_constant_folding=True)",
        "versions": {
            "torch": str(torch.__version__),
            "onnx": str(onnx.__version__),
            "onnxruntime": _ort_version(),
            "python": platform.python_version(),
        },
        "git": git_state(),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", "utf-8")
    return meta


def _ort_version() -> str:
    import onnxruntime as ort

    return str(ort.__version__)


def load_meta(graph_dir: Path) -> dict[str, Any]:
    """Read ``<graph_dir>/meta.json``.

    Raises:
        ExportError: If it is missing or of another format.
    """
    path = Path(graph_dir) / META_FILE
    if not path.is_file():
        raise ExportError(f"{path} does not exist (run the export first)")
    meta: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if meta.get("format") != META_FORMAT:
        raise ExportError(f"{path}: format {meta.get('format')!r} != {META_FORMAT!r}")
    return meta


def dp_config_from_meta(meta: Mapping[str, Any]) -> DPConfig:
    """``dp``'s sampler config from ``meta.json`` (an ORT backend never loads the checkpoint).

    Raises:
        ExportError: If the recorded timesteps differ from the rebuilt scheduler's.
    """
    d = meta["dp"]
    cfg = DPConfig(
        n_obs_steps=int(d["n_obs_steps"]),
        horizon=int(d["horizon"]),
        n_action_steps=int(d["n_action_steps"]),
        action_dim=int(d["action_dim"]),
        num_train_timesteps=int(d["num_train_timesteps"]),
        beta_schedule=str(d["beta_schedule"]),
        clip_sample=bool(d["clip_sample"]),
        num_inference_steps=int(d["K"]),
        timestep_spacing=str(d["timestep_spacing"]),
    )
    got = [int(t) for t in ddim_scheduler(cfg, int(d["K"])).timesteps]
    if got != [int(t) for t in d["timesteps"]]:
        raise ExportError(f"meta.json timesteps {d['timesteps']} != rebuilt {got}")
    return cfg


def verify_graphs(graph_dir: Path, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Re-hash the graphs against ``meta.json``.

    Args:
        graph_dir: The graph directory.
        names: Graph file names to check (all when ``None``).

    Returns:
        ``meta.json``.

    Raises:
        ExportError: On a missing graph or a hash mismatch.
    """
    meta = load_meta(graph_dir)
    for name, rec in meta["graphs"].items():
        if names is not None and name not in names:
            continue
        path = Path(graph_dir) / name
        if not path.is_file():
            raise ExportError(f"{path} does not exist")
        got = sha256_file(path)
        if got != rec["sha256"]:
            raise ExportError(f"{path}: SHA-256 {got[:12]}... != meta.json {rec['sha256'][:12]}...")
    return meta


def as_numpy(x: torch.Tensor) -> NDArray[np.float32]:
    """Detach to a float32 numpy array on the host."""
    out: NDArray[np.float32] = x.detach().cpu().numpy().astype(np.float32)
    return out
