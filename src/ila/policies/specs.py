"""Controller and network specifications, re-derived from committed files (Phase 11).

Writes ``results/readme/model_specs.json``: one record per student arm with a config under
``configs/train/`` (the seven core arms, the eleven ablations under ``configs/train/ablations/``,
and ``R1_lerobot_dp``, which ``ila`` does not build), and one per P8 controller this project
distils from or prints beside every student (``ppo`` seed 0, ``pid_feedforward``,
``pid_track_descend``, ``gated``, ``oracle_gated``). Every number traces to a file:

* a student's architecture is the network **built** on the CPU from its config with the very
  constructor training uses (:func:`ila.policies.registry.build_net_from_config`, seed 0, no
  checkpoint); parameter counts are summed over that module's ``nn.Parameter`` tensors
  (buffers -- the fixed proprio/target normalisers -- are counted apart);
* training and sampler settings are the config's, parsed by :class:`ila.train.config.TrainConfig`
  (the training loop's parser) and :class:`ila.policies.dp.policy.DPConfig`;
* P8's facts come only through :mod:`ila.p8_adapter` (:func:`ila.p8_adapter.ppo_teacher_facts`,
  :func:`ila.p8_adapter.controller_facts`, :func:`ila.p8_adapter.obs_fields`,
  :func:`ila.p8_adapter.load_eval_configs`); every P8 file is cited relative to P8's root at
  P8's pinned commit. The PPO actor/critic parameter counts are derived from P8's config
  (``net_arch`` for both ``pi`` and ``vf``, as ``rld.rl.train`` builds SB3's ``MlpPolicy``).

Checkpoint cross-checks (workstation only). Where the pinned seed-0 checkpoint exists locally
(``results/e08/selection_pins.json`` -> the committed ``selection.json`` -> the checkpoint under
``artifacts/runs/``, byte-verified), its EMA and raw state dicts are loaded on the CPU and every
tensor's shape, and the parameter total, must equal the freshly built network's; P8's PPO
``final/`` checkpoint is loaded through :func:`ila.p8_adapter.teacher` and checked against the
derived SB3 shapes. A machine without the checkpoints (CI) cannot re-run these; the committed
cross-check records are then carried over unchanged, after checking that they describe the same
config SHA-256 and parameter total as the re-derived record (so a config change can never hide
behind a stale cross-check). The output is deterministic (sorted keys, fixed indentation, no
timestamps, no ``ila`` commit), so ``--check`` is byte-stable.

Units: image sizes in px; horizons ``To``/``Tp``/``Ta`` in control steps of 1/30 s model scale
(lambda = 1/25: 1/6 s full scale); training lengths in optimiser steps; learning rates per
optimiser step; P8 gains in the units their key names carry (m, m/s, s model scale).
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from torch import nn

from ila import p8_adapter
from ila.policies.bc import BCHeadConfig
from ila.policies.dp.policy import DiffusionPolicyNet, DPConfig
from ila.policies.dp.unet1d import ConditionalResidualBlock1D
from ila.policies.encoder import VisionEncoder
from ila.policies.modular import ModularEstimator, ModularHeadConfig
from ila.policies.registry import (
    CAMERA_CONFIG,
    REGISTRY,
    REPO_ROOT,
    StudentEntry,
    build_net_from_config,
    read_checkpoint,
)
from ila.policies.state_mlp import StateMLP
from ila.train.config import ADAMW_BETAS, ADAMW_EPS, TrainConfig

__all__ = [
    "ABSENT",
    "CONTROLLERS",
    "OUTPUT",
    "SCHEMA",
    "CheckResult",
    "build_document",
    "check",
    "dumps",
    "student_arms",
    "write",
]

#: Format tag of the output file.
SCHEMA: Final[str] = "ila-model-specs/1"

#: The committed output.
OUTPUT: Final[Path] = REPO_ROOT / "results" / "readme" / "model_specs.json"

#: Selection pins Phase 8 flew (arm/s<seed> -> selection.json SHA-256).
PINS: Final[Path] = REPO_ROOT / "results" / "e08" / "selection_pins.json"

#: P8's checkpoint provenance (expected SHA-256 of the PPO seed-0 ``final/`` files).
P8_PROVENANCE: Final[Path] = REPO_ROOT / "results" / "p8_provenance.json"

#: Cross-check status recorded when no checkpoint was available when the file was generated.
ABSENT: Final[str] = "checkpoint_absent_when_generated"

#: Seed whose pinned checkpoint is cross-checked.
CROSSCHECK_SEED: Final[int] = 0

#: P8 registry controllers recorded, with their role in this project.
CONTROLLERS: Final[Mapping[str, str]] = {
    "pid_feedforward": "teacher T_pid (pid_1000 demos; *_pid ablations); baseline row",
    "gated": "teacher in T_mix (mix_1000 demos with T_ppo; *_mix ablations); baseline row",
    "pid_track_descend": "baseline row (naive)",
    "oracle_gated": "baseline row (privileged commit-timing oracle; never deployable)",
}

#: One-line structural descriptions of P8's controllers (their laws; numbers come from P8).
_STRUCTURE: Final[Mapping[str, str]] = {
    "pid_track_descend": (
        "Lateral PI on the pad-relative xy error (speed-capped, anti-windup clamp); descend at a "
        "constant rate once the lateral error is inside commit_radius_m (released beyond "
        "release_factor x radius), hold height otherwise. No deck feedforward. Stateful "
        "(integrator, commit latch)."
    ),
    "pid_feedforward": (
        "pid_track_descend's law plus k_ff x pad deck-point velocity (velocity + rel_velocity "
        "from the observation) on all three axes. Stateful (integrator, commit latch)."
    ),
    "gated": (
        "Hover hover_height_m above the pad (P on the height error, capped), commit to "
        "pid_feedforward's descent when dmf's quiescence rule holds on the CURRENT deck state "
        "over the trailing sustain window. Gains loaded from pid_feedforward.yaml. Stateful."
    ),
    "oracle_gated": (
        "gated's hover and commit, but the quiescence predicate is applied to the TRUE FUTURE "
        "deck states at the predicted touchdown time (privileged context). A commit-timing "
        "oracle, never a deployable result."
    ),
}


def _rel(path: Path) -> str:
    """Return ``path`` relative to this repository (POSIX), else as given."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    """``{"path": repo-relative, "sha256": ...}`` of one committed file."""
    return {"path": _rel(path), "sha256": _sha256(path)}


# --------------------------------------------------------------------------- parameter counts


def _numel(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of parameter elements (``requires_grad`` only if ``trainable_only``)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))


def _by_module(net: nn.Module) -> dict[str, int]:
    """Parameter counts of each top-level child and, for the encoder and U-Net, their children.

    The top-level entries sum to the total (asserted).
    """
    out: dict[str, int] = {}
    top = 0
    for name, child in net.named_children():
        n = _numel(child)
        out[name] = n
        top += n
        if name in ("encoder", "unet"):
            sub = 0
            for sub_name, grandchild in child.named_children():
                m = _numel(grandchild)
                out[f"{name}.{sub_name}"] = m
                sub += m
            if sub != n:
                raise AssertionError(f"{name}: children sum {sub} != {n}")
    direct = int(sum(p.numel() for p in net.parameters(recurse=False)))
    if top + direct != _numel(net):
        raise AssertionError(f"top-level children sum {top} + {direct} != {_numel(net)}")
    return out


def _parameters(net: nn.Module) -> dict[str, Any]:
    """Total, trainable and per-module parameter counts, and the buffer element count."""
    total = _numel(net)
    trainable = _numel(net, trainable_only=True)
    return {
        "total": total,
        "trainable": trainable,
        "by_module": _by_module(net),
        "buffers_elements": int(sum(b.numel() for b in net.buffers())),
        "buffers": sorted(name for name, _ in net.named_buffers()),
    }


def _shapes(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    """Tensor name -> shape."""
    return {k: [int(d) for d in v.shape] for k, v in state.items()}


# --------------------------------------------------------------------------- students


def student_arms() -> tuple[str, ...]:
    """Registered students in table order: core, then ablations, then the external reference."""
    order = {"core": 0, "ablation": 1, "reference_external": 2}
    return tuple(sorted(REGISTRY, key=lambda a: (order[REGISTRY[a].tag], list(REGISTRY).index(a))))


def _config_block(entry: StudentEntry, cfg: TrainConfig | None) -> dict[str, Any]:
    """Config path and SHA-256, its base and any referenced file (all repo-relative)."""
    block: dict[str, Any] = _file_ref(entry.config)
    if cfg is not None:
        others = [Path(p) for p, _ in cfg.files_sha256[1:]]
        block["also_read"] = [_file_ref(p) for p in others]
    return block


def _encoder_block(enc: VisionEncoder) -> dict[str, Any]:
    """The shared vision encoder, read from the built module."""
    c = enc.cfg
    with torch.no_grad():
        trunk_out = enc.trunk(torch.zeros(1, 3, c.crop_px, c.crop_px))
    linears = [m for m in enc.proprio_mlp if isinstance(m, nn.Linear)]
    widths = [int(m.in_features) for m in linears]
    widths.append(int(next(reversed(linears)).out_features))
    return {
        "trunk": c.trunk,
        "trunk_impl": "torchvision.models.resnet18(weights=None), conv1..layer4 (no pool, no fc)",
        "pretrained": c.pretrained,
        "norm": f"{c.norm}({c.groups} groups) replacing every BatchNorm",
        "norm_groups": c.groups,
        "trunk_output_shape_chw": [int(d) for d in trunk_out.shape[1:]],
        "spatial_softmax_keypoints": c.keypoints,
        "keypoint_projection": f"Conv2d 1x1 {int(trunk_out.shape[1])}->{c.keypoints}",
        "frame_feature_dim": c.frame_feature_dim,
        "proprio_normalisation": "fixed buffers proprio_mean/proprio_std (train-set mean/std)",
        "proprio_mlp_widths": widths,
        "proprio_mlp_activation": "relu between layers, none after the last",
        "proprio_mlp_shared_over_To": True,
        "proprio_feature_dim": c.proprio_feature_dim,
        "feature_dim": c.feature_dim,
        "feature_layout": "[keypoints(t-To+1..t) || proprio_mlp(t-To+1..t)], oldest first",
    }


def _vision_inputs(
    entry: StudentEntry, enc: VisionEncoder, raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Image and proprio inputs of a vision arm (contract by P8 field name)."""
    from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

    contract = load_contract(variant=entry.obs_variant)
    c = enc.cfg
    camera: Mapping[str, Any] = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))
    crop = raw["augment"]["crop"]
    if int(crop["size"]) != c.crop_px:
        raise AssertionError(f"{entry.name}: augment crop {crop['size']} != camera {c.crop_px}")
    return {
        "image_px": c.input_px,
        "image_channels": 3,
        "image_dtype": "uint8 RGB, scaled to [0, 1]; no ImageNet statistics",
        "camera_config": _file_ref(CAMERA_CONFIG),
        "camera_fov_vertical_deg": float(camera["camera"]["fov_vertical_deg"]),
        "crop_px": c.crop_px,
        "crop_train": str(crop["train"]),
        "crop_eval": str(crop["eval"]),
        "frames_To": c.n_obs_steps,
        "proprio_dim": c.proprio_dim,
        "proprio_fields": [
            {"name": n, "size": contract.layout[n][1]} for n in contract.student_proprio
        ],
        "obs_variant": entry.obs_variant,
        "contract": _file_ref(REPO_ROOT / "configs" / "obs" / "student.yaml"),
    }


def _training_block(cfg: TrainConfig) -> dict[str, Any]:
    """Optimiser, schedule, precision, EMA and data, from the parsed config."""
    raw = cfg.raw
    ema: dict[str, Any] = {"kind": cfg.ema.kind}
    if cfg.ema.kind == "constant":
        ema["decay"] = cfg.ema.decay
    else:
        ema.update(power=cfg.ema.power, max_value=cfg.ema.max_value, inv_gamma=cfg.ema.inv_gamma)
    loss = raw.get("loss", "mse_epsilon (ila.policies.dp.policy.ddpm_loss)")
    out: dict[str, Any] = {
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "optimizer": "adamw",
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
        "adamw_betas": list(ADAMW_BETAS),
        "adamw_eps": ADAMW_EPS,
        "schedule": cfg.optim.schedule,
        "warmup_steps": cfg.optim.warmup_steps,
        "precision": cfg.precision,
        "ema": ema,
        "evaluated_weights": "ema",
        "loss": loss if isinstance(loss, str) else dict(loss),
        "checkpoint_every_steps": cfg.checkpoint_every,
        "seeds": list(cfg.seeds),
        "data": {
            "collection": cfg.collection,
            "first_n_episodes": cfg.first_n,
            "success_filter": cfg.success_filter,
        },
    }
    if cfg.dart_sigma_rule is not None:
        out["data"]["dart_sigma_rule"] = cfg.dart_sigma_rule
    if cfg.iterations is not None:
        it = cfg.iterations
        out["iterations"] = {
            "iteration_0": "the same-seed bc run (its steps above), last-step EMA",
            "n": it.n,
            "episodes_added_per_iteration": it.add_per_iteration,
            "steps_per_iteration": it.steps_per_iteration,
            "warmup_steps_per_iteration": it.warmup_steps,
            "labelled_episodes_total": it.labelled_episodes_total,
        }
        out["total_steps"] = cfg.steps + it.n * it.steps_per_iteration
    else:
        out["total_steps"] = cfg.steps
    if cfg.dagger is not None:
        out["dagger"] = {
            "beta": list(cfg.dagger.beta),
            "rollout_weights": cfg.dagger.rollout_weights,
            "rollouts_success_filter": cfg.dagger.rollouts_success_filter,
        }
    return out


def _bc_head(raw: Mapping[str, Any], feature_dim: int) -> dict[str, Any]:
    head = BCHeadConfig.from_mapping(raw)
    return {
        "type": "mlp",
        "widths": [feature_dim, *head.hidden, head.out],
        "activation": "relu",
        "output": head.activation,
    }


def _dp_block(net: DiffusionPolicyNet) -> tuple[dict[str, Any], dict[str, Any]]:
    """(``head``: the U-Net; ``diffusion``: the noise schedule and sampler)."""
    cfg: DPConfig = net.cfg
    u = cfg.unet
    first = net.unet.down_modules[0]
    assert isinstance(first, nn.ModuleList)
    block = first[0]
    assert isinstance(block, ConditionalResidualBlock1D)
    film = block.cond_encoder[1]
    assert isinstance(film, nn.Linear)
    head = {
        "type": "conditional_unet1d (Chi et al. 2023, adapted; MIT notice in unet1d.py)",
        "down_dims": list(u.down_dims),
        "kernel_size": u.kernel_size,
        "n_groups": u.n_groups,
        "n_groups_source": "UNetConfig default (diffusion_policy's value; not a config key)",
        "activation": "mish",
        "cond": u.cond,
        "film": "per-channel scale and bias from [step embedding || obs feature]",
        "diffusion_step_embed_dim": u.diffusion_step_embed_dim,
        "diffusion_step_encoder": (
            f"sinusoidal({u.diffusion_step_embed_dim}) -> Linear({4 * u.diffusion_step_embed_dim})"
            f" -> Mish -> Linear({u.diffusion_step_embed_dim})"
        ),
        "global_cond_dim": int(film.in_features) - u.diffusion_step_embed_dim,
        "film_cond_dim": int(film.in_features),
        "sample_shape_Tp_A": [cfg.horizon, cfg.action_dim],
        "residual_blocks_per_level": 2,
        "mid_blocks": len(net.unet.mid_modules),
    }
    diffusion = {
        "train_scheduler": "ddpm (diffusers DDPMScheduler)",
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "prediction_type": cfg.prediction_type,
        "clip_sample": cfg.clip_sample,
        "sampler": "ddim (diffusers DDIMScheduler), eta 0",
        "num_inference_steps_K": cfg.num_inference_steps,
        "timestep_spacing": cfg.timestep_spacing,
        "final_clip": [-1.0, 1.0],
        "noise": (
            "explicit input; per-request torch.Generator seeded by (run_seed, episode_seed, step)"
        ),
        "action_normalisation": "identity (actions already in [-1, 1])",
    }
    return head, diffusion


def _student_record(arm: str) -> tuple[dict[str, Any], nn.Module | None]:
    """The derivable record of one student and its freshly built network (``None`` for R1)."""
    entry = REGISTRY[arm]
    base: dict[str, Any] = {"tag": entry.tag, "kind": entry.kind, "obs_variant": entry.obs_variant}
    if not entry.has_ila_network:
        return _r1_record(entry, base), None
    cfg = TrainConfig.load(entry.config)
    raw = cfg.raw
    net = build_net_from_config(entry.kind, raw, CROSSCHECK_SEED, entry.obs_variant)
    rec: dict[str, Any] = {
        **base,
        "built_here": True,
        "builder": "ila.policies.registry.build_net_from_config (the training loop's builder)",
        "config": _config_block(entry, cfg),
        "action_dim": 3,
        "action": (
            "normalised world-frame velocity setpoint in [-1, 1]^3 (env: x 1.5 m/s, norm cap)"
        ),
        "training": _training_block(cfg),
        "parameters": _parameters(net),
        "inference_path": "cpu fp32 single-sample (selection and evaluation)",
    }
    if isinstance(net, StateMLP):
        from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

        contract = load_contract()
        widths = [net.cfg.obs_dim, *net.cfg.hidden, net.cfg.out]
        rec["inputs"] = {
            "images": None,
            "teacher_obs_dim": net.cfg.obs_dim,
            "teacher_obs_fields": [{"name": n, "size": s} for n, (_, s) in contract.layout.items()],
            "privileged_by_design": True,
            "normalisation": "fixed buffers obs_mean/obs_std (train-set mean/std)",
            "frames_To": 1,
        }
        rec["architecture"] = {
            "trunk": None,
            "head": {"type": "mlp", "widths": widths, "activation": "relu", "output": "tanh"},
        }
        rec["horizons"] = {"To_steps": 1, "Tp_steps": 1, "Ta_steps": 1}
        return rec, net
    assert isinstance(net, nn.Module)
    enc = net.encoder
    assert isinstance(enc, VisionEncoder)
    rec["inputs"] = _vision_inputs(entry, enc, raw)
    arch: dict[str, Any] = {"encoder": _encoder_block(enc)}
    to = enc.cfg.n_obs_steps
    if isinstance(net, DiffusionPolicyNet):
        arch["head"], rec["diffusion"] = _dp_block(net)
        rec["horizons"] = {
            "To_steps": net.cfg.n_obs_steps,
            "Tp_steps": net.cfg.horizon,
            "Ta_steps": net.cfg.n_action_steps,
            "Ta_K_at_selection": dict(raw["selection"]["at"]),
            "executed_chunk_indices": [to - 1, to - 2 + net.cfg.n_action_steps],
        }
        rec["inference_sweeps"] = {
            "Ta_steps": list(raw["inference_sweeps"]["Ta"]),
            "K": list(raw["inference_sweeps"]["K"]),
            "temporal_ensembling_m": float(raw["inference_sweeps"]["temporal_ensembling"]["m"]),
        }
    elif isinstance(net, ModularEstimator):
        from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

        contract = load_contract()
        mh = ModularHeadConfig.from_mapping(raw)
        arch["head"] = {
            "type": "mlp",
            "widths": [enc.cfg.feature_dim, *mh.hidden, mh.out_dim],
            "activation": "relu",
            "output": "linear: normalised privileged-block estimate (train-set mean/std buffers)",
            "out_fields": [{"name": n, "size": contract.layout[n][1]} for n in mh.out_fields],
        }
        rec["action_dim"] = 3
        rec["action"] = "from the frozen P8 teacher fed the de-normalised estimate"
        rec["frozen_teacher"] = {
            "source": "P8 ppo seed 0 final/ (VecNormalize frozen, training=False)",
            "trainable": False,
            "counted_in_parameters": False,
            "see": "p8_controllers.ppo",
            "clock": "time_fraction passes to the teacher only (declared clock exception 2)",
        }
        rec["horizons"] = {"To_steps": to, "Tp_steps": 1, "Ta_steps": 1}
    else:
        arch["head"] = _bc_head(raw, enc.cfg.feature_dim)
        rec["horizons"] = {"To_steps": to, "Tp_steps": 1, "Ta_steps": 1}
    rec["architecture"] = arch
    return rec, net


def _r1_record(entry: StudentEntry, base: dict[str, Any]) -> dict[str, Any]:
    """``R1_lerobot_dp``: config facts only; LeRobot's network is not built in this venv."""
    raw: Mapping[str, Any] = yaml.safe_load(entry.config.read_text(encoding="utf-8"))
    pol = raw["policy"]
    train = raw["train"]
    kept = raw["lerobot_defaults_kept"]["policy"]
    preset = raw["lerobot_defaults_kept"]["optimizer_preset"]
    return {
        **base,
        "built_here": False,
        "note": (
            "not built here: LeRobot 0.6.1 DiffusionPolicy, trained in .venv-lerobot and flown "
            "from its ONNX export; see the config. Parameter counts are not derived by ila."
        ),
        "config": _file_ref(entry.config),
        "framework": dict(raw["framework"]),
        "inputs": {
            "input_features": list(raw["data"]["input_features"]),
            "crop_px": list(pol["crop_shape"]),
            "crop_is_random_train": bool(pol["crop_is_random"]),
            "frames_To": int(pol["n_obs_steps"]),
        },
        "architecture": {
            "vision_backbone": pol["vision_backbone"],
            "pretrained_backbone_weights": pol["pretrained_backbone_weights"],
            "use_group_norm": bool(pol["use_group_norm"]),
            "spatial_softmax_num_keypoints": int(pol["spatial_softmax_num_keypoints"]),
            "down_dims": list(pol["down_dims"]),
            "kernel_size": int(pol["kernel_size"]),
            "diffusion_step_embed_dim": int(pol["diffusion_step_embed_dim"]),
            "unet_n_groups": int(kept["n_groups"]),
            "use_film_scale_modulation": bool(kept["use_film_scale_modulation"]),
        },
        "horizons": {
            "To_steps": int(pol["n_obs_steps"]),
            "Tp_steps": int(pol["horizon"]),
            "Ta_steps": int(pol["n_action_steps"]),
        },
        "diffusion": {
            "num_train_timesteps": int(pol["num_train_timesteps"]),
            "beta_schedule": pol["beta_schedule"],
            "prediction_type": pol["prediction_type"],
            "clip_sample": bool(pol["clip_sample"]),
            "sampler": f"{raw['inference']['sampler']} (ila's, from the ONNX export)",
            "num_inference_steps_K": int(raw["inference"]["num_inference_steps"]),
            "timestep_spacing": raw["inference"]["timestep_spacing"],
        },
        "training": {
            "steps": int(train["steps"]),
            "batch_size": int(train["batch_size"]),
            "optimizer": f"{preset['type']} (LeRobot preset, grad clip {preset['grad_clip_norm']})",
            "lr": float(kept["optimizer_lr"]),
            "weight_decay": float(kept["optimizer_weight_decay"]),
            "schedule": kept["scheduler_name"],
            "warmup_steps": int(kept["scheduler_warmup_steps"]),
            "precision": "fp32" if not kept["use_amp"] else "amp",
            "ema": raw["lerobot_defaults_kept"]["ema"],
            "evaluated_weights": "raw (LeRobot 0.6.1 has no EMA for diffusion)",
            "seeds": [int(s) for s in raw["seeds"]],
        },
        "parameters": None,
        "checkpoint_crosscheck": None,
    }


# --------------------------------------------------------------------------- cross-checks


def _pinned_checkpoint(arm: str, seed: int) -> Path | dict[str, Any]:
    """Resolve (arm, seed)'s pinned selected checkpoint (``ila.eval.arms``, as Phase 8 flew it).

    Returns:
        The recorded path if the file is not on this machine; else the record facts (with the
        local path under ``_path``), after re-hashing the file.

    Raises:
        AssertionError: If the committed ``selection.json`` differs from its pin, or the
            checkpoint bytes from the recorded SHA-256.
    """
    from ila.eval import arms as ev_arms  # lazy: the matrix module is heavy

    entry = REGISTRY[arm]
    inference = ev_arms.Inference(8, 10, False) if ev_arms.is_diffusion(arm) else None
    spec = ev_arms.ArmSpec(
        arm=arm,
        tag=entry.tag,
        seeds=(seed,),
        teacher=ev_arms.expected_teacher(arm),
        obs_variant=ev_arms.expected_obs_variant(arm),
        inference=inference,
    )
    ckpt = ev_arms.resolve_checkpoint(spec, seed)
    pins: Mapping[str, str] = json.loads(PINS.read_text(encoding="utf-8"))["pins"]
    key = f"{arm}/s{seed}"
    if pins.get(key) != ckpt.selection_json_sha256:
        raise AssertionError(f"{key}: selection.json {ckpt.selection_json_sha256} != pin")
    if not ckpt.path.is_file():
        return ckpt.path
    actual = _sha256(ckpt.path)
    if actual != ckpt.sha256:
        raise AssertionError(f"{key}: {ckpt.path} SHA-256 {actual} != recorded {ckpt.sha256}")
    return {
        "seed": seed,
        "checkpoint_file": _rel(ckpt.path),
        "sha256": ckpt.sha256,
        "global_step": ckpt.global_step,
        "selection_json": ckpt.selection_json,
        "selection_json_sha256": ckpt.selection_json_sha256,
        "pinned_in": _rel(PINS),
        "_path": str(ckpt.path),
    }


def _student_crosscheck(arm: str, net: nn.Module, config_sha256: str) -> dict[str, Any] | None:
    """Load the pinned seed-0 checkpoint on the CPU and compare it to the built network.

    Returns:
        The cross-check record, or ``None`` if the checkpoint is not on this machine.

    Raises:
        AssertionError: If any tensor's name or shape, or the parameter total, differs.
    """
    found = _pinned_checkpoint(arm, CROSSCHECK_SEED)
    if isinstance(found, Path):
        return None
    rec = dict(found)
    path = Path(rec.pop("_path"))
    ckpt = read_checkpoint(path)
    want = _shapes(net.state_dict())
    param_names = {name for name, _ in net.named_parameters()}
    checked: list[str] = []
    n_tensors = 0
    for key in ("ema", "model"):
        state = ckpt.get(key)
        if state is None:
            continue
        got = _shapes(state)
        if got != want:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            bad = sorted(k for k in set(want) & set(got) if want[k] != got[k])
            raise AssertionError(
                f"{arm}: checkpoint {key} != built net: missing {missing[:5]}, "
                f"extra {extra[:5]}, shape {bad[:5]}"
            )
        total = int(sum(state[k].numel() for k in param_names))
        if total != _numel(net):
            raise AssertionError(f"{arm}: checkpoint {key} parameters {total} != {_numel(net)}")
        checked.append(key)
        n_tensors = len(got)
    if "ema" not in checked:
        raise AssertionError(f"{arm}: checkpoint has no EMA weights")
    if ckpt.get("arm") != arm or int(ckpt.get("run_seed", -1)) != CROSSCHECK_SEED:
        raise AssertionError(f"{arm}: checkpoint is {ckpt.get('arm')} s{ckpt.get('run_seed')}")
    rec.update(
        weights_checked=checked,
        tensors_checked=n_tensors,
        parameters_total=_numel(net),
        config_sha256_at_crosscheck=config_sha256,
        checkpoint_config_sha256=str(ckpt.get("config_sha256")),
        checkpoint_config_sha256_matches=ckpt.get("config_sha256") == config_sha256,
        match=True,
    )
    return rec


# --------------------------------------------------------------------------- P8 controllers


def _ppo_expected_shapes(obs_dim: int, arch: list[int], action_dim: int) -> dict[str, list[int]]:
    """SB3 ``ActorCriticPolicy`` tensors for ``net_arch={"pi": arch, "vf": arch}``.

    ``FlattenExtractor`` (no parameters, shared); ``mlp_extractor.{policy,value}_net`` =
    Linear-Tanh pairs at indices 0, 2, ...; ``action_net`` (Gaussian mean), ``log_std``
    (state-independent), ``value_net``.
    """
    out: dict[str, list[int]] = {"log_std": [action_dim]}
    for net in ("policy_net", "value_net"):
        widths = [obs_dim, *arch]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            out[f"mlp_extractor.{net}.{2 * i}.weight"] = [b, a]
            out[f"mlp_extractor.{net}.{2 * i}.bias"] = [b]
    out["action_net.weight"] = [action_dim, arch[-1]]
    out["action_net.bias"] = [action_dim]
    out["value_net.weight"] = [1, arch[-1]]
    out["value_net.bias"] = [1]
    return out


def _count(shapes: Mapping[str, list[int]], keep: Callable[[str], bool]) -> int:
    total = 0
    for name, shape in shapes.items():
        if keep(name):
            n = 1
            for d in shape:
                n *= d
            total += n
    return total


def _prefix(prefix: str) -> Callable[[str], bool]:
    """Predicate: the tensor name starts with ``prefix``."""
    return lambda name: name.startswith(prefix)


def _is_actor(name: str) -> bool:
    return name == "log_std" or name.startswith(("mlp_extractor.policy_net", "action_net"))


def _is_critic(name: str) -> bool:
    return name.startswith(("mlp_extractor.value_net", "value_net"))


def _ppo_record() -> tuple[dict[str, Any], dict[str, list[int]]]:
    """The PPO teacher from P8's config and code (derivable on a fresh clone)."""
    facts = p8_adapter.ppo_teacher_facts()
    cfg: Mapping[str, Any] = facts["config"]
    fields = p8_adapter.obs_fields()
    obs_dim = int(sum(int(f.size) for f in fields))
    landing = p8_adapter.load_eval_configs().landing
    arch: list[int] = list(facts["net_arch"])
    action_dim = int(p8_adapter.ACTION_DIM)
    shapes = _ppo_expected_shapes(obs_dim, arch, action_dim)
    actor = _count(shapes, _is_actor)
    critic = _count(shapes, _is_critic)
    if actor + critic != _count(shapes, lambda _: True):
        raise AssertionError("PPO actor + critic != total")
    groups = {
        "mlp_extractor.policy_net": "mlp_extractor.policy_net",
        "action_net": "action_net",
        "log_std": "log_std",
        "mlp_extractor.value_net": "mlp_extractor.value_net",
        "value_net": "value_net.",
    }
    by_module = {k: _count(shapes, _prefix(p)) for k, p in groups.items()}
    p8_config = p8_adapter.P8_ROOT / facts["config_file"]
    rec: dict[str, Any] = {
        "role": "teacher T_ppo (every T_ppo demo, DAgger/DART labels, modular's frozen policy)",
        "p8_files": {
            "config": facts["config_file"],
            "build": facts["build_file"],
            "policy_wrapper": facts["policy_file"],
            "env": facts["env_file"],
        },
        "p8_config_sha256": _sha256(Path(p8_config)),
        "algorithm": "stable-baselines3 PPO, MlpPolicy (ActorCriticPolicy), trained on CPU",
        "seed": 0,
        "checkpoint": "final/",
        "observation": {
            "dim": obs_dim,
            "fields": [{"name": f.name, "size": int(f.size)} for f in fields],
            "privileged": True,
            "normalisation": {
                "kind": "SB3 VecNormalize, statistics frozen at evaluation (training=False)",
                "norm_obs": bool(cfg["normalize"]["norm_obs"]),
                "clip_obs": float(cfg["normalize"]["clip_obs"]),
                "norm_reward_in_training": bool(cfg["normalize"]["norm_reward"]),
            },
        },
        "action": {
            "dim": action_dim,
            "space": "Box(-1, 1, (3,)): normalised world-frame velocity setpoint, yaw held at 0",
            "v_max_m_s": float(landing.v_max_m_s),
            "v_max_is_norm_cap": bool(landing.v_max_is_norm_cap),
            "tracker": facts["tracker"],
            "ctrl_freq_hz": int(landing.ctrl_freq_hz),
            "physics_freq_hz": int(landing.physics_freq_hz),
            "evaluation": "deterministic mean action, clipped to [-1, 1]",
        },
        "policy": {
            "net_arch_pi": arch,
            "net_arch_vf": arch,
            "shared_layers": False,
            "activation": str(cfg["policy"]["activation"]),
            "log_std_init": float(cfg["ppo"]["log_std_init"]),
            "features_extractor": "FlattenExtractor (no parameters)",
        },
        "training": {
            "total_steps": int(cfg["total_steps"]),
            "n_envs": int(cfg["n_envs"]),
            "ppo": dict(cfg["ppo"]),
            "curriculum_stages": list(cfg["curriculum"]["stages"]),
            "steps_unit": "env control steps of 1/30 s model scale",
        },
        "parameters": {
            "actor": actor,
            "critic": critic,
            "total": actor + critic,
            "by_module": by_module,
            "derivation": "from P8's config: net_arch for pi and vf, obs dim 25, action dim 3",
        },
    }
    return rec, shapes


def _ppo_crosscheck(shapes: Mapping[str, list[int]]) -> dict[str, Any] | None:
    """Load P8's PPO seed-0 ``final/`` through the adapter and compare to the derived shapes.

    Returns:
        The record, or ``None`` if ``configs/paths.local.yaml`` or the run is absent.
    """
    try:
        run_dir = p8_adapter.load_paths()["p8_ppo_seed0_run"]
    except (FileNotFoundError, KeyError):
        return None
    final = Path(run_dir) / "final"
    if not (final / "model.zip").is_file():
        return None
    policy: Any = p8_adapter.teacher("ppo", 0)
    got = _shapes(policy.model.policy.state_dict())
    if got != dict(shapes):
        raise AssertionError(f"PPO checkpoint tensors {got} != derived {dict(shapes)}")
    kwargs: Mapping[str, Any] = policy.model.policy_kwargs
    norm: Any = policy.normalizer
    prov = json.loads(P8_PROVENANCE.read_text(encoding="utf-8"))
    expected = {c["name"]: c["expected"] for c in prov["checks"]}
    zip_sha = _sha256(final / "model.zip")
    vn_sha = _sha256(final / "vecnormalize.pkl")
    if zip_sha != expected["ppo_s0_model_zip_sha256"]:
        raise AssertionError(f"PPO model.zip {zip_sha} != {_rel(P8_PROVENANCE)}")
    if vn_sha != expected["ppo_s0_vecnormalize_sha256"]:
        raise AssertionError(f"PPO vecnormalize.pkl {vn_sha} != {_rel(P8_PROVENANCE)}")
    obs_dim = got["mlp_extractor.policy_net.0.weight"][1]
    if norm is None or norm.training or tuple(norm.obs_rms.mean.shape) != (obs_dim,):
        raise AssertionError("PPO VecNormalize is not frozen or has the wrong width")
    activation = getattr(kwargs.get("activation_fn"), "__name__", None)
    total = _count(got, lambda _: True)
    return {
        "checkpoint": "p8_ppo_seed0_run/final (configs/paths.local.yaml)",
        "model_zip_sha256": zip_sha,
        "vecnormalize_sha256": vn_sha,
        "sha256_match_p8_provenance": True,
        "p8_provenance": _rel(P8_PROVENANCE),
        "checkpoint_steps": int(policy.steps),
        "tensors_checked": len(got),
        "parameters_total": total,
        "policy_kwargs": {
            "net_arch": {k: list(v) for k, v in dict(kwargs["net_arch"]).items()},
            "activation_fn": activation,
            "log_std_init": float(kwargs["log_std_init"]),
        },
        "vecnormalize": {
            "training": bool(norm.training),
            "obs_rms_dim": int(norm.obs_rms.mean.shape[0]),
            "clip_obs": float(norm.clip_obs),
        },
        "match": True,
    }


def _controller_record(name: str, role: str) -> dict[str, Any]:
    """One P8 registry controller (no learned parameters)."""
    facts = p8_adapter.controller_facts(name)
    cfg_path = p8_adapter.P8_ROOT / facts["config_file"]
    return {
        "role": role,
        "structure": _STRUCTURE[name],
        "p8_description": facts["description"],
        "class": facts["class"],
        "p8_files": {"source": facts["source_file"], "config": facts["config_file"]},
        "p8_config_sha256": _sha256(Path(cfg_path)),
        "privileged": facts["privileged"],
        "needs_motion_feed": facts["needs_motion_feed"],
        "resolved": facts["resolved"],
        "p8_resolved_config_sha256": facts["resolved_config_sha256"],
        "learned_parameters": 0,
        "observation": "P8's 25-d observation (it reads the fields it needs by name)",
        "action": "the same normalised world-frame velocity setpoint as the PPO teacher",
    }


# --------------------------------------------------------------------------- document


@dataclass(frozen=True)
class CheckResult:
    """Outcome of :func:`check` / :func:`write`.

    Attributes:
        ok: The re-derived document equals the committed file (``check``), or was written.
        notes: One line per cross-check (run, carried over, or skipped) and any diff.
    """

    ok: bool
    notes: tuple[str, ...]


def _carry(
    what: str,
    fresh: dict[str, Any] | None,
    previous: Mapping[str, Any] | None,
    *,
    total: int,
    config_sha256: str | None,
    attempted: bool,
    notes: list[str],
) -> dict[str, Any]:
    """Return the fresh cross-check, else the committed one if it describes the same network.

    Raises:
        AssertionError: If the committed cross-check was made for another parameter total or
            another config SHA-256 than the re-derived record (a stale record is never carried).
    """
    if fresh is not None:
        notes.append(f"{what}: checkpoint cross-check run, match")
        return fresh
    why = "checkpoint absent here" if attempted else "cross-check not run (--no-checkpoints)"
    if previous is None or previous.get("status") == ABSENT:
        notes.append(f"{what}: {why} and no committed cross-check; recorded as absent")
        return {"status": ABSENT}
    if int(previous.get("parameters_total", -1)) != total:
        raise AssertionError(
            f"{what}: committed cross-check is for {previous.get('parameters_total')} "
            f"parameters, re-derived {total}"
        )
    if config_sha256 is not None and previous.get("config_sha256_at_crosscheck") != config_sha256:
        raise AssertionError(f"{what}: committed cross-check was made for another config")
    notes.append(f"{what}: {why}; committed cross-check carried over (skipped)")
    return dict(previous)


def build_document(
    *, checkpoints: bool = True, previous: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Re-derive the whole document.

    Args:
        checkpoints: Run the checkpoint cross-checks where the checkpoints exist locally.
        previous: The committed document, whose cross-checks are carried over where a
            checkpoint is absent (or ``checkpoints`` is False).

    Returns:
        ``(document, notes)``.
    """
    notes: list[str] = []
    prev_students: Mapping[str, Any] = (previous or {}).get("students", {})
    prev_p8: Mapping[str, Any] = (previous or {}).get("p8_controllers", {})
    students: dict[str, Any] = {}
    for arm in student_arms():
        rec, net = _student_record(arm)
        if net is not None:
            sha = str(rec["config"]["sha256"])
            fresh = _student_crosscheck(arm, net, sha) if checkpoints else None
            prev = prev_students.get(arm, {}).get("checkpoint_crosscheck")
            rec["checkpoint_crosscheck"] = _carry(
                arm,
                fresh,
                prev,
                total=_numel(net),
                config_sha256=sha,
                attempted=checkpoints,
                notes=notes,
            )
        students[arm] = rec
    ppo, shapes = _ppo_record()
    fresh_ppo = _ppo_crosscheck(shapes) if checkpoints else None
    ppo["checkpoint_crosscheck"] = _carry(
        "ppo",
        fresh_ppo,
        prev_p8.get("ppo", {}).get("checkpoint_crosscheck"),
        total=int(ppo["parameters"]["total"]),
        config_sha256=None,
        attempted=checkpoints,
        notes=notes,
    )
    controllers: dict[str, Any] = {"ppo": ppo}
    for name, role in CONTROLLERS.items():
        controllers[name] = _controller_record(name, role)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_by": "scripts/model_specs.py (ila.policies.specs)",
        "p8": {
            "root": _rel(Path(p8_adapter.P8_ROOT)),
            "commit": p8_adapter.p8_commit(),
            "paths": "every p8_files path is relative to the P8 root at this commit",
        },
        "units": {
            "image": "px",
            "horizons": "control steps of 1/30 s model scale (lambda = 1/25; 1/6 s full scale)",
            "training_steps": "optimiser steps",
            "lr": "per optimiser step",
            "p8_gains": "units in the key names; lengths m, speeds m/s, times s, model scale",
            "parameters": "elements of nn.Parameter tensors; buffers counted apart",
        },
        "crosscheck_seed": CROSSCHECK_SEED,
        "students": students,
        "p8_controllers": controllers,
    }
    return doc, notes


def dumps(doc: Mapping[str, Any]) -> str:
    """Canonical text: sorted keys, 2-space indent, ASCII, trailing newline."""
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def write(path: Path = OUTPUT, *, checkpoints: bool = True) -> CheckResult:
    """Re-derive and write the document (committed cross-checks carried where absent)."""
    doc, notes = build_document(checkpoints=checkpoints, previous=_load(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(doc), encoding="utf-8")
    return CheckResult(True, (*notes, f"wrote {_rel(path)}"))


def check(path: Path = OUTPUT, *, checkpoints: bool = True) -> CheckResult:
    """Re-derive the document and compare it byte for byte with the committed file."""
    committed = _load(path)
    if committed is None:
        return CheckResult(False, (f"{_rel(path)}: missing",))
    doc, notes = build_document(checkpoints=checkpoints, previous=committed)
    want = path.read_text(encoding="utf-8")
    got = dumps(doc)
    if got == want:
        return CheckResult(True, (*notes, f"{_rel(path)}: re-derived, identical"))
    diff = difflib.unified_diff(
        want.splitlines(), got.splitlines(), "committed", "re-derived", lineterm="", n=2
    )
    return CheckResult(False, (*notes, f"{_rel(path)}: DIFFERS", *list(diff)[:200]))
