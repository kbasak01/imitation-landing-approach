r"""Export R1's LeRobot Diffusion Policy checkpoints to ONNX (Phase 6; ``.venv-lerobot`` ONLY).

R1 (``R1_lerobot_dp``, tag ``reference_external``) is trained by lerobot 0.6.1 and flown by
``ila`` through ONNX Runtime in ``.venv``. This script is the only bridge: it never imports
``ila`` (not installed in this venv), runs on the CPU only (``CUDA_VISIBLE_DEVICES`` is
cleared before torch is imported) and writes, for a checkpoint at training step ``N``,
``<run>/onnx/step<N>/``:

* ``encoder.onnx``: ``image`` ``uint8[B, To, 96, 96, 3]`` (HWC, oldest frame first) and
  ``state`` ``float32[B, To, 13]`` (raw student proprio, ``student.yaml`` order) ->
  ``global_cond`` ``float32[B, C]``. Inside the graph, exactly LeRobot's eval path:
  ``lerobot.envs.utils.preprocess_observation``'s ``uint8 -> float32 / 255``, HWC -> CHW;
  the preprocessor's ``NormalizerProcessorStep`` (VISUAL MEAN_STD, STATE MIN_MAX, with the
  checkpoint's own statistics; LeRobot's ``_apply_transform`` is called, not re-derived);
  ``DiffusionModel._prepare_global_conditioning`` (eval-mode centre crop, ResNet + spatial
  softmax, per-step ``[state, image feature]`` concatenation, flattened).
* ``denoiser.onnx``: ``sample`` ``float32[B, Tp, 3]`` (normalised actions), ``timestep``
  ``int64[B]``, ``global_cond`` ``float32[B, C]`` -> ``eps`` ``float32[B, Tp, 3]``
  (LeRobot's ``DiffusionConditionalUnet1d``).
* ``meta.json``: horizons, ``C``, the schedulers, the action unnormalisation (MIN_MAX
  per-dim min/max from the checkpoint's postprocessor), SHA-256 of every checkpoint file and
  of both graphs, versions, the export wall clock, and the LeRobot-path check below.
* ``reference.npz``: torch fp32 CPU outputs at batch 1 for ``--n-ref`` (64) fixed inputs, the
  first windows (anchors ``t = 0`` and ``t = 1``; ``t = 0`` repeats frame 0, as LeRobot's
  observation queue and ``il-conventions`` do) of the first ``n-ref / 2`` episodes of R1's
  dataset: ``global_cond``; ``eps`` at k in {99, 89, 49, 9, 0} on fixed ``N(0, 1)``
  samples; a full ``K = 10`` DDIM-trailing chunk from fixed provided noise with every
  intermediate ``x_k`` and ``eps``, clipped (normalised) and unnormalised.

Sampling is never in a graph (no RNG op; noise is an input; checked after export). The
reference chunk reimplements ``ila.policies.dp.policy.ddim_scheduler`` / ``ddim_sample``
exactly (diffusers ``DDIMScheduler``, ``set_alpha_to_one=True``, ``steps_offset=0``,
``timestep_spacing="trailing"``, eta = 0, final clip to [-1, 1]); ``ila.deploy.r1_parity``
in ``.venv`` replays the stored epsilons through ``ila``'s own sampler to prove the two agree.

**LeRobot-path check.** The same 64 inputs are also run through LeRobot's own eval path:
``preprocess_observation`` -> the checkpoint's preprocessor -> ``DiffusionPolicy.select_action``
(its observation queues, image-key stacking, ``generate_actions`` slicing ``To - 1 ...
To - 2 + Ta``) -> the checkpoint's postprocessor, at batch 1 with the same ``noise`` argument.
LeRobot's native sampler is the config's ``noise_scheduler_type`` (DDPM, 100 ancestral steps,
global RNG) and cannot take per-step noise, so for this check its ``noise_scheduler`` is
replaced by the ila DDIM-trailing scheduler and ``num_inference_steps`` set to K; everything
else is LeRobot's unmodified code. Its executed actions must equal the reference chunk's
executed slice (recorded in ``meta.json``; the export fails above ``--lerobot-tol``).

Idempotent: a step directory whose ``meta.json`` records the same checkpoint file hashes and
whose files match their recorded hashes is skipped; otherwise it is rebuilt in a temporary
directory and swapped in. Examples::

    .venv-lerobot/bin/python scripts/r1/export_onnx.py export --run artifacts/runs/R1_lerobot_dp/s0
    .venv-lerobot/bin/python scripts/r1/export_onnx.py export \
        --checkpoint artifacts/runs/R1_lerobot_dp/s0/checkpoints/020000/pretrained_model
    .venv-lerobot/bin/python scripts/r1/export_onnx.py fixture --out /tmp/r1_fixture_run

Units: actions are world-frame velocity setpoints (env-normalised, x 1.5 m/s model scale);
one control step is 1/30 s model scale (lambda = 1/25). Simulation only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch the GPU (set before torch)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never touch the Hub: local files only

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"
DATASET = "r1_ppo_2000_first1000_lerobot_v3"
REPO_ID = "ila/r1_ppo_2000_first1000"
LEROBOT_VERSION = "0.6.1"
FORMAT = "ila-r1-onnx/1"
OPSET = 17
MAX_IR_VERSION = 10  # opset 17 needs IR >= 8; ORT 1.29 and onnx 1.22 in .venv both load 10
REF_TIMESTEPS = (99, 89, 49, 9, 0)
REF_K = 10
REF_NOISE_SEED = 20261030
REF_SAMPLE_SEED = 20261031
FIXTURE_SEED = 0
SPACING = "trailing"  # P3-D1 / dp.yaml inference.timestep_spacing
#: ONNX op types that draw random numbers (or may, in training mode). None may appear.
RNG_OPS = frozenset(
    {
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
        "Bernoulli",
        "Dropout",
    }
)
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


# ---------------------------------------------------------------------------------- utilities
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_root_from_paths(path: Path = PATHS_FILE) -> Path:
    """Resolve ``data_root`` as ``ila.p8_adapter.load_paths`` does (``${key}``, repo-relative)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = re.sub(r"\$\{(\w+)\}", lambda m: resolved[m.group(1)], str(value))
    return (REPO_ROOT / resolved["data_root"]).resolve()


def check_venv() -> str:
    """Refuse to run outside ``.venv-lerobot`` or with another lerobot."""
    if Path(sys.prefix).name != ".venv-lerobot":
        sys.exit(f"run with .venv-lerobot/bin/python (sys.prefix is {sys.prefix})")
    import lerobot

    if lerobot.__version__ != LEROBOT_VERSION:
        sys.exit(f"lerobot {lerobot.__version__} != pinned {LEROBOT_VERSION}")
    return str(lerobot.__version__)


def git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def checkpoint_files(ckpt: Path) -> dict[str, str]:
    """SHA-256 of every regular file in a ``pretrained_model`` directory (name -> hex)."""
    missing = [f for f in CHECKPOINT_FILES if not (ckpt / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{ckpt}: missing {missing} (not a lerobot pretrained_model dir)")
    return {p.name: sha256_file(p) for p in sorted(ckpt.iterdir()) if p.is_file()}


def step_of(ckpt: Path) -> int:
    """Training step of ``.../checkpoints/<NNNNNN>/pretrained_model``."""
    name = ckpt.parent.name
    if not name.isdigit():
        raise ValueError(f"{ckpt}: parent {name!r} is not a step directory; pass --step")
    return int(name)


def run_dir_of(ckpt: Path) -> Path:
    """``<run>`` of ``<run>/checkpoints/<NNNNNN>/pretrained_model``."""
    if ckpt.name != "pretrained_model" or ckpt.parent.parent.name != "checkpoints":
        raise ValueError(f"{ckpt}: not <run>/checkpoints/<step>/pretrained_model; pass --out")
    return ckpt.parent.parent.parent


def run_checkpoints(run: Path) -> list[Path]:
    """Every ``<run>/checkpoints/<NNNNNN>/pretrained_model``, symlinks (``last``) skipped."""
    root = run / "checkpoints"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist")
    out = [
        d / "pretrained_model"
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.is_symlink() and d.name.isdigit()
    ]
    return [p for p in out if p.is_dir()]


def rng_ops(model: Any) -> list[str]:
    """Op types in ``RNG_OPS`` anywhere in the model (graph, subgraphs, local functions)."""
    from onnx import AttributeProto

    found: list[str] = []

    def scan(nodes: Any) -> None:
        for n in nodes:
            if n.op_type in RNG_OPS:
                found.append(f"{n.domain or 'ai.onnx'}::{n.op_type}")
            for a in n.attribute:
                if a.type == AttributeProto.GRAPH:
                    scan(a.g.node)
                elif a.type == AttributeProto.GRAPHS:
                    for g in a.graphs:
                        scan(g.node)

    scan(model.graph.node)
    for f in model.functions:
        scan(f.node)
    return sorted(set(found))


# ------------------------------------------------------------------------------- the policy
def load_checkpoint(ckpt: Path) -> tuple[Any, Any, Any, Any]:
    """Load a lerobot ``pretrained_model`` dir on the CPU in eval mode.

    Returns:
        ``(config, policy, preprocessor, postprocessor)``.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    if cfg.type != "diffusion":
        raise ValueError(f"{ckpt}: policy type {cfg.type!r}, expected 'diffusion'")
    cfg.device = "cpu"
    policy = DiffusionPolicy.from_pretrained(ckpt, config=cfg)
    policy.to("cpu").eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return cfg, policy, pre, post


def check_supported(cfg: Any) -> str:
    """Refuse configurations the graph contract does not cover; return the image key."""
    from lerobot.utils.constants import OBS_STATE

    image_keys = list(cfg.image_features)
    problems = []
    if len(image_keys) != 1:
        problems.append(f"exactly one camera supported, got {image_keys}")
    if cfg.env_state_feature is not None:
        problems.append("observation.environment_state is not part of the contract")
    if cfg.robot_state_feature is None or OBS_STATE not in cfg.input_features:
        problems.append("observation.state missing")
    if cfg.prediction_type != "epsilon":
        problems.append(f"prediction_type {cfg.prediction_type!r} (ila's sampler: epsilon)")
    if not cfg.clip_sample or float(cfg.clip_sample_range) != 1.0:
        problems.append("ila's DDIM clips x0 to [-1, 1]: need clip_sample and range 1.0")
    if cfg.beta_schedule != "squaredcos_cap_v2":
        problems.append(
            f"beta_schedule {cfg.beta_schedule!r}: ila's scheduler omits beta_start/end"
        )
    if cfg.resize_shape is not None:
        problems.append("resize_shape is not part of the contract")
    if len(cfg.action_feature.shape) != 1:
        problems.append(f"action shape {cfg.action_feature.shape}")
    if problems:
        raise ValueError("; ".join(problems))
    return str(image_keys[0])


def normalizer_of(pre: Any) -> Any:
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    steps = [s for s in pre.steps if isinstance(s, NormalizerProcessorStep)]
    if len(steps) != 1:
        raise ValueError(f"expected one NormalizerProcessorStep, got {len(steps)}")
    return steps[0]


def unnormalizer_of(post: Any) -> Any:
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep

    steps = [s for s in post.steps if isinstance(s, UnnormalizerProcessorStep)]
    if len(steps) != 1:
        raise ValueError(f"expected one UnnormalizerProcessorStep, got {len(steps)}")
    return steps[0]


def graph_modules(policy: Any, pre: Any, image_key: str) -> tuple[Any, Any]:
    """The two exported computations as eval-mode ``nn.Module`` wrappers around LeRobot code."""
    import torch
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
    from torch import nn

    norm = normalizer_of(pre)
    image_type = norm.features[image_key].type
    state_type = norm.features[OBS_STATE].type
    diffusion = policy.diffusion

    class EncoderGraph(nn.Module):
        """uint8 HWC window + raw state -> LeRobot's global conditioning (eval path)."""

        def __init__(self) -> None:
            super().__init__()
            self.diffusion = diffusion

        def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            # preprocess_observation: "b h w c -> b c h w", .contiguous(), float32, /= 255.
            # The contiguous copy matters for the torch reference: a channels-last-strided
            # input sends torch's CPU conv down another kernel (different summation order).
            x = image.permute(0, 1, 4, 2, 3).contiguous().to(torch.float32) / 255
            x = norm._apply_transform(x, image_key, image_type, inverse=False)
            s = norm._apply_transform(state, OBS_STATE, state_type, inverse=False)
            # select_action: stack the camera keys at dim -4 -> (B, To, n_cam = 1, C, H, W).
            batch = {OBS_STATE: s, OBS_IMAGES: x.unsqueeze(2)}
            out: torch.Tensor = self.diffusion._prepare_global_conditioning(batch)
            return out

    class DenoiserGraph(nn.Module):
        """(x_k, k, global_cond) -> eps-hat (LeRobot's conditional 1-D U-Net)."""

        def __init__(self) -> None:
            super().__init__()
            self.unet = diffusion.unet

        def forward(
            self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
        ) -> torch.Tensor:
            out: torch.Tensor = self.unet(sample, timestep, global_cond=global_cond)
            return out

    return EncoderGraph().eval(), DenoiserGraph().eval()


def ila_ddim_scheduler(cfg: Any, k: int) -> Any:
    """Exactly ``ila.policies.dp.policy.ddim_scheduler`` (same kwargs, ``set_timesteps(K)``)."""
    from diffusers import DDIMScheduler

    sched = DDIMScheduler(
        num_train_timesteps=int(cfg.num_train_timesteps),
        beta_schedule=str(cfg.beta_schedule),
        clip_sample=bool(cfg.clip_sample),
        prediction_type=str(cfg.prediction_type),
        set_alpha_to_one=True,
        steps_offset=0,
        timestep_spacing=SPACING,
    )
    sched.set_timesteps(int(k))
    return sched


def ila_ddim_sample(eps_fn: Any, noise: Any, sched: Any) -> tuple[Any, list[Any], list[Any]]:
    """``ila.policies.dp.policy.ddim_sample`` with every ``x_k`` and ``eps`` recorded.

    Returns:
        ``(clipped sample, [x before each step ..., x after the last step], [eps per step])``.
    """
    import torch

    b = noise.shape[0]
    x = noise.to(dtype=torch.float32)
    xs, epss = [x], []
    for k in sched.timesteps:
        kk = torch.full((b,), int(k), dtype=torch.long, device=x.device)
        eps = eps_fn(x, kk)
        epss.append(eps)
        x = sched.step(eps, int(k), x, eta=0.0).prev_sample
        xs.append(x)
    return x.clamp(-1.0, 1.0), xs, epss


# ------------------------------------------------------------------------------- reference
def reference_inputs(dataset_root: Path, n_ref: int) -> dict[str, np.ndarray]:
    """First windows (anchors 0 and 1) of the first ``n_ref / 2`` episodes (module docstring)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if n_ref % 2:
        raise ValueError(f"--n-ref must be even, got {n_ref}")
    ds = LeRobotDataset(REPO_ID, root=dataset_root)
    raw = ds.hf_dataset.with_format("numpy")
    n_ep = n_ref // 2
    if ds.num_episodes < n_ep:
        raise ValueError(f"{dataset_root}: {ds.num_episodes} episodes < {n_ep}")
    images, states, eps_idx, frames = [], [], [], []
    for e in range(n_ep):
        a = int(ds.meta.episodes[e]["dataset_from_index"])
        b = int(ds.meta.episodes[e]["dataset_to_index"])
        if b - a < 2:
            raise ValueError(f"episode {e} has {b - a} frames")
        rows = raw[a : a + 2]
        img = np.stack([np.asarray(im, dtype=np.uint8) for im in rows["observation.image"]])
        st = np.asarray(rows["observation.state"], dtype=np.float32)
        if img.shape != (2, 96, 96, 3):
            raise ValueError(f"episode {e}: image shape {img.shape}")
        for t in (0, 1):
            window = [max(t - 1, 0), t]  # il-conventions: indices < 0 repeat index 0
            images.append(img[window])
            states.append(st[window])
            eps_idx.append(e)
            frames.append(t)
    return {
        "dataset_num_episodes": np.asarray(ds.num_episodes, dtype=np.int64),
        "dataset_num_frames": np.asarray(ds.num_frames, dtype=np.int64),
        "image": np.ascontiguousarray(np.stack(images)),
        "state": np.ascontiguousarray(np.stack(states)),
        "episode_index": np.asarray(eps_idx, dtype=np.int64),
        "frame_index": np.asarray(frames, dtype=np.int64),
    }


def lerobot_select_actions(
    cfg: Any, policy: Any, pre: Any, post: Any, inputs: dict[str, np.ndarray], noise: Any
) -> np.ndarray:
    """LeRobot's own eval path, batch 1 per input, ila's DDIM swapped in (module docstring).

    Returns:
        ``float32[N, Ta, A]``: the executed actions, unnormalised by the checkpoint's
        postprocessor.
    """
    import torch
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    native = (policy.diffusion.noise_scheduler, policy.diffusion.num_inference_steps)
    policy.diffusion.noise_scheduler = ila_ddim_scheduler(cfg, REF_K)
    policy.diffusion.num_inference_steps = REF_K
    out = []
    try:
        for i in range(inputs["image"].shape[0]):
            policy.reset()
            n_i = noise[i : i + 1]
            frames = [0, 1] if int(inputs["frame_index"][i]) == 1 else [1]
            for j in frames:
                obs = preprocess_observation(
                    {"pixels": inputs["image"][i, j], "agent_pos": inputs["state"][i, j]}
                )
                batch = pre(obs)
                if j == 0:  # warm the observation queue with frame 0, then replan at frame 1
                    policy.select_action(batch, noise=n_i)
                    policy._queues[ACTION].clear()
                    continue
                first = policy.select_action(batch, noise=n_i)
                rest = list(policy._queues[ACTION])
                acts = [post(a) for a in [first, *rest]]
                out.append(torch.stack(acts, dim=1)[0].numpy())
    finally:
        policy.diffusion.noise_scheduler, policy.diffusion.num_inference_steps = native
        policy.reset()
    return np.stack(out).astype(np.float32)


def build_reference(
    cfg: Any,
    policy: Any,
    pre: Any,
    post: Any,
    enc: Any,
    den: Any,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Torch fp32 CPU, batch 1 per input. Returns the arrays and the LeRobot-path summary."""
    import torch

    n = inputs["image"].shape[0]
    tp, a_dim = int(cfg.horizon), int(cfg.action_feature.shape[0])
    g_noise = torch.Generator().manual_seed(REF_NOISE_SEED)
    noise = torch.randn((n, tp, a_dim), generator=g_noise, dtype=torch.float32)
    g_s = torch.Generator().manual_seed(REF_SAMPLE_SEED)
    eps_sample = torch.randn((len(REF_TIMESTEPS), n, tp, a_dim), generator=g_s)
    sched = ila_ddim_scheduler(cfg, REF_K)
    unnorm = unnormalizer_of(post)
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import ACTION

    gcs, epss, xs_all, eps_all, chunks = [], [], [], [], []
    with torch.no_grad():
        for i in range(n):
            img = torch.from_numpy(inputs["image"][[i]])
            st = torch.from_numpy(inputs["state"][[i]])
            gc = enc(img, st)
            gcs.append(gc)
            row = []
            for j, k in enumerate(REF_TIMESTEPS):
                kk = torch.full((1,), k, dtype=torch.long)
                row.append(den(eps_sample[j, i : i + 1], kk, gc))
            epss.append(torch.cat(row, dim=0))
            chunk, xs, es = ila_ddim_sample(
                lambda x, kk, gc=gc: den(x, kk, gc), noise[i : i + 1], sched
            )
            xs_all.append(torch.cat(xs, dim=0))
            eps_all.append(torch.cat(es, dim=0))
            chunks.append(chunk)
        chunk_norm = torch.cat(chunks, dim=0)
        chunk = unnorm._apply_transform(chunk_norm, ACTION, FeatureType.ACTION, inverse=True)
    arrays = {
        **inputs,
        "global_cond": torch.cat(gcs, dim=0).numpy(),
        "eps_timesteps": np.asarray(REF_TIMESTEPS, dtype=np.int64),
        "eps_sample": eps_sample.numpy(),
        "eps": torch.stack(epss, dim=1).numpy(),
        "noise": noise.numpy(),
        "ddim_k": np.asarray(REF_K, dtype=np.int64),
        "ddim_timesteps": np.asarray([int(k) for k in sched.timesteps], dtype=np.int64),
        "ddim_x": torch.stack(xs_all, dim=1).numpy(),
        "ddim_eps": torch.stack(eps_all, dim=1).numpy(),
        "chunk_norm": chunk_norm.numpy(),
        "chunk": chunk.numpy().astype(np.float32),
    }
    to, ta = int(cfg.n_obs_steps), int(cfg.n_action_steps)
    lr = lerobot_select_actions(cfg, policy, pre, post, inputs, noise)
    arrays["lerobot_actions"] = lr
    executed = arrays["chunk"][:, to - 1 : to - 1 + ta]
    diff = np.abs(lr.astype(np.float64) - executed.astype(np.float64))
    summary = {
        "method": (
            "preprocess_observation -> checkpoint preprocessor -> DiffusionPolicy.select_action "
            "(own observation/action queues, t=1 windows: frame 0 queued first, then the "
            "action queue cleared so frame 1 replans) -> checkpoint postprocessor; batch 1; "
            "same noise argument; LeRobot's noise_scheduler replaced by ila's DDIM-trailing "
            f"K={REF_K} for this check only"
        ),
        "n": int(n),
        "executed_slice": [to - 1, to - 1 + ta],
        "max_abs": float(diff.max()),
        "bit_identical": bool(np.array_equal(lr, executed)),
    }
    return arrays, summary


# ---------------------------------------------------------------------------------- export
def export_graphs(enc: Any, den: Any, cfg: Any, out: Path, c_dim: int) -> dict[str, float]:
    """``torch.onnx.export`` (dynamo, opset 17) both graphs; return per-graph seconds."""
    import torch
    from onnxscript import opset17 as op17

    def mish17(x: Any) -> Any:
        # opset 17 has no Mish (added in 18): the definition x * tanh(softplus(x)).
        return op17.Mul(x, op17.Tanh(op17.Softplus(x)))

    table = {torch.ops.aten.mish.default: mish17}
    batch = torch.export.Dim("batch")
    to, tp = int(cfg.n_obs_steps), int(cfg.horizon)
    a_dim, s_dim = int(cfg.action_feature.shape[0]), int(cfg.robot_state_feature.shape[0])
    c, h, w = cfg.image_features[next(iter(cfg.image_features))].shape
    g = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (2, to, h, w, c), dtype=torch.uint8, generator=g)
    state = torch.randn((2, to, s_dim), generator=g)
    sample = torch.randn((2, tp, a_dim), generator=g)
    timestep = torch.tensor([99, 0], dtype=torch.long)
    gc = torch.randn((2, c_dim), generator=g)
    secs = {}
    t0 = time.perf_counter()
    torch.onnx.export(
        enc,
        (image, state),
        str(out / "encoder.onnx"),
        input_names=["image", "state"],
        output_names=["global_cond"],
        dynamic_shapes=({0: batch}, {0: batch}),
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
        custom_translation_table=table,
        verbose=False,
    )
    secs["encoder"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch.onnx.export(
        den,
        (sample, timestep, gc),
        str(out / "denoiser.onnx"),
        input_names=["sample", "timestep", "global_cond"],
        output_names=["eps"],
        dynamic_shapes=({0: batch}, {0: batch}, {0: batch}),
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
        custom_translation_table=table,
        verbose=False,
    )
    secs["denoiser"] = time.perf_counter() - t0
    return secs


def check_graph(
    path: Path, inputs: dict[str, tuple[int, list[Any]]], outputs: dict[str, Any]
) -> dict[str, Any]:
    """Opset, IR, RNG ops, names, dtypes and shapes (dim 0 symbolic) of one exported graph."""
    import onnx

    model = onnx.load(str(path))
    if model.ir_version > MAX_IR_VERSION:
        model.ir_version = MAX_IR_VERSION
        onnx.save(model, str(path))
    onnx.checker.check_model(str(path), full_check=True)
    opsets = {o.domain or "ai.onnx": int(o.version) for o in model.opset_import}
    problems = []
    if opsets != {"ai.onnx": OPSET}:
        problems.append(f"opset imports {opsets}")
    if model.functions:
        problems.append(f"{len(model.functions)} local functions")
    rng = rng_ops(model)
    if rng:
        problems.append(f"RNG ops {rng}")

    def sig(vals: Any) -> dict[str, tuple[int, list[Any]]]:
        return {
            v.name: (
                int(v.type.tensor_type.elem_type),
                [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim],
            )
            for v in vals
        }

    got_in, got_out = sig(model.graph.input), sig(model.graph.output)
    for want, got in ((inputs, got_in), (outputs, got_out)):
        if set(want) != set(got):
            problems.append(f"names {sorted(got)} != {sorted(want)}")
            continue
        for name, (dtype, dims) in want.items():
            g_dtype, g_dims = got[name]
            if g_dtype != dtype or len(g_dims) != len(dims) or g_dims[1:] != dims[1:]:
                problems.append(f"{name}: {g_dtype} {g_dims} != {dtype} {dims}")
            elif not isinstance(g_dims[0], str):
                problems.append(f"{name}: batch axis is static ({g_dims[0]})")
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return {
        "ir_version": int(model.ir_version),
        "opset": OPSET,
        "op_types": sorted({n.op_type for n in model.graph.node}),
        "n_nodes": len(model.graph.node),
        "rng_ops": rng,
        "inputs": {k: {"elem_type": v[0], "shape": v[1]} for k, v in got_in.items()},
        "outputs": {k: {"elem_type": v[0], "shape": v[1]} for k, v in got_out.items()},
    }


def stats_record(pre: Any, post: Any, image_key: str, dataset_root: Path | None) -> dict[str, Any]:
    """Where normalisation lives, its values, and whether they equal the dataset's stats.json."""
    from lerobot.utils.constants import ACTION, OBS_STATE

    norm, unnorm = normalizer_of(pre), unnormalizer_of(post)

    def vec(t: Any) -> list[float]:
        return [float(v) for v in np.asarray(t, dtype=np.float64).reshape(-1)]

    rec: dict[str, Any] = {
        "location": (
            "lerobot 0.6.1 keeps no statistics in the policy (model.safetensors holds weights "
            "only): inputs are normalised by the preprocessor's NormalizerProcessorStep "
            "(policy_preprocessor.json + policy_preprocessor_step_*_normalizer_processor"
            ".safetensors), actions unnormalised by the postprocessor's "
            "UnnormalizerProcessorStep (policy_postprocessor_step_*_unnormalizer_processor"
            ".safetensors)"
        ),
        "norm_map": {k.value: v.value for k, v in norm.norm_map.items()},
        "eps": float(norm.eps),
        "image": {
            "key": image_key,
            "mean": vec(norm._tensor_stats[image_key]["mean"]),
            "std": vec(norm._tensor_stats[image_key]["std"]),
        },
        "state": {
            "min": vec(norm._tensor_stats[OBS_STATE]["min"]),
            "max": vec(norm._tensor_stats[OBS_STATE]["max"]),
        },
        "action_pre": {
            "min": vec(norm._tensor_stats[ACTION]["min"]),
            "max": vec(norm._tensor_stats[ACTION]["max"]),
        },
    }
    a_post = unnorm._tensor_stats[ACTION]
    rec["action_post"] = {"min": vec(a_post["min"]), "max": vec(a_post["max"])}
    rec["action_pre_equals_post"] = rec["action_pre"] == rec["action_post"]
    if dataset_root is not None and (dataset_root / "meta" / "stats.json").is_file():
        ds = json.loads((dataset_root / "meta" / "stats.json").read_text("utf-8"))
        f32 = lambda v: [float(x) for x in np.asarray(v, dtype=np.float32).reshape(-1)]  # noqa: E731
        rec["equals_dataset_stats_json"] = {
            "image_mean_std": f32(ds[image_key]["mean"]) == rec["image"]["mean"]
            and f32(ds[image_key]["std"]) == rec["image"]["std"],
            "state_min_max": f32(ds[OBS_STATE]["min"]) == rec["state"]["min"]
            and f32(ds[OBS_STATE]["max"]) == rec["state"]["max"],
            "action_min_max": f32(ds[ACTION]["min"]) == rec["action_post"]["min"]
            and f32(ds[ACTION]["max"]) == rec["action_post"]["max"],
        }
        rec["dataset_stats_json_sha256"] = sha256_file(dataset_root / "meta" / "stats.json")
    return rec


def up_to_date(out: Path, files: dict[str, str]) -> bool:
    """True iff ``out`` holds a complete export of exactly these checkpoint files."""
    meta_path = out / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("format") != FORMAT or meta.get("source", {}).get("files") != files:
        return False
    want = {**meta.get("onnx_sha256", {}), "reference.npz": meta.get("reference_sha256")}
    return all((out / f).is_file() and sha256_file(out / f) == s for f, s in want.items())


def export_checkpoint(
    ckpt: Path,
    out: Path,
    step: int,
    dataset_root: Path,
    n_ref: int,
    lerobot_tol: float,
    force: bool = False,
) -> dict[str, Any]:
    """Export one checkpoint (module docstring). Returns the meta record (or the skip note)."""
    import diffusers
    import lerobot
    import onnx
    import onnxscript
    import torch
    from onnx import TensorProto

    ckpt = ckpt.resolve()
    files = checkpoint_files(ckpt)
    if not force and up_to_date(out, files):
        print(f"up to date: {out}", flush=True)
        return {"skipped": True, "out": str(out)}
    t_all = time.perf_counter()
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        t0 = time.perf_counter()
        cfg, policy, pre, post = load_checkpoint(ckpt)
        image_key = check_supported(cfg)
        enc, den = graph_modules(policy, pre, image_key)
        t_load = time.perf_counter() - t0
        to, tp, ta = int(cfg.n_obs_steps), int(cfg.horizon), int(cfg.n_action_steps)
        a_dim, s_dim = int(cfg.action_feature.shape[0]), int(cfg.robot_state_feature.shape[0])
        c, h, w = cfg.image_features[image_key].shape
        with torch.no_grad():
            c_dim = int(
                enc(
                    torch.zeros((1, to, h, w, c), dtype=torch.uint8),
                    torch.zeros((1, to, s_dim)),
                ).shape[1]
            )
        secs = export_graphs(enc, den, cfg, tmp, c_dim)
        u8, f32, i64 = TensorProto.UINT8, TensorProto.FLOAT, TensorProto.INT64
        graphs = {
            "encoder.onnx": check_graph(
                tmp / "encoder.onnx",
                {"image": (u8, ["B", to, h, w, c]), "state": (f32, ["B", to, s_dim])},
                {"global_cond": (f32, ["B", c_dim])},
            ),
            "denoiser.onnx": check_graph(
                tmp / "denoiser.onnx",
                {
                    "sample": (f32, ["B", tp, a_dim]),
                    "timestep": (i64, ["B"]),
                    "global_cond": (f32, ["B", c_dim]),
                },
                {"eps": (f32, ["B", tp, a_dim])},
            ),
        }
        t0 = time.perf_counter()
        inputs = reference_inputs(dataset_root, n_ref)
        arrays, lerobot_check = build_reference(cfg, policy, pre, post, enc, den, inputs)
        t_ref = time.perf_counter() - t0
        if not lerobot_check["max_abs"] <= lerobot_tol:
            raise ValueError(
                f"LeRobot select_action path differs from the reference chunk by "
                f"{lerobot_check['max_abs']:.3e} > {lerobot_tol:.1e}"
            )
        np.savez(tmp / "reference.npz", **arrays)
        sched = ila_ddim_scheduler(cfg, REF_K)
        stats = stats_record(pre, post, image_key, dataset_root)
        meta = {
            "format": FORMAT,
            "step": int(step),
            "n_obs_steps": to,
            "horizon": tp,
            "n_action_steps": ta,
            "action_dim": a_dim,
            "state_dim": s_dim,
            "image_shape_hwc": [int(h), int(w), int(c)],
            "C": c_dim,
            "prediction_type": str(cfg.prediction_type),
            "crop_shape": list(cfg.crop_shape) if cfg.crop_shape is not None else None,
            "image_key": image_key,
            "scheduler": {
                "lerobot_native": {
                    "type": str(cfg.noise_scheduler_type),
                    "num_train_timesteps": int(cfg.num_train_timesteps),
                    "beta_schedule": str(cfg.beta_schedule),
                    "beta_start": float(cfg.beta_start),
                    "beta_end": float(cfg.beta_end),
                    "clip_sample": bool(cfg.clip_sample),
                    "clip_sample_range": float(cfg.clip_sample_range),
                    "prediction_type": str(cfg.prediction_type),
                    "num_inference_steps": cfg.num_inference_steps,
                    "note": "training schedule; LeRobot's own select_action samples with it",
                },
                "ila_inference": {
                    "sampler": "ddim",
                    "num_train_timesteps": int(cfg.num_train_timesteps),
                    "beta_schedule": str(cfg.beta_schedule),
                    "clip_sample": bool(cfg.clip_sample),
                    "prediction_type": str(cfg.prediction_type),
                    "timestep_spacing": SPACING,
                    "set_alpha_to_one": True,
                    "steps_offset": 0,
                    "eta": 0.0,
                    "final_clip": [-1.0, 1.0],
                    "default_num_inference_steps": REF_K,
                    "timesteps_at_default_k": [int(k) for k in sched.timesteps],
                    "alphas_cumprod": [float(v) for v in sched.alphas_cumprod.numpy()],
                },
            },
            "action_unnormalisation": {
                "mode": "MIN_MAX",
                "min": stats["action_post"]["min"],
                "max": stats["action_post"]["max"],
                "eps": stats["eps"],
                "formula": (
                    "float32: d = max - min; d = where(d == 0, eps, d); "
                    "a = (x + 1) / 2 * d + min (LeRobot UnnormalizerProcessorStep); no clip"
                ),
            },
            "normalisation": stats,
            "graph_contract": {
                "encoder.onnx": graphs["encoder.onnx"],
                "denoiser.onnx": graphs["denoiser.onnx"],
                "image": "uint8 HWC, frames oldest first; /255, CHW, MEAN_STD, eval centre "
                "crop all inside the graph",
                "state": "raw float32 student proprio; MIN_MAX inside the graph",
                "global_cond_layout": "flatten over To of [state_norm (D), image feature] per step",
            },
            "source": {
                "checkpoint_dir": str(ckpt),
                "checkpoint_sha256": files["model.safetensors"],
                "checkpoint_sha256_of": "model.safetensors",
                "files": files,
            },
            "onnx_sha256": {
                "encoder.onnx": sha256_file(tmp / "encoder.onnx"),
                "denoiser.onnx": sha256_file(tmp / "denoiser.onnx"),
            },
            "reference_sha256": sha256_file(tmp / "reference.npz"),
            "reference": {
                "dataset_root": str(dataset_root),
                "dataset_build_json_sha256": (
                    sha256_file(dataset_root.parent / f"{dataset_root.name}.build.json")
                    if (dataset_root.parent / f"{dataset_root.name}.build.json").is_file()
                    else None
                ),
                "dataset_episodes_frames": [
                    int(inputs["dataset_num_episodes"]),
                    int(inputs["dataset_num_frames"]),
                ],
                "n": int(inputs["image"].shape[0]),
                "selection": "first windows: anchors t = 0 (frame 0 repeated) and t = 1 of "
                "episodes 0 .. n/2 - 1, in that order",
                "torch_path": "torch fp32 CPU, batch 1 per input, eval mode",
                "eps_timesteps": list(REF_TIMESTEPS),
                "noise_seed": REF_NOISE_SEED,
                "eps_sample_seed": REF_SAMPLE_SEED,
                "ddim_k": REF_K,
            },
            "lerobot_path_check": {**lerobot_check, "tolerance": lerobot_tol},
            "versions": {
                "lerobot": str(lerobot.__version__),
                "torch": str(torch.__version__),
                "diffusers": str(diffusers.__version__),
                "onnx": str(onnx.__version__),
                "onnxscript": str(onnxscript.__version__),
                "numpy": str(np.__version__),
                "python": platform.python_version(),
                "opset": OPSET,
                "ir_version": graphs["encoder.onnx"]["ir_version"],
            },
            "exporter": (
                "torch.onnx.export(dynamo=True, opset_version=17, external_data=False): "
                "torch 2.11 builds the graph at its native opset (18) and down-converts to 17 "
                "with the onnx C API version converter (onnxscript's converter declines); "
                "aten::mish is translated to Mul(x, Tanh(Softplus(x))) because ai.onnx Mish "
                "exists only from opset 18. Opset, IR, RNG ops and the I/O contract are "
                "checked after export"
            ),
            "torch_threads": int(torch.get_num_threads()),
            "machine": {"cpu": cpu_model(), "platform": platform.platform()},
            "ila_commit": git_head(),
            "timing_s": {
                "load": t_load,
                "export_encoder": secs["encoder"],
                "export_denoiser": secs["denoiser"],
                "reference": t_ref,
                "total": time.perf_counter() - t_all,
            },
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    t = meta["timing_s"]
    print(
        f"exported {ckpt} -> {out} in {t['total']:.1f} s (load {t['load']:.1f}, encoder "
        f"{t['export_encoder']:.1f}, denoiser {t['export_denoiser']:.1f}, reference "
        f"{t['reference']:.1f}); C={c_dim}; LeRobot path max-abs "
        f"{lerobot_check['max_abs']:.3e}",
        flush=True,
    )
    return meta


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


# --------------------------------------------------------------------------------- fixture
def make_fixture(out_run: Path, dataset_root: Path, step: int, seed: int) -> Path:
    """A random-init R1-shaped checkpoint with the dataset's stats, saved as lerobot does.

    Layout ``<out_run>/checkpoints/<step:06d>/pretrained_model`` (policy ``save_pretrained``
    plus both processor pipelines). Architecture: To 2, Tp 16, Ta 8, crop 88 (random in
    training, centre at eval), ResNet-18 without pretrained weights, GroupNorm, 32
    keypoints, down_dims (128, 256, 512), kernel 5, step embedding 128, DDPM 100
    ``squaredcos_cap_v2`` epsilon with ``clip_sample``; LeRobot's default normalisation.
    """
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    meta = LeRobotDatasetMetadata(REPO_ID, root=dataset_root)
    feats = dataset_to_policy_features(meta.features)
    output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in feats.items() if k not in output_features}
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        crop_shape=(88, 88),
        crop_is_random=True,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        down_dims=(128, 256, 512),
        kernel_size=5,
        diffusion_step_embed_dim=128,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        device="cpu",
        push_to_hub=False,
    )
    torch.manual_seed(seed)
    policy = DiffusionPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    ckpt = out_run / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    pre.save_pretrained(ckpt)
    post.save_pretrained(ckpt)
    print(f"fixture checkpoint: {ckpt}", flush=True)
    return ckpt


# ------------------------------------------------------------------------------------ CLI
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="export one checkpoint or every checkpoint of a run")
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="a lerobot pretrained_model directory")
    src.add_argument("--run", type=Path, help="a run dir holding checkpoints/<NNNNNN>/")
    ex.add_argument("--step", type=int, default=None, help="override the step (with --checkpoint)")
    ex.add_argument("--out", type=Path, default=None, help="step dir (with --checkpoint)")
    ex.add_argument("--dataset-root", type=Path, default=None)
    ex.add_argument("--n-ref", type=int, default=64)
    ex.add_argument("--lerobot-tol", type=float, default=1e-6)
    ex.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    ex.add_argument("--force", action="store_true", help="re-export even if up to date")
    fx = sub.add_parser("fixture", help="write a random-init R1-shaped checkpoint (tests)")
    fx.add_argument("--out", type=Path, required=True, help="fixture run dir")
    fx.add_argument("--dataset-root", type=Path, default=None)
    fx.add_argument("--step", type=int, default=0)
    fx.add_argument("--seed", type=int, default=FIXTURE_SEED)
    args = parser.parse_args()
    check_venv()
    import torch

    torch.set_num_threads(getattr(args, "threads", 1))
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else data_root_from_paths() / "lerobot" / DATASET
    )
    if args.cmd == "fixture":
        make_fixture(args.out, dataset_root, args.step, args.seed)
        return 0
    if args.checkpoint is not None:
        ckpt = args.checkpoint
        if (ckpt / "pretrained_model").is_dir():
            ckpt = ckpt / "pretrained_model"
        step = args.step if args.step is not None else step_of(ckpt)
        out = args.out if args.out is not None else run_dir_of(ckpt) / "onnx" / f"step{step}"
        jobs = [(ckpt, out, step)]
    else:
        if args.out is not None or args.step is not None:
            sys.exit("--out/--step apply to --checkpoint only")
        jobs = [
            (c, args.run / "onnx" / f"step{step_of(c)}", step_of(c))
            for c in run_checkpoints(args.run)
        ]
        if not jobs:
            sys.exit(f"no checkpoints under {args.run / 'checkpoints'}")
    for ckpt, out, step in jobs:
        export_checkpoint(
            ckpt, out, step, dataset_root, args.n_ref, args.lerobot_tol, force=args.force
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
