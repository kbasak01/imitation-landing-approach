"""Train R1_lerobot_dp with LeRobot 0.6.1's own training building blocks (``.venv-lerobot`` ONLY).

Background job (prepare with ``/train-job``; ``make r1-train SEED=n``). It never imports ``ila``:
it uses the standard library, ``yaml``, ``torch`` and ``lerobot`` only. The loop is
``lerobot.scripts.lerobot_train.train`` step for step, built from the same functions
(``make_train_eval_datasets``, ``make_policy``, ``make_pre_post_processors``,
``make_optimizer_and_scheduler``, ``EpisodeAwareSampler``, ``update_policy``,
``save_checkpoint``, ``load_training_state``, ``compute_sampler_state``), so R1 is LeRobot's
recipe, not a re-implementation.
What is added around it is ila's run bookkeeping:

* run dir ``artifacts/runs/R1_lerobot_dp/s<seed>/`` (``--purpose test``:
  ``artifacts/runs/R1_lerobot_dp_test/s<seed>/``; ``smoke``:
  ``artifacts/runs/smoke/R1_lerobot_dp/``);
* ``status.json`` (state, step / total, steps/s, ETA in s, PID), ``train_log.jsonl``;
* ``manifest.json``: config SHA-256 against ``ila.train.manifest.FROZEN_CONFIG_SHA256`` (read by
  ``ast``, not imported), the dataset path and its ``.build.json`` SHA-256 (and every dataset file
  re-hashed against the build record), the ``ila`` commit and clean-tree flag, the P8 commit,
  lerobot / torch versions, host and GPU, wall clock and GPU-hours (one GPU: wall-clock hours of
  this process), every checkpoint's SHA-256s;
* ``checkpoints/<step>/`` every 10 000 steps (LeRobot's layout: ``pretrained_model/`` with
  ``model.safetensors`` and the processors, ``training_state/`` with optimizer, scheduler, step and
  the python / numpy / torch / CUDA RNG states), written to a temporary directory and renamed.

**Resume** (P3-D1 §4, interruption rule): an existing run dir resumes from its last complete
checkpoint: weights, optimizer, scheduler and RNG state restored by ``load_training_state``, the
data order by ``compute_sampler_state``; the resume is logged in the manifest. GPU training is not
claimed bit-identical to an uninterrupted run (P4-D1 item 8).

**Guards.** ``--purpose train`` refuses a dirty tree, a config whose SHA-256 differs from the frozen
table, any step/episode override, and a lerobot other than 0.6.1. Every purpose refuses a LeRobot
default that differs from ``lerobot_defaults_kept`` in the config, and (leak audit R1) any policy
input feature other than ``observation.image`` and ``observation.state``. ``HF_HUB_OFFLINE=1``;
wandb is off. Examples::

    .venv-lerobot/bin/python scripts/r1/train_r1.py --seed 0
    .venv-lerobot/bin/python scripts/r1/train_r1.py --seed 0 --purpose test --max-steps 300
    .venv-lerobot/bin/python scripts/r1/train_r1.py --purpose smoke --episodes 0-7 --steps 10000
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # never touch the Hub
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import argparse  # noqa: E402
import ast  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "train" / "r1_lerobot_dp.yaml"
CONFIG_REL = "configs/train/r1_lerobot_dp.yaml"
MANIFEST_PY = REPO_ROOT / "src" / "ila" / "train" / "manifest.py"
PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"
P8_DIR = REPO_ROOT / "external" / "rl-land-on-moving-deck"
RUNS_ROOT = REPO_ROOT / "artifacts" / "runs"
ARM = "R1_lerobot_dp"
LEROBOT_VERSION = "0.6.1"
INPUT_FEATURES = frozenset({"observation.image", "observation.state"})
OUTPUT_FEATURES = frozenset({"action"})
#: Steps of each invocation excluded from the steady-state rate (worker spawn, cuDNN autotune).
WARMUP_STEPS = 20


# --------------------------------------------------------------------------- small utilities


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Sorted, indented JSON with a trailing newline, replaced atomically (ila's format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
    tmp.replace(path)


def log_event(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"unix": time.time(), **record}, sort_keys=True, default=str) + "\n")


def frozen_config_sha256(rel: str) -> str | None:
    """``FROZEN_CONFIG_SHA256[rel]`` from ``src/ila/train/manifest.py`` (parsed, not imported)."""
    tree = ast.parse(MANIFEST_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == "FROZEN_CONFIG_SHA256" and value is not None:
            table: dict[str, str] = ast.literal_eval(value)
            return table.get(rel)
    raise RuntimeError(f"{MANIFEST_PY}: FROZEN_CONFIG_SHA256 not found")


def data_root_from_paths(path: Path = PATHS_FILE) -> Path:
    """Resolve ``data_root`` as ``ila.p8_adapter.load_paths`` does (``${key}``, repo-relative)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = re.sub(r"\$\{(\w+)\}", lambda m: resolved[m.group(1)], str(value))
    return (REPO_ROOT / resolved["data_root"]).resolve()


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln]
    return {
        "ila_commit": _git("rev-parse", "HEAD"),
        "ila_dirty": bool(dirty),
        "ila_dirty_paths": dirty[:50],
        "p8_commit": _git("rev-parse", "HEAD", cwd=P8_DIR),
    }


def host_info(device: str) -> dict[str, Any]:
    import lerobot
    import torch

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "lerobot": lerobot.__version__,
        "device": device,
        "cuda": torch.version.cuda,
        "venv": sys.prefix,
        "env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
            )
        },
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        info["gpu"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(idx).total_memory
    return info


class Status:
    """``status.json`` in ila's format: state, step, total_steps, pid, steps_per_s, eta_s."""

    def __init__(self, path: Path, total_steps: int, **fields: Any) -> None:
        """Start the file (state ``starting``)."""
        self.path = path
        self.fields: dict[str, Any] = {
            "state": "starting",
            "step": 0,
            "total_steps": int(total_steps),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_unix": time.time(),
            "last_select_success": None,
            **fields,
        }
        self.write()

    def update(self, **fields: Any) -> None:
        """Merge fields and rewrite."""
        self.fields.update(fields)
        self.write()

    def progress(self, step: int, steps_per_s: float | None, **fields: Any) -> None:
        """Record progress and the ETA (s) at the current rate."""
        remaining = max(0, self.fields["total_steps"] - step)
        eta = None if not steps_per_s else remaining / steps_per_s
        self.update(state="running", step=step, steps_per_s=steps_per_s, eta_s=eta, **fields)

    def write(self) -> None:
        """Rewrite atomically."""
        self.fields["updated_unix"] = time.time()
        write_json_atomic(self.path, self.fields)


# --------------------------------------------------------------------------- config


def parse_episodes(text: str | None) -> list[int] | None:
    """``"0-7"`` -> ``[0, ..., 7]``; ``"0,3,5"``; ``None`` -> all episodes."""
    if text is None:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _default_of(cls: Any, name: str) -> Any:
    f = {x.name: x for x in dataclasses.fields(cls)}[name]
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    raise KeyError(f"{cls.__name__}.{name} has no default")


def _plain(value: Any) -> Any:
    """Enum -> its name, tuple -> list, mapping values recursively (YAML-comparable)."""
    if hasattr(value, "name") and hasattr(value, "value") and not isinstance(value, str | int):
        return value.name
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def check_lerobot_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assert every ``lerobot_defaults_kept`` value equals the installed lerobot's default.

    Returns:
        ``{name: installed default}`` for the record.

    Raises:
        RuntimeError: On any difference (a different lerobot must not train R1 silently).
    """
    import lerobot
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.scripts import lerobot_train
    from lerobot.transforms.transforms import ImageTransformsConfig

    kept = cfg["lerobot_defaults_kept"]
    diffs: list[str] = []
    seen: dict[str, Any] = {}

    def check(name: str, have: Any, want: Any) -> None:
        seen[name] = have
        if have != want:
            diffs.append(f"{name}: installed {have!r} != config {want!r}")

    check("lerobot.__version__", lerobot.__version__, cfg["framework"]["version"])
    for key, want in kept["policy"].items():
        check(f"DiffusionConfig.{key}", _plain(_default_of(DiffusionConfig, key)), want)
    preset = DiffusionConfig().get_optimizer_preset()
    check("optimizer_preset.type", preset.type, kept["optimizer_preset"]["type"])
    check(
        "optimizer_preset.grad_clip_norm",
        preset.grad_clip_norm,
        kept["optimizer_preset"]["grad_clip_norm"],
    )
    t = dict(kept["train"])
    check(
        "TrainPipelineConfig.seed", _default_of(TrainPipelineConfig, "seed"), t.pop("seed_default")
    )
    for key, want in t.items():
        check(f"TrainPipelineConfig.{key}", _default_of(TrainPipelineConfig, key), want)
    d = kept["dataset"]
    check(
        "DatasetConfig.use_imagenet_stats",
        _default_of(DatasetConfig, "use_imagenet_stats"),
        d["use_imagenet_stats"],
    )
    check("DatasetConfig.eval_split", _default_of(DatasetConfig, "eval_split"), d["eval_split"])
    check(
        "ImageTransformsConfig.enable",
        _default_of(ImageTransformsConfig, "enable"),
        d["image_transforms_enable"],
    )
    # EMA: lerobot 0.6.1's diffusion policy and training loop have none.
    import inspect

    from lerobot.policies.diffusion import modeling_diffusion

    ema_found = any(
        re.search(r"\bEMA|\bema\b|ema_model|AveragedModel", inspect.getsource(m))
        for m in (modeling_diffusion, lerobot_train)
    )
    check("ema_in_lerobot_diffusion_or_train", "found" if ema_found else "none", kept["ema"])
    for key, want in cfg.get("lerobot_defaults_overridden", {}).get("train", {}).items():
        have = _default_of(TrainPipelineConfig, key)
        check(f"TrainPipelineConfig.{key} (overridden)", have, want)
    if diffs:
        raise RuntimeError("LeRobot defaults differ from the R1 config:\n  " + "\n  ".join(diffs))
    return seen


def build_configs(
    cfg: dict[str, Any],
    *,
    dataset_root: Path,
    episodes: list[int] | None,
    seed: int,
    device: str,
    steps: int,
    save_freq: int,
    run_dir: Path,
    num_workers: int | None = None,
) -> Any:
    """Build LeRobot's ``TrainPipelineConfig`` with ``DiffusionConfig`` from the YAML."""
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    pol = {k: v for k, v in cfg["policy"].items() if k != "type"}
    if cfg["policy"]["type"] != "diffusion":
        raise ValueError(f"policy type {cfg['policy']['type']!r}")
    pol["crop_shape"] = tuple(pol["crop_shape"])
    pol["down_dims"] = tuple(pol["down_dims"])
    policy_cfg = DiffusionConfig(**pol, device=device)
    tr = cfg["train"]
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=cfg["data"]["repo_id"], root=str(dataset_root), episodes=episodes
        ),
        policy=policy_cfg,
        output_dir=None,
        seed=seed,
        batch_size=int(tr["batch_size"]),
        steps=steps,
        save_freq=save_freq,
        wandb=WandBConfig(enable=bool(tr["wandb"]["enable"])),
    )
    argv = sys.argv
    try:
        sys.argv = argv[:1]  # validate() re-reads the CLI for --policy.path etc.; there is none
        train_cfg.validate()
    finally:
        sys.argv = argv
    train_cfg.output_dir = run_dir  # after validate(): the run dir may exist (resume)
    # Loader workers: the config's (12; LeRobot's default 4 is asserted in
    # check_lerobot_defaults), or a test run's measurement override (recorded in the manifest).
    train_cfg.num_workers = int(tr["num_workers"] if num_workers is None else num_workers)
    return train_cfg


def dataset_record(data_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Dataset path, build-record SHA-256, and every listed file re-hashed against the record."""
    root = data_root / "lerobot" / cfg["data"]["dataset"]
    build = root.with_name(root.name + ".build.json")
    rec = json.loads(build.read_text(encoding="utf-8"))
    mismatched = [rel for rel, sha in rec["files"].items() if sha256_file(root / rel) != sha]
    if mismatched:
        raise RuntimeError(f"{root}: files differ from the build record: {mismatched}")
    dp_sha = frozen_config_sha256("configs/train/dp.yaml")
    if rec["dp_config"]["sha256"] != dp_sha:
        raise RuntimeError("the dataset was not built from dp's frozen episodes")
    return {
        "path": str(root),
        "build_json": str(build),
        "build_json_sha256": sha256_file(build),
        "files_verified": sorted(rec["files"]),
        "episodes_in_dataset": rec["kept"],
        "frames_in_dataset": rec["frames"],
        "dp_config_sha256": rec["dp_config"]["sha256"],
        "store_sha256": rec["store"]["sha256"],
    }


# --------------------------------------------------------------------------- checkpoints


def find_resume(run_dir: Path) -> Path | None:
    """The last complete checkpoint (``training_state/training_step.json`` present)."""
    ckpts = sorted(
        p
        for p in (run_dir / "checkpoints").glob("[0-9]*")
        if p.is_dir() and (p / "training_state" / "training_step.json").is_file()
    )
    return ckpts[-1] if ckpts else None


def checkpoint_record(ckpt_dir: Path, step: int) -> dict[str, Any]:
    """Return ila's checkpoint entry; ``sha256`` is ``pretrained_model/model.safetensors``'s."""
    files = {
        p.relative_to(ckpt_dir).as_posix(): sha256_file(p)
        for p in sorted((ckpt_dir / "pretrained_model").rglob("*"))
        if p.is_file()
    }
    return {
        "kind": "full",
        "file": ckpt_dir.name,
        "stage": 0,
        "stage_step": step,
        "global_step": step,
        "sha256": files["pretrained_model/model.safetensors"],
        "sha256_of": "pretrained_model/model.safetensors",
        "files": files,
        "weights": "raw (lerobot 0.6.1 has no EMA)",
        "unix": time.time(),
    }


# --------------------------------------------------------------------------- main


def main() -> int:  # noqa: C901, PLR0912, PLR0915 (one linear job script)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--purpose", choices=["train", "test", "smoke"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="test/smoke: schedule length")
    parser.add_argument(
        "--save-freq", type=int, default=None, help="test/smoke: checkpoint cadence"
    )
    parser.add_argument(
        "--episodes", default=None, help="test/smoke: e.g. 0-7 (dataset episode indices)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="test: stop after this step")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="test only: loader workers (measurement)"
    )
    parser.add_argument("--allow-dirty", action="store_true", help="train: record and continue")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg["arm"] != ARM:
        raise SystemExit(f"{CONFIG}: arm {cfg['arm']!r}")
    config_sha = sha256_file(CONFIG)
    frozen_sha = frozen_config_sha256(CONFIG_REL)
    git = git_state()
    serious = args.purpose == "train"
    if serious:
        if args.seed not in cfg["seeds"]:
            raise SystemExit(f"seed {args.seed} not in the config's seeds {cfg['seeds']}")
        overrides = (args.steps, args.save_freq, args.episodes, args.max_steps, args.num_workers)
        if any(v is not None for v in overrides):
            raise SystemExit("a train run takes the config's steps, cadence and episodes")
        if config_sha != frozen_sha:
            raise SystemExit(f"config SHA-256 {config_sha} != frozen {frozen_sha}")
        if git["ila_dirty"] and not args.allow_dirty:
            raise SystemExit("the ila tree is dirty; commit first (or --allow-dirty, recorded)")
    elif (args.max_steps is not None or args.num_workers is not None) and args.purpose != "test":
        raise SystemExit("--max-steps and --num-workers are for test runs")
    steps = int(cfg["train"]["steps"]) if args.steps is None else args.steps
    save_freq = int(cfg["train"]["save_freq"]) if args.save_freq is None else args.save_freq
    episodes = parse_episodes(args.episodes)
    budget = steps if args.max_steps is None else min(steps, args.max_steps)
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    elif args.purpose == "train":
        run_dir = RUNS_ROOT / ARM / f"s{args.seed}"
    elif args.purpose == "test":
        run_dir = RUNS_ROOT / f"{ARM}_test" / f"s{args.seed}"
    else:
        run_dir = RUNS_ROOT / "smoke" / ARM
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    man_file = run_dir / "manifest.json"
    t_start = time.perf_counter()

    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from lerobot.common.train_utils import (
        get_step_checkpoint_dir,
        load_training_state,
        save_checkpoint,
        should_save_checkpoint,
        update_last_checkpoint,
    )
    from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import cycle
    from safetensors.torch import load_file

    defaults = check_lerobot_defaults(cfg)
    data_root = data_root_from_paths()
    data_rec = dataset_record(data_root, cfg)
    train_cfg = build_configs(
        cfg,
        dataset_root=Path(data_rec["path"]),
        episodes=episodes,
        seed=args.seed,
        device=args.device,
        steps=steps,
        save_freq=save_freq,
        run_dir=run_dir,
        num_workers=args.num_workers,
    )

    manifest: dict[str, Any]
    if man_file.is_file():
        manifest = json.loads(man_file.read_text(encoding="utf-8"))
        if (manifest["config"]["sha256"], manifest["run_seed"], manifest["purpose"]) != (
            config_sha,
            args.seed,
            args.purpose,
        ):
            raise SystemExit(f"{run_dir}: written by another config, seed or purpose")
    else:
        manifest = {
            "arm": ARM,
            "kind": "onnx_dp",
            "tag": cfg["tag"],
            "run_seed": args.seed,
            "purpose": args.purpose,
            "selectable": args.purpose == "train",
            "sigma": None,
            "config": {
                "files": [
                    {
                        "path": CONFIG_REL,
                        "sha256": config_sha,
                        "frozen_sha256": frozen_sha,
                        "matches": config_sha == frozen_sha,
                    }
                ],
                "all_frozen": config_sha == frozen_sha,
                "sha256": config_sha,
            },
            "config_resolved": cfg,
            "lerobot_train_config": train_cfg.to_dict(),
            "lerobot_defaults_verified": defaults,
            "data": data_rec,
            "episodes_override": episodes,
            "steps_override": args.steps,
            "checkpoint_every_override": args.save_freq,
            "max_steps": args.max_steps,
            "num_workers_override": args.num_workers,
            "total_steps": steps,
            "stages": [{"index": 0, "steps": steps, "warmup": 500, "checkpoint_every": save_freq}],
            "optimizer": {
                "name": "adam (lerobot DiffusionConfig preset; torch.optim.Adam, coupled L2)",
                "lr": train_cfg.optimizer.lr,
                "betas": list(train_cfg.optimizer.betas),
                "eps": train_cfg.optimizer.eps,
                "weight_decay": train_cfg.optimizer.weight_decay,
                "grad_clip_norm": train_cfg.optimizer.grad_clip_norm,
                "schedule": f"{train_cfg.scheduler.name} (diffusers get_scheduler), "
                f"{train_cfg.scheduler.num_warmup_steps} warm-up steps",
            },
            "precision": "fp32 (use_amp false)",
            "ema": None,
            "batch_size": train_cfg.batch_size,
            "checkpoints": [],
            "resumes": [],
            "segments": [],
        }
    host = host_info(args.device)
    manifest.setdefault("invocations", []).append({"git": git, "host": host, "unix": time.time()})
    manifest.setdefault("git", git)
    manifest.setdefault("host", host)
    if (manifest["git"]["ila_commit"], manifest["git"]["ila_dirty_paths"]) != (
        git["ila_commit"],
        git["ila_dirty_paths"],
    ):
        manifest["code_changed_since_start"] = True
    manifest["state"] = "running"
    status = Status(
        run_dir / "status.json", budget, arm=ARM, run_seed=args.seed, purpose=args.purpose
    )
    write_json_atomic(man_file, manifest)

    # ---- lerobot_train.train(), step for step -------------------------------------------
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision=None,  # use_amp false, DiffusionConfig has no dtype -> fp32
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        cpu=args.device == "cpu",
    )
    set_seed(train_cfg.seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True  # cudnn_deterministic false
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset, eval_dataset = make_train_eval_datasets(train_cfg)
    if eval_dataset is not None:
        raise SystemExit("eval_split must stay 0 (no held-out split in R1)")
    policy = make_policy(
        cfg=train_cfg.policy, ds_meta=dataset.meta, rename_map=train_cfg.rename_map
    )
    # Leak audit R1: pixels and the 13-d proprio in, the action out; nothing else.
    in_feats = set(train_cfg.policy.input_features)
    out_feats = set(train_cfg.policy.output_features)
    if in_feats != INPUT_FEATURES or out_feats != OUTPUT_FEATURES:
        raise SystemExit(f"policy features {sorted(in_feats)} -> {sorted(out_feats)}")
    if train_cfg.policy.env_state_feature is not None:
        raise SystemExit("an environment_state feature reached the policy")
    if tuple(train_cfg.policy.robot_state_feature.shape) != (13,):
        raise SystemExit(f"state shape {train_cfg.policy.robot_state_feature.shape}")
    manifest["input_features"] = {
        k: list(v.shape) for k, v in train_cfg.policy.input_features.items()
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy, pretrained_path=None, dataset_stats=dataset.meta.stats
    )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)
    step = 0
    resume = find_resume(run_dir)
    if resume is not None:
        # Interruption rule (P3-D1 §4): weights, optimizer, scheduler, RNG, then the data order.
        policy.load_state_dict(load_file(str(resume / "pretrained_model" / "model.safetensors")))
        step, optimizer, lr_scheduler = load_training_state(resume, optimizer, lr_scheduler)
        manifest["resumes"].append({"from": resume.name, "global_step": step, "unix": time.time()})
        log_event(log_path, {"event": "resume", "from": resume.name, "step": step})
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(train_cfg.policy, "drop_n_last_frames", 0),
        shuffle=True,
        seed=train_cfg.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    if step > 0:
        sampler.load_state_dict(compute_sampler_state(step, len(sampler), train_cfg.batch_size, 1))
    workers = train_cfg.num_workers > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_cfg.num_workers,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=None,
        prefetch_factor=train_cfg.prefetch_factor if workers else None,
        persistent_workers=train_cfg.persistent_workers and workers,
        multiprocessing_context=train_cfg.dataloader_multiprocessing_context if workers else None,
    )
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()
    meters = {
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        meters["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")
    tracker = MetricsTracker(
        train_cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        meters,
        initial_step=step,
        accelerator=accelerator,
    )
    manifest["dataset_frames_used"] = dataset.num_frames
    manifest["dataset_episodes_used"] = dataset.num_episodes
    manifest["sampler_len"] = len(sampler)
    write_json_atomic(man_file, manifest)
    checkpoints: list[dict[str, Any]] = list(manifest["checkpoints"])
    first = step
    state = "done"
    log_freq = int(train_cfg.log_freq)
    t_loop = time.perf_counter()
    t_steady: float | None = None
    data_s = upd_s = 0.0
    window_t0 = time.perf_counter()
    window_steps = 0
    status.progress(step, None)
    while step < budget:
        t0 = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        tracker.dataloading_s = time.perf_counter() - t0
        tracker, _ = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            train_cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        step += 1
        window_steps += 1
        tracker.step()
        data_s += tracker.dataloading_s.val
        upd_s += tracker.update_s.val
        if step - first == WARMUP_STEPS:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_steady = time.perf_counter()
        loss = float(tracker.loss.val)
        if not math.isfinite(loss):
            state = "diverged"
            log_event(log_path, {"event": "diverged", "step": step, "loss": loss})
            break
        if step % log_freq == 0 or step == budget:
            rate = window_steps / (time.perf_counter() - window_t0)
            rec = {
                "event": "train",
                "step": step,
                "loss_avg": tracker.loss.avg,
                "grad_norm_avg": tracker.grad_norm.avg,
                "lr": tracker.lr.val,
                "update_s_avg": tracker.update_s.avg,
                "dataloading_s_avg": tracker.dataloading_s.avg,
                "steps_per_s_window": rate,
                "epochs": tracker.epochs,
            }
            log_event(log_path, rec)
            status.progress(step, rate, loss=tracker.loss.avg, stage=0)
            print(json.dumps(rec), flush=True)
            tracker.reset_averages()
            window_t0, window_steps = time.perf_counter(), 0
        final = step == budget and args.purpose != "train"
        if should_save_checkpoint(step, save_freq, steps) or final:
            final_dir = get_step_checkpoint_dir(run_dir, steps, step)
            if not final_dir.is_dir():
                tmp = final_dir.with_name(".tmp_" + final_dir.name)
                shutil.rmtree(tmp, ignore_errors=True)
                save_checkpoint(
                    checkpoint_dir=tmp,
                    step=step,
                    cfg=train_cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=train_cfg.batch_size,
                )
                tmp.replace(final_dir)
                update_last_checkpoint(final_dir)
                entry = checkpoint_record(final_dir, step)
                checkpoints.append(entry)
                manifest["checkpoints"] = checkpoints
                write_json_atomic(man_file, manifest)
                log_event(
                    log_path,
                    {
                        "event": "checkpoint",
                        **{k: entry[k] for k in ("file", "global_step", "sha256")},
                    },
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()
    del dl_iter, dataloader  # shut the persistent loader workers down before exit
    n_steps = step - first
    loop_s = t_end - t_loop
    steady = None
    if t_steady is not None and n_steps > WARMUP_STEPS:
        steady = (n_steps - WARMUP_STEPS) / (t_end - t_steady)
    manifest["segments"].append(
        {
            "num_workers": train_cfg.num_workers,
            "first_step": first,
            "steps": n_steps,
            "loop_s": loop_s,
            "steps_per_s": n_steps / loop_s if loop_s else None,
            "steps_per_s_after_warmup": steady,
            "warmup_steps_excluded": WARMUP_STEPS,
            "dataloading_s_mean": data_s / n_steps if n_steps else None,
            "update_s_mean": upd_s / n_steps if n_steps else None,
        }
    )
    wall = time.perf_counter() - t_start
    prior = float(manifest.get("wall_clock_s", 0.0))
    manifest.update(
        {
            "state": state,
            "global_step": step,
            "wall_clock_s": prior + wall,
            "gpu_hours": (prior + wall) / 3600.0 if device.type == "cuda" else 0.0,
            "loop_s_total": float(manifest.get("loop_s_total", 0.0)) + loop_s,
            "steps_timed_total": int(manifest.get("steps_timed_total", 0)) + n_steps,
            "finished_unix": time.time(),
        }
    )
    if manifest["loop_s_total"]:
        manifest["steps_per_s_loop"] = manifest["steps_timed_total"] / manifest["loop_s_total"]
    if device.type == "cuda":
        manifest["gpu_mem_max_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json_atomic(man_file, manifest)
    status.update(state=state, step=step, eta_s=0.0 if state == "done" else None)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "state": state,
                "step": step,
                "steps_per_s": n_steps / loop_s if loop_s else None,
                "steps_per_s_after_warmup": steady,
                "wall_clock_s": wall,
            }
        ),
        flush=True,
    )
    accelerator.end_training()
    return 0 if state == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
