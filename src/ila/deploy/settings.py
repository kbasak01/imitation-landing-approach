"""Phase 9 bench settings: ``configs/deploy/bench.yaml`` -> a frozen :class:`BenchSettings`.

Every module of :mod:`ila.deploy` reads its numbers from here, so the thresholds P3-D1 §6 H5
and §7 state live in one reviewed file. Simulation only. Times are model scale (lambda =
1/25): one 30 Hz control step is 33.333 ms model scale (166.7 ms full scale).

Provider names (P3-D1 §6 H5(a); P9-D1):

* ``torch_cpu_fp32``: torch eager, CPU, fp32, one intra-op thread, batch 1. The **reference**
  of every parity test and the fallback of P3-D1 §6 ("If no provider passes ... §7's d uses
  the p99 of the torch CPU fp32 single-sample path").
* scored: ``ort_cpu_fp32``, ``ort_cuda_fp32``, ``torch_cuda_fp32``, ``ort_trt_fp16``.
* unscored, reported beside: ``ort_trt_fp32`` (TensorRT with TF32 held off), ``ort_cuda_fp16``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from ila import p8_adapter

__all__ = [
    "ALL_PROVIDERS",
    "BENCH_YAML",
    "DEPLOY_ARMS",
    "GPU_PROVIDERS",
    "BenchSettings",
    "Precision",
    "Provider",
    "load_settings",
    "precision_of",
]

#: The committed bench config.
BENCH_YAML: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "deploy" / "bench.yaml"

#: Every provider this module knows, in report order (reference first).
type Provider = Literal[
    "torch_cpu_fp32",
    "ort_cpu_fp32",
    "ort_cuda_fp32",
    "torch_cuda_fp32",
    "ort_trt_fp16",
    "ort_trt_fp32",
    "ort_cuda_fp16",
]
ALL_PROVIDERS: Final[tuple[Provider, ...]] = (
    "torch_cpu_fp32",
    "ort_cpu_fp32",
    "ort_cuda_fp32",
    "torch_cuda_fp32",
    "ort_trt_fp16",
    "ort_trt_fp32",
    "ort_cuda_fp16",
)

#: Providers that place compute on the GPU.
GPU_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"ort_cuda_fp32", "torch_cuda_fp32", "ort_trt_fp16", "ort_trt_fp32", "ort_cuda_fp16"}
)

#: The two exported arms (P3-D1 §7).
DEPLOY_ARMS: Final[tuple[str, str]] = ("bc_dagger", "dp")

type Precision = Literal["fp32", "fp16"]


def precision_of(provider: str) -> Precision:
    """The graph precision a provider runs (``*_fp16`` -> ``fp16``)."""
    return "fp16" if provider.endswith("_fp16") else "fp32"


@dataclass(frozen=True)
class BenchSettings:
    """``configs/deploy/bench.yaml``, validated.

    Attributes:
        arm_seeds: Arm -> training seed exported (0 for both; P3-D1 §3 "seed 0 only").
        dp_ta: ``Ta`` of the latency arm and of the closed-loop parity flights (8 control steps,
            0.267 s model scale, 1.33 s full scale).
        dp_k: ``K`` of the exported unrolled graph and of every ``dp`` flight (10).
        selection_pins: ``results/e08/selection_pins.json``.
        opset: ONNX opset (17).
        precisions: Graph precisions exported (``fp32``, ``fp16``).
        out_dir: Graph directory (gitignored ``artifacts/deploy``).
        input_list: The list the open-loop inputs come from (``select``).
        input_policy: The policy flown to record them (``ppo``, P8 s0).
        n_inputs: Open-loop inputs (1 000).
        draw_seed: Their draw seed.
        noise_episode_key: Episode-key slot of input ``i``'s noise key ``(run_seed, key, i)``.
        eps_max_abs: Judged per-step epsilon-hat max-abs (1e-4).
        action_max_abs: Judged ``bc_dagger`` action max-abs (1e-4).
        chunk_reported: Final-chunk max-abs printed beside (1e-3), never judged.
        closed_loop_max_flips: Outcome flips allowed on the 150 ``select`` episodes (3).
        reference: Reference provider (``torch_cpu_fp32``).
        scored: The four scored providers.
        unscored: Variants reported, not scored.
        workers_cpu: Closed-loop processes for CPU providers.
        workers_gpu: Closed-loop processes for GPU providers (one CUDA context each).
        chunk: Episodes per closed-loop task.
        warmup_iters: Latency warm-up calls (>= 200).
        timed_iters: Latency timed calls (>= 2 000).
        batch_sizes: Latency batch sizes (1 and the evaluation batch, 24).
        threads: Intra-op CPU threads of every latency configuration (1).
        k_loop: ``K`` values of the full-loop latency rows.
        reference_batch_sizes: Batch sizes timed for the reference provider (1).
        control_period_ms: One control step, ms model scale (33.333).
        h5a_threshold_ms: H5(a)'s p99 bound as written (33.3 ms model scale).
        results_dir: ``results/latency``.
    """

    arm_seeds: Mapping[str, int]
    dp_ta: int
    dp_k: int
    selection_pins: Path
    opset: int
    precisions: tuple[Precision, ...]
    out_dir: Path
    input_list: str
    input_policy: str
    n_inputs: int
    draw_seed: int
    noise_episode_key: int
    eps_max_abs: float
    action_max_abs: float
    chunk_reported: float
    closed_loop_max_flips: int
    reference: Provider
    scored: tuple[Provider, ...]
    unscored: tuple[Provider, ...]
    workers_cpu: int
    workers_gpu: int
    chunk: int
    warmup_iters: int
    timed_iters: int
    batch_sizes: tuple[int, ...]
    threads: int
    k_loop: tuple[int, ...]
    reference_batch_sizes: tuple[int, ...]
    control_period_ms: float
    h5a_threshold_ms: float
    results_dir: Path

    def __post_init__(self) -> None:
        """Validate against P3-D1 (never relaxing it).

        Raises:
            ValueError: On a threshold, iteration count, list or provider that P3-D1 does not
                allow.
        """
        if set(self.arm_seeds) != set(DEPLOY_ARMS):
            raise ValueError(f"arms must be {DEPLOY_ARMS}, got {sorted(self.arm_seeds)}")
        if self.opset != 17:
            raise ValueError(f"opset must be 17, got {self.opset}")
        if self.input_list != "select":
            raise ValueError("open-loop inputs come from select only (P3-D1 §6 H5(a))")
        if self.n_inputs < 1000:
            raise ValueError(f"P3-D1 §6 H5(a) needs 1 000 inputs, got {self.n_inputs}")
        if self.noise_episode_key >= 0:
            raise ValueError("the parity noise key's episode slot must be negative")
        if self.eps_max_abs > 1e-4 or self.action_max_abs > 1e-4:
            raise ValueError("parity tolerances may not exceed P3-D1's 1e-4")
        if self.closed_loop_max_flips > 3:
            raise ValueError("closed-loop parity allows at most 3 flips (P3-D1 §3, §6)")
        if self.warmup_iters < 200 or self.timed_iters < 2000:
            raise ValueError("latency needs >= 200 warm-up and >= 2 000 timed calls")
        if 1 not in self.batch_sizes or 1 not in self.reference_batch_sizes:
            raise ValueError("latency must time batch 1")
        if self.reference != "torch_cpu_fp32":
            raise ValueError("the reference is torch fp32 CPU (P3-D1 §6 H5(a))")
        want = {"ort_cpu_fp32", "ort_cuda_fp32", "torch_cuda_fp32", "ort_trt_fp16"}
        if set(self.scored) != want:
            raise ValueError(f"scored providers must be {sorted(want)}, got {self.scored}")
        if set(self.unscored) & (want | {self.reference}):
            raise ValueError("an unscored variant cannot also be scored or the reference")
        if abs(self.control_period_ms - 33.333) > 1e-9 or abs(self.h5a_threshold_ms - 33.3) > 1e-9:
            raise ValueError("control period 33.333 ms (P3-D1 §7) and H5(a) 33.3 ms (§6)")
        if self.dp_k != 10 or self.dp_ta != 8:
            raise ValueError("dp is exported and flown at K = 10, Ta = 8 (P3-D1 §6, §7)")

    @property
    def providers(self) -> tuple[Provider, ...]:
        """Reference, scored, then unscored, in :data:`ALL_PROVIDERS` order."""
        chosen = {self.reference, *self.scored, *self.unscored}
        return tuple(p for p in ALL_PROVIDERS if p in chosen)

    def is_scored(self, provider: str) -> bool:
        """Whether ``provider`` is one of the four scored providers."""
        return provider in self.scored

    @property
    def eval_batch(self) -> int:
        """The evaluation batch size (the largest latency batch, 24)."""
        return max(self.batch_sizes)


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else p8_adapter.REPO_ROOT / p


def from_mapping(raw: Mapping[str, Any]) -> BenchSettings:
    """Build :class:`BenchSettings` from a parsed ``bench.yaml``."""
    arms = raw["arms"]
    exp, inp, par, prov = raw["export"], raw["inputs"], raw["parity"], raw["providers"]
    cl, lat = raw["closed_loop"], raw["latency"]
    return BenchSettings(
        arm_seeds={str(a): int(v["seed"]) for a, v in arms.items()},
        dp_ta=int(arms["dp"]["Ta"]),
        dp_k=int(arms["dp"]["K"]),
        selection_pins=_rel(str(raw["selection_pins"])),
        opset=int(exp["opset"]),
        precisions=tuple(exp["precisions"]),
        out_dir=_rel(str(exp["out_dir"])),
        input_list=str(inp["list"]),
        input_policy=str(inp["policy"]),
        n_inputs=int(inp["n"]),
        draw_seed=int(inp["draw_seed"]),
        noise_episode_key=int(inp["noise_episode_key"]),
        eps_max_abs=float(par["eps_max_abs"]),
        action_max_abs=float(par["action_max_abs"]),
        chunk_reported=float(par["chunk_reported"]),
        closed_loop_max_flips=int(par["closed_loop_max_flips"]),
        reference=prov["reference"],
        scored=tuple(prov["scored"]),
        unscored=tuple(prov["unscored"]),
        workers_cpu=int(cl["workers_cpu"]),
        workers_gpu=int(cl["workers_gpu"]),
        chunk=int(cl["chunk"]),
        warmup_iters=int(lat["warmup_iters"]),
        timed_iters=int(lat["timed_iters"]),
        batch_sizes=tuple(int(b) for b in lat["batch_sizes"]),
        threads=int(lat["threads"]),
        k_loop=tuple(int(k) for k in lat["k_loop"]),
        reference_batch_sizes=tuple(int(b) for b in lat["reference_batch_sizes"]),
        control_period_ms=float(raw["control_period_ms"]),
        h5a_threshold_ms=float(raw["h5a_threshold_ms"]),
        results_dir=_rel(str(raw["results_dir"])),
    )


def load_settings(path: Path = BENCH_YAML) -> BenchSettings:
    """Read and validate ``configs/deploy/bench.yaml``."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return from_mapping(raw)
