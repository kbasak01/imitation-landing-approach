"""Phase 10: closed-loop outcome sensitivity of the Diffusion-Policy students (LIMITATIONS L4).

Phase 9 measured how often core ``dp`` s0 changes an episode's outcome when the identical
sampler runs on another device (P9-D2 item 1: 32-44 of 150 ``select`` episodes). This stage
measures the same quantity, by the same definition, for the ``dp_*`` ablations, with core ``dp``
flown again as the control. It is descriptive: no threshold, no verdict.

Definition (equal to Phase 9's for core ``dp``, :mod:`ila.deploy.closed_loop`):

* **Checkpoint**: each arm's seed-0 selected checkpoint, pinned in
  ``results/e08/selection_pins.json`` and resolved as Phases 8 and 9 resolve it: the arm's spec
  from ``configs/eval/matrix.yaml`` (:func:`matrix_arm_spec`, as :func:`ila.eval.arms.expand`
  builds it; tag ``core`` or ``ablation``), :func:`ila.eval.arms.resolve_checkpoint` over the
  committed ``selection.json`` and ``manifest.json``, the pin compared with that file's
  SHA-256, the checkpoint bytes re-hashed and its arm and contract variant checked
  (:func:`ila.deploy.export.load_arm`). EMA weights.
* **Inference**: selection's, ``Ta = 8`` (0.267 s model scale, 1.33 s full scale), ``K = 10``
  DDIM-trailing steps, no temporal ensembling. Confirmed from the ``selection.json``
  ``inference`` (else the manifest's config, :func:`ila.train.select.inference_settings`) and
  from the checkpoint's own config, which is what the student flies.
* **Reference**: the 150 ``select`` episodes on ``torch_cpu_fp32``, selection's own path
  (:func:`ila.train.select.fly_checkpoint` through :func:`ila.deploy.closed_loop.fly_provider`;
  CPU fp32 single-sample, one torch thread per worker, render ``full``). Its rows file is
  compared byte for byte with the committed selection rows of the same checkpoint
  (:func:`ila.deploy.closed_loop.rows_files_identical`; P2-D3 item 1 says they are identical).
* **Perturbation**: the same episodes on ``torch_cuda_fp32``: the identical Python sampler
  (``ProviderDPStudent``: ``ChunkedDiffusionStudent``'s executor, noise keys
  ``(run_seed, episode_seed, t)``), the EMA networks in torch eager on the GPU, fp32, TF32 off,
  batch 1 per worker (:func:`ila.deploy.closed_loop.provider_policy`). No ONNX graph is used.
* **Flips**: episodes whose P8 outcome class differs between the two flights, in list order
  (:func:`ila.deploy.closed_loop.count_flips`); ``lost`` = reference ``success`` and perturbed
  not, ``gained`` = the converse. Outcome classes are P8's, matched in P8's order.
* **Control**: core ``dp`` s0 must reproduce Phase 9's ``torch_cuda_fp32`` flight
  (``results/latency/closed_loop/dp/torch_cuda_fp32.json``: 32 flips); the record says whether
  it does (flip count, flipped episodes, rows bytes), and the log says so if it does not.

Outputs under ``results/phase10/dp_sensitivity/``: ``<arm>_s0_torch_cpu_fp32_rows.csv`` and
``<arm>_s0_torch_cuda_fp32_rows.csv`` (the runner's columns, CRLF as selection writes them),
``<arm>_s0.json`` (the record, written last: an arm with a record is done and is skipped unless
forced), ``sensitivity.csv`` (one row per arm, LF, fixed formatting, re-rendered from the
records by :func:`check`) and ``run_info.json`` (commit, host, wall clock per invocation).

The stage refuses a dirty tree outside ``results/phase10/`` unless told otherwise (smoke runs
only), because each record carries the commit it ran at. Simulation only: PyBullet physics and
TinyRenderer frames; nothing here is a statement about a real vehicle or camera. Control steps
are 1/30 s model scale (lambda = 1/25).
"""

from __future__ import annotations

import csv
import datetime as dt
import functools
import gzip
import io
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from ila import p8_adapter
from ila.deploy import closed_loop as cl
from ila.deploy import export as ex
from ila.deploy.pipeline import DirtyTreeError
from ila.deploy.settings import BenchSettings, load_settings
from ila.eval import arms as ev_arms
from ila.eval.score import OUTCOMES

__all__ = [
    "DIRTY_ALLOWED",
    "DP_ARMS",
    "SENSITIVITY_COLUMNS",
    "SENSITIVITY_YAML",
    "ArmInfo",
    "Flight",
    "SensitivityError",
    "SensitivitySettings",
    "arm_spec",
    "check",
    "check_against_bench",
    "from_mapping",
    "git_checked",
    "load_sensitivity_settings",
    "matrix_arm_spec",
    "outcome_counts",
    "pinned_selection",
    "record_path",
    "render_csv",
    "resolve_arm",
    "rows_path",
    "run_sensitivity",
    "sensitivity_csv",
    "stage_sensitivity",
    "summarise",
]

#: The committed config.
SENSITIVITY_YAML: Final[Path] = p8_adapter.REPO_ROOT / "configs" / "deploy" / "sensitivity.yaml"

#: The Diffusion-Policy students this stage may fly (core ``dp`` and its trained ablations; the
#: ONNX ``R1_lerobot_dp`` reference arm is not one of them).
DP_ARMS: Final[tuple[str, ...]] = (
    "dp",
    "dp_mix",
    "dp_pid",
    "dp_n100",
    "dp_n300",
    "dp_minus_last_action",
)

#: Uncommitted paths under this prefix do not make the tree dirty for this stage.
DIRTY_ALLOWED: Final[str] = "results/phase10/"

#: ``sensitivity.csv`` columns.
SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "seed",
    "checkpoint_sha256",
    "n",
    "reference_equals_selection_rows",
    "flips",
    "flip_rate",
    "lost",
    "gained",
    "success_ref",
    "success_perturbed",
    *(f"ref_{o}" for o in OUTCOMES),
    *(f"perturbed_{o}" for o in OUTCOMES),
)

CSV_FILE: Final[str] = "sensitivity.csv"
RUN_INFO_FILE: Final[str] = "run_info.json"

#: The definition, as recorded in every record.
DEFINITION: Final[str] = (
    "outcome flips on the 150 select episodes between the arm's seed-0 pinned checkpoint flown "
    "on torch_cpu_fp32 (selection's CPU fp32 single-sample path; rows asserted byte-identical "
    "to the committed selection rows) and the identical Python sampler in torch eager on "
    "torch_cuda_fp32 (fp32, TF32 off, batch 1 per worker); Ta 8, K 10, no ensembling, EMA "
    "weights; Phase 9's definition for core dp (P9-D2 item 1). Descriptive: no threshold"
)

#: Record fields recomputed from the rows files by :func:`check`.
_SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "n",
    "flips",
    "flip_rate",
    "flipped",
    "rows_text_differ",
    "lost",
    "gained",
    "success_ref",
    "success_perturbed",
    "outcomes_ref",
    "outcomes_perturbed",
    "transitions",
)


class SensitivityError(RuntimeError):
    """A setting, pin, selection record or inference setting is not what this stage needs."""


# --------------------------------------------------------------------------- settings


def _rel_to_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else p8_adapter.REPO_ROOT / p


@dataclass(frozen=True)
class SensitivitySettings:
    """``configs/deploy/sensitivity.yaml``, validated.

    Attributes:
        arms: Arms flown, in table order (a subset of :data:`DP_ARMS`; includes the control).
        seed: Training seed (0).
        ta: Executed actions per chunk (8 control steps; 0.267 s model, 1.33 s full scale).
        k: DDIM denoising steps (10).
        temporal_ensembling: Off (selection never ensembles).
        list_name: ``select`` (never a frozen list).
        n_episodes: Episodes in the list (150).
        reference: ``torch_cpu_fp32`` (selection's path).
        perturbed: ``torch_cuda_fp32``.
        workers_cpu: Processes of the reference flight (rows do not depend on it).
        workers_gpu: Processes of the perturbed flight (one CUDA context each).
        chunk: Episodes per runner task.
        selection_pins: ``results/e08/selection_pins.json``.
        bench_yaml: ``configs/deploy/bench.yaml`` (cross-checked, never edited).
        control_arm: ``dp``: must reproduce Phase 9's flight.
        control_record: Phase 9's ``closed_loop/dp/torch_cuda_fp32.json``.
        control_rows: Phase 9's ``closed_loop/dp/torch_cuda_fp32_rows.csv``.
        control_reference_rows: Phase 9's ``closed_loop/dp/torch_cpu_fp32_rows.csv``.
        results_dir: ``results/phase10/dp_sensitivity``.
        source: The YAML file read (its SHA-256 goes into ``run_info.json``).
    """

    arms: tuple[str, ...]
    seed: int
    ta: int
    k: int
    temporal_ensembling: bool
    list_name: str
    n_episodes: int
    reference: str
    perturbed: str
    workers_cpu: int
    workers_gpu: int
    chunk: int
    selection_pins: Path
    bench_yaml: Path
    control_arm: str
    control_record: Path
    control_rows: Path
    control_reference_rows: Path
    results_dir: Path
    source: Path

    def __post_init__(self) -> None:
        """Validate against the definition (module docstring).

        Raises:
            ValueError: On an arm, seed, inference setting, list or provider the definition
                does not allow.
        """
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError(f"arms must be non-empty and unique, got {self.arms}")
        unknown = [a for a in self.arms if a not in DP_ARMS]
        if unknown:
            raise ValueError(f"not Diffusion-Policy students of this stage: {unknown}")
        if self.control_arm != "dp" or self.control_arm not in self.arms:
            raise ValueError("the control is core dp and must be among the arms")
        if self.seed != 0:
            raise ValueError(f"seed 0 only (Phase 9's definition), got {self.seed}")
        if (self.ta, self.k, self.temporal_ensembling) != (8, 10, False):
            raise ValueError("selection inference is Ta = 8, K = 10, no ensembling")
        if self.list_name != "select" or self.n_episodes != 150:
            raise ValueError("the 150 select episodes only (never a frozen list)")
        if self.reference != "torch_cpu_fp32" or self.perturbed != "torch_cuda_fp32":
            raise ValueError("reference torch_cpu_fp32, perturbation torch_cuda_fp32")
        if min(self.workers_cpu, self.workers_gpu, self.chunk) < 1:
            raise ValueError("workers and chunk must be positive")

    @property
    def inference(self) -> dict[str, Any]:
        """``{"Ta", "K", "temporal_ensembling"}`` as ``selection.json`` records it."""
        return {"Ta": self.ta, "K": self.k, "temporal_ensembling": self.temporal_ensembling}


def from_mapping(raw: Mapping[str, Any], source: Path = SENSITIVITY_YAML) -> SensitivitySettings:
    """Build :class:`SensitivitySettings` from a parsed ``sensitivity.yaml``."""
    inf, ctl = raw["inference"], raw["control"]
    return SensitivitySettings(
        arms=tuple(str(a) for a in raw["arms"]),
        seed=int(raw["seed"]),
        ta=int(inf["Ta"]),
        k=int(inf["K"]),
        temporal_ensembling=bool(inf["temporal_ensembling"]),
        list_name=str(raw["list"]),
        n_episodes=int(raw["n_episodes"]),
        reference=str(raw["reference"]),
        perturbed=str(raw["perturbed"]),
        workers_cpu=int(raw["workers_cpu"]),
        workers_gpu=int(raw["workers_gpu"]),
        chunk=int(raw["chunk"]),
        selection_pins=_rel_to_repo(str(raw["selection_pins"])),
        bench_yaml=_rel_to_repo(str(raw["bench_yaml"])),
        control_arm=str(ctl["arm"]),
        control_record=_rel_to_repo(str(ctl["record"])),
        control_rows=_rel_to_repo(str(ctl["rows"])),
        control_reference_rows=_rel_to_repo(str(ctl["reference_rows"])),
        results_dir=_rel_to_repo(str(raw["results_dir"])),
        source=Path(source),
    )


def check_against_bench(settings: SensitivitySettings, bench: BenchSettings) -> None:
    """Require the values Phase 9's flights used (``bench.yaml``) where the definitions meet.

    Raises:
        ValueError: If a reference, worker count, chunk, pin file or ``dp`` inference setting
            differs from ``bench.yaml``'s.
    """
    pairs = {
        "reference": (settings.reference, bench.reference),
        "workers_cpu": (settings.workers_cpu, bench.workers_cpu),
        "workers_gpu": (settings.workers_gpu, bench.workers_gpu),
        "chunk": (settings.chunk, bench.chunk),
        "selection_pins": (settings.selection_pins.resolve(), bench.selection_pins.resolve()),
        "Ta": (settings.ta, bench.dp_ta),
        "K": (settings.k, bench.dp_k),
        "seed": (settings.seed, bench.arm_seeds["dp"]),
    }
    bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"sensitivity.yaml differs from bench.yaml: {bad}")


def load_sensitivity_settings(path: Path = SENSITIVITY_YAML) -> SensitivitySettings:
    """Read, validate and cross-check (against ``bench_yaml``) ``sensitivity.yaml``."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    s = from_mapping(raw, Path(path))
    check_against_bench(s, load_settings(s.bench_yaml))
    return s


def rows_path(settings: SensitivitySettings, arm: str, provider: str) -> Path:
    """``<results_dir>/<arm>_s<seed>_<provider>_rows.csv``."""
    return settings.results_dir / f"{arm}_s{settings.seed}_{provider}_rows.csv"


def record_path(settings: SensitivitySettings, arm: str) -> Path:
    """``<results_dir>/<arm>_s<seed>.json``."""
    return settings.results_dir / f"{arm}_s{settings.seed}.json"


# --------------------------------------------------------------------------- arms


@dataclass(frozen=True)
class ArmInfo:
    """One arm's pinned, verified checkpoint and what its reference is compared with.

    Attributes:
        arm: Arm.
        seed: Training seed.
        tag: Its matrix tag (``core`` or ``ablation``).
        checkpoint: The resolved checkpoint (bytes verified by :func:`resolve_arm`).
        selection_pin: Its pin in ``selection_pins.json`` (= the ``selection.json`` SHA-256).
        inference: Selection's inference, confirmed (``{"Ta", "K", "temporal_ensembling"}``).
        selection_rows: The selected candidate's ``select`` rows file, located as Phase 9
            locates it (:func:`ila.deploy.closed_loop._selection_rows_path`).
        committed_selection_rows: The committed copy beside ``selection.json`` (plain, or the
            ``.csv.gz`` Phase 7 committed), or ``None``.
    """

    arm: str
    seed: int
    tag: str
    checkpoint: ev_arms.Checkpoint
    selection_pin: str
    inference: Mapping[str, Any]
    selection_rows: Path
    committed_selection_rows: Path | None


def matrix_arm_spec(arm: str, raw: Mapping[str, Any] | None = None) -> ev_arms.ArmSpec:
    """An arm's :class:`ila.eval.arms.ArmSpec` from ``configs/eval/matrix.yaml`` alone.

    The spec :func:`ila.eval.arms.expand` builds (``tests/test_deploy_sensitivity.py`` checks
    equality for every arm here): a ``core_arms`` entry is tag ``core`` with its own seeds and
    ``Ta``/``K``; an ``ablations`` entry is tag ``ablation``, :data:`ila.eval.arms.ABLATION_SEEDS`
    seeds and, for a diffusion arm, core ``dp``'s inference; never ensembling. Read through
    :func:`ila.eval.stats.read_matrix_yaml` (SHA-256 checked); no frozen list is read, as
    ``ila.deploy`` reads none (``tests/test_deploy_parity.py``).

    Raises:
        SensitivityError: If the arm is in neither block, or core ``dp`` has no ``Ta``/``K``.
    """
    from ila.eval.stats import read_matrix_yaml

    m = read_matrix_yaml() if raw is None else raw
    core = {str(e["arm"]): e for e in m["core_arms"]}
    dp = core.get("dp", {})
    if "Ta" not in dp or "K" not in dp:
        raise SensitivityError("matrix.yaml: core dp has no Ta/K")
    core_inf = ev_arms.Inference(int(dp["Ta"]), int(dp["K"]), False)
    if arm in core:
        tag, n_seeds = "core", int(core[arm]["seeds"])
        e = core[arm]
        inf = ev_arms.Inference(int(e["Ta"]), int(e["K"]), False) if "Ta" in e else None
    elif arm in {str(e["arm"]) for e in m["ablations"]}:
        tag, n_seeds = "ablation", ev_arms.ABLATION_SEEDS
        inf = core_inf if ev_arms.is_diffusion(arm) else None
    else:
        raise SensitivityError(f"{arm}: not in configs/eval/matrix.yaml")
    return ev_arms.ArmSpec(
        arm=arm,
        tag=tag,
        seeds=tuple(range(n_seeds)),
        teacher=ev_arms.expected_teacher(arm),
        obs_variant=ev_arms.expected_obs_variant(arm),
        inference=inf if ev_arms.is_diffusion(arm) else None,
    )


def arm_spec(
    settings: SensitivitySettings, arm: str, raw: Mapping[str, Any] | None = None
) -> ev_arms.ArmSpec:
    """The arm's matrix spec, checked to be a diffusion arm at selection inference.

    Args:
        settings: Sensitivity settings.
        arm: An arm of ``settings.arms``.
        raw: The parsed ``matrix.yaml`` (default: read it, :func:`matrix_arm_spec`).

    Raises:
        SensitivityError: If the arm is not in the settings or the matrix's spec disagrees.
    """
    if arm not in settings.arms:
        raise SensitivityError(f"{arm!r} is not one of {settings.arms}")
    spec = matrix_arm_spec(arm, raw)
    want = ev_arms.Inference(settings.ta, settings.k, settings.temporal_ensembling)
    if not ev_arms.is_diffusion(arm) or spec.inference != want:
        raise SensitivityError(f"{arm}: matrix inference {spec.inference} != {want}")
    if settings.seed not in spec.seeds:
        raise SensitivityError(f"{arm}: seed {settings.seed} not in {spec.seeds}")
    return spec


def _selection_rows_name(selection_json: Path) -> str:
    sel: dict[str, Any] = json.loads(Path(selection_json).read_text("utf-8"))
    return str(sel["selected"]["rows_file"])


def _committed_selection_rows(selection_json: Path) -> Path | None:
    """The selected rows committed beside ``selection.json`` (plain, else ``.gz``), or ``None``."""
    base = Path(selection_json).parent / "select" / _selection_rows_name(selection_json)
    for p in (base, base.with_name(base.name + ".gz")):
        if p.is_file():
            return p
    return None


def _read_bytes(path: Path) -> bytes:
    """A rows file's bytes (decompressed when it is a ``.gz``)."""
    data = Path(path).read_bytes()
    return gzip.decompress(data) if Path(path).suffix == ".gz" else data


def pinned_selection(
    settings: SensitivitySettings, spec: ev_arms.ArmSpec
) -> tuple[ev_arms.Checkpoint, str, dict[str, Any]]:
    """Resolve the checkpoint from the committed records only (no checkpoint bytes).

    :func:`ila.eval.arms.resolve_checkpoint` (arm, seed, purpose, list, manifest, teacher,
    tag, contract variant, recorded inference), the pin compared with the ``selection.json``
    SHA-256, and selection's inference: the record's ``inference``, else the manifest config's
    (:func:`ila.train.select.inference_settings`; core ``dp`` predates the field).

    Returns:
        ``(checkpoint, pin, inference)``.

    Raises:
        SensitivityError: If the pin is missing or differs, or selection flew another inference.
    """
    from ila.train import manifest as mf
    from ila.train import select as train_select

    ck = ev_arms.resolve_checkpoint(spec, settings.seed)
    pins: Mapping[str, str] = json.loads(settings.selection_pins.read_text("utf-8"))["pins"]
    key = f"{spec.arm}/s{settings.seed}"
    pin = pins.get(key)
    if pin is None or pin != ck.selection_json_sha256:
        raise SensitivityError(f"{key}: selection.json SHA-256 {ck.selection_json_sha256} != {pin}")
    sel_json = p8_adapter.REPO_ROOT / ck.selection_json
    rec: dict[str, Any] = json.loads(sel_json.read_text("utf-8"))
    inf = rec.get("inference") or train_select.inference_settings(
        mf.read_json(sel_json.parent / "manifest.json")
    )
    if inf != settings.inference:
        raise SensitivityError(f"{key}: selection inference {inf} != {settings.inference}")
    return ck, pin, dict(inf)


def resolve_arm(
    settings: SensitivitySettings, arm: str, raw: Mapping[str, Any] | None = None
) -> ArmInfo:
    """Resolve, pin-check and byte-verify one arm; confirm its sampler settings.

    :func:`arm_spec`, :func:`pinned_selection`, then :func:`ila.deploy.export.load_arm` (pin
    again, checkpoint bytes, arm, contract variant, network family). The checkpoint's own
    config must give ``Ta`` and ``K`` (the student flies them) and selection's rows must exist.

    Raises:
        SensitivityError: On any disagreement, or a missing selection rows file.
    """
    from ila.policies.dp.policy import DPConfig

    spec = arm_spec(settings, arm, raw)
    ck, pin, inf = pinned_selection(settings, spec)
    loaded = ex.load_arm(arm, settings.seed, settings.selection_pins, tag=spec.tag)
    if loaded.ckpt.sha256 != ck.sha256:
        raise SensitivityError(f"{arm}: load_arm resolved {loaded.ckpt.sha256} != {ck.sha256}")
    cfg = loaded.net.cfg
    if not isinstance(cfg, DPConfig) or (cfg.n_action_steps, cfg.num_inference_steps) != (
        settings.ta,
        settings.k,
    ):
        raise SensitivityError(f"{arm}: checkpoint config is not Ta {settings.ta}, K {settings.k}")
    sel_json = p8_adapter.REPO_ROOT / ck.selection_json
    sel_rows = cl._selection_rows_path(sel_json)
    if not sel_rows.is_file():
        raise SensitivityError(f"{arm}: selection rows {sel_rows} missing")
    return ArmInfo(
        arm=arm,
        seed=settings.seed,
        tag=spec.tag,
        checkpoint=ck,
        selection_pin=pin,
        inference=inf,
        selection_rows=sel_rows,
        committed_selection_rows=_committed_selection_rows(sel_json),
    )


# --------------------------------------------------------------------------- flights


@dataclass(frozen=True)
class Flight:
    """One flight request.

    Attributes:
        arm: Arm.
        seed: Training seed.
        tag: Matrix tag.
        provider: ``torch_cpu_fp32`` (the reference) or ``torch_cuda_fp32``.
        workers: Processes.
        episodes: The listed ``select`` episodes, in order.
    """

    arm: str
    seed: int
    tag: str
    provider: str
    workers: int
    episodes: Sequence[p8_adapter.ListedEpisode]


type Flyer = Callable[[Flight], list[dict[str, Any]]]
type Resolver = Callable[[SensitivitySettings, str], ArmInfo]


def _fly_default(bench: BenchSettings, req: Flight) -> list[dict[str, Any]]:
    """:func:`ila.deploy.closed_loop.fly_provider`, Phase 9's own flight."""
    return cl.fly_provider(
        req.arm,
        req.provider,
        bench,
        req.episodes,
        graph_dir=bench.out_dir / "graphs",
        workers=req.workers,
        seed=req.seed,
        tag=req.tag,
    )


# --------------------------------------------------------------------------- summaries


def outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Episodes per P8 outcome class (every class, zeros included, in P8's order).

    Raises:
        ValueError: On an outcome that is not one of P8's classes.
    """
    c = Counter(str(r["outcome"]) for r in rows)
    unknown = set(c) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown outcome classes {sorted(unknown)}")
    return {o: int(c.get(o, 0)) for o in OUTCOMES}


def summarise(
    reference: Sequence[Mapping[str, Any]], perturbed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Flips, successes lost and gained, outcome counts and transitions (module docstring).

    Raises:
        ValueError: If the two flights are not the same episodes in the same order.
    """
    fl = cl.count_flips(perturbed, reference)
    n = len(reference)
    lost = sum(
        str(r["outcome"]) == "success" and str(p["outcome"]) != "success"
        for r, p in zip(reference, perturbed, strict=True)
    )
    gained = sum(
        str(r["outcome"]) != "success" and str(p["outcome"]) == "success"
        for r, p in zip(reference, perturbed, strict=True)
    )
    ref_c, pert_c = outcome_counts(reference), outcome_counts(perturbed)
    trans = Counter(f"{f['reference']}->{f['provider']}" for f in fl["flipped"])
    return {
        "n": n,
        "flips": int(fl["flips"]),
        "flip_rate": (int(fl["flips"]) / n) if n else 0.0,
        "flipped": fl["flipped"],
        "rows_text_differ": int(fl["rows_text_differ"]),
        "lost": int(lost),
        "gained": int(gained),
        "success_ref": ref_c["success"],
        "success_perturbed": pert_c["success"],
        "outcomes_ref": ref_c,
        "outcomes_perturbed": pert_c,
        "transitions": dict(sorted(trans.items())),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _flips_key(flipped: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    return sorted(
        (str(f["ss"]), str(f["index"]), str(f["reference"]), str(f["provider"])) for f in flipped
    )


def control_check(
    settings: SensitivitySettings,
    summary: Mapping[str, Any],
    checkpoint_sha256: str,
    ref_path: Path,
    pert_path: Path,
) -> dict[str, Any]:
    """Core ``dp`` against Phase 9's ``torch_cuda_fp32`` flight (from files; no flight).

    ``reproduced`` is the flip count's equality (the number L4 cites); the flipped episodes
    and both rows files' bytes are compared beside it.
    """
    p9: dict[str, Any] = json.loads(settings.control_record.read_text("utf-8"))

    def same(a: Path, b: Path) -> bool | None:
        return cl.rows_files_identical(a, b) if a.is_file() and b.is_file() else None

    return {
        "phase9_record": _rel(settings.control_record),
        "phase9_record_sha256": ex.sha256_file(settings.control_record),
        "phase9_provider": p9.get("provider"),
        "phase9_flips": int(p9["flips"]),
        "flips": int(summary["flips"]),
        "reproduced": int(p9["flips"]) == int(summary["flips"]),
        "same_flipped_episodes": _flips_key(p9.get("flipped", []))
        == _flips_key(summary["flipped"]),
        "checkpoint_sha256_equal": p9.get("checkpoint_sha256") == checkpoint_sha256,
        "perturbed_rows_equal_phase9": same(pert_path, settings.control_rows),
        "reference_rows_equal_phase9": same(ref_path, settings.control_reference_rows),
    }


def _selection_compare(
    ref_path: Path, selection_rows: Path, committed: Path | None
) -> dict[str, Any]:
    sel_rows = _read_rows(selection_rows)
    try:
        sel_flips: int | None = int(cl.count_flips(_read_rows(ref_path), sel_rows)["flips"])
    except ValueError:
        sel_flips = None
    return {
        "reference_equals_selection_rows": cl.rows_files_identical(ref_path, selection_rows),
        "selection_rows_file": _rel(selection_rows),
        "selection_rows_sha256": ex.sha256_file(selection_rows),
        "selection_rows_flips": sel_flips,
        "committed_selection_rows_file": None if committed is None else _rel(committed),
        "reference_equals_committed_selection_rows": None
        if committed is None
        else ref_path.read_bytes() == _read_bytes(committed),
    }


# --------------------------------------------------------------------------- CSV


def _fmt_bool(v: Any) -> str:
    return "" if v is None else ("true" if bool(v) else "false")


def sensitivity_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """``sensitivity.csv`` text (LF, fixed formatting: ``flip_rate`` from the integers, %.4f)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SENSITIVITY_COLUMNS)
    for r in records:
        n, flips = int(r["n"]), int(r["flips"])
        w.writerow(
            [
                r["arm"],
                int(r["seed"]),
                r["checkpoint_sha256"],
                n,
                _fmt_bool(r.get("reference_equals_selection_rows")),
                flips,
                f"{(flips / n) if n else 0.0:.4f}",
                int(r["lost"]),
                int(r["gained"]),
                int(r["success_ref"]),
                int(r["success_perturbed"]),
                *(int(r["outcomes_ref"][o]) for o in OUTCOMES),
                *(int(r["outcomes_perturbed"][o]) for o in OUTCOMES),
            ]
        )
    return buf.getvalue()


def _records(settings: SensitivitySettings) -> list[dict[str, Any]]:
    out = []
    for arm in settings.arms:
        p = record_path(settings, arm)
        if p.is_file():
            out.append(json.loads(p.read_text("utf-8")))
    return out


def render_csv(settings: SensitivitySettings) -> str:
    """``sensitivity.csv`` from the per-arm records on disk, in ``settings.arms`` order."""
    return sensitivity_csv(_records(settings))


def check(settings: SensitivitySettings) -> list[str]:
    """Re-decide every record from its files and re-render ``sensitivity.csv`` (no flight).

    For each arm with a record: both rows files exist and hash to the recorded SHA-256s; the
    summary (flips, lost, gained, outcome counts, transitions, ``n``) recomputed from them
    equals the record's; the selection comparison is re-decided from bytes where the selection
    rows are on this machine (the ablations' plain files live under ``artifacts/``; their
    committed ``.csv.gz`` is checked too); the control is re-decided from Phase 9's committed
    files. Then ``sensitivity.csv`` must re-render byte for byte.

    Returns:
        The problems found (empty: everything agrees).
    """
    bad: list[str] = []
    recs = _records(settings)
    for r in recs:
        arm = str(r.get("arm"))
        if r.get("status") != "flown":
            bad.append(f"{arm}: status {r.get('status')!r}")
            continue
        paths = {
            side: settings.results_dir / str(r[side]["rows_file"])
            for side in ("reference", "perturbed")
        }
        missing = [s for s, p in paths.items() if not p.is_file()]
        if missing:
            bad.append(f"{arm}: rows files missing: {missing}")
            continue
        for side, p in paths.items():
            if ex.sha256_file(p) != r[side]["rows_sha256"]:
                bad.append(f"{arm}: {p.name} differs from the recorded SHA-256")
        summ = summarise(_read_rows(paths["reference"]), _read_rows(paths["perturbed"]))
        for k in _SUMMARY_KEYS:
            if json.loads(json.dumps(summ[k])) != r.get(k):
                bad.append(f"{arm}: {k} disagrees with the rows files")
        sel = _rel_to_repo(str(r["selection_rows_file"]))
        if sel.is_file():
            if ex.sha256_file(sel) != r["selection_rows_sha256"]:
                bad.append(f"{arm}: selection rows differ from the recorded SHA-256")
            elif cl.rows_files_identical(paths["reference"], sel) != bool(
                r["reference_equals_selection_rows"]
            ):
                bad.append(f"{arm}: reference_equals_selection_rows disagrees with the bytes")
        com = r.get("committed_selection_rows_file")
        if com is not None and _rel_to_repo(str(com)).is_file():
            same = paths["reference"].read_bytes() == _read_bytes(_rel_to_repo(str(com)))
            if same != r.get("reference_equals_committed_selection_rows"):
                bad.append(f"{arm}: reference_equals_committed_selection_rows disagrees")
        if arm == settings.control_arm:
            ctl = control_check(
                settings, summ, str(r["checkpoint_sha256"]), paths["reference"], paths["perturbed"]
            )
            if ctl != r.get("control"):
                bad.append(f"{arm}: control disagrees with Phase 9's files")
    csv_path = settings.results_dir / CSV_FILE
    if recs or csv_path.is_file():
        want = sensitivity_csv(recs).encode("utf-8")
        if not csv_path.is_file() or csv_path.read_bytes() != want:
            bad.append(f"{CSV_FILE} does not re-render byte-identically")
    return bad


# --------------------------------------------------------------------------- stage


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, rec: Any) -> None:
    """Write JSON via a temporary file and a rename (a record exists only when complete)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")
    tmp.replace(path)


def git_checked(
    allow_dirty: bool, *, git: Mapping[str, Any] | None = None, allowed: str = DIRTY_ALLOWED
) -> dict[str, Any]:
    """The checkout's git state; refuse uncommitted changes outside ``allowed``.

    Args:
        allow_dirty: Record instead of refusing (smoke runs only).
        git: :func:`ila.deploy.export.git_state` output (default: read it now).
        allowed: Repository-relative prefix whose changes are tolerated (``results/phase10/``).

    Raises:
        DirtyTreeError: On uncommitted changes outside ``allowed`` without ``allow_dirty``.
    """
    g = dict(ex.git_state() if git is None else git)
    dirty = [ln for ln in g["dirty_paths"] if not ln[3:].strip().startswith(allowed)]
    if dirty and not allow_dirty:
        raise DirtyTreeError(f"uncommitted changes outside {allowed}: {dirty[:10]}")
    return {**g, "dirty_outside_results": bool(dirty)}


def host_env() -> dict[str, Any]:
    """P8's environment stamp (CPU, GPU, driver, torch, CUDA, ...) plus CUDA availability."""
    import torch

    stamp: dict[str, Any] = dict(p8_adapter.latency_harness().environment_stamp())
    stamp["torch_cuda_available"] = bool(torch.cuda.is_available())
    return stamp


def run_arm(
    settings: SensitivitySettings,
    info: ArmInfo,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    fly: Flyer,
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly one arm's reference and perturbation, write its rows, then its record (last)."""
    arm = info.arm
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sides: dict[str, dict[str, Any]] = {}
    for side, provider, workers in (
        ("reference", settings.reference, settings.workers_cpu),
        ("perturbed", settings.perturbed, settings.workers_gpu),
    ):
        t1 = time.perf_counter()
        rows = fly(Flight(arm, info.seed, info.tag, provider, workers, episodes))
        path = rows_path(settings, arm, provider)
        path.write_text(cl.rows_text(rows), "utf-8")  # CRLF rows, as selection writes them
        sides[side] = {
            "provider": provider,
            "rows_file": path.name,
            "rows_sha256": ex.sha256_file(path),
            "workers": workers,
            "wall_s": time.perf_counter() - t1,
        }
        log(f"{arm} {provider}: {len(rows)} rows, {sides[side]['wall_s']:.0f} s")
    ref_path = settings.results_dir / sides["reference"]["rows_file"]
    pert_path = settings.results_dir / sides["perturbed"]["rows_file"]
    summary = summarise(_read_rows(ref_path), _read_rows(pert_path))
    if summary["n"] != settings.n_episodes:
        raise SensitivityError(f"{arm}: {summary['n']} rows, not {settings.n_episodes}")
    sel = _selection_compare(ref_path, info.selection_rows, info.committed_selection_rows)
    for side, key in (("reference", "outcomes_ref"), ("perturbed", "outcomes_perturbed")):
        sides[side]["outcomes"] = summary[key]
        sides[side]["success"] = summary[key]["success"]
    control = (
        control_check(settings, summary, info.checkpoint.sha256, ref_path, pert_path)
        if arm == settings.control_arm
        else None
    )
    rec = {
        "status": "flown",
        "what": DEFINITION,
        "arm": arm,
        "seed": info.seed,
        "tag": info.tag,
        "teacher": info.checkpoint.teacher,
        "obs_variant": info.checkpoint.obs_variant,
        "checkpoint_sha256": info.checkpoint.sha256,
        "checkpoint": info.checkpoint.record(),
        "selection_pin": info.selection_pin,
        "selection_pins_file": _rel(settings.selection_pins),
        "inference": dict(info.inference),
        "list": settings.list_name,
        "reference": sides["reference"],
        "perturbed": sides["perturbed"],
        **sel,
        **summary,
        "control": control,
        "git": dict(git),
        "env": dict(env),
        "time_utc": _now(),
        "wall_s": time.perf_counter() - t0,
    }
    _write_json(record_path(settings, arm), rec)
    log(
        f"{arm}: {summary['flips']}/{summary['n']} flips (lost {summary['lost']}, gained "
        f"{summary['gained']}); success {summary['success_ref']} -> "
        f"{summary['success_perturbed']}; reference equals selection rows: "
        f"{sel['reference_equals_selection_rows']}"
    )
    if not sel["reference_equals_selection_rows"]:
        log(f"WARNING {arm}: the reference rows are NOT byte-identical to the selection rows")
    if control is not None:
        word = "reproduced" if control["reproduced"] else "NOT REPRODUCED"
        log(
            f"control {arm}: {control['flips']} flips vs Phase 9's {control['phase9_flips']} "
            f"({word}; same episodes {control['same_flipped_episodes']}, perturbed rows equal "
            f"Phase 9's {control['perturbed_rows_equal_phase9']})"
        )
    return rec


def run_sensitivity(
    settings: SensitivitySettings,
    *,
    arms: Sequence[str] | None = None,
    force: bool = False,
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    fly: Flyer,
    resolve: Resolver,
    episodes: Sequence[p8_adapter.ListedEpisode],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fly every requested arm without a record (all of them with ``force``); write the table.

    Returns:
        ``{"flown": [...], "kept": [...], "records": {arm: record}}``.

    Raises:
        SensitivityError: On an arm outside the settings, or a list that is not 150 episodes.
    """
    wanted = list(settings.arms if arms is None else arms)
    unknown = [a for a in wanted if a not in settings.arms]
    if unknown:
        raise SensitivityError(f"arms not in sensitivity.yaml: {unknown}")
    if len(episodes) != settings.n_episodes:
        raise SensitivityError(f"{len(episodes)} {settings.list_name} episodes, not 150")
    flown, kept = [], []
    records: dict[str, Any] = {}
    for arm in [a for a in settings.arms if a in wanted]:
        path = record_path(settings, arm)
        if path.is_file() and not force:
            old = json.loads(path.read_text("utf-8"))
            if old.get("status") == "flown":
                records[arm] = old
                kept.append(arm)
                log(f"{arm}: record exists, kept (--force re-flies)")
                continue
        info = resolve(settings, arm)
        records[arm] = run_arm(settings, info, episodes, fly=fly, git=git, env=env, log=log)
        flown.append(arm)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / CSV_FILE).write_text(render_csv(settings), "utf-8")
    return {"flown": flown, "kept": kept, "records": records}


def _run_info(settings: SensitivitySettings, invocation: Mapping[str, Any]) -> None:
    path = settings.results_dir / RUN_INFO_FILE
    info: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    info["sensitivity_yaml"] = _rel(settings.source)
    info["sensitivity_yaml_sha256"] = ex.sha256_file(settings.source)
    info["bench_yaml_sha256"] = ex.sha256_file(settings.bench_yaml)
    info["p8_commit"] = p8_adapter.p8_commit()
    info.setdefault("invocations", []).append(dict(invocation))
    info["arms"] = {
        r["arm"]: {
            "record_sha256": ex.sha256_file(record_path(settings, r["arm"])),
            "ila_commit": r["git"].get("ila_commit"),
            "dirty_outside_results": r["git"].get("dirty_outside_results"),
            "time_utc": r["time_utc"],
            "wall_s": r["wall_s"],
        }
        for r in _records(settings)
    }
    _write_json(path, info)


def stage_sensitivity(
    settings: SensitivitySettings,
    *,
    arms: Sequence[str] | None = None,
    force: bool = False,
    allow_dirty: bool = False,
    log: Callable[[str], None] = print,
    fly: Flyer | None = None,
    resolve: Resolver | None = None,
    episodes: Sequence[p8_adapter.ListedEpisode] | None = None,
    git: Mapping[str, Any] | None = None,
    env: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``sensitivity`` stage of ``scripts/deploy/bench.py`` (module docstring).

    Refuses a dirty tree outside ``results/phase10/`` unless ``allow_dirty``; flies with
    Phase 9's flight (:func:`ila.deploy.closed_loop.fly_provider`, TF32 off) and the real
    resolver unless tests inject ``fly``/``resolve``/``episodes``/``env``; records the
    invocation (commit, host, wall clock) in ``run_info.json``.

    Raises:
        DirtyTreeError: On a dirty tree without ``allow_dirty``.
        SensitivityError: If CUDA is unavailable for the real flight, or on any resolution
            problem.
    """
    g = git_checked(allow_dirty, git=git)
    if fly is None:
        import torch

        if not torch.cuda.is_available():
            raise SensitivityError("torch_cuda_fp32 needs CUDA; none is available")
        p8_adapter.latency_harness().disable_tf32()
        fly = functools.partial(_fly_default, load_settings(settings.bench_yaml))
    if resolve is None:
        resolve = resolve_arm
    if episodes is None:
        from ila.data import episodes as data_episodes

        episodes = data_episodes.read_split(settings.list_name)
    e = host_env() if env is None else dict(env)
    t0 = time.perf_counter()
    res = run_sensitivity(
        settings,
        arms=arms,
        force=force,
        git=g,
        env=e,
        fly=fly,
        resolve=resolve,
        episodes=episodes,
        log=log,
    )
    _run_info(
        settings,
        {
            "time_utc": _now(),
            "git": g,
            "env": e,
            "arms_requested": list(settings.arms if arms is None else arms),
            "arms_flown": res["flown"],
            "arms_kept": res["kept"],
            "force": bool(force),
            "allow_dirty": bool(allow_dirty),
            "wall_s": time.perf_counter() - t0,
        },
    )
    return res
