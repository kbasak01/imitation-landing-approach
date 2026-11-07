"""Inference latency per provider with P8's harness method; H5(a) and the delay d (Phase 9).

Method (P8 ``rld.deploy.latency``, which mirrors dmf's ``benchmark_onnxruntime`` /
``benchmark_torch``; the pieces come through :func:`ila.p8_adapter.latency_harness`):

* one configuration per fresh child process (``python -m ila.deploy.latency --child``) under
  dmf's ``tf32_environment()`` (``NVIDIA_TF32_OVERRIDE = 0`` before any CUDA context), run one
  at a time;
* in the child: dmf's ``disable_tf32()``; ``>= 200`` untimed warm-up calls, then a
  ``torch.cuda.synchronize()`` on GPU providers; ``>= 2 000`` timed calls, each bracketed by
  ``time.perf_counter()`` and, on GPU providers, ``torch.cuda.synchronize()`` before the
  bracket closes; host to host (numpy in, numpy out); the inputs cycle through fixed distinct
  batches (dmf's ``_batches`` rule) of the 1 000 ``select`` windows and the reference path's
  conditioning, noise and ``x_k``;
* reduced by dmf's ``_stats`` to p50 / p90 / p99 / mean / std, plus the max (P9-D1), in ms
  **model-scale wall clock of this machine** (the sim's 30 Hz control step is 33.333 ms model
  scale); stamped by dmf's ``environment_stamp``;
* a configuration whose arm failed open-loop parity on that provider is **refused** and never
  launched; a child that fails is written ``failed: <reason>``. Neither carries a number.

Rows (``row`` column), each at batch 1 and the evaluation batch (24):

* ``bc_e2e``: ``bc_dagger`` image stack + proprio -> action;
* ``dp_encoder``; ``dp_denoise_step`` (one U-Net call at ``k = 99``);
* ``dp_loop_K{1,5,10,25}``: ``ila``'s DDIM loop around the denoise step (global_cond and noise
  in, chunk out; ORT providers sample on the host, torch on the device);
* ``dp_unrolled_K10``: ``dp_K10.onnx`` (torch: :class:`ila.deploy.export.UnrolledDDIM`);
* ``dp_e2e``: image stack + proprio + noise -> K = 10 chunk, one full replanning call
  (ORT: ``dp_encoder.onnx`` then ``dp_K10.onnx``; torch: ``DiffusionPolicyNet.sample``).

Providers: the reference ``torch_cpu_fp32`` (batch 1 only: the P3-D1 §6 fallback), the four
scored ones and the unscored variants.

Scoring (:func:`score`):

* **H5(a)** (P3-D1 §6): ``dp_e2e`` at batch 1, ``p99 < 33.3 ms`` on at least one scored
  provider that passes parity, where parity = open-loop **and** closed-loop (``<= 3`` flips on
  the 150 ``select`` episodes; binding). Supported iff one does; "not supported" if none passes
  parity or none meets the bound. A scored provider with parity passed but no timed row, or
  without its closed-loop record, leaves H5(a) "not scored - ..." (never dropped silently).
* **d** (P3-D1 §7): per arm (``bc_dagger``: ``bc_e2e``; ``dp``: ``dp_e2e``), the scored
  parity-passing provider with the lowest batch-1 p99; ``d = ceil(p99 / 33.333 ms)``. With no
  such provider, the ``torch_cpu_fp32`` batch-1 p99 (the fallback). ``delay.json`` is the
  interface the latency arm reads.

No claim transfers to an embedded target: this is one workstation (stamped), simulation only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.deploy.settings import GPU_PROVIDERS, BenchSettings, load_settings, precision_of

__all__ = [
    "CHILD_TIMEOUT_S",
    "H5A_COLUMNS",
    "LATENCY_COLUMNS",
    "ROWS",
    "LatencyJob",
    "delay_steps",
    "h5a_csv",
    "latency_csv",
    "latency_jobs",
    "make_batches",
    "row_arm",
    "run_child",
    "run_latency",
    "score",
    "time_calls",
]

#: Child timeout, s. P8's is 3 600 s; ``ort_cpu_fp32`` ``dp_loop_K25`` at batch 24 runs longer
#: (P9-D1), so the limit is 8 h.
CHILD_TIMEOUT_S: Final[float] = 8 * 3600.0


def ROWS(k_loop: Sequence[int]) -> tuple[str, ...]:  # noqa: N802 (a table, named as one)
    """Latency rows in report order."""
    return (
        "bc_e2e",
        "dp_encoder",
        "dp_denoise_step",
        *(f"dp_loop_K{k}" for k in k_loop),
        "dp_unrolled_K10",
        "dp_e2e",
    )


def row_arm(row: str) -> str:
    """The arm a row times (``bc_e2e`` -> ``bc_dagger``, else ``dp``)."""
    return "bc_dagger" if row == "bc_e2e" else "dp"


@dataclass(frozen=True)
class LatencyJob:
    """One configuration.

    Attributes:
        row: The latency row (module docstring).
        provider: The ``ila`` provider.
        batch: Batch size.
        warmup: Untimed calls.
        timed: Timed calls.
        threads: Intra-op CPU threads.
    """

    row: str
    provider: str
    batch: int
    warmup: int
    timed: int
    threads: int

    @property
    def label(self) -> str:
        """``<row>-<provider>-b<batch>-t<threads>``."""
        return f"{self.row}-{self.provider}-b{self.batch}-t{self.threads}"

    @property
    def arm(self) -> str:
        """The arm the row times."""
        return row_arm(self.row)


def latency_jobs(settings: BenchSettings) -> list[LatencyJob]:
    """Every configuration: batch 1 first (all H5(a) and d need), then the evaluation batch."""
    jobs = []
    for b in sorted(settings.batch_sizes):
        for p in settings.providers:
            if p == settings.reference and b not in settings.reference_batch_sizes:
                continue
            for row in ROWS(settings.k_loop):
                jobs.append(
                    LatencyJob(
                        row, p, b, settings.warmup_iters, settings.timed_iters, settings.threads
                    )
                )
    return jobs


def make_batches(n: int, batch: int) -> list[NDArray[np.int64]]:
    """Row indices of fixed distinct batches, cycling (dmf ``_batches``: ``max(1, n // B)``)."""
    if batch < 1 or n < 1:
        raise ValueError(f"need n >= 1 and batch >= 1, got {n}, {batch}")
    nb = max(1, n // batch)
    return [np.array([(i * batch + j) % n for j in range(batch)], np.int64) for i in range(nb)]


def time_calls(
    fn: Callable[[int], Any], n_batches: int, warmup: int, timed: int, sync: Callable[[], None]
) -> list[float]:
    """P8's loop: ``warmup`` untimed calls, ``sync``, then ``timed`` bracketed calls (seconds).

    Args:
        fn: Runs batch ``i`` host to host.
        n_batches: Distinct batches cycled.
        warmup: Untimed calls.
        timed: Timed calls.
        sync: ``torch.cuda.synchronize`` on a GPU provider, else a no-op.

    Returns:
        Per-call wall time, seconds.
    """
    for i in range(warmup):
        fn(i % n_batches)
    sync()
    samples: list[float] = []
    for i in range(timed):
        start = time.perf_counter()
        fn(i % n_batches)
        sync()
        samples.append(time.perf_counter() - start)
    return samples


# --------------------------------------------------------------------------- child


def _job_fn(
    job: LatencyJob,
    backend: Any,
    data: Mapping[str, NDArray[Any]],
    batches: list[NDArray[np.int64]],
) -> Callable[[int], Any]:
    img, prop = data["images"], data["proprio"]
    cond, noise, x0 = data["cond"], data["noise"], data["x0"]
    b = [
        {
            "images": np.ascontiguousarray(img[ix]),
            "proprio": np.ascontiguousarray(prop[ix]),
            "cond": np.ascontiguousarray(cond[ix]),
            "noise": np.ascontiguousarray(noise[ix]),
            "x0": np.ascontiguousarray(x0[ix]),
            "t99": np.full((len(ix),), 99, np.int64),
        }
        for ix in batches
    ]
    row = job.row
    if row == "bc_e2e":
        return lambda i: backend.bc(b[i]["images"], b[i]["proprio"])
    if row == "dp_encoder":
        return lambda i: backend.encode(b[i]["images"], b[i]["proprio"])
    if row == "dp_denoise_step":
        return lambda i: backend.denoise(b[i]["x0"], b[i]["t99"], b[i]["cond"])
    if row.startswith("dp_loop_K"):
        k = int(row.removeprefix("dp_loop_K"))
        return lambda i: backend.loop(b[i]["cond"], b[i]["noise"], k)
    if row == "dp_unrolled_K10":
        return lambda i: backend.unrolled(b[i]["cond"], b[i]["noise"])
    if row == "dp_e2e":
        return lambda i: backend.dp_e2e(b[i]["images"], b[i]["proprio"], b[i]["noise"])
    raise ValueError(f"unknown latency row {row!r}")


def run_child(job: LatencyJob, settings: BenchSettings) -> dict[str, Any]:
    """Measure one configuration in this process (the child's body)."""
    import torch

    from ila.deploy import backends as bk
    from ila.deploy.export import dp_config_from_meta, load_arm, verify_graphs
    from ila.deploy.inputs import load_inputs

    h = p8_adapter.latency_harness()
    h.disable_tf32()
    torch.set_num_threads(job.threads)
    meta = verify_graphs(settings.out_dir / "graphs")
    inputs = load_inputs(settings.out_dir / "inputs" / "select_inputs.npz")
    with np.load(settings.out_dir / "parity" / "reference.npz", allow_pickle=False) as z:
        data: dict[str, NDArray[Any]] = {
            "images": inputs.images,
            "proprio": inputs.proprio,
            "cond": np.asarray(z["cond"], np.float32),
            "noise": np.asarray(z["noise"], np.float32),
            "x0": np.asarray(z["x"][0], np.float32),
        }
    gpu = job.provider in GPU_PROVIDERS
    on_ort_gpu = gpu and not job.provider.startswith("torch_")
    free_before = h.cuda_free_bytes() if on_ort_gpu else 0

    def arms() -> dict[str, Any]:
        return {job.arm: load_arm(job.arm, settings.arm_seeds[job.arm], settings.selection_pins)}

    backend = bk.make_backend(
        job.provider,
        graph_dir=settings.out_dir / "graphs",
        dp_cfg=dp_config_from_meta(meta),
        arms_loader=arms,
        batch=job.batch,
        threads=job.threads,
        trt_cache_root=settings.out_dir / "trt_cache",
    )
    if gpu and job.provider.startswith("torch_"):
        torch.cuda.reset_peak_memory_stats()
    batches = make_batches(int(inputs.n), job.batch)
    fn = _job_fn(job, backend, data, batches)

    def sync() -> None:
        if gpu:
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    samples = time_calls(fn, len(batches), job.warmup, job.timed, sync)
    wall = time.perf_counter() - t0
    free_min = min(free_before, h.cuda_free_bytes()) if on_ort_gpu else 0
    p50, p90, p99, mean, std = h.stats(samples)
    if not gpu:
        peak_dev = 0.0
    elif job.provider.startswith("torch_"):
        peak_dev = float(torch.cuda.max_memory_allocated()) / h.BYTES_PER_MIB
    else:
        peak_dev = max(0.0, float(free_before - free_min) / h.BYTES_PER_MIB)
    return {
        "job": asdict(job),
        "label": job.label,
        "status": "timed",
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "max_ms": float(np.max(samples) * 1e3),
        "mean_ms": mean,
        "std_ms": std,
        "throughput_per_s": job.batch / (mean / 1e3),
        "peak_host_mem_mib": float(h.peak_host_mem_mb()),
        "peak_device_mem_mib": peak_dev,
        "providers_realized": backend.realized(),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "n_distinct_batches": len(batches),
        "wall_s": wall,
        "env": h.environment_stamp(),
    }


def _child_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="one latency configuration (internal)")
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", type=Path, default=None)
    a = ap.parse_args(argv)
    job = LatencyJob(**json.loads(a.child.read_text("utf-8")))
    settings = load_settings() if a.settings is None else load_settings(a.settings)
    rec = run_child(job, settings)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0


def run_job_subprocess(job: LatencyJob, timeout_s: float = CHILD_TIMEOUT_S) -> dict[str, Any]:
    """Run one configuration in a fresh child under dmf's ``tf32_environment()``.

    Raises:
        RuntimeError: If the child exits non-zero or writes nothing (stderr tail included).
    """
    env = p8_adapter.latency_harness().tf32_environment()
    with tempfile.TemporaryDirectory(prefix="ila-bench-") as tmp:
        jp, op = Path(tmp) / "job.json", Path(tmp) / "out.json"
        jp.write_text(json.dumps(asdict(job)), "utf-8")
        done = subprocess.run(
            [sys.executable, "-m", "ila.deploy.latency", "--child", str(jp), "--out", str(op)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
            cwd=p8_adapter.REPO_ROOT,
        )
        if done.returncode != 0 or not op.is_file():
            raise RuntimeError(
                f"latency child {job.label} exited {done.returncode}\n{done.stderr.strip()[-4000:]}"
            )
        rec: dict[str, Any] = json.loads(op.read_text("utf-8"))
        return rec


def _open_loop_pass(parity: Mapping[str, Any], provider: str, arm: str) -> bool:
    rec = parity.get("providers", {}).get(provider, {}).get(arm)
    return bool(rec is not None and rec.get("open_loop_pass"))


def run_latency(
    settings: BenchSettings,
    jobs: Sequence[LatencyJob],
    parity: Mapping[str, Any],
    *,
    runner: Callable[[LatencyJob], dict[str, Any]] = run_job_subprocess,
    force: bool = False,
    git: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Time every non-refused job, one at a time; one JSON per job under ``latency/jobs/``.

    A job whose record already exists is kept (resumable) unless ``force``.
    """
    out = settings.results_dir / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for job in jobs:
        path = out / f"{job.label}.json"
        if path.is_file() and not force:
            records.append(json.loads(path.read_text("utf-8")))
            continue
        base = {"job": asdict(job), "label": job.label, "git": dict(git or {})}
        if job.provider != settings.reference and not _open_loop_pass(
            parity, job.provider, job.arm
        ):
            rec = {**base, "status": "refused (open-loop parity not passed)"}
            log(f"refused: {job.label}")
        else:
            t0 = time.perf_counter()
            try:
                rec = {**runner(job), "git": dict(git or {})}
                log(
                    f"{job.label}: p50 {rec['p50_ms']:.3f} ms, p99 {rec['p99_ms']:.3f} ms "
                    f"({time.perf_counter() - t0:.0f} s)"
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                lines = str(exc).strip().splitlines()
                rec = {**base, "status": f"failed: {lines[-1] if lines else 'child failed'}"[:400]}
                log(f"FAILED: {job.label}: {rec['status']}")
        path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")
        records.append(rec)
    return records


# --------------------------------------------------------------------------- tables

#: Columns of ``latency.csv``.
LATENCY_COLUMNS: Final[tuple[str, ...]] = (
    "label",
    "row",
    "arm",
    "provider",
    "scored",
    "precision",
    "batch_size",
    "threads",
    "status",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "max_ms",
    "mean_ms",
    "std_ms",
    "throughput_per_s",
    "peak_host_mem_mib",
    "peak_device_mem_mib",
    "tf32",
    "providers_realized",
    "warmup_iters",
    "timed_iters",
    "ila_commit",
)


def _f(x: Any) -> str:
    return f"{float(x):.6g}"


def latency_csv(records: Sequence[Mapping[str, Any]], settings: BenchSettings) -> str:
    """Render ``latency.csv`` (a refused or failed row has empty numbers)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(LATENCY_COLUMNS)
    for r in records:
        j = r["job"]
        timed = r.get("status") == "timed"
        nums = (
            [
                _f(r[k])
                for k in (
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                    "max_ms",
                    "mean_ms",
                    "std_ms",
                    "throughput_per_s",
                    "peak_host_mem_mib",
                    "peak_device_mem_mib",
                )
            ]
            + [str(r["tf32"])]
            if timed
            else [""] * 10
        )
        realized = (
            ";".join(f"{g}={'|'.join(p)}" for g, p in sorted(r["providers_realized"].items()))
            if timed
            else ""
        )
        w.writerow(
            [
                r["label"],
                j["row"],
                row_arm(j["row"]),
                j["provider"],
                settings.is_scored(j["provider"]),
                precision_of(j["provider"]),
                j["batch"],
                j["threads"],
                r["status"],
                *nums,
                realized,
                j["warmup"],
                j["timed"],
                (r.get("git") or {}).get("ila_commit", ""),
            ]
        )
    return buf.getvalue()


def delay_steps(p99_ms: float, control_period_ms: float) -> int:
    """``d = ceil(p99 / 33.333 ms)`` control steps (P3-D1 §7)."""
    if not (math.isfinite(p99_ms) and p99_ms > 0):
        raise ValueError(f"p99 must be finite and positive, got {p99_ms}")
    return int(math.ceil(p99_ms / control_period_ms))


#: Columns of ``h5a.csv``.
H5A_COLUMNS: Final[tuple[str, ...]] = (
    "hypothesis",
    "row",
    "provider",
    "scored",
    "batch_size",
    "latency_status",
    "p50_ms",
    "p99_ms",
    "max_ms",
    "open_loop_pass",
    "closed_loop_pass",
    "parity_pass",
    "threshold_ms",
    "p99_below_threshold",
    "verdict",
)


def _find(
    records: Sequence[Mapping[str, Any]], row: str, provider: str, batch: int
) -> Mapping[str, Any] | None:
    for r in records:
        j = r["job"]
        if j["row"] == row and j["provider"] == provider and int(j["batch"]) == batch:
            return r
    return None


def _closed_pass(closed: Mapping[str, Mapping[str, Any]], arm: str, provider: str) -> bool | None:
    rec = closed.get(arm, {}).get(provider)
    if rec is None or "closed_loop_pass" not in rec:
        return None
    return bool(rec["closed_loop_pass"])


def score(
    settings: BenchSettings,
    records: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
    closed: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[list[str]], dict[str, Any]]:
    """H5(a) and the per-arm delay (module docstring).

    Args:
        settings: Bench settings.
        records: Latency job records.
        parity: ``parity.json``.
        closed: ``{arm: {provider: closed-loop record}}``.

    Returns:
        ``(h5a_verdict, h5a table rows, delay record)``.
    """
    table: list[list[str]] = []
    unscorable: list[str] = []
    meets: list[bool] = []
    passers: dict[str, list[tuple[float, str]]] = {"bc_dagger": [], "dp": []}
    for arm, row in (("dp", "dp_e2e"), ("bc_dagger", "bc_e2e")):
        for provider in settings.providers:
            if provider == settings.reference:
                continue
            rec = _find(records, row, provider, 1)
            ol = _open_loop_pass(parity, provider, arm)
            cl = _closed_pass(closed, arm, provider)
            par = bool(ol and cl)
            timed = rec is not None and rec.get("status") == "timed"
            p99 = float(rec["p99_ms"]) if timed and rec is not None else math.nan
            if settings.is_scored(provider) and par and timed:
                passers[arm].append((p99, provider))
            if arm != "dp":
                continue
            scored = settings.is_scored(provider)
            below = timed and p99 < settings.h5a_threshold_ms
            if scored:
                if ol and cl is None:
                    unscorable.append(f"{provider}: closed-loop parity not flown")
                elif par and not timed:
                    unscorable.append(f"{provider}: parity passed but not timed")
                elif par and timed:
                    meets.append(bool(below))
            table.append(
                [
                    "H5a",
                    row,
                    provider,
                    str(scored),
                    "1",
                    "missing" if rec is None else str(rec.get("status")),
                    "" if not timed or rec is None else _f(rec["p50_ms"]),
                    "" if not timed else _f(p99),
                    "" if not timed or rec is None else _f(rec["max_ms"]),
                    str(ol),
                    "" if cl is None else str(cl),
                    str(par),
                    _f(settings.h5a_threshold_ms),
                    "" if not timed else str(below),
                ]
            )
    if unscorable:
        verdict = "not scored - " + "; ".join(unscorable)
    elif any(meets):
        verdict = "supported"
    else:
        verdict = "not supported"
    for line in table:
        line.append(verdict if line[3] == "True" else "context (not scored)")
    delay: dict[str, Any] = {
        "rule": "d = ceil(p99_ms / 33.333), batch-1 full call, fastest (lowest p99) scored "
        "provider passing open- and closed-loop parity; else torch_cpu_fp32 (P3-D1 §6, §7)",
        "control_period_ms": settings.control_period_ms,
        "arms": {},
    }
    for arm, row in (("bc_dagger", "bc_e2e"), ("dp", "dp_e2e")):
        if passers[arm]:
            p99, chosen = min(passers[arm])
            fallback = False
        else:
            ref = _find(records, row, settings.reference, 1)
            if ref is None or ref.get("status") != "timed":
                delay["arms"][arm] = {"d": None, "status": "fallback row not timed"}
                continue
            p99, chosen, fallback = float(ref["p99_ms"]), str(settings.reference), True
        delay["arms"][arm] = {
            "d": delay_steps(p99, settings.control_period_ms),
            "provider": chosen,
            "row": row,
            "p99_ms": p99,
            "fallback": fallback,
            "candidates": {p: v for v, p in sorted(passers[arm])},
        }
    delay["h5a_verdict"] = verdict
    return verdict, table, delay


def h5a_csv(table: Sequence[Sequence[str]]) -> str:
    """Render ``h5a.csv``."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(H5A_COLUMNS)
    w.writerows(table)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1:]))
