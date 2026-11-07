"""Phase 9 latency: P8's harness by reference, the loop, refusals, H5(a) and d (synthetic rows).

CPU only; one child-body run on a tiny export with a handful of iterations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deploy_support import tiny_settings
from ila import p8_adapter
from ila.deploy import latency as lat
from ila.deploy import pipeline
from ila.deploy.settings import load_settings


def test_harness_is_p8s_by_reference() -> None:
    import dmf.deploy.bench as b
    import dmf.deploy.providers as pr
    import rld.deploy.latency as rl

    h = p8_adapter.latency_harness()
    assert h.BenchConfig is b.BenchConfig and h.BenchResult is b.BenchResult
    assert h.stats is b._stats and h.environment_stamp is b.environment_stamp
    assert h.GPU_PROVIDERS is b.GPU_PROVIDERS
    assert h.disable_tf32 is pr.disable_tf32 and h.tf32_environment is pr.tf32_environment
    assert h.session_providers is pr.session_providers
    assert h.CUDA_PROVIDER_OPTIONS is pr.CUDA_PROVIDER_OPTIONS
    assert h.CONTROL_PERIOD_MS == rl.CONTROL_PERIOD_MS
    assert pr.CUDA_PROVIDER_OPTIONS == {"use_tf32": "0"}
    assert h.tf32_environment()["NVIDIA_TF32_OVERRIDE"] == "0"


def test_make_batches_is_dmfs_rule() -> None:
    bs = lat.make_batches(1000, 24)
    assert len(bs) == 41 and all(len(b) == 24 for b in bs)
    assert bs[0].tolist() == list(range(24)) and bs[-1][-1] == (40 * 24 + 23) % 1000
    assert len(lat.make_batches(3, 24)) == 1 and lat.make_batches(3, 24)[0].tolist()[:4] == [
        0,
        1,
        2,
        0,
    ]


def test_time_calls_warms_up_then_brackets_every_call() -> None:
    calls: list[tuple[str, int]] = []
    out = lat.time_calls(lambda i: calls.append(("f", i)), 3, 5, 7, lambda: calls.append(("s", -1)))
    assert len(out) == 7 and all(x >= 0 for x in out)
    fs = [c for c in calls if c[0] == "f"]
    assert [i for _, i in fs] == [0, 1, 2, 0, 1] + [0, 1, 2, 0, 1, 2, 0]
    # One sync after the warm-up, then one inside every timed bracket.
    assert sum(1 for c in calls if c[0] == "s") == 1 + 7
    assert calls[5] == ("s", -1)


@pytest.mark.parametrize(
    ("p99", "d"), [(1.0, 1), (33.333, 1), (33.334, 2), (66.666, 2), (100.0, 4), (240.0, 8)]
)
def test_delay_is_ceil_of_p99_over_the_control_step(p99: float, d: int) -> None:
    assert lat.delay_steps(p99, 33.333) == d


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_delay_refuses_a_non_positive_p99(bad: float) -> None:
    with pytest.raises(ValueError):
        lat.delay_steps(bad, 33.333)


def test_latency_jobs_grid() -> None:
    s = load_settings()
    jobs = lat.latency_jobs(s)
    rows = lat.ROWS(s.k_loop)
    assert rows == (
        "bc_e2e", "dp_encoder", "dp_denoise_step", "dp_loop_K1", "dp_loop_K5", "dp_loop_K10",
        "dp_loop_K25", "dp_unrolled_K10", "dp_e2e",
    )  # fmt: skip
    assert all(j.warmup >= 200 and j.timed >= 2000 and j.threads == 1 for j in jobs)
    first24 = next(i for i, j in enumerate(jobs) if j.batch == 24)
    assert all(j.batch == 1 for j in jobs[:first24])
    assert not [j for j in jobs if j.provider == "torch_cpu_fp32" and j.batch != 1]
    n_prov = len(s.providers)
    assert len(jobs) == len(rows) * (n_prov + (n_prov - 1))
    assert len({j.label for j in jobs}) == len(jobs)


def _rec(
    row: str, provider: str, batch: int, p99: float | None, status: str = "timed"
) -> dict[str, Any]:
    job = {
        "row": row,
        "provider": provider,
        "batch": batch,
        "warmup": 200,
        "timed": 2000,
        "threads": 1,
    }
    label = lat.LatencyJob(**job).label
    if p99 is None or status != "timed":
        return {"job": job, "label": label, "status": status if p99 is None else status, "git": {}}
    return {
        "job": job, "label": label, "status": "timed", "p50_ms": p99 / 2, "p90_ms": p99 * 0.9,
        "p99_ms": p99, "max_ms": p99 * 1.5, "mean_ms": p99 / 2, "std_ms": 0.1,
        "throughput_per_s": 1.0, "peak_host_mem_mib": 1.0, "peak_device_mem_mib": 0.0,
        "providers_realized": {"g.onnx": ["X"]}, "tf32": False, "git": {"ila_commit": "abc"},
    }  # fmt: skip


def _parity(passing: dict[str, tuple[bool, bool]]) -> dict[str, Any]:
    return {
        "providers": {
            p: {"bc_dagger": {"open_loop_pass": bc}, "dp": {"open_loop_pass": dp}}
            for p, (bc, dp) in passing.items()
        }
    }


def _closed(passing: dict[str, tuple[bool, bool]]) -> dict[str, dict[str, Any]]:
    return {
        "bc_dagger": {p: {"closed_loop_pass": bc} for p, (bc, _) in passing.items()},
        "dp": {p: {"closed_loop_pass": dp} for p, (_, dp) in passing.items()},
    }


def _records(p99s: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    out = []
    for p, (bc, dp) in p99s.items():
        out.append(_rec("bc_e2e", p, 1, bc))
        out.append(_rec("dp_e2e", p, 1, dp))
    return out


PROVIDERS = (
    "ort_cpu_fp32",
    "ort_cuda_fp32",
    "torch_cuda_fp32",
    "ort_trt_fp16",
    "ort_trt_fp32",
    "ort_cuda_fp16",
)
ALL_PASS = dict.fromkeys(PROVIDERS, (True, True))


def test_score_supported_and_fastest_provider() -> None:
    s = load_settings()
    recs = _records(
        {
            "torch_cpu_fp32": (12.0, 300.0),
            "ort_cpu_fp32": (10.0, 200.0),
            "ort_cuda_fp32": (3.0, 30.0),
            "torch_cuda_fp32": (4.0, 60.0),
            "ort_trt_fp16": (2.0, 20.0),
            "ort_trt_fp32": (1.0, 5.0),
            "ort_cuda_fp16": (1.0, 5.0),
        }
    )
    verdict, table, delay = lat.score(s, recs, _parity(ALL_PASS), _closed(ALL_PASS))
    assert verdict == "supported"
    # The unscored variants are faster but never chosen.
    assert delay["arms"]["dp"]["provider"] == "ort_trt_fp16" and delay["arms"]["dp"]["d"] == 1
    assert delay["arms"]["bc_dagger"]["provider"] == "ort_trt_fp16"
    assert not delay["arms"]["dp"]["fallback"]
    assert [ln[-1] for ln in table if ln[2] == "ort_trt_fp32"] == ["context (not scored)"]
    assert lat.h5a_csv(table).splitlines()[0].startswith("hypothesis,row,provider")


def test_closed_loop_failure_removes_a_provider() -> None:
    s = load_settings()
    recs = _records({"torch_cpu_fp32": (12.0, 300.0), "ort_cpu_fp32": (10.0, 200.0),
                     "ort_cuda_fp32": (3.0, 30.0), "torch_cuda_fp32": (4.0, 60.0),
                     "ort_trt_fp16": (2.0, 20.0)})  # fmt: skip
    passing = {**ALL_PASS, "ort_trt_fp16": (True, False), "ort_cuda_fp32": (True, False)}
    verdict, _, delay = lat.score(s, recs, _parity(ALL_PASS), _closed(passing))
    assert verdict == "not supported"  # torch_cuda 60 ms and ort_cpu 200 ms remain
    assert delay["arms"]["dp"]["provider"] == "torch_cuda_fp32"
    assert delay["arms"]["dp"]["d"] == math.ceil(60.0 / 33.333) == 2
    assert delay["arms"]["bc_dagger"]["provider"] == "ort_trt_fp16"  # bc passed everywhere


def test_no_provider_passes_uses_torch_cpu_fallback() -> None:
    s = load_settings()
    recs = _records({"torch_cpu_fp32": (12.0, 300.0)})
    fail = dict.fromkeys(ALL_PASS, (False, False))
    verdict, _, delay = lat.score(s, recs, _parity(fail), _closed(fail))
    assert verdict == "not supported"
    assert delay["arms"]["dp"] == {
        **delay["arms"]["dp"],
        "provider": "torch_cpu_fp32",
        "fallback": True,
        "d": 10,
    }
    assert delay["arms"]["bc_dagger"]["d"] == 1


def test_missing_closed_loop_leaves_h5a_unscored() -> None:
    s = load_settings()
    recs = _records({"torch_cpu_fp32": (12.0, 300.0), "ort_trt_fp16": (2.0, 20.0)})
    closed = _closed({p: v for p, v in ALL_PASS.items() if p != "ort_trt_fp16"})
    verdict, _, _ = lat.score(s, recs, _parity(ALL_PASS), closed)
    assert verdict.startswith("not scored") and "ort_trt_fp16" in verdict


def _quiet(msg: str) -> None:
    del msg


def test_run_latency_refuses_and_records_failures(tmp_path: Path) -> None:
    s = tiny_settings(tmp_path)
    jobs = [
        lat.LatencyJob("dp_e2e", "ort_cuda_fp32", 1, 200, 2000, 1),
        lat.LatencyJob("dp_e2e", "ort_cpu_fp32", 1, 200, 2000, 1),
        lat.LatencyJob("bc_e2e", "torch_cpu_fp32", 1, 200, 2000, 1),
    ]
    parity = _parity({"ort_cuda_fp32": (True, False), "ort_cpu_fp32": (True, True)})
    launched: list[str] = []

    def runner(job: lat.LatencyJob) -> dict[str, Any]:
        launched.append(job.label)
        if job.provider == "ort_cpu_fp32":
            raise RuntimeError("child exited 1\nboom")
        return _rec(job.row, job.provider, job.batch, 5.0)

    recs = lat.run_latency(s, jobs, parity, runner=runner, git={"ila_commit": "c0"}, log=_quiet)
    assert launched == [jobs[1].label, jobs[2].label]  # the parity-failed job is never launched
    assert [r["status"] for r in recs] == [
        "refused (open-loop parity not passed)",
        "failed: boom",
        "timed",
    ]
    text = lat.latency_csv(recs, s)
    row = [ln for ln in text.splitlines() if ln.startswith("dp_e2e-ort_cuda_fp32")][0].split(",")
    assert row[8].startswith("refused") and row[9:19] == [""] * 10
    # Resumable: records on disk are kept, nothing is relaunched.
    launched.clear()
    lat.run_latency(s, jobs, parity, runner=runner, log=_quiet)
    assert launched == []


def test_check_rerenders_byte_identically(tmp_path: Path) -> None:
    s = tiny_settings(tmp_path)
    r = s.results_dir
    (r / "jobs").mkdir(parents=True)
    for rec in _records({"torch_cpu_fp32": (12.0, 300.0), "ort_trt_fp16": (2.0, 20.0)}):
        (r / "jobs" / f"{rec['label']}.json").write_text(json.dumps(rec))
    (r / "parity.json").write_text(json.dumps(_parity(ALL_PASS)))
    for arm in ("bc_dagger", "dp"):
        (r / "closed_loop" / arm).mkdir(parents=True)
        for p, v in ALL_PASS.items():
            rec = {
                "closed_loop_pass": v[0],
                "n": 150,
                "flips": 0,
                "status": "flown",
                "outcomes": {"success": 150},
            }
            (r / "closed_loop" / arm / f"{p}.json").write_text(json.dumps(rec))
    delay = pipeline.stage_score(s)
    from ila.deploy import parity as par

    prov = json.loads((r / "parity.json").read_text())["providers"]
    (r / "parity.csv").write_text(par.parity_csv(prov, s))
    (r / "parity_steps.csv").write_text(par.steps_csv(prov, s))
    assert pipeline.check(s) == []
    (r / "parity.csv").write_text("stale\n")
    (r / "h5a.csv").write_text("stale\n")
    assert pipeline.check(s) == ["parity.csv", "h5a.csv"]
    assert delay["arms"]["dp"]["d"] == 1


def test_child_body_times_a_tiny_ort_cpu_graph(tmp_path: Path) -> None:
    from deploy_support import tiny_arms, tiny_inputs
    from ila.deploy import export as ex
    from ila.deploy import parity as par

    s = tiny_settings(tmp_path)
    arms = tiny_arms()
    ex.export_loaded(arms, s, s.out_dir / "graphs")
    x = tiny_inputs(5)
    (s.out_dir / "inputs").mkdir(parents=True)
    np.savez(s.out_dir / "inputs" / "select_inputs.npz", images=x.images, proprio=x.proprio,
             episode=x.episode, step=x.step, episode_seed=x.episode_seed)  # fmt: skip
    ref = par.reference(arms, x, s)
    (s.out_dir / "parity").mkdir(parents=True)
    np.savez(s.out_dir / "parity" / "reference.npz", **ref.arrays())
    for row in ("dp_e2e", "dp_loop_K5", "bc_e2e"):
        rec = lat.run_child(lat.LatencyJob(row, "ort_cpu_fp32", 2, 2, 3, 1), s)
        assert rec["status"] == "timed" and rec["p50_ms"] > 0 and rec["max_ms"] >= rec["p99_ms"]
        assert rec["n_distinct_batches"] == 2 and rec["env"]["cpu"]
        assert all(v == ["CPUExecutionProvider"] for v in rec["providers_realized"].values())
