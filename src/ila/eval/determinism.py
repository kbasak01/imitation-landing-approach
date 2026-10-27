"""Phase 2 determinism contract (plan Phase 2 item 5): dummy dp, CPU vs batched GPU.

The policy is ``build_dummy("dp", 0)`` (random init, seed 0 = the run seed of every noise
key), K = 10 DDIM steps, Ta = 8, flown on all 150 ``select`` episodes
(``splits/select.parquet``, checked against ``splits/MANIFEST.csv``) with render ``full``:

* ``cpu_w1`` / ``cpu_w8``: the CPU fp32 single-sample path (in-worker
  ``DPStudent.act_vision``, torch threads 1) at 1 and 8 workers. Their row text must be
  bit-identical (the contract's first half).
* ``gpu_B1`` / ``gpu_B8`` / ``gpu_B24``: lockstep workers (24) + one CUDA server holding one
  shared network (:class:`ila.eval.batched.SharedDPPolicy`), batch cap B, numerics
  ``strict`` (TF32 off). Compared with ``cpu_w1``: rows whose P8-column text differs, outcome
  class flips, and per-episode differences.
* ``gpu_B1_repeat``: ``gpu_B1`` again (GPU run-to-run reproducibility at a fixed batch cap).
* ``gpu_B24_torch_default``: the same at B = 24 with torch's default numerics (cuDNN TF32
  on), the labelled control.

Beside the GPU flip rate: P8's one-ulp floor, 53 of 800 ``id`` SS6 policy-episodes (6.6 %)
flip outcome class under at least one of 20 one-ulp input perturbations
(``external/rl-land-on-moving-deck/docs/protocol.md`` lines 5314 and 5455). ``select`` is
SS3-SS5, so that SS6 floor is a conservative comparator. Episode lengths are 30 Hz control
steps, model scale (lambda = 1/25); times wall seconds.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import batched
from ila.eval.jobs import JobStatus, host_info, rel, write_json
from ila.eval.runner import (
    RowTags,
    init_worker_threads,
    p8_rows,
    run_list_vision,
    set_single_thread_env,
    vision_rows_csv,
)

__all__ = ["K", "P8_FLOOR", "TA", "Plan", "compare_runs", "run", "select_episodes"]

#: DDIM steps and executed actions per chunk of the contract run.
K: Final[int] = 10
TA: Final[int] = 8

#: P8's one-ulp outcome-flip floor (P8 docs/protocol.md L5314, L5455).
P8_FLOOR: Final[dict[str, Any]] = {
    "flipped": 53,
    "of": 800,
    "rate": 53 / 800,
    "cell": "id SS6 (4 policies x 200 episodes)",
    "definition": "outcome class flips under at least one of 20 one-ulp input perturbations",
    "source": "external/rl-land-on-moving-deck/docs/protocol.md L5314, L5455",
}


@dataclass(frozen=True)
class Plan:
    """Which runs, on how many episodes."""

    n_episodes: int | None = None
    cpu_workers: tuple[int, ...] = (1, 8)
    gpu_workers: int = 24
    gpu_chunk: int = 5
    gpu_caps: tuple[int, ...] = (1, 8, 24)
    control_cap: int = 24

    @classmethod
    def smoke(cls) -> Plan:
        """Three episodes, two CPU worker counts, caps 1 and 3, the control at 3."""
        return cls(
            n_episodes=3,
            cpu_workers=(1, 2),
            gpu_workers=3,
            gpu_chunk=1,
            gpu_caps=(1, 3),
            control_cap=3,
        )


def select_episodes() -> tuple[list[p8_adapter.ListedEpisode], dict[str, str]]:
    """Read ``splits/select.parquet`` and check it against ``splits/MANIFEST.csv``.

    Returns:
        ``(episodes, {file_sha256, content_sha256})``.

    Raises:
        RuntimeError: If the file or content hash differs from the manifest's.
    """
    path = data_episodes.SPLITS_DIR / "select.parquet"
    manifest = {
        r["file"]: r for r in data_episodes.read_manifest(data_episodes.SPLITS_DIR / "MANIFEST.csv")
    }
    rows = p8_adapter.read_list(path)
    got = {
        "file_sha256": data_episodes.file_sha256(path),
        "content_sha256": data_episodes.content_sha256(rows),
    }
    entry = manifest["select.parquet"]
    for key, value in got.items():
        if entry[key] != value:
            raise RuntimeError(f"select.parquet {key} {value} != manifest {entry[key]}")
    return rows, got


def _spec() -> p8_adapter.PolicySpec:
    return p8_adapter.callable_spec(
        "dummy_dp",
        functools.partial(batched.build_dummy_student, "dp", 0, "cpu", TA, K),
        run_seed=0,
    )


_TAGS: Final = RowTags(arm="dummy_dp", student_seed=0, Ta=TA, K=K, ensemble=False)

#: Numeric P8 columns whose per-episode absolute difference is reported.
_NUMERIC: Final[tuple[str, ...]] = (
    "steps",
    "return",
    "closing_speed_normal_m_s",
    "lateral_offset_m",
    "rel_tilt_deg",
    "effort_mean_sq",
    "action_jerk_mean",
)


def _lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    return p8_adapter.episode_rows_csv(p8_rows(rows)).split("\n")[1:-1]


def compare_runs(
    reference: Sequence[dict[str, Any]], other: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare a run with the reference row by row (same list order).

    Returns:
        ``(summary, per-episode differences)``: rows whose P8-column text differs, outcome
        flips (with the class pairs), and per differing episode the differing columns and
        the absolute differences of :data:`_NUMERIC`.
    """
    ref_lines, new_lines = _lines(reference), _lines(other)
    if len(ref_lines) != len(new_lines):
        raise ValueError(f"{len(ref_lines)} reference rows, {len(new_lines)} new")
    diffs: list[dict[str, Any]] = []
    flips: dict[str, int] = {}
    for a, b, la, lb in zip(reference, other, ref_lines, new_lines, strict=True):
        if (a["regime"], a["ss"], a["index"]) != (b["regime"], b["ss"], b["index"]):
            raise ValueError("runs are not in the same list order")
        if la == lb:
            continue
        cols = [c for c in p8_adapter.EPISODE_COLUMNS if repr(a[c]) != repr(b[c])]
        record: dict[str, Any] = {
            "ss": a["ss"],
            "index": a["index"],
            "episode_seed": a["episode_seed"],
            "outcome_ref": a["outcome"],
            "outcome_new": b["outcome"],
            "flip": a["outcome"] != b["outcome"],
            "differing_columns": "|".join(cols),
        }
        for c in _NUMERIC:
            x, y = float(a[c]), float(b[c])
            record[f"absdiff_{c}"] = "" if math.isnan(x) and math.isnan(y) else abs(x - y)
        diffs.append(record)
        if record["flip"]:
            pair = f"{a['outcome']}->{b['outcome']}"
            flips[pair] = flips.get(pair, 0) + 1
    n = len(ref_lines)
    n_flips = sum(flips.values())
    summary = {
        "n_rows": n,
        "n_text_differ": len(diffs),
        "n_text_identical": n - len(diffs),
        "n_outcome_flips": n_flips,
        "flip_rate": n_flips / n if n else float("nan"),
        "flips_by_class": flips,
        "bit_identical": not diffs,
    }
    return summary, diffs


def _outcomes(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    text = vision_rows_csv(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _sha(p8_adapter.episode_rows_csv(p8_rows(rows)))


def run(*, out_dir: Path, run_dir: Path, smoke: bool = False) -> dict[str, Any]:
    """Fly every run, compare, and write ``summary.json``, ``diffs.csv`` and ``run_info.json``.

    Args:
        out_dir: ``results/determinism`` (or the smoke's run directory).
        run_dir: Rows and ``status.json`` (gitignored).
        smoke: Run :meth:`Plan.smoke`.

    Returns:
        The summary record.
    """
    plan = Plan.smoke() if smoke else Plan()
    status = JobStatus(run_dir, "determinism" + ("_smoke" if smoke else ""))
    try:
        import torch

        threads = set_single_thread_env()
        init_worker_threads()
        cfgs = p8_adapter.load_eval_configs()
        episodes, select_hashes = select_episodes()
        if plan.n_episodes is not None:
            episodes = episodes[: plan.n_episodes]
        spec = _spec()
        started = time.perf_counter()
        runs: dict[str, list[dict[str, Any]]] = {}
        meta: dict[str, dict[str, Any]] = {}

        def tick(name: str) -> None:
            status.update(
                "running",
                progress={
                    "run": name,
                    "done": list(runs),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                },
            )

        for w in plan.cpu_workers:
            name = f"cpu_w{w}"
            tick(name)
            t = time.perf_counter()
            runs[name] = run_list_vision(
                episodes, spec, cfgs, _TAGS, workers=w, chunk=5 if smoke else 25
            )
            meta[name] = {
                "path": "cpu fp32 single-sample (in-worker act_vision)",
                "workers": w,
                "wall_s": round(time.perf_counter() - t, 1),
            }
        gpu_runs: list[tuple[str, int, str]] = [(f"gpu_B{b}", b, "strict") for b in plan.gpu_caps]
        gpu_runs.append((f"gpu_B{plan.gpu_caps[0]}_repeat", plan.gpu_caps[0], "strict"))
        gpu_runs.append(
            (f"gpu_B{plan.control_cap}_torch_default", plan.control_cap, "torch_default")
        )
        if not torch.cuda.is_available():
            gpu_runs = []
        for name, cap, numerics in gpu_runs:
            tick(name)
            built: list[batched.TimedPolicy] = []
            inner = batched.shared_net_factory(
                "dp", 0, "cuda", n_action_steps=TA, num_inference_steps=K, numerics=numerics
            )

            def server(
                c: p8_adapter.EvalConfigs, n: int, inner: Any = inner, built: Any = built
            ) -> Any:
                timed = batched.TimedPolicy(inner(c, n))
                built.append(timed)
                return timed

            arm = batched.BatchedArm(
                spec=spec, episodes=episodes, tags=_TAGS, server_build=server, batch_cap=cap
            )
            t = time.perf_counter()
            runs[name] = batched.run_batched(
                arm, cfgs, workers=plan.gpu_workers, chunk=plan.gpu_chunk
            )
            timed = built[0]
            inner_policy = timed.inner
            sizes = getattr(inner_policy, "sample_calls", [])
            hist: dict[str, int] = {}
            for s in sizes:
                hist[str(s)] = hist.get(str(s), 0) + 1
            meta[name] = {
                "path": "lockstep workers + CUDA server, shared net (SharedDPPolicy)",
                "workers": plan.gpu_workers,
                "chunk": plan.gpu_chunk,
                "batch_cap": cap,
                "numerics": getattr(inner_policy, "numerics", {}),
                "sample_batch_calls": len(sizes),
                "sample_batch_size_histogram": dict(
                    sorted(hist.items(), key=lambda kv: int(kv[0]))
                ),
                "wall_s": round(time.perf_counter() - t, 1),
            }
        # Rows and comparisons.
        row_sha = {
            name: _write_rows(run_dir / f"rows_{name}.csv", rows) for name, rows in runs.items()
        }
        reference_name = f"cpu_w{plan.cpu_workers[0]}"
        reference = runs[reference_name]
        comparisons: dict[str, Any] = {}
        all_diffs: list[dict[str, Any]] = []
        for name, rows in runs.items():
            if name == reference_name:
                continue
            summary, diffs = compare_runs(reference, rows)
            comparisons[name] = {**summary, **meta[name]}
            all_diffs += [{"run": name, **d} for d in diffs]
        gpu_ref = f"gpu_B{plan.gpu_caps[0]}"
        vs_gpu_ref: dict[str, Any] = {}
        if gpu_ref in runs:
            for name, rows in runs.items():
                if name.startswith("gpu_") and name != gpu_ref:
                    vs_gpu_ref[name], _ = compare_runs(runs[gpu_ref], rows)
        cpu_names = [f"cpu_w{w}" for w in plan.cpu_workers]
        cpu_identical = all(comparisons[n]["bit_identical"] for n in cpu_names[1:])
        record = {
            "item": "plan Phase 2 item 5 (determinism contract)",
            "smoke": smoke,
            "policy": {
                "arm": "dp",
                "build": "ila.policies.registry.build_dummy('dp', 0)",
                "init": "random (seed 0)",
                "K": K,
                "Ta": TA,
                "run_seed": 0,
            },
            "episodes": {
                "list": "splits/select.parquet",
                "n": len(episodes),
                **select_hashes,
                "sea_states": sorted({e.ss for e in episodes}),
            },
            "reference_run": reference_name,
            "cpu_single_sample_bit_identical_across_workers": cpu_identical,
            "outcomes": {name: _outcomes(rows) for name, rows in runs.items()},
            "mean_steps": {
                name: sum(int(r["steps"]) for r in rows) / max(len(rows), 1)
                for name, rows in runs.items()
            },
            "comparisons_vs_reference": comparisons,
            "gpu_comparisons_vs_" + gpu_ref: vs_gpu_ref,
            "p8_one_ulp_floor": P8_FLOOR,
            "p8_column_rows_sha256": row_sha,
            "caveats": [
                "select is SS3-SS5; P8's floor is measured at id SS6, so it is a conservative "
                "comparator",
                "the network is random-init: its episodes end early (mostly crash), so outcome "
                "flips have little room to show; the row-text and numeric differences are the "
                "informative measures until a trained dp exists",
            ],
        }
        write_json(out_dir / "summary.json", record)
        _write_diffs(out_dir / "diffs.csv", all_diffs)
        write_json(
            out_dir / "run_info.json",
            {
                "wall_s": round(time.perf_counter() - started, 1),
                "plan": {k: getattr(plan, k) for k in plan.__dataclass_fields__},
                "threads": {**threads, "torch_num_threads": 1},
                "rows_dir": rel(run_dir),
                "runs": meta,
                **host_info(),
            },
        )
        status.update("done", passed_cpu=cpu_identical)
        return record
    except BaseException as exc:
        status.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _write_diffs(path: Path, diffs: Sequence[dict[str, Any]]) -> None:
    columns = [
        "run",
        "ss",
        "index",
        "episode_seed",
        "outcome_ref",
        "outcome_new",
        "flip",
        "differing_columns",
        *(f"absdiff_{c}" for c in _NUMERIC),
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for d in diffs:
        writer.writerow({c: d.get(c, "") for c in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")
