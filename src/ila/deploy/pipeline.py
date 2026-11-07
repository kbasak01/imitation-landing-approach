"""Phase 9 stages behind ``scripts/deploy/bench.py`` and ``make bench-*`` (P9-D1).

Stages, in order (each writes its section of ``results/latency/run_info.json``: the ``ila``
commit, the dirty flag, the environment stamp, the graph hashes, its wall clock):

1. ``export``: :func:`ila.deploy.export.export_all` -> ``artifacts/deploy/graphs/`` (gitignored)
   and ``results/latency/graphs.json`` (the hashes, committed).
2. ``inputs``: :func:`ila.deploy.inputs.record_select_inputs` -> ``artifacts/deploy/inputs/``
   and ``results/latency/inputs.json``.
3. ``parity``: the torch fp32 CPU reference (cached in ``artifacts/deploy/parity/reference.npz``,
   its array hash in ``results/latency/reference.json``), then every provider
   (:mod:`ila.deploy.parity`) -> ``parity.{json,csv}``, ``parity_steps.csv``.
4. ``closed-loop``: :func:`ila.deploy.closed_loop.run_closed_loop` -> ``closed_loop/`` and
   ``closed_loop.csv``.
5. ``latency``: :func:`ila.deploy.latency.run_latency` -> ``jobs/*.json`` and ``latency.csv``.
6. ``score``: H5(a) and d -> ``h5a.csv``, ``delay.json``.

``check``: every CSV and ``delay.json`` re-rendered from the committed records and compared byte
for byte (no flight, no timing).

Every stage but ``check`` refuses a dirty tree (outside ``results/latency/``) unless told
otherwise, because each output records the commit it ran at.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ila import p8_adapter
from ila.deploy import closed_loop as cl
from ila.deploy import export as ex
from ila.deploy import inputs as inp
from ila.deploy import latency as lat
from ila.deploy import parity as par
from ila.deploy.settings import BenchSettings

__all__ = [
    "DirtyTreeError",
    "check",
    "stage_closed_loop",
    "stage_export",
    "stage_inputs",
    "stage_latency",
    "stage_parity",
    "stage_score",
]


class DirtyTreeError(RuntimeError):
    """The checkout has uncommitted changes outside ``results/latency/``."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, rec: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", "utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def _git_checked(settings: BenchSettings, allow_dirty: bool) -> dict[str, Any]:
    git = ex.git_state()
    try:
        rel = settings.results_dir.relative_to(p8_adapter.REPO_ROOT).as_posix().rstrip("/") + "/"
    except ValueError:  # an output root outside the checkout (tests)
        rel = "\0"
    dirty = [ln for ln in git["dirty_paths"] if not ln[3:].strip().startswith(rel)]
    if dirty and not allow_dirty:
        raise DirtyTreeError(f"uncommitted changes outside {rel}: {dirty[:10]}")
    return {**git, "dirty_outside_results": bool(dirty)}


def _run_info(settings: BenchSettings, stage: str, rec: Mapping[str, Any]) -> None:
    path = settings.results_dir / "run_info.json"
    info = _read_json(path) if path.is_file() else {"stages": {}}
    info["stages"][stage] = dict(rec)
    info["bench_yaml_sha256"] = ex.sha256_file(p8_adapter.REPO_ROOT / "configs/deploy/bench.yaml")
    info["p8_commit"] = p8_adapter.p8_commit()
    _write_json(path, info)


def _env() -> dict[str, str]:
    stamp: dict[str, str] = p8_adapter.latency_harness().environment_stamp()
    return stamp


def _graph_hashes(settings: BenchSettings) -> dict[str, str]:
    meta = ex.load_meta(settings.out_dir / "graphs")
    return {k: v["sha256"] for k, v in meta["graphs"].items()}


def stage_export(settings: BenchSettings, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Stage 1 (module docstring)."""
    git = _git_checked(settings, allow_dirty)
    t0 = time.perf_counter()
    meta = ex.export_all(settings)
    rec = {
        "graphs": {
            k: {
                kk: v[kk]
                for kk in ("sha256", "precision", "inputs", "outputs", "opsets", "rng_ops")
            }
            for k, v in meta["graphs"].items()
        },
        "sources": meta["sources"],
        "dp": meta["dp"],
        "meta_sha256": ex.sha256_file(settings.out_dir / "graphs" / ex.META_FILE),
        "versions": meta["versions"],
    }
    _write_json(settings.results_dir / "graphs.json", rec)
    _run_info(
        settings,
        "export",
        {
            "time_utc": _now(),
            "git": git,
            "wall_s": time.perf_counter() - t0,
            "graphs": _graph_hashes(settings),
        },
    )
    return rec


def stage_inputs(
    settings: BenchSettings, *, workers: int = 24, allow_dirty: bool = False
) -> dict[str, Any]:
    """Stage 2 (module docstring)."""
    git = _git_checked(settings, allow_dirty)
    t0 = time.perf_counter()
    rec = inp.record_select_inputs(
        settings.out_dir / "inputs",
        n=settings.n_inputs,
        draw_seed=settings.draw_seed,
        policy=settings.input_policy,
        workers=workers,
    )
    rec["file"] = Path(rec["file"]).resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    _write_json(settings.results_dir / "inputs.json", rec)
    _run_info(
        settings, "inputs", {"time_utc": _now(), "git": git, "wall_s": time.perf_counter() - t0}
    )
    return rec


def _inputs(settings: BenchSettings) -> inp.SelectInputs:
    rec = _read_json(settings.results_dir / "inputs.json")
    return inp.load_inputs(settings.out_dir / "inputs" / inp.INPUTS_FILE, rec["sha256"])


def _reference(settings: BenchSettings, log: Callable[[str], None]) -> par.Reference:
    path = settings.out_dir / "parity" / "reference.npz"
    rec_path = settings.results_dir / "reference.json"
    if path.is_file() and rec_path.is_file():
        with np.load(path, allow_pickle=False) as z:
            ref = par.Reference({k: z[k] for k in z.files})
        want = _read_json(rec_path)["arrays_sha256"]
        if par.array_sha256(ref.arrays()) != want:
            raise RuntimeError(f"{path}: arrays differ from reference.json")
        log("reference: cached")
        return ref
    arms = {a: ex.load_arm(a, s, settings.selection_pins) for a, s in settings.arm_seeds.items()}
    t0 = time.perf_counter()
    ref = par.reference(arms, _inputs(settings), settings, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=False, **ref.arrays())
    _write_json(
        rec_path,
        {
            "arrays_sha256": par.array_sha256(ref.arrays()),
            "path": "torch fp32 CPU, batch 1, 1 intra-op thread, EMA weights, TF32 n/a",
            "noise_key": f"(run_seed, {settings.noise_episode_key}, i) for input i",
            "timesteps": [int(t) for t in ref.timesteps],
            "n": ref.n,
            "wall_s": time.perf_counter() - t0,
            "checkpoints": {a: la.ckpt.sha256 for a, la in arms.items()},
        },
    )
    return ref


def stage_parity(
    settings: BenchSettings,
    *,
    providers: Sequence[str] | None = None,
    allow_dirty: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Stage 3 (module docstring).

    Providers already in ``parity.json`` are kept unless named in ``providers``.
    """
    from ila.deploy import backends as bk

    git = _git_checked(settings, allow_dirty)
    p8_adapter.latency_harness().disable_tf32()
    meta = ex.verify_graphs(settings.out_dir / "graphs")
    inputs = _inputs(settings)
    ref = _reference(settings, log)
    path = settings.results_dir / "parity.json"
    record = _read_json(path) if path.is_file() else {"providers": {}}
    wanted = [p for p in (providers or settings.providers) if p != settings.reference]
    dp_cfg = ex.dp_config_from_meta(meta)
    for provider in wanted:
        log(f"parity: {provider}")
        try:
            backend = bk.make_backend(
                provider,
                graph_dir=settings.out_dir / "graphs",
                dp_cfg=dp_cfg,
                arms_loader=lambda: {
                    a: ex.load_arm(a, s, settings.selection_pins)
                    for a, s in settings.arm_seeds.items()
                },
                batch=1,
                threads=settings.threads,
                trt_cache_root=settings.out_dir / "trt_cache",
            )
            rec = par.provider_parity(backend, inputs, ref, settings, log=log)
            rec["status"] = "measured"
        except Exception as exc:  # recorded with no number
            rec = {
                "provider": provider,
                "status": f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
            }
            log(f"parity {provider} FAILED: {rec['status']}")
        rec["scored"] = settings.is_scored(provider)
        rec["git"] = git
        rec["time_utc"] = _now()
        record["providers"][provider] = rec
    record["criterion"] = (
        f"bc_dagger action max-abs <= {settings.action_max_abs:g}; dp per-step eps-hat max-abs <= "
        f"{settings.eps_max_abs:g} at each of the K = {settings.dp_k} DDIM-trailing steps, own "
        "encoder, reference x_k (P3-D1 §6 H5(a), §7); final chunk reported, not judged; batch 1 "
        "against torch fp32 CPU batch 1 with the same initial noise; TF32 off"
    )
    record["inputs"] = _read_json(settings.results_dir / "inputs.json")
    record["reference"] = _read_json(settings.results_dir / "reference.json")
    record["graphs"] = _graph_hashes(settings)
    _write_json(path, record)
    (settings.results_dir / "parity.csv").write_text(
        par.parity_csv(record["providers"], settings), "utf-8"
    )
    (settings.results_dir / "parity_steps.csv").write_text(
        par.steps_csv(record["providers"], settings), "utf-8"
    )
    _run_info(
        settings,
        "parity",
        {"time_utc": _now(), "git": git, "env": _env(), "graphs": _graph_hashes(settings)},
    )
    return record


def stage_closed_loop(
    settings: BenchSettings,
    *,
    arms: Sequence[str] = ("bc_dagger", "dp"),
    providers: Sequence[str] | None = None,
    include_failed: bool = False,
    force: bool = False,
    allow_dirty: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Stage 4 (module docstring)."""
    git = _git_checked(settings, allow_dirty)
    ex.verify_graphs(settings.out_dir / "graphs")
    parity = _read_json(settings.results_dir / "parity.json")
    t0 = time.perf_counter()
    res = cl.run_closed_loop(
        settings,
        arms=arms,
        providers=list(providers or settings.providers),
        graph_dir=settings.out_dir / "graphs",
        open_loop=parity["providers"],
        include_failed=include_failed,
        force=force,
        log=log,
    )
    (settings.results_dir / "closed_loop.csv").write_text(cl.closed_loop_csv(settings), "utf-8")
    _run_info(
        settings,
        "closed_loop",
        {
            "time_utc": _now(),
            "git": git,
            "env": _env(),
            "graphs": _graph_hashes(settings),
            "wall_s": time.perf_counter() - t0,
        },
    )
    return res


def stage_latency(
    settings: BenchSettings,
    *,
    providers: Sequence[str] | None = None,
    rows: Sequence[str] | None = None,
    batches: Sequence[int] | None = None,
    force: bool = False,
    allow_dirty: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Stage 5 (module docstring)."""
    git = _git_checked(settings, allow_dirty)
    ex.verify_graphs(settings.out_dir / "graphs")
    parity = _read_json(settings.results_dir / "parity.json")
    jobs = [
        j
        for j in lat.latency_jobs(settings)
        if (providers is None or j.provider in providers)
        and (rows is None or j.row in rows)
        and (batches is None or j.batch in batches)
    ]
    t0 = time.perf_counter()
    lat.run_latency(settings, jobs, parity, force=force, git=git, log=log)
    _render_latency(settings)
    _run_info(
        settings,
        "latency",
        {
            "time_utc": _now(),
            "git": git,
            "env": _env(),
            "graphs": _graph_hashes(settings),
            "wall_s": time.perf_counter() - t0,
            "jobs_this_invocation": len(jobs),
        },
    )
    return _latency_records(settings)


def _latency_records(settings: BenchSettings) -> list[dict[str, Any]]:
    order = {j.label: i for i, j in enumerate(lat.latency_jobs(settings))}
    recs = [_read_json(p) for p in sorted((settings.results_dir / "jobs").glob("*.json"))]
    return sorted(recs, key=lambda r: order.get(r["label"], len(order)))


def _render_latency(settings: BenchSettings) -> str:
    text = lat.latency_csv(_latency_records(settings), settings)
    (settings.results_dir / "latency.csv").write_text(text, "utf-8")
    return text


def _closed_records(settings: BenchSettings) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    root = settings.results_dir / "closed_loop"
    for arm in ("bc_dagger", "dp"):
        out[arm] = (
            {p.stem: _read_json(p) for p in sorted((root / arm).glob("*.json"))}
            if (root / arm).is_dir()
            else {}
        )
    return out


def _score_outputs(settings: BenchSettings) -> dict[str, str]:
    parity = _read_json(settings.results_dir / "parity.json")
    verdict, table, delay = lat.score(
        settings, _latency_records(settings), parity, _closed_records(settings)
    )
    return {
        "h5a.csv": lat.h5a_csv(table),
        "delay.json": json.dumps(delay, indent=2, sort_keys=True) + "\n",
    }


def stage_score(settings: BenchSettings) -> dict[str, Any]:
    """Stage 6 (module docstring)."""
    outs = _score_outputs(settings)
    for name, text in outs.items():
        (settings.results_dir / name).write_text(text, "utf-8")
    _render_latency(settings)
    (settings.results_dir / "closed_loop.csv").write_text(cl.closed_loop_csv(settings), "utf-8")
    delay: dict[str, Any] = json.loads(outs["delay.json"])
    return delay


def check(settings: BenchSettings) -> list[str]:
    """Re-render every table from the committed records; return the files that differ."""
    r = settings.results_dir
    want: dict[str, str] = {}
    if (r / "parity.json").is_file():
        prov = _read_json(r / "parity.json")["providers"]
        want["parity.csv"] = par.parity_csv(prov, settings)
        want["parity_steps.csv"] = par.steps_csv(prov, settings)
    if (r / "closed_loop").is_dir():
        want["closed_loop.csv"] = cl.closed_loop_csv(settings)
    if (r / "jobs").is_dir():
        want["latency.csv"] = lat.latency_csv(_latency_records(settings), settings)
    if (r / "h5a.csv").is_file() or (r / "delay.json").is_file():
        want.update(_score_outputs(settings))
    bad = []
    for name, text in want.items():
        path = r / name
        if not path.is_file() or path.read_text("utf-8") != text:
            bad.append(name)
    return bad
