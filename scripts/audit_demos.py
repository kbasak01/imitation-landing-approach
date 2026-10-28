"""Audit the demo stores and write results/demos_summary.json (plan Phase 3 item 3).

Per store (``ila.data.audit``): disjointness and split identity from stored attributes; open-loop
replay of stored ``action_exec`` on 20 random steps (byte-identical frame and ``teacher_obs``);
closed-loop re-flight of the same episodes (every array byte-identical); re-label of 1 000
random steps with the stored teacher shadowed in order; counts by sea state / heading / speed,
outcomes per sea state, success per sea state with Wilson 95 %, kept fraction, frames, bytes,
SHA-256s, clear-colour statistics, touchdown-time quantiles (and, for ``mix_1000``, the H2
precondition: ppo- vs gated-assigned touchdown times and their W1). Random steps use seed
20261023 (recorded). The output is deterministic: no timestamps, no host, store paths relative
to ``data_root`` and run files relative to the repository. Example::

    python scripts/audit_demos.py                      # all six collections, 6 processes
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

from ila import p8_adapter
from ila.data import audit, collect
from ila.eval import runner

ROOT = p8_adapter.REPO_ROOT
MANIFEST_FIELDS = ("ila_commit", "ila_dirty_paths", "complete", "limit", "workers", "chunk",
                   "wall_s", "hdf5_write_s", "n_planned")  # fmt: skip


def _clean(value: Any) -> Any:
    """NaN/inf -> None, recursively, so the JSON is strict."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def audit_store(job: tuple[str, str, str, int, int, int]) -> dict[str, Any]:
    """Run every audit and the statistics for one store (picklable worker)."""
    name, data_root, run_root, seed, n_states, n_labels = job
    store = Path(data_root) / "demos" / f"{name}.h5"
    manifest = store.with_suffix(".manifest.json")
    rows_csv = Path(run_root) / name / "episodes.csv"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    entry: dict[str, Any] = {
        "store": f"${{data_root}}/demos/{name}.h5",
        "manifest": {
            "path": f"${{data_root}}/demos/{name}.manifest.json",
            "sha256": p8_adapter.sha256_file(manifest),
            **{k: m.get(k) for k in MANIFEST_FIELDS},
        },
        "runner_rows": {
            "path": rows_csv.resolve().relative_to(ROOT).as_posix(),
            "sha256": p8_adapter.sha256_file(rows_csv),
        },
        "summary": audit.store_summary(store, rows_csv),
        "attr_audit": audit.attr_audit(store),
        "replay_audit": audit.replay_audit(store, n_states, seed),
        "refly_audit": audit.refly_audit(store, n_states, seed),
        "relabel_audit": audit.relabel_audit(store, n_labels, seed),
    }
    if m.get("file_sha256") != entry["summary"]["file_sha256"]:
        entry["manifest_file_sha256_mismatch"] = True
    entry["passed"] = (
        all(
            entry[k]["passed"]
            for k in ("attr_audit", "replay_audit", "refly_audit", "relabel_audit")
        )
        and not entry.get("manifest_file_sha256_mismatch")
        and not entry["summary"].get("runner_rows_mismatch")
    )
    return entry


def main() -> int:
    names = list(collect.load_collections())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=names, help=f"collections (default {names})")
    parser.add_argument("--data-root", type=Path, default=None, help="default: paths.local.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "runs" / "collect")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demos_summary.json")
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--n-labels", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=audit.AUDIT_SEED)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    data_root = args.data_root or p8_adapter.load_paths()["data_root"]
    jobs = [
        (n, str(data_root), str(args.run_root), args.seed, args.n_states, args.n_labels)
        for n in args.names
    ]
    runner.set_single_thread_env()
    if args.workers == 1:
        runner.init_worker_threads()
        entries = [audit_store(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.workers, len(jobs)), initializer=runner.init_worker_threads) as p:
            entries = p.map(audit_store, jobs, chunksize=1)
    report: dict[str, Any] = {
        "phase": 3,
        "what": "demo-store audits and statistics (plan Phase 3 item 3)",
        "seed": args.seed,
        "sampling": (
            "sample_steps: n distinct (episode, step) pairs uniform over all stored frames from "
            "default_rng(SeedSequence([seed, sha256_entropy(purpose), total_frames])), purpose "
            "'replay' (replay and refly audits share the pairs) or 'relabel'"
        ),
        "n_states": args.n_states,
        "n_labels": args.n_labels,
        "statistics": (
            "success intervals: P8 wilson_interval (z = 1.96, analytic; no bootstrap used); "
            "quantiles: P8 quantile_linear; W1: exact 1-D empirical Wasserstein-1, s model scale"
        ),
        "units": "times s model scale (lambda = 1/25); touchdown time = P8 time_to_touchdown_s",
        "p8_commit": p8_adapter.p8_commit(),
        "audit_code_sha256": {
            rel: p8_adapter.sha256_file(ROOT / rel)
            for rel in ("src/ila/data/audit.py", "scripts/audit_demos.py")
        },
        "data_root": "${data_root} = configs/paths.local.yaml data_root",
        "collections": {e["summary"]["collection"]: e for e in entries},
    }
    report["passed"] = all(e["passed"] for e in entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(report), indent=2, allow_nan=False) + "\n"
    args.out.write_text(text, encoding="utf-8")
    for e in entries:
        s = e["summary"]
        print(
            f"{s['collection']}: attr {e['attr_audit']['passed']}, replay "
            f"{e['replay_audit']['frames_identical']}/{e['replay_audit']['n_states']} frames "
            f"{e['replay_audit']['teacher_obs_identical']} obs, refly "
            f"{e['refly_audit']['frames_identical']}/{e['refly_audit']['n_states']}, relabel "
            f"{e['relabel_audit']['identical']}/{e['relabel_audit']['n_steps']}, success "
            f"{s['success']['rate']:.4f}, passed {e['passed']}"
        )
    print(f"overall passed: {report['passed']} -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
