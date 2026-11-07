"""Phase 8 evaluation matrix and Phase 9 latency arm (background job; ``make eval-bg``).

Expands the frozen ``configs/eval/matrix.yaml`` (SHA-256 checked) into jobs
(:mod:`ila.eval.arms`) and flies them on P8's frozen lists through the selection path, CPU fp32
single-sample (:mod:`ila.eval.matrix`), writing ``results/e08/rows/<arm>/s<seed>/<stem>.csv.gz``
with a sidecar each, ``results/e08/run_info.json`` and ``artifacts/runs/eval/status.json``.
Resumable: re-run the same command and finished jobs are skipped. Refuses a dirty tree (the
output root excepted). Simulation only.

The ``latency_arm`` block (Phase 9, P3-D1 §7) needs its delay d per arm from the committed
``results/latency/delay.json`` (``make bench-score``); without it every mode refuses, except a
dry run given ``--assume-delay ARM=D`` (projection only, labelled ASSUMED; never a flight).

    python scripts/eval_matrix.py --dry-run              # jobs, episodes per tag, projection
    python scripts/eval_matrix.py --dry-run --only latency_arm --assume-delay bc_dagger=1 dp=4
    python scripts/eval_matrix.py --only latency_arm     # the latency arm (needs delay.json)
    python scripts/eval_matrix.py --smoke --out DIR --allow-dirty   # 2 select episodes per arm
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix

#: Episodes per arm in a smoke run (the first ``select`` episodes; never a frozen list).
SMOKE_EPISODES = 2


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="print jobs and the projection")
    parser.add_argument("--smoke", action="store_true", help="2 select episodes per (tag, arm)")
    parser.add_argument("--out", type=Path, default=None, help="output root (default results/e08)")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=10, help="episodes per task (rows unchanged)")
    parser.add_argument("--only", nargs="*", default=[], help="tags, arms, seeds (s<n>)")
    parser.add_argument(
        "--order",
        default=",".join(ev_arms.DEFAULT_ORDER),
        help="comma-separated tag order (default %(default)s)",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="smoke only")
    parser.add_argument(
        "--fly-ta1-ensemble",
        action="store_true",
        help="fly dp (Ta=1, ensembling on) instead of aliasing it to (Ta=1, off)",
    )
    parser.add_argument(
        "--assume-delay",
        nargs="*",
        default=None,
        metavar="ARM=D",
        help="dry run only, when results/latency/delay.json does not exist yet: the latency "
        "arm's d per arm (control steps) for the projection, printed as ASSUMED",
    )
    parser.add_argument("--status-dir", type=Path, default=None)
    parser.add_argument("--quiet-jobs", action="store_true", help="dry run: totals only")
    return parser.parse_args()


def _dry_run(matrix: ev_arms.Matrix, jobs: list[ev_arms.Job], workers: int, quiet: bool) -> None:
    ckpts = {(a.arm, s): ev_arms.resolve_checkpoint(a, s) for a in matrix.arms for s in a.seeds}
    proj = ev_matrix.project(jobs, ckpts, ev_matrix.ppo_cell_lengths())
    if not quiet:
        print("jobs (tag arm seed rows-file episodes projected-cpu-h):")
        for p in proj:
            j = p.job
            print(
                f"  {j.tag:18s} {j.arm:22s} s{j.seed} {j.rel_path.as_posix():70s} "
                f"{p.episodes:5d} {p.cpu_s / 3600:7.2f}"
            )
    by_tag: dict[str, list[ev_matrix.JobProjection]] = defaultdict(list)
    for p in proj:
        by_tag[p.job.tag].append(p)
    shared = defaultdict(int)
    for s in matrix.shared:
        shared[s.tag] += 1
    print(f"\nprojection at {workers} workers (CPU fp32 single-sample path):")
    print(
        f"  {'tag':18s} {'jobs':>5s} {'episodes':>9s} {'shared':>6s} {'CPU-h':>8s} {'wall-h':>7s}"
    )
    tot_j = tot_e = 0
    tot_c = 0.0
    for tag in [t for t in ev_arms.TAGS if t in by_tag]:
        ps = by_tag[tag]
        n_e = sum(p.episodes for p in ps)
        cpu = sum(p.cpu_s for p in ps)
        tot_j, tot_e, tot_c = tot_j + len(ps), tot_e + n_e, tot_c + cpu
        print(
            f"  {tag:18s} {len(ps):5d} {n_e:9d} {shared[tag]:6d} {cpu / 3600:8.1f} "
            f"{cpu / 3600 / workers:7.2f}"
        )
    print(
        f"  {'total':18s} {tot_j:5d} {tot_e:9d} {len(matrix.shared):6d} {tot_c / 3600:8.1f} "
        f"{tot_c / 3600 / workers:7.2f}"
    )
    print("\nshared cells (not flown; rows from another job):")
    for s in matrix.shared:
        if s.seed == 0:
            print(
                f"  {s.tag} {s.arm} Ta={s.inference.Ta} K={s.inference.K} "
                f"ens={int(s.inference.ensemble)} -> {s.source.rel_path.as_posix()} "
                f"(x{len(matrix.arm(s.arm).seeds)} seeds): {s.reason}"
            )
    print("\nskipped blocks:")
    for k, v in matrix.skipped.items():
        print(f"  {k}: {v}")
    print("\nassumptions:")
    for a in ev_matrix.PROJECTION_ASSUMPTIONS:
        print(f"  - {a}")


def _assumed(pairs: list[str]) -> dict[str, int]:
    out = {}
    for pair in pairs:
        arm, _, d = pair.partition("=")
        if not arm or not d.isdigit():
            raise SystemExit(f"--assume-delay takes ARM=D with D >= 0, got {pair!r}")
        out[arm] = int(d)
    return out


def main() -> int:
    args = _parse()
    order = [t.strip() for t in args.order.split(",") if t.strip()]
    if args.assume_delay is not None and not args.dry_run:
        print("--assume-delay is for --dry-run only; a flight reads delay.json", file=sys.stderr)
        return 2
    assumed = None
    if args.assume_delay is not None and not ev_arms.deploy_latency_arm.DELAY_FILE.is_file():
        assumed = _assumed(args.assume_delay)
    try:
        matrix = ev_arms.load_matrix(alias_ta1_ensemble=not args.fly_ta1_ensemble, delays=assumed)
    except ev_arms.LatencyDelayError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    jobs = ev_matrix.order_jobs(ev_matrix.filter_jobs(matrix.jobs, args.only), order)
    if args.dry_run:
        if assumed is not None:
            print(f"ASSUMED latency-arm delays (projection only, not delay.json): {assumed}\n")
        _dry_run(matrix, jobs, args.workers, args.quiet_jobs)
        return 0
    if args.smoke:
        if args.out is None:
            print("--smoke needs --out (a scratch directory)", file=sys.stderr)
            return 2
        results = (p8_adapter.REPO_ROOT / "results").resolve()
        if args.out.resolve().is_relative_to(results):
            print("--smoke must not write under results/", file=sys.stderr)
            return 2
        select = data_episodes.read_split(data_episodes.SELECT_NAME)[:SMOKE_EPISODES]
        cells = tuple(dict.fromkeys(e.ss for e in select))
        jobs = ev_matrix.smoke_jobs(jobs, cells)
        status_dir = args.status_dir or args.out / "status"
        record = ev_matrix.run_jobs(
            jobs,
            matrix,
            out_root=args.out,
            status_dir=status_dir,
            workers=min(args.workers, SMOKE_EPISODES),
            chunk=args.chunk,
            allow_dirty=args.allow_dirty,
            episodes_of=lambda _job: list(select),
            frozen=False,
        )
    else:
        if args.allow_dirty:
            print("--allow-dirty is for --smoke only", file=sys.stderr)
            return 2
        out = args.out or ev_matrix.OUT_ROOT
        record = ev_matrix.run_jobs(
            jobs,
            matrix,
            out_root=out,
            status_dir=args.status_dir or ev_matrix.STATUS_DIR,
            workers=args.workers,
            chunk=args.chunk,
        )
    print(
        f"done: {record['jobs_done']} of {record['jobs_total']} matrix jobs have rows", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
