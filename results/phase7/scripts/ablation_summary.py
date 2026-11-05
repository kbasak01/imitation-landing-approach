"""Phase 7 run summary on `select`, recomputed from committed files only (descriptive; P7-D3).

Inputs: results/phase7/runs/<arm>/s<n>/{selection.json, select/*_rows.csv.gz} (Phase 7 arms),
results/phase6/runs/dp/s<n>/ (core `dp` rows) and results/phase{4,5}/runs/*/s<n>/selection.json
(core BC-family candidate counts; their rows are not committed).

Per arm:
- selected success per seed;
- candidate successes, candidate mean and inflation (selected minus the seed's candidate mean);
- and, where rows are committed, pooled over every candidate's 150 episodes:
  - outcome rates;
  - success by sea state;
  - touchdown-time bands (fast < 1.8 s, slow > 2.8 s, `time_to_touchdown_s`, model scale);
  - per-band success;
  - lateral offset and closing speed (`closing_speed_normal_m_s`) p50 / p95, model scale
    (lambda = 1/25: x25 lengths, x5 times and speeds at full scale);
  - `dp_*` arms also get the paired per-episode success difference vs core `dp` (all candidates,
    and selected-only), with a 95 % percentile bootstrap over the 150 episodes
    (rng seed 0, 20 000 replicates). This is ad hoc and descriptive, not P8's paired_bootstrap.

Everything is in-sample on `select`. Nothing is scored or selected from this file.
Writes results/phase7/diagnostics/ablation_summary.json.
"""
import csv, gzip, io, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase7/diagnostics/ablation_summary.json"
FAST, SLOW = 1.8, 2.8
P7 = {"modular": 5, **{a: 3 for a in (
    "bc_mix", "bc_pid", "bc_n100", "bc_n300", "bc_minus_last_action", "bc_plus_clock",
    "dp_mix", "dp_pid", "dp_n100", "dp_n300", "dp_minus_last_action")}}
CORE = {"bc": ("phase4", 5), "bc_state": ("phase4", 5), "bc_matched": ("phase4", 5),
        "bc_dart": ("phase4", 5), "bc_dagger": ("phase5", 5), "dp": ("phase6", 5)}


def read_rows(path: Path) -> list[dict[str, str]]:
    raw = gzip.open(path, "rt") if path.suffix == ".gz" else open(path)
    with raw as f:
        return list(csv.DictReader(io.StringIO(f.read())))


def run_dirs(arm: str) -> list[Path]:
    if arm in P7:
        return [ROOT / f"results/phase7/runs/{arm}/s{s}" for s in range(P7[arm])]
    phase, n = CORE[arm]
    return [ROOT / f"results/{phase}/runs/{arm}/s{s}" for s in range(n)]


def q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p)) if x.size else float("nan")


def summarise(arm: str) -> tuple[dict, dict[int, np.ndarray] | None]:
    seeds, pooled, per_ep_all, per_ep_sel = [], [], [], []
    for d in run_dirs(arm):
        sel = json.loads((d / "selection.json").read_text())
        cands = [c["success"] for c in sel["candidates"]]
        chosen = sel["selected"]["candidate"]["sha256"]
        seeds.append({"selected_success": sel["selected"]["success"],
                      "selected_step": sel["selected"]["candidate"]["global_step"],
                      "candidates": cands, "candidate_mean": float(np.mean(cands)),
                      "inflation": sel["selected"]["success"] - float(np.mean(cands))})
        for c in sel["candidates"]:
            f = d / "select" / c["rows_file"]
            f = f if f.exists() else f.with_name(f.name + ".gz")
            if not f.exists():
                continue
            rows = read_rows(f)
            pooled += rows
            ok = np.array([r["outcome"] == "success" for r in rows], float)
            per_ep_all.append(ok)
            if c["candidate"]["sha256"] == chosen:
                per_ep_sel.append(ok)
    rec: dict = {"seeds": seeds,
                 "candidate_mean_pooled": float(np.mean([c for s in seeds for c in s["candidates"]])),
                 "selected_mean": float(np.mean([s["selected_success"] for s in seeds]))}
    if pooled:
        out = [r["outcome"] for r in pooled]
        td = np.array([float(r["time_to_touchdown_s"]) for r in pooled])
        lat = np.array([float(r["lateral_offset_m"]) for r in pooled])
        cs = np.array([float(r["closing_speed_normal_m_s"]) for r in pooled])
        succ = np.array([o == "success" for o in out])
        fin = np.isfinite(td)
        bands = {"fast": fin & (td < FAST), "between": fin & (td >= FAST) & (td <= SLOW),
                 "slow": fin & (td > SLOW)}
        rec["all_candidates"] = {
            "n_episodes": len(pooled),
            "outcome_rate": {k: out.count(k) / len(out) for k in
                             ("success", "off_pad", "hard_landing", "bounce", "crash", "timeout")},
            "success_by_ss": {ss: float(succ[[r["ss"] == ss for r in pooled]].mean())
                              for ss in sorted({r["ss"] for r in pooled})},
            "touchdown_s_p10_p50_p90": [q(td[fin], .1), q(td[fin], .5), q(td[fin], .9)],
            "band_share": {k: float(m.sum() / fin.sum()) for k, m in bands.items()},
            "band_success": {k: float(succ[m].mean()) if m.any() else None for k, m in bands.items()},
            "fast_band_lateral_p50_m": q(lat[bands["fast"] & np.isfinite(lat)], .5),
            "fast_band_closing_p50_m_s": q(cs[bands["fast"] & np.isfinite(cs)], .5),
            "lateral_offset_m_p50_p95": [q(lat[np.isfinite(lat)], .5), q(lat[np.isfinite(lat)], .95)],
            "closing_speed_m_s_p50_p95": [q(cs[np.isfinite(cs)], .5), q(cs[np.isfinite(cs)], .95)],
            "frac_closing_above_0p5_m_s": float((cs[np.isfinite(cs)] > 0.5).mean()),
            "n_tunnelled": sum(r["tunnelled"] in ("True", "true", "1") for r in pooled),
        }
        return rec, {"all": np.mean(per_ep_all, 0), "sel": np.mean(per_ep_sel, 0)}
    return rec, None


def boot(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, diff.size, (20_000, diff.size))
    m = diff[idx].mean(1)
    return [float(diff.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


res, per_ep = {}, {}
for arm in list(CORE) + list(P7):
    res[arm], per_ep[arm] = summarise(arm)
for arm in P7:
    if arm.startswith("dp_") and per_ep[arm] is not None:
        d_all = per_ep[arm]["all"] - per_ep["dp"]["all"]
        res[arm]["paired_vs_dp"] = {
            "all_candidates_mean_ci95": boot(d_all),
            "selected_mean_ci95": boot(per_ep[arm]["sel"] - per_ep["dp"]["sel"]),
            "episodes_better_worse_equal": [int((d_all > 0).sum()), int((d_all < 0).sum()),
                                            int((d_all == 0).sum())]}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "bands_s": [FAST, SLOW],
                           "arms": res}, indent=1) + "\n")
for arm, r in res.items():
    a = r.get("all_candidates", {})
    print(f"{arm:22s} sel {[s['selected_success'] for s in r['seeds']]} "
          f"cand {r['candidate_mean_pooled']:.1f} "
          f"bands {[round(v, 3) for v in a.get('band_share', {}).values()]} "
          f"off {a.get('outcome_rate', {}).get('off_pad', float('nan')):.3f} "
          f"pair {r.get('paired_vs_dp', {}).get('all_candidates_mean_ci95')}")
