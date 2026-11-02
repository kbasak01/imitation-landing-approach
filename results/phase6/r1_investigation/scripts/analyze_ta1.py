"""Ta = 1 diagnostic flights (replan every step) vs Ta = 8: success, outcomes, lateral offsets."""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
ORDER = ["crash", "off_pad", "hard_landing", "bounce", "success", "timeout"]


def load(name: str) -> dict:
    return {(r["variant"], r["run_seed"]): r for r in pickle.load(open(S / name, "rb"))["results"]}


def main() -> None:
    res = {}
    for f in ("flights_s0-9.pkl", "flights_ta1_s0-4.pkl", "flights_ta1_s5-9.pkl",
              "flights_ta1_dpraw_s0-9.pkl"):
        if (S / f).exists():
            res.update(load(f))
    seeds = list(range(10))
    variants = ["r1", "dp_ema", "dp_raw", "r1_ta1", "dp_ema_ta1", "dp_raw_ta1"]
    mats, out = {}, {}
    for v in variants:
        ss = [s for s in seeds if (v, s) in res]
        mat = np.array([[r["outcome"] == "success" for r in res[(v, s)]["rows"]] for s in ss])
        mats[v] = (ss, mat)
        offs = np.array([float(r["lateral_offset_m"]) for s in ss for r in res[(v, s)]["rows"]])
        offs = offs[np.isfinite(offs)]
        cs = np.array([float(r["closing_speed_normal_m_s"]) for s in ss for r in res[(v, s)]["rows"]])
        cs = cs[np.isfinite(cs)]
        c = Counter(r["outcome"] for s in ss for r in res[(v, s)]["rows"])
        out[v] = {"seeds": ss, "success_counts": mat.sum(1).tolist(), "pooled": f"{int(mat.sum())}/{mat.size}",
                  "rate": float(mat.mean()), "outcomes": {o: c[o] for o in ORDER if c[o]},
                  "lat_off_mean_m": float(offs.mean()), "lat_off_median_m": float(np.median(offs)),
                  "frac_lat_gt_0.10": float((offs > 0.10).mean()),
                  "seeds_ge_7": int((mat.sum(1) >= 7).sum()),
                  "closing_mean_m_s": float(cs.mean()), "closing_p95_m_s": float(np.quantile(cs, 0.95)),
                  "frac_closing_gt_0.45": float((cs > 0.45).mean())}
    rng = np.random.default_rng(0)
    for a, b in (("r1_ta1", "dp_ema_ta1"), ("dp_raw_ta1", "dp_ema_ta1"), ("r1_ta1", "dp_raw_ta1"),
                 ("r1_ta1", "r1"), ("dp_ema_ta1", "dp_ema"), ("dp_raw_ta1", "dp_raw")):
        sa, ma = mats[a]
        sb, mb = mats[b]
        common = [s for s in sa if s in sb]
        da = ma[[sa.index(s) for s in common]].astype(float)
        db = mb[[sb.index(s) for s in common]].astype(float)
        diff = da - db
        boots = [diff[np.ix_(rng.integers(0, len(common), len(common)), rng.integers(0, 8, 8))].mean()
                 for _ in range(20000)]
        out[f"diff_{a}_minus_{b}"] = {"seeds": common, "point": float(diff.mean()),
                                      "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    (S / "analysis_ta1.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
