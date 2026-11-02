"""Pre-gate review fixes (results-skeptic S1, S2, M3, M4): per-sea-state Ta = 1 vs Ta = 8 on the
committed sweep rows, dp vs bc at Ta = 1, seed-level sign test for late vs 40 k, and the dp
held-out curves (EMA and raw, by k-bucket) extracted from the train logs. Reads committed
results/ files and artifacts/runs/dp/s*/train_log.jsonl (the latter written out here).
"""
import csv, json, statistics
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parents[5]
SW = R / "results/phase6/sweeps/dp"
out = {}

def rows(s, cell):
    return list(csv.DictReader(open(SW / f"s{s}/{cell}_rows.csv")))

# S1: per sea state, and the calmest slice (SS3, 0 kn)
by = {}
for cell in ("Ta01_K010_off", "Ta08_K010_off"):
    for s in range(5):
        for r in rows(s, cell):
            for key in (r["ss"], f'{r["ss"]}_{float(r["speed_kn"]):g}kn'):
                d = by.setdefault(cell, {}).setdefault(key, {"n": 0, "success": 0, "lat": []})
                d["n"] += 1; d["success"] += r["outcome"] == "success"
                if r["lateral_offset_m"] not in ("", "nan"):
                    d["lat"].append(float(r["lateral_offset_m"]))
out["per_sea_state"] = {c: {k: {"n": v["n"], "success": v["success"],
                                "lateral_offset_median_m": round(statistics.median(v["lat"]), 4) if v["lat"] else None}
                            for k, v in sorted(d.items())} for c, d in by.items()}

# S2: dp at Ta = 1 vs bc (selected, unpaired), hard landings
dp1 = [sum(r["outcome"] == "success" for r in rows(s, "Ta01_K010_off")) for s in range(5)]
dp1_hard = sum(r["outcome"] == "hard_landing" for s in range(5) for r in rows(s, "Ta01_K010_off"))
bc = [r for r in csv.DictReader(open(R / "results/phase4/runs_summary.csv")) if r["arm"] == "bc"]
out["dp_Ta1_vs_bc"] = {"dp_Ta1_success": dp1, "dp_Ta1_hard_landing_of_750": dp1_hard,
                       "bc_selected_success": [int(r["select_success"]) for r in bc],
                       "bc_selected_hard_landing_of_750": sum(json.loads(r["outcomes"])["hard_landing"] for r in bc)}

# M3: late vs 40 k / 50 k at seed level (sign test), from the committed select rows
def succ(s, step):
    f = R / f"results/phase6/runs/dp/s{s}/select/s0_step{step:06d}_rows.csv"
    return sum(r["outcome"] == "success" for r in csv.DictReader(open(f)))
for ref in (40000, 50000):
    diffs = [succ(s, 100000) - succ(s, ref) for s in range(5)]
    k = sum(d > 0 for d in diffs); n = sum(d != 0 for d in diffs)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n)
    out[f"100k_minus_{ref // 1000}k"] = {"per_seed": diffs, "sign_test_two_sided_p": p}

# M4: held-out curves (every logged evaluation), minimum step and k0-24 share of the rise
curves = {}
for s in range(5):
    ev = [json.loads(l) for l in open(R / f"artifacts/runs/dp/s{s}/train_log.jsonl") if '"heldout"' in l]
    ev = [e for e in ev if e.get("event") == "heldout"]
    ema = [(e["step"], e["ema"]) for e in ev]
    smin, emin = min(ema, key=lambda x: x[1]["loss"])
    elast = ema[-1][1]
    ks = [k for k in elast if k.startswith("loss_k")]
    rise = {k: elast[k] - emin[k] for k in ks}
    curves[f"s{s}"] = {"steps": [e["step"] for e in ev], "ema": [e["ema"] for e in ev], "raw": [e["raw"] for e in ev],
                       "ema_min_step": smin, "ema_min": emin["loss"], "ema_last": elast["loss"],
                       "rise_pct": round(100 * (elast["loss"] / emin["loss"] - 1), 1),
                       "k0_24_share_of_rise": round(rise["loss_k0_24"] / sum(rise.values()), 3)}
json.dump(curves, open(R / "results/phase6/diagnostics/dp/heldout_curves.json", "w"))
out["heldout_summary"] = {s: {k: v for k, v in c.items() if k not in ("steps", "ema", "raw")} for s, c in curves.items()}
print(json.dumps(out, indent=1))
