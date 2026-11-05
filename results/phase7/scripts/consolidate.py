"""Copy the Phase 7 run evidence (modular s0-s4, 11 ablations s0-s2) into results/phase7/ and
build runs_summary.csv and runs_candidates.csv.

Read-only over artifacts/; writes only under results/phase7/. Per run: manifest.json,
selection.json and every candidate's `select` rows (gzipped, select/<ckpt>_rows.csv.gz) so the
selection rule can be re-applied from committed files. runs_summary.csv columns follow
results/phase4/runs_summary.csv plus `obs_variant`; runs_candidates.csv has one row per
(run, checkpoint) with success, outcome classes and pooled p95 closing speed (m/s, model scale,
lambda = 1/25) on the 150 `select` episodes. All numbers are in-sample on `select`.
"""
import csv, gzip, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase7"
P4COLS = next(csv.reader(open(ROOT / "results/phase4/runs_summary.csv")))
ARMS = [("modular", range(5))] + [
    (a, range(3))
    for a in ("bc_mix", "bc_pid", "bc_n100", "bc_n300", "bc_minus_last_action", "bc_plus_clock",
              "dp_mix", "dp_pid", "dp_n100", "dp_n300", "dp_minus_last_action")
]
OUTCOMES = ("success", "bounce", "hard_landing", "off_pad", "crash", "timeout")
rows, cands = [], []
for arm, seeds in ARMS:
    for s in seeds:
        run, dst = ROOT / f"artifacts/runs/{arm}/s{s}", OUT / f"runs/{arm}/s{s}"
        (dst / "select").mkdir(parents=True, exist_ok=True)
        for f in ("manifest.json", "selection.json"):
            shutil.copy2(run / f, dst / f)
        m = json.loads((run / "manifest.json").read_text())
        d = json.loads((run / "selection.json").read_text())
        for c in d["candidates"]:
            src = run / "select" / c["rows_file"]
            with open(src, "rb") as fi, gzip.GzipFile(
                dst / "select" / (c["rows_file"] + ".gz"), "wb", mtime=0
            ) as fo:
                shutil.copyfileobj(fi, fo)
            cands.append({
                "run": f"{arm}/s{s}", "arm": arm, "seed": s,
                "global_step": c["candidate"]["global_step"],
                "selected": c["candidate"]["sha256"] == d["selected"]["candidate"]["sha256"],
                "success": c["success"], "n": c["n"],
                **{f"n_{k}": c["outcomes"].get(k, 0) for k in OUTCOMES},
                "p95_closing_speed_m_s": c["p95_closing_speed_m_s"],
                "mean_steps": c.get("mean_steps", ""), "ckpt_sha256": c["candidate"]["sha256"]})
        sel = d["selected"]
        c = sel["candidate"]
        step = c.get("global_step", c.get("step"))
        rows.append({
            "run": f"{arm}/s{s}", "arm": arm, "seed": s, "purpose": m["purpose"], "sigma": "",
            "ila_commit": m["git"]["ila_commit"][:7], "ila_dirty": m["git"]["ila_dirty"],
            "total_steps": m["total_steps"], "wall_clock_s": round(m["wall_clock_s"], 1),
            "gpu_hours": round(m["gpu_hours"], 4), "resumes": len(m["resumes"]),
            "selected_checkpoint": Path(c.get("file", c.get("dir", ""))).name or f"step{step}",
            "selected_global_step": step, "select_success": sel["success"], "select_n": sel["n"],
            "outcomes": json.dumps(dict(sorted(sel["outcomes"].items()))),
            "p95_closing_speed_m_s": sel["p95_closing_speed_m_s"],
            "mean_steps": sel.get("mean_steps", ""),
            "selected_ckpt_sha256": c["sha256"], "n_tunnelled": sel["n_tunnelled"],
            "n_detectors_disagree": sel["n_detectors_disagree"],
            "obs_variant": d.get("obs_variant", "")})
with open(OUT / "runs_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=P4COLS + ["obs_variant"])
    w.writeheader()
    w.writerows(rows)
with open(OUT / "runs_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cands[0]))
    w.writeheader()
    w.writerows(cands)
print(f"{len(rows)} runs, {len(cands)} candidates; "
      f"gpu_hours {sum(r['gpu_hours'] for r in rows):.2f}")
