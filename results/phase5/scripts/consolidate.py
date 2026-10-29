"""Copy the Phase 5 bc_dagger run evidence into results/phase5/ and build runs_summary.csv.

Read-only over artifacts/ and ${data_root}; writes only under results/phase5/. Columns of
runs_summary.csv follow results/phase4/runs_summary.csv, plus the CPU seconds (rollouts, audit,
aggregate load) inside each run's wall clock (P5-D1 item 11) and GPU-h net of them.
"""
import csv, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts/runs/bc_dagger"
DATA = Path("/home/kbasak/data/ila/dagger/bc_dagger")
OUT = ROOT / "results/phase5"
P4COLS = next(csv.reader(open(ROOT / "results/phase4/runs_summary.csv")))
rows = []
for s in range(5):
    run, dst = RUNS / f"s{s}", OUT / f"runs/bc_dagger/s{s}"
    (dst / "dagger").mkdir(parents=True, exist_ok=True); (dst / "rollouts").mkdir(exist_ok=True)
    for f in ("manifest.json", "selection.json"): shutil.copy2(run / f, dst / f)
    for f in sorted((run / "dagger").glob("iter*.json")): shutil.copy2(f, dst / "dagger" / f.name)
    for f in sorted((DATA / f"s{s}").glob("*.json")): shutil.copy2(f, dst / "rollouts" / f.name)
    for f in sorted((DATA / f"s{s}").glob("iter*.episodes.csv")):
        shutil.copy2(f, dst / "rollouts" / f.name.replace(".episodes.csv", ".rollout_rows.csv"))
    m = json.loads((run / "manifest.json").read_text()); sj = json.loads((run / "selection.json").read_text())
    sel = sj["selected"]; c = sel["candidate"]
    cpu_s = sum(sum(v for k, v in m["data"][str(i)]["dagger"]["timing_s"].items() if k != "note") for i in range(1, 6))
    rows.append({
        "run": f"bc_dagger/s{s}", "arm": "bc_dagger", "seed": s, "purpose": m["purpose"], "sigma": "",
        "ila_commit": m["git"]["ila_commit"][:7], "ila_dirty": m["git"]["ila_dirty"], "total_steps": m["total_steps"],
        "wall_clock_s": round(m["wall_clock_s"], 1), "gpu_hours": round(m["gpu_hours"], 4), "resumes": len(m["resumes"]),
        "selected_checkpoint": Path(c["file"]).name, "selected_global_step": c["global_step"],
        "select_success": sel["success"], "select_n": sel["n"], "outcomes": json.dumps(dict(sorted(sel["outcomes"].items()))),
        "p95_closing_speed_m_s": sel["p95_closing_speed_m_s"], "mean_steps": sel.get("mean_steps", ""),
        "selected_ckpt_sha256": c["sha256"], "n_tunnelled": sel["n_tunnelled"], "n_detectors_disagree": sel["n_detectors_disagree"],
        "cpu_in_wall_s": round(cpu_s, 1), "gpu_hours_net_of_cpu": round((m["wall_clock_s"] - cpu_s) / 3600, 4)})
cols = P4COLS + ["cpu_in_wall_s", "gpu_hours_net_of_cpu"]
with open(OUT / "runs_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(open(OUT / "runs_summary.csv").read())
