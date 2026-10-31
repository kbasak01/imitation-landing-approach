"""Copy the Phase 6 run evidence (dp s0-s4, R1_lerobot_dp s0-s2) into results/phase6/ and build
runs_summary.csv.

Read-only over artifacts/; writes only under results/phase6/. Columns follow
results/phase4/runs_summary.csv. R1 also gets every export's parity.json and meta.json (the parity
gate before selection, P6-D3) and the run's model.safetensors SHA-256 per step from its manifest.
"""
import csv, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase6"
P4COLS = next(csv.reader(open(ROOT / "results/phase4/runs_summary.csv")))
rows = []
for arm, seeds in (("dp", range(5)), ("R1_lerobot_dp", range(3))):
    for s in seeds:
        run, dst = ROOT / f"artifacts/runs/{arm}/s{s}", OUT / f"runs/{arm}/s{s}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("manifest.json", "selection.json"):
            shutil.copy2(run / f, dst / f)
        if arm == "R1_lerobot_dp":
            for step_dir in sorted((run / "onnx").glob("step*")):
                d = dst / "onnx" / step_dir.name
                d.mkdir(parents=True, exist_ok=True)
                for f in ("parity.json", "meta.json"):
                    shutil.copy2(step_dir / f, d / f)
        m = json.loads((run / "manifest.json").read_text())
        sel = json.loads((run / "selection.json").read_text())["selected"]
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
            "p95_closing_speed_m_s": sel["p95_closing_speed_m_s"], "mean_steps": sel.get("mean_steps", ""),
            "selected_ckpt_sha256": c["sha256"], "n_tunnelled": sel["n_tunnelled"],
            "n_detectors_disagree": sel["n_detectors_disagree"]})
with open(OUT / "runs_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=P4COLS)
    w.writeheader()
    w.writerows(rows)
print(open(OUT / "runs_summary.csv").read())
