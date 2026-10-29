"""Teacher agreement on bc_dagger rollouts, split by who executed the previous step (results-skeptic S1).

Read-only over ${data_root}/dagger/bc_dagger/s<n>/iter<i>.{sidecar.npz,h5}; writes
results/phase5/agreement_by_previous_executor.csv. Agreement = fraction of steps with
max|a_student_proposed - label| <= 0.1 normalised units (0.15 m/s model scale, 0.75 m/s full
scale, lambda = 1/25). Step 0 of each episode has no previous step and is its own group (t0).
"""
import csv
from pathlib import Path

import h5py
import numpy as np

DATA = Path("/home/kbasak/data/ila/dagger/bc_dagger")
OUT = Path(__file__).resolve().parents[1] / "agreement_by_previous_executor.csv"
rows = []
for s in range(5):
    for i in range(1, 6):
        z = np.load(DATA / f"s{s}/iter{i}.sidecar.npz")
        off, prop, te = z["offsets"], z["proposed"], z["teacher_exec"].astype(bool)
        with h5py.File(DATA / f"s{s}/iter{i}.h5", "r") as f:
            lab = np.concatenate([f[k]["label"][()] for k in sorted(f.keys())])
        assert len(lab) == len(prop)
        ok = np.abs(prop.astype(np.float64) - lab).max(1) <= 0.1
        prev_te, first = np.zeros_like(te), np.zeros_like(te)
        for a, b in zip(off[:-1], off[1:], strict=True):
            prev_te[a + 1 : b] = te[a : b - 1]
            first[a] = True
        groups = {"all": np.ones_like(ok), "prev_teacher": prev_te & ~first, "prev_student": ~prev_te & ~first, "t0": first}
        row = {"seed": s, "iteration": i}
        for g, m in groups.items():
            row[f"agree_{g}"] = round(float(ok[m].mean()), 4) if m.any() else ""
            row[f"n_{g}"] = int(m.sum())
        rows.append(row)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
