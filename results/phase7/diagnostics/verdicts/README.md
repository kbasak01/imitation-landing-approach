# Phase 7 diagnostician working files (read-only reviews, 2026-11-03/04)

These are the `training-run-diagnostician`'s scripts and small outputs for the `dp_*` ablation
verdicts in P7-D3. They are kept as provenance. They read `artifacts/runs/` and session-scratch
caches, so they do not re-run from a fresh clone.

- The numbers P7-D3 cites from the committed runs are recomputed from committed files alone by
  `results/phase7/scripts/ablation_summary.py`, which writes
  `results/phase7/diagnostics/ablation_summary.json`.
- What only these files hold:
  - open-loop held-out diagnostics (`*/diag_out/`);
  - the open-loop copycat replay (`dp_minus_last_action/ol_out/ol.json`);
  - recomputed checkpoint SHA-256s (`*sha*.txt`).
- All numbers are in simulation, model scale (λ = 1/25), and on `select` or `train`-pool held-out
  demos. No frozen list was read.
