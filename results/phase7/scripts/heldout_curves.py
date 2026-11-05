"""Extract held-out and training-loss curves for the Phase 7 runs and the core `bc` reference into
results/phase7/diagnostics/heldout_curves.json (P7-D3 traceability).

Reads artifacts/runs/<arm>/s<n>/train_log.jsonl (read-only). For each run it records every
`heldout` event (step, EMA and raw loss, and per-noise-bucket losses where the arm logs them) and
the last `train` event's loss. It also records summaries:
- the step and value of the EMA held-out minimum;
- the final EMA held-out loss and its ratio to the minimum;
- the EMA held-out loss at the selected step (from selection.json).

Loss units are the arm's training loss on normalised targets:
- BC: MSE on actions;
- DP: epsilon-MSE;
- modular: Huber on the 11-d estimate.
Held-out data for every run is `ppo_2000` stream indices 1950-1999 (`train` pool).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase7/diagnostics/heldout_curves.json"
RUNS = [("bc", 5), ("modular", 5)] + [
    (a, 3) for a in ("bc_mix", "bc_pid", "bc_n100", "bc_n300", "bc_minus_last_action",
                     "bc_plus_clock", "dp_mix", "dp_pid", "dp_n100", "dp_n300",
                     "dp_minus_last_action")]


def scalar(v):
    return v["loss"] if isinstance(v, dict) else v


res = {}
for arm, n in RUNS:
    for s in range(n):
        d = ROOT / f"artifacts/runs/{arm}/s{s}"
        held, last_train = [], None
        for line in open(d / "train_log.jsonl"):
            e = json.loads(line)
            if e.get("event") == "heldout":
                held.append({"step": e["step"], "ema": e["ema"], "raw": e.get("raw")})
            elif e.get("event") == "train":
                last_train = {"step": e.get("step"), "loss": e.get("loss")}
        sel_step = json.loads((d / "selection.json").read_text())["selected"]["candidate"][
            "global_step"]
        ema = [(h["step"], scalar(h["ema"])) for h in held]
        mstep, mval = min(ema, key=lambda t: t[1])
        at_sel = [v for st, v in ema if st == sel_step]
        res[f"{arm}/s{s}"] = {
            "summary": {"ema_min_step": mstep, "ema_min": mval, "ema_final": ema[-1][1],
                        "final_over_min": ema[-1][1] / mval, "selected_step": sel_step,
                        "ema_at_selected": at_sel[0] if at_sel else None,
                        "last_train": last_train},
            "heldout": held}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "runs": res}) + "\n")
for k, v in res.items():
    s = v["summary"]
    print(f"{k:26s} min {s['ema_min']:.3g}@{s['ema_min_step']} final {s['ema_final']:.3g} "
          f"x{s['final_over_min']:.2f} sel@{s['selected_step']} {s['ema_at_selected']} "
          f"train {s['last_train']['loss'] if s['last_train'] else None}")
