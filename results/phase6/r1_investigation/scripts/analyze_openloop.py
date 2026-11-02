"""Bucket open-loop sampled executed-chunk error by steps to the episode end; label-shift probe."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent


def main() -> None:
    r = pickle.load(open(S / "openloop.pkl", "rb"))
    t, T = r["t"], r["T"]
    to_end = T - 1 - t
    buckets = {
        "last7 (T-7..T-1; never R1 anchors)": to_end <= 6,
        "T-15..T-8": (to_end >= 7) & (to_end <= 14),
        "T-23..T-16": (to_end >= 15) & (to_end <= 22),
        "earlier (<= T-24)": to_end >= 23,
        "anchored all (t <= T-8)": to_end >= 7,
    }
    out: dict = {"frames": int(len(t)), "bucket_frames": {k: int(m.sum()) for k, m in buckets.items()}}
    names = sorted({k[0] for k in r if isinstance(k, tuple)})
    seeds = sorted({k[1] for k in r if isinstance(k, tuple)})
    for name in names:
        rec: dict = {}
        errs = {sh: [] for sh in (-2, -1, 0, 1, 2)}
        for s in seeds:
            ch = r[(name, s)][:, 1:9].astype(np.float64)  # executed slice To-1 .. To+6
            for sh in errs:
                errs[sh].append((ch - r["targets"][sh]) ** 2)
        e0 = np.mean(errs[0], axis=0)  # mean over seeds: [N, 8, 3]
        rec["mse_all"] = float(e0.mean())
        for k, m in buckets.items():
            rec[k] = {"mse_exec8": float(e0[m].mean()), "mse_first_step": float(e0[m, 0].mean()),
                      "rms_first_step": float(np.sqrt(e0[m, 0].mean()))}
        rec["shift_probe_mse_anchored"] = {
            str(sh): float(np.mean(errs[sh], axis=0)[buckets["anchored all (t <= T-8)"]].mean())
            for sh in errs}
        out[name] = rec
    (S / "analysis_openloop.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
