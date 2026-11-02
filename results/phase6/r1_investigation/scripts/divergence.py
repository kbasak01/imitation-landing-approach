"""Where do closed-loop students leave the demo? Per-step |action - label| and xy distance to demo."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent


def main() -> None:
    d = pickle.load(open(S / "flights_s0-9.pkl", "rb"))
    from ila import p8_adapter
    from ila.data.store import DemoStore
    from ila.obs.split import load_contract
    from ila.policies.registry import REGISTRY
    from ila.train import loop
    from ila.train.config import TrainConfig

    c = load_contract()
    cfg = TrainConfig.load(REGISTRY["dp"].config)
    with DemoStore(loop.store_path(p8_adapter.load_paths()["data_root"], cfg.collection)) as st:
        dobs = [st.privileged_observations(n) for n in d["store_groups"]]
        dlab = [st.labels(n) for n in d["store_groups"]]
    res = {(r["variant"], r["run_seed"]): r for r in d["results"]}
    out = {}
    bins = [(0, 1), (1, 4), (4, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48)]
    for v in ("r1", "dp_ema", "dp_raw"):
        rows = {}
        for lo, hi in bins:
            a_err, p_err, first_act = [], [], []
            for s in range(10):
                for e in range(8):
                    tr = res[(v, s)]["traces"][e]
                    n = min(len(tr["act"]), len(dlab[e]))
                    for t in range(lo, min(hi, n)):
                        a_err.append(np.abs(tr["act"][t] - dlab[e][t]))
                        ps = c.audit_field(tr["obs"][t], "rel_position")
                        pd = c.audit_field(dobs[e][t], "rel_position")
                        p_err.append(np.linalg.norm(ps[:2] - pd[:2]))
            rows[f"t{lo}-{hi - 1}"] = {
                "mean_abs_action_minus_label": np.mean(a_err, 0).round(4).tolist(),
                "mean_xy_dist_to_demo_m": round(float(np.mean(p_err)), 4),
            }
        out[v] = rows
    (S / "divergence.json").write_text(json.dumps(out, indent=1))
    for v, rows in out.items():
        print(v)
        for k, r in rows.items():
            print(f"  {k:8s} |a-label| {r['mean_abs_action_minus_label']}  xy_dist {r['mean_xy_dist_to_demo_m']}")


if __name__ == "__main__":
    main()
