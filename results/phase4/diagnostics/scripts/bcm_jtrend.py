import json, sys, time
from pathlib import Path
import numpy as np, torch
src = (Path(__file__).parent / "bcm_diag.py").read_text().split("t0 = time.time()")[0]
g = {"__file__": __file__}; exec(compile(src, "bcm_diag_head", "exec"), g)
torch.set_num_threads(6)
ho, hv = g["ds_from"]("ppo_2000.h5", 1950, 50, 2); dt, dv = g["ds_from"]("dart_0.10.h5", 0, 1000, 25)
rng = np.random.default_rng(0); jh = rng.choice(len(ho), 512, replace=False); jd = rng.choice(len(dt), 512, replace=False)
d2 = g["d2"]; RUNS = g["RUNS"]
for s in [0, 2]:
    for k in range(5000, 30001, 5000):
        net = d2["net_from"](RUNS / "bc" / f"s{s}" / "checkpoints" / f"s0_step{k:06d}.pt", "ema")
        Jh = d2["jac"](net, ho, "bc", jh)["J_last_action"]; Jd = d2["jac"](net, dt, "bc", jd)["J_last_action"]
        print(f"bc s{s} step {k}: trJ/3 heldout {np.trace(Jh)/3:.3f} dart {np.trace(Jd)/3:.3f} diag_ho {np.round(np.diag(Jh),3)}", flush=True)
