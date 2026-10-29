"""Read-only open-loop diagnostics of bc_dart s0-4 (CPU fp32, eval mode), same metrics as diag2/bcm_diag."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
src = (Path(__file__).parent / "diag2.py").read_text().replace("\nmain()", "\n")
d2 = {}; exec(compile(src, "diag2", "exec"), d2)
torch.set_num_threads(4)
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, BCWindowDataset
from ila.policies.registry import read_checkpoint
DATA = Path("/home/kbasak/data/ila/demos"); RUNS = d2["RUNS"]
def ds_from(store, start, n, every):
    with DemoStore(DATA / store) as st:
        names = list(choose_episodes(st, n, False, start=start).names)[::every]
        tab = FrameTable.from_store(st, names)
    ds = BCWindowDataset(tab, 2); v = d2["View"]()
    v.label = tab.label; v.la = tab.proprio[:, 9:12]; v.step = tab.step_of; v.ep = tab.episode_of
    return ds, v
t0 = time.time()
ho, hv = ds_from("ppo_2000.h5", 1950, 50, 2)
dt, dv = ds_from("dart_0.10.h5", 0, 1000, 25)   # bc_dart TRAIN states
print("frames heldout", len(ho), "dart(train)", len(dt), flush=True)
rng = np.random.default_rng(0); jh = rng.choice(len(ho), 512, replace=False); jd = rng.choice(len(dt), 512, replace=False)
res = {}
# global CPU RNG state inside checkpoints: constant within a run? (unused by training if so)
for s in range(5):
    sts = [read_checkpoint(RUNS / "bc_dart" / f"s{s}" / "checkpoints" / f"s0_step{k:06d}.pt")["rng"]["torch"] for k in (5000, 30000)]
    print(f"s{s} cpu-rng state equal 5k vs 30k: {torch.equal(*sts)}; first 16 bytes {sts[0][:16].tolist()}", flush=True)
for s in range(5):
    run = RUNS / "bc_dart" / f"s{s}"; sj = json.loads((run / "selection.json").read_text())
    sel = sj["selected"]["candidate"]["global_step"]
    cps = [("sel_ema", Path(sj["selected"]["candidate"]["file"]), "ema")]
    if sel != 30000: cps.append(("final_ema", run / "ema_last.pt", "ema"))
    cps.append(("final_raw", run / "checkpoints/s0_step030000.pt", "raw"))
    R = {"selected_step": sel}
    for tag, path, w in cps:
        net = d2["net_from"](path, w); r = {}
        r["J_heldout"] = d2["jac"](net, ho, "bc", jh)["J_last_action"]
        r["J_dart"] = d2["jac"](net, dt, "bc", jd)["J_last_action"]
        r["heldout"] = d2["stats"](d2["predict"](net, ho, "bc"), hv)
        r["dart"] = d2["stats"](d2["predict"](net, dt, "bc"), dv)
        R[tag] = r
        print(f"  s{s} {tag} {time.time()-t0:.0f}s diagJ_ho={np.round(np.diag(r['J_heldout']),3)} diagJ_dart={np.round(np.diag(r['J_dart']),3)} ho_mse={r['heldout']['mse']:.3e} dart_mse={r['dart']['mse']:.3e}", flush=True)
    res[f"s{s}"] = R
res["label_stats_heldout"] = {k: d2["stats"](hv.label, hv)[k] for k in ["frac_norm>=0.98_s/l", "mse_label_to_last_action", "episode_jerk_s/l", "mean_dS/dL"]}
res["label_stats_dart"] = {k: d2["stats"](dv.label, dv)[k] for k in ["frac_norm>=0.98_s/l", "mse_label_to_last_action", "episode_jerk_s/l", "mean_dS/dL"]}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1])
