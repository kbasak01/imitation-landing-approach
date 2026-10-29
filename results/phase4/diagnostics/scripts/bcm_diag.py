"""Read-only open-loop diagnostics: bc (iteration 0) vs bc_matched iterations 1-5. CPU fp32, eval mode."""
import json, sys, time, importlib.util
from pathlib import Path
import numpy as np, torch
spec = importlib.util.spec_from_file_location("d2", str(Path(__file__).parent / "diag2.py"))
src = (Path(__file__).parent / "diag2.py").read_text().replace("\nmain()", "\n")
d2 = {}; exec(compile(src, "diag2", "exec"), d2)
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, BCWindowDataset
DATA = Path("/home/kbasak/data/ila/demos"); RUNS = d2["RUNS"]
torch.set_num_threads(6)
def ds_from(store, start, n, every):
    with DemoStore(DATA / store) as st:
        names = list(choose_episodes(st, n, False, start=start).names)[::every]
        tab = FrameTable.from_store(st, names)
    ds = BCWindowDataset(tab, 2); v = d2["View"]()
    v.label = tab.label; v.la = tab.proprio[:, 9:12]; v.step = tab.step_of; v.ep = tab.episode_of
    return ds, v
t0 = time.time()
ho, hv = ds_from("ppo_2000.h5", 1950, 50, 2)       # bc: unseen; bc_matched: trained on from it5
dt, dv = ds_from("dart_0.10.h5", 0, 1000, 25)      # neither arm trained on these states
print("frames heldout", len(ho), "dart", len(dt), f"{time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(0); jh = rng.choice(len(ho), 512, replace=False); jd = rng.choice(len(dt), 512, replace=False)
res = {"dart_label_stats": {k: d2["stats"](dv.label, dv)[k] for k in ["frac_norm>=0.98_s/l", "mse_label_to_last_action", "mse_to_label_t-1", "episode_jerk_s/l"]}}
for s in range(5):
    sj = json.loads((RUNS / "bc_matched" / f"s{s}" / "selection.json").read_text())
    selfile = Path(sj["selected"]["candidate"]["file"]).name
    cps = [("it0_bc_final", RUNS / "bc" / f"s{s}" / "ema_last.pt")] + [(f"it{k}_end", RUNS / "bc_matched" / f"s{s}" / "checkpoints" / f"s{k}_step010000.pt") for k in range(1, 6)]
    if selfile != "s5_step010000.pt": cps.append(("it5_5k_selected", RUNS / "bc_matched" / f"s{s}" / "checkpoints" / selfile))
    R = {"selected": selfile}
    for tag, path in cps:
        net = d2["net_from"](path, "ema"); r = {}
        r["J_heldout"] = d2["jac"](net, ho, "bc", jh)["J_last_action"]
        r["J_dart"] = d2["jac"](net, dt, "bc", jd)["J_last_action"]
        full = tag in ("it0_bc_final", "it5_end", "it5_5k_selected") or True
        r["heldout"] = d2["stats"](d2["predict"](net, ho, "bc"), hv)
        if tag in ("it0_bc_final", "it5_end", "it5_5k_selected"):
            r["dart"] = d2["stats"](d2["predict"](net, dt, "bc"), dv)
        R[tag] = r
        print(f"  s{s} {tag} {time.time()-t0:.0f}s diagJ_ho={np.round(np.diag(r['J_heldout']),3)} diagJ_dart={np.round(np.diag(r['J_dart']),3)} ho_mse={r['heldout']['mse']:.3e}" + (f" dart_mse={r['dart']['mse']:.3e}" if 'dart' in r else ""), flush=True)
    res[f"s{s}"] = R
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1])
