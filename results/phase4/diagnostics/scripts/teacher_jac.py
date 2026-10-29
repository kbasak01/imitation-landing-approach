"""PPO teacher (and bc_state) d action / d last_action on the diagnostician's state sets. CPU, read-only.

Teacher path: ila.p8_adapter.teacher("ppo", 0) -> P8 LearnedPolicy (SB3 on CPU, frozen VecNormalize,
training=False) -> act(obs) = clip(mean action, -1, 1). It is called one observation at a time on the
stored raw 25-d teacher_obs, so the derivative is by central finite difference (eps = 0.05 on the raw
last_action entries obs[21:24], the same raw-unit perturbation applied to the students' proprio[9:12]).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
from ila import p8_adapter
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable
from ila.train.state_dataset import StateFrames
from ila.policies.registry import read_checkpoint, build_net_from_config
DATA = Path("/home/kbasak/data/ila/demos"); RUNS = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
EPS = 0.05
def sets(store, start, n, every):
    with DemoStore(DATA / store) as st:
        names = list(choose_episodes(st, n, False, start=start).names)[::every]
        sf = StateFrames(st, names); tab = FrameTable.from_store(st, names)
    assert np.array_equal(sf.label, tab.label) and len(sf) == len(tab.label), "frame order differs"
    assert np.array_equal(sf.obs[:, 21:24], tab.proprio[:, 9:12]), "last_action differs between teacher_obs and proprio"
    return sf, tab
t0 = time.time()
ho_sf, ho_tab = sets("ppo_2000.h5", 1950, 50, 2); dt_sf, dt_tab = sets("dart_0.10.h5", 0, 1000, 25)
rng = np.random.default_rng(0); jh = rng.choice(len(ho_sf), 512, replace=False); jd = rng.choice(len(dt_sf), 512, replace=False)
print("frames", len(ho_sf), len(dt_sf), flush=True)
pol = p8_adapter.teacher("ppo", 0)
print("teacher", type(pol).__name__, "normalizer training =", getattr(pol.normalizer, "training", None), "norm_reward =", getattr(pol.normalizer, "norm_reward", None), flush=True)
pol.reset(0)
def teacher_J(obs, idx):
    a0 = np.stack([pol.act(obs[i]) for i in idx])
    J = np.zeros((3, 3)); Jabs = np.zeros((3, 3))
    for j in range(3):
        d = []
        for i in idx:
            xp, xm = obs[i].copy(), obs[i].copy(); xp[21 + j] += EPS; xm[21 + j] -= EPS
            d.append((pol.act(xp) - pol.act(xm)) / (2 * EPS))
        d = np.stack(d); J[:, j] = d.mean(0); Jabs[:, j] = np.abs(d).mean(0)
    return a0, J, Jabs
res = {"method": "central finite difference, eps=0.05 raw units on teacher_obs[21:24] (last_action), "
       "through ila.p8_adapter.teacher('ppo',0).act (frozen VecNormalize, training=False, clip to [-1,1]); "
       "J[i,j] = mean over states of d a_i / d last_action_j", "sets": {}}
for nm, sf, idx in [("heldout_ppo_1950_1999_every2", ho_sf, jh), ("dart_0.10_0_999_every25", dt_sf, jd)]:
    for tag, ii in [("sample512", idx), ("all_frames", np.arange(len(sf)))]:
        a0, J, Ja = teacher_J(sf.obs, ii)
        lab_err = float(np.abs(a0 - sf.label[ii]).max())
        sat = float((np.abs(a0) >= 1.0).mean())
        res["sets"][f"{nm}/{tag}"] = {"n": int(len(ii)), "J": np.round(J, 4).tolist(), "diag": np.round(np.diag(J), 4).tolist(),
                                      "trace_over_3": round(float(np.trace(J) / 3), 4), "mean_abs_J": np.round(Ja, 4).tolist(),
                                      "max_abs(act - stored label)": lab_err, "frac_components_at_clip": round(sat, 4)}
        print(nm, tag, "diag", np.round(np.diag(J), 4), "tr/3", round(float(np.trace(J) / 3), 4), "label reproduction max err", lab_err, f"{time.time()-t0:.0f}s", flush=True)
# bc_state on the same sets (final EMA and selected EMA), same eps and raw-unit perturbation
def state_J(net, obs, idx):
    x = torch.from_numpy(obs[idx]); J = np.zeros((3, 3))
    with torch.no_grad():
        for j in range(3):
            xp, xm = x.clone(), x.clone(); xp[:, 21 + j] += EPS; xm[:, 21 + j] -= EPS
            J[:, j] = ((net(xp) - net(xm)).numpy() / (2 * EPS)).mean(0)
    return J
res["bc_state"] = {}
for s in range(5):
    run = RUNS / "bc_state" / f"s{s}"; sj = json.loads((run / "selection.json").read_text())
    out = {"selected_step": sj["selected"]["candidate"]["global_step"]}
    for tag, path in [("final_ema", run / "ema_last.pt"), ("sel_ema", Path(sj["selected"]["candidate"]["file"]))]:
        ck = read_checkpoint(path); net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
        net.load_state_dict(ck["ema"]); net.eval()
        Jh, Jd = state_J(net, ho_sf.obs, jh), state_J(net, dt_sf.obs, jd)
        out[tag] = {"diag_heldout": np.round(np.diag(Jh), 4).tolist(), "diag_dart": np.round(np.diag(Jd), 4).tolist(),
                    "tr3_heldout": round(float(np.trace(Jh) / 3), 4), "tr3_dart": round(float(np.trace(Jd) / 3), 4)}
    res["bc_state"][f"s{s}"] = out; print("bc_state", s, out, flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1], f"{time.time()-t0:.0f}s")
