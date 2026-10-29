"""Read-only behaviour signatures on the DAgger rollouts (stores + sidecars), per seed and iteration,
with the teacher's demos at the same starting conditions (ppo_2000 1000-1999 = bc_matched's second 1000) beside them.
Proposed = the student's action on every visited state (sidecar); label = teacher at the same state."""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np, h5py, pandas as pd
DAG = Path("/home/kbasak/data/ila/dagger/bc_dagger"); PPO = Path("/home/kbasak/data/ila/demos/ppo_2000.h5")
BETA = [0.5, 0.25, 0.125, 0.0, 0.0]
def sig(prop, lab, ex, offs):
    r = {}; dS, dL, js, jl, lag_t, lag_tm1 = [], [], [], [], [], []
    for k in range(len(offs) - 1):
        a, b = offs[k], offs[k + 1]; p, l = prop[a:b], lab[a:b]
        if b - a < 2: continue
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1); dl = np.linalg.norm(np.diff(l, axis=0), axis=1)
        dS.append(ds); dL.append(dl); js.append(ds.mean()); jl.append(dl.mean())
        lag_t.append(((p[1:] - l[1:]) ** 2).sum(1)); lag_tm1.append(((p[1:] - l[:-1]) ** 2).sum(1))
    dS, dL = np.concatenate(dS), np.concatenate(dL); mov = dL > 0.02
    nP, nL, nE = (np.linalg.norm(x, axis=1) for x in (prop, lab, ex))
    l2 = np.linalg.norm(prop - lab, axis=1)
    return {"frames": int(len(prop)), "l2_mean": float(l2.mean()), "l2_p90": float(np.quantile(l2, 0.9)), "linf_le_0p1": float((np.abs(prop - lab).max(1) <= 0.1).mean()),
            "copycat_frac_dS_lt_0p25dL_given_dL_gt_0p02": float((dS[mov] < 0.25 * dL[mov]).mean()), "n_moving_steps": int(mov.sum()),
            "mean_step_change_student": float(dS.mean()), "mean_step_change_label": float(dL.mean()),
            "episode_jerk_student_vs_label": [float(np.mean(js)), float(np.mean(jl))],
            "lag_mse_to_label_t": float(np.concatenate(lag_t).mean()), "lag_mse_to_label_t_minus_1": float(np.concatenate(lag_tm1).mean()),
            "sat_norm_ge_0p98_student": float((nP >= 0.98).mean()), "sat_norm_ge_0p98_label": float((nL >= 0.98).mean()), "sat_norm_ge_0p98_executed": float((nE >= 0.98).mean()),
            "frac_component_clipped_student": float((np.abs(prop) >= 0.999).any(1).mean())}
res = {}
# teacher reference at the same starting conditions (ppo_2000 1000-1999), by iteration slice
with h5py.File(PPO, "r") as f:
    for i in range(1, 6):
        names = [f"ep_{j:05d}" for j in range(1000 + 200 * (i - 1), 1000 + 200 * i)]
        lab = np.concatenate([f[n]["label"][()] for n in names]).astype(np.float64)
        ex = np.concatenate([f[n]["action_exec"][()] for n in names]).astype(np.float64)
        offs = np.concatenate([[0], np.cumsum([f[n]["label"].shape[0] for n in names])])
        oc = Counter(str(f[n].attrs["outcome"]) for n in names); steps = np.diff(offs)
        s = sig(lab, lab, ex, offs)
        res[f"teacher_demos_iter{i}_slice"] = {"outcomes": dict(oc), "frames": int(offs[-1]), "steps_mean": float(steps.mean()), "steps_max": int(steps.max()),
            "label_step_change_mean": s["mean_step_change_label"], "label_episode_jerk": s["episode_jerk_student_vs_label"][1], "sat_norm_ge_0p98_label": s["sat_norm_ge_0p98_label"]}
    allnames = [f"ep_{j:05d}" for j in range(1000, 2000)]
    res["teacher_demos_1000_1999_frames"] = int(sum(f[n]["label"].shape[0] for n in allnames))
    res["teacher_demos_1000_1999_outcomes"] = dict(Counter(str(f[n].attrs["outcome"]) for n in allnames))
for sd in range(5):
    for i in range(1, 6):
        sc = np.load(DAG / f"s{sd}" / f"iter{i}.sidecar.npz")
        with h5py.File(DAG / f"s{sd}" / f"iter{i}.h5", "r") as f:
            names = sorted(f.keys())
            lab = np.concatenate([f[n]["label"][()] for n in names]).astype(np.float64)
            ex = np.concatenate([f[n]["action_exec"][()] for n in names]).astype(np.float64)
            oc = Counter(str(f[n].attrs["outcome"]) for n in names); steps = np.array([f[n]["label"].shape[0] for n in names])
            clear = int(sum(int(f[n].attrs["clear_frames"]) for n in names)); clearmax = float(max(float(f[n].attrs["clear_frac_max"]) for n in names))
        prop = sc["proposed"].astype(np.float64); offs = sc["offsets"]; te = sc["teacher_exec"]
        assert offs[-1] == len(lab) == len(prop)
        r = {"beta": BETA[i - 1], "outcomes": dict(oc), "steps_mean": float(steps.mean()), "steps_max": int(steps.max()),
             "teacher_executed_fraction": float(te.mean()), "clear_frames": clear, "clear_frac_max": clearmax}
        r.update(sig(prop, lab, ex, offs))
        res[f"s{sd}/iter{i}"] = r
        print(f"s{sd} it{i} b={BETA[i-1]:.3f} oc={dict(oc)} steps {r['steps_mean']:.1f}/{r['steps_max']} l2 {r['l2_mean']:.4f} copycat {r['copycat_frac_dS_lt_0p25dL_given_dL_gt_0p02']:.4f} "
              f"lag t/t-1 {r['lag_mse_to_label_t']:.4f}/{r['lag_mse_to_label_t_minus_1']:.4f} jerk s/l {r['episode_jerk_student_vs_label'][0]:.4f}/{r['episode_jerk_student_vs_label'][1]:.4f} "
              f"sat s/l/x {r['sat_norm_ge_0p98_student']:.4f}/{r['sat_norm_ge_0p98_label']:.4f}/{r['sat_norm_ge_0p98_executed']:.4f} clear {clear}", flush=True)
for k in [k for k in res if k.startswith("teacher")]: print(k, res[k])
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1])
