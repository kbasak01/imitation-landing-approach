"""Read-only open-loop diagnostics of Phase 4 bc / bc_state runs (CPU fp32, eval mode)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, BCWindowDataset
from ila.train.state_dataset import StateFrames
from ila.policies.registry import read_checkpoint, build_net_from_config

RUNS = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
DATA = Path("/home/kbasak/data/ila/demos")
FIELDS = {'attitude': (0, 3), 'body_rates': (3, 3), 'velocity': (6, 3), 'rel_position': (9, 3), 'rel_velocity': (12, 3),
          'deck_normal': (15, 3), 'rel_tilt': (18, 1), 'height': (19, 1), 'time_fraction': (20, 1), 'last_action': (21, 3), 'in_contact': (24, 1)}

class View:  # common view: label, last_action, step_of, episode_of
    pass

def names_for(start, n, every):
    with DemoStore(DATA / "ppo_2000.h5") as st:
        ch = choose_episodes(st, n, False, start=start)
        return list(ch.names)[::every]

def vision_ds(names):
    with DemoStore(DATA / "ppo_2000.h5") as st:
        tab = FrameTable.from_store(st, names)
    ds = BCWindowDataset(tab, 2)
    v = View(); v.label = tab.label; v.la = tab.proprio[:, 9:12]; v.step = tab.step_of; v.ep = tab.episode_of
    return ds, v

def state_ds(names):
    with DemoStore(DATA / "ppo_2000.h5") as st:
        sf = StateFrames(st, names)
    v = View(); v.label = sf.label; v.la = sf.obs[:, 21:24]
    v.ep = np.repeat(np.arange(len(sf.episode_len)), sf.episode_len)
    starts = np.concatenate([[0], np.cumsum(sf.episode_len)[:-1]]); v.step = np.arange(len(sf)) - starts[v.ep]
    return sf, v

def net_from(path, weights="ema"):
    ck = read_checkpoint(path)
    net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
    net.load_state_dict(ck["model" if weights == "raw" else weights]); net.eval(); return net

def predict(net, ds, kind, idx=None):
    n = len(ds); idx = np.arange(n) if idx is None else idx; out = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = ds.batch(idx[s:s+512])
            if kind == "bc": out.append(net(torch.from_numpy(b["images"]), torch.from_numpy(b["proprio"])).numpy())
            else: out.append(net(torch.from_numpy(b["obs"])).numpy())
    return np.concatenate(out)

def stats(pred, v):
    lab, la = v.label, v.la
    prev = np.where(v.step > 0)[0]
    r = {"mse": float(((pred - lab) ** 2).mean()), "mae_norm": float(np.linalg.norm(pred - lab, axis=1).mean())}
    nP, nL = np.linalg.norm(pred, axis=1), np.linalg.norm(lab, axis=1)
    r["norm_mean_s/l"] = [round(float(nP.mean()), 4), round(float(nL.mean()), 4)]
    r["frac_norm>=0.98_s/l"] = [round(float((nP >= 0.98).mean()), 4), round(float((nL >= 0.98).mean()), 4)]
    r["frac_norm>1_s/l"] = [round(float((nP > 1).mean()), 4), round(float((nL > 1).mean()), 4)]
    r["frac_anycomp>=0.98_s/l"] = [round(float((np.abs(pred) >= 0.98).any(1).mean()), 4), round(float((np.abs(lab) >= 0.98).any(1).mean()), 4)]
    r["mean_s"] = np.round(pred.mean(0), 4).tolist(); r["mean_l"] = np.round(lab.mean(0), 4).tolist()
    r["std_s"] = np.round(pred.std(0), 4).tolist(); r["std_l"] = np.round(lab.std(0), 4).tolist()
    r["q01_99_s"] = np.round(np.quantile(pred, [0.01, 0.99], axis=0), 3).tolist(); r["q01_99_l"] = np.round(np.quantile(lab, [0.01, 0.99], axis=0), 3).tolist()
    # histogram distance per component (W1 via sorted samples)
    r["W1_per_comp"] = [round(float(np.mean(np.abs(np.sort(pred[:, i]) - np.sort(lab[:, i])))), 4) for i in range(3)]
    r["mse_to_label_t"] = float(((pred[prev] - lab[prev]) ** 2).mean())
    r["mse_to_label_t-1"] = float(((pred[prev] - lab[prev - 1]) ** 2).mean())
    r["mse_to_last_action"] = float(((pred[prev] - la[prev]) ** 2).mean())
    r["mse_label_to_last_action"] = float(((lab[prev] - la[prev]) ** 2).mean())
    dS = np.linalg.norm(pred[prev] - pred[prev - 1], axis=1); dL = np.linalg.norm(lab[prev] - lab[prev - 1], axis=1)
    mov = dL > 0.02
    r["mean_dS/dL"] = [round(float(dS.mean()), 4), round(float(dL.mean()), 4)]
    r["frac_dS<0.25dL|dL>0.02"] = round(float((dS[mov] < 0.25 * dL[mov]).mean()), 4)
    # per-episode jerk (mean ||diff||), the P8 action_jerk_mean definition, on teacher states
    js, jl = [], []
    for e in np.unique(v.ep):
        m = v.ep == e
        js.append(np.linalg.norm(np.diff(pred[m], axis=0), axis=1).mean()); jl.append(np.linalg.norm(np.diff(lab[m], axis=0), axis=1).mean())
    r["episode_jerk_s/l"] = [round(float(np.mean(js)), 4), round(float(np.mean(jl)), 4)]
    # lag-1 cross-correlation check: corr(pred_t, lab_t) vs corr(pred_t, lab_{t-1})
    r["n_frames"] = int(len(lab))
    return r

def jac(net, ds, kind, idx, eps=0.05):
    b = ds.batch(idx); out = {}
    with torch.no_grad():
        if kind == "bc":
            im = torch.from_numpy(b["images"]); p = torch.from_numpy(b["proprio"])
            J = np.zeros((3, 3))
            for j in range(3):
                pp, pm = p.clone(), p.clone(); pp[:, -1, 9 + j] += eps; pm[:, -1, 9 + j] -= eps
                J[:, j] = ((net(im, pp) - net(im, pm)).numpy() / (2 * eps)).mean(0)
            out["J_last_action"] = np.round(J, 3).tolist()
        else:
            x = torch.from_numpy(b["obs"]); sd = x.std(0).numpy()
            J = np.zeros((3, 25))
            for j in range(25):
                h = max(eps * float(sd[j]), 1e-4)
                xp, xm = x.clone(), x.clone(); xp[:, j] += h; xm[:, j] -= h
                J[:, j] = np.abs((net(xp) - net(xm)).numpy() / (2 * h)).mean(0) * sd[j]  # |da/dx| * std_x
            out["J_last_action"] = None
            Js = np.zeros((3, 3))
            for j in range(3):
                xp, xm = x.clone(), x.clone(); xp[:, 21 + j] += eps; xm[:, 21 + j] -= eps
                Js[:, j] = ((net(xp) - net(xm)).numpy() / (2 * eps)).mean(0)
            out["J_last_action"] = np.round(Js, 3).tolist()
            out["sens_by_field(|da/dx|*std, summed over a)"] = {k: round(float(J[:, o:o+n].sum()), 4) for k, (o, n) in FIELDS.items()}
    return out

def main():
    t0 = time.time(); res = {}
    ho_names = names_for(1950, 50, 2); tr_names = names_for(0, 1000, 25)
    for arm in ["bc_state", "bc"]:
        kind = "bc" if arm == "bc" else "state"
        mk = vision_ds if arm == "bc" else state_ds
        ho, hv = mk(ho_names); tr, tv = mk(tr_names)
        print(arm, "heldout frames", len(ho), "train frames", len(tr), f"{time.time()-t0:.0f}s", flush=True)
        if arm == "bc_state":
            res["teacher_demo_jerk_heldout"] = stats(hv.label, hv)["episode_jerk_s/l"][1]
        for s in range(5):
            run = RUNS / arm / f"s{s}"; sj = json.loads((run / "selection.json").read_text())
            sel = sj["selected"]["candidate"]["global_step"]
            R = {"selected_step": sel}
            for tag, path, w in [("sel_ema", Path(sj["selected"]["candidate"]["file"]), "ema"), ("final_ema", run / "ema_last.pt", "ema"), ("final_raw", run / "checkpoints/s0_step030000.pt", "raw")]:
                if tag == "sel_ema" and sel == 30000: R[tag] = "= final_ema"; continue
                net = net_from(path, w)
                R[tag + "_heldout"] = stats(predict(net, ho, kind), hv)
                if tag != "final_raw": R[tag + "_train"] = stats(predict(net, tr, kind), tv)
                if tag == "final_ema":
                    rng = np.random.default_rng(0)
                    R["jac_heldout"] = jac(net, ho, kind, rng.choice(len(ho), 512, replace=False))
            print(f"  {arm} s{s} done {time.time()-t0:.0f}s", flush=True)
            res[f"{arm}/s{s}"] = R
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1])

main()
