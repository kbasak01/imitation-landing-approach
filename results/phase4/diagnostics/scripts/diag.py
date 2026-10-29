"""Read-only diagnostics of bc_dart tuning checkpoints (CPU, fp32, eval mode)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, BCWindowDataset
from ila.policies.registry import load_student, read_checkpoint, build_net_from_config

ROOT = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs/bc_dart_tune")
DATA = Path("/home/kbasak/data/ila/demos")
LA = slice(9, 12)  # last_action in proprio

def load_ds(coll, start, n, every):
    with DemoStore(DATA / f"{coll}.h5") as st:
        ch = choose_episodes(st, n, False, start=start)
        names = list(ch.names)[::every]
        outc = [st.meta(nm).outcome for nm in names]
        tab = FrameTable.from_store(st, names)
    return BCWindowDataset(tab, 2), outc

def predict(net, ds, idx=None):
    net.eval(); out = []
    n = len(ds); idx = np.arange(n) if idx is None else idx
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = ds.batch(idx[s:s+256])
            out.append(net(torch.from_numpy(b["images"]), torch.from_numpy(b["proprio"])).numpy())
    return np.concatenate(out)

def net_from(path, weights):
    ck = read_checkpoint(path)
    net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
    net.load_state_dict(ck["model" if weights == "raw" else weights]); return net

def stats(pred, ds, tag, noise_sigma):
    t = ds.table; lab = t.label; prop = t.proprio
    step = t.step_of; ep = t.episode_of
    has_prev = step > 0
    prev = np.where(has_prev)[0]
    r = {}
    r["mse"] = float(((pred - lab) ** 2).mean())
    nP, nL = np.linalg.norm(pred, axis=1), np.linalg.norm(lab, axis=1)
    r["norm_mean_student"], r["norm_mean_label"] = float(nP.mean()), float(nL.mean())
    r["frac_norm_ge_0.98_student"] = float((nP >= 0.98).mean())
    r["frac_norm_ge_0.98_label"] = float((nL >= 0.98).mean())
    r["frac_norm_gt_1_student"] = float((nP > 1.0).mean())
    r["frac_norm_gt_1_label"] = float((nL > 1.0).mean())
    r["frac_anycomp_ge_0.98_student"] = float((np.abs(pred) >= 0.98).any(1).mean())
    r["frac_anycomp_ge_0.98_label"] = float((np.abs(lab) >= 0.98).any(1).mean())
    r["q_student"] = np.round(np.quantile(pred, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0), 3).tolist()
    r["q_label"] = np.round(np.quantile(lab, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0), 3).tolist()
    r["mean_student"] = np.round(pred.mean(0), 4).tolist(); r["mean_label"] = np.round(lab.mean(0), 4).tolist()
    r["std_student"] = np.round(pred.std(0), 4).tolist(); r["std_label"] = np.round(lab.std(0), 4).tolist()
    # copycat: excess dependence on last_action, using injected noise n = last_action[t] - label[t-1]
    la = prop[:, LA]
    err = pred - lab
    nz = la[prev] - lab[prev - 1]
    e = err[prev]
    beta = [float(np.cov(e[:, i], nz[:, i])[0, 1] / max(np.var(nz[:, i]), 1e-12)) for i in range(3)]
    r["beta_err_on_injected_noise"] = np.round(beta, 4).tolist()
    r["injected_noise_std"] = np.round(nz.std(0), 4).tolist()
    # lag test (on step-pairs within episode)
    r["mse_to_label_t"] = float(((pred[prev] - lab[prev]) ** 2).mean())
    r["mse_to_label_t-1"] = float(((pred[prev] - lab[prev - 1]) ** 2).mean())
    r["mse_to_last_action"] = float(((pred[prev] - la[prev]) ** 2).mean())
    r["mse_label_to_last_action"] = float(((lab[prev] - la[prev]) ** 2).mean())
    dS = np.linalg.norm(pred[prev] - pred[prev - 1], axis=1)
    dL = np.linalg.norm(lab[prev] - lab[prev - 1], axis=1)
    mov = dL > 0.02
    r["mean_dS"], r["mean_dL"] = float(dS.mean()), float(dL.mean())
    r["frac_dS_lt_0.25dL_where_dL>0.02"] = float((dS[mov] < 0.25 * dL[mov]).mean()) if mov.any() else None
    r["n_frames"] = int(len(lab))
    return r

def jac_last_action(net, ds, idx, eps=0.05):
    """Mean |d a / d last_action[t]| (3x3) by central difference on the current frame only."""
    b = ds.batch(idx)
    im = torch.from_numpy(b["images"]); p = torch.from_numpy(b["proprio"]).clone()
    J = np.zeros((3, 3))
    with torch.no_grad():
        for j in range(3):
            pp, pm = p.clone(), p.clone()
            pp[:, -1, 9 + j] += eps; pm[:, -1, 9 + j] -= eps
            d = (net(im, pp) - net(im, pm)).numpy() / (2 * eps)
            J[:, j] = d.mean(0)
    return np.round(J, 3).tolist()

def main():
    res = {}
    t0 = time.time()
    ho, ho_out = load_ds("ppo_2000", 1950, 50, 2)   # 25 held-out episodes
    print("heldout frames", len(ho), "outcomes", {o: ho_out.count(o) for o in set(ho_out)}, flush=True)
    for s in ["0.05", "0.10", "0.20"]:
        run = ROOT / f"sigma_{s}" / "s0"
        tr, tr_out = load_ds(f"dart_{s}", 0, 1000, 25)   # 40 train episodes
        print(f"sigma {s}: train frames {len(tr)} outcomes", {o: tr_out.count(o) for o in set(tr_out)}, flush=True)
        R = {"train_outcomes": {o: tr_out.count(o) for o in set(tr_out)}}
        # final EMA (ema_last) and final raw
        for w, path in [("ema", run / "ema_last.pt"), ("raw", run / "checkpoints/s0_step030000.pt")]:
            net = net_from(path, w)
            R[f"final_{w}_train"] = stats(predict(net, tr), tr, "train", float(s))
            R[f"final_{w}_heldout"] = stats(predict(net, ho), ho, "heldout", 0.0)
            if w == "ema":
                rng = np.random.default_rng(0)
                R["jac_last_action_train"] = jac_last_action(net, tr, rng.choice(len(tr), 512, replace=False))
                R["jac_last_action_heldout"] = jac_last_action(net, ho, rng.choice(len(ho), 512, replace=False))
            print(f"  final {w} done {time.time()-t0:.0f}s", flush=True)
        # every checkpoint's EMA on a subsample
        rng = np.random.default_rng(1)
        sub_tr = np.sort(rng.choice(len(tr), 800, replace=False)); sub_ho = np.sort(rng.choice(len(ho), 800, replace=False))
        per = {}
        for st in range(5000, 30001, 5000):
            net = net_from(run / f"checkpoints/s0_step{st:06d}.pt", "ema")
            ptr, pho = predict(net, tr, sub_tr), predict(net, ho, sub_ho)
            per[st] = {"mse_train": float(((ptr - tr.table.label[sub_tr]) ** 2).mean()),
                       "mse_heldout": float(((pho - ho.table.label[sub_ho]) ** 2).mean()),
                       "frac_norm_ge_0.98_train": float((np.linalg.norm(ptr, axis=1) >= 0.98).mean()),
                       "frac_norm_ge_0.98_heldout": float((np.linalg.norm(pho, axis=1) >= 0.98).mean())}
        R["per_ckpt_ema"] = per
        # per-outcome error on train (failing DART episodes)
        oc = np.array(tr_out)[tr.table.episode_of]
        net = net_from(run / "ema_last.pt", "ema"); p = predict(net, tr)
        R["mse_by_outcome_train"] = {o: [float(((p[oc == o] - tr.table.label[oc == o]) ** 2).mean()), int((oc == o).sum())] for o in set(tr_out)}
        print(f"  ckpts done {time.time()-t0:.0f}s", flush=True)
        res[s] = R
    out = Path(sys.argv[1]); out.write_text(json.dumps(res, indent=1))
    print("wrote", out)

main()
