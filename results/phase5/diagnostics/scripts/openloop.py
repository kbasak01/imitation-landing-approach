"""Read-only open-loop diagnostics, bc_dagger vs bc_matched, per end-of-iteration EMA (eval mode).
State sets:
  heldout  = ppo_2000 1950-1999, every 2nd episode (P4-D5's set; teacher states; bc_matched it5 trained on them)
  sibling  = bc_dagger s((s+1)%5) iteration-5 rollouts (beta = 0, a sibling DAgger student's own states);
             never in any training set of seed s (either arm): a fixed, like-for-like off-teacher state set.
Metrics: MSE and mean L2 to the teacher label, fraction ||.||_inf <= 0.1, norm-cap saturation (||a|| >= 0.98)
student vs label, copycat (FD d a / d last_action, eps 0.05 raw units, 512 frames rng 0 = P4-D5's method;
fraction ||a_t - a_{t-1}|| < 0.25 ||label_t - label_{t-1}|| where the label moves > 0.02; MSE to last_action).
Device: CUDA fp32, TF32 off (diagnostic only; selection and rollouts use the CPU path)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, BCWindowDataset
from ila.policies.registry import read_checkpoint, build_net_from_config
RUNS = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
DATA = Path("/home/kbasak/data/ila"); DEV = torch.device("cuda")
class V: pass
def dataset(store, names):
    with DemoStore(store) as st:
        tab = FrameTable.from_store(st, names)
    ds = BCWindowDataset(tab, 2); v = V()
    v.label = tab.label.astype(np.float64); v.la = tab.proprio[:, 9:12].astype(np.float64); v.step = np.asarray(tab.step_of); v.ep = np.asarray(tab.episode_of)
    return ds, v
def net_from(path):
    ck = read_checkpoint(path)
    net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
    net.load_state_dict(ck["ema"]); net.eval(); return net.to(DEV)
def predict(net, ds):
    out = []
    with torch.no_grad():
        for s in range(0, len(ds), 1024):
            b = ds.batch(np.arange(s, min(s + 1024, len(ds))))
            out.append(net(torch.from_numpy(b["images"]).to(DEV), torch.from_numpy(b["proprio"]).to(DEV)).double().cpu().numpy())
    return np.concatenate(out)
def jac(net, ds, idx, eps=0.05):
    b = ds.batch(idx); im = torch.from_numpy(b["images"]).to(DEV); p = torch.from_numpy(b["proprio"]).to(DEV); J = np.zeros((3, 3))
    with torch.no_grad():
        for j in range(3):
            pp, pm = p.clone(), p.clone(); pp[:, -1, 9 + j] += eps; pm[:, -1, 9 + j] -= eps
            J[:, j] = ((net(im, pp) - net(im, pm)).double().cpu().numpy() / (2 * eps)).mean(0)
    return np.round(np.diag(J), 4).tolist()
def stats(pred, v):
    lab = v.label; d = pred - lab; l2 = np.linalg.norm(d, axis=1)
    prev = np.where(v.step > 0)[0]
    dS = np.linalg.norm(pred[prev] - pred[prev - 1], axis=1); dL = np.linalg.norm(lab[prev] - lab[prev - 1], axis=1); mov = dL > 0.02
    nP, nL = np.linalg.norm(pred, axis=1), np.linalg.norm(lab, axis=1)
    return {"mse": float((d ** 2).mean()), "l2_mean": float(l2.mean()), "l2_p90": float(np.quantile(l2, 0.9)), "linf_le_0p1": float((np.abs(d).max(1) <= 0.1).mean()),
            "sat_norm_ge_0p98_student": float((nP >= 0.98).mean()), "sat_norm_ge_0p98_label": float((nL >= 0.98).mean()),
            "mse_to_last_action": float(((pred[prev] - v.la[prev]) ** 2).mean()), "mse_label_to_last_action": float(((lab[prev] - v.la[prev]) ** 2).mean()),
            "frac_dS_lt_0p25dL_given_dL_gt_0p02": float((dS[mov] < 0.25 * dL[mov]).mean()), "mean_dS": float(dS.mean()), "mean_dL": float(dL.mean()),
            "n_frames": int(len(lab))}
t0 = time.time()
with DemoStore(DATA / "demos/ppo_2000.h5") as st:
    ho_names = list(choose_episodes(st, 50, False, start=1950).names)[::2]
HO = dataset(DATA / "demos/ppo_2000.h5", ho_names)
rng = np.random.default_rng(0); jh = rng.choice(len(HO[0]), 512, replace=False)
res = {"heldout_frames": len(HO[0])}
for s in range(5):
    sib = (s + 1) % 5; p = DATA / f"dagger/bc_dagger/s{sib}/iter5.h5"
    with DemoStore(p) as st: names = sorted(st.episodes())
    SB = dataset(p, names); js = np.random.default_rng(0).choice(len(SB[0]), 512, replace=False)
    sj = {a: Path(json.loads((RUNS / a / f"s{s}" / "selection.json").read_text())["selected"]["candidate"]["file"]).name for a in ("bc_dagger", "bc_matched")}
    cps = [("it0", RUNS / "bc" / f"s{s}" / "checkpoints" / "s0_step030000.pt")]
    for a in ("bc_dagger", "bc_matched"):
        cps += [(f"{a}:it{k}", RUNS / a / f"s{s}" / "checkpoints" / f"s{k}_step010000.pt") for k in range(1, 6)]
        if sj[a] != "s5_step010000.pt": cps.append((f"{a}:selected_it5_5k", RUNS / a / f"s{s}" / "checkpoints" / sj[a]))
    R = {"sibling_state_source": f"bc_dagger/s{sib}/iter5", "sibling_frames": len(SB[0]), "selected": sj}
    for tag, path in cps:
        net = net_from(path)
        R[tag] = {"heldout": stats(predict(net, HO[0]), HO[1]), "sibling": stats(predict(net, SB[0]), SB[1]),
                  "J_diag_heldout": jac(net, HO[0], jh), "J_diag_sibling": jac(net, SB[0], js)}
        x = R[tag]; print(f"s{s} {tag:28s} ho mse {x['heldout']['mse']:.3e} l2 {x['heldout']['l2_mean']:.4f} | sib mse {x['sibling']['mse']:.3e} l2 {x['sibling']['l2_mean']:.4f} linf {x['sibling']['linf_le_0p1']:.3f} | J ho {x['J_diag_heldout']} sib {x['J_diag_sibling']} | sat sib {x['sibling']['sat_norm_ge_0p98_student']:.4f}/{x['sibling']['sat_norm_ge_0p98_label']:.4f} {time.time()-t0:.0f}s", flush=True)
        del net; torch.cuda.empty_cache()
    res[f"s{s}"] = R
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print("wrote", sys.argv[1])
