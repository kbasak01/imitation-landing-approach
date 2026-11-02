"""CPU diagnostic: R1 eps-MSE by k-bucket on held-out (ppo_2000 1950-1999) vs train50 (0-49), eval mode."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
S = Path(sys.argv[1]); run = Path(sys.argv[2]); steps = sys.argv[3].split(',')
NOBS, H = 2, 16
KS = [3, 15, 28, 40, 53, 65, 78, 90]   # 2 per bucket [0,25),[25,50),[50,75),[75,100)
def build(npz, anchors_drop=7):
    d = np.load(S / f'{npz}.npz'); ep = d['ep']
    starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]; ends = np.r_[starts[1:], len(ep)]
    oi, ai, anc_ok = [], [], []
    for s, e in zip(starts, ends):
        T = e - s
        for t in range(T):
            oi.append([s + min(max(t + j, 0), T - 1) for j in range(1 - NOBS, 1)])
            ai.append([s + min(max(t + j, 0), T - 1) for j in range(1 - NOBS, 1 - NOBS + H)])
            anc_ok.append(t <= T - 1 - anchors_drop)
    return d, np.array(oi), np.array(ai), np.array(anc_ok)
sets = {n: build(n) for n in ('heldout', 'train50')}
results = {}
for st in steps:
    t0 = time.time()
    pm = run / 'checkpoints' / st / 'pretrained_model'
    pol = DiffusionPolicy.from_pretrained(str(pm)); pol.config.device = 'cpu'; pol.to('cpu').eval()
    pre = PolicyProcessorPipeline.from_pretrained(str(pm), config_filename='policy_preprocessor.json',
                                                  overrides={'device_processor': {'device': 'cpu'}})
    dm = pol.diffusion; sched = dm.noise_scheduler
    res = {}
    for name, (d, oi, ai, ok) in sets.items():
        imgs = torch.from_numpy(d['images']).permute(0, 3, 1, 2).float() / 255.0  # [N,3,96,96]
        prop = torch.from_numpy(d['proprio']).float(); lab = torch.from_numpy(d['label']).float()
        # normalise per frame through LeRobot's own pipeline (image MEAN_STD, state/action MIN_MAX)
        normed = pre({'observation.image': imgs, 'observation.state': prop, 'action': lab})
        imn, stn, acn = normed['observation.image'], normed['observation.state'], normed['action']
        with torch.no_grad():
            feats = torch.cat([dm.rgb_encoder[0](imn[i:i + 256]) for i in range(0, len(imn), 256)])  # eval: centre crop
            gc = torch.cat([stn[oi], feats[oi]], dim=-1).flatten(1)  # [B, 2*(13+64)]
            traj = acn[ai]  # [B,16,3]
            g = torch.Generator().manual_seed(12345)
            eps_all = torch.randn((len(KS),) + traj.shape, generator=g)
            per = np.zeros((len(KS), len(traj)))
            for j, k in enumerate(KS):
                ts = torch.full((len(traj),), k, dtype=torch.long)
                x = sched.add_noise(traj, eps_all[j], ts)
                for i in range(0, len(traj), 512):
                    pr = dm.unet(x[i:i + 512], ts[i:i + 512], global_cond=gc[i:i + 512])
                    per[j, i:i + 512] = ((pr - eps_all[j][i:i + 512]) ** 2).mean(dim=(1, 2)).numpy()
        r = {}
        for lab_, mask in (('anchors', ok), ('all', np.ones_like(ok))):
            p = per[:, mask]
            r[lab_] = {'loss': float(p.mean()), **{f'k{b*25}_{b*25+24}': float(p[2*b:2*b+2].mean()) for b in range(4)},
                       'n': int(mask.sum())}
        res[name] = r
    results[st] = res
    h, tr = res['heldout']['anchors'], res['train50']['anchors']
    print(st, f"held {h['loss']:.4f} [{h['k0_24']:.4f} {h['k25_49']:.4f} {h['k50_74']:.4f} {h['k75_99']:.4f}]",
          f"train50 {tr['loss']:.4f} [{tr['k0_24']:.4f} {tr['k25_49']:.4f} {tr['k50_74']:.4f} {tr['k75_99']:.4f}]",
          f"held_all {res['heldout']['all']['loss']:.4f}  {time.time()-t0:.0f}s", flush=True)
    Path(sys.argv[4]).write_text(json.dumps(results, indent=1))
