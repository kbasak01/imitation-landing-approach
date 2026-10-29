import json, math, hashlib
from pathlib import Path
import numpy as np
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def lr_ref(step, peak=3e-4, warm=500, total=30000):
    if step < warm: return peak * step / warm
    p = (step - warm) / (total - warm); return peak * 0.5 * (1 + math.cos(math.pi * p))
def load(a, s):
    ev = [json.loads(l) for l in (R / a / f"s{s}" / "train_log.jsonl").read_text().splitlines()]
    return ev, [e for e in ev if e["event"] == "train"], [e for e in ev if e["event"] == "heldout"]
bcfin = {s: (load("bc", s)[1][-1]["loss"], load("bc", s)[2][-1]["ema"]["loss"]) for s in range(5)}
gh = 0; walls = []
for s in range(5):
    d = R / "bc_dart" / f"s{s}"; m = json.loads((d / "manifest.json").read_text()); st = json.loads((d / "status.json").read_text())
    g = m["git"]; c = m["config"]["files"][0]; x = m["data"]["0"]
    gh += m["gpu_hours"]; walls.append(m["wall_clock_s"])
    print(f"#### s{s}: state={m['state']}/{st['state']} purpose={m['purpose']} seed={m['run_seed']} sigma={m['sigma']} steps={m['global_step']}/{m['total_steps']} cfg={c['sha256'][:12]} frozen={c['matches'] and m['config']['all_frozen'] and c['sha256']==c['frozen_sha256']} ila={g['ila_commit'][:8]} dirty={g['ila_dirty']} p8={g['p8_commit'][:8]} inv={len(m['invocations'])} resumes={m['resumes']} init_from={m['init_from']}")
    print(f"   store={[ (sd['collection'], sd['sha256'][:12], sd['matches']) for sd in x['stores']]} kept={x['kept_episodes']} train_frames={x['train_frames']} choice={[(ch['start'], ch['last_kept'], ch['attempted'], ch['kept'], ch['success_filter'], ch['outcomes']) for ch in x['choices']]} ho_overlap={x['heldout_overlap_episodes']}")
    print(f"   wall={m['wall_clock_s']:.0f}s loop={m['loop_s_total']:.0f}s gpu_h={m['gpu_hours']:.3f} rate={m['steps_per_s_loop']:.2f} gpu={m['host']['gpu']} numerics={m['numerics']} ema={m['ema']['kind']}/{m['ema']['decay']} prec={m['precision']} normaliser_std_la={np.round(m['normaliser']['proprio_std'][9:12],3)}")
    ev, tr, ho = load("bc_dart", s)
    steps = np.array([e["step"] for e in tr]); loss = np.array([e["loss"] for e in tr]); gx = np.array([e["grad_norm_max"] for e in tr])
    nonfin = sum(not all(math.isfinite(e[v]) for v in ("loss", "grad_norm_mean", "grad_norm_max", "loss_last")) for e in tr) + sum(not (math.isfinite(h["ema"]["loss"]) and math.isfinite(h["raw"]["loss"])) for h in ho)
    spikes = [int(steps[i]) for i in range(10, len(loss)) if loss[i] > 3 * np.median(loss[i-10:i])]
    lrerr = max(abs(e["lr"] - lr_ref(e["step"] - 1)) / 3e-4 for e in tr)
    ck = [e for e in ev if e["event"] == "checkpoint"]; mc = {c["file"]: c["sha256"] for c in m["checkpoints"]}
    print(f"   log: n_train={len(tr)} n_ho={len(ho)} nonfinite={nonfin} spikes={spikes} lr_relerr={lrerr:.1e} other={sorted({e['event'] for e in ev} - {'train','heldout'})} ckpt_events_match={all(mc.get(e['file'])==e['sha256'] for e in ck)} n={len(ck)}")
    print(f"   train loss: " + " ".join(f"{int(steps[min(np.searchsorted(steps,q),len(steps)-1)])//1000}k:{loss[min(np.searchsorted(steps,q),len(steps)-1)]:.2e}" for q in [1000,5000,10000,15000,20000,25000,30000]) + f" | late median(>=25k) {np.median(loss[steps>=25000]):.2e} | gn_max all {gx.max():.3f}@{steps[gx.argmax()]} late {gx[steps>=20000].max():.3f} | bc final train {bcfin[s][0]:.2e}")
    print(f"   heldout ema(x1e-4): " + " ".join(f"{h['step']//1000}k:{h['ema']['loss']*1e4:.2f}" for h in ho) + f" | raw end {ho[-1]['raw']['loss']*1e4:.2f} ema/raw end {ho[-1]['ema']['loss']/ho[-1]['raw']['loss']:.3f} | min ema {min(h['ema']['loss'] for h in ho)*1e4:.2f}@{ho[int(np.argmin([h['ema']['loss'] for h in ho]))]['step']} | bc ho ema end {bcfin[s][1]*1e4:.2f} | ovl {sorted({h['overlap_episodes'] for h in ho})}")
print(f"\nTotal bc_dart GPU-h {gh:.3f} (P4-D2 projected 3.92); walls {np.round(walls)}")
t = json.loads((R / "bc_dart_tune/sigma_0.10/s0/manifest.json").read_text())
print("tune sigma 0.10 gpu_h", round(t["gpu_hours"], 3))
# log identity tune vs s0: compare deterministic fields only
_, a, ha = load("bc_dart", 0); _, b, hb = load("bc_dart_tune/sigma_0.10", 0)
keys = ["loss", "loss_last", "grad_norm_mean", "grad_norm_max", "lr", "epoch", "step"]
print("train-log det. fields identical:", len(a) == len(b) and all(x[k] == y[k] for x, y in zip(a, b) for k in keys), "| heldout identical:", all(x["ema"] == y["ema"] and x["raw"] == y["raw"] for x, y in zip(ha, hb)))
