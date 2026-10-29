import json, math, subprocess
from pathlib import Path
import numpy as np
ROOT = Path("/home/kbasak/projects_02/imitation-landing-approach")
R = ROOT / "artifacts/runs"
FROZEN = "f956e182d1d9e72f1c75010dce743428ee629639de13abdca75374c08ced8de5"
DSHA = json.loads(Path("/home/kbasak/data/ila/demos/ppo_2000.manifest.json").read_text())
print("ppo_2000 manifest keys", list(DSHA)[:12])
def lr_ref(k, peak=3e-4, warm=500, total=10000):
    if k < warm: return peak * k / warm
    p = (k - warm) / (total - warm); return peak * 0.5 * (1 + math.cos(math.pi * p))
bc_final = {}
for s in range(5):
    ev = [json.loads(l) for l in (R / "bc" / f"s{s}" / "train_log.jsonl").read_text().splitlines()]
    tr = [e for e in ev if e["event"] == "train"]; ho = [e for e in ev if e["event"] == "heldout"]
    bc_final[s] = (tr[-1]["loss"], ho[-1]["ema"]["loss"], ho[-1]["raw"]["loss"])
gh = 0
for s in range(5):
    d = R / "bc_matched" / f"s{s}"; m = json.loads((d / "manifest.json").read_text())
    st = json.loads((d / "status.json").read_text())
    g = m["git"]; c = m["config"]["files"][0]
    print(f"\n######## s{s}: state={m['state']} status={st['state']} purpose={m['purpose']} seed={m['run_seed']} kind={m['kind']} total_steps={m['total_steps']} global_step={m['global_step']}")
    print(f"  config sha={c['sha256'][:12]} frozen_match={c['sha256']==FROZEN and c['matches'] and m['config']['all_frozen']}; ila={g['ila_commit'][:8]} dirty={g['ila_dirty']} p8={g['p8_commit'][:8]}; invocations={len(m['invocations'])} resumes={m['resumes']}")
    print(f"  init_from={m['init_from'].split('runs/')[1]} sha={m['init_from_sha256'][:12]} iteration0={ {k:v for k,v in m['iteration0'].items() if k!='file'} }")
    print(f"  wall_clock_s={m['wall_clock_s']:.0f} loop_s={m['loop_s_total']:.0f} gpu_hours={m['gpu_hours']:.3f} steps/s={m['steps_per_s_loop']:.2f} gpu={m['host']['gpu']} numerics={m['numerics']}")
    gh += m["gpu_hours"]
    print(f"  ema={m['ema']} optimizer={ {k:m['optimizer'][k] for k in ['lr','grad_clip','schedule']} } stages={[ (x['index'],x['steps'],x['warmup']) for x in m['stages']]} normaliser_in_manifest={'normaliser' in m}")
    prev_frames = None
    for k in sorted(m["data"], key=int):
        x = m["data"][k]
        ch = x["choices"]; add = ch[-1]
        okstore = all(sd["matches"] and sd["sha256"] == sd["expected_sha256"] for sd in x["stores"])
        dsha = {sd["sha256"][:12] for sd in x["stores"]}
        expected_first = 1000 + 200 * (int(k) - 1)
        print(f"  it{k}: labelled={x['labelled_episodes']}/{x['labelled_episodes_total']} kept={x['kept_episodes']} train_frames={x['train_frames']} (+{x['train_frames']-(prev_frames or 0) if prev_frames else '-'}) added=[{add['start']}..{add['last_kept']}] attempted={add['attempted']} kept={add['kept']} outcomes={add['outcomes']} expected_start={expected_first} ok={add['start']==expected_first and add['attempted']==200} n_choices={len(ch)} base={ch[0]['attempted']},{ch[0]['kept']},{ch[0]['outcomes']} ho_overlap={x['heldout_overlap_episodes']} store_ok={okstore} {dsha}")
        prev_frames = x["train_frames"]
    # curves
    ev = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()]
    tr = [e for e in ev if e["event"] == "train"]; ho = [e for e in ev if e["event"] == "heldout"]
    nonfin = sum(not all(math.isfinite(e[v]) for v in ("loss", "grad_norm_mean", "grad_norm_max", "loss_last")) for e in tr)
    nonfin += sum(not (math.isfinite(h["ema"]["loss"]) and math.isfinite(h["raw"]["loss"])) for h in ho)
    lrerr = max(abs(e["lr"] - lr_ref(e["stage_step"] - 1)) / 3e-4 for e in tr)
    others = sorted({e["event"] for e in ev} - {"train", "heldout", "stage", "checkpoint"})
    print(f"  log: n_train={len(tr)} n_heldout={len(ho)} nonfinite={nonfin} lr_maxrelerr(vs step-1)={lrerr:.2e} other_events={others}; bc final train {bc_final[s][0]:.3e} ho ema {bc_final[s][1]:.3e}")
    for k in range(1, 6):
        t = [e for e in tr if e["stage"] == k]; h = [e for e in ho if e["stage"] == k]
        L = np.array([e["loss"] for e in t]); G = np.array([e["grad_norm_max"] for e in t]); Gm = np.array([e["grad_norm_mean"] for e in t])
        spikes = [t[i]["stage_step"] for i in range(10, len(L)) if L[i] > 3 * np.median(L[i-10:i])]
        # warm-up window: first 1000 stage steps vs steady (last 3000)
        w = L[:10]; late = L[-30:]
        print(f"   it{k}: loss first100={L[0]:.3e} max(0-1k)={w.max():.3e}@{t[int(w.argmax())]['stage_step']} end={L[-1]:.3e} late_med={np.median(late):.3e} | gn_max warm={G[:10].max():.4f} overall={G.max():.4f}@{t[int(G.argmax())]['stage_step']} late={G[-30:].max():.4f} gn_mean first={Gm[0]:.4f} end={Gm[-1]:.4f} | spikes={spikes}")
        print(f"        heldout ema: " + " ".join(f"{e['ema']['loss']*1e4:.3f}" for e in h) + f"  (x1e-4); raw end {h[-1]['raw']['loss']*1e4:.3f}; ema/raw end {h[-1]['ema']['loss']/h[-1]['raw']['loss']:.4f}; raw max {max(e['raw']['loss'] for e in h)*1e4:.3f}@{h[int(np.argmax([e['raw']['loss'] for e in h]))]['step']}; ovl {sorted({e['overlap_episodes'] for e in h})}; ho/train end {h[-1]['ema']['loss']/L[-1]:.2f}")
    ck = [e for e in ev if e["event"] == "checkpoint"]
    mc = {c["file"]: c["sha256"] for c in m["checkpoints"]}
    print(f"  checkpoint events={len(ck)} match manifest={all(mc.get(e['file'])==e['sha256'] for e in ck)} files={[c['file'] for c in m['checkpoints']]}")
print(f"\nTotal bc_matched GPU-h {gh:.3f} vs P4-D2 projection 6.53")
