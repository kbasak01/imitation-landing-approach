import json, math
from pathlib import Path
import numpy as np
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def lr_ref(step, peak=3e-4, warm=500, total=30000):
    if step < warm: return peak * step / warm
    p = (step - warm) / (total - warm); return peak * 0.5 * (1 + math.cos(math.pi * p))
for a in ["bc", "bc_state", "bc_dart_tune/sigma_0.10"]:
    for s in range(5):
        f = R / a / f"s{s}" / "train_log.jsonl"
        if not f.exists(): continue
        ev = [json.loads(l) for l in f.read_text().splitlines()]
        tr = [e for e in ev if e["event"] == "train"]; ho = [e for e in ev if e["event"] == "heldout"]
        kinds = sorted({e["event"] for e in ev})
        steps = np.array([e["step"] for e in tr]); loss = np.array([e["loss"] for e in tr])
        gm = np.array([e["grad_norm_mean"] for e in tr]); gx = np.array([e["grad_norm_max"] for e in tr])
        lr = np.array([e["lr"] for e in tr]); lrr = np.array([lr_ref(st) for st in steps])
        nonfin = sum(not all(math.isfinite(v) for v in (e["loss"], e["grad_norm_mean"], e["grad_norm_max"], e["loss_last"])) for e in tr)
        # spikes: loss > 3x running median of previous 10 windows
        spikes = [int(steps[i]) for i in range(10, len(loss)) if loss[i] > 3 * np.median(loss[i-10:i])]
        lrerr = float(np.max(np.abs(lr - lrr)) / 3e-4)
        print(f"== {a} s{s} events={kinds} n_train={len(tr)} first={steps[0]} last={steps[-1]} nonfinite={nonfin} spikes={spikes[:8]}{'...' if len(spikes)>8 else ''} n_spikes={len(spikes)} lr_maxrelerr={lrerr:.2e}")
        for q in [1000, 5000, 10000, 15000, 20000, 25000, 30000]:
            i = np.searchsorted(steps, q); i = min(i, len(steps)-1)
            print(f"   step {steps[i]:6d} loss {loss[i]:.3e} gn_mean {gm[i]:.3f} gn_max {gx[i]:.3f} lr {lr[i]:.2e}")
        print(f"   gn_max overall {gx.max():.3f} at {steps[gx.argmax()]}; late(>=20k) gn_max {gx[steps>=20000].max():.3f}")
        for h in ho:
            print(f"   heldout step {h['step']:6d} ema {h['ema']['loss']:.3e} raw {h['raw']['loss']:.3e} ratio_ho/train {h['ema']['loss']/loss[min(np.searchsorted(steps,h['step']),len(loss)-1)]:.2f} ovl {h['overlap_episodes']}")
