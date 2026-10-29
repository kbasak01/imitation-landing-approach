"""Read-only training-curve audit: bc_dagger s0-s4 per stage, with bc_matched and bc (iteration 0) beside it.
Checks non-finite values, LR schedule (diffusers cosine, 500 warm-up, 10 k per stage, peak 3e-4),
loss spikes (> 3x median of the previous 10 logged windows), grad norms, held-out EMA vs raw.
"""
import json, math, sys
from pathlib import Path
import numpy as np
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def lr_ref(s, peak=3e-4, warm=500, total=10000):
    if s < warm: return peak * s / warm
    p = (s - warm) / (total - warm); return peak * 0.5 * (1 + math.cos(math.pi * p))
def curves(run, staged=True):
    ev = [json.loads(l) for l in (run / "train_log.jsonl").read_text().splitlines()]
    tr = [e for e in ev if e["event"] == "train"]; ho = [e for e in ev if e["event"] == "heldout"]
    out = {"n_train_points": len(tr), "n_heldout_points": len(ho), "events": sorted({e["event"] for e in ev})}
    nonfin = sum(not all(math.isfinite(e[k]) for k in ("loss", "loss_last", "grad_norm_mean", "grad_norm_max", "lr")) for e in tr)
    nonfin += sum(not (math.isfinite(h["ema"]["loss"]) and math.isfinite(h["raw"]["loss"])) for h in ho)
    out["nonfinite"] = nonfin
    stages = sorted({e.get("stage", 0) for e in tr}); out["stages"] = {}
    errs = {"s": [], "s-1": []}
    for st in stages:
        t = [e for e in tr if e.get("stage", 0) == st]; h = [e for e in ho if e.get("stage", 0) == st]
        ss = np.array([e.get("stage_step", e["step"]) for e in t]); loss = np.array([e["loss"] for e in t])
        lr = np.array([e["lr"] for e in t]); gm = np.array([e["grad_norm_max"] for e in t]); gmean = np.array([e["grad_norm_mean"] for e in t])
        total = 10000 if staged else 30000
        for k, off in (("s", 0), ("s-1", 1)):
            errs[k].append(float(np.max(np.abs(lr - np.array([lr_ref(x - off, total=total) for x in ss]))) / 3e-4))
        spikes = [int(ss[i]) for i in range(10, len(loss)) if loss[i] > 3 * np.median(loss[i-10:i])]
        late = ss >= 0.5 * total
        d = {"train_loss_first": float(loss[0]), "train_loss_end": float(loss[-1]), "train_loss_late_median": float(np.median(loss[late])),
             "train_loss_late_cv": float(np.std(loss[late]) / np.mean(loss[late])),
             "loss_spikes_3x_median10": spikes, "grad_norm_max_overall": float(gm.max()), "grad_norm_max_at": int(ss[gm.argmax()]),
             "grad_norm_max_late": float(gm[late].max()), "grad_norm_mean_late_median": float(np.median(gmean[late])),
             "lr_peak_logged": float(lr.max()), "lr_end_logged": float(lr[-1])}
        if h:
            d.update({"heldout_ema_first": h[0]["ema"]["loss"], "heldout_ema_end": h[-1]["ema"]["loss"], "heldout_raw_end": h[-1]["raw"]["loss"],
                      "ema_over_raw_end": h[-1]["ema"]["loss"] / h[-1]["raw"]["loss"],
                      "heldout_ema_min": min(x["ema"]["loss"] for x in h), "heldout_ema_min_step": min(h, key=lambda x: x["ema"]["loss"])["step"],
                      "heldout_raw_max_over_ema": max(x["raw"]["loss"] / x["ema"]["loss"] for x in h),
                      "heldout_ema_trace": [round(x["ema"]["loss"], 7) for x in h],
                      "heldout_overlap_episodes": sorted({x["overlap_episodes"] for x in h})})
        out["stages"][str(st)] = d
    out["lr_schedule_max_rel_err"] = {k: max(v) for k, v in errs.items()}
    return out
res = {}
for s in range(5):
    res[f"bc_dagger/s{s}"] = curves(R / "bc_dagger" / f"s{s}")
    res[f"bc_matched/s{s}"] = curves(R / "bc_matched" / f"s{s}")
    res[f"bc/s{s}"] = curves(R / "bc" / f"s{s}", staged=False)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
for s in range(5):
    d, m, b = res[f"bc_dagger/s{s}"], res[f"bc_matched/s{s}"], res[f"bc/s{s}"]
    print(f"s{s} nonfinite dag={d['nonfinite']} bcm={m['nonfinite']} lr_err dag={d['lr_schedule_max_rel_err']} n={d['n_train_points']}/{d['n_heldout_points']}")
    print(f"   bc 30k heldout ema {b['stages']['0']['heldout_ema_end']:.3e}")
    for st in "12345":
        a, c = d["stages"][st], m["stages"][st]
        print(f"   st{st} dag: train end {a['train_loss_end']:.2e} late med {a['train_loss_late_median']:.2e} spikes {a['loss_spikes_3x_median10']} gmax {a['grad_norm_max_overall']:.3f}@{a['grad_norm_max_at']} late {a['grad_norm_max_late']:.3f} | ho ema {a['heldout_ema_first']:.3e}->{a['heldout_ema_end']:.3e} raw {a['heldout_raw_end']:.3e} ema/raw {a['ema_over_raw_end']:.3f} maxraw/ema {a['heldout_raw_max_over_ema']:.2f}"
              f" || bcm: train late {c['train_loss_late_median']:.2e} spikes {c['loss_spikes_3x_median10']} ho ema end {c['heldout_ema_end']:.3e} ovl {c['heldout_overlap_episodes']}")
