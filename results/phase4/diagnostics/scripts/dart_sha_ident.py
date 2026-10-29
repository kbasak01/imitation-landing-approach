"""Recompute checkpoint SHA-256 for bc_dart s0-4; compare bc_dart/s0 vs bc_dart_tune/sigma_0.10/s0 tensors; cross-seed sanity."""
import json, hashlib
from pathlib import Path
import torch
torch.set_num_threads(2)
from ila.policies.registry import read_checkpoint
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
for s in range(5):
    d = R / "bc_dart" / f"s{s}"; m = json.loads((d / "manifest.json").read_text())
    for c in m["checkpoints"]:
        p = d / ("checkpoints" if c["kind"] == "full" else "") / c["file"]
        ok = sha(p) == c["sha256"] and p.stat().st_size == c["bytes"]
        print("sha", s, c["file"], c["global_step"], "OK" if ok else "MISMATCH", flush=True)
    print("sha", s, "unlisted", sorted({p.name for p in (d / "checkpoints").iterdir()} - {c["file"] for c in m["checkpoints"]}), flush=True)
def cmp_sd(a, b):
    if a.keys() != b.keys(): return f"keys differ {set(a)^set(b)}"
    mx = 0.0; nbad = 0
    for k in a:
        if a[k].dtype.is_floating_point:
            d = (a[k].float() - b[k].float()).abs().max().item() if a[k].numel() else 0.0
        else:
            d = float((a[k] != b[k]).any().item())
        mx = max(mx, d); nbad += d > 0
    return f"maxabs={mx:.3g} n_tensors_differ={nbad}/{len(a)}"
def cmp_opt(a, b):
    mx = 0.0
    for pid, st in a["state"].items():
        for k, v in st.items():
            if torch.is_tensor(v): mx = max(mx, (v.float() - b["state"][pid][k].float()).abs().max().item())
    return f"maxabs={mx:.3g} groups_equal={a['param_groups']==b['param_groups']}"
T = R / "bc_dart_tune/sigma_0.10/s0"
for st in range(5000, 30001, 5000):
    f = f"s0_step{st:06d}.pt"
    a = read_checkpoint(R / "bc_dart/s0/checkpoints" / f); b = read_checkpoint(T / "checkpoints" / f)
    meta = {k: (a[k], b[k]) for k in a if k not in ("model", "ema", "optimizer", "rng", "config") and a[k] != b[k]}
    rng = {k: (torch.equal(a["rng"][k], b["rng"][k]) if torch.is_tensor(a["rng"][k]) else a["rng"][k] == b["rng"][k]) for k in a["rng"]}
    print(f"ident {st}: model {cmp_sd(a['model'], b['model'])} | ema {cmp_sd(a['ema'], b['ema'])} | optim {cmp_opt(a['optimizer'], b['optimizer'])} | rng_equal {rng} | config_equal {a['config']==b['config']} | meta_diff {meta}", flush=True)
a = read_checkpoint(R / "bc_dart/s0/ema_last.pt"); b = read_checkpoint(T / "ema_last.pt")
print("ident ema_last:", cmp_sd(a["ema"], b["ema"]), {k: (a[k], b[k]) for k in a if k not in ("model", "ema", "optimizer", "rng", "config") and a[k] != b[k]})
# control: s0 vs s1 at 30k must differ; s1 tune does not exist
a = read_checkpoint(R / "bc_dart/s0/checkpoints/s0_step030000.pt"); b = read_checkpoint(R / "bc_dart/s1/checkpoints/s0_step030000.pt")
print("control s0 vs s1 30k ema:", cmp_sd(a["ema"], b["ema"]))
# other sigma trials vs bc_dart s0 at 5k (expect differ: different data)
for sg in ["0.05", "0.20"]:
    b = read_checkpoint(R / f"bc_dart_tune/sigma_{sg}/s0/checkpoints/s0_step005000.pt"); a = read_checkpoint(R / "bc_dart/s0/checkpoints/s0_step005000.pt")
    print(f"control sigma {sg} vs bc_dart s0 5k model:", cmp_sd(a["model"], b["model"]))
