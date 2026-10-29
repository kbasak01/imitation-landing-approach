import json, hashlib
from pathlib import Path
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
for s in range(5):
    d = R / "bc_matched" / f"s{s}"; m = json.loads((d / "manifest.json").read_text())
    ip = Path(m["init_from"]); bcm = json.loads((ip.parent.parent / "manifest.json").read_text())
    bc_rec = [c for c in bcm["checkpoints"] if c["file"] == ip.name]
    hi = sha(ip)
    print(s, "init", ip.relative_to(R), "manifest_rec", hi == m["init_from_sha256"], "bc_manifest_rec", bool(bc_rec) and bc_rec[0]["sha256"] == hi, flush=True)
    for c in m["checkpoints"]:
        p = d / ("checkpoints" if c["kind"] == "full" else "") / c["file"]
        ok = sha(p) == c["sha256"] and p.stat().st_size == c["bytes"]
        print(s, c["file"], c["kind"], c["global_step"], "OK" if ok else "MISMATCH", flush=True)
    extra = sorted({p.name for p in (d / "checkpoints").iterdir()} - {c["file"] for c in m["checkpoints"]})
    print(s, "unlisted files", extra, flush=True)
