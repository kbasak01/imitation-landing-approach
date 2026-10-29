import json, hashlib, sys
from pathlib import Path
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
out = {}
for sg in ["0.05", "0.10", "0.20"]:
    d = R / f"bc_dart_tune/sigma_{sg}/s0"; m = json.loads((d / "manifest.json").read_text())
    r = {}
    for c in m["checkpoints"]:
        p = d / ("checkpoints" if c["kind"] == "full" else "") / c["file"]
        r[c["file"]] = sha(p) == c["sha256"] and p.stat().st_size == c["bytes"]
        print(sg, c["file"], "OK" if r[c["file"]] else "MISMATCH", flush=True)
    r["unlisted"] = sorted({p.name for p in (d / "checkpoints").iterdir()} - {c["file"] for c in m["checkpoints"]})
    out[sg] = r
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
