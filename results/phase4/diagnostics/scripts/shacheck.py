import json, hashlib, sys
from pathlib import Path
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
for a in ["bc_state", "bc"]:
    for s in range(5):
        d = R / a / f"s{s}"; m = json.loads((d / "manifest.json").read_text())
        for c in m["checkpoints"]:
            p = d / ("checkpoints" if c["kind"] == "full" else "") / c["file"]
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
            ok = h.hexdigest() == c["sha256"] and p.stat().st_size == c["bytes"]
            print(a, s, c["file"], c["kind"], c["global_step"], "OK" if ok else "MISMATCH " + h.hexdigest(), flush=True)
