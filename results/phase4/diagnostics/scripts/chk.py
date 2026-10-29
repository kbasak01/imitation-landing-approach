import json, collections
from pathlib import Path
from ila.data.store import DemoStore
from ila.data import episodes as de
from ila import p8_adapter
def key(r): return (r.ss, float(r.heading_deg), float(r.speed_kn), r.vessel, int(r.realization_seed))
sel = de.read_split('select')
selk = {key(r) for r in sel}; sel_ep = {r.episode_seed for r in sel}
frozen = {}
for name, path in p8_adapter.list_paths().items():
    rows = p8_adapter.read_list(path)
    frozen[name] = ({key(r) for r in rows}, {r.episode_seed for r in rows})
print("frozen lists:", list(frozen))
for st in ['ppo_2000','dart_0.05','dart_0.10','dart_0.20','mix_1000','pid_1000']:
    with DemoStore(Path('/home/kbasak/data/ila/demos')/f'{st}.h5') as s:
        metas=[s.meta(n) for n in s.episodes()]
        fa = s.file_attrs
    ks = {(m.ss, float(m.heading_deg), float(m.speed_kn), m.vessel, int(m.realization_seed)) for m in metas}
    eps = {m.episode_seed for m in metas}
    hd = collections.Counter(m.heading_deg for m in metas); ss = collections.Counter(m.ss for m in metas)
    out = {"n":len(metas),"split":fa.get("split_file"),"ss":dict(ss),"head":dict(hd),
           "streams":dict(collections.Counter(m.stream for m in metas)),
           "sel_key_overlap":len(ks&selk),"sel_epseed_overlap":len(eps&sel_ep)}
    for fn,(fk,fe) in frozen.items():
        out[f"{fn}_key"]=len(ks&fk); out[f"{fn}_ep"]=len(eps&fe)
    if st=='ppo_2000':
        ho=[m for m in metas if 1950<=m.demo_index<=1999]
        out["heldout_n"]=len(ho); out["heldout_success"]=sum(m.outcome=="success" for m in ho)
        out["max_idx"]=max(m.demo_index for m in metas)
    print(st, json.dumps(out))
