import json, csv, collections, os, sys
import numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
disk={}
for line in open('/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/ckpt_sha.txt'):
    h,f=line.split(); disk[f]=h
def p95(x): return float(np.percentile(x,95)) if len(x) else float('nan')
for arm,seeds in [('dp_n100',[0,1,2])]:
  for s in seeds:
    d=f'{R}/{arm}/s{s}'
    m=json.load(open(f'{d}/manifest.json')); sel=json.load(open(f'{d}/selection.json'))
    man={c['file']:c['sha256'] for c in m['checkpoints']}
    # jsonl checkpoint events
    ev={}
    for l in open(f'{d}/train_log.jsonl'):
        e=json.loads(l)
        if e.get('event')=='checkpoint': ev[e['file']]=e['sha256']
    ok_disk=all(disk.get(f's{s}/checkpoints/{f}')==h for f,h in man.items())
    ok_ev=ev==man
    print(f'== s{s}: run_seed={m["run_seed"]} sel.run_seed={sel["run_seed"]} n_ckpt={len(man)} disk==manifest:{ok_disk} log==manifest:{ok_ev} inference={sel["inference"]} path={sel["inference_path"]} weights={sel["weights"]} list_sha={sel["list"]["file_sha256"][:12]}')
    best=None
    for c in sel['candidates']:
        f=os.path.basename(c['candidate']['file']); assert c['candidate']['file'].startswith(d+'/'), c['candidate']['file']
        rows=list(csv.DictReader(open(f'{d}/select/{c["rows_file"]}')))
        rs={r['run_seed'] for r in rows}; ss={r['student_seed'] for r in rows}; ck={r['ckpt_sha256'] for r in rows}; arms={r['arm'] for r in rows}
        oc=collections.Counter(r['outcome'] for r in rows)
        td=[float(r['closing_speed_normal_m_s']) for r in rows if r['touchdown_contact']=='True' or r['touchdown_analytic']=='True']
        tdc=[float(r['closing_speed_normal_m_s']) for r in rows if r['touchdown_contact']=='True']
        cs=[float(r['closing_speed_normal_m_s']) for r in rows if r['closing_speed_normal_m_s'] not in ('','nan')]
        succ=oc['success']
        match = (succ==c['success'] and dict(oc)=={k:v for k,v in c['outcomes'].items() if v} and ck=={c['candidate']['sha256']} and man[f]==c['candidate']['sha256'])
        print(f'  {c["candidate"]["global_step"]:>6} succ={succ:3d} oc={dict(sorted(oc.items()))} p95sel={c["p95_closing_speed_m_s"]:.4f} p95(any td)={p95(td):.4f} p95(contact)={p95(tdc):.4f} p95(all finite)={p95(cs):.4f} n={len(rows)} run_seed={rs} stud={ss} arm={arms} rows==json/manifest:{match}')
        key=(-c['success'],c['p95_closing_speed_m_s'],c['candidate']['global_step'])
        if best is None or key<best[0]: best=(key,c['candidate']['global_step'])
    print('  recomputed selection:',best[1],' recorded:',sel['selected']['candidate']['global_step'], sel['selected']['success'], round(sel['selected']['p95_closing_speed_m_s'],4))
