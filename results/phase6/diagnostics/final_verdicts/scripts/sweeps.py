import json,csv,hashlib,numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach'
d=json.load(open(f'{R}/results/dp_sweeps.json'))
SKIP={'Ta','K','ensemble'}
order=['Ta01_K010_off','Ta02_K010_off','Ta04_K010_off','Ta08_K010_off','Ta12_K010_off','Ta02_K010_ens','Ta04_K010_ens','Ta08_K010_ens','Ta12_K010_ens','Ta08_K001_off','Ta08_K002_off','Ta08_K005_off','Ta08_K025_off','Ta08_K100_off']
table={}
for s in range(5):
    r=d['runs'][f'dp/s{s}']
    sel=json.load(open(f'{R}/results/phase6/runs/dp/s{s}/selection.json'))['selected']
    assert r['checkpoint']['sha256']==sel['candidate']['sha256']
    selrows_p=f'{R}/'+r['selection_rows']['file']
    assert hashlib.sha256(open(selrows_p,'rb').read()).hexdigest()==r['selection_rows']['sha256']
    # independent reference check
    A=list(csv.DictReader(open(selrows_p))); B=list(csv.DictReader(open(f'{R}/results/phase6/sweeps/dp/s{s}/Ta08_K010_off_rows.csv')))
    diff=[(i,k,a[k],b[k]) for i,(a,b) in enumerate(zip(A,B)) for k in a if k not in SKIP and a[k]!=b[k]]
    hdr_ok=list(A[0].keys())==list(B[0].keys())
    print(f's{s}: ckpt step {r["checkpoint"]["global_step"]} reproduces_selection_rows={r["reproduces_selection_rows"]} my-diff-cells={len(diff)} hdr={hdr_ok} sweep_commit={r["sweep_git"]["ila_commit"][:7]} dirty={r["sweep_git"]["ila_dirty"]} train={r["train_git"]["ila_commit"][:7]}; Ta/K/ens in sel rows: {A[0]["Ta"]!r},{A[0]["K"]!r},{A[0]["ensemble"]!r} vs sweep {B[0]["Ta"]},{B[0]["K"]},{B[0]["ensemble"]}')
    for c in order:
        cc=r['cells'][c]
        rr=list(csv.DictReader(open(f'{R}/'+cc['rows_file'])))
        assert hashlib.sha256(open(f'{R}/'+cc['rows_file'],'rb').read()).hexdigest()==cc['rows_sha256']
        assert len(rr)==150 and [x['episode_seed'] for x in rr]==[x['episode_seed'] for x in A]
        assert all(x['ckpt_sha256']==sel['candidate']['sha256'] for x in rr)
        su=sum(x['outcome']=='success' for x in rr); assert su==cc['success']
        ta=int(c[2:4]); K=int(c[6:9]); ens=c.endswith('ens')
        assert all(int(x['Ta'])==ta and int(x['K'])==K and (x['ensemble']=='True')==ens for x in rr), c
        oc={}
        for x in rr: oc[x['outcome']]=oc.get(x['outcome'],0)+1
        table.setdefault(c,[]).append((su,oc,cc['n_tunnelled'],cc['n_detectors_disagree'],round(cc['p95_closing_speed_m_s'],3),cc['schedule_limited'],round(cc['mean_steps'],1)))
    assert r['cells']['Ta01_K010_ens']['alias_of']=='Ta01_K010_off'
print()
for c in order:
    v=table[c]; S=[x[0] for x in v]
    tot={}
    for x in v:
        for k,n in x[1].items(): tot[k]=tot.get(k,0)+n
    print(f'{c:15s} S={S} mean={np.mean(S):.1f} sl={v[0][5]} off={tot.get("off_pad",0)} hard={tot.get("hard_landing",0)} bnc={tot.get("bounce",0)} crash={tot.get("crash",0)} to={tot.get("timeout",0)} tun={[x[2] for x in v]} dd={[x[3] for x in v]} p95={[x[4] for x in v]} steps={[x[6] for x in v]}')
