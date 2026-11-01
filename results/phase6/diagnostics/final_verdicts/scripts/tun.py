import json,csv,numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach'
def load(arm,s):
    sel=json.load(open(f'{R}/results/phase6/runs/{arm}/s{s}/selection.json'))['selected']
    return list(csv.DictReader(open(f'{R}/artifacts/runs/{arm}/s{s}/select/{sel["rows_file"]}')))
for arm,n in [('dp',5),('R1_lerobot_dp',3)]:
    rows=[r for s in range(n) for r in load(arm,s)]
    tab={}
    for r in rows:
        k=(r['outcome'],r['tunnelled']); tab[k]=tab.get(k,0)+1
    print(arm, 'outcome x tunnelled', sorted(tab.items()))
    t=[r for r in rows if r['tunnelled']=='True']
    cs=np.array([float(r['closing_speed_normal_m_s']) for r in t]); pen=np.array([float(r['max_penetration_m']) for r in t])
    cs_all=np.array([float(r['closing_speed_normal_m_s']) for r in rows if r['closing_speed_normal_m_s']])
    print(f'  tunnelled n={len(t)} closing median {np.median(cs):.3f} vs all {np.nanmedian(cs_all):.3f}; max pen mm median {1000*np.median(pen):.1f} max {1000*pen.max():.1f}')
    lo=np.array([float(r['lateral_offset_m']) for r in rows if r['lateral_offset_m'] not in ('',)])
    oc=[r['outcome'] for r in rows if r['lateral_offset_m'] not in ('',)]
    lo_s=lo[[o=='success' for o in oc]]; lo_o=lo[[o=='off_pad' for o in oc]]
    print(f'  lateral offset m: all median {np.nanmedian(lo):.3f} mean {np.nanmean(lo):.3f}; success max {np.nanmax(lo_s):.3f}; off_pad min {np.nanmin(lo_o):.3f} median {np.nanmedian(lo_o):.3f}')
    hl=[float(r['closing_speed_normal_m_s']) for r in rows if r['outcome']=='hard_landing']
    print(f'  hard_landing closing min {min(hl):.3f} median {np.median(hl):.3f}; success closing max {max(float(r["closing_speed_normal_m_s"]) for r in rows if r["outcome"]=="success"):.3f}')
    to=[r for r in rows if r['outcome'] in('timeout','crash')]
    for r in to: print('   ',r['outcome'],r['ss'],r['index'],r['steps'],r['termination_reason'],r['lateral_offset_m'][:6],r['closing_speed_normal_m_s'][:6])
    st=np.array([int(r['steps']) for r in rows]); print(f'  steps median {np.median(st)} max {st.max()} n>200 {(st>200).sum()}')
# Ta=1 sweep lateral offsets
rows=[r for s in range(5) for r in csv.DictReader(open(f'{R}/results/phase6/sweeps/dp/s{s}/Ta01_K010_off_rows.csv'))]
lo=np.array([float(r['lateral_offset_m']) for r in rows if r['lateral_offset_m']]); print('dp Ta=1 lateral offset median',np.median(lo).round(3),'mean',lo.mean().round(3))
rows=[r for s in range(5) for r in csv.DictReader(open(f'{R}/results/phase6/sweeps/dp/s{s}/Ta08_K010_ens_rows.csv'))]
lo=np.array([float(r['lateral_offset_m']) for r in rows if r['lateral_offset_m']]); print('dp Ta=8 ens lateral offset median',np.median(lo).round(3))
for arm in ['bc','bc_dagger']:
    import glob
    ph='phase4' if arm=='bc' else 'phase5'
    rr=[]
    for s in range(5):
        sel=json.load(open(f'{R}/results/{ph}/runs/{arm}/s{s}/selection.json'))['selected'] if __import__('os').path.exists(f'{R}/results/{ph}/runs/{arm}/s{s}/selection.json') else None
        if sel is None: continue
        p=glob.glob(f'{R}/artifacts/runs/{arm}/s{s}/select/{sel["rows_file"]}') or glob.glob(f'{R}/artifacts/runs/{arm}/s{s}/**/{sel["rows_file"]}',recursive=True)
        if p: rr+=list(csv.DictReader(open(p[0])))
    if rr:
        lo=np.array([float(r['lateral_offset_m']) for r in rr if r['lateral_offset_m']]); print(arm,'n',len(rr),'lateral offset median',np.median(lo).round(3), 'mean jerk',np.mean([float(r['action_jerk_mean']) for r in rr]).round(3))
