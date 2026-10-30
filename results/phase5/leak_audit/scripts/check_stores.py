import json, sys, glob
import numpy as np, pandas as pd, h5py
R='/home/kbasak/projects_02/imitation-landing-approach'
D='/home/kbasak/data/ila/dagger/bc_dagger'
SIZES=dict(attitude=3,body_rates=3,velocity=3,rel_position=3,rel_velocity=3,deck_normal=3,rel_tilt=1,height=1,time_fraction=1,last_action=3,in_contact=1)
demos=pd.read_parquet(f'{R}/splits/demos_0-1999.parquet')
sel=pd.read_parquet(f'{R}/splits/select.parquet')
KEY=['ss','heading_deg','speed_kn','vessel','realization_seed']
def keys(df): return set(map(tuple, df[KEY].astype(str).values.tolist()))
frozen={}
for p in glob.glob(f'{R}/external/rl-land-on-moving-deck/results/episodes/*.parquet'):
    frozen[p.split('/')[-1]]=pd.read_parquet(p)
fz_keys=set().union(*[keys(d) for d in frozen.values()])
fz_seeds=set().union(*[set(d.episode_seed.astype(int)) for d in frozen.values()])
sel_keys=keys(sel); sel_seeds=set(sel.episode_seed.astype(int))
tot=dict(eps=0,frames=0)
bad=[]
starts={}
for s in range(5):
  for i in range(1,6):
    f=h5py.File(f'{D}/s{s}/iter{i}.h5','r')
    obs_fields=json.loads(f.attrs['obs_fields']); prop_fields=json.loads(f.attrs['proprio_fields'])
    off={}; o=0
    for n in obs_fields: off[n]=(o,o+SIZES[n]); o+=SIZES[n]
    assert o==25
    pcols=np.concatenate([np.arange(*off[n]) for n in prop_fields])
    privcols=[c for n in ['rel_position','rel_velocity','deck_normal','rel_tilt','height','time_fraction'] for c in range(*off[n])]
    sc=np.load(f'{D}/s{s}/iter{i}.sidecar.npz')
    offs=sc['offsets']
    names=sorted(f.keys())
    want=[f'ep_{j:05d}' for j in range(1000+200*(i-1),1000+200*i)]
    if names!=want: bad.append((s,i,'names'))
    for k,n in enumerate(names):
        g=f[n]; a=dict(g.attrs)
        j=int(a['demo_index']); row=demos.iloc[j]
        for c in ['ss','index','heading_deg','speed_kn','vessel','realization_seed','episode_seed']:
            if str(row[c])!=str(a[c]): bad.append((s,i,n,'row',c,row[c],a[c]))
        key=tuple(str(row[c]) for c in KEY)
        if key in fz_keys: bad.append((s,i,n,'frozen key'))
        if key in sel_keys: bad.append((s,i,n,'select key'))
        if int(a['episode_seed']) in fz_seeds: bad.append((s,i,n,'frozen seed'))
        if int(a['episode_seed']) in sel_seeds: bad.append((s,i,n,'select seed'))
        if a['ss']=='SS6' or float(a['heading_deg'])==90.0 or not (0<=int(a['realization_seed'])<=26): bad.append((s,i,n,'pool'))
        if a['dagger_iteration']!=i or float(a['beta'])!=[0.5,0.25,0.125,0,0][i-1]: bad.append((s,i,n,'beta/iter'))
        to=g['teacher_obs'][()]; pr=g['proprio'][()]; pv=g['priv'][()]; ae=g['action_exec'][()]; lb=g['label'][()]
        if not np.array_equal(pr, to[:,pcols]): bad.append((s,i,n,'proprio!=allowlist'))
        # no proprio column equals a privileged/clock column
        for pc in range(pr.shape[1]):
            for qc in privcols:
                if len(pr)>3 and np.array_equal(pr[:,pc], to[:,qc]) and np.ptp(pr[:,pc])>0: bad.append((s,i,n,'proprio col equals priv',pc,qc))
        la=to[:,slice(*off['last_action'])]
        if np.any(la[0]!=0) or not np.array_equal(la[1:],ae[:-1]): bad.append((s,i,n,'last_action'))
        lo,hi=int(offs[k]),int(offs[k+1])
        if hi-lo!=len(ae): bad.append((s,i,n,'sidecar len')); continue
        prop=sc['proposed'][lo:hi]; te=sc['teacher_exec'][lo:hi]; u=sc['u'][lo:hi]
        if not np.array_equal(te, u<[0.5,0.25,0.125,0,0][i-1]): bad.append((s,i,n,'coin'))
        chosen=np.where(te[:,None], lb, prop).astype(np.float64)
        ex=np.clip(chosen,-1,1).astype(np.float32)
        if not np.array_equal(ex,ae): bad.append((s,i,n,'exec', float(np.abs(ex-ae).max())))
        if int(sc['demo_index'][k])!=j or int(sc['episode_seed'][k])!=int(a['episode_seed']): bad.append((s,i,n,'sidecar ids'))
        starts.setdefault(j,set()).add((key,int(a['episode_seed']), g['images'][0].tobytes().__hash__()))
        tot['eps']+=1; tot['frames']+=len(ae)
    f.close()
print(tot, 'bad', len(bad)); print(bad[:20])
print('starting conditions identical across seeds (key+seed):', all(len({(x[0],x[1]) for x in v})==1 for v in starts.values()), len(starts))
print('t0 frame identical across seeds:', sum(len(v)==1 for v in starts.values()), 'of', len(starts))
