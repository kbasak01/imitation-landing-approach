"""Scratch, read-only: touchdown bands, lateral offset, closing speed from stored select rows."""
import json, glob
import numpy as np, pandas as pd
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs/'
WT='/home/kbasak/projects_02/imitation-landing-approach/results/phase7/diagnostics/clock_intervention_episodes.csv.gz'
def load(arm, seeds, which='all'):
    out=[]
    for s in seeds:
        sel=json.load(open(f'{R}{arm}/s{s}/selection.json'))
        cs=sel['candidates'] if which=='all' else [sel['selected']]
        for c in cs:
            d=pd.read_csv(f'{R}{arm}/s{s}/select/'+c['rows_file']); d['seed']=s; d['step']=c['candidate']['global_step']; out.append(d)
    return pd.concat(out)
def band(t): return np.where(t<1.8,'fast',np.where(t<=2.8,'between','slow'))
def summ(name,d):
    td=d[d.touchdown_contact.astype(bool)]
    q=lambda x,p: np.nanquantile(x,p)
    t=td.time_to_touchdown_s; b=band(t)
    s=f"{name:22s} n{len(d):5d} succ {100*(d.outcome=='success').mean():5.1f}% | td p10/50/90 {q(t,.1):.2f}/{q(t,.5):.2f}/{q(t,.9):.2f} | band f/b/s {100*(b=='fast').mean():4.1f}/{100*(b=='between').mean():4.1f}/{100*(b=='slow').mean():4.1f}% | lat p50/p95 {q(td.lateral_offset_m,.5):.3f}/{q(td.lateral_offset_m,.95):.3f} | close p50/p95 {q(td.closing_speed_normal_m_s,.5):.3f}/{q(td.closing_speed_normal_m_s,.95):.3f}"
    oc=d.outcome.value_counts(normalize=True)*100
    s+=' | '+' '.join(f"{k[:3]} {oc.get(k,0):.1f}" for k in ['off_pad','hard_landing','bounce','crash','timeout'])
    if 'action_jerk_mean' in d: s+=f" | jerk {d.action_jerk_mean.median():.3f} effort {d.effort_mean_sq.median():.3f}"
    print(s)
    return td
arms={'dp':range(5),'dp_minus_last_action':[0,1,2],'dp_mix':range(3),'dp_pid':range(3),'dp_n300':range(3),'bc':range(5),'bc_minus_last_action':range(3)}
D={}
for a,ss in arms.items():
    for w in ('all','sel'):
        try: D[(a,w)]=load(a,ss,w)
        except Exception as e: print(a,w,'ERR',e)
tp=pd.read_csv(WT); tp=tp[(tp.subject=='T_ppo')&(tp['mode']=='none')].copy(); tp['touchdown_contact']=np.isfinite(tp.time_to_touchdown_s)
print('Bands: fast <1.8 s, between, slow >2.8 s (model scale)')
summ('T_ppo select (branch)',tp)
for k,d in D.items(): summ(f'{k[0][:16]}/{k[1]}',d)
# per-band success and lateral for fast band
print('\nFast band only (td<1.8 s): success, off_pad share, lateral p50, closing p50')
for k,d in D.items():
    if k[1]!='all': continue
    td=d[d.touchdown_contact.astype(bool)]; x=td[td.time_to_touchdown_s<1.8]
    if len(x): print(f'{k[0]:22s} n{len(x):5d} succ {100*(x.outcome=="success").mean():5.1f} off {100*(x.outcome=="off_pad").mean():5.1f} lat50 {x.lateral_offset_m.median():.3f} close50 {x.closing_speed_normal_m_s.median():.3f}')
# off_pad lateral
print('\nFailures in selected rows')
for k,d in D.items():
    if k[1]!='sel': continue
    f=d[d.outcome!='success']
    print(f'{k[0]:22s}', f.outcome.value_counts().to_dict(), 'by SS fail', f.ss.value_counts().sort_index().to_dict(), 'off_pad lat p50', round(f[f.outcome=="off_pad"].lateral_offset_m.median(),3), 'HL close p50', round(f[f.outcome=="hard_landing"].closing_speed_normal_m_s.median(),3))
# success by SS
print('\nSuccess by SS, all candidates (%)')
for k,d in D.items():
    if k[1]=='all': print(f'{k[0]:22s}', (d.groupby('ss').outcome.apply(lambda o:(o=="success").mean()*100)).round(1).to_dict())
# per-episode overlap: do dp_minus_la failing eps coincide with dp failing eps?
a=D[('dp_minus_last_action','all')]; b=D[('dp','all')]
fa=a.groupby('episode_seed').outcome.apply(lambda o:(o=='success').mean()); fb=b.groupby('episode_seed').outcome.apply(lambda o:(o=='success').mean())
j=pd.concat([fa.rename('mla'),fb.rename('dp')],axis=1)
print('\nper-episode success rate corr (all cands) mla vs dp: spearman', round(j.corr('spearman').iloc[0,1],3), ' eps mla<0.5:', (j.mla<0.5).sum(), ' of which dp<0.5:', ((j.mla<0.5)&(j.dp<0.5)).sum(), ' dp<0.5 eps:', (j.dp<0.5).sum())
pd.to_pickle(D,'/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/minus_la/D3.pkl')
