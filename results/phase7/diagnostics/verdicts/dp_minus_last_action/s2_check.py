"""Scratch, read-only; no ila import. Rows vs selection.json, independent rule re-application,
episode order vs s0 rows, render/ckpt SHA columns; curves."""
import json, math
import numpy as np, pandas as pd
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs/dp_minus_last_action/'
sel=json.load(open(R+'s2/selection.json'))
ref=pd.read_csv(R+'s0/select/s0_step070000_rows.csv')
recs=[]
for c in sel['candidates']:
    d=pd.read_csv(R+'s2/select/'+c['rows_file'])
    assert len(d)==150 and (d.episode_seed.values==ref.episode_seed.values).all()
    assert set(d.ckpt_sha256)=={c['candidate']['sha256']}, 'sha col'
    assert set(d.render_cfg_sha256.str[:12])=={'53f802656f63'}
    assert (d.method=='dp_minus_last_action').all() and set(d.run_seed)=={2}
    oc=d.outcome.value_counts().to_dict()
    assert all(oc.get(k,0)==v for k,v in c['outcomes'].items()), (oc,c['outcomes'])
    cs=d.closing_speed_normal_m_s.values; cs=cs[np.isfinite(cs)]
    p95=float(np.quantile(cs,0.95))
    assert abs(p95-c['p95_closing_speed_m_s'])<1e-12, (p95,c['p95_closing_speed_m_s'])
    recs.append((-(d.outcome=='success').sum(),p95,c['candidate']['global_step']))
best=sorted(recs)[0]
print('rule recomputed:', -best[0], round(best[1],4), best[2], '== selected', sel['selected']['success'], sel['selected']['candidate']['global_step'])
print('Ta/K cols', d.Ta.unique(), d.K.unique())
# curves
tr=[];ho=[]
for l in open(R+'s2/train_log.jsonl'):
    x=json.loads(l)
    if x.get('event')=='train': tr.append(x)
    elif x.get('event')=='heldout': ho.append(x)
L=np.array([x['loss'] for x in tr]); st=np.array([x['step'] for x in tr])
sp=sum(1 for i in range(20,len(L)) if L[i]>3*np.median(L[i-20:i]))
he=np.array([x['ema']['loss'] for x in ho]); hr=np.array([x['raw']['loss'] for x in ho]); hs=np.array([x['step'] for x in ho]); i=int(he.argmin())
at=lambda k: he[np.where(hs==k)[0][0]]
print(f"logs {len(tr)} nan {sum(not math.isfinite(v) for v in L)} spikes {sp} gnmax {max(x['grad_norm_max'] for x in tr):.1f} train last {L[-10:].mean():.4f} | heldout EMA min {he[i]:.4f}@{hs[i]}; @60k {at(60000):.4f} @100k {he[-1]:.4f} (x{he[-1]/he[i]:.2f}) raw@100k {hr[-1]:.4f} k0-24 min {min(x['ema']['loss_k0_24'] for x in ho):.3f} @100k {ho[-1]['ema']['loss_k0_24']:.3f}")
print('lr', [(x['step'],x['lr']) for x in tr if x['step'] in (500,50000,100000)])
