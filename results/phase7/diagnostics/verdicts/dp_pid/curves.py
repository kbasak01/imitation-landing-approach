import json, math, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def flat(x,p=''):
    if isinstance(x,dict):
        o={}
        for k,v in x.items(): o.update(flat(v,p+k+'.'))
        return o
    return {p[:-1]:x}
for arm in ['dp_pid','dp_mix','dp']:
  for s in range(3):
    tr=[];ho=[]
    for l in open(f'{R}/{arm}/s{s}/train_log.jsonl'):
        d=json.loads(l)
        if d['event']=='train': tr.append(d)
        elif 'ema' in d and d['event']!='checkpoint': ho.append(d)
    nan=any(not math.isfinite(d['loss']) for d in tr)
    L=np.array([d['loss'] for d in tr]); st=np.array([d['step'] for d in tr])
    spikes=int(np.sum(L[10:]>3*np.median(L[max(0,0):])))
    lr=[(d['step'],d['lr']) for d in tr if d['step'] in (500,1000,50000,99900,100000)]
    if s==0 and arm=='dp_pid': print('heldout record keys', list(flat(ho[0]).keys())[:30])
    f=[flat(h) for h in ho]
    def key(name):
        ks=[k for k in f[0] if k.startswith(name) and isinstance(f[0][k],(int,float)) and not isinstance(f[0][k],bool)]
        return ks
    ek=key('ema'); rk=key('raw')
    # choose total loss key
    e=[x.get('ema.loss', x.get(ek[0])) for x in f]; r=[x.get('raw.loss', x.get(rk[0])) for x in f]; hs=[x['step'] for x in f]
    i=int(np.argmin(e))
    def at(step):
        j=hs.index(step) if step in hs else None
        return None if j is None else (round(e[j],5), round(r[j],5))
    print(f'{arm} s{s}: NaN {nan} train loss@1k {L[st==1000][0] if (st==1000).any() else None:.4f} @50k {L[st==50000][0]:.5f} @100k {L[-1]:.5f} spikes>3xmed {spikes}; lr {lr}; heldout ema min {e[i]:.5f}@{hs[i]} | (ema,raw)@10k {at(10000)} @30k {at(30000)} @60k {at(60000)} @70k {at(70000)} @100k {at(100000)} rise100k/min {e[-1]/e[i]:.3f}; overlap {set(x.get("overlap_episodes") for x in f)}')
