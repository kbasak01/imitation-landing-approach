import json, math
import numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs/'
def rd(arm,s):
    tr=[];ho=[]
    for l in open(f'{R}{arm}/s{s}/train_log.jsonl'):
        d=json.loads(l)
        if d.get('event')=='train': tr.append(d)
        elif d.get('event')=='heldout': ho.append(d)
    return tr,ho
for arm,ss in [('dp_minus_last_action',[0,1]),('dp',range(5))]:
    for s in ss:
        tr,ho=rd(arm,s)
        nan=sum(1 for d in tr if not math.isfinite(d['loss']))
        L=np.array([d['loss'] for d in tr]); st=np.array([d['step'] for d in tr])
        # spikes: loss > 3x rolling median of previous 20 logs
        sp=sum(1 for i in range(20,len(L)) if L[i]>3*np.median(L[i-20:i]))
        he=np.array([d['ema']['loss'] for d in ho]); hr=np.array([d['raw']['loss'] for d in ho]); hs=np.array([d['step'] for d in ho])
        i=int(he.argmin())
        at=lambda k: he[np.where(hs==k)[0][0]] if k in hs else float('nan')
        b=lambda k,key: [d['ema'][key] for d in ho if d['step']==k][0]
        lr=[(d['step'],d['lr']) for d in tr if d['step'] in (500,50000,99900,100000)]
        print(f"{arm[:9]:9s} s{s} logs {len(tr)} nan {nan} spikes {sp} gnmax {max(d['grad_norm_max'] for d in tr):.1f} | train last1k {L[-10:].mean():.4f} @10k {L[st==10000][0]:.4f} | heldout EMA min {he[i]:.4f}@{hs[i]//1000}k ; @10k {at(10000):.4f} @30k {at(30000):.4f} @60k {at(60000):.4f} @70k {at(70000):.4f} @100k {he[-1]:.4f} (x{he[-1]/he[i]:.2f}) raw@100k {hr[-1]:.4f} | k0-24 min {min(d['ema']['loss_k0_24'] for d in ho):.3f} @100k {b(hs[-1],'loss_k0_24'):.3f} k75-99 @100k {b(hs[-1],'loss_k75_99'):.4f} | lr {lr}")
