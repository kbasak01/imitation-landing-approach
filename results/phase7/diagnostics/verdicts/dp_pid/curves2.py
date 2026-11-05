import json, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
for arm in ['dp_pid','dp']:
  for s in range(3):
    tr=[];ho=[]
    for l in open(f'{R}/{arm}/s{s}/train_log.jsonl'):
        d=json.loads(l)
        if d['event']=='train': tr.append(d)
        elif d['event']=='heldout': ho.append(d)
    L=np.array([d['loss'] for d in tr]); st=np.array([d['step'] for d in tr]); g=np.array([d['grad_norm_max'] for d in tr])
    sp=[]
    for i in range(5,len(L)):
        med=np.median(L[max(0,i-10):i+11])
        if L[i]>2*med: sp.append((int(st[i]),round(L[i]/med,2)))
    h=ho[-1]
    print(arm,s,'local spikes(>2x rolling med)',sp[:8],'gradmax late',g[-100:].max().round(3),'| heldout@100k buckets ema',{k:round(v,4) for k,v in h['ema'].items()}, 'ema-raw gap@100k',round(h['raw']['loss']-h['ema']['loss'],5))
    h=[x for x in ho if x['step']==10000][0]; print('     @10k',{k:round(v,4) for k,v in h['ema'].items()})
