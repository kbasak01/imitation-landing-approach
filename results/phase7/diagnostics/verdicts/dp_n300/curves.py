import json, math, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def load(arm,s):
    ev=[json.loads(l) for l in open(f'{R}/{arm}/s{s}/train_log.jsonl')]
    tr=[e for e in ev if e.get('event')=='train']; ho=[e for e in ev if e.get('event')=='heldout']
    return tr,ho
for arm,seeds in [('dp_n300',[0,1,2])]:
  for s in seeds:
    try: tr,ho=load(arm,s)
    except FileNotFoundError: print(arm,s,'no log'); continue
    L=np.array([e['loss'] for e in tr]); st=np.array([e['step'] for e in tr])
    nonfin=sum(1 for e in tr if not math.isfinite(e['loss']) or not math.isfinite(e.get('grad_norm_max',0)))
    gmax=max(e.get('grad_norm_max',0) for e in tr)
    # spikes: loss > 3x running median of prev 20 logs
    spikes=[(int(st[i]),round(float(L[i]),5)) for i in range(20,len(L)) if L[i]>3*np.median(L[i-20:i])]
    lr={e['step']:e['lr'] for e in tr}
    def at(x): return min(tr,key=lambda e:abs(e['step']-x))['loss']
    print(f'{arm} s{s}: n_train_logs={len(tr)} interval={st[1]-st[0]} nonfinite={nonfin} grad_norm_max={gmax:.3f} spikes(>3x med20)={len(spikes)} {spikes[:5]}')
    print('   lr@500,1000,50k,100k:', [f"{lr.get(k,float('nan')):.3g}" for k in (400,500,1000,50000,100000)])
    print('   train loss@1k,10k,30k,60k,100k:',[f'{at(k):.5f}' for k in (1000,10000,30000,60000,100000)])
    hs={e['step']:e for e in ho}
    ks=sorted(hs); 
    emin=min(ks,key=lambda k:hs[k]['ema']['loss'])
    print(f'   heldout n={len(ks)} first={ks[0]} EMA min {hs[emin]["ema"]["loss"]:.4f}@{emin}')
    for k in (1000,5000,10000,20000,30000,60000,100000):
        if k in hs:
            e=hs[k]; print(f'     {k:>6} ema={e["ema"]["loss"]:.4f} raw={e["raw"]["loss"]:.4f} ema_k0_24={e["ema"]["loss_k0_24"]:.4f} k25_49={e["ema"]["loss_k25_49"]:.4f} k50_74={e["ema"]["loss_k50_74"]:.5f} k75_99={e["ema"]["loss_k75_99"]:.5f}')
