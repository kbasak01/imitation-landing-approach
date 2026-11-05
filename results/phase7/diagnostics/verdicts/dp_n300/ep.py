import json, csv, collections, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def rows(arm,s,sel_only=False):
    sel=json.load(open(f'{R}/{arm}/s{s}/selection.json'))
    cs=[sel['selected']] if sel_only else sel['candidates']
    return [list(csv.DictReader(open(f'{R}/{arm}/s{s}/select/{c["rows_file"]}'))) for c in cs]
keyf=lambda r:(r['ss'],r['index'],r['episode_seed'])
P={}
for arm,seeds in [('dp_n100',[0,1,2]),('dp_n300',[0,1,2]),('dp',[0,1,2,3,4])]:
    allr=[r for s in seeds for rr in rows(arm,s) for r in rr]
    keys=sorted({keyf(r) for r in allr})
    succ=collections.defaultdict(list)
    for r in allr: succ[keyf(r)].append(r['outcome']=='success')
    P[arm]=np.array([np.mean(succ[k]) for k in keys]); K=keys
    byss=collections.defaultdict(list)
    for r in allr: byss[r['ss']].append(r['outcome']=='success')
    sel=[r for s in seeds for rr in rows(arm,s,True) for r in rr]
    selss=collections.defaultdict(list)
    for r in sel: selss[r['ss']].append(r['outcome']=='success')
    jm=[float(r['action_jerk_mean']) for r in sel]; st=[int(r['steps']) for r in sel]
    print(arm,'n episodes',len(keys),'all-cand success by SS',{k:round(np.mean(v),3) for k,v in sorted(byss.items())},'selected by SS',{k:round(np.mean(v),3) for k,v in sorted(selss.items())})
    print('   selected action_jerk_mean median %.4f p95 %.4f ; steps median %d'%(np.median(jm),np.percentile(jm,95),np.median(st)))
    print('   per-episode success-prob distribution: frac always fail %.3f, always succeed %.3f, hist'%(np.mean(P[arm]==0),np.mean(P[arm]==1)), np.histogram(P[arm],bins=[0,.001,.2,.4,.6,.8,.999,1.0])[0])
a,b=P['dp_n300'],P['dp']
print('per-episode corr n300 vs dp: %.3f'%np.corrcoef(a,b)[0,1])
# binomial check: if episodes were exchangeable with p=mean, var of per-episode rate
for arm,n in [('dp_n100',30),('dp_n300',30),('dp',50)]:
    p=P[arm].mean(); print(arm,'between-episode var %.4f vs binomial-iid %.4f'%(P[arm].var(), p*(1-p)/n))
print('per-episode corr n100 vs n300: %.3f'%np.corrcoef(P['dp_n100'],P['dp_n300'])[0,1])
# success vs heldout loss across checkpoints
from scipy.stats import spearmanr
for arm,seeds in [('dp_n100',[0,1,2]),('dp_n300',[0,1,2]),('dp',[0,1,2,3,4])]:
    xs=[];ys=[]
    for s in seeds:
        ho={}
        for l in open(f'{R}/{arm}/s{s}/train_log.jsonl'):
            e=json.loads(l)
            if e.get('event')=='heldout': ho[e['step']]=e['ema']['loss']
        sel=json.load(open(f'{R}/{arm}/s{s}/selection.json'))
        for c in sel['candidates']: xs.append(ho[c['candidate']['global_step']]); ys.append(c['success'])
    print(arm,'spearman(success, heldout EMA loss) over candidates: rho=%.3f p=%.3g n=%d'%(*spearmanr(xs,ys),len(xs)))
