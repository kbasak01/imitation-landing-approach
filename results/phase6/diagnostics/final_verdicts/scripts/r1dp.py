import json,csv,numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach'
def M(arm,n):
    out=[]
    for s in range(n):
        sel=json.load(open(f'{R}/results/phase6/runs/{arm}/s{s}/selection.json'))['selected']
        rr=list(csv.DictReader(open(f'{R}/artifacts/runs/{arm}/s{s}/select/{sel["rows_file"]}')))
        out.append([r['outcome']=='success' for r in rr])
    return np.array(out,float)
D=M('dp',5); Rr=M('R1_lerobot_dp',3)
rng=np.random.default_rng(0); B=20000; diffs=np.empty(B)
for b in range(B):
    e=rng.integers(0,150,150); sd=rng.integers(0,5,5); sr=rng.integers(0,3,3)
    diffs[b]=Rr[sr][:,e].mean()-D[sd][:,e].mean()
print('R1-dp selected success rate diff', (Rr.mean()-D.mean()).round(4), np.percentile(diffs,[2.5,97.5]).round(4))
# candidate means (all checkpoints) as unbiased-ish reference
sd=json.load(open('sel_out_dp.json')); sr=json.load(open('sel_out_r1.json'))
cd=np.mean([c['succ'] for s in range(5) for c in sd[f'dp/s{s}']['cands']]); cr=np.mean([c['succ'] for s in range(3) for c in sr[f'R1_lerobot_dp/s{s}']['cands']])
print('candidate-mean success dp',cd,'R1',cr, 'diff pts',(cr-cd)/1.5)
for arm,d in [('dp',sd),('R1',sr)]:
    k='dp' if arm=='dp' else 'R1_lerobot_dp'; n=5 if arm=='dp' else 3
    sel=[max(c['succ'] for c in d[f'{k}/s{s}']['cands']) for s in range(n)]
    mean=[np.mean([c['succ'] for c in d[f'{k}/s{s}']['cands']]) for s in range(n)]
    sdv=[np.std([c['succ'] for c in d[f'{k}/s{s}']['cands']],ddof=1) for s in range(n)]
    print(arm,'selected',sel,'cand mean',np.round(mean,1),'selected-minus-mean',np.round(np.array(sel)-mean,1),'cand sd',np.round(sdv,1))
# expected max of 10 iid binomial(150,p)
for p in [0.373,0.299]:
    sims=rng.binomial(150,p,(200000,10)).max(1); print('E[max of 10 Bin(150,%.3f)] = %.1f (mean %.1f)'%(p,sims.mean(),150*p))
