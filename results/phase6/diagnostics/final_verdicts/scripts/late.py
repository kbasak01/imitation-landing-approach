import json,csv,numpy as np
from scipy.stats import binomtest
R='/home/kbasak/projects_02/imitation-landing-approach'
def succ(arm,s,f): return np.array([r['outcome']=='success' for r in csv.DictReader(open(f'{R}/artifacts/runs/{arm}/s{s}/select/{f}'))])
for a_,b_ in [(100000,40000),(100000,50000)]:
    B=C=0
    for s in range(5):
        a=succ('dp',s,f's0_step{a_:06d}_rows.csv'); b=succ('dp',s,f's0_step{b_:06d}_rows.csv')
        B+=int((a&~b).sum()); C+=int((~a&b).sum())
    print(f'dp pooled {a_//1000}k-only {B} vs {b_//1000}k-only {C}, exact p={binomtest(B,B+C).pvalue:.3f}, diff {(B-C)/750*100:.1f} pts')
# late (70-100k avg) vs 40-50k avg per seed
d=json.load(open('sel_out_dp.json'))
for s in range(5):
    c={x['step']:x['succ'] for x in d[f'dp/s{s}']['cands']}
    print(s, np.mean([c[k] for k in (70000,80000,90000,100000)])-np.mean([c[k] for k in (40000,50000)]))
