import json,csv,sys,numpy as np
from scipy.stats import chi2, spearmanr, kendalltau
R='/home/kbasak/projects_02/imitation-landing-approach'
for arm,n,f in [('dp',5,'sel_out_dp.json'),('R1_lerobot_dp',3,'sel_out_r1.json')]:
    d=json.load(open(f)); print('=====',arm)
    steps=list(range(10000,100001,10000))
    S=np.array([[c['succ'] for c in d[f'{arm}/s{s}']['cands']] for s in range(n)])
    H=np.array([[c['ho'] for c in d[f'{arm}/s{s}']['cands']] for s in range(n)])
    print('mean success by step', dict(zip([s//1000 for s in steps],S.mean(0).round(1))))
    # pooled within-seed spearman: rank within seed
    rs=[spearmanr(S[i],H[i]).correlation for i in range(n)]; print('per-seed rho(succ,heldout)',np.round(rs,2),'mean',np.mean(rs).round(2))
    rs2=[spearmanr(S[i],steps).correlation for i in range(n)]; print('per-seed rho(succ,step)',np.round(rs2,2),'mean',np.mean(rs2).round(2))
    # Cochran's Q per seed
    for s in range(n):
        sel=json.load(open(f'{R}/results/phase6/runs/{arm}/s{s}/selection.json'))
        M=[]
        for c in sel['candidates']:
            rr=list(csv.DictReader(open(f'{R}/artifacts/runs/{arm}/s{s}/select/{c["rows_file"]}')))
            M.append([r['outcome']=='success' for r in rr])
        X=np.array(M,dtype=int).T  # eps x k
        k=X.shape[1]; C=X.sum(0); Rw=X.sum(1); N=X.sum()
        Q=(k-1)*(k*(C**2).sum()-N**2)/(k*N-(Rw**2).sum())
        p=1-chi2.cdf(Q,k-1)
        # episode effect: distribution of per-episode success counts vs binomial(10,pbar)
        pbar=X.mean(); var_obs=Rw.var(); var_bin=k*pbar*(1-pbar)
        # mean pairwise outcome agreement
        print(f'  s{s}: Cochran Q={Q:.1f} df={k-1} p={p:.3f}; per-episode success-count var {var_obs:.2f} vs independent binomial {var_bin:.2f} (ratio {var_obs/var_bin:.2f}); eps with 0/10:{(Rw==0).sum()} 10/10:{(Rw==10).sum()} hist={np.bincount(Rw,minlength=11).tolist()}')
    # outcomes 40-50k vs 70-100k pooled
    def agg(ss):
        tot={}
        for s in range(n):
            for c in d[f'{arm}/s{s}']['cands']:
                if c['step'] in ss:
                    for k_,v in c['oc'].items(): tot[k_]=tot.get(k_,0)+v
                    tot['tun']=tot.get('tun',0)+c['ntun']; tot['N']=tot.get('N',0)+150
        return tot
    for lab,ss in [('10-30k',[10000,20000,30000]),('40-50k',[40000,50000]),('60k',[60000]),('70-100k',[70000,80000,90000,100000])]:
        t=agg(ss); N=t['N']; print(f'  {lab}: N={N} '+' '.join(f"{k}={100*t.get(k,0)/N:.1f}%" for k in ['success','off_pad','hard_landing','bounce','crash','timeout','tun']))
    pp=[]
    for s in range(n):
        c=d[f'{arm}/s{s}']['cands']
        pp.append([x['p95'] for x in c])
    print('  mean p95 by step',dict(zip([s//1000 for s in steps],np.array(pp).mean(0).round(3))))
