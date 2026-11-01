import json, csv, os, sys, math, itertools
import numpy as np
from scipy.stats import spearmanr, binomtest
R='/home/kbasak/projects_02/imitation-landing-approach'
out={}
def rows(p):
    with open(p) as f: return list(csv.DictReader(f))
def heldout_dp(seed):
    d={}
    for l in open(f'{R}/artifacts/runs/dp/s{seed}/train_log.jsonl'):
        j=json.loads(l)
        if j.get('event')=='heldout': d[j['step']]=j['ema']['loss']
    return d
def heldout_r1(seed):
    base=f'{R}/results/phase6/diagnostics/R1_lerobot_dp/heldout_reconstruction'
    d={}
    fs=[f's{seed}.json'] if seed else ['s0_step100000.json','s0_steps010000-090000.json']
    for f in fs:
        for k,v in json.load(open(f'{base}/{f}')).items(): d[int(k)]=v['heldout']['anchors']['loss']
    return d
for arm,seeds in [(a,range(n)) for a,n in ([('dp',5),('R1_lerobot_dp',3)] if len(sys.argv)<2 else [(sys.argv[1],int(sys.argv[2]))])]:
    for s in seeds:
        sel=json.load(open(f'{R}/results/phase6/runs/{arm}/s{s}/selection.json'))
        man=json.load(open(f'{R}/results/phase6/runs/{arm}/s{s}/manifest.json'))
        mck={c['global_step']:c['sha256'] for c in man['checkpoints'] if c.get('kind','full')!='ema_last'}
        ho=heldout_dp(s) if arm=='dp' else heldout_r1(s)
        cands=[]; outcome_mat=[]; eps=None
        for c in sel['candidates']:
            st=c['candidate']['global_step']; sha=c['candidate']['sha256']
            rr=rows(f'{R}/artifacts/runs/{arm}/s{s}/select/{c["rows_file"]}')
            assert len(rr)==150
            e=[(r['ss'],r['index'],r['episode_seed']) for r in rr]
            if eps is None: eps=e
            assert e==eps
            assert all(r['ckpt_sha256']==sha for r in rr), 'row sha'
            if arm=='dp': assert mck[st]==sha,'manifest sha'
            else:
                import hashlib
                mp=f'{R}/results/phase6/runs/{arm}/s{s}/onnx/step{st}/meta.json'
                meta=json.load(open(mp))
                assert hashlib.sha256(open(mp,'rb').read()).hexdigest()==sha,('metasha',st)
                assert meta['source']['checkpoint_sha256']==mck[st],('src',st)
            oc={}
            for r in rr: oc[r['outcome']]=oc.get(r['outcome'],0)+1
            succ=oc.get('success',0)
            td=np.array([float(r['closing_speed_normal_m_s']) for r in rr if r['closing_speed_normal_m_s'] not in ('','nan','None')])
            td=td[np.isfinite(td)]
            p95=float(np.quantile(td,0.95))
            ntun=sum(r['tunnelled']=='True' for r in rr); ndd=sum(r['detectors_disagree']=='True' for r in rr)
            assert succ==c['success'] and oc=={k:v for k,v in c['outcomes'].items() if v}, (st,oc,c['outcomes'])
            assert abs(p95-c['p95_closing_speed_m_s'])<1e-9,(p95,c['p95_closing_speed_m_s'])
            assert ntun==c['n_tunnelled'] and ndd==c['n_detectors_disagree']
            outcome_mat.append([r['outcome']=='success' for r in rr])
            # per SS
            ss={}
            for r in rr: ss.setdefault(r['ss'],[0,0]); ss[r['ss']][0]+=r['outcome']=='success'; ss[r['ss']][1]+=1
            cands.append(dict(step=st,succ=succ,p95=p95,oc=oc,ntun=ntun,ndd=ndd,ho=ho.get(st),ss={k:v[0] for k,v in ss.items()},mean_steps=c['mean_steps']))
        best=sorted(cands,key=lambda c:(-c['succ'],c['p95'],c['step']))[0]
        assert best['step']==sel['selected']['candidate']['global_step']
        M=np.array(outcome_mat)
        si=[c['step'] for c in cands].index(best['step'])
        mcn=[]
        for j,c in enumerate(cands):
            b=int((M[si]&~M[j]).sum()); cc=int((~M[si]&M[j]).sum())
            p=binomtest(b,b+cc).pvalue if b+cc else 1.0
            mcn.append((c['step'],b,cc,round(p,3)))
        # per-episode always/never success
        always=int(M.all(0).sum()); never=int((~M).any(0).sum()- (~M.any(0)).sum()); nev=int((~M.any(0)).sum())
        sc=[c['succ'] for c in cands]; hs=[c['ho'] for c in cands]
        rho=spearmanr(sc,hs) if None not in hs else None
        out[f'{arm}/s{s}']=dict(cands=cands,selected=best['step'],mcnemar_vs_selected=mcn,ep_always=always,ep_never=nev,
            succ_min=min(sc),succ_max=max(sc),succ_mean=float(np.mean(sc)),succ_sd=float(np.std(sc,ddof=1)),
            spearman_succ_vs_heldout=(None if rho is None else (round(rho.correlation,3),round(rho.pvalue,3))))
        print(f'== {arm}/s{s} selected {best["step"]} succ range {min(sc)}-{max(sc)} mean {np.mean(sc):.1f} sd {np.std(sc,ddof=1):.1f}; always-success eps {always}, never {nev}; rho(succ,heldout)={out[f"{arm}/s{s}"]["spearman_succ_vs_heldout"]}')
        for c,m in zip(cands,mcn):
            print(f"  {c['step']:>6} S={c['succ']:3d} p95={c['p95']:.3f} off={c['oc'].get('off_pad',0):2d} hard={c['oc'].get('hard_landing',0):2d} bnc={c['oc'].get('bounce',0):2d} cr={c['oc'].get('crash',0)} to={c['oc'].get('timeout',0)} tun={c['ntun']:2d} dd={c['ndd']} ho={c['ho']:.4f} ss={c['ss']} sel-only/cand-only={m[1]}/{m[2]} p={m[3]}")
json.dump(out,open(os.path.dirname(__file__)+'/sel_out.json','w'),indent=1)
