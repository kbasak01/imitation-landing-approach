import json, csv, collections, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def ep(arm,s):
    epc=json.load(open(f'{R}/{arm}/s{s}/train_log.jsonl'.replace('train_log.jsonl','train_log.jsonl'))) if False else None
for arm,seeds in [('dp_n100',[0,1,2]),('dp_n300',[0,1,2]),('dp',[0,1,2,3,4])]:
  allc=[];sels=[];selfail=collections.Counter();allfail=collections.Counter();lat_off=[];lat_succ=[]
  early=[];late=[]
  for s in seeds:
    sel=json.load(open(f'{R}/{arm}/s{s}/selection.json'))
    succ=[c['success'] for c in sel['candidates']]; steps=[c['candidate']['global_step'] for c in sel['candidates']]
    allc+=succ; sels.append(sel['selected']['success'])
    early+= [c['success'] for c in sel['candidates'] if c['candidate']['global_step']<=30000]
    late += [c['success'] for c in sel['candidates'] if c['candidate']['global_step']>=80000]
    for c in sel['candidates']:
        for k,v in c['outcomes'].items():
            if k!='success': allfail[k]+=v
    for k,v in sel['selected']['outcomes'].items():
        if k!='success': selfail[k]+=v
    rows=list(csv.DictReader(open(f'{R}/{arm}/s{s}/select/{sel["selected"]["rows_file"]}')))
    lat_off+=[float(r['lateral_offset_m']) for r in rows if r['outcome']=='off_pad' and r['lateral_offset_m'] not in ('','nan')]
    lat_succ+=[float(r['lateral_offset_m']) for r in rows if r['outcome']=='success']
    xs=np.array(steps); ys=np.array(succ); slope=np.polyfit(xs/1e4,ys,1)[0]
    print(f'{arm} s{s}: per-ckpt success {succ} mean={np.mean(succ):.1f} sd={np.std(succ,ddof=1):.1f} slope/10k={slope:+.2f} selected={sel["selected"]["success"]}@{sel["selected"]["candidate"]["global_step"]} sel-mean={sel["selected"]["success"]-np.mean(succ):+.1f}')
  nf=sum(allfail.values()); nsf=sum(selfail.values())
  print(f' {arm}: all-cand mean {np.mean(allc):.1f}/150 ; early(<=30k) mean {np.mean(early):.1f} late(>=80k) mean {np.mean(late):.1f}; selected {sels}')
  print(f'  all-candidate failures: {dict(allfail)} off_pad share of failures={allfail["off_pad"]/nf:.3f}, of episodes={allfail["off_pad"]/(150*len(allc)):.3f}')
  print(f'  selected failures: {dict(selfail)} off_pad share of failures={selfail["off_pad"]/nsf:.3f}')
  lo=np.array(lat_off); ls_=np.array(lat_succ)
  print(f'  selected lateral offset: off_pad n={len(lo)} median={np.median(lo):.3f} p10={np.percentile(lo,10):.3f} p90={np.percentile(lo,90):.3f} frac<0.15={np.mean(lo<0.15):.2f}; success median={np.median(ls_):.3f} p90={np.percentile(ls_,90):.3f}')
