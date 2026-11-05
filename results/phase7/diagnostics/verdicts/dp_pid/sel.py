import json, math
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def wilson(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
for s in range(3):
    d=json.load(open(f'{R}/dp_pid/s{s}/selection.json'))
    cs=[(c['candidate']['global_step'],c['success'],c['p95_closing_speed_m_s'],c['outcomes']) for c in d['candidates']]
    # recompute rule
    best=sorted(cs,key=lambda c:(-c[1],c[2],c[0]))[0]
    sel=d['selected']['candidate']['global_step']
    lo,hi=wilson(best[1],150)
    within=[c[0] for c in cs if wilson(c[1],150)[1]>=lo]
    mean=sum(c[1] for c in cs)/len(cs); late=[c[1] for c in cs if c[0]>=50000]
    print(f's{s}: rule->{best[0]} json->{sel} ok={best[0]==sel}; succ by step {[c[1] for c in cs]}; sel {best[1]} Wilson [{lo*150:.1f},{hi*150:.1f}]/150 = [{lo:.3f},{hi:.3f}]; cands whose CI overlaps: {within}; mean all {mean:.1f}, mean 50-100k {sum(late)/len(late):.1f}; winner-minus-late {best[1]-sum(late)/len(late):+.1f}')
    print('   p95 by step',[round(c[2],3) for c in cs]); print('   off_pad',[c[3]['off_pad'] for c in cs],'HL',[c[3]['hard_landing'] for c in cs],'bounce',[c[3]['bounce'] for c in cs])
