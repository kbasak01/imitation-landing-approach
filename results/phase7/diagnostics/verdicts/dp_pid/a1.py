import sys; sys.path.insert(0,'/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/phase7/dp_pid')
from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
for arm in ['dp','dp_mix','dp_pid','bc_pid']:
    d=load(arm); d['ok']=d.outcome=='success'; d['band']=band(d.td_t_episode_s)
    print('=====',arm, 'n rows',len(d))
    for name,sub in [('all cands',d),('selected',d[d.selected])]:
        g=sub
        print(f' {name}: succ {g.ok.mean():.3f} td p10/50/90 {np.nanpercentile(g.td_t_episode_s,[10,50,90]).round(2)} steps mean {g.steps.mean():.1f}  lat p50/p95 {np.nanpercentile(g.lateral_offset_m,[50,95]).round(3)} close p50/p95 {np.nanpercentile(g.closing_speed_normal_m_s,[50,95]).round(3)}')
        bb=g.groupby('band').agg(share=('ok','size'),succ=('ok','mean'),lat50=('lateral_offset_m','median'),close50=('closing_speed_normal_m_s','median'))
        bb['share']=bb.share/len(g); print(bb.round(3).to_string())
        print(' outcomes',g.outcome.value_counts().to_dict())
        print(' by SS', g.groupby('ss').ok.mean().round(3).to_dict())
