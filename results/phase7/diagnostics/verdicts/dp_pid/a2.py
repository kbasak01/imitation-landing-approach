import sys; sys.path.insert(0,'/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/phase7/dp_pid')
from load import *
dp=load('dp_pid'); bp=load('bc_pid')
for nm,d in [('dp_pid',dp),('bc_pid',bp)]:
    s=d[d.selected]
    print(nm,'selected outcomes by SS'); print(pd.crosstab([s.seed,s.ss],s.outcome).to_string())
    f=s[s.outcome!='success']
    print(' failing HL closing speeds',sorted(f[f.outcome=='hard_landing'].closing_speed_normal_m_s.round(2).tolist()))
    print(' bounce closing',sorted(f[f.outcome=='bounce'].closing_speed_normal_m_s.round(2).tolist()))
    print(' failing (ss,index):',sorted(map(tuple,f[['ss','index']].values.tolist())))
s=dp[dp.selected]; b=bp[bp.selected]
fa=set(map(tuple,s[s.outcome!='success'][['ss','index']].values.tolist())); fb=set(map(tuple,b[b.outcome!='success'][['ss','index']].values.tolist()))
print('overlap of failing episodes dp_pid vs bc_pid selected', len(fa&fb), 'of', len(fa), len(fb))
# jerk
print('action_jerk_mean selected dp_pid',s.action_jerk_mean.median().round(4),'bc_pid',b.action_jerk_mean.median().round(4))
print('success by step pooled dp_pid', dp.groupby('step').apply(lambda g:(g.outcome=='success').sum()).to_dict())
print('jerk by step dp_pid', dp.groupby('step').action_jerk_mean.median().round(4).to_dict())
print('touchdown p50 by step', dp.groupby('step').td_t_episode_s.median().round(2).to_dict())
print('lat p95 by step', dp.groupby('step').lateral_offset_m.quantile(.95).round(3).to_dict())
