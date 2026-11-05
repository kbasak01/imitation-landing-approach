import sys; sys.path.insert(0,'/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/phase7/dp_pid')
from load import *
import json
d=load('dp_pid'); m=load('dp_mix'); c=load('dp')
print('render sha', d.render_cfg_sha256.unique(), 'same as dp/dp_mix', set(d.render_cfg_sha256)==set(c.render_cfg_sha256)==set(m.render_cfg_sha256))
print('frames==steps+1', (d.frames_rendered==d.steps+1).all(), 'teacher', d.teacher.unique(), 'arm', d.arm.unique(), 'run_seed==seed', (d.run_seed==d.seed).all(), 'student_seed', (d.student_seed==d.seed).all(), 'latency', d.latency_steps.unique(), 'privileged', d.privileged.unique())
for s in range(3):
    man=json.load(open(f'{R}/dp_pid/s{s}/manifest.json'))
    shas={c['global_step']:c['sha256'] for c in man['checkpoints'] if c['kind']=='full'}
    x=d[d.seed==s]
    print(s,'row ckpt sha == manifest', all(x[x.step==k].ckpt_sha256.eq(v).all() for k,v in shas.items()))
# episodes same order as select list across arms
a=d[(d.seed==0)&(d.step==70000)].episode_seed.values; b=c[(c.seed==0)&(c.selected)].episode_seed.values
print('same select episodes/order as dp', (a==b).all())
