import numpy as np, h5py
from ila import p8_adapter
cfgs=p8_adapter.load_eval_configs()
T=p8_adapter.teacher('ppo',0,cfgs)
rng=np.random.default_rng(777)
D='/home/kbasak/data/ila/dagger/bc_dagger'
n=0; nbad=0; mx=0.0; ne=0
for s in range(5):
  for i in range(1,6):
    f=h5py.File(f'{D}/s{s}/iter{i}.h5','r'); names=sorted(f.keys())
    for name in rng.choice(names,4,replace=False):
        g=f[name]; to=g['teacher_obs'][()]; lb=g['label'][()]
        T.reset(int(g.attrs['episode_seed']), None)
        for t in range(len(to)):
            a=np.asarray(T.act(to[t]),dtype=np.float32)
            d=float(np.abs(a-lb[t]).max()); mx=max(mx,d); nbad+= d!=0; n+=1
        ne+=1
    f.close()
print('episodes',ne,'steps',n,'mismatch',nbad,'maxabs',mx, 'obs dtype', to.dtype)
