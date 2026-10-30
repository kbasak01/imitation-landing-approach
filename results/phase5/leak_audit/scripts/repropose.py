import json, numpy as np, h5py, torch
torch.set_num_threads(1)
from ila import p8_adapter
from ila.train.select import checkpoint_policy
cfgs=p8_adapter.load_eval_configs()
D='/home/kbasak/data/ila/dagger/bc_dagger'
for s,i in [(0,1),(3,4),(4,5)]:
    man=json.load(open(f'{D}/s{s}/iter{i}.manifest.json'))
    st=checkpoint_policy(man['student']['ckpt'],cfgs)
    sc=np.load(f'{D}/s{s}/iter{i}.sidecar.npz'); offs=sc['offsets']
    f=h5py.File(f'{D}/s{s}/iter{i}.h5','r'); names=sorted(f.keys())
    nb=0; n=0
    for k in [0,57,133,199]:
        g=f[names[k]]; im=g['images'][()]; pr=g['proprio'][()]
        prop=sc['proposed'][offs[k]:offs[k+1]]
        st.reset(int(g.attrs['episode_seed']),None)
        for t in range(len(im)):
            w=[max(t-1,0),t]
            a=st.act_vision(im[w],pr[w])
            nb+= not np.array_equal(a,prop[t]); n+=1
    print(s,i,'steps',n,'mismatch',nb)
