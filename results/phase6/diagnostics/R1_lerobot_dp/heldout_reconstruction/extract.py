import h5py, numpy as np, sys
out=sys.argv[1]
f=h5py.File('/home/kbasak/data/ila/demos/ppo_2000.h5','r')
print(list(f['ep_00000'].attrs))
by_stream={}
for k in f.keys():
    a=f[k].attrs; by_stream[int(a['demo_index'])]=k
def take(rng, name):
    imgs=[];prop=[];lab=[];epi=[];info=[]
    for s in rng:
        g=f[by_stream[s]]; oc=g.attrs.get('outcome', g.attrs.get('outcome_class'))
        oc = oc.decode() if isinstance(oc,bytes) else oc
        info.append((s,by_stream[s],oc,g['label'].shape[0]))
        if oc!='success': continue
        imgs.append(g['images'][()]); prop.append(g['proprio'][()]); lab.append(g['label'][()]); epi.append(np.full(len(g['label']),s))
    print(name, [i for i in info if i[2]!='success'], 'kept', len(imgs), 'frames', sum(len(x) for x in lab))
    np.savez(f'{out}/{name}.npz', images=np.concatenate(imgs), proprio=np.concatenate(prop), label=np.concatenate(lab), ep=np.concatenate(epi))
take(range(1950,2000),'heldout')
take(range(0,50),'train50')
