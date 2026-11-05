"""FD copycat for DP: d a[t] / d last_action[t] of the first executed action (chunk idx To-1),
EMA weights, CPU fp32, DDIM K=10, fixed initial noise per state (keys (run_seed, i, 0)).
Central differences eps=0.05 (P4-D5). Scratch, read-only."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
from ila.policies.registry import load_student
from ila.policies.dp.policy import initial_noise
from ila.train.diagnostics import heldout_states, sample_indices, proprio_field_slice, COPYCAT_SEED
DATA=Path('/home/kbasak/data/ila')
states=heldout_states(DATA)
idx=sample_indices(len(states), COPYCAT_SEED, int(sys.argv[1]))
batch=states.batch(np.asarray(idx,dtype=np.int64))
IM=torch.from_numpy(np.ascontiguousarray(batch['images'])); PR=torch.from_numpy(np.ascontiguousarray(batch['proprio'])).float()
cols=proprio_field_slice('last_action','default')
out={}
for ck in sys.argv[2:]:
    t0=time.time()
    st=load_student(Path(ck),'cpu',weights='ema'); net=st.net; net.eval()
    To=st.chunk_spec.n_obs_steps if hasattr(st,'chunk_spec') else 2
    rs=int(st.run_seed) if hasattr(st,'run_seed') else 0
    noise=initial_noise([(rs,int(i),0) for i in idx], net.cfg.horizon, net.cfg.action_dim)
    def act(pr):
        res=[]
        with torch.inference_mode():
            for b in range(0,len(idx),64):
                res.append(net.sample(IM[b:b+64],pr[b:b+64],noise[b:b+64],10)[:,To-1,:])
        return torch.cat(res).double().numpy()
    J=np.zeros((3,3)); eps=0.05
    base=act(PR)
    for j in range(3):
        p=PR.clone(); p[:,-1,cols.start+j]+=eps; m=PR.clone(); m[:,-1,cols.start+j]-=eps
        J[:,j]=((act(p)-act(m))/(2*eps)).mean(0)
    # also |a_student - label| on heldout (ppo labels)
    lab=np.asarray(batch['labels'] if 'labels' in batch else batch.get('label'))
    out[ck]={'diag':np.diag(J).round(3).tolist(),'tr3':round(float(np.trace(J)/3),3),'J':J.round(3).tolist(),'s':round(time.time()-t0,1)}
    print(ck.split('runs/')[1], out[ck], flush=True)
