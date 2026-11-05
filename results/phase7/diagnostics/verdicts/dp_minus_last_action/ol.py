"""Scratch, read-only, open-loop only (no episodes flown). Held-out ppo_2000 1950-1999 every 2nd.
(A) teacher-forced replay at Ta=8,K=10, EMA, CPU fp32: per-axis RMSE of executed vs label.
(B) core dp only: same, but last_action in the window replaced by the student's OWN previous
    executed action (what it sees in closed loop); images and other proprio teacher-forced.
(S) variant only: spread/clip/bias (dp_diagnostics.spread_and_clip) + open-loop jerk."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
from ila.policies.registry import load_student, read_checkpoint
from ila.data.store import DemoStore
from ila.train.sampling import choose_episodes
from ila.train.dataset import FrameTable, DPChunkDataset
from ila.train import dp_diagnostics as dd
from ila.train.diagnostics import proprio_field_slice
R=Path('/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs')
OUT=Path('/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/faab0cc0-7e62-448c-9376-9d45c0926856/scratchpad/minus_la/ol_out'); OUT.mkdir(exist_ok=True)
def data_for(variant):
    with DemoStore(Path('/home/kbasak/data/ila/demos/ppo_2000.h5')) as st:
        names=list(choose_episodes(st,50,False,start=1950).names)[::2]
        tab=FrameTable.from_store(st,names,variant)
    return DPChunkDataset(tab,2,16)
def replay(student,data,own_la=None):
    tab=data.table; out=[]
    for e in range(tab.n_episodes):
        s0,T=int(tab.episode_start[e]),int(tab.episode_len[e])
        student.reset(dd.OPEN_LOOP_KEY_BASE-e); acts=[]
        for t in range(T):
            smp=data[s0+t]; pr=np.array(smp['proprio'],dtype=np.float32,copy=True)
            if own_la is not None:
                # window steps [t-1,t] (index<0 -> 0); last_action[k] = exec[k-1], 0 at k=0
                ws=[max(t-1,0),t]
                for i,k in enumerate(ws):
                    pr[i,own_la]=acts[k-1] if k>0 else 0.0
            acts.append(np.asarray(student.act_vision(np.asarray(smp['images'],dtype=np.uint8),pr),dtype=np.float64))
        out.append((np.asarray(acts),np.asarray(tab.label[s0:s0+T],dtype=np.float64)))
    return out
def err(rep):
    A=np.concatenate([a for a,_ in rep]); L=np.concatenate([l for _,l in rep]); d=A-L
    return {'rmse_xyz':np.sqrt((d**2).mean(0)).round(4).tolist(),'rmse_lat':round(float(np.sqrt((d[:,:2]**2).sum(1).mean())),4),'mean_norm':round(float(np.linalg.norm(d,axis=1).mean()),4),
            'act_change_mean':round(float(np.mean([np.linalg.norm(np.diff(a,axis=0),axis=1).mean() for a,_ in rep])),4),
            'label_change_mean':round(float(np.mean([np.linalg.norm(np.diff(l,axis=0),axis=1).mean() for _,l in rep])),4)}
res={}
for tag in sys.argv[1:]:
    arm,s=tag.split('/'); t0=time.time()
    sel=json.load(open(R/arm/s/'selection.json')); ck=Path(sel['selected']['candidate']['file'])
    variant=str(read_checkpoint(ck)['obs_variant'])
    st=load_student(ck,'cpu',weights='ema',n_action_steps=8,num_inference_steps=10)
    data=data_for(variant)
    r={'ckpt':ck.name,'variant':variant,'proprio_dim':int(np.asarray(data[0]['proprio']).shape[-1]),'frames':len(data)}
    A=replay(st,data); r['A_teacher_forced']=err(A)
    r['A_jerk_d1_ratio']=dd.jerk_summary([dd.chunk_jerk(a,8) for a,_ in A])['d1_ratio_boundary_over_within']
    if variant=='default':
        sl=proprio_field_slice('last_action','default')
        B=replay(st,data,own_la=sl); r['B_own_last_action']=err(B)
    else:
        idx=dd.draw_states(len(data)); b=data.batch(idx)
        sp=dd.spread_and_clip(st.net,int(sel['run_seed']),np.asarray(b['images']),np.asarray(b['proprio']),np.asarray(b['target']))
        rec=dd.spread_record(sp); r['spread_exec_median']=rec['spread_executed']['median']; r['spread_first_median']=rec['spread_first']['median']; r['bias_exec_median']=rec['bias_executed']['median']; r['clip_exec']=rec['clip_executed']; r['label_clip_exec']=rec['label_clip_executed']
    r['s']=round(time.time()-t0,1); res[tag]=r
    print(tag,json.dumps(r),flush=True)
    json.dump(res,open(OUT/'ol.json','w'),indent=1)
