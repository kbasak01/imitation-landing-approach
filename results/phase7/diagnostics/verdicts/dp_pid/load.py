import json, glob, os
import pandas as pd, numpy as np
R='/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs'
def load(arm):
    out=[]
    for sd in sorted(glob.glob(f'{R}/{arm}/s[0-9]')):
        sel=json.load(open(f'{sd}/selection.json'))
        selstep=sel['selected']['candidate']['global_step']
        for f in sorted(glob.glob(f'{sd}/select/*_rows.csv')):
            d=pd.read_csv(f)
            step=int(os.path.basename(f).split('step')[1][:6])
            d['seed']=int(os.path.basename(sd)[1:]); d['step']=step; d['selected']=(step==selstep)
            out.append(d)
    return pd.concat(out,ignore_index=True)
def band(td):
    return np.where(td<1.8,'fast',np.where(td>2.8,'slow','between'))
