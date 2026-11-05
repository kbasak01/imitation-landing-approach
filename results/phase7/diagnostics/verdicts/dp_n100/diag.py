import sys, time, json, torch
from pathlib import Path
torch.set_num_threads(4)
from ila.train.dp_diagnostics import run_diagnostics
out=Path(sys.argv[1])
for s in (0,1,2):
    t=time.time()
    rec=run_diagnostics(Path(f'/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs/dp_n100/s{s}'), closed_loop=False, out_root=out, workers=1)
    h=rec['heldout']
    print(s, round(time.time()-t), json.dumps({k:h[k] for k in ('spread_executed','spread_first','bias_executed','clip_executed','label_clip_executed')}), h['open_loop_jerk']['d1_ratio_boundary_over_within'], h['teacher_label_jerk']['d1_ratio_boundary_over_within'], flush=True)
