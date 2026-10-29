from pathlib import Path
from ila.data.store import DemoStore
with DemoStore(Path('/home/kbasak/data/ila/demos/ppo_2000.h5')) as s:
    m=[s.meta(n) for n in s.episodes()]
k=lambda x:x.realization_key
tr={k(x) for x in m if x.demo_index<1000}
ho=[x for x in m if 1950<=x.demo_index<=1999]
print("heldout eps sharing a realization with train 0-999:", sum(k(x) in tr for x in ho), "/", len(ho))
print("episode seeds shared:", len({x.episode_seed for x in ho} & {x.episode_seed for x in m if x.demo_index<1000}))
print("exact (key,t0) shared:", len({(k(x),x.t0_model_s) for x in ho} & {(k(x),x.t0_model_s) for x in m if x.demo_index<1000}))
