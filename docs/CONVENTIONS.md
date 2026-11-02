
# IL conventions (mirrored verbatim in `docs/CONVENTIONS.md`; keep them identical)

## Time and scale
- Control step k at 30 Hz: `t_k = t0 + k / 30` s **model scale** (λ = 1/25: 1 model s = 5 full-scale s).
- Physics 240 Hz: 8 `stepSimulation` calls per control step, inside P8's `env.step`.
- Episode ≤ 12 s + 0.5 s dwell grace, model scale (≤ 375 control steps).
- Lengths metres model scale; angles radians inside code, degrees at config/report boundaries.

## Frames (inherited from P8)
- World: x = bow, y = port, z = up.
- Drone-yaw frame: world rotated by −yaw about z (P8's relative-position frame; not the body frame).
- Body frame: the drone's attitude. The camera is body-fixed, looking along body −z, image "up" =
  body +x.

## Observation split (by P8 field name; `configs/obs/student.yaml`)
- Student proprio (13): `attitude 3, body_rates 3, velocity 3, last_action 3, in_contact 1`.
- Privileged (11, estimator target only): `rel_position 3, rel_velocity 3, deck_normal 3,
  rel_tilt 1, height 1`.
- Teacher-only: `time_fraction 1`.
- `StudentInput` = `(images: uint8[To, H, W, 3], proprio: float32[To, 13])`. It has no field for
  privileged data. A student `act(obs)` with a 25-d vector raises.
- Only `rgb` (`ila.render.camera.Frame`, what `Scene.render()` returns) and the allowlisted proprio
  may enter a student input, a demo store's student columns or a training batch. `RenderMeta`
  (view/projection matrices = the drone's world pose, the physics step count = the clock, the
  segmentation mask = a perfect deck mask) never does, and neither does the store's `t_model_s`.

## Alignment (the rule every dataset and runner follows)
At control step t:
1. `obs[t]` and `image[t]` come from the state after the t-th `env.step` (t = 0: after `reset`);
   same `stepSimulation` count.
2. The policy sees `image[t−To+1 … t]`, `proprio[t−To+1 … t]` (indices < 0 → repeat index 0).
3. It outputs `action_exec[t]`, which `env.step` applies to produce `obs[t+1]`.
4. `label[t] = teacher(teacher_obs[t])`, where a stateful teacher has been `reset` at t = 0 and
   called once per step in order, along the trajectory actually flown.
5. `last_action` inside `proprio[t]` is `action_exec[t−1]` (zero at t = 0, as P8 defines it).

## Action
- Shape (3,), normalised world-frame velocity setpoint in [−1, 1]; the env scales by v_max = 1.5 m/s
  with a **norm** cap. Students output `tanh` (BC) or clipped samples (DP); the env applies the cap.
- DART noise is added to `action_exec`, never to `label`.

## Chunks (Diffusion Policy)
- `To` observation steps, prediction horizon `Tp`, executed `Ta`.
- At replanning step t the model predicts `â[t−To+1 … t−To+Tp]` (chunk index j ↔ step t−To+1+j).
- Executed: chunk indices `To−1 … To−2+Ta` ↔ steps `t … t+Ta−1`. Constraint: **Ta ≤ Tp − To + 1**
  (To = 2, Tp = 16 ⇒ Ta ≤ 15).
- Training targets for a sample anchored at t: `label[t−To+1 … t−To+Tp]`; steps < 0 repeat
  `label[0]`; steps ≥ T repeat `label[T−1]`.
- Replanning happens at t, t+Ta, t+2Ta, … (without ensembling).
- **Temporal ensembling:** replan every step; for step t average the predictions made for t by
  the replans s ∈ [max(0, t−Ta+1), t] (those whose executed window covers t; chunk index
  To−1+(t−s)), weight `w_i = exp(−m·i)`, i = 0 the oldest, m = 0.01, normalised. With Ta = 1
  this is no ensembling (P6-D1).
- Only `ila/policies/dp/chunking.py` computes these indices.

## Latency-in-the-loop semantics
A decision computed from `obs[t]` takes effect at step `t + d`. Until then the previous chunk keeps
executing (DP) or the previous action is held (BC). `d = ceil(p99_ms / 33.333)`.

## Randomness
- Diffusion initial noise and per-step noise: `torch.Generator` seeded by
  `hash64(run_seed, episode_seed, t)`; never a global RNG. Graph exports take noise as an input.
- Data loading: worker seeds from `(run_seed, epoch, worker_id)`.
- **The clock and the episode seed never enter a network.** The control step t and the episode
  seed reach a student only as a noise key and a schedule index. They may (a) seed the per-request
  `torch.Generator` via `noise_seed(run_seed, episode_seed, t)`, (b) feed `chunking`
  (`is_replan_step`, ensembling indices), and (c) drive slot bookkeeping. They never enter an
  `nn.Module` forward, a feature, a conditioning vector, a normaliser or a training batch. Besides
  pixels and proprio, the only integer a network sees is the diffusion index
  k ∈ [0, num_train_timesteps). DDPM training noise comes from the data-loading generator
  (run_seed, epoch, worker_id), never from `noise_seed(·, ·, t)`.
- **Three declared exceptions** (P3-D1 §3), each labelled wherever it is reported: (1) the `+clock`
  ablation adds `time_fraction` to the student's proprio; (2) `modular` passes the true
  `time_fraction` to the frozen teacher MLP only, through a `ModularInput` built in
  `ila/obs/split.py`, never to the estimator; (3) `bc_state` reads P8's full 25-d observation,
  `time_fraction` included (privileged by design, never a vision row). No other path carries the
  clock into a network.

## Demo store keys (HDF5, one group per episode)
`images uint8[T,96,96,3]`, `proprio f32[T,13]`, `priv f32[T,11]`, `teacher_obs f32[T,25]`,
`action_exec f32[T,3]`, `label f32[T,3]`, `t_model_s f64[T]`; attrs: realization key, episode seed,
teacher, β, DART σ, outcome class, P8 commit, camera-config SHA-256, `ila` commit.

## Smoke-test checklist when a student "can't land"
1. Print steps 0–3 of one stored episode: `image[t]` thumbnail, `proprio[t]`, `label[t]` beside a
   fresh teacher rollout of the same episode. They must match exactly.
2. Is `last_action` in `proprio[t]` equal to `action_exec[t−1]`?
3. Are executed chunk indices `To−1 …` (not `0 …`)?
4. Is the crop centred at eval and random at train only?
5. Is the EMA model the one being evaluated?
