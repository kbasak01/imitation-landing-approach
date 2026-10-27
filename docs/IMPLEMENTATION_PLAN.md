# `imitation-landing-approach` — Sequential Implementation Plan

Vision-based imitation learning of a ship-deck approach-and-landing policy. A **privileged,
state-based teacher** from Project 8 (`rl-land-on-moving-deck`: its PPO policy, its
`pid_feedforward` controller and its `gated` controller) is distilled into **camera-based students**
— behaviour cloning, DART, DAgger, a Diffusion Policy and a modular "estimate the state, then run the
teacher" pipeline. Every student is flown on P8's frozen episode lists, under P8's frozen success
criteria and statistics, beside P8's classical baselines. Single RTX A4000 (16 GB) plus the 36-thread
CPU P8 ran on. Builds on P8 as a pinned, read-only submodule; never modifies it.

Plan date: 2026-10-23. External facts were checked on that date and are listed in `SOURCES.md`;
items marked **(verify)** are Phase 0 checks.

---

## 0. Read this first: the five things that decide whether this project works

**(1) The teacher is privileged and the student is not. That information gap *is* the experiment,
and any leak through it voids every number.** P8's 25-d observation contains the relative pad
position and velocity, the deck normal, the relative tilt and the height above the deck: exactly
what a camera has to recover. The student sees **only** (a) rendered camera images and (b) an
explicit proprioception allowlist (`configs/obs/student.yaml`: own attitude, body rates, own
velocity, last action, contact flag). Three leak routes are easy to miss and are tested:

| Leak route | Why it is a leak | Guard |
|---|---|---|
| a privileged field in the proprio vector | the student reads the answer | allowlist by field **name** from P8's `OBS_FIELDS`; a test asserts no privileged name reaches any student input |
| a world-fixed texture in the image (P8's checkerboard ground plane) | it is an inertial reference that does not exist at sea; the student can read its own drift off it | ground plane recoloured to a uniform, untextured colour in render only; a test renders two frames with the drone translated over a static deck and asserts the background pixels are identical |
| `time_fraction` (the episode clock) | descend-on-a-timer is a shortcut that looks like landing | excluded from the student by default; an ablation adds it back and is labelled as such |

**(2) "DAgger beats BC" usually means "more labelled data beats less".** Every on-policy method is
compared with a **data-matched** behaviour-cloning control trained on the same number of labelled
episodes from the teacher alone (`bc_matched`). The comparison that answers the compounding-error
question is `bc_dagger` vs `bc_matched`, never `bc_dagger` vs `bc`. DART gets the same treatment.
Labels for DAgger come from the teacher evaluated at the **student's** states. That is clean for the
stateless PPO teacher. For `pid_feedforward`, whose integrator follows whatever trajectory it is
shadowing, the label is what the PID would command *along the student's path*. Stated, not hidden;
it is one reason the PPO policy is the primary teacher.

**(3) Action chunking meets a moving deck, and a synchronous simulator hides latency.** A Diffusion
Policy predicts `Tp` actions and executes `Ta` of them open loop. At 30 Hz, `Ta = 8` is 267 ms. The
deck's encounter period at model scale is of order 1–2 s (λ = 1/25; P8 `deck_stats.csv`), so an
8-step chunk is a sizeable fraction of a deck cycle, flown blind. Whether that costs anything at
touchdown is hypothesis H3, and it is cheap to test: `Ta` and the number of DDIM steps `K` are
**inference-time** choices on one trained model. Two traps:
- Chunk indexing. Observation window `[t−To+1 … t]`, predicted actions indexed from `t−To+1`,
  executed actions `[t … t+Ta−1]`, so **`Ta ≤ Tp − To + 1`** (with `To = 2`, `Tp = 16`, `Ta ≤ 15`).
  The `il-conventions` skill is the authority; an off-by-one here produces a policy that works
  slightly late, which looks like "chunking hurts".
- PyBullet does not advance while the network thinks. The deployment phase therefore adds a
  **latency-in-the-loop** arm: the measured p99 inference latency becomes a delay of
  `d = ⌈p99 / 33.3 ms⌉` control steps before a new chunk (or a new BC action) takes effect.

**(4) The P8 protocol is inherited, not re-derived — and rendering must not touch the physics.**
Frozen lists (MANIFEST SHA-256 `e6f30e55…`), `configs/env/success.yaml`
(SHA-256 `c7fbdcc4…`), the six outcome classes, Wilson intervals, IQM with stratified bootstrap and
the paired bootstrap are all P8's, imported through one adapter, unchanged. This repo adds a camera
and a visual-only deck marking. The **Phase 2 reproduction gate** flies P8's `pid_feedforward` and
`ppo` seed 0 through *this* repo's runner, with rendering **on**, and requires the episode rows to
equal P8's committed rows in every column. If rendering perturbs a single contact, nothing after it
is comparable with P8. Also inherited: the ceiling. P8's PPO teacher lands 100 % at `id` SS3–SS5,
so those cells can only show a student *losing* to its teacher; the scored cells are SS5 and SS6.
A student that **beats** its own teacher in a cell is not a result until the `results-skeptic` has
looked for an exploit.

**(5) Determinism has limits, and P8 measured them.** P8 found that about 6.6 % of `id` SS6
policy-episodes flip outcome class under a one-ulp perturbation of the policy input
(findings, Phase 8). Batched GPU inference is not bit-reproducible across batch sizes. So:
- the **determinism contract** is bit-identical rows on the CPU fp32 single-sample path, and an
  outcome-flip rate no larger than P8's measured one-ulp floor on the batched GPU path (measured in
  Phase 2, recorded as a protocol entry);
- Diffusion Policy noise is drawn from a per-episode generator seeded by
  `(run_seed, episode_seed, step)`, never from a global stream, so an episode's samples do not
  depend on which worker or batch slot flew it.

**Positioning (do not over-claim).** The closest prior art, Angelis, Bauersfeld, Scaramuzza and
Boukas (arXiv:2605.23717), lands on a moving platform from a downward camera with **PPO** on sparse
keypoints, trained end to end, with no teacher, demonstrations or distillation, on sinusoidal
platform motion. Privileged-teacher → vision-student distillation for flight is established
(Kaufmann et al. RSS 2020; Loquercio et al. Sci. Robotics 2021; Xing et al. arXiv:2403.12203;
Messikommer et al. arXiv:2603.07578), as are DAgger, DART, Diffusion Policy and ACT. ACT has also
been applied to image-based spacecraft GNC (Posadas-Nava et al., arXiv:2509.04628). A search on
2026-10-23 found no controlled IL study of vision-based landing on a seakeeping-driven deck. This
project's contributions:
1. a matched-data comparison of BC, DART, DAgger, Diffusion Policy and a modular estimator pipeline
   on P8's frozen protocol, with P8's classical baselines printed beside them;
2. an action-chunk-length-vs-deck-motion study at fixed model (H3);
3. a decomposition of the student's shortfall into an **imitation gap** (state student vs teacher)
   and a **perception gap** (vision student vs state student);
4. deployment numbers for the visuomotor policy, with the measured latency put back into the loop.

Do not claim novelty for distillation, for Diffusion Policy, or for vision-based landing. P8
withdrew its motion-realism novelty claim (H4); do not resurrect it here.

**Simulation only.** PyBullet's TinyRenderer is a toy rasteriser. No claim about real cameras, real
decks, or real flight is made anywhere.

---

## 1. System design (decided up front, so Claude Code isn't improvising)

### 1.1 Architecture

```
                 P8 (frozen, pinned submodule: external/rl-land-on-moving-deck)
   frozen episode lists ─▶ make_env ─▶ DeckLandingAviary (240 Hz physics, 30 Hz control, DSLPIDControl)
                                              │ full 25-d obs (privileged)          ▲ action (3,) in [-1,1]
                                              ▼                                     │ norm-capped, v_max 1.5 m/s
  P9 (this repo)                      ila.p8_adapter (ONLY importer of `rld`)        │
                                              │                                     │
          ┌───────────────────────────────────┼────────────────────────────┐        │
          ▼                                   ▼                            ▼        │
   ila.render.camera                 ila.obs.split                 teacher (P8)    │
   body-fixed down camera,           privileged block │ proprio     ppo s0 | pid_ff │
   visual twin of the plate,         (5 fields)       │ allowlist   | gated         │
   ground recoloured                                  │                            │
          │ 96×96 RGB                                 │                            │
          └──────────────┬────────────────────────────┘                            │
                         ▼                                                          │
   students: bc_state(MLP) │ bc │ bc_matched │ bc_dart │ bc_dagger │ dp │ modular ──┘
                         │
   ila.eval.runner (vision-aware twin of P8's runner; P8 rows + P9 columns)
                         │
   P8 stats (Wilson, IQM + stratified bootstrap, paired bootstrap, relative p95) ─▶ results/
```

`ila.p8_adapter` is the **only** module that imports `rld`. It exposes: `load_eval_configs`,
`make_env`, `read_list`, `list_paths`, `obs_fields`, `teacher(name, seed)`, `stats` (re-exports of
P8's statistics functions, by reference — a test asserts identity), `success_yaml_sha256`,
`p8_commit`. Everything under `ila/policies/` and `ila/data/` is torch/numpy with no `rld` import.

### 1.2 Observation contract (`configs/obs/student.yaml`)

| P8 field | dims | teacher | student (default) | notes |
|---|---:|:-:|:-:|---|
| `attitude` | 3 | ✔ | ✔ | own roll/pitch/yaw, world; an IMU/AHRS quantity |
| `body_rates` | 3 | ✔ | ✔ | gyro |
| `velocity` | 3 | ✔ | ✔ | own velocity, drone-yaw frame; GNSS/VIO-plausible |
| `rel_position` | 3 | ✔ | ✘ | **privileged** |
| `rel_velocity` | 3 | ✔ | ✘ | **privileged** |
| `deck_normal` | 3 | ✔ | ✘ | **privileged** |
| `rel_tilt` | 1 | ✔ | ✘ | **privileged** |
| `height` | 1 | ✔ | ✘ | **privileged** (above the deck, along its normal) |
| `time_fraction` | 1 | ✔ | ✘ | episode clock; shortcut risk; ablation `+clock` only |
| `last_action` | 3 | ✔ | ✔ | ablation `−last_action` (copycat test, Wen et al. 2020) |
| `in_contact` | 1 | ✔ | ✔ | landing-gear switch; without it the 0.5 s dwell is unlearnable (P8) |

Student proprio dimension = 13 by default. Field names, never slice indices, select the blocks;
offsets come from P8's `OBS_FIELDS` at import time and a test pins them.

### 1.3 Camera and rendering (`configs/render/camera.yaml`)

- **Camera:** body-fixed, looking along body −z (rotates with the drone's attitude), 2 cm below the
  body origin, vertical FOV 90°, near 0.01 m, far 5 m, 96×96 RGB, `ER_TINY_RENDERER`, DIRECT client,
  shadows off, fixed light direction. Rendered after `env.reset()` and after every `env.step()`,
  from the same physics state as the observation it is paired with.
- **Deck marking:** P8's plate may have no textured visual shape. If so, a **visual twin** is
  created: a body with *no collision shape* and mass 0, carrying an original procedural helipad
  texture (ring + "H" + edge band, generated by `ila.render.texture`), re-posed to the plate's
  pose immediately before each render. It never takes part in contact.
- **Background:** the ground plane's visual is recoloured to a uniform colour with no texture
  (render only; its collision shape is P8's and untouched).
- **Appearance-shift arm (evaluation only):** light direction rotated 60°, background hue shifted,
  marking contrast reduced 30 %, camera mount tilted ±2°. Never trained on.
- **Invariant:** rendering, the twin and the recolour change no physics. Tested at two levels: the
  state trajectory of a scripted episode is bit-identical with render on/off, and the Phase 2
  reproduction gate matches P8's committed rows.

### 1.4 Teachers

| ID | Source | Stateful? | Role |
|---|---|:-:|---|
| `T_ppo` | P8 `ppo`, **seed 0**, SB3 checkpoint + frozen `VecNormalize` | no | primary teacher. Seed 0 is fixed by declaration, not chosen by performance |
| `T_pid` | P8 `pid_feedforward` (tuned gains, `configs/control/pid_feedforward.yaml`) | yes (integrator) | secondary teacher: the classical expert |
| `T_mix` | per-episode coin flip between `T_ppo` and P8 `gated` (seeded) | `gated` yes | the multimodality probe (H2): `ppo` commits early, `gated` waits for a quiescent window |

For DAgger labelling speed, `T_ppo` may be evaluated through P8's exported ONNX graph on ORT CPU
(0.046 ms p50 in P8) **only after** label parity ≤ 1e-6 max-abs against the SB3 path on 10 000
recorded observations.

### 1.5 Data: pools, episodes and the demo store

- **Pools (inherited from P8 P3-D2):** `train` = 729 frigate realizations (SS3–SS5; headings 45°,
  135°, 180°; speeds 0/6/12 kn; seed ordinals 0–26); `tune` = 135 (ordinals 27–31). Neither shares a
  realization with any frozen list. SS6 and 90° are never trained on, tuned on, or used for
  selection.
- **Episodes** = `(realization key, episode seed)`; the env draws `t0` and the initial state from the
  seed exactly as P8 does. Demo episodes are dealt round-robin over `train` realizations, stratified by
  sea state, generator seed **20261023**, manifests with SHA-256 in `splits/`.
- **Selection list:** 150 `tune` episodes (50 per SS3/SS4/SS5), fixed in `splits/select.parquet`.
  Closed-loop checkpoint selection uses only this list.
- **Demo store** (`ila.data.store`, HDF5 via h5py, lzf compression, one group per episode):

| Key | dtype / shape | meaning |
|---|---|---|
| `images` | uint8 (T, 96, 96, 3) | frame paired with step t's observation |
| `proprio` | float32 (T, 13) | the allowlisted fields, in the order of `student.yaml` |
| `priv` | float32 (T, 11) | the five privileged fields (estimator target; **never a student input**) |
| `teacher_obs` | float32 (T, 25) | P8's full observation |
| `action_exec` | float32 (T, 3) | the action actually executed (teacher, student, or mixture) |
| `label` | float32 (T, 3) | the teacher's action at `teacher_obs[t]` |
| `t_model_s` | float64 (T,) | model-scale control time |
| attrs | — | realization key, episode seed, teacher ID, β (DAgger), DART σ, outcome class, P8 commit, camera-config SHA-256, `ila` git SHA |

The convention `label[t] = teacher(teacher_obs[t])` and "image[t] and proprio[t] are what the policy
sees before choosing action[t]" is the authority (`il-conventions` skill). The store keeps every
episode; BC training uses only teacher episodes whose outcome is `success` and reports the kept
fraction. DAgger rollouts are kept whatever their outcome: their labels are still correct.

### 1.6 Students and arms

| Arm | Input | Model | Training data (teacher `T_ppo` unless noted) | Seeds |
|---|---|---|---|---:|
| `bc_state` | P8 25-d obs | MLP 2×256, MSE | N = 1 000 demos | 5 |
| `bc` | 2 frames + proprio | encoder + MLP head, MSE | N = 1 000 demos | 5 |
| `bc_matched` | same | same | N = 2 000 demos (the data-matched control) | 5 |
| `bc_dart` | same | same | N = 1 000 noise-injected demos (σ chosen on `select`) | 5 |
| `bc_dagger` | same | same | 1 000 demos + 5 iterations × 200 on-policy episodes = 2 000 | 5 |
| `dp` | 2 frames + proprio | Diffusion Policy (1D conditional U-Net) | N = 1 000 demos | 5 |
| `modular` | 2 frames + proprio | encoder → privileged block (Huber) → teacher MLP unchanged | N = 1 000 demos | 5 |
| `R1_lerobot_dp` *(recommended, not gating)* | same | LeRobot `DiffusionPolicy`, separate venv, evaluated via ONNX | same 1 000 demos | 3 |

**Encoder (all vision arms):** ResNet-18 trunk (torchvision, `weights=None`, BatchNorm → GroupNorm
with 16 groups), 32-keypoint spatial softmax → 64-d per frame, frames concatenated (`To = 2`),
proprio through a 2×64 MLP, concatenated → observation feature. Random crop 96 → 88 at train,
centre crop at eval. Identical encoder across `bc*`, `dp` and `modular`, so architecture is not a
confound.

**BC head:** MLP 2×256, output 3, `tanh`; loss MSE on normalised actions; the env applies its own
norm cap. AdamW 3e-4, weight decay 1e-4, cosine, 30 k steps, batch 256, bf16 autocast, EMA 0.999.

**Diffusion Policy** (defaults follow the reference `train_diffusion_unet_hybrid_workspace.yaml`
where they transfer: `To = 2`, `Tp = 16`, `Ta = 8`, DDPM 100 training steps, ε-prediction,
GroupNorm encoder, EMA power 0.75 max 0.9999, lr 1e-4): ConditionalUnet1D with FiLM conditioning,
`down_dims = [128, 256, 512]` (smaller than the reference's for the A4000 budget), kernel 5,
`squaredcos_cap_v2` schedule from `diffusers`, inference with DDIM, `K = 10` by default. 100 k
steps, batch 128, bf16. Action normalisation: identity (actions already lie in [−1, 1]; tested).
Inference-time variants on the **same** checkpoints: `Ta ∈ {1, 2, 4, 8, 12}`,
`K ∈ {1, 2, 5, 10, 25, 100}`, and ACT-style temporal ensembling (replan every step, weights
`exp(−m·i)`, `m = 0.01`, i = 0 the oldest prediction).

**DART:** Gaussian noise of std σ added to the *executed* action during collection, label = clean
teacher action. σ ∈ {0.05, 0.10, 0.20} (normalised units), chosen by `bc_dart` closed-loop success on
`select` (one seed per σ, three trials total, counted in the tuning log).

**DAgger:** iteration 0 = the 1 000 demos. Iterations 1–5 each roll out the current student on 200
fresh `train` episodes, executing the teacher's action with probability β per step,
β = (0.5, 0.25, 0.125, 0, 0); every visited state is labelled by the teacher; the dataset is
aggregated; the student is warm-started and trained 10 k steps on the aggregate. Total labelled
episodes = 2 000, which is `bc_matched`'s N.

**Modular:** the encoder regresses the 11-d privileged block (normalised; Huber loss); the frozen
teacher MLP then receives `[proprio fields, estimated privileged fields, true time_fraction]` in
P8's field order. `time_fraction` is the episode clock, not deck information; it goes to the
teacher because the teacher was trained with it, and never to an end-to-end student.

**Ablations (3 seeds each):** data scale N ∈ {100, 300} for `bc` and `dp`; teacher `T_pid` for `bc`
and `dp`; teacher `T_mix` for `bc` and `dp` (H2); `−last_action` for `bc` and `dp` (copycat);
`+clock` for `bc` (shortcut probe).

### 1.7 Evaluation

- **Lists:** P8's frozen lists, all of them: `id` SS3–SS6, `unseen_seastate` SS6, `unseen_heading`
  SS3–SS6, `unseen_vessel` SS3–SS6, plus `static`. Core arms fly all 2 800 episodes per seed.
  Ablations fly `id` SS4–SS6 only. The appearance-shift arm flies `id` SS5–SS6.
- **Rows:** every P8 episode column, unchanged and in P8's order, then P9 columns: `arm`,
  `teacher`, `student_seed`, `Ta`, `K`, `ensemble`, `latency_steps`, `render_cfg_sha256`,
  `ckpt_sha256`.
- **Always printed beside every student:** its teacher (P8's committed rows), `pid_feedforward`,
  `pid_track_descend`, and the privileged `oracle_gated`, exactly as P8 prints them.
- **Metrics:** P8 P3-D1 §3, verbatim (success with Wilson CI and N, all six outcome classes,
  closing speed p50/p95, lateral offset, relative tilt p95, time to touchdown, effort, jerk,
  detector disagreement, tunnelling). Added: touchdown-time distribution (for H2), teacher-agreement
  on the student's own states (mean ‖a_student − a_teacher‖ at visited states, a DAgger-style
  diagnostic), and inference calls per episode.
- **Statistics:** P8 P3-D1 §4 functions via the adapter. Bootstrap seed **20261023**; 2 000 replicates
  for IQM CIs, 10 000 for paired contrasts. "Separates" = the 95 % CI excludes 0. Never pooled across
  sea states in a headline. No multiplicity correction; said so.

### 1.8 Hypotheses (drafted here, frozen in Phase 3 as P3-D1 before any student is trained)

Magnitudes below are drafts. Phase 3 may restate them against measured teacher ceilings, **before**
any student number exists, as P8 did with its H1.

- **H1 (compounding error).** `bc_dagger` − `bc_matched` success at `id` SS5 ≥ **5 points**, paired
  bootstrap CI excluding 0. Secondary: the same at `id` SS6, and `bc_dart` − `bc_matched`.
- **H2 (multimodality).** With teacher `T_mix`, at `id` SS4: W₁(touchdown-time distribution of
  `dp` vs `T_mix`) < W₁(`bc` vs `T_mix`), the paired-bootstrap CI of the difference excluding 0.
  Secondary: success drop from `T_ppo` → `T_mix` training is larger for `bc` than for `dp`.
- **H3 (chunk length vs deck bandwidth).** On the same `dp` checkpoints at `id` SS6:
  success(`Ta = 1`) − success(`Ta = 8`) ≥ **5 points**, CI excluding 0. Reported as a curve over
  `Ta ∈ {1, 2, 4, 8, 12}` at SS4–SS6, with temporal ensembling beside it.
- **H4 (where the shortfall comes from).** At `id` SS6: perception gap (`bc_state` − `bc`) >
  imitation gap (teacher − `bc_state`), CI of the difference excluding 0. `modular` vs the best
  end-to-end student is reported two-sided, with no predicted direction.
- **H5 (deployment).** (a) The exported `dp` (K = 10) runs a full replanning call at batch 1 under
  33.3 ms p99 on at least one parity-passing provider. (b) With the measured latency put in the loop,
  `dp` at `id` SS5 is non-inferior to its synchronous self: CI lower bound of the difference ≥ −3
  points.

### 1.9 Stretch (only after Phase 10 passes)

- `dp_dagger` (chunk labels by restoring the PyBullet state and rolling the teacher `Tp` steps).
- ACT (CVAE + transformer, temporal ensembling) on the same encoder.
- One-step samplers: Consistency Policy or flow matching, for the latency axis of H5.
- Exporting the demo store to LeRobotDataset v3 as a public artefact (licence permitting).

---

## 2. Phase plan

Each phase ends with an exit gate. Run `/phase-gate N` before the next phase. Long jobs (demo
collection, training, the evaluation matrix) are launched by the user from commands Claude prepares
with `/train-job`; Claude never runs training in the foreground.

| Phase | Name | Days | GPU | Exit gate in one line |
|---|---|---:|:-:|---|
| 0 | Scaffold, submodule, environments, licences | 0.5 | – | `make smoke` green; `import rld, ila`; P8 checkpoints hashed |
| 1 | Adapter, camera, observation contract ⚠️ the crux | 1 | – | leak tests and render-invariance tests green; `privileged-leak-auditor` CLEAN |
| 2 | Vision runner, P8 reproduction gate, throughput budget | 1 | – | P8 rows reproduced with render on; throughput JSON; determinism contract recorded |
| 3 | Teacher demos, demo store, protocol freeze | 1 | – | demos + manifests committed; P3-D1 frozen with user approval |
| 4 | BC family: `bc_state`, `bc`, `bc_matched`, `bc_dart` | 1.5 | ✔ | all runs selected on `select` only; diagnostician TRUSTWORTHY |
| 5 | DAgger | 1 | ✔ | 5 seeds × 5 iterations; labels audited; data counts match `bc_matched` |
| 6 | Diffusion Policy + inference-time sweeps | 2 | ✔ | smoke overfit passes; 5 seeds; `Ta`/`K` sweeps on `select` only |
| 7 | Modular, multimodality, ablations | 1.5 | ✔ | every ablation run present or its absence explained |
| 8 | Evaluation matrix + hypothesis scoring ⭐ | 1.5 | ✔ | every cell filled from CSV; H1–H5 scored as written |
| 9 | ONNX export, parity, latency, latency-in-the-loop | 1 | ✔ | parity + latency + latency-arm CSVs committed |
| 10 | Validation sweep | 1 | – | every `VALIDATION_CHECKLIST.md` line has a verdict |
| 11 | Publication | 0.5–1 | – | `repo-publication-reviewer`: SHIP |

**Total ≈ 14 working days**, plus about 90 GPU-hours of queued training (estimate; Phase 4 measures
it). **Fast path (≈ 7 days):** 0 → 1 → 2 → 3 → 4 (`bc_state`, `bc` only) → 6 (`dp`, 3 seeds, `Ta`
sweep only) → 8-lite (`id` only) → 11. That still yields the perception-gap decomposition (H4) and
the chunking result (H3).

---

### Phase 0 — Scaffold, submodule, environments, licences (0.5 day)

1. `git init imitation-landing-approach`; copy this starter kit in (`CLAUDE.md`, `docs/`,
   `REPO_SETUP.md`, `SOURCES.md`, `VALIDATION_CHECKLIST.md`, `KICKSTART.md`, `configs/`, `.claude/`).
2. `git submodule add https://github.com/kbasak01/rl-land-on-moving-deck external/rl-land-on-moving-deck`;
   check out P8's release commit (`7237b7065c8af8172b3848ac6e9d3ceb2cf82075`, the merge of PR #9,
   "Gate 9 passed", or a release tag on it if one exists — **(verify)**); `git submodule update
   --init --recursive` (P8 itself pins `deck-motion-forecast` and `gym-pybullet-drones`). Commit the pin.
3. Environment per `REPO_SETUP.md`: Python 3.12, P8's exact pins (torch 2.13.0, gymnasium 1.3.0,
   stable-baselines3 2.9.0, pybullet 3.2.7, onnxruntime-gpu 1.29.0, …), editable installs of P8's two
   submodules and of P8 itself (`--no-deps`), then this repo's additions (`diffusers`, `h5py`,
   `torchvision` matching torch). `diffusers` 0.41.0 is current **(verify that it resolves against
   torch 2.13.0 and P8's numpy)**. LeRobot is **never** installed in the main venv; it gets
   `.venv-lerobot` for R1 only.
4. Package skeleton `src/ila/` (layout in `REPO_SETUP.md`), `pyproject.toml` (ruff + mypy strict +
   pytest markers `gpu`, `slow`, `checkpoints` — P8's config), `Makefile` with stubs, CPU CI with
   `submodules: recursive`.
5. `configs/paths.local.yaml` (gitignored, from the example): P8 `artifacts/runs/` location (PPO
   checkpoints live in P8's gitignored `artifacts/`, on the workstation only), demo-store root.
6. `scripts/verify_p8.py`: P8 commit, `results/episodes/MANIFEST.csv` SHA-256 = `e6f30e55…`,
   `success.yaml` SHA-256 = `c7fbdcc4…`, and SHA-256 of the `ppo` seed-0 checkpoint +
   `VecNormalize` stats → `results/p8_provenance.json`.
7. `docs/LICENSES.md` from `SOURCES.md`: P8 MIT, gym-pybullet-drones MIT, PyBullet zlib, diffusers
   Apache-2.0, LeRobot Apache-2.0, diffusion_policy MIT (any adapted U-Net code carries its notice in
   `THIRD_PARTY_NOTICES.md`).

**Exit gate:** `make lint test smoke` green locally and in CI (CI skips `checkpoints`-marked tests);
`python -c "import rld, ila"`; `results/p8_provenance.json` matches P8's published hashes.

---

### Phase 1 — Adapter, camera, observation contract (1 day) ⚠️ the crux

Every student number depends on two things this phase builds: what the student is allowed to see,
and a renderer that does not change the world it renders.

1. `ila/p8_adapter.py` (§1.1). Re-exports P8's statistics functions by reference; a test asserts
   `ila.p8_adapter.stats.wilson is rld.eval.stats.<fn>` (names from P8's source, not from memory).
2. `ila/obs/split.py`: `split(obs) -> (proprio, priv, extras)` driven by `configs/obs/student.yaml`
   field names; `StudentInput` dataclass that **has no field** able to hold `priv`. Tests:
   - every P8 field is classified exactly once (teacher-only, student, privileged);
   - offsets and sizes equal P8's `OBS_FIELDS`;
   - a student forward pass raises if handed a 25-d vector;
   - `+clock` and `−last_action` ablations change exactly one field each.
3. `ila/render/texture.py` (procedural helipad PNG, deterministic), `ila/render/camera.py`
   (body-fixed camera from the drone's physics pose, intrinsics from config), `ila/render/scene.py`
   (visual twin if needed — inspect P8's plate with `getVisualShapeData` first and record what you
   find in `docs/protocol.md` P1-D1 —, ground recolour, appearance-shift variants).
4. Tests:
   - **render invariance:** a scripted 300-step episode (fixed action sequence) on a moving deck:
     full physics state (drone, plate) bit-identical with rendering off vs on vs on+twin+recolour;
   - **no inertial reference:** drone translated 5 cm over a static deck, background pixels outside
     the plate's projection identical;
   - **geometry:** a marker of known pose projects to the expected pixel within 0.5 px (pinhole
     check from the view and projection matrices); a 10° roll of the drone moves the deck image the
     right way; deck tilt is visible as a perspective change;
   - **timing:** the image at step t is rendered from the same `stepSimulation` count as `obs[t]`;
   - determinism: the same episode rendered twice gives byte-identical frames.
5. `docs/CONVENTIONS.md` = the `il-conventions` skill; keep them identical.

**Exit gate:** all tests green; `privileged-leak-auditor` verdict **CLEAN**; P1-D1 (plate visuals,
twin or not, camera numbers) recorded.

---

### Phase 2 — Vision runner, P8 reproduction gate, throughput budget (1 day)

1. `ila/eval/runner.py`: a vision-aware twin of P8's runner. Same contract (the list is the only
   source of episodes; start state checked on every reset; rows independent of worker count),
   reusing P8's public functions (`read_list`, `make_env`, `EPISODE_COLUMNS`, episode-record →
   row). Rendering in the worker; policy inference either in-worker (CPU path) or in a batched GPU
   server process fed by lockstep workers (`ila/eval/batched.py`). Wherever P8's runner uses a
   private helper, re-implement it and **test it equal** to P8's on P8's own fixtures.
2. **Reproduction gate** (`scripts/reproduce_p8.py`):
   - `pid_feedforward` on all 2 800 listed episodes, render **on**: every row equal to P8's committed
     `results/e01/` rows, every column, compared as text (P8's own comparison rule);
   - `ppo` seed 0 on `id` SS5 and SS6 (400 episodes), render on: equal to P8's committed `e07` rows,
     using P8's exact inference path (SB3, CPU, thread count from P8's run_info).
   A miss stops the project here. First suspects: thread count, a visual body changing Bullet's
   object ordering, the render being taken between physics substeps.
3. **Teacher label parity:** `T_ppo` via P8's ONNX graph on ORT CPU vs SB3 on 10 000 recorded
   observations, max-abs ≤ 1e-6, or the SB3 path is used for labels.
4. **Throughput** (`scripts/throughput.py` → `results/throughput.csv`): render cost per frame at
   64/96/128 px; steps/s for teacher-with-render at 1/8/16/24 workers; batched GPU inference cost for
   a dummy `bc` and a dummy `dp` at K = 10 and K = 100, `Ta` = 1 and 8. From it, project wall clock
   for demo collection, DAgger iterations and the evaluation matrix, and write the projection into
   P2-D-budget. If 96 px is too slow, drop to 64 px **now**, for every arm, by a dated decision.
5. **Determinism contract:** fly a dummy `dp` with batch sizes 1, 8 and 24 on 200 `select` episodes:
   rows bit-identical on CPU single-sample; on the GPU batched path record the outcome-flip count
   against P8's one-ulp floor. Write the contract as P2-D-determinism.

**Exit gate:** both reproduction comparisons pass with zero differing rows; label parity met (or
SB3 path chosen); throughput CSV committed; budget and determinism decisions recorded.

---

### Phase 3 — Teacher demos, demo store, protocol freeze (1 day)

1. `ila/data/episodes.py`: train/select episode generators (seed 20261023), manifests + SHA-256 in
   `splits/`; tests for disjointness from every frozen list (by realization key) and from `tune` (for
   train) and for the absence of SS6 and 90°.
2. `ila/data/collect.py` + `/train-job collect-<teacher>-<N>`: `T_ppo` N = 2 000 (the first 1 000
   are the `bc` set; all 2 000 are `bc_matched`'s), `T_pid` N = 1 000, `T_mix` N = 1 000, DART sets
   for the three σ values. Store per §1.5 → `data/` (gitignored); commit `results/demos_summary.json`
   (counts, teacher success rates on train episodes, kept fraction, frames, bytes, SHA-256 of each
   file).
3. Data checks (run by `demo-data-engineer`, reviewed by `privileged-leak-auditor`): image/obs alignment on 20
   random steps (re-render from saved state → identical frame), label = teacher(teacher_obs) on
   1 000 random steps, no episode from a frozen list or from `tune`.
4. **Draft P3-D1 in `docs/protocol.md`** — the frozen IL protocol: arms (§1.6) with every
   hyper-parameter, seeds, the selection rule (`select` success, ties by p95 closing speed, then
   lowest step), tuning budgets (DART 3 trials; no other tuning), data counts, the
   evaluation lists per arm (§1.7), metrics, statistics, the hypotheses (§1.8) with cells and
   magnitudes restated against the measured teacher ceilings, the latency-arm rule, and the
   training budget rule (below). **Show it to the user and wait for approval.** Hash the block.
5. **Budget rule (pre-registered):** the first `dp` seed's wall clock is reported; if the projected
   total GPU time exceeds 120 h, training steps are cut **once**, by the same factor for every arm of
   that family, before any evaluation number is read.

**Exit gate:** demos and manifests complete; audits clean; P3-D1 frozen with the user's dated
approval and its SHA-256 in the Gate 3 entry.

---

### Phase 4 — BC family (1.5 days, GPU)

1. `ila/policies/encoder.py`, `bc.py`, `state_mlp.py`; `ila/train/loop.py` (bf16 autocast, AdamW,
   cosine, EMA, gradient-norm logging, deterministic dataloader seeding), `ila/train/dataset.py`
   (memory-mapped HDF5 reader, frame stacking with first-frame padding, crop augmentation).
2. `make train-smoke ARM=bc`: overfit 8 episodes to action MSE < 1e-4, and fly them closed loop: the
   student must land ≥ 7/8 of its own training episodes. A student that cannot land the episodes it
   memorised has an alignment bug, not a capacity problem.
3. `/train-job` for `bc_state`, `bc`, `bc_matched`, `bc_dart` (σ selection first, 3 trials, logged
   as tuning). Checkpoints every 5 k steps; selection on `select` closed-loop success only.
4. `training-run-diagnostician` on every run: loss curves (train vs held-out `tune` demos), EMA
   vs raw, action histograms vs teacher, saturation at the norm cap, copycat signature (action at t
   ≈ last_action much more often than the teacher's), teacher-agreement on the student's own states.

**Exit gate:** every run present with a manifest (config SHA, data SHA, checkpoint SHA, wall
clock); diagnostician TRUSTWORTHY on each, or the run is excluded with a reason in the protocol;
measured GPU-hours per run written into the budget entry.

---

### Phase 5 — DAgger (1 day, GPU)

1. `ila/data/dagger.py`: iteration loop (§1.6); each iteration writes its rollouts to the store with
   β, the student checkpoint SHA and the iteration index; aggregate manifests.
2. Label audit per iteration: 1 000 random visited states re-labelled from `teacher_obs` → identical.
   Count check: labelled episodes = 2 000 per seed, matching `bc_matched`.
3. Report per iteration: on-`train` rollout success, teacher-agreement on student states, and
   `select` success. A DAgger curve that rises on `train` but not on `select` is a finding.

**Exit gate:** 5 seeds × 5 iterations complete; audits clean; final checkpoints selected on
`select` only.

---

### Phase 6 — Diffusion Policy and inference-time sweeps (2 days, GPU)

1. `ila/policies/dp/` — `unet1d.py` (ConditionalUnet1D with FiLM; adapted from the MIT reference,
   notice kept), `policy.py` (conditioning, DDPM training, DDIM sampling with an explicit noise
   input and a per-episode generator), `chunking.py` (the `To`/`Tp`/`Ta` window logic and temporal
   ensembling — **the only place indices are computed**).
2. Tests (CPU): window indices against hand-written tables for `To ∈ {1, 2}`, `Ta ∈ {1, 8, 15}`;
   `Ta > Tp − To + 1` refused; first-step padding; ensembling weights; DDIM with K = num_train_steps
   matches DDPM in distribution on a 1-D toy (two-Gaussian action mixture recovered; BC-MSE on the same
   toy collapses to the mean — this is the H2 mechanism in miniature).
3. `make train-smoke ARM=dp` (overfit 8 episodes; closed-loop ≥ 7/8 at `Ta = 8`, K = 10), then
   `/train-job` for 5 seeds. Selection on `select` at the **default** `Ta = 8`, K = 10.
4. Sweeps on `select` only (reported, never used to re-select the checkpoint): `Ta`, K, ensembling.
5. `training-run-diagnostician`: denoising loss by diffusion step, EMA vs raw, sample diversity
   (per-state sample spread vs teacher multimodality), chunk-boundary jerk.
6. **R1 (recommended):** in `.venv-lerobot`, export the store to LeRobotDataset v3, train LeRobot's
   `DiffusionPolicy` with matched `To/Tp/Ta`, export to ONNX with noise as an input, evaluate through
   this repo's runner. Tagged `reference_external`. A large gap between R1 and `dp` is investigated
   before Phase 8.

**Exit gate:** CPU tests green; 5 seeds trained and selected; diagnostician TRUSTWORTHY; sweep JSON
committed.

---

### Phase 7 — Modular, multimodality, ablations (1.5 days, GPU)

1. `ila/policies/modular.py`: estimator + frozen teacher; per-field estimation error (RMSE, p95) on
   held-out `tune` demos and on the student's own closed-loop states, compared with P8's σ_p = 4 cm
   noise arm magnitude (P8 found its PID baselines collapse at that σ; say where the estimator
   falls).
2. `T_mix` arms (`bc`, `dp`), `T_pid` arms, data-scale arms, `−last_action`, `+clock` — §1.6, 3
   seeds each, same selection rule.
3. For H2: touchdown-time distributions of `T_mix`, `bc`-mix and `dp`-mix on `select`, plotted only;
   the scored cell is `id` SS4 in Phase 8.

**Exit gate:** every planned run present with a manifest, or its absence explained in the protocol;
diagnostician verdicts recorded.

---

### Phase 8 — Evaluation matrix and hypothesis scoring (1.5 days) ⭐

1. `scripts/eval_matrix.py` → `results/e08/<arm>/<seed>/<list>.csv.gz` (episode rows) and
   `results/e08/summary.csv`, `aggregate.csv` (IQM + CI), `contrasts.csv`, `hypotheses.csv`, with
   `run_info.json` (commit, P8 commit, list hashes, render SHA, wall clock, workers).
2. Arms (each tagged): core × all lists; `dp` at `Ta ∈ {1, 2, 4, 8, 12}` and ensembling on `id`
   SS4–SS6; K sweep on `id` SS5; appearance shift on `id` SS5–SS6 for `bc_dagger`, `dp`, `modular`;
   ablations on `id` SS4–SS6.
3. Score H1–H5(b) exactly as written in P3-D1 (H5(a) is scored in Phase 9). Supported / not
   supported / inconclusive, with the number and CI. Nulls go in the README body.
4. `scripts/report.py` → `results/results.md`, re-rendered byte-identically by `make report`;
   `scripts/make_figures.py`: success vs sea state per arm with the teacher and baselines; the
   imitation/perception gap bars (H4); success vs `Ta` per sea state (H3); touchdown-time histograms
   (H2); a 4-panel GIF gallery (teacher / `bc` / `bc_dagger` / `dp`) on hand-picked episodes,
   **labelled as hand-picked**, each re-flown and checked against its committed row.
5. `eval-auditor` builds the tables; `results-skeptic` reviews before the gate.

**Exit gate:** every cell filled from CSV, or the empty cell explained; hypotheses scored as written;
`results-skeptic` reports nothing BLOCKING.

---

### Phase 9 — ONNX export, parity, latency, latency-in-the-loop (1 day, GPU)

1. Export (opset 17, fp32 and fp16): `bc_dagger` (encoder + head, one graph), `dp` as
   `dp_encoder.onnx` + `dp_denoise_step.onnx` (inputs: noisy chunk, step index, conditioning) and an
   unrolled `dp_K10.onnx` (inputs: conditioning **and the initial noise**, so it is deterministic).
2. Parity on 1 000 fixed `select` inputs, TF32 off: fp32 max-abs ≤ 1e-4 per denoise step and ≤ 1e-3
   on the final chunk; fp16 reported. Closed-loop parity on 50 `id` SS5 episodes judged against the
   Phase 2 determinism contract — not against bit-identity.
3. Latency (P8's harness methodology: ≥ 200 warm-up, ≥ 2 000 timed, synchronised brackets, batch 1
   and the eval batch size): encoder; one denoise step; the full K = {1, 5, 10, 25} loop; end-to-end
   image → action (BC) and image → chunk (DP); providers torch-eager CUDA, ORT CUDA EP, ORT TensorRT
   EP fp16, ORT CPU. P8 found ORT CPU fastest for its tiny MLP; do not assume either way here.
   H5(a) is scored from this table.
4. **Latency-in-the-loop arm:** `d = ⌈p99 / 33.3 ms⌉` from the fastest parity-passing provider;
   async semantics as §0(3); flown on `id` SS4–SS6 for `bc_dagger` and `dp` (`Ta = 8`, K = 10);
   H5(b) scored.
5. `deploy-benchmarker` owns it; `onnx-parity` checks inside `results-skeptic`'s review.

**Exit gate:** `results/latency/` with parity, latency, latency-arm CSVs and `run_info.json`
committed; H5 scored.

---

### Phase 10 — Validation sweep (1 day)

Walk `VALIDATION_CHECKLIST.md` line by line; verdict + evidence (met / met-with-caveat / unmet) into
`docs/LIMITATIONS.md`. Run `/full-audit`. Re-run `make report` and every `--check` mode; diff.

**Exit gate:** every line has a verdict; no critical finding open.

### Phase 11 — Publication (0.5–1 day)

README in P8's style: status line with every gate's honest history; "the short version" with every
hypothesis verdict; headline figure (success vs sea state, teacher and baselines beside the
students); the gap decomposition; the chunking curve; latency table; "what a rendered camera can and
cannot show" box; limitations before reproduction; citations. Release assets: student checkpoints,
ONNX graphs, `SHA256SUMS.txt`. **No demo images in git** (the store is gitignored; a 256-frame
sample may ship as a release asset). `repo-publication-reviewer`: SHIP. Tag `phase-11-complete`.

---

## 3. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Rendering or the visual twin perturbs physics; P8 rows not reproduced | medium | high | Phase 1 bit-identity tests; Phase 2 gate; render strictly after `step`; twin has no collision shape |
| TinyRenderer too slow for DAgger + the matrix | medium | high | Phase 2 throughput budget; 64 px fallback decided once, for all arms; render in parallel workers |
| Vision students far below teacher everywhere (no room for H1/H3) | medium | medium | that is H4's finding; H1/H3 scored at SS4 if SS5 floors — only by a P3-D1 rule written before training |
| Students at the ceiling at SS5 (no room) | medium | medium | SS6 secondaries pre-registered; P8's ceiling lesson |
| Chunk-index off-by-one masquerades as "chunking hurts" | medium | high | `chunking.py` is the single source; table tests; smoke closed-loop landing of memorised episodes |
| DAgger gain is just data volume | high (if unchecked) | high | `bc_matched` control is part of H1 |
| Leak via proprio, background texture or clock | medium | high | allowlist by name; recoloured ground; tests; `privileged-leak-auditor` before Gates 1, 3, 8 |
| GPU nondeterminism blurs SS6 contrasts | high | medium | determinism contract (Phase 2), per-episode DP noise, P8's ulp floor quoted beside every SS6 contrast |
| LeRobot cannot coexist with P8's pins | high | low | separate venv; only ONNX crosses; R1 is not gating |
| DP training exceeds budget | medium | medium | pre-registered single cut; smaller U-Net already chosen |
| Over-claiming vision realism or sim-to-real | medium | high | §0, `claim-trace` skill word list, `results-skeptic`, README box |

## 4. Calendars

- **Full (≈ 14 days + queued GPU):** D1 P0 + P1(start) · D2 P1 · D3 P2 · D4 P3 (collection overnight)
  · D5–6 P4 · D7 P5 · D8–9 P6 · D10 P7 · D11–12 P8 · D12–13 P9 + P10 · D14 P11. Training queues
  overnight from D5 on; Phases 7 and 8 overlap with the tail of the queue.
- **Fast (≈ 7 days):** D1 P0 + P1 · D2 P2 · D3 P3 · D4 P4-lite · D5 P6-lite · D6 P8-lite · D7 P11
  trimmed.

## 5. How this feeds the ship-deck work

- The teacher → student pipeline is the practical route for your real system: your classical
  pose-based controller is a privileged teacher in simulation, and the camera policy is what flies.
- The gap decomposition says where to spend effort: if the perception gap dominates, better state
  estimation (P1/P6's pose and uncertainty work) beats better imitation.
- The chunking result sets the replanning rate a deck-landing policy needs at a given sea state.
- The latency-in-the-loop arm is the template for the Jetson question: measure, put it back in the
  loop, and report what it costs.
- The modular arm is a direct hook for P6: replace the point estimator with a conformal set and
  wave off when the set is larger than the deck's tolerance.
