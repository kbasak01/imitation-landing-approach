# SOURCES.md — what this project depends on, checked 2026-10-23

Each entry: what it is, what this project uses it for, licence, and how it was checked. (each such item is resolved in [`docs/LICENSES.md`](docs/LICENSES.md) §2)
items are Phase 0 checks; resolve them in `docs/LICENSES.md` and protocol P0-D1.

## 1. Your own projects

| Source | Used for | Licence | Checked |
|---|---|---|---|
| **P8 `rl-land-on-moving-deck`** — <https://github.com/kbasak01/rl-land-on-moving-deck>, commit `7237b7065c8af8172b3848ac6e9d3ceb2cf82075` (merge of PR #9, "Gate 9 passed") | environment, frozen episode lists (MANIFEST `e6f30e55…`), success YAML (`c7fbdcc4…`), statistics, teachers (`ppo` seed 0, `pid_feedforward`, `gated`), committed episode rows for the reproduction gate, P8's ONNX `ppo` graph for fast labels | MIT | cloned and read 2026-10-23 (README, CLAUDE.md, protocol P3-D1/P3-D2, `observation.py`, `runner.py`, configs). No release tag exists on that commit; it is pinned by SHA (`docs/LICENSES.md` §2) |
| **P4 `deck-motion-forecast`** (via P8's submodule) | JONSWAP deck motion; read-only | MIT (per P8 notices) | via P8 |
| P8 trained checkpoints (`artifacts/runs/ppo/<seed-0 run>/`, path recorded in `paths.local.yaml`) | the PPO teacher | yours | gitignored in P8; present only on the workstation — Phase 0 hashes them |
| **P1 `spacecraft-pose-baseline`, P6 `pose-with-error-bars`** | design patterns only (adapter boundary, parity gates, auditors); no code import | MIT | — |

Facts from P8 this plan relies on (all from its README / protocol):
- 25-d observation; fields `attitude, body_rates, velocity, rel_position, rel_velocity,
  deck_normal, rel_tilt, height, time_fraction, last_action, in_contact` (`src/rld/envs/observation.py`).
- One action space: world-frame velocity setpoint, `v_max` 1.5 m/s norm cap, 30 Hz, tracked by
  `DSLPIDControl`; physics 240 Hz; episodes ≤ 12 s + 0.5 s dwell grace, model scale; λ = 1/25.
- `ppo` 98.2 % and `pid_feedforward` 90.5 % at `id` SS6; PPO family at 99.5–100 % in SS3–SS5.
- `train` pool 729 realizations, `tune` 135 (P3-D2); SS6 and 90° never trained on.
- About 6.6 % of `id` SS6 policy-episodes flip under a one-ulp input perturbation (Phase 8).
- Under σ_p = 4 cm perception noise, `pid_feedforward` falls to 0–0.5 % (P7 noise arm).
- P8 renders GIFs with `getCameraImage` / `ER_TINY_RENDERER` in a DIRECT client (`src/rld/viz/gifs.py`).
- P8's runner hands a policy only the observation vector (`EvalPolicy.act(obs)`), so this project
  needs its own vision-aware runner (hence the Phase 2 reproduction gate).

## 2. Software

| Package | Version / status (2026-10-23) | Use | Licence |
|---|---|---|---|
| PyTorch | 2.13.0 (P8 pin) | everything | BSD-3 |
| torchvision | 0.28.0+cu130, matching torch 2.13.0 (`docs/LICENSES.md` §2) | ResNet-18 trunk | BSD-3 |
| diffusers | 0.41.0, released 2026-10-06, Python ≥ 3.10 ([PyPI](https://pypi.org/project/diffusers/)) | DDPM/DDIM schedulers | Apache-2.0 |
| gymnasium / SB3 / pybullet | 1.3.0 / 2.9.0 / 3.2.7 (P8 pins) | env, PPO teacher loading | MIT / MIT / zlib |
| gym-pybullet-drones | P8's submodule (`learnsyslab/gym-pybullet-drones` at `7ebad1ec…`) | the aviary | MIT |
| onnx / onnxruntime-gpu / tensorrt-cu13 | 1.22.0 / 1.29.0 / 10.16.1.11 (P8 pins) | export + latency | Apache-2.0 / MIT / NVIDIA licence |
| h5py | resolve in Phase 0 | demo store | BSD-3 |
| **LeRobot** | 0.6.1, released 2026-08-03, Python ≥ 3.12; policies include ACT and Diffusion ([PyPI](https://pypi.org/project/lerobot/)) | reference arm R1 only, separate venv; dataset format v3 (`CODEBASE_VERSION = "v3.0"`, `docs/LICENSES.md` §2) | Apache-2.0 |
| **diffusion_policy** (reference implementation) — <https://github.com/real-stanford/diffusion_policy> | defaults read from `train_diffusion_unet_hybrid_workspace.yaml`: `horizon 16`, `n_obs_steps 2`, `n_action_steps 8`, DDPM `num_train_timesteps 100`, ε-prediction, `obs_encoder_group_norm True`, `crop_shape [76, 76]`, EMA power 0.75 / max 0.9999, lr 1e-4, batch 64 | architecture reference; ConditionalUnet1D may be adapted with notice | MIT |

## 3. Literature (cite in README; do not over-claim against)

**Closest prior art**
- A. Angelis, L. Bauersfeld, D. Scaramuzza, N. Boukas, *Vision-Based Agile Landing on Turbulent
  Waters*, arXiv:2605.23717. Read 2026-10-23: PPO with an SRPose attention encoder on 25 sparse
  keypoints from a downward camera, trained in Aerial Gym; **no imitation learning, teacher or
  distillation**; platform motion sinusoidal per DoF; 30 Hz pipeline; hardware on a Stewart platform
  and a pool. Code release not mentioned.

**Privileged teacher → sensor-based student (flight and driving)**
- D. Chen, B. Zhou, V. Koltun, P. Krähenbühl, *Learning by Cheating*, CoRL 2019.
- E. Kaufmann, A. Loquercio, R. Ranftl, M. Müller, V. Koltun, D. Scaramuzza, *Deep Drone Acrobatics*,
  RSS 2020.
- A. Loquercio, E. Kaufmann, R. Ranftl, M. Müller, V. Koltun, D. Scaramuzza, *Learning High-Speed
  Flight in the Wild*, Science Robotics 2021.
- J. Xing, A. Romero, L. Bauersfeld, D. Scaramuzza, *Bootstrapping Reinforcement Learning with
  Imitation for Vision-Based Agile Flight*, arXiv:2403.12203 (RL teacher with privileged state →
  vision student by IL → RL fine-tuning). CoRL 2024 (PMLR vol. 270; `docs/LICENSES.md` §2).
- N. Messikommer, J. Xing, L. Bauersfeld, M. Cannici, E. Aljalbout, D. Scaramuzza, *Approximate
  Imitation Learning for Event-based Quadrotor Flight in Cluttered Environments*, arXiv:2603.07578
  (DAgger with online rendering as the costly baseline; why render throughput matters).

**IL methods**
- S. Ross, G. Gordon, D. Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger), AISTATS 2011.
- M. Laskey, J. Lee, R. Fox, A. Dragan, K. Goldberg, *DART: Noise Injection for Robust Imitation
  Learning*, CoRL 2017.
- C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023;
  IJRR 2024; arXiv:2303.04137.
- T. Z. Zhao, V. Kumar, S. Levine, C. Finn, *Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware* (ACT, temporal ensembling), RSS 2023.
- A. Prasad et al., *Consistency Policy*, RSS 2024 (stretch).
- J. Song, C. Meng, S. Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
- P. de Haan, D. Jayaraman, S. Levine, *Causal Confusion in Imitation Learning*, NeurIPS 2019.
- C. Wen, J. Lin, T. Darrell, D. Jayaraman, Y. Gao, *Fighting Copycat Agents in Behavioral Cloning
  from Observation Histories*, NeurIPS 2020.

**Aerospace IL analog**
- A. Posadas-Nava, A. Scorsoglio, L. Ghilardi, R. Linares, R. Furfaro, *Action Chunking with
  Transformers for Image-Based Spacecraft Guidance and Control*, arXiv:2509.04628 (ACT from 100
  demonstrations of a meta-RL agent; no code released, per its review page).

**Inherited from P8 (cite as P8 does)** — gym-pybullet-drones (Panerati et al., IROS 2021), PyBullet,
SB3 (JMLR 2021), PPO, IQM/stratified bootstrap (Agarwal et al., NeurIPS 2021), Wilson interval,
JONSWAP, Fossen's handbook, MSS.

## 4. Things this project must not cite as support

- Real-flight results of any paper as evidence for this repo's numbers.
- P8's withdrawn motion-realism novelty claim (P8 H4).
- Any latency multiplier not measured here.
