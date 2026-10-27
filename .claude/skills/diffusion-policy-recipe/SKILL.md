---
name: diffusion-policy-recipe
description: Implementation recipe and pitfalls for this project's Diffusion Policy student — conditional 1D U-Net with FiLM, GroupNorm ResNet-18 + spatial-softmax encoder, DDPM training, DDIM sampling, EMA, normalisation, chunk execution, temporal ensembling, deterministic export. Use when writing, debugging, exporting or reviewing the dp arm or any diffusion/sampling code.
---

# Diffusion Policy recipe (this project's frozen defaults; P3-D1 wins on conflict)

## Reference
Chi et al., RSS 2023 / IJRR 2024; repo `real-stanford/diffusion_policy` (MIT). Its
`train_diffusion_unet_hybrid_workspace.yaml` defaults: horizon 16, n_obs_steps 2, n_action_steps 8,
DDPM 100 train steps, ε-prediction, GroupNorm encoder, crop 76 of 96 (we use 88 of 96 for a 96 px
deck image), EMA power 0.75 / max 0.9999, lr 1e-4, batch 64. Ours differs where stated in
`configs/train/dp.yaml` (smaller U-Net `[128, 256, 512]`, batch 128, 100 k steps, DDIM K = 10).
If you adapt `ConditionalUnet1D`, keep the MIT notice in the file and `THIRD_PARTY_NOTICES.md`.

## Model
- Encoder per frame: ResNet-18 (no pretraining), BatchNorm → GroupNorm(16), spatial softmax (32
  keypoints → 64-d). Concatenate `To` frames + proprio MLP(64, 64) → global conditioning vector.
- U-Net over the action sequence (channels = 3, length = Tp): down/up blocks with Conv1d +
  GroupNorm + Mish; FiLM from [diffusion-step sinusoidal embedding (128) ‖ obs conditioning].
- Train: sample k ~ U{0..99}, ε ~ N(0, I); x_k = scheduler.add_noise(a, ε, k); loss = MSE(ε̂, ε).
- Scheduler: `diffusers` DDPM for training (`squaredcos_cap_v2`, `clip_sample=True`), DDIM for
  inference with K steps (`set_timesteps(K)`), η = 0.

## Inference
- Initial noise and any per-step noise from the per-episode generator (il-conventions). Clip the
  final sample to [−1, 1].
- Replan every `Ta` steps; execute chunk indices `To−1 … To−2+Ta`. Temporal ensembling: replan every
  step, exponential weights, m = 0.01.
- Always evaluate the EMA weights.

## Pitfalls (each has bitten someone)
1. Executing chunk indices from 0 instead of `To−1` → a policy one step late; looks like "chunking
   hurts".
2. Normalisation fitted on actions that include DART noise or DAgger student actions → train on
   `label`, never on `action_exec`.
3. Evaluating raw weights instead of EMA → noisy chunks, jerk at boundaries.
4. BatchNorm in the encoder with small or correlated batches → train/eval mismatch. GroupNorm.
5. DDIM with `K` > train steps, or timesteps in the wrong order → silent garbage. Test: K = 100 DDIM
   sample statistics ≈ DDPM on the 1-D two-Gaussian toy.
6. Global RNG for sampling → results depend on batch composition and worker count.
7. Export with the sampling loop reading a Python RNG → nondeterministic graph. Noise is an input.
8. Comparing K or Ta settings with re-selected checkpoints → selection effect, not a sampling effect.

## Diagnostics worth logging
Denoising loss by k-bucket; per-state spread of 16 samples; chunk-boundary jerk; fraction of samples
clipped at ±1; inference calls per episode; wall time per call at batch 1 and eval batch.
