---
name: new-student
description: Scaffold a new student policy type — module, config, registry entry, smoke target and tests — following the project's student contract.
argument-hint: <student-name>
# disable-model-invocation: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

Create student `$ARGUMENTS`. Read `CLAUDE.md`, the `il-conventions` skill and
`src/ila/policies/registry.py` first.

1. `src/ila/policies/$ARGUMENTS.py`: a class implementing the student contract —
   `reset(seed, context=None, motion_feed=None)` (refuses a non-None context or feed: students are
   never privileged), `act_vision(image_stack, proprio) -> action (3,)` in normalised units, and
   `act(obs)` raising `TypeError` with a message pointing at `act_vision` (so P8's obs-only path can
   never feed it a 25-d vector). Inputs are `StudentInput` from `ila.obs.split`; reuse
   `ila.policies.encoder`. Diffusion-type students take a per-episode `torch.Generator`.
2. `configs/train/$ARGUMENTS.yaml` mirroring `bc.yaml`'s keys (data, encoder, optim, steps, batch,
   precision, EMA, checkpointing, selection, seeds).
3. Registry entry with `privileged=False`, input spec (image size, `n_obs_steps`, proprio fields),
   output spec (single action or chunk with `Tp`/`Ta`).
4. Tests in `tests/test_student_$ARGUMENTS.py`: contract (shapes, dtype, range), refuses privileged
   input, determinism (same episode seed → same actions on CPU), state cleared by `reset`,
   export-safe op set (no data-dependent Python control flow in `forward`).
5. `make train-smoke ARM=$ARGUMENTS` wiring (overfit 8 episodes + closed-loop ≥ 7/8).
6. If the new student is an arm that was not in P3-D1, say so: adding it after the freeze needs a
   dated deviation and is reported as exploratory.
