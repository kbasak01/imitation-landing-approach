# CLAUDE.md — imitation-landing-approach

## What this project is

Vision-based imitation learning for quadrotor landing on a heaving ship deck, in simulation. A
privileged state-based teacher from Project 8 (`rl-land-on-moving-deck`, pinned read-only submodule
in `external/`) — its `ppo` seed 0, its `pid_feedforward`, and its `gated` controller — is distilled
into camera-based students: `bc_state`, `bc`, `bc_matched`, `bc_dart`, `bc_dagger`, `dp` (Diffusion
Policy) and `modular` (estimate the deck state from images, then run the teacher). Everything is
flown on P8's frozen episode lists under P8's frozen success criteria and statistics. ONNX latency
study and a latency-in-the-loop arm at the end.

Full spec: `docs/IMPLEMENTATION_PLAN.md`. **Read the phase you are in before writing code, and read
`docs/protocol.md` for everything already decided.** Where they disagree, the protocol wins.

## Non-negotiables

1. **Simulation only.** PyBullet physics and PyBullet's TinyRenderer. No sentence may imply a real
   camera, a real deck, real flight, or sim-to-real transfer.
2. **The student never sees privileged state.** Student inputs are images plus the allowlist in
   `configs/obs/student.yaml`, selected by P8 field *name*. `rel_position`, `rel_velocity`,
   `deck_normal`, `rel_tilt`, `height` never reach a student input; `time_fraction` only in the
   labelled `+clock` ablation. No world-fixed texture in any rendered frame.
3. **P8 is read-only and its protocol is inherited.** Never edit `external/`. Frozen lists, success
   YAML, outcome classes and statistics come through `ila.p8_adapter` — the only module that imports
   `rld`. Rendering must not change physics: the Phase 2 reproduction gate stays green forever.
4. **Splits.** Demos, DAgger rollouts and DART noise use P8's `train` pool only. Checkpoint selection
   and the one permitted tuning search (DART σ) use `splits/select.parquet` (from P8's `tune` pool)
   only. Frozen lists are for evaluation only, after P3-D1 is frozen. SS6 and 90° are never trained,
   tuned or selected on.
5. **Data-matched comparisons.** On-policy methods (DAgger, DART) are compared with `bc_matched` at
   equal labelled-episode count. Never claim "DAgger beats BC" against the smaller-data `bc`.
6. **Five seeds for core arms, three for ablations, none dropped.** IQM with stratified-bootstrap 95 %
   CIs; paired bootstrap for contrasts; a difference exists only when its CI excludes 0.
7. **Baselines always printed.** Teacher, `pid_feedforward`, `pid_track_descend`, `oracle_gated`
   beside every student, from P8's committed rows.
8. **Report what you find.** Nulls, teacher-beats-student, CPU-beats-GPU — all in the README body.
   A student beating its own teacher is investigated before it is reported.

## Architecture rules

- Logic in `src/ila/`; `scripts/` are argparse wrappers; YAML in `configs/` → frozen dataclasses.
- `ila/obs/split.py` is the only place observation fields are selected.
- `ila/policies/dp/chunking.py` is the only place `To`/`Tp`/`Ta` indices are computed.
- `ila/render/` renders after `env.step()` from the same physics state as the observation. Visual-only
  bodies have no collision shape.
- Every student implements P8's `EvalPolicy` shape (`reset(seed, …)`, `act(...)`) plus
  `act_vision(image_stack, proprio)`; new student = file + config + registry entry + test. Use
  `/new-student`.
- Diffusion sampling noise comes from a per-episode `torch.Generator` seeded by
  `(run_seed, episode_seed, step)`.
- Units in every docstring. Model scale vs full scale stated for every time and length (λ = 1/25).

## Commands

```
make test | make lint | make smoke      # pytest; ruff + mypy --strict; quick end-to-end
make verify-p8                          # P8 commit, list hashes, success.yaml, checkpoint SHAs
make reproduce-p8                       # Phase 2 gate: P8 rows with render on
make throughput                         # Phase 2 budget
make collect TEACHER=… N=…              # demo collection (background)
make train-bg CFG=configs/train/<arm>.yaml SEED=n   # background training; status in artifacts/runs/*/status.json
make train-smoke ARM=…                  # overfit-8 + closed-loop landing check
make eval                               # Phase 8 matrix
make bench                              # Phase 9 export, parity, latency, latency arm
make report | make figures              # re-render from committed CSVs
```

**Never run demo collection, training or the evaluation matrix in the foreground.** Prepare the
command with `/train-job`, let the user launch it, check with `/job-status`.

## Gates

Each phase ends with a gate in the plan. Run `/phase-gate N`. Do not start phase N+1 until gate N
passes. Never relax a threshold silently; record any change in `docs/protocol.md` as a dated
deviation with its reason.

## Specialists — delegate rather than doing their work in the main thread

| Agent | Owns |
|---|---|
| `sim-render-engineer` | `ila/p8_adapter.py`, `ila/obs/`, `ila/render/`, `ila/eval/runner.py`, `ila/eval/batched.py` — Gates 1, 2 |
| `demo-data-engineer` | `ila/data/`: episodes, collection, store, DART, DAgger loop, LeRobot export — Gates 3, 5 |
| `policy-trainer` | `ila/policies/`, `ila/train/`: encoder, BC, DP, modular, training loops — Gates 4, 6, 7 |
| `eval-auditor` | `ila/eval/` except the runner, statistics wiring, tables, hypotheses — Gate 8. Read-only over `policies/` and `data/` |
| `deploy-benchmarker` | `ila/deploy/`: ONNX, parity, latency, latency-in-the-loop — Gate 9 |
| `privileged-leak-auditor` | Read-only. Student inputs, renders, splits, selection, label alignment. Before Gates 1, 3, 5, 8 |
| `training-run-diagnostician` | Read-only. Every training run: curves, EMA, copycat, collapse, diversity, budget |
| `results-skeptic` | Read-only adversarial review before every gate from Phase 3 on |
| `repo-publication-reviewer` | Read-only. Phase 11 and the Phase 10 dry run |

## Known traps in this project

- **Render timing.** Render after `env.step()` returns, never inside a physics substep. Image t and
  obs t share one `stepSimulation` count.
- **The ground plane is an inertial reference.** P8's checkerboard plane must be recoloured, not
  left in the frame.
- **Chunk indices.** `Ta ≤ Tp − To + 1`. Executed actions start at chunk index `To − 1`.
- **Stateful teachers.** `pid_feedforward` and `gated` keep integrators/latches; call `reset` on every
  episode and shadow them step by step during DAgger, never query them on isolated states.
- **VecNormalize.** The PPO teacher needs P8's frozen statistics with `training=False`.
- **Batch-dependent GPU numerics.** Rows are bit-identical only on the CPU single-sample path; the
  GPU path is judged by the Phase 2 determinism contract.
- **P8 inherits from Python 3.12 and pins.** pybullet builds from sdist; `dmf` must be an editable
  install; never `pip install` anything that re-resolves torch.
- **Success without outcome classes** hides hovering, timeouts and crashes. Always print the
  breakdown.
- **Throughput.** SB3/P8 MLPs are faster on CPU; vision encoders are not. Measure; don't assume.

## Style

Type hints, mypy strict, ruff (P8's config). Google docstrings with units and frames. No notebooks
defining logic. Commit `results/` CSV/JSON; never commit `data/` or `artifacts/`.
