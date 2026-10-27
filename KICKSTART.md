# KICKSTART.md — prompts to drive Claude Code through the build

One Claude Code session per phase. Start each in **plan mode** (Shift+Tab until "plan mode"),
approve the plan, then let it execute. Run `/clear` after each gate passes so the next phase starts
with a clean context and re-reads `CLAUDE.md`, the plan and the protocol.

## Before the first prompt

1. Follow `REPO_SETUP.md` §1 (repo + submodule). Copy this kit in.
2. `chmod +x .claude/hooks/guard_paths.py`.
3. Start `claude` in the repo root. Run `/agents` and confirm all nine agents are listed. Type `/` and
   confirm `/phase-gate`, `/train-job`, `/job-status`, `/new-student`, `/full-audit` appear (the
   reference skills `il-conventions`, `landing-protocol-p8`, `diffusion-policy-recipe`, `claim-trace`
   load automatically when relevant).
4. Optional: `/model` → Opus for Phases 1, 2, 3 and 8 (the crux and the scoring); any model for
   routine phases.

## How the agents and skills fit together

| When | Use |
|---|---|
| building a phase's code | the owning specialist (CLAUDE.md table), delegated by the main session |
| any code that touches observations, frames, chunks or time | `il-conventions` (auto) |
| evaluation wording, statistics, tables | `landing-protocol-p8` (auto) |
| DP architecture/scheduler/sampling questions | `diffusion-policy-recipe` (auto) |
| a new student type | `/new-student <name>` |
| a long job (collection, training, DAgger, the matrix) | `/train-job <what>` → you launch it → `/job-status` |
| end of every phase | `/phase-gate N` (calls the right auditors itself) |
| Phase 10, or whenever results look too good | `/full-audit` |
| README or any text with numbers | `claim-trace` (auto) |

---

## Prompt 0 — Bootstrap (Phase 0)

```
Read CLAUDE.md, docs/IMPLEMENTATION_PLAN.md (§0, §1 and Phase 0), REPO_SETUP.md, SOURCES.md and
docs/protocol.md.

Execute Phase 0:
1. Write pyproject.toml: package `ila`, src layout, Python >=3.12. Read
   external/rl-land-on-moving-deck/pyproject.toml and copy its ruff/mypy/pytest config; dependencies
   are ONLY the additions listed in REPO_SETUP.md §2 (P8's pins come from installing P8). Add pytest
   markers gpu, slow, checkpoints, data.
2. Makefile with P8's PY/RUFF/MYPY/PYTEST auto-detection and stubs for every target in REPO_SETUP §5.
3. Give me the exact shell commands for REPO_SETUP §2 and §3. I will run pip myself. Wait.
4. After I confirm: package skeleton per REPO_SETUP §4 (empty modules with docstrings), .gitignore,
   CI workflow, configs/paths.local.yaml from the example (ask me for the P8 artifacts path).
5. scripts/verify_p8.py → results/p8_provenance.json (P8 commit, MANIFEST and success.yaml SHA-256
   against the values in the plan, PPO seed-0 checkpoint + VecNormalize SHA-256).
6. tests/test_smoke.py: import rld and ila; build one P8 env through the (stub) adapter; one reset.
7. docs/LICENSES.md and THIRD_PARTY_NOTICES.md from SOURCES.md; resolve every (verify) you can and
   list the rest.
8. Fill P0-D1 in docs/protocol.md.
Then run /phase-gate 0.
```

## Prompt 1 — Adapter, camera, observation contract (Phase 1, the crux)

```
Read CLAUDE.md, Phase 1 of the plan, §0(1) and §1.1–§1.3, docs/protocol.md, and the il-conventions
skill. Delegate Phase 1 to the sim-render-engineer agent.

Order of work, tests first where stated:
1. ila/p8_adapter.py — the only importer of rld. Read P8's source for the real names of read_list,
   make_env, load_eval_configs, OBS_FIELDS and the statistics functions; never guess them. Test that
   the stats re-exports are P8's objects by identity.
2. ila/obs/split.py from configs/obs/student.yaml; write tests/test_obs_split.py FIRST (every field
   classified once; offsets equal OBS_FIELDS; StudentInput cannot hold privileged data; ablations
   change exactly one field) and watch it fail.
3. Inspect P8's plate and ground with getVisualShapeData; record what you find and the twin-or-not
   decision as P1-D1 before writing ila/render/scene.py.
4. ila/render/texture.py, camera.py, scene.py.
5. Render tests: physics bit-identical render off/on/on+twin+recolour over a scripted 300-step
   moving-deck episode; background invariance under drone translation; pinhole projection within
   0.5 px; roll moves the image the right way; image t and obs t share a stepSimulation count;
   byte-identical re-render.
6. docs/CONVENTIONS.md identical to the il-conventions skill.
Then ask the privileged-leak-auditor for a verdict, and run /phase-gate 1.
```

## Prompt 2 — Runner, reproduction gate, throughput (Phase 2)

```
Read CLAUDE.md, Phase 2 of the plan, §0(4)–(5), docs/protocol.md and the landing-protocol-p8 skill.
Delegate to sim-render-engineer.

1. Read external/rl-land-on-moving-deck/src/rld/eval/runner.py in full. Build ila/eval/runner.py
   with the same contract, reusing P8's public functions; for each private helper you re-implement,
   add a test that it equals P8's on P8's fixtures.
2. ila/eval/batched.py: lockstep workers render; one GPU process does batched inference; rows must
   not depend on worker count on the CPU path.
3. scripts/reproduce_p8.py and `make reproduce-p8`: pid_feedforward on all 2 800 listed episodes and
   ppo seed 0 on id SS5+SS6, render ON, rows compared as text to P8's committed CSVs. Prepare the
   command with /train-job and let me launch it. If any row differs, STOP and diagnose (threads,
   body ordering, render timing) — do not proceed and do not loosen the comparison.
4. Teacher label parity (P8 ONNX on ORT CPU vs SB3, 10 000 observations, ≤ 1e-6).
5. scripts/throughput.py → results/throughput.csv; projection of demo, DAgger and matrix wall clock.
6. Determinism contract: dummy dp at batch 1/8/24 on 200 select episodes; CPU rows bit-identical;
   GPU flip count vs P8's one-ulp floor.
Record P2-D1..D3, then /phase-gate 2.
```

## Prompt 3 — Demos and protocol freeze (Phase 3)

```
Read CLAUDE.md, Phase 3, §1.4–§1.8 and docs/protocol.md. Delegate steps 1–3 to demo-data-engineer.

1. ila/data/episodes.py + splits/ manifests (seed 20261023) with tests: train pool only, disjoint
   from tune and every frozen list by realization key, no SS6, no 90°; select list from tune.
2. ila/data/store.py (schema in plan §1.5) and collect.py. Prepare /train-job commands for the
   collections in configs/data/demos.yaml; I launch them.
3. Audits: re-render 20 saved states → identical frames; 1 000-step label audit; disjointness.
   results/demos_summary.json.
4. Main session: draft P3-D1 in docs/protocol.md exactly as its skeleton lists, restating
   hypothesis magnitudes against the measured teacher ceilings where needed. Ask the
   privileged-leak-auditor and results-skeptic to review the draft. Show me the draft and WAIT for
   my approval before marking it frozen. Then /phase-gate 3.
```

## Prompt 4 — BC family (Phase 4)

```
Read CLAUDE.md, Phase 4, P3-D1 and the il-conventions skill. Delegate to policy-trainer.

1. ila/policies/encoder.py, bc.py, state_mlp.py; ila/train/{loop,dataset,select,manifest}.py.
2. make train-smoke ARM=bc and ARM=bc_state: overfit 8 episodes (MSE < 1e-4) and land ≥ 7/8 of them
   closed loop. If not, look for an alignment bug before touching capacity.
3. DART σ selection (3 trials on select, logged as tuning), then /train-job for bc_state, bc,
   bc_matched, bc_dart × seeds 0–4. Record the first run's wall clock and apply the budget rule if
   triggered.
4. After each run finishes, the training-run-diagnostician reviews it.
Then /phase-gate 4.
```

## Prompt 5 — DAgger (Phase 5)

```
Read CLAUDE.md, Phase 5 and P3-D1. Delegate the loop to demo-data-engineer and training to
policy-trainer.
Implement ila/data/dagger.py per plan §1.6 and configs/data/demos.yaml: β schedule, step-wise
shadowing of the teacher with reset per episode, aggregation manifests, warm-start training. Label
audit per iteration; count check against bc_matched. Prepare /train-job for 5 seeds (the loop runs
unattended). Report per-iteration train/select success and teacher agreement.
privileged-leak-auditor reviews the aggregates; then /phase-gate 5.
```

## Prompt 6 — Diffusion Policy (Phase 6)

```
Read CLAUDE.md, Phase 6, P3-D1, the il-conventions and diffusion-policy-recipe skills. Delegate to
policy-trainer.

1. ila/policies/dp/chunking.py FIRST, with table tests (To ∈ {1,2}, Ta ∈ {1,8,15}; Ta > Tp−To+1
   refused; first-step padding; ensembling weights).
2. unet1d.py (adapt the MIT reference ConditionalUnet1D, keep its notice) and policy.py (DDPM
   training, DDIM sampling with explicit noise input and a per-episode generator).
3. Toy bimodal test: DP recovers both modes; MSE-BC collapses to the mean.
4. make train-smoke ARM=dp, then /train-job for seeds 0–4. Selection at Ta=8, K=10 on select.
5. Ta / K / ensembling sweeps on select only, same checkpoints; results/dp_sweeps.json.
6. Optional R1: in .venv-lerobot, LeRobot DiffusionPolicy on the exported dataset → ONNX → our
   runner, tagged reference_external.
training-run-diagnostician on every run; then /phase-gate 6.
```

## Prompt 7 — Modular, multimodality, ablations (Phase 7)

```
Read CLAUDE.md, Phase 7 and P3-D1. Delegate to policy-trainer.
Build ila/policies/modular.py (estimator → frozen teacher, time_fraction passed through to the
teacher only), with per-field estimation error on held-out demos and on closed-loop states, compared
with P8's σ_p = 4 cm noise level. Then /train-job for every ablation in configs/eval/matrix.yaml
(3 seeds each). Plot T_mix touchdown-time distributions on select for bc-mix and dp-mix.
Then /phase-gate 7.
```

## Prompt 8 — Evaluation matrix and hypotheses (Phase 8)

```
Read CLAUDE.md, Phase 8, P3-D1 (all of it) and the landing-protocol-p8 skill. Delegate to
eval-auditor.
1. scripts/eval_matrix.py from configs/eval/matrix.yaml; prepare /train-job for the matrix; I launch.
2. Summaries, IQM CIs, contrasts and hypotheses with P8's statistics through the adapter; unit tests
   for each hypothesis scorer on synthetic cases with known answers.
3. results/results.md via scripts/report.py (byte-identical re-render); figures; GIF gallery
   (hand-picked, labelled, re-flown and matched to committed rows).
4. Any student above its teacher in any cell: investigate (touchdown audit, tunnelling, detector
   disagreement) before reporting.
results-skeptic reviews; then /phase-gate 8.
```

## Prompt 9 — Deployment (Phase 9)

```
Read CLAUDE.md, Phase 9 and P3-D1 §7. Delegate to deploy-benchmarker.
Export bc_dagger and dp (encoder, denoise step, unrolled K=10 with noise as input), fp32/fp16;
parity on 1 000 select inputs with TF32 off; closed-loop parity against the P2 determinism
contract; latency with P8's methodology on torch CUDA, ORT CUDA, ORT TensorRT fp16, ORT CPU; score
H5(a); derive d and fly the latency-in-the-loop arm; score H5(b). /phase-gate 9.
```

## Prompt 10 — Validation sweep (Phase 10)

```
Walk VALIDATION_CHECKLIST.md line by line. For each, collect evidence and write the verdict into
docs/LIMITATIONS.md (met / met-with-caveat / unmet). Run /full-audit. Re-run make report and every
--check mode and diff. List anything unmet and the smallest fix. /phase-gate 10.
```

## Prompt 11 — Publication (Phase 11)

```
Read CLAUDE.md, Phase 11, docs/protocol.md, docs/LIMITATIONS.md and the claim-trace skill. Write
README.md in P8's style (status line with each gate's real history; the short version; headline
figure; tables copied from results with sources; hypotheses table; deployment; "what a rendered
camera can and cannot show"; limitations before reproduction; citations; licence). Every number
traced to a committed file. Then ask repo-publication-reviewer for a verdict, fix, and on SHIP
prepare the release (assets + SHA256SUMS.txt) and the tag command for me.
```

## When things go wrong

- **Reproduction gate fails:** `Prompt: "Use the sim-render-engineer to bisect the first differing
  row: same episode with render off, with render on but no twin, with twin but no recolour. Report
  the first configuration that differs and the first physics step where state diverges."`
- **A student lands nothing:** `"Run make train-smoke for this arm and the il-conventions checklist;
  print image[t], proprio[t], label[t] for steps 0–3 of one stored episode next to the teacher's
  obs and action from a fresh rollout of the same episode."`
- **Results look too good:** `/full-audit`.
- **Context is getting long mid-phase:** commit, write a short "where I am" note in the phase's
  protocol entry, `/clear`, and resume with that phase's prompt plus "resume from the note".
