# imitation-landing-approach -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
#
# Interpreter and tooling resolve to .venv when it exists and to PATH otherwise (inherited from
# P8, where an unactivated shell exited 127 and failed a gate for an environment reason).
# Overridable: `make PY=python RUFF=ruff MYPY=mypy PYTEST=pytest`.
PY     ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUFF   ?= $(shell [ -x .venv/bin/ruff ]   && echo .venv/bin/ruff   || echo ruff)
MYPY   ?= $(shell [ -x .venv/bin/mypy ]   && echo .venv/bin/mypy   || echo mypy)
PYTEST ?= $(shell [ -x .venv/bin/pytest ] && echo .venv/bin/pytest || echo pytest)

# `make collect COLLECTION=ppo_2000` (or `TEACHER=ppo N=2000`), `make train-bg
# CFG=configs/train/bc.yaml SEED=0`, `make train-smoke ARM=bc`.
TEACHER ?= ppo
N       ?= 1000
COLLECTION ?= $(TEACHER)_$(N)
WORKERS ?= 24
CFG     ?= configs/train/bc.yaml
SEED    ?= 0
ARM     ?= bc
# Extra flags for train-bg (e.g. ARGS="--max-steps 2000" for a P3-D1 §5 timing run, or
# ARGS="--sigma 0.10 --tuning" for a DART trial, ARGS="--rollout-workers 16" for dagger-bg) and
# select-bg (e.g. SELECT="--arm bc --seeds 0 1"); dagger-report takes REPORT (e.g. REPORT="--seeds 0").
ARGS    ?=
SELECT  ?=
REPORT  ?=
# dp-sweeps / dp-diagnostics take SWEEP / DIAG (e.g. SWEEP="--arm dp --seeds 0 1 2 3 4").
SWEEP   ?= --arm dp --seeds 0 1 2 3 4
DIAG    ?= --arm dp --seeds 0 1 2 3 4
# eval / eval-bg take EVAL (e.g. EVAL="--dry-run", EVAL="--only core s0").
EVAL    ?=

.PHONY: r1-export r1-dataset r1-train lint test format smoke verify-p8 reproduce-p8 label-parity throughput determinism collect train-smoke train-bg \
        select-bg budget dagger-bg dagger-report dp-sweeps dp-diagnostics eval eval-bg bench report figures gifs all

# --- implemented ------------------------------------------------------------------

lint:   ; $(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts && $(MYPY) src
test:   ; $(PYTEST)
format: ; $(RUFF) format src tests scripts && $(RUFF) check --fix src tests scripts
# Phase 0: import rld + ila, one P8 env built through the adapter and reset once.
smoke:  ; $(PYTEST) tests/test_smoke.py
# Phase 0: P8 commit, frozen-file hashes, PPO seed-0 checkpoint -> results/p8_provenance.json.
verify-p8: ; $(PY) scripts/verify_p8.py --out results/p8_provenance.json
# Phase 2 gate: P8's pid_feedforward (2 800) and ppo seed 0 (id SS5+SS6, 400) rows re-flown with
# render full, 24 workers, chunk 25, one thread -> results/reproduce_p8/. Background job:
#   mkdir -p artifacts/runs/reproduce_p8 && nohup make reproduce-p8 > artifacts/runs/reproduce_p8/run.log 2>&1 &
# Pre-flight (foreground, ~1 min): $(PY) scripts/reproduce_p8.py --smoke
reproduce-p8: ; mkdir -p artifacts/runs/reproduce_p8 && $(PY) scripts/reproduce_p8.py
# Phase 2: SB3 vs ONNX/ORT CPU on 10 000 train-pool ppo observations -> results/label_parity.json
label-parity: ; $(PY) scripts/label_parity.py
# Phase 2 budget -> results/throughput.csv, throughput_projection.csv. Background job:
#   mkdir -p artifacts/runs/throughput && nohup make throughput > artifacts/runs/throughput/run.log 2>&1 &
throughput: ; mkdir -p artifacts/runs/throughput && $(PY) scripts/throughput.py
# Phase 2 determinism contract -> results/determinism/. Background job:
#   mkdir -p artifacts/runs/determinism && nohup make determinism > artifacts/runs/determinism/run.log 2>&1 &
determinism: ; mkdir -p artifacts/runs/determinism && $(PY) scripts/determinism.py
# Phase 3 demo collection: COLLECTION in ppo_2000 pid_1000 mix_1000 dart_0.05 dart_0.10 dart_0.20
# -> <data_root>/demos/<COLLECTION>.h5 (+ .manifest.json); status/log/manifest/episodes.csv in
# artifacts/runs/collect/<COLLECTION>/. Background job (prepare with /train-job):
#   mkdir -p artifacts/runs/collect/ppo_2000 && nohup make collect COLLECTION=ppo_2000 > artifacts/runs/collect/ppo_2000/nohup.log 2>&1 &
# Pilot (foreground): $(PY) scripts/collect.py pid_1000 --limit 8 --data-root <scratch> --run-root <scratch>/runs
# The store is never overwritten unless OVERWRITE=1 (passes --overwrite).
collect: ; mkdir -p artifacts/runs/collect/$(COLLECTION) && $(PY) scripts/collect.py $(COLLECTION) --workers $(WORKERS) $(if $(OVERWRITE),--overwrite,)
# Phase 4 smoke pre-flight (foreground, a few minutes): overfit the first 8 ppo_2000 train
# episodes (action MSE < 1e-4) and land >= 7/8 closed loop -> results/phase4/smoke_<arm>.json.
# ARM=bc_dagger (Phase 5, about 30 min): 2 DAgger iterations x 4 rollouts from bc s0 + a resume,
# then the 8 rollout episodes overfit and flown -> results/phase5/smoke_bc_dagger.json.
# ARM=dp (Phase 6, P6-D1, about 16 min): 10 000 steps on 8 episodes; sampled executed-chunk MSE
# < 1e-3 and >= 7/8 landed at Ta = 8, K = 10 -> results/phase6/smoke_dp.json. Background job:
#   mkdir -p artifacts/runs/smoke && nohup make train-smoke ARM=dp > artifacts/runs/smoke/dp.log 2>&1 &
# ARM=R1_lerobot_dp (Phase 6, P6-D3, ~20-25 min at 12 loader workers): LeRobot trains
# 10 000 steps on dataset episodes 0-7 (.venv-lerobot), ONNX export + parity, then >= 7/8 landed
# through OnnxDPStudent at Ta = 8, K = 10 -> results/phase6/smoke_R1_lerobot_dp.json. Background job:
#   mkdir -p artifacts/runs/smoke && nohup make train-smoke ARM=R1_lerobot_dp > artifacts/runs/smoke/R1_lerobot_dp.log 2>&1 &
train-smoke: ; $(PY) scripts/train_smoke.py $(ARM)
# Phase 4 training (background; prepare with /train-job). Run dir artifacts/runs/<arm>/s<seed>/
# (status.json, manifest.json, train_log.jsonl, checkpoints/); an existing run dir resumes.
#   mkdir -p artifacts/runs/bc/s0 && nohup make train-bg CFG=configs/train/bc.yaml SEED=0 > artifacts/runs/bc/s0/train.log 2>&1 &
train-bg: ; $(PY) scripts/train.py $(CFG) --seed $(SEED) $(ARGS)
# Closed-loop selection on splits/select.parquet, CPU fp32 single-sample path (background).
#   nohup make select-bg SELECT="--arm bc --seeds 0 1 2 3 4" > artifacts/runs/bc/select.log 2>&1 &
select-bg: ; $(PY) scripts/select.py $(SELECT)
# P3-D1 §5 budget projection from the timing runs -> results/phase4/budget_projection.csv.
budget: ; $(PY) scripts/budget.py
# Phase 5 bc_dagger training (background; prepare with /train-job; needs bc s<SEED> finished).
# Each iteration flies 200 CPU rollouts (--rollout-workers, default 24) from the previous
# iteration's last-step EMA, audits them, then trains 10 000 steps on the aggregate. Run dir
# artifacts/runs/bc_dagger/s<seed>/ (+ dagger/iter<i>.json); rollouts under
# ${data_root}/dagger/bc_dagger/s<seed>/. An existing run dir resumes (rollouts reused).
#   mkdir -p artifacts/runs/bc_dagger/s0 && nohup make dagger-bg SEED=0 > artifacts/runs/bc_dagger/s0/train.log 2>&1 &
dagger-bg: ; $(PY) scripts/train.py configs/train/bc_dagger.yaml --seed $(SEED) $(ARGS)
# Phase 5 per-iteration report -> results/phase5/dagger_iterations.{csv,json}. Background job:
# it flies the select list for every iteration's last-step EMA without reusable rows.
#   mkdir -p artifacts/runs/bc_dagger && nohup make dagger-report > artifacts/runs/bc_dagger/report.log 2>&1 &
dagger-report: ; $(PY) scripts/dagger_report.py $(REPORT)

# Phase 6 dp inference sweeps on select (P6-D1; background, ~15-25 min per seed at 24 workers,
# up to ~6x if episodes time out): each run's SELECTED checkpoint re-flown at Ta x ensembling and
# K cells -> results/phase6/sweeps/<arm>/s<n>/, results/dp_sweeps.json. Resumable per cell.
#   mkdir -p artifacts/runs/dp && nohup make dp-sweeps SWEEP="--arm dp --seeds 0 1 2 3 4" > artifacts/runs/dp/sweeps.log 2>&1 &
dp-sweeps: ; $(PY) scripts/dp_sweeps.py $(SWEEP)
# Phase 6 dp diagnostics (P6-D1; background, a few minutes per seed): sample spread, clip
# fraction, chunk-boundary vs within-chunk jerk of the selected checkpoint, on P4-D5's held-out
# states and on select re-flown with an action recorder -> results/phase6/diagnostics/<arm>/s<n>.json.
#   mkdir -p artifacts/runs/dp && nohup make dp-diagnostics DIAG="--arm dp --seeds 0 1 2 3 4" > artifacts/runs/dp/diagnostics.log 2>&1 &
dp-diagnostics: ; $(PY) scripts/dp_diagnostics.py $(DIAG)

# Phase 6 R1 data: dump dp's exact training episodes (student inputs + labels only) to a neutral
# intermediate in .venv (~40 s), then build the lossless LeRobotDataset v3 in .venv-lerobot (~4 min).
r1-export:  ; $(PY) scripts/r1/export_dataset.py $(if $(OVERWRITE),--overwrite,)
r1-dataset: ; .venv-lerobot/bin/python scripts/r1/build_lerobot_dataset.py
# Phase 6 R1 training (P6-D3; .venv-lerobot; background, prepare with /train-job): LeRobot 0.6.1's
# DiffusionPolicy on the dataset above, configs/train/r1_lerobot_dp.yaml (SHA in
# ila.train.manifest), 100 000 steps, checkpoints every 10 000 -> artifacts/runs/R1_lerobot_dp/s<SEED>/
# (status.json, manifest.json, train_log.jsonl, checkpoints/). An existing run dir resumes from its
# last checkpoint with RNG state. Refuses a dirty tree. 12 loader workers (P6-D3): ~9.8 steps/s,
# ~2.8 h per run.
#   mkdir -p artifacts/runs/R1_lerobot_dp/s0 && nohup make r1-train SEED=0 > artifacts/runs/R1_lerobot_dp/s0/train.log 2>&1 &
# Then export + parity (CPU) before selection:
#   .venv-lerobot/bin/python scripts/r1/export_onnx.py export --run artifacts/runs/R1_lerobot_dp/s0
#   .venv/bin/python scripts/r1/parity.py --run artifacts/runs/R1_lerobot_dp/s0
# R1 smoke: make train-smoke ARM=R1_lerobot_dp (background; see the train-smoke comment).
r1-train: ; .venv-lerobot/bin/python scripts/r1/train_r1.py --seed $(SEED) $(ARGS)

# Phase 8 evaluation matrix (P8-D1; background, prepare with /train-job; ~1 day of CPU wall clock
# at 24 workers, `EVAL=--dry-run` prints the jobs and the projection). Every arm on P8's frozen
# lists through the selection path (CPU fp32 single-sample) -> results/e08/rows/<arm>/s<seed>/,
# results/e08/run_info.json, status in artifacts/runs/eval/status.json. Resumable per job (re-run
# the same command); refuses a dirty tree. EVAL takes --only <tags/arms/s<n>>, --order, --workers.
#   mkdir -p artifacts/runs/eval && nohup make eval-bg > artifacts/runs/eval/run.log 2>&1 &
eval: ; $(PY) scripts/eval_matrix.py $(EVAL)
eval-bg: ; $(PY) scripts/eval_matrix.py $(EVAL)

# --- not yet implemented ----------------------------------------------------------
# One stub per target in REPO_SETUP.md §5; each exits 0 so `make all` lists every stage.
# Replace the stub in its own phase.

bench:        ; @echo "not implemented: phase 9"
report:       ; @echo "not implemented: phase 8"
figures:      ; @echo "not implemented: phase 8"
gifs:         ; @echo "not implemented: phase 10"

# The project in methodology order. Long jobs (collect, train-bg, dagger-bg) are launched by
# hand from /train-job and are not in `all`.
all: verify-p8 reproduce-p8 throughput eval bench report figures
