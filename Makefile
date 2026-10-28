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

.PHONY: lint test format smoke verify-p8 reproduce-p8 label-parity throughput determinism collect train-smoke train-bg \
        dagger-bg eval bench report figures gifs all

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

# --- not yet implemented ----------------------------------------------------------
# One stub per target in REPO_SETUP.md §5; each exits 0 so `make all` lists every stage.
# Replace the stub in its own phase.

train-smoke:  ; @echo "not implemented: phase 4 (ARM=$(ARM))"
train-bg:     ; @echo "not implemented: phase 4 (CFG=$(CFG) SEED=$(SEED))"
dagger-bg:    ; @echo "not implemented: phase 5"
eval:         ; @echo "not implemented: phase 8"
bench:        ; @echo "not implemented: phase 9"
report:       ; @echo "not implemented: phase 8"
figures:      ; @echo "not implemented: phase 8"
gifs:         ; @echo "not implemented: phase 10"

# The project in methodology order. Long jobs (collect, train-bg, dagger-bg) are launched by
# hand from /train-job and are not in `all`.
all: verify-p8 reproduce-p8 throughput eval bench report figures
