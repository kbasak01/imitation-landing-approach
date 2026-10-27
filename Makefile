# imitation-landing-approach -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
#
# Interpreter and tooling resolve to .venv when it exists and to PATH otherwise (inherited from
# P8, where an unactivated shell exited 127 and failed a gate for an environment reason).
# Overridable: `make PY=python RUFF=ruff MYPY=mypy PYTEST=pytest`.
PY     ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUFF   ?= $(shell [ -x .venv/bin/ruff ]   && echo .venv/bin/ruff   || echo ruff)
MYPY   ?= $(shell [ -x .venv/bin/mypy ]   && echo .venv/bin/mypy   || echo mypy)
PYTEST ?= $(shell [ -x .venv/bin/pytest ] && echo .venv/bin/pytest || echo pytest)

# `make collect TEACHER=ppo N=1000`, `make train-bg CFG=configs/train/bc.yaml SEED=0`,
# `make train-smoke ARM=bc`.
TEACHER ?= ppo
N       ?= 1000
CFG     ?= configs/train/bc.yaml
SEED    ?= 0
ARM     ?= bc

.PHONY: lint test format smoke verify-p8 reproduce-p8 throughput collect train-smoke train-bg \
        dagger-bg eval bench report figures gifs all

# --- implemented ------------------------------------------------------------------

lint:   ; $(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts && $(MYPY) src
test:   ; $(PYTEST)
format: ; $(RUFF) format src tests scripts && $(RUFF) check --fix src tests scripts
# Phase 0: import rld + ila, one P8 env built through the adapter and reset once.
smoke:  ; $(PYTEST) tests/test_smoke.py
# Phase 0: P8 commit, frozen-file hashes, PPO seed-0 checkpoint -> results/p8_provenance.json.
verify-p8: ; $(PY) scripts/verify_p8.py --out results/p8_provenance.json

# --- not yet implemented ----------------------------------------------------------
# One stub per target in REPO_SETUP.md §5; each exits 0 so `make all` lists every stage.
# Replace the stub in its own phase.

reproduce-p8: ; @echo "not implemented: phase 2"
throughput:   ; @echo "not implemented: phase 2"
collect:      ; @echo "not implemented: phase 3 (TEACHER=$(TEACHER) N=$(N))"
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
