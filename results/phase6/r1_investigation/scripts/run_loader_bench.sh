#!/usr/bin/env bash
# Sequential loader-alone benchmarks (CPU only, one at a time). Writes only into this directory.
set -u
S=/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/dac10d8d-026e-44c5-904e-0fe74946efe3/scratchpad/r1_investigation
PY=/home/kbasak/projects_02/imitation-landing-approach/.venv-lerobot/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$S"
run() {  # name, args...
  local name=$1; shift
  echo "=== $name $(date +%T) load: $(cut -d' ' -f1-3 /proc/loadavg)"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu util (dp training): /'
  "$PY" loader_bench.py "$@" --out "$S/loader_$name.json" 2>&1 | grep -v "^\[INFO\]" | tail -40
}
run full_cycle_w12
run full_continuous_w12 --mode continuous
run ep8_cycle_w12 --episodes 0-7
run ep8_continuous_w12 --episodes 0-7 --mode continuous
run full_cycle_w4 --workers 4
echo "=== done $(date +%T)"
