#!/usr/bin/env bash
S=/tmp/claude-1000/-home-kbasak-projects-02-imitation-landing-approach/dac10d8d-026e-44c5-904e-0fe74946efe3/scratchpad/r1_investigation
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
cd $S
PY=/home/kbasak/projects_02/imitation-landing-approach/.venv/bin/python
$PY fly_seeds.py --variants dp_raw_ta1 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 8 --out $S/flights_ta1_dpraw_s0-9.pkl
$PY fly_seeds.py --variants r1_ta1 dp_ema_ta1 --seeds 5 6 7 8 9 --workers 8 --out $S/flights_ta1_s5-9.pkl
echo ALLDONE
