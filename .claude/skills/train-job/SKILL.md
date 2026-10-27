---
name: train-job
description: Prepare (not run) a long background job — demo collection, a training run or sweep, a DAgger loop, the reproduction gate or the evaluation matrix — with its manifest, log and status file, and print the exact command for the user to launch.
argument-hint: <collect|train|dagger|reproduce|eval|bench> <arm or collection> [seeds]
# isable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---

Prepare the job described by `$ARGUMENTS`. Do **not** launch it.

1. Resolve the config(s) under `configs/` and confirm they match what P3-D1 froze (SHA-256 of each
   config vs the protocol, from Phase 3 on). If anything differs, stop and say so.
2. Pre-flight, in the foreground, all cheap: `make lint`; the arm's smoke test if it exists
   (`make train-smoke ARM=…`); free disk under `data_root`/`artifacts_root`; `nvidia-smi` for other
   GPU jobs; the P8 checkpoint hash vs `results/p8_provenance.json`.
3. Estimate wall clock from `results/throughput.csv` and any finished run of the same arm; state the
   estimate and its basis. Flag if it would breach the P3-D1 budget.
4. Print one command per job, detached and logged, in this shape (Makefile targets wrap it):

   ```bash
   nohup make train-bg CFG=configs/train/<arm>.yaml SEED=<s> \
     > artifacts/runs/<arm>/s<s>/train.log 2>&1 &
   ```

   Each job writes `status.json` (state, step/budget, last `select` success, ETA, PID) and a manifest
   at start (config SHA, data SHA, P8 and `ila` commits, host, GPU). Several seeds: print a queue
   script under `artifacts/queues/` that runs them sequentially on the one GPU, and the command to
   start it.
5. Tell the user what to run, roughly how long it takes, and to come back with `/job-status`.
