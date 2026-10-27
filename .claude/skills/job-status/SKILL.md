---
name: job-status
description: Summarise every background job (collection, training, DAgger, evaluation, benchmark) from its status file and log, flag stalled or failed runs, and recommend — never execute — the next command.
argument-hint: <filter, optional>
allowed-tools: Read, Glob, Bash
---

Summarise jobs under `artifacts/runs/`, `artifacts/collect/`, `artifacts/eval/` and
`artifacts/queues/` (filter: `$ARGUMENTS` if given).

For each job read `status.json`, the tail of its log, and check the PID is alive. One table:
job | arm/collection | seed | state (queued/running/done/failed/stalled) | progress / budget |
throughput | last `select` success (training) | ETA. A job is **stalled** if `status.json` has not
updated in > 3 report intervals.

For failed or stalled jobs, show the last 20 log lines and the likely cause (OOM, NaN, PyBullet
client lost, disk full, P8 checkpoint path). Do not restart or kill anything; recommend the command
and let the user decide. For finished training runs not yet reviewed, recommend handing them to
`training-run-diagnostician`.
