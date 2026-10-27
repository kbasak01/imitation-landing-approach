---
name: full-audit
description: Run the full pre-release integrity audit — leakage, training-run verdicts, statistics and claims — by dispatching the read-only auditors and consolidating their findings into docs/audit_report.md.
# disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---

Run a full audit of the current tree.

1. `make lint test`; `make verify-p8`; `make report` and `git diff --exit-code results/results.md`;
   every script's `--check` mode that exists.
2. Dispatch, in parallel where possible, and collect verbatim verdicts:
   - `privileged-leak-auditor` — whole pipeline;
   - `training-run-diagnostician` — every final checkpoint referenced in `results/e08/run_info.json`
     (summary table of verdicts);
   - `results-skeptic` — `results/`, `docs/protocol.md`, README draft if present;
   - `repo-publication-reviewer` — dry run.
3. Re-run the Phase 2 reproduction gate on a 200-episode subset (`id` SS5 `pid_feedforward`) and
   compare rows as text.
4. Write `docs/audit_report.md`: date, commit, each auditor's verdict and findings (BLOCKING /
   SHOULD FIX / NOTE), the reproduction spot-check, and a single consolidated list of actions ordered
   by severity. Do not fix anything in this skill.
