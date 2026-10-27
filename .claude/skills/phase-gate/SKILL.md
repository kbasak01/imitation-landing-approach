---
name: phase-gate
description: Run the exit-gate check for a build phase before moving on. Verifies the phase's gate criteria against committed artifacts, calls the right auditors, and reports pass/fail per criterion.
argument-hint: <phase-number 0-11>
# disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---

Run the gate check for phase $ARGUMENTS of `docs/IMPLEMENTATION_PLAN.md`.

1. Read the phase section and its **Exit gate**. Read `docs/protocol.md` for any recorded change to
   those criteria; the protocol wins.
2. Run `make lint` and `make test`. Report failures by test name only.
3. Check each criterion against **committed artifacts** (files in `results/`, `splits/`, manifests,
   test outcomes, recorded SHAs) — not against what the code or a summary claims. Unverified until an
   artifact shows it.
4. Delegate reviews by phase and fold in their findings:

   | Phase | Reviewers |
   |---|---|
   | 1 | `privileged-leak-auditor` |
   | 2 | `privileged-leak-auditor` (timing, render invariance) |
   | 3 | `privileged-leak-auditor`, `results-skeptic` (on the P3-D1 draft) |
   | 4, 6, 7 | `training-run-diagnostician` (every run), `results-skeptic` |
   | 5 | `privileged-leak-auditor`, `training-run-diagnostician`, `results-skeptic` |
   | 8 | `privileged-leak-auditor`, `results-skeptic` |
   | 9 | `results-skeptic` (parity criteria, latency claims) |
   | 10 | `/full-audit` already run — read its report |
   | 11 | `repo-publication-reviewer` |

5. Table: criterion | measured value / evidence | PASS / FAIL / UNVERIFIED.

Then exactly one of:

- **GATE PASSED** — one-line justification; append a dated `Gate $ARGUMENTS passed` entry (with the
  table and reviewer verdicts) to `docs/protocol.md`; add a short "Before you start" note at the head
  of the next phase's protocol section if this phase measured anything that changes it; name the
  first task of the next phase; suggest the commit message and the PR title.
- **GATE FAILED** — append a dated `Gate $ARGUMENTS failed, attempt k` entry; list the failing
  criteria and the smallest fix for each. Do not start the next phase. Do not propose relaxing a
  threshold; if one genuinely must change, say so explicitly and leave the decision to the user, to be
  recorded as a dated deviation.

Be concise. This is a checkpoint, not a report.
