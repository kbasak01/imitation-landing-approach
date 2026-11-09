# Audit report: Phase 10 full audit

- **Date:** 2026-11-09
- **Commit audited:** `8f4c9e0` (`main` after PR #11, Phase 9), branch `phase-10`
- **Run by:** `/full-audit`, with four read-only reviewers dispatched in parallel. The reviewers
  wrote no files, and the tree was frozen while `make test` ran.

All results are in simulation only.

## 1. Mechanical checks

| Check | Result |
|---|---|
| `make lint` | clean (ruff, format 163 files, mypy strict 81 files) |
| `make test` | `make test` at `8f4c9e0`: 1 244 passed, 0 failed (23 min 25 s), on an unchanging tree |
| `make verify-p8` | `all_match: true`; `results/p8_provenance.json` unchanged |
| `make report` + `git diff --exit-code results/results.md` | exit 0, byte-identical |
| `make figures` | `results/e08/figures/` unchanged |
| `make bench-check` (the `--check` mode of `scripts/deploy/bench.py`) | `results/latency` re-renders byte-identically; `d0_check.json` agrees with its rows files |
| `make bench-recheck-reference` | decision and SHAs unchanged; only provenance stamps rewritten, then restored (write mode) |
| `scripts/score.py` + `scripts/above_teacher.py` into scratch | 17/17 scored files byte-identical to `results/e08/` |

Other `--check` modes: `scripts/modular_estimation.py` and `scripts/r1/export_onnx.py` have
`--checkpoint` flags only, not `--check` modes. `bench.py check` is the only check mode in
`scripts/`.

## 2. Reproduction spot-check

The full Phase 2 gate was re-run on the final tree at `8f4c9e0` (render full, 24 workers, chunk 25,
one thread, 253 s). It covers the 200-episode `id` SS5 `pid_feedforward` subset that the skill asks for.

| Teacher | Expected | Identical (row text) | Differ | Missing |
|---|---|---|---|---|
| `pid_feedforward` | 2 800 | 2 800 | 0 | 0 |
| `ppo` s0 (`id` SS5–SS6) | 400 | 400 | 0 | 0 |

- **Frames:** `pid_feedforward` 428 781 frames = 425 981 steps + 2 800 episodes; `ppo` 25 197 frames
  = 24 797 steps + 400 episodes.
- **`comparison.json`:** byte-identical to the Phase 2 file.
- **Record:** `results/reproduce_p8/rerun_phase10/`.

## 3. Reviewer verdicts

### `privileged-leak-auditor`: GO, no LEAK

It traced every route into a student:
- training, through the store's `student_inputs`;
- the single and batched runners;
- DAgger;
- the latency arm;
- deploy inputs;
- ONNX graph inputs;
- R1's `input_features`.

The only privileged inputs are `bc_state` (declared) and the `modular` targets (reviewed allowlist).
- **BLOCKING:** none.
- **SHOULD FIX 1:** the 25 DAgger iteration audits are only in the data root, and P5-D2 cites a
  path that does not exist (`runs/bc_dagger/s<n>/rollouts/…`). *Fix:* commit a digest, and correct
  the path.
- **SHOULD FIX 2:** `make reproduce-p8` had not been re-run since `5d09486`, although `runner.py`
  and `p8_adapter.py` changed in 5 later commits. *Fix:* re-run on the final tree.
- **NOTE 1:** frozen lists were flown before P3-D1 only by scripted actions and teachers (P1-D2,
  P2-D1), never by a student.
- **NOTE 2:** P8's `episodes_mss` lists are not in `list_paths()`. They are disjoint from the train
  pool by construction.
- **NOTE 3:** DAgger flies only `ppo` (stateless). Stateful shadowing is covered by a test double
  and by the `mix_1000` relabel audit.
- **NOTE 4:** core-arm manifests have no `obs_variant` key. The checkpoints carry it, and every
  core arm's is `default`.

### `training-run-diagnostician`: all 71 pinned checkpoints usable; 0 EXCLUDE

It re-hashed all 71 pins, manifests and selected checkpoints, re-applied the selection rule (71/71
match), and checked every row's `ckpt_sha256` against its sidecar. It read all 71 training curves:
no NaN, no late spike, LR schedules as configured, EMA ≈ raw at the end.
- **BLOCKING:** none.
- **SHOULD FIX 1:** the core `dp` s0–s4 verdict (TRUSTWORTHY, P6-D5) predates evidence that meets
  the CAVEAT bar:
  - the `last_action` copycat channel: lateral RMSE +29–38 % (P7-D3 item 7);
  - 32–44/150 flips (P9-D2 item 1);
  - chunk-boundary jerk ratio 1.35–1.47, against 1.09 for the teacher;
  - under-dispersed samples;
  - picks past the held-out minimum.

  No defect: the ±1 label shift gives 506× worse MSE. *Fix:* a dated protocol entry.
- **NOTES:**
  - Phase 6/7 verdicts exist only as protocol tables, and there is no `dp_mix` diagnostics folder.
  - `bc_mix`'s averaging caveat is arm-wide, not s0 only; it is H2's subject.
  - Selection is noise at the ceiling for `bc_state`, `bc_dart` and `bc_dagger` (disclosed).
  - DAgger raw agreement fell, explained in P5-D2.
  - The recorded deviations (smoke trees, R1 smoke 6/8, `bc_dart` s0 = the σ trial) are all
    disclosed.
  - Kept fractions are not yet in `results.md` or the README.
- **Tally:** protocol 39 T / 32 C / 0 E; this review 34 T / 37 C / 0 E. The only difference is
  `dp` s0–s4. The per-run table is in the reviewer's hand-back and summarised in
  `docs/LIMITATIONS.md` D4.

### `results-skeptic`: nothing BLOCKING

- **Checks:**
  - The P3-D1 block hash was recomputed: `00e22330…3fc5`, unchanged.
  - Every frozen config SHA matches.
  - 33/33 SS6 contrasts carry the flip floor.
  - No success has a detector disagreement.
  - The above-teacher scan covers every group.
  - The latency fallback was fixed before any results.
- **SHOULD FIX:**
  1. Gate 9 says "PASS" for closed-loop parity, but `dp` failed it (32/33/44 flips). LIMITATIONS
     G4 must say so plainly.
  2. Byte-identical `dp` rows hold only on this software stack. Scope every reproducibility claim
     for `dp`, `dp_*` and R1.
  3. `dp`'s flip rate is printed only beside core `dp`. H2 (`dp_mix`) and the `dp_pid` /
     `dp_minus_last_action` rows need "sensitivity not measured".
  4. A pooled-across-sea-states README sentence is staged in P8-D1 item 11. Split it per sea state,
     or label it descriptive.
  5. Protocol prose quotes SS6 contrasts without the flip floor. Quote from the CSV/JSON instead.
  6. The H1-DART caveats must state that σ was tuned on `select` and `bc` was not.
- **NOTES:**
  - H1-SS6 is a narrow "supported" (+5.9 [+2.3, +9.7] vs +5); lead with the inconclusive primary.
  - H5(b) measures the CPU fallback, not a GPU deployment.
  - The ≤ 3-flip rule could not be passed by this `dp`.
  - The closed-loop parity and batch-1 latency ran before the reviews (disclosed).
  - Parity inputs are teacher-visited states.
  - The appearance-shift set is small (5 variants).
  - Protocol "real" is used in the software sense only.

### `repo-publication-reviewer` (dry run): HOLD for publication; nothing blocks Gate 10

- **Verified:**
  - A1, met.
  - No `data/`, `artifacts/`, weights or demo images at HEAD or in history; the largest blob is
    669 KB.
  - GIF SHAs match, and the picks recomputed from the committed rows match.
  - `graphs.json` SHAs match the local ONNX files.
  - `results.md` re-renders from committed files only.
  - CI is green on `main` at `8f4c9e0`.
- **SHOULD FIX:**
  1. `THIRD_PARTY_NOTICES.md` has the diffusion_policy commit marked "(verify)".
  2. `docs/LICENSES.md` is stale and has no "no pretrained weights" line.
  3. `SOURCES.md` has 5 "(verify)" markers (plus `REPO_SETUP.md:45`).
  4. The background-only frames at contact (camera below the deck surface at rest) are
     undocumented.
  5. CI does not run `make report`, `bench-check` or `figures`.
- **NOTES:**
  - 380 committed records carry workstation-absolute paths.
  - GIF d shows one episode where the teacher fails; caption it with no rate.
  - The GIF manifest has no `ila` commit.
  - `REPO_SETUP.md` is starter-kit text and names a non-existent `scripts/export_lerobot.py`.
- **PHASE-11 TODO:** README rewrite, a reproduction section, release assets with `SHA256SUMS.txt`,
  and making the repo public and tagging it.

## 4. Consolidated actions, by severity

**BLOCKING:** none.

**SHOULD FIX, closed in Phase 10:**
1. Reproduction gate re-run on the final tree: 3 200/3 200 identical
   (`results/reproduce_p8/rerun_phase10/`).
2. DAgger audit digest committed to `results/phase5/dagger_audits.csv` (25/25 clean); the P5-D2
   path is corrected in P10-D1 item 2.
3. Core `dp` s0–s4 re-rated TRUSTWORTHY → CAVEAT (P10-D1 item 1).
4. G4 written plainly as "`bc_dagger` pass, `dp` fail" (`docs/LIMITATIONS.md`).
5. The scope of `dp` reproducibility, the unmeasured `dp_*` sensitivity, the pooled-sentence rule,
   the flip-floor quoting rule, the H1-DART σ caveat and the contact-frame limitation are recorded
   as README rules (P10-D1 items 3–7; `docs/LIMITATIONS.md` L1–L10).

**SHOULD FIX, owned by Phase 11** (non-critical lines; see `docs/LIMITATIONS.md` *Open items*):

6. H6: release assets and `SHA256SUMS.txt`.
7. README rewrite, which re-checks H1/H2 and carries H3, H4 and C5.
8. Licence and notice cleanup: "(verify)" markers, the `docs/LICENSES.md` row, the
   no-pretrained-weights line.
9. CI runs `make report`, `make bench-check` and `make figures` with a diff.

**NOTE:** `REPO_SETUP.md` starter-kit text; the GIF manifest `ila` commit; the absolute paths in
records; per-run Phase 6/7 verdict JSONs.
