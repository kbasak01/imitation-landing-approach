# VALIDATION_CHECKLIST.md — imitation-landing-approach

Walked line by line in Phase 10. Each line gets **met / met-with-caveat / unmet** and the evidence
(file, test, artifact, commit) in `docs/LIMITATIONS.md`. A line marked ★ is critical: unmet ★ blocks
publication.

## A. Inheritance from P8

- [ ] ★ A1. Submodule pinned at the commit in `results/p8_provenance.json`; `external/` unmodified
  (`git -C external/rl-land-on-moving-deck status --porcelain` empty; hook log shows no blocked edits overridden).
- [ ] ★ A2. MANIFEST SHA-256 `e6f30e55…` and success-YAML SHA-256 `c7fbdcc4…` re-verified by `make verify-p8`.
- [ ] ★ A3. Reproduction gate: `pid_feedforward` (2 800 episodes) and `ppo` s0 (`id` SS5–SS6) rows equal
  P8's committed rows with rendering on — re-run on the final tree.
- [ ] A4. Statistics functions are P8's by identity (test), not re-implementations.
- [ ] A5. Teacher label path parity (ONNX vs SB3) ≤ 1e-6, or SB3 path used — recorded.

## B. Observation contract and leakage

- [ ] ★ B1. No privileged field reaches any student input (field-name test; `StudentInput` has no slot).
- [ ] ★ B2. No world-fixed texture in rendered frames (background-invariance test).
- [ ] B3. `time_fraction` only in the `+clock` arm; `last_action` absent in `−last_action`.
- [ ] ★ B4. Image t and obs t share one `stepSimulation` count (timing test).
- [ ] ★ B5. Rendering, the visual twin and the recolour leave physics bit-identical (scripted-episode test).
- [ ] B6. Appearance-shift variants never appear in any training store (store attrs).

## C. Data

- [ ] ★ C1. Every demo/DAgger/DART episode comes from P8's `train` pool; none from `tune` or any frozen list
  (realization-key test against all lists).
- [ ] ★ C2. Selection used `splits/select.parquet` only; no frozen-list episode was flown before P3-D1 froze
  (timestamps of `results/e08/` vs the P3-D1 approval entry).
- [ ] C3. `label[t] == teacher(teacher_obs[t])` on a 1 000-step random audit per store.
- [ ] C4. Re-rendering 20 random saved states reproduces the stored frames byte for byte.
- [ ] C5. Kept-fraction after success filtering reported per teacher.
- [ ] ★ C6. Labelled-episode counts: `bc_dagger` = `bc_matched` = 2 000 per seed; `bc_dart` = `bc` = 1 000.
- [ ] C7. DAgger labels for stateful teachers produced by step-wise shadowing with `reset` per episode.

## D. Training

- [ ] ★ D1. Every planned run present; none dropped; excluded runs listed with reason in the protocol.
- [ ] D2. Each run's manifest: config SHA, data SHA, P8 commit, `ila` commit, checkpoint SHA, wall clock, GPU.
- [ ] D3. Smoke tests: overfit-8 MSE and closed-loop ≥ 7/8 for `bc` and `dp`.
- [ ] D4. Diagnostician verdict per run (TRUSTWORTHY / CAVEAT / EXCLUDE) recorded.
- [ ] D5. Budget rule applied at most once, before any evaluation number was read.
- [ ] D6. DART σ chosen on `select` within its 3-trial budget, logged.

## E. Diffusion Policy specifics

- [ ] ★ E1. Chunk-index tests pass; `Ta ≤ Tp − To + 1` enforced.
- [ ] E2. Toy bimodal test: DP recovers both modes; MSE-BC collapses to the mean.
- [ ] E3. Per-episode noise generator: same episode → same chunk across batch sizes on CPU.
- [ ] E4. `Ta`/`K`/ensembling sweeps run on the same checkpoints that were selected at the defaults.
- [ ] E5. R1 (LeRobot reference) result reported, or its absence explained; never mixed into core tables.

## F. Evaluation and statistics

- [ ] ★ F1. Every student flew the exact frozen lists its arm declares; start-state check on every reset.
- [ ] ★ F2. Teacher, `pid_feedforward`, `pid_track_descend`, `oracle_gated` printed beside every student.
- [ ] F3. Success with Wilson CI and N; all six outcome classes; never pooled across sea states.
- [ ] ★ F4. H1–H5 scored exactly as frozen in P3-D1; scoring code unit-tested on synthetic cases.
- [ ] F5. Every SS6 contrast quoted with P8's one-ulp flip rate beside it.
- [ ] F6. Determinism contract (P2) met for every evaluation configuration, or the deviation recorded.
- [ ] F7. Any student above its teacher in a cell investigated (exploit search, touchdown audit) before reporting.
- [ ] F8. `make report` re-renders `results/results.md` byte-identically.

## G. Deployment

- [ ] G1. ONNX parity (fp32) within thresholds per graph; fp16 reported.
- [ ] G2. Latency methodology: ≥ 200 warm-up, ≥ 2 000 timed, synchronised; machine stated; no transfer claim.
- [ ] G3. Latency-in-the-loop delay derived from the measured p99, by rule.
- [ ] G4. Closed-loop export parity judged against the determinism contract.

## H. Claims and publication

- [ ] ★ H1. No sentence implies real cameras, real decks, real flight or sim-to-real (claim-trace word list clean).
- [ ] ★ H2. Every number in README traces to a committed file (claim-trace).
- [ ] H3. Novelty stated as in plan §0 positioning; Angelis et al. described accurately.
- [ ] H4. GIFs labelled hand-picked; each re-flown and matched to its committed row.
- [ ] H5. Licences and third-party notices complete; no demo images in git.
- [ ] H6. Release assets with `SHA256SUMS.txt`.
