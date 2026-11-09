# LIMITATIONS.md — validation checklist verdicts and known limits

Phase 10 walk of `VALIDATION_CHECKLIST.md`, 2026-11-09. First walk at `8f4c9e0` (= `main` after Phase 9);
re-walked after the Phase 10 fixes on branch `phase-10`. Everything here is **simulation only** (PyBullet physics, PyBullet TinyRenderer).

**How verdicts are assigned.**
- **met:** a committed artifact, a passing test or a recorded SHA shows the line holds as written.
- **met-with-caveat:** the line holds, with a stated scope limit or an evidence gap. The gap is
  listed, with its fix, in [Open items](#open-items).
- **unmet:** the line does not hold yet.
- ★ marks a critical line; an unmet ★ blocks publication.

**Sources of evidence.**
- The test names are from this phase's full `make test` runs: at `8f4c9e0`, 1 244 passed and 0
  failed (23 min 25 s) on an unchanging tree. The final-code run is recorded in the Gate 10 entry
  of `docs/protocol.md`.
- The checks re-run on the final tree are listed in [Re-runs on the final tree](#re-runs-on-the-final-tree).
- Reviewer findings are in `docs/audit_report.md`.

**Tally.** 47 lines (17 ★): 34 met, 13 met-with-caveat, 0 unmet. The 17 ★ lines are 15 met and 2
met-with-caveat (A1, C2). After the first walk, the open items were closed in this phase (see *Open
items*).

## Re-runs on the final tree

| Command | Result |
|---|---|
| `make verify-p8` | `all_match: true`; 8/8 checks; `results/p8_provenance.json` unchanged (`git diff` empty) |
| `make reproduce-p8` (out-dir in scratch, render full, 24 workers) | `pid_feedforward` 2 800/2 800 and `ppo` 400/400 rows identical, 0 differ, 0 missing. `comparison.json` byte-identical to the committed one. Run record: `results/reproduce_p8/rerun_phase10/` |
| `make report` (ends in `git diff --exit-code results/results.md`) | exit 0, byte-identical (951 lines) |
| `make figures` then `git status` | `results/e08/figures/` unchanged |
| `make bench-check` | `results/latency` re-renders byte-identically; `d0_check.json` agrees with its rows files |
| `make bench-recheck-reference` | Decision unchanged (`reference_equals_selection_rows` re-decided True, same rows SHA-256s). It rewrites only its `ila_commit`/`time_utc` stamps; restored, since it is a write mode, not a check |
| `scripts/score.py --out <scratch>` + `scripts/above_teacher.py --out <scratch>` | 17/17 files byte-identical to `results/e08/`: `summary`, `baselines`, `aggregate`, `contrasts`, `hypotheses`, `above_teacher`, `missing` and `h2_touchdown_times` CSVs, `hypotheses.json`, `score_info.json`, and `above_teacher/*.json` + `audit.md` |
| `make sensitivity-check` (Phase 10) | records agree with their rows; `sensitivity.csv` re-renders byte for byte (also runs without `artifacts/`) |
| `make release-check` (Phase 10) | 86 assets, 16.68 GiB, every SHA-256 equal to its committed record; `results/release/{SHA256SUMS.txt,manifest.json}` identical |
| `make lint` | ruff clean, format clean (163 files), mypy strict clean (81 source files) |
| `make test` | `make test` at `8f4c9e0`: 1 244 passed, 0 failed (23 min 25 s), on an unchanging tree |

## A. Inheritance from P8

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| A1 Submodule pinned; `external/` unmodified | ★ | met-with-caveat | The gitlink `7237b70…` equals `p8_commit` in `results/p8_provenance.json`, and both nested submodules are at their pinned SHAs (dmf `e9fa15cc`, gym-pybullet-drones `7ebad1ec`). `git status --porcelain` is empty in P8 and in both nested submodules. The gitlink has one commit in history (`7f31ce4`, the pin). Caveat: `.claude/hooks/guard_paths.py` blocks edits under `external/` (exit 2) but keeps no log, so "no blocked edit overridden" rests on the clean porcelain and the single gitlink commit |
| A2 MANIFEST `e6f30e55…`, success YAML `c7fbdcc4…` re-verified | ★ | met | `make verify-p8` on the final tree: `manifest_sha256` and `success_yaml_sha256` match, along with the PPO s0 zip, VecNormalize and ONNX SHAs (`all_match: true`) |
| A3 Reproduction gate re-run on the final tree | ★ | met | Re-run at `8f4c9e0` with render full: 3 200/3 200 rows identical (`results/reproduce_p8/rerun_phase10/comparison.json`, byte-identical to the Phase 2 file). Covers the full-audit 200-episode `id` SS5 `pid_feedforward` spot-check |
| A4 Statistics are P8's by identity | | met | `tests/test_p8_adapter.py::test_stats_reexport_is_identity` (`is` over `rld.eval.stats.__all__`) and `::test_stats_namespace_covers_p8_all_exactly`; `src/ila/eval/stats.py` calls only `p8_adapter.stats.*`. The non-P8 statistics (joint-seed, W₁, H4 three-arm) are declared in P3-D1 §1 and have known-answer tests |
| A5 Teacher label parity ≤ 1e-6 or SB3 used | | met | `results/label_parity.json`: SB3 vs ORT CPU 1-thread max-abs 3.58e-7 ≤ 1e-6 (`passed: true`). Collection and DAgger labels use SB3 anyway (P2-D1 item 2) |

## B. Observation contract and leakage

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| B1 No privileged field reaches any student input | ★ | met | Tests:<br>• `tests/test_obs_split.py::test_student_input_has_no_field_for_privileged_data`, `::test_student_input_rejects_full_observation`, `::test_contract_refuses_privileged_in_proprio`, `::test_no_literal_obs_slices_outside_split`<br>• `tests/test_data_store.py::test_exact_student_columns`<br>• `tests/test_eval_runner.py::test_student_receives_only_contract_input_and_pixels`<br>• `tests/test_render_access.py` (audit scan; two reviewed allowlist entries: `bc_state`, `modular` targets)<br>• ONNX graph inputs: `tests/test_deploy_export.py::test_graph_entry_points_take_only_allowlisted_inputs`<br>R1 `input_features` = image + 13-dim state. Leak audit 2026-11-09: GO |
| B2 No world-fixed texture | ★ | met | `tests/test_render_scene.py::test_background_identical_under_translation`, with a power control: `::test_background_test_has_power_on_p8_checkerboard`; `::test_no_world_fixed_visual_on_moving_deck` |
| B3 `time_fraction` only in `+clock`; `last_action` absent in `−last_action` | | met | `tests/test_obs_split.py::test_plus_clock_appends_time_fraction`, `::test_minus_last_action_drops_it`, `::test_contract_refuses_clock_outside_plus_clock`. The leak auditor read every checkpoint's `obs_variant`: `plus_clock` only in `bc_plus_clock` ×3; `minus_last_action` only in `bc`/`dp_minus_last_action` ×3 each; all core arms `default` |
| B4 Image t and obs t share one `stepSimulation` count | ★ | met | `tests/test_render_scene.py::test_frame_and_obs_share_physics_step_count`; `tests/test_eval_runner.py::test_timing_every_step`, `::test_timing_assertion_fires_on_a_mismatched_step`. At runtime `runner._timed_frame` raises on a mismatch. All 185 200 `results/e08/rows/**` rows have `frames_rendered == steps + 1` |
| B5 Render, twin, recolour leave physics bit-identical | ★ | met | `tests/test_render_invariance.py::test_airborne_300_steps_identical`, `::test_descent_to_contact_identical`, `::test_reused_env_reattach_identical`, `::test_negative_control_collision_twin_is_detected`. A3 re-run on the final tree, render full, rows identical |
| B6 Appearance-shift variants never in a training store | | met | All 31 stores (6 demo + 25 DAgger iterations), file level and every episode: `camera_cfg_sha256` is the default `53f80265…` and matches none of the 5 shift SHAs. The leak auditor read the attributes from the data root. Guards are tested by `tests/test_data_collect.py::test_appearance_shift_refused` and `tests/test_data_store.py::test_writer_refuses_non_committed_camera` |

## C. Data

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| C1 Every demo/DAgger/DART episode from P8's `train` pool | ★ | met | Demo and DART stores: `results/demos_summary.json` → `collections.*.attr_audit`, 0 violations for all 6 stores (outside train, `tune`, each of 5 frozen lists, SS6, 90°). DAgger: 25/25 iteration audits, 0 attribute violations (`results/phase5/dagger_audits.csv`, committed this phase). Tests: `tests/test_data_episodes.py::test_train_disjoint_from_frozen[*]`, `::test_train_is_train_only`, `::test_select_is_tune_only` |
| C2 Selection on `select` only; no frozen list flown before P3-D1 froze | ★ | met-with-caveat | All 478 selection rows files (71 700 rows) have `regime=select`. The σ search used `select` (`results/phase4/dart_tune/sigma_star.json`). `tests/test_render_access.py::test_training_code_reads_no_frozen_list`. P3-D1 froze 2026-10-27/28 (`2b9ae07`, 2026-10-28T01:17Z); e08 began 2026-11-06T01:10Z (`results/e08/run_info.json` `invocations[0].started_utc`). Caveat: before the freeze, frozen lists were flown by scripted actions (Phase 1 tests) and by `pid_feedforward`/`ppo` (the Phase 2 gate), never by a student (P1-D2, P2-D1) |
| C3 `label[t] == teacher(teacher_obs[t])`, 1 000-step audit per store | | met | `results/demos_summary.json` `relabel_audit`: 1000/1000 identical, max-abs 0.0, for each of the 6 stores. DAgger: 25 × 1000/1000, max-abs 0.0 (`results/phase5/dagger_audits.csv`) |
| C4 Re-rendering 20 saved states byte for byte | | met-with-caveat | `results/demos_summary.json` `replay_audit`/`refly_audit`: 20/20 frames byte-identical for each of the 6 stores. DAgger: 25 × 20/20 (`results/phase5/dagger_audits.csv`). Caveat: a "saved state" is re-simulated from the listed reset plus the stored `action_exec` (`src/ila/data/audit.py`), not restored from a PyBullet state snapshot |
| C5 Kept fraction per teacher | | met | P3-D1 table and `results/demos_summary.json` `kept_fraction_success_filter`: `ppo` 0.9995 (1 999/2 000), `pid` 0.998 (998/1 000), `mix` 0.978 (978/1 000). DART is unfiltered by design (would-be kept 995/751/154). The README prints the kept counts (*Setup in brief*) |
| C6 Labelled-episode counts | ★ | met | Run manifests `labelled_episodes_total`, every seed: `bc_dagger` and `bc_matched` stages 1200/1400/1600/1800/**2000**; `bc` **1000** (999 kept); `bc_dart` **1000**. Labelled frames 616 053 vs 614 365 printed with H1 (P8-D1 item 8) |
| C7 Stateful DAgger labels by step-wise shadowing with reset | | met-with-caveat | `src/ila/data/dagger.py` resets the teacher per episode and calls it every step before the coin. `tests/test_data_dagger.py::test_recorder_shadows_teacher_and_selects_by_coin` uses a stateful double; the `mix_1000` relabel audit re-labels 725 `gated` steps identically from reset. Caveat: the only DAgger teacher flown is `ppo` (stateless), so the stateful path is exercised in tests, not in a production DAgger run |

## D. Training

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| D1 Every planned run present; none dropped | ★ | met | 71/71 selections pinned in `results/e08/selection_pins.json`: 35 core (7 arms × 5), 33 ablation (11 × 3), 3 R1, plus 3 σ trials. "No exclusions" recorded in P4-D4, P5-D2, P6-D5, P7-D3 and Gate 7. `results/e08/missing.csv` is header only; 961/961 cells. The diagnostician re-hashed all 71 pins and checkpoints and re-applied the selection rule: 71/71 match |
| D2 Manifest fields | | met-with-caveat | All 71 manifests carry the config SHA (`all_frozen: true`), data SHA (`matches: true`), P8 commit `7237b70`, clean `ila` commit, checkpoint SHAs, `wall_clock_s`, `gpu_hours` and `host.gpu`. Caveat (disclosed): `bc_matched`'s normaliser is not in its manifest (P4-D5), and Phase 7 `selection.json` lacks the `ila` commit (P7-D3 item 1) |
| D3 Smoke tests for `bc` and `dp` | | met-with-caveat | `results/phase4/smoke_bc.json`: MSE 8.9e-7 < 1e-4, 8/8. `results/phase6/smoke_dp.json`: MSE 5.2e-5 < 1e-3 (threshold frozen in P6-D1 before the result), 7/8. Caveats: `dp`'s 7/8 was a favourable draw, 3 of 10 noise keys passing (P6-D4); `smoke_bc` ran on a tree dirty only with `?? results/phase4/` |
| D4 Diagnostician verdict per run | | met-with-caveat | P4-D4, P5-D2, P6-D5 and P7-D3 tables cover all 71 runs: 39 TRUSTWORTHY, 32 CAVEAT, 0 EXCLUDE. Per-run JSONs exist for Phases 4–5 only. The Phase 10 re-review re-rates core `dp` s0–s4 TRUSTWORTHY → CAVEAT on later evidence (P10-D1 item 1), giving 34 T / 37 C / 0 EXCLUDE |
| D5 Budget rule at most once, before evaluation | | met | P4-D2 "no cut", committed `32c8cf9` 2026-10-28, before the first `select` rows. No later cut. Measured 95.41 GPU-h of 120 (sum of the four `runs_summary.csv`) |
| D6 DART σ on `select` within 3 trials | | met | P4-D3; `results/phase4/dart_tune/sigma_star.json`: σ 0.05/0.10/0.20 → 149/150/149 of 150, σ* = 0.10. Decided by one episode (disclosed); `bc_dart` s0 is the σ = 0.10 trial (P4-D5 item 2, user-approved) |

## E. Diffusion Policy specifics

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| E1 Chunk-index tests; `Ta ≤ Tp − To + 1` enforced | ★ | met | `tests/test_dp_chunking.py` (table tests): `::test_executed_starts_at_to_minus_one_not_zero`, `::test_executed_indices_table`, `::test_max_action_steps`, `::test_ta_out_of_range_refused`, `::test_ensemble_sources_refuse_inadmissible_ta`. The diagnostician's ±1 label shift gives 506× worse MSE, and R1 parity confirms the execution indices |
| E2 Toy bimodal: DP keeps both modes, MSE-BC collapses | | met | `tests/test_dp_toy.py::test_diffusion_recovers_both_modes`, `::test_mse_bc_collapses_to_the_mean` |
| E3 Per-episode noise generator, same chunk across batch sizes on CPU | | met-with-caveat | `tests/test_policies.py::test_sample_slot_independent_of_batch`, `::test_noise_seed_is_stable_and_distinct`, `::test_noise_seed_pinned_value`; `tests/test_policies.py::test_gpu_slot_noise_independent_of_batch`. Caveat: at a fixed batch size a slot is bit-identical wherever it sits; across batch sizes the same noise goes in, but outputs agree only to the last ulps (CPU kernels differ by batch size). Every committed row is from batch 1 on the CPU single-sample path |
| E4 Sweeps on the selected checkpoints | | met | `results/dp_sweeps.json`: checkpoint SHAs `b459f276`/`f2d66322`/`40c71347`/`ac8dfbe6`/`0179fdc7` equal the selected ones; `reproduces_selection_rows: true` ×5. e08 `dp_Ta`, `dp_K`, `appearance_shift` and `latency_arm` sidecars carry the pinned SHA per seed |
| E5 R1 reported or absence explained; never in core tables | | met | `results/results.md` §7e (tag `reference_external`, "not gating"). R1 appears in `aggregate.csv` only under `reference_external`, and 0 times in `contrasts.csv`/`hypotheses.csv`. The recipe gap is in P6-D5 items 8–9 |

## F. Evaluation and statistics

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| F1 Exact frozen lists; start-state check on every reset | ★ | met | `runner.check_start` on every reset (batched path included); `matrix.check_rows` checks list order, 200 per cell, regime and render SHA. Leak auditor's re-check: 185 200 e08 rows, 0 mismatches against P8's lists on (ss, index) → (episode_seed, realization_seed, heading, vessel); every arm flew exactly its declared lists |
| F2 Teacher and three baselines beside every student | ★ | met | `results/results.md` §2–§4 and §7a–§7f: teacher, `pid_feedforward`, `pid_track_descend` and `oracle_gated` in every tagged section, from `results/e08/baselines.csv` (P8's committed rows). `oracle_gated` labelled "commit-timing oracle, privileged, not a bound" |
| F3 Wilson CI and N; six outcome classes; no pooling | | met | `results/e08/summary.csv`: `success_wilson_lo/hi` and `n_episodes` on all 961 rows. All six classes in `results.md` §3 and §7f. No core table pools sea states. The one pooled sentence planned for the README (P8-D1 item 11) must be split per sea state or labelled descriptive (P10-D1 item 4) |
| F4 H1–H5 scored as frozen in P3-D1; synthetic tests | ★ | met | P3-D1 block hash recomputed `00e22330…3fc5`, as at Gate 3; every §1–§3 config SHA matches. `results/e08/hypotheses.{csv,json}`. Synthetic branch tests in `tests/test_eval_hypotheses.py` (`test_h1_*` … `test_h5a_*`, `test_magnitude_rule`, `test_noninferiority_rule`); `tests/test_deploy_latency.py::test_delay_is_ceil_of_p99_over_the_control_step`; `tests/test_eval_score.py::test_full_matrix_scores_every_hypothesis` |
| F5 Every SS6 contrast with P8's one-ulp flip rate | | met-with-caveat | `results/e08/contrasts.csv`: 33/33 SS6 rows carry `flip_floor`; `results.md` §5–§6 and every SS6 number in the README print it. The `dp_*` ablations' own sensitivity is now measured (`results/phase10/dp_sensitivity/sensitivity.csv`, P10-D2) and quoted beside them in the README (L4). Caveat: older protocol prose (P8-D1 item 9, P9-D2 item 5, the Gate 8 row) quotes some SS6 contrasts without the floor. Rule (P10-D1 item 5): quote from `contrasts.csv`/`hypotheses.json`, never from that prose |
| F6 Determinism contract met for every configuration | | met-with-caveat | Every e08 row flew the CPU fp32 single-sample path (`results/e08/run_info.json` `inference_path`, P8-D1 item 2), so the GPU re-measurement is moot. Bit-identity is re-checked by the P9-D2 reference re-flights, `results/latency/d0_check.json` (0 flips) and the GIF re-flights. Caveat: `dp` flips 32–44 of 150 `select` episodes under numeric perturbations of order 1e-5 (P9-D2 item 1). Byte-identical `dp`, `dp_*` and R1 rows therefore hold only on this software stack (torch 2.13, CPU fp32, i9-10980XE); another BLAS build would move about a quarter of `dp` outcomes while leaving rates near 58–67 of 150 vs 63 |
| F7 Above-teacher cells investigated before reporting | | met | `results/e08/above_teacher.csv` (6 cells); `above_teacher/audit.md` and per-cell JSONs (tunnelling, detector disagreement, near-threshold, grace, McNemar); `reviewed.json` (6 released, results-skeptic, 2026-11-06). The scan covers every group, `latency_arm` included |
| F8 `make report` byte-identical | | met | `make report` at `8f4c9e0`: exit 0, `git diff --exit-code results/results.md` clean |

## G. Deployment

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| G1 ONNX fp32 parity per graph; fp16 reported | | met-with-caveat | `results/latency/parity.csv`: the three scored fp32 providers pass; `dp` per-step ε̂ ≤ 2.96e-5 and `bc_dagger` action ≤ 3.04e-6 (bound 1e-4). fp16 is reported and fails (6.38e-2 / 9.65e-4). Caveats: `dp_K10`'s final chunk is reported (≤ 2.77e-5), not judged (P3-D1 §6); inputs are teacher-visited `select` states (P9-D1 item 3) |
| G2 Latency: ≥ 200 warm-up, ≥ 2 000 timed, synchronised; machine; no transfer claim | | met | `results/latency/latency.csv`: `warmup_iters` 200 and `timed_iters` 2000 on all rows; synchronised brackets via `p8_adapter.latency_harness` (identity test in `tests/test_deploy_latency.py`); machine in `results/latency/jobs/*.json` `env` (i9-10980XE, RTX A4000, WSL2); "no embedded target… nothing transfers" in `results.md` §7f. Latencies are wall clock on this machine and are not Froude-scaled |
| G3 Delay from measured p99, by rule | | met | `results/latency/delay.json`: `bc_dagger` d = 1 (`torch_cuda_fp32`, 4.32 ms); `dp` d = 6 (fallback `torch_cpu_fp32`, 181.3 ms; ⌈181.33/33.33⌉ = 6). The fallback and its 1 thread were fixed in `configs/deploy/bench.yaml` at `9a34b8c`, before results. Pinned in `results/e08/latency_pins.json` |
| G4 Closed-loop export parity against the determinism contract | | met-with-caveat | `results/latency/closed_loop.csv`, ≤ 3 of 150 flips (P3-D1 §6): `bc_dagger` 0–2 flips, **pass**. `dp` 32/33/44 on the scored providers (46 on unscored TRT fp32), **fail**, read as outcome sensitivity (P9-D2 item 1). Gate 9's "PASS (test run as frozen)" means the test ran as frozen, **not** that the exported `dp` matches |

## H. Claims and publication

| Line | ★ | Verdict | Evidence |
|---|---|---|---|
| H1 No real-camera/deck/flight/sim-to-real wording | ★ | met | Forbidden-list grep over `README.md`, `results/results.md`, `docs/protocol.md`, `results/e08/gifs/RULE.md`, `results/e08/above_teacher/audit.md`, `results/phase7/diagnostics/verdicts/README.md` and this file. Only ordinal "first", negated "embedded"/"hardware", and software-sense "real" ("real path", "real schedule") in the protocol. Re-check in Phase 11 on the real README |
| H2 Every README number traces to a committed file | ★ | met | Phase 10 README. Every number was traced by `results-skeptic` and `repo-publication-reviewer` against `results/results.md`, `results/e08/*`, `results/latency/*`, `results/demos_summary.json`, `results/phase*/runs_summary.csv`, `results/phase10/dp_sensitivity/` and `results/reproduce_p8/rerun_phase10/`. All BLOCKING and SHOULD FIX items of both reviews were applied; the re-review is recorded in `docs/audit_report.md` §5 |
| H3 Novelty as plan §0; Angelis et al. accurate | | met | README *How this relates to prior work*. It follows plan §0 and `SOURCES.md` §3: Angelis et al. (arXiv:2605.23717) use PPO on 25 sparse keypoints end to end, with sinusoidal per-DoF platform motion, hardware tests and no teacher, demonstrations or distillation. No novelty is claimed for distillation, Diffusion Policy or vision-based landing, and P8 is credited. The paper was not re-read in Phase 10; the description rests on the 2026-10-23 reading |
| H4 GIFs labelled hand-picked; re-flown and matched | | met | `results/e08/gifs/RULE.md` was fixed first (its SHA is in `manifest.json`). Every frame is burned with "hand-picked by stated rule · simulation (PyBullet TinyRenderer)". The README gallery is captioned "hand-picked by a stated rule" and links RULE.md. The 4 GIF SHAs match the manifest, and the picks recomputed from the committed rows match. `src/ila/viz/gifs.py` raises before writing on any re-flight mismatch. The contact-dwell background frames are disclosed (L3). The manifest still records no `ila` commit; the evidence is the raise-before-write path plus the matching SHAs |
| H5 Licences and notices; no demo images in git | | met | MIT `LICENSE`; `THIRD_PARTY_NOTICES.md` covers P8, dmf, gym-pybullet-drones, diffusion_policy (MIT text; upstream `main` at `5ba07ac`, named in the `unet1d.py` header too), diffusers, LeRobot, PyBullet and PyTorch. `docs/LICENSES.md` §3: the Phase 6 rows; no pretrained weights; the release assets are MIT. `SOURCES.md`/`REPO_SETUP.md` "(verify)" markers are resolved. No demo images at HEAD or in history; the only images are 6 PNGs and 4 GIFs (`53b56bd`, P10-D2) |
| H6 Release assets with `SHA256SUMS.txt` | | met-with-caveat | Draft GitHub release `phase-11-complete` (not public): 86 assets (68 pinned checkpoints as stored, 3 R1 exports × 3 files, 8 Phase 9 graphs + `graphs_meta.json`) plus `SHA256SUMS.txt` and `manifest.json`. Every SHA-256 comes from a committed record (`selection.json` pinned by `selection_pins.json`; `graphs.json`) and was re-hashed (`make release-check`); `sha256sum -c` passes on the staged files. Uploaded sizes equal the manifest; the uploaded `SHA256SUMS.txt`/`manifest.json` are byte-identical to `results/release/`. Caveat: the release is a draft until Phase 11 publishes it with the tag. P8's `ppo` checkpoint is not released (README *Reproducing this*) |

## Open items

All open items from the first walk are closed in Phase 10. The commits are on `phase-10`.

| # | Line(s) | Item | Fix | Status |
|---|---|---|---|---|
| 1 | H6 | No release, no `SHA256SUMS.txt` | `ila.release` + `make release`/`release-check`; `results/release/`; draft release with 86 assets (`53b56bd`) | closed (draft; publish in Phase 11) |
| 2 | H2, H3, H4, C5 | README was the Phase 0 stub | Phase 10 README, reviewed by `results-skeptic` and `repo-publication-reviewer` | closed |
| 3 | H5 | "(verify)" markers; stale licence row; no-pretrained-weights line | `53b56bd` and the README commit | closed |
| 4 | D4 | Core `dp` verdict TRUSTWORTHY predates the evidence | P10-D1 item 1 re-rates `dp` s0–s4 CAVEAT (`a5bd97f`) | closed |
| 5 | C1, C3, C4 | DAgger audits uncommitted; P5-D2 cited a wrong path | `results/phase5/dagger_audits.csv`; P10-D1 item 2 (`a5bd97f`) | closed |
| 6 | A3, B5 | Reproduction gate not re-run since `5d09486` | Re-run at `8f4c9e0`, 3 200/3 200 identical (`a5bd97f`) | closed |
| 7 | F5, F6, G4 | Wording risks for `dp` and SS6 contrasts | README rules (P10-D1 items 3–7) applied in the README | closed |
| 8 | — | CI does not exercise the results path | CI runs `make report`, `bench-check`, `sensitivity-check`, `figures` + diff (`53b56bd`, `d8a21df`) | closed |
| 9 | L4, F5 | `dp_*` closed-loop sensitivity not measured | `make sensitivity` (`fa364b1`); results `d8a21df`; P10-D2 | closed |
| 10 | — (P4-D5 item 2) | The binding `bc_dart` s0 footnote and seeds 1–4 sensitivity row were never produced in Phase 8 (found by both README reviews) | The scorer and report add them from the committed rows; footnote ‡ on every `bc_dart` row in `results/results.md` and the README (P10-D2) | closed |

## Limitations (for the README)

Every number below traces to the committed file named beside it.

- **L1 Simulation only.** PyBullet physics and PyBullet's TinyRenderer, P8's heave model and vessel
  motions. Nothing here was flown, filmed or measured on a real vehicle, camera or deck. No
  sim-to-real claim is made or supported.
- **L2 What a rendered camera can show.** The frame comes from TinyRenderer at the camera of
  `configs/render/camera.yaml`, with a single-colour ground (P8's checkerboard recoloured) and the
  deck marking. There is no sensor noise, blur, exposure change, rolling shutter or lens model. The
  appearance-shift test covers 5 variants only (hue, light, contrast, ±2° mount tilt;
  `results/results.md` §7c), and robustness claims stay inside them.
- **L3 Background-only frames at contact.** Once the vehicle rests on the deck, the camera
  (`offset_body_m` −0.02 m) sits below the deck surface, and the frame is uniform background until
  the episode ends. This affects no physics or outcome. The same frames are in the demo stores, so
  students see them too. The GIFs show it at every touchdown.
- **L4 Outcome sensitivity of the Diffusion-Policy students.** Seed 0, 150 `select` episodes,
  torch CPU fp32 reference vs the same sampler in torch on the GPU (numeric differences of order
  1e-5). The flips are `dp` 32, `dp_mix` 22, `dp_pid` 5, `dp_n100` 43, `dp_n300` 35 and
  `dp_minus_last_action` 37 of 150 (`results/phase10/dp_sensitivity/sensitivity.csv`, P10-D2).
  - `dp`'s 32 reproduces Phase 9 exactly, on the same episodes.
  - Every reference flight equals its selection rows byte for byte.
  - The ORT providers gave `dp` 33–44 (P9-D2 item 1).
  - Byte-identical `dp`, `dp_*` and R1 rows hold only on this software stack: torch 2.13.0, CPU
    fp32 single-sample path; for R1, ONNX Runtime CPU, one thread.
- **L5 Ceilings and thin margins.** H1's primary is inconclusive and ceiling-limited. H1-SS6 is
  supported by +5.9 [+2.3, +9.7] against a +5 margin (+5.1 with tunnelled successes removed)
  (`results/e08/hypotheses.json`). Lead with the primary.
- **L6 Latency.** All latencies are wall clock on one workstation (i9-10980XE, RTX A4000, WSL2;
  `results/latency/jobs/*.json`) and are not Froude-scaled; nothing transfers to an embedded target.
  - H5(b)'s d = 6 is the delay of the 1-thread CPU fallback (181.3 ms p99,
    `results/latency/delay.json`), not of a GPU deployment. d = 2 under ORT CUDA was not flown.
  - The `bc_dagger` delay cost is the cost of one 33.3 ms control step (model scale, 30 Hz; 166.7 ms
    full scale at λ = 1/25), with no teacher-under-delay control.
- **L7 Teacher-visited parity inputs.** Open-loop ONNX parity is measured on teacher-visited
  `select` states, not student-visited ones (P9-D1 item 3).
- **L8 Tuning asymmetry.** `bc_dart` had the one permitted search (σ* = 0.10 on `select` over 3
  seed-0 trials, decided by one episode; `results/phase4/dart_tune/sigma_star.json`). `bc` is
  untuned.
- **L9 Tunnelled successes.** P8 counts a success even when the touchdown penetration exceeds
  5 mm (P8's `tunnelled` flag, not reclassified). Core `dp` has 354 such successes of 5 356
  (6.6 %), and the latency arm 8.4 % (results-skeptic scan of `results/e08/rows/**`). The H1 and
  H1-DART contrasts print success minus tunnelled successes beside them
  (`results/results.md` §5). Tunnelled successes favour the Ta = 8 side of H3, so its direction
  stands.
- **L10 Run paths.** Committed run records carry workstation-absolute paths (`/home/kbasak/...`).
  They are provenance, not instructions.
