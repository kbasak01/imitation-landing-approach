---
name: landing-protocol-p8
description: The inherited P8 evaluation protocol as it applies to imitation students — frozen episode lists and hashes, success criteria, outcome classes, metrics, statistics (Wilson, IQM with stratified bootstrap, paired bootstrap, relative p95), always-printed baselines, the ceiling and determinism caveats, and reporting rules. Use whenever writing or reviewing evaluation code, tables, figures, hypothesis scoring, the README, or any sentence containing a success rate, closing speed, CI or comparison.
---

# Landing evaluation protocol (inherited from P8; this repo adds no criteria)

Authorities, in order: this repo's `docs/protocol.md` P3-D1, then P8's `docs/protocol.md` P3-D1
(read it in `external/rl-land-on-moving-deck/docs/protocol.md`). If this skill disagrees, they win.

## Frozen inputs (verified by `make verify-p8`)
- Episode lists: `external/.../results/episodes/`, MANIFEST SHA-256 `e6f30e55…`; 200 episodes per
  cell; `id` SS3–SS6, `unseen_seastate` SS6, `unseen_heading` SS3–SS6, `unseen_vessel` SS3–SS6,
  `static`. Aft pad. Regimes share realizations: never present regime-vs-regime differences as
  independent draws.
- Success: `configs/env/success.yaml` SHA-256 `c7fbdcc4…` — closing speed along the deck normal
  ≤ 0.5 m/s, lateral offset ≤ 0.10 m, relative tilt ≤ 15°, contact ≥ 0.5 s with 50 ms grace.
- Outcome classes, matched in order: `crash → off_pad → hard_landing → bounce → success → timeout`
  (only `timeout` truncates).

## Every cell reports
Success with Wilson 95 % CI and N; all six outcome fractions; closing speed p50/p95 (deck normal;
world-z beside it); lateral offset p50/p95; relative tilt p95; time to touchdown p50; effort; jerk;
detector disagreement; tunnelling n. Students also: teacher agreement on own states; inference calls
per episode.

## Statistics (P8 functions through `ila.p8_adapter`; bootstrap seed 20261023)
- Per cell/seed: Wilson 95 %.
- Across seeds: IQM with stratified-bootstrap 95 % CI, 2 000 replicates, resampling runs within task.
- Contrasts: paired bootstrap over identical lists — seeds, then episodes — 10 000 replicates.
- Closing-speed contrasts: relative p95 over touched-down episodes only, never imputed.
- "Separates" = 95 % CI excludes 0. Magnitude thresholds judged on the point estimate.
- No multiplicity correction; say so.

## Reporting rules
- Teacher (P8 committed rows), `pid_feedforward`, `pid_track_descend`, `oracle_gated` beside every
  student. `oracle_gated` is a commit-timing oracle, not a bound.
- Learned cell format: `IQM [CI]; losses by class per 1 000` (P8's format). Baselines:
  `rate [Wilson] k/200; losses per 200`.
- Never pool sea states in a headline. Shade SS6 and 90° cells as outside the training distribution.
- Ceilings: P8's PPO teacher is at 100 % in `id` SS3–SS5; a degenerate CI there records a ceiling,
  not precision.
- Every SS6 contrast carries P8's one-ulp flip rate (~6.6 % of `id` SS6 policy-episodes) beside it.
- Tags never mixed into core sections: `reference_external`, `appearance_shift`, `latency_arm`,
  ablations, `+clock`.
- A student above its teacher in any cell is investigated before it is reported.
- Simulation only. A rendered TinyRenderer frame is not a camera image.
