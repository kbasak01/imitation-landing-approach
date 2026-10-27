# protocol.md — imitation-landing-approach decision log

Every decision, deviation, threshold and gate, dated. Where this file and
`docs/IMPLEMENTATION_PLAN.md` disagree, this file is what happened. Entries are appended, never
edited; a correction is a new entry that names the one it corrects.

Entry format:

```
### P<phase>-D<n> — <title> (<YYYY-MM-DD>)
*Decision.* … *Reason.* … *Evidence.* <files, SHAs> … *Consequence.* …
```

Gate entries: `### Gate N passed (<date>)` or `### Gate N failed, attempt k (<date>)`, with the
criterion table and the reviewer verdicts.

---

## Phase 0

### P0-D1 — Pins, submodule and provenance (2026-10-27)
*Decision.* P8 is pinned by SHA at `7237b7065c8af8172b3848ac6e9d3ceb2cf82075` (merge of PR #9,
"Gate 9 passed"). The PPO teacher is P8's `ppo` seed-0 run, checkpoint `final/`. This project adds
exactly four packages on top of P8's resolved environment (pinned below). LeRobot lives only in
`.venv-lerobot`.

*Reason.* P8's results must reproduce bit for bit (non-negotiable 3). Any package that
re-resolves torch, numpy or pybullet breaks that. Seed 0 is the teacher by declaration (plan §1.4),
not by performance.

*Evidence.*
- **Submodules** (`git submodule status --recursive`): P8 `7237b7065c8af8172b3848ac6e9d3ceb2cf82075`
  (`heads/main`); `third_party/deck-motion-forecast` `e9fa15cc35312a5c1ecf0c42668a690b5061c090`;
  `third_party/gym-pybullet-drones` `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`
  (`v1.0.0-324-g7ebad1e`). P8 has no release tag. `git ls-remote --tags` is empty, and the remote
  `main` is the pinned commit.
- **P8 working copy holding the artifacts:** `/home/kbasak/projects_02/rl-land-on-moving-deck` is
  at `9118c25` ("Gate 9 passed"), the second parent of the pinned merge. `git diff 9118c25 7237b70`
  is empty (identical trees). The checkpoints were trained from P8 `1d67488` (run
  `provenance.json`), and P8 itself certified them through Gate 9.
- **Main env `.venv`:** Python 3.12.3, pip 26.2.2. P8 is installed per REPO_SETUP §2 (dmf and
  gym-pybullet-drones editable, then `P8[dev]` with its exact pins): torch 2.13.0+cu130, numpy 2.5.2,
  pybullet 3.2.7 (built from sdist), gymnasium 1.3.0, SB3 2.9.0. CUDA is available on an RTX A4000
  (driver 596.71). Additions were resolved under a constraints file frozen from that environment:
  **torchvision 0.28.0+cu130, diffusers 0.41.0, h5py 3.16.0, einops 0.8.2**. They moved nothing
  already installed; the dry run listed only new packages. The full freeze is
  `results/env/pip_freeze_main.txt` (SHA-256 `ea1f699c…`).
- **Provenance** (`make verify-p8` → `results/p8_provenance.json`, SHA-256 `2b7aba29…`,
  `all_match: true`):
  - MANIFEST `e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2` and success.yaml
    `c7fbdcc4df48bd4a2cfe21811613e5e50e3e4868822fe6dddd13745d96692118` match P8 P3-D1.
  - `ppo/0/final/model.zip` `1a012394335c44d1e08d20f2199debaf3809d7bef96e04749817ae35b584fd49`,
    `vecnormalize.pkl` `9ddb2a992d4263426bc28033f8cd9f7f54daf7011ca67eb81f3099ecf289db6f` and
    `onnx/ppo_s0.onnx` `06230fd2de23468a169589ac41f1d19da7e8c60c7dc5ea68a1534f0b39281621` match the
    seed-0 entry P8 committed in `results/latency/run_info.json`.
  - `final/checkpoint.json` is hashed for the record. It is not compared, because P8 publishes no
    reference value for it.
- **LeRobot env `.venv-lerobot`:** lerobot 0.6.1 (Apache-2.0) resolved torch 2.11.0+cu130,
  torchvision 0.26.0, numpy 2.2.6 and gymnasium 1.4.0, all incompatible with P8's pins.
  Coexistence in `.venv` was therefore **not attempted**. Its dataset format is
  `CODEBASE_VERSION = "v3.0"`. Freeze: `results/env/pip_freeze_lerobot.txt` (SHA-256 `c00401da…`).
- `(verify)` items: all resolved; see `docs/LICENSES.md` §2.

*Consequence.* Only the two ONNX/LeRobot artifacts cross between the venvs (REPO_SETUP §3).
`ila.provenance` holds the expected hashes as constants. A `checkpoints`-marked test re-runs the
check on the workstation; CI skips it.

## Phase 1

*Before you start (from Phase 0).* `ila.p8_adapter` is a stub: it exposes `make_env(cfgs, listed)`,
built exactly as P8's runner builds it (`motion_for` + `make_env`), plus `load_eval_configs`,
`read_list`, `list_path` and the hash helpers. Phase 1 adds `obs_fields`, `teacher` and `stats`. The
teacher checkpoint is `ppo/0/final/` (P0-D1).

### P1-D1 — Plate visuals, camera, background (to fill)
- What `getVisualShapeData` reports for P8's plate and ground; twin or texture-in-place; why.
- Camera numbers as committed in `configs/render/camera.yaml`, with its SHA-256.

## Phase 2

### P2-D1 — Reproduction gate result (to fill)
### P2-D2 — Throughput and the budget it implies (to fill)
### P2-D3 — Determinism contract (to fill)

## Phase 3

### P3-D1 — FROZEN IMITATION-LEARNING PROTOCOL (draft in Phase 3; frozen only with the user's dated approval)

Sections to fill, in this order (mirrors P8's P3-D1):
1. Inherited: success YAML SHA-256, MANIFEST SHA-256, outcome classes, metrics, statistics (by
   reference to P8 P3-D1 §1–§4), bootstrap seed 20261023.
2. Pools and episodes: `train`/`select` manifests + SHA-256; demo counts per teacher.
3. Arms: every arm of plan §1.6 with every hyper-parameter, seeds, and its evaluation lists.
4. Selection rule and the only tuning (DART σ, 3 trials).
5. Training budget and the single-cut rule.
6. Hypotheses H1–H5: cell, statistic, magnitude, supported / not supported / inconclusive rules,
   restated against the measured teacher ceilings if needed (say why).
7. Latency-arm rule.
8. Deviations and incidents before freeze.

The SHA-256 of this block (heading to the line before the next `###`, UTF-8) is recorded in the
Gate 3 entry.

### Gate 0 failed, attempt 1 (2026-10-27)

| Criterion (plan Phase 0 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` green locally | ruff check + format check clean; `mypy --strict` clean on 46 files | PASS |
| `make test` green locally | 5 passed, including the `checkpoints` provenance test | PASS |
| `make smoke` green locally | 5 passed (`tests/test_smoke.py`: imports, one P8 env built through `ila.p8_adapter` and reset, obs (25,) finite, only the adapter imports `rld`, frozen-file hashes) | PASS |
| Green in CI (CI skips `checkpoints`) | `.github/workflows/ci.yml` is written, but no GitHub remote exists (`kbasak01/imitation-landing-approach` not found) and nothing is pushed. The CPU subset passes locally: 4 passed, 1 deselected | UNVERIFIED |
| `python -c "import rld, ila"` | `import OK` in `.venv` | PASS |
| `results/p8_provenance.json` matches P8's published hashes | `all_match: true` (P8 commit, two nested SHAs, MANIFEST, success.yaml, model.zip, vecnormalize.pkl, ONNX; P0-D1). The file is **not yet committed** | PASS on content; UNVERIFIED as a committed artifact |

Reviewers: none are assigned to Phase 0 by the phase-gate table.

Failing items and the smallest fix for each: (1) commit the Phase 0 tree on branch `phase-0`;
(2) create the public GitHub repo, push `phase-0` and get a green CI run. Both need the user's
approval. No threshold changes.

### Gate 0 passed (2026-10-27)

| Criterion (plan Phase 0 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` green locally and in CI | Local: ruff and format clean, `mypy --strict` 46 files. CI run 40518656883 / job 122245991307 (run 40520506247, commit `28ff2cc`): same, 46 files | PASS |
| `make test` / `make smoke` green locally | 5 passed (includes the `checkpoints` provenance test) | PASS |
| Tests green in CI (CI skips `checkpoints`) | Run 40520506247: 4 passed, 1 deselected | PASS |
| `python -c "import rld, ila"` | Works locally and in the CI install step | PASS |
| `results/p8_provenance.json` matches P8's published hashes | Committed; `all_match: true` (P0-D1) | PASS |

Reviewers: none are assigned to Phase 0.

Incident (attempt 1 → 2): the first CI run was green, but its mypy saw 39 files against 46 locally.
An unanchored `data/` in `.gitignore` had also matched `src/ila/data/` and `configs/data/`, so
seven skeleton modules and the starter kit's `configs/data/demos.yaml` were not committed. Root
patterns are now anchored (`28ff2cc`) and the re-run checks all 46 files. No threshold changed.
