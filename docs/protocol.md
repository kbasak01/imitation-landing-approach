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

### P1-D1 — Plate visuals, camera, background (2026-10-27)
*Decision.*
1. **Deck marking: a visual twin, not a texture on P8's plate** (`marking.strategy: visual_twin`).
   The twin is one body with mass 0 and no collision shape (`baseCollisionShapeIndex = -1`). Its
   visual is a `GEOM_MESH` quad: two triangles with explicit UVs and an upward normal, covering the
   plate's top face (0.80 × 0.52 m model scale), 1.0 mm above the plate's top surface in the plate
   frame. It carries the procedural helipad texture from `ila.render.texture`. That texture is
   generated at the face's aspect ratio, so texels are square and the ring (radius 0.15 m = P8's
   pad radius) stays circular. The twin is created after every `env.reset()`, after P8's four bodies,
   so P8's ids do not change (plane 0, drone 1, plate 2, P8 marker 3, twin 4). Immediately before
   each render it is re-posed with `resetBasePositionAndOrientation` to the plate's **physics** pose
   (`getBasePositionAndOrientation(plate)`).
2. **P8's pad marker is hidden**, not re-posed. After every reset its visual is set to alpha 0
   (`changeVisualShape(rgbaColor=[…, 0])`). It is found by geometry, not colour, through
   `ila.p8_adapter.scene_handles`. The twin's ring stands in for P8's painted pad.
3. **Ground:** after every reset, `changeVisualShape(plane, -1, textureUniqueId=-1,
   rgbaColor=ground_rgba)`, where `ground_rgba` is [0.08, 0.16, 0.24, 1]. The plane's collision box
   is P8's and is not touched.
4. **Drone:** no change (see evidence: its own camera cannot see it).
5. **Camera**, as committed in `configs/render/camera.yaml` (SHA-256
   `53f802656f6343f31a5ea98069024978d03cecf636807f4adcabb399e67a0380`):
   - body-fixed, looking along body −z with image up = body +x, mounted 0.02 m below the body
     origin;
   - vertical FOV 90°, aspect 1, near 0.01 m, **far 20.0 m**, 96 × 96 RGB;
   - `ER_TINY_RENDERER`, shadows off, light direction [0.3, −0.2, 1.0];
   - texture 512 × 333 px (512 along plate x; texels square to 0.1 %); centre crop 88 px at
     evaluation, random crop at training;
   - lengths are model scale (λ = 1/25).

   **`light_rot_60` corrected.** The appearance-shift value [−0.2, 0.3, 1.0] was 39° from the default
   light direction in 3-D (157° in azimuth), not the 60° of plan §1.3. It is now the default rotated
   +60° about world z: [0.3232, 0.1598, 1.0], with the elevation unchanged
   (`tests/test_render_texture.py`). The variant is evaluation only, so nothing trained changes.

   **Far plane 20 m, not 5 m: a deviation from plan §1.3, approved by the user 2026-10-27.** At
   5 m the far plane clips the ground at altitudes and attitudes the drone can reach, and where
   that boundary falls encodes height above the world-fixed ground (pixel evidence below). At 20 m
   no clipping was measured up to 3.0 m altitude and 30° pitch.
6. **Naming correction.** P8 has no `OBS_FIELDS` constant. The field table is
   `rld.envs.observation.obs_fields(cfg: ObservationConfig) -> tuple[ObsField, ...]`, and its
   `last_action` and `in_contact` blocks are optional. Both are on in P8's evaluation config, which
   gives 25 entries. `ila.p8_adapter.obs_fields()` returns that table for the eval config, and
   `ila.obs.split` takes offsets from it. Wherever the plan, `configs/obs/student.yaml` or the
   skills say "`OBS_FIELDS`", they mean this function. The YAML comment and the `split.py`
   docstring are corrected; CLAUDE.md and the plan are not edited.

*Reason.*
- **A texture on a `GEOM_BOX` is unusable in TinyRenderer.** A box twin textured with a 4-colour
  test image rendered its whole top face as one colour (2 unique colours in the frame). The mesh
  quad with explicit UVs showed all four quadrants (5 unique colours). P8's plate is a `GEOM_BOX`,
  so texturing it in place cannot show a helipad. The twin also leaves P8's plate visual untouched.
- **P8's marker is a world-fixed visual**, which non-negotiable 2 forbids. It is created once at the
  reset pose and never moved, so it parts from the plate as soon as the deck moves. Over one `id`
  episode per sea state (frigate, zero action, 360 control steps = 12 s model):

  | Sea state | Max separation from plate | Max attitude difference |
  |---|---:|---:|
  | SS3 | 50.1 mm | 1.7° |
  | SS4 | 70.5 mm | 3.0° |
  | SS5 | 89.9 mm | 4.6° |
  | SS6 | 89.7 mm | 16.2° |

  In SS6 the separation was already 80.1 mm and 11.2° within the first 90 steps. Hiding it is one
  call per reset. Re-posing it as well would double the twin's ring.
- **The ground is an inertial reference.** It is a checkerboard: 2 colours in a 128 px patch away
  from the deck.

*Evidence.* PyBullet 3.2.7 (API 202010061), DIRECT. Episode `id` SS6 index 0 (frigate, 45°,
episode seed 1490732229). Throwaway scripts were run from the session scratchpad and are not
committed.
- **`getVisualShapeData` after reset**, one row per body. Columns: body, what it is, geometry,
  dimensions (m), rgba, texture id, collision shape.

  | Body | What | Geometry | Dimensions (m) | rgba | Texture | Collision |
  |---|---|---|---|---|---|---|
  | 0 | plane | `GEOM_MESH` `plane100.obj` | scale 1 | (1, 1, 1, 1) | 1 (checkerboard) | box 200 × 200 × 10 m, mass 0 |
  | 1 | drone `cf2` | `GEOM_CYLINDER` | length 0.025, radius 0.06 | (1, 1, 1, 1) | none | P8's |
  | 2 | plate | `GEOM_BOX` | 0.80 × 0.52 × 0.02, visual frame z −0.01 | `PLATE_RGBA` (0.35, 0.38, 0.42, 1) | none | identical box, mass 1000 kg |
  | 3 | P8 marker | `GEOM_CYLINDER` | length 0.001, radius 0.15 | `PAD_MARKER_RGBA` (0.95, 0.75, 0.15, 1) | none | none, mass 0 |

  - Drone: there is no mesh (`URDF_IGNORE_VISUAL_SHAPES`). PyBullet still reports, and
    TinyRenderer draws, the collision cylinder as a white visual. This corrects the earlier note
    that the drone has no visual.
  - Plate: the body origin is on the top surface.
  - P8 marker: centred on the plate's top surface at reset.
- **The plate's physics pose vs P8's analytic deck sample** at the same substep count: within
  6 µm at steps 30, 60 and 90. That is irrelevant at 96 px; the twin follows the physics pose.
- **Texture removal.**
  - `textureUniqueId=-1` removes the plane's texture in the render: unique colours go from 2 to 1,
    and with `ground_rgba` the ground renders uniform [19, 38, 59].
  - `getVisualShapeData` keeps reporting texture id 1 afterwards. Its texture field is stale, so
    tests must check pixels, not that field.
- **Alpha 0 is honoured.** With alpha 0, TinyRenderer drops the marker from both the RGB image
  (7 279 px changed in a 192 px top-down view) and the segmentation mask (bodies seen go from
  {0, 2, 3} to {0, 2}).
- **The drone is invisible to its own camera.** At camera offsets −0.02 (committed), −0.013,
  −0.012, −0.005 and 0.0 m along body z, the drone covers 0 px of the segmentation mask:
  - the cylinder spans body z ±0.0125 m, so at −0.02 m it is behind the camera;
  - from inside the cylinder, back faces are culled.

  Seen from 0.4 m above, the drone covers 2 361 px, in white.
- **Far plane.** Listed start altitudes are 1.6 to 2.0 m model (all 2 800 episodes). With far
  5 m, TinyRenderer's white clear colour replaces the ground as follows (96 × 96 frame = 9 216 px,
  pitch about body y):

  | Altitude | Pitch | Clear-colour px |
  |---|---:|---:|
  | 1.0 or 1.75 m | 0 to 30° | 0 |
  | 2.5 m | 30° | 1 248 |
  | 3.0 m | 20° | 96 |
  | 3.0 m | 30° | 2 208 |

  Where that boundary falls depends on height above the world-fixed ground, so it is a weak
  inertial cue. With far 20 m: 0 px at 3.0 m up to 30° (192 px at 40°, the horizon).
- **Physics pre-check** (a throwaway script, not the Phase 1 test). A scripted 200-step action
  sequence was flown on SS6 two ways:
  - (a) no render;
  - (b) ground recolour, marker alpha 0, a mesh-quad twin re-posed every step, and a 96 px
    TinyRenderer frame before every step.

  The run ended by crash at step 51, with first contact at step 50. A SHA-256 over the per-step
  drone and plate pose and velocity, the observation, the reward and the info was identical:
  `7207c4cd…`.

*Consequence.*
- **Hand-off B builds the scene to this entry.** `scene.attach(env)` re-applies the recolour,
  marker alpha and twin after every reset, because `resetSimulation` destroys all of them.
  `pre_render()` re-poses the twin. Nothing calls `stepSimulation` or `changeDynamics`.
- **Render tests** check pixels and segmentation masks, never `getVisualShapeData`'s texture
  field. The "no world-fixed visual" test takes every body id in each frame's segmentation mask;
  each must be the ground, the plate or the twin, the twin's pose must equal the plate's, and the
  drone id must never appear.
- **Image and observation share a count.** `ila.p8_adapter.physics_step_count(env)` (P8's substep
  counter: 0 after reset, 8 per step) is the count an image and its observation share.
- **`body_rates` frame.** P8's layout table labels `body_rates` "body", but P8's builder passes
  PyBullet's world-axis angular velocity through unrotated (`rld/envs/observation.py`). The student
  receives it as P8 defines it. Plan §1.2's "gyro" is therefore loose wording.
  `configs/obs/student.yaml` now says world axes (P8 `observation.py`, `build_observation`
  docstring).
- **Pixel convention, measured.** TinyRenderer samples at integer viewport coordinates and then
  flips rows: `col = (ndc_x + 1)/2 · W` and `row = (1 − ndc_y)/2 · H − 1`. The optical axis is
  therefore at pixel (col 48, row 47) at 96 px, not the OpenGL pixel-centre (47.5, 47.5).
  `ila.render.camera.project_points` uses the measured convention. Pinhole residuals (rendered disc
  centroid vs projection from the frame's own matrices, four camera poses, two tilted and yawed)
  were 0.000, 0.051, 0.252 and 0.195 px. With the OpenGL convention they were 0.71–0.78 px.
  Anything that labels pixels (the `modular` estimator, crops) uses `project_points`.
- **Twin pose.** The twin's position equals the plate's physics position exactly at render time.
  Its quaternion matches to about 1 ulp, because PyBullet renormalises the quaternion it is given.
- **Texture ids.** They keep counting across `resetSimulation` (+2 per reset: the plane's texture
  and the helipad). Resident memory stayed flat (681 MB) over 206 resets, so nothing leaks. The
  helipad PNG is written once to `artifacts/render/helipad_<sha16>.png` and re-loaded after each
  reset.
- **Bootstrap seed.** P8's `rld.eval.stats.DEFAULT_BOOTSTRAP_SEED` is 20260926; this project uses
  20261023. Every caller of a P8 bootstrap function (through `ila.p8_adapter.stats`) must therefore
  pass the seed explicitly, never rely on P8's default.
- **Far-plane residual at 20 m** (the worst of yaw 0° and 45°, which puts an image corner in the
  pitch plane; reproduced independently of the leak audit):

  | Pitch | Clear-colour px, altitude 1–3 m |
  |---|---|
  | ≤ 38° | 0 |
  | 40° | 0 for z ≤ 2.0 m, 96 at 2.5 m, 192 at 3.0 m |
  | 45° | 384–1 056 |

  The residual therefore appears only at tumbling or crash attitudes. Phase 3 collection logs the
  per-frame fraction of clear-colour pixels.
- **Student-facing render output is pixels only** (privileged-leak audit, RISK 1).
  - `Scene.render()` returns `ila.render.camera.Frame`, whose only field is `rgb: uint8[H, W, 3]`
    (read-only).
  - `Scene.render_with_meta()` also returns a `RenderMeta`: view and projection matrices, the
    physics step count and the optional segmentation mask.
  - `RenderMeta` is never a student input. The view matrix gives the drone's world position,
    which with the plate pose is `rel_position`/`height`; the step count is the clock; the mask is
    a perfect deck mask.
  - The plate-pose accessor is private.
  - A test pins `Frame`'s single field.
  - The il-conventions skill and `docs/CONVENTIONS.md` state the rule: only `rgb` and the
    allowlisted proprio may enter a student store or batch, and the store's `t_model_s` never
    does.
  - `StudentInput` now carries `proprio_fields` (set by `ObsContract.student_input`) and refuses
    names other than the variant's contract. A hand-built array of the right width without the
    contract's names cannot pass.

### P1-D2 — Frozen-list rows used in Phase 1 tests (2026-10-27)
*Decision.* The render, invariance, scene, geometry and adapter tests, and the P1-D1 inspection
scripts, fly realizations from P8's frozen `id` and `static` lists. They use scripted or zero
actions only. The adapter's teacher test makes one `act` call on a reset observation to check the
interface, and nothing is scored. When `splits/` exists (Phase 3), these tests move to a
`train`-pool realization.

*Reason.*
- The frozen lists are the only episode source the adapter exposes before Phase 3 builds the
  `train`/`select` manifests.
- No policy is trained, selected or tuned on these rows, and no outcome on them is reported.
- P8 has already published its teacher results on these lists.
- So non-negotiable 4 ("frozen lists are for evaluation only") is not touched in substance. It is
  recorded here so that the use is visible.

*Evidence.* `tests/render_support.py` (`listed("id", "SS6")`, `listed("id", "SS5")`,
`listed("static")`); `tests/test_p8_adapter.py`; `tests/test_obs_split.py` (one reset of `id` #0).

*Consequence.* Phase 3 obligations:
1. Move these fixtures to a `train`-pool realization once `splits/` exists.
2. **(Leak audit NOTE 3.)** Demo collection, DAgger and DART must assert
   `render_cfg.appearance_shift is None` before writing a frame, with a test. The appearance-shift
   variants are evaluation only (plan §1.3), and `RenderConfig` carries the variant's name so the
   writer can refuse it.
3. **Poison test (leak re-audit NOTE 2).** In a demo store:
   - set the privileged columns (`rel_position`, `rel_velocity`, `deck_normal`, `rel_tilt`,
     `height`) and `time_fraction` of every stored `teacher_obs` to NaN, selected by name through
     `ila.obs.split`;
   - build every student dataset and batch from the poisoned store;
   - assert every student batch is finite.

   This catches what the syntactic slice scanner in `tests/test_obs_split.py` cannot.
4. **Student-readable columns.** A test asserts that the store's student-readable columns are
   exactly `images` and `proprio`. `t_model_s`, `priv`, `teacher_obs` and any render metadata are
   never read into a student batch. Item 6 says where the test is written.
5. **Guard scope (leak re-audit, for the record).**
   - The obs-slice scanner has no escape hatch. If a literal time index is ever needed, it returns
     only as a committed per-node allowlist.
   - `tests/test_render_access.py` forbids `RenderMeta`, `render_with_meta`, `render_pose`,
     `scene_handles`, `physics_step_count`, `_plate_pose` and PyBullet in `ila/train`,
     `ila/policies`, `ila/deploy`, `ila/data` and `ila/eval`. "PyBullet" means any `pybullet*`
     module except `pybullet_data`, including through `sys.modules[...]`.
   - `ila/data` may use `attach` and `Scene.render()`, which returns pixels only.
   - Only `eval/runner.py` and `eval/batched.py` are exempt. The runner calls `render_with_meta`
     only inside its timing assertion, and never returns, yields or passes on a `RenderMeta`.
6. **The exact student-columns test is written against the dataset/batch class, not the HDF5
   file.** Any extra column, under any name, must fail it. `ila/data` can read live env attributes
   (`env.pos`, `env._substep`) without importing anything, so a privileged value could enter a
   batch under an innocent name. The poison test (item 3) poisons only `teacher_obs`, so it would
   not catch that.

Phase 2 obligations (runner, `ila/eval/runner.py` and `ila/eval/batched.py`):
1. A test that pins the runner's public rollout and record types: no `RenderMeta` field and no
   physics step count in anything it returns or yields.
2. A test that students receive only what `ObsContract.student_input` builds: a `StudentInput`
   carrying the contract's `proprio_fields`, with `Frame.rgb` pixels.

## Phase 2

*Before you start (from Phase 1).*
- **Runner contract.** Call `scene.attach(env)` after every `env.reset()`, including P8's
  `set_motion` + reset reuse pattern. Then call `Scene.render()` after the reset and after each
  `env.step()`. Use `render_with_meta()` only inside the timing assertion, and never return,
  yield or store a `RenderMeta` (P1-D2 item 5).
- **Adapter gap.** The adapter does not wrap P8's `set_motion` / `pad_offset_for` yet. Reusing one
  env per chunk the way P8's `run_chunk` does needs them added there.
- **Pixel convention.** TinyRenderer puts the optical axis at pixel (48, 47) at 96 px. Any pixel
  label must use `ila.render.camera.project_points` (P1-D1).
- **Render cost** (single process, i9-10980XE): about 3.5 ms per 96 px frame for the camera, plus
  about 0.9 ms for the full scene. Physics alone runs 300 steps in 0.71 s. Measure properly in the
  throughput run.
- **Texture shimmer.** Nearest-neighbour sampling of the 512 px texture shimmers at start
  altitude. It is deterministic, but weigh it in the 64 px decision.
- **Phase 2 obligations from P1-D2:**
  - a test pinning the runner's public rollout and record types (no `RenderMeta`, no step count);
  - a test that students receive only what `ObsContract.student_input` builds.
- **CI coverage.** The render-invariance tests are marked `slow`, so CI deselects them. They run
  locally, and every gate re-runs the full suite.

### P2-D1 — Reproduction gate and teacher label path (2026-10-27)
*Decision.*
1. **The reproduction gate passes.** With rendering on (`full`: camera, twin, ground recolour,
   P8 marker hidden), this repo's runner reproduces P8's committed episode rows as text, every
   column:
   - `pid_feedforward`: 2 800 of 2 800 listed episodes vs P8 `results/e01/episodes.csv`;
   - `ppo` seed 0: 400 of 400 on `id` SS5+SS6 vs P8 `results/e07/matrix/episodes.csv.gz`.

   The comparison is P8's own `rld.eval.reproduce.compare_to_reference` (pad `aft`). The gate also
   requires `n_compared` equal to 2 800 / 400, `n_differ = 0` and `n_missing = 0`. Without those,
   a short run would pass on `all_identical` alone.
2. **Teacher labels may use either path; collection uses SB3.** P8's ONNX graph on ORT CPU
   (1 thread) matches P8's SB3 reference within the plan's 1e-6 max-abs, so the ONNX path is
   permitted for labels (plan §1.4). Phase 3 nevertheless flies and labels `T_ppo` with the SB3
   path, the one this gate certified, unless P3-D1 records otherwise. The reason is that a path
   that passes parity is not trajectory-identical:
   - only 7.8 % of action values are bit-equal between SB3 `act` and ORT;
   - P8 measured that one-ulp input changes flip outcomes (P2-D3).
3. **Runner contract**, as built (`src/ila/eval/runner.py`, `src/ila/eval/batched.py`):
   - P8's contract: the list is the only episode source, the start is checked on every reset,
     and rows do not depend on workers or chunk (chunk 25).
   - P8's columns come verbatim and in order, then the P9 columns of plan §1.7.
   - The scene is re-attached after every reset. A frame is rendered after the reset and after
     every `env.step()`, including the terminal one.
   - Image/obs step-count equality is asserted through `render_with_meta()` on sampled steps.
   - Students receive only `ObsContract.student_input` + `Frame.rgb`; the obs window comes from
     `chunking.obs_window_steps`.
   - Not mirrored from P8: the ship-motion feed (feed specs are refused), and the Phase 7 arm
     fields (`pad_override`, non-JONSWAP motion, the `lambda` start check, `cfgs_override`). No arm
     of this project uses them.
   - P8 private helpers re-implemented (`_check_start`, `_check_privilege`, `_none_to_nan`,
     `_episode_row`) are each tested equal to P8's on P8's own `tiny_list` fixture
     (`tests/test_eval_runner.py`).

*Reason.* Non-negotiable 3. If rendering moved a single contact, no row here would be
comparable with P8's. Prediction from earlier phases: the P1-D1 pre-check (one 200-step
episode) and the Phase 1 render-invariance tests had found no physics change. This gate is the
test at the scale of the frozen lists.

*Evidence.*
- `results/reproduce_p8/comparison.json`:
  - `passed: true`;
  - `n_compared = n_identical` = 2 800 / 400, `n_differ = 0`, `n_missing_in_reference = 0`;
  - `header_identical: true`.
- `results/reproduce_p8/run_info.json`:
  - render `full`, `render_cfg_sha256` `53f80265…0380` (= P1-D1);
  - workers 24, chunk 25, spawn;
  - `OMP_NUM_THREADS = MKL_NUM_THREADS = 1`, torch threads 1;
  - P8 `7237b706…`, `ppo` `model.zip` `1a012394…`;
  - 3 200 episodes in 254.8 s, `ila_commit` `c88c106`.
- P8-column row SHA-256: `pid_feedforward` `d5ddc55d…ec31`, `ppo` `a836275f…bf50` (run_info
  `rows`).
- The first full run (256.4 s) was made on an uncommitted tree (`ila_dirty_paths` in its
  run_info). It was re-run on the committed Phase 2 tree `c88c106`, which had no dirty paths,
  in 254.8 s. The re-run gave the same result and the identical P8-column row SHA-256s. The
  committed `results/reproduce_p8/` files are the re-run's:
  - `comparison.json` `75b34e8b…31fa`;
  - `run_info.json` `4ea43413…e002`;
  - repo-relative paths.
- `results/label_parity.json`:
  - 10 000 observations from `ppo` seed 0 flying 163 episodes of P8's **train** pool (SS3–SS5,
    stream `label_parity`, seed 20261023);
  - observations SHA-256 `70e9f35a…6e24`, ONNX `06230fd2…`.

  | Comparison (over all action values) | Max-abs | Bit-equal fraction |
  |---|---|---|
  | SB3 `reference_actions` vs ORT CPU, single (primary) | **3.58e-7** (≤ 1e-6: pass) | 0.112 |
  | SB3 batched vs SB3 `act` | 4.77e-7 | 0.119 |
  | SB3 `act` vs ORT single | 4.77e-7 | 0.078 |
  | ORT single vs ORT batched | 0 | 1.000 |

  P8's own `check_ort` also passes.

*Consequence.*
- `make reproduce-p8` stays green from here on. Any later change under `src/ila/render/`,
  `src/ila/eval/runner.py` or the adapter re-runs it before merging.
- Phase 3 collection flies `T_ppo` on SB3.
- **Frozen-list use in Phase 2**, recorded as P1-D2 did: the reproduction gate flew all five
  frozen lists with the teachers `pid_feedforward` and `ppo` only. Nothing was trained, selected
  or tuned on them, and P8 had already published those rows.

### P2-D2 — Throughput and the budget it implies (2026-10-27)
*Decision.* **96 px stays for every arm.** The measured budget does not justify the 64 px
fallback of plan Phase 2 item 4. GPU inference runs with TF32 off (P2-D3).

*Reason.*
- The projected wall clock of all non-training work is 12.2 h at the central estimate; the
  upper bound is in the Evidence.
- 96 px costs about twice 64 px per frame. That shows only in the teacher and `bc` rates, which
  are already fast.
- The rate that dominates the budget is `dp` inference, which is limited by kernel launches and
  does not depend on frame size.

*Evidence.* All on this machine (i9-10980XE, RTX A4000, WSL2). Sources:
`results/throughput.csv` (SHA-256 `11898e1e…1879`) and `results/throughput_projection.csv`
(`59fb8fa3…9191`); run info in `results/throughput_run_info.json`.
- **Render per frame** (single process, 300 frames, `full` mode, p50):

  | Size | p50 |
  |---|---|
  | 64 px | 1.98 ms |
  | 96 px | 4.09 ms |
  | 128 px | 7.24 ms |

  `camera` mode is within 0.3 ms of `full` at every size.
- **Teacher with render, 96 px** (P8 train-pool episodes; spawn excluded):

  | Teacher | 1 worker | 8 workers | 16 workers | 24 workers |
  |---|---|---|---|---|
  | `ppo` | 93 steps/s | 519 | 918 | **1 205** |
  | `pid_feedforward` | 133 | 730 | 1 324 | **1 765** |
- **GPU inference, dummy networks** (real architectures, random init, P2-D4; one call, p50):

  | Network | B = 1 | B = 8 | B = 24 |
  |---|---|---|---|
  | `bc` | 3.85 ms | 5.22 ms | 10.78 ms |
  | `dp`, K = 10 | 80.6 ms | 80.9 ms | 91.4 ms |
  | `dp`, K = 100 | 772 ms | 751 ms | 752 ms |

  For comparison, `dp` at K = 10 on CPU (1 thread) takes 170 ms.
- **End to end, lockstep workers + one CUDA server, 96 px, B = 24** (env-steps/s):

  | Policy | env-steps/s |
  |---|---|
  | `bc` | 973 |
  | `dp` K = 10, Ta = 8 | 172 |
  | `dp` K = 10, Ta = 1 | 136 |
  | `dp` K = 100, Ta = 8 | 28.2 |
  | `dp` K = 100, Ta = 1 | 19.0 |
- **Projection** (`results/throughput_projection.csv`; training time excluded, HDF5 writes not
  timed).
  - Episode lengths come from P8's committed `ppo`, `pid_feedforward` and `gated` rows:
    `ppo` 61.5 steps on SS3–SS5, `pid_feedforward` 151, `gated` 138.
  - The upper bound assumes every student episode times out at 361 steps.

  | Work | Central (h) | Upper (h) |
  |---|---|---|
  | Demo collection (ppo 2 000, pid 1 000, mix 1 000, DART 3 × 1 000) | 0.11 | — |
  | DAgger rollouts, 5 seeds × 5 iterations × 200 | 0.09 | 0.52 |
  | Matrix, core arms, 5 seeds × 2 800 | 2.83 | 16.5 |
  | Ablations, `Ta`/K sweeps, appearance shift, latency arm | 9.16 | 53.5 |
  | **Total** | **12.2** | **70.6** |

  The upper total is the sum of the per-row upper bounds, using the central value where a row
  has no upper bound.
- **Frozen-list reads.** The episode lengths come from P8's committed `id` rows, SS6 included.
  These are P8's published teacher rows, read for the budget only; nothing is selected or tuned on
  them.

*Consequence.*
- **Inference latency.** `dp` at K = 10 already takes 80.6 ms per call at B = 1 on the GPU,
  against H5(a)'s 33.3 ms. Batching is nearly free, which points to per-call launch overhead.
  This is a measurement of the dummy network in eager PyTorch, not of the exported graph. Phase 9
  measures the exported graph, and CUDA graphs or `torch.compile` are Phase 9 options, which change
  numerics and so need their own parity.
- **K = 100 cost.** The K = 25/100 sweep rows are costed at the K = 100 rate.
- **Re-estimate later.** `/train-job` uses these rates for every estimate. Phase 4 replaces the
  student length assumption with measured lengths.

### P2-D3 — Determinism contract (2026-10-27)
*Decision.* The contract has four parts.
1. **CPU fp32 single-sample path:** rows are bit-identical regardless of worker count, chunking or
   batch slot. Measured: 150 of 150 `select` rows identical at 1 vs 8 workers. This is the path
   for every result that claims bit-reproducibility.
2. **Batched GPU path:**
   - Run with `torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 =
     False`, `cudnn.benchmark = False` and `float32_matmul_precision = "highest"`, recorded in
     every run_info (`ila.eval.batched.configure_torch_numerics`).
   - It is reproducible run to run at a fixed batch cap (B = 1 twice: 150/150 identical).
   - It is **not** bit-identical across batch caps or with the CPU path, and is never claimed to
     be.
   - It is judged by its outcome-flip rate against P8's one-ulp floor: 53 of 800 = 6.6 % of `id`
     SS6 policy-episodes (P8 `docs/protocol.md` L5314, L5455).
3. **The flip rate measured here is not informative, and is re-measured before use.**
   - The dummy `dp` has random weights and crashed in all 150 episodes on every path, so outcome
     flips had no room to show (0 at every batch cap).
   - **Binding:** before any GPU-path row enters a Phase 8 table, this study is re-run with a
     trained `dp` (Phase 6, selected checkpoint) on the same 150 `select` episodes. Its flip rate
     must be no larger than P8's floor, or the GPU path is not used for that arm.
4. **Rule for the noise key** (privileged-leak audit, Phase 2, RISK 1):

   > The control step t and the episode seed reach a student only as a noise key and a schedule
   > index. They may (a) seed the per-request `torch.Generator` via
   > `noise_seed(run_seed, episode_seed, t)`, (b) feed `chunking` (`is_replan_step`, ensembling
   > indices), and (c) drive slot bookkeeping. They never enter an `nn.Module` forward, a feature,
   > a conditioning vector, a normaliser or a training batch. Besides pixels and proprio, the only
   > integer a network sees is the diffusion index k ∈ [0, num_train_timesteps). DDPM training
   > noise comes from the data-loading generator (run_seed, epoch, worker_id), never from
   > `noise_seed(·, ·, t)`.

   The rule is mirrored in the il-conventions skill and `docs/CONVENTIONS.md`, and guarded by
   tests (signature allowlist, invariance under a fixed noise, and a field allowlist).

*Deviation.* Plan Phase 2 item 5 says "200 `select` episodes". `select` is fixed at 150 by plan
§1.5 (50 each of SS3–SS5), so the study flies all 150. The user approved this on 2026-10-27.

*Reason.* Plan §0(5). Batched GPU numerics depend on the batch. The defensible claim is a flip
rate no larger than the one P8 measured for a one-ulp input change. A study whose policy never
survives long enough to flip cannot support that claim, so it is not presented as if it did.

*Evidence.* `results/determinism/summary.json` (SHA-256 `b00dff4c…b96c`), `diffs.csv`
(`9c5e571b…29dd`) and `run_info.json` (`f28f0f34…7b97`). Run settings: dummy `dp`
(`build_dummy("dp", 0)`), K = 10, Ta = 8, render `full`, `splits/select.parquet`
(`07ecf049…422b`).

| Run vs CPU workers = 1 | Rows whose text differs | Outcome flips | Mean steps |
|---|---|---|---|
| CPU, 8 workers | 0/150 | 0 | 45.45 (= reference) |
| GPU B = 1 | 150/150 | 0 | 46.39 |
| GPU B = 8 | 150/150 | 0 | 47.53 |
| GPU B = 24 | 150/150 | 0 | 46.61 |
| GPU B = 1, repeat | 150/150 | 0 | 46.39 |
| GPU B = 24, TF32 on (torch default) | 150/150 | 0 | 46.21 |

GPU vs GPU B = 1:

| Run | Rows whose text differs |
|---|---|
| B = 1 repeat | 0/150 |
| B = 8 | 122/150 |
| B = 24 | 140/150 |
| B = 24, TF32 on | 150/150 |

Every run's 150 outcomes are `crash`.

Two caveats on the comparator:
- `select` is SS3–SS5 while P8's floor is measured at `id` SS6, so the floor is a conservative
  comparator.
- The `select` outcomes here choose nothing: no checkpoint, σ or setting is selected from them.

*Consequence.*
- The CPU single-sample path flies every bit-reproducible result.
- The GPU path may be used for throughput-bound arms only after item 3's re-measurement. Its use
  is labelled in the run_info.
- `policy-trainer` found CPU-vs-GPU gaps on one random-init `dp` chunk of up to 2.8e-3 with TF32
  on, and 5e-6 with TF32 off (not committed; from its CPU/GPU test session). That is why TF32 is
  off.

### P2-D4 — Work pulled forward from Phases 3, 4 and 6 (2026-10-27)
*Decision.* Two pieces of later-phase work were done in Phase 2 because Phase 2 items 4 and 5
need them. The user approved both on 2026-10-27.
1. **Splits (Phase 3 step 1):** `src/ila/data/episodes.py`, `scripts/make_splits.py`,
   `splits/select.parquet` and `splits/MANIFEST.csv`.
   - `select` is 150 episodes of P8's `tune` pool, 50 each of SS3, SS4 and SS5, covering all
     135 tune realizations. It is dealt by P8's own round-robin rule with generator seed
     20261023.
   - `train` draws come from a generator with named streams. Nothing demo-sized is written yet.
     Whether the demo sets, DART and DAgger share one stream is a P3-D1 decision.
2. **Student architectures with random init (parts of Phases 4 and 6):**
   - `ila/policies/encoder.py` (ResNet-18 with GroupNorm 16, spatial softmax with 32 keypoints);
   - `bc.py`;
   - `dp/unet1d.py` (FiLM, adapted from diffusion_policy, MIT; `THIRD_PARTY_NOTICES.md`);
   - `dp/chunking.py`, `dp/policy.py` (DDIM, per-episode noise generator);
   - `registry.py`.

   There is no training code. Parameters: `bc` 11 330 339, `dp` 29 353 123.

*Reason.* The throughput and determinism studies must measure the networks the project will
train, and the determinism study must fly `select`.

*Evidence.*
- `splits/select.parquet` SHA-256 `07ecf049cbf91e646036582b4967607e26cf68773bd486c8d7375d043f29422b`
  (content `e8b93d6f…5827`).
- `splits/MANIFEST.csv` `cb70cd51…7ed5`; regenerated by `scripts/make_splits.py` byte for byte.
- Realization-key overlap with all five frozen lists: 0 (`tests/test_data_episodes.py`; the
  Phase 2 leak audit checked this independently).

*Consequence.*
- **Gates 3 and 4/6 audit this code, not re-derive it.** Gate 3 re-checks `splits/` and records
  the `select` SHA in P3-D1 §2. Gates 4 and 6 audit the networks as trained.
- **Teacher's own view of `tune`.** P8's teacher has seen `tune` realizations in P8's own tuning
  and validation. `select` episodes are new seeds and select nothing for the teacher, but the
  teacher's `select` success may read slightly high. To be noted in P3-D1.
- **Open for P3-D1: DDIM timestep spacing.** `dp.yaml` does not set it, and the code uses
  diffusers' default, `leading`. Under it, K = 1 runs only timestep 0 and outputs near-pure noise.
  `trailing` gives [99] for K = 1. Since K = 1 and 2 are in the frozen sweep, the choice is made in
  P3-D1, before any training.
- **Module ownership.** The Phase 2 job modules `ila/eval/{reproduce,label_parity,throughput,
  determinism,jobs}.py` sit in `ila/eval/`, which CLAUDE.md assigns to `eval-auditor`. They were
  written by `sim-render-engineer`, and `eval-auditor` reviews any change to them from Phase 3 on.

## Phase 3

*Before you start (from Phase 2).*
- **Already built** (P2-D4): `splits/select.parquet` (150 episodes, `07ecf049…422b`), the
  `train` generator with named streams, and the `bc`/`dp` architectures.
- **P3-D1 must decide:**
  - whether the demo sets, DART and DAgger share one `train` stream;
  - DDIM `timestep_spacing` (`leading` gives near-pure noise at K = 1; `trailing` is
    recommended).

  P3-D1 must also note that P8's teacher has seen the `tune` realizations in P8's own tuning.
- **Teacher path:** collection flies and labels `T_ppo` on SB3, the gate-certified path. ONNX
  passes parity but is not trajectory-identical (P2-D1).
- **Budget:** collection projects to about 0.11 h at 24 workers, 96 px (P2-D2). HDF5 writes are
  not timed, so measure them in the first collection.
- **Obligations carried from P1-D2:** move the Phase 1 test fixtures to a `train` realization
  (`draw_train`); the appearance-shift refusal; the poison test; the exact student-columns test.
- **GPU path:** P2-D3 item 3 binds before Phase 8. Phase 3 needs only the CPU teacher path.
- **Re-run the reproduction gate** (`make reproduce-p8`, about 4 min) after any change to
  `ila/render/`, `ila/eval/runner.py` or the adapter.

### P3-D1 — FROZEN IMITATION-LEARNING PROTOCOL (2026-10-27)

**Status: FROZEN 2026-10-27, revision 3, approved by the user on 2026-10-27.**
- Nothing below changes without a dated deviation entry. That applies from the freeze, before any
  student is trained.
- The SHA-256 of this block (from this heading to the line before the next `###`, UTF-8) is
  recorded in the Gate 3 entry.
- **Revisions 2 and 3** fold in the pre-gate reviews and their re-reviews:
  - `privileged-leak-auditor`: FINDINGS, no LEAK, 3 RISKs, 6 NOTEs;
  - `results-skeptic`: 5 BLOCKING, 10 SHOULD-FIX, notes.

  §8 lists each disposition.

Everything here is **simulation only**: PyBullet physics, PyBullet TinyRenderer frames. Lengths and
times are model scale (λ = 1/25); full scale is ×25 for lengths and ×5 for times. One control step
is 33.3 ms model (167 ms full).

**1. Inherited from P8 (by reference, unchanged), and the statistics.**
- **Success criteria:** `configs/env/success.yaml`, SHA-256
  `c7fbdcc4df48bd4a2cfe21811613e5e50e3e4868822fe6dddd13745d96692118`.
- **Frozen lists:** MANIFEST SHA-256
  `e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2`. 2 800 episodes: `id`
  SS3–SS6, `unseen_seastate` SS6, `unseen_heading` SS3–SS6, `unseen_vessel` SS3–SS6 and `static`,
  200 per cell, aft pad.
- **Outcomes and metrics:** outcome classes in the order `crash → off_pad → hard_landing → bounce →
  success → timeout`; P8's metrics. Both as in P8 P3-D1 §1 and §3.
- **P8 commit and teacher:** `7237b7065c8af8172b3848ac6e9d3ceb2cf82075`. Teacher `ppo/0/final/`:
  `model.zip` `1a012394…`, `vecnormalize.pkl` `9ddb2a99…` (P0-D1). Every teacher-relative
  statement is conditional on teacher seed 0.
- **Rendering:** `configs/render/camera.yaml` `53f80265…0380`, 96 px (P1-D1, P2-D2).
- **Observation contract:** `configs/obs/student.yaml`
  `2bdfb259319b8fe3e0fc34e41cd6a80a75c7e94893502a7e02e63cf531bd3fc4`.
- **Evaluation matrix:** `configs/eval/matrix.yaml`
  `4d8a78fdfc59ddfb5cc51add8bba121af8da43ee0e35787ffd0e2611d7e1e96a`.
- **Statistics.** Bootstrap seed **20261023**, always passed explicitly (P8's default 20260926 is
  never relied on).
  - Per cell and seed: Wilson 95 %.
  - Across seeds: IQM with stratified-bootstrap 95 % CI, 2 000 replicates.
  - Closing speed: relative p95, over touched-down episodes only, never imputed.
  - **Success contrasts** (H1 and its secondaries): P8 `paired_bootstrap`.
    - Mean over seeds × episodes. Seeds are resampled per arm; one set of episode indices is
      shared.
    - 10 000 replicates. The point estimate is `PairedContrast.diff`.
    - `paired_iqm_difference` is printed beside it as "post hoc, not scored" (P8 P7-D1a item 3).
  - **Same-checkpoint contrasts** (H3: `Ta = 1` vs `Ta = 8`; H5(b): latency vs synchronous): the
    same statistic, but each replicate resamples seed indices **jointly** for both sides, because
    both sides are the same trained runs.
  - **H2:** W₁ is the exact 1-D empirical W₁ between an arm's pooled touched-down
    `time_to_touchdown_s` (all its seeds × episodes in the cell) and the reference's touched-down
    values.
    - Each replicate draws `bc`-mix and `dp`-mix seeds independently, then one set of 200 episode
      indices shared by both arms and the reference.
    - A replicate with an empty pool is NaN and excluded, and the count is reported.
    - The point estimate is computed on the unresampled pools.
  - **H4:** each replicate draws `bc_state`'s seeds once (used in both P and I) and `bc`'s seeds
    independently. The teacher's single row set is held fixed. One set of episode indices is
    shared, and S = mean success.
  - **Three statistics are not P8's.** The joint-seed contrast, H2's W₁ bootstrap and H4's
    three-arm bootstrap are written in `ila.eval`. Each gets a unit test on a synthetic case of
    known value before Phase 8. These are the only exceptions to "P8 functions only".
  - "Separates" means the 95 % CI excludes 0. Magnitude thresholds are judged on the point
    estimate. There is no multiplicity correction, and the README says so.
  - **SS6 flip floor.** Every SS6 contrast prints P8's one-ulp flip floor beside it, described as
    "6.6 % of `id` SS6 policy-episodes flip under at least one of 20 one-ulp draws (single-draw
    median 1.9 %), measured on P8's MLPs, not on these students".
- **Determinism:** P2-D3, with item 3's acceptance rule tightened in §3 (inference path).

**2. Pools, episodes and demonstrations.**
- **Pools** (P8 P3-D2):
  - `train` = 729 frigate realizations (SS3–SS5; headings 45°, 135°, 180°; 0/6/12 kn; seed
    ordinals 0–26);
  - `tune` = 135 (ordinals 27–31).
  - Disjoint by realization key from every frozen list and from each other. No SS6 and no 90° in
    any training, tuning or selection.
  - Tests in `tests/test_data_episodes.py`: `test_train_is_train_only`,
    `test_train_disjoint_from_frozen`, `test_train_no_ss6_no_beam`, `test_demo_files_train_only`,
    `test_select_is_tune_only`, `test_select_disjoint_from_frozen`.
- **`select`:** `splits/select.parquet`, SHA-256 `07ecf049…422b`. 150 `tune` episodes, 50 each of
  SS3, SS4 and SS5. Checkpoint selection, the DART σ search, the inference-path acceptance test
  and Phase 9 closed-loop parity use only this list.
  - *Caveat.* P8's teacher saw the `tune` realizations in P8's own tuning and validation. The
    episode seeds here are new and nothing is selected for the teacher, but the teacher's
    `select` success may read slightly high.
- **Streams.** Every demonstration set draws from one `train` stream, `demos` (generator seed
  20261023):
  - `ppo_2000` is indices 0–1999;
  - `pid_1000`, `mix_1000` and the three DART sets are indices 0–999 of the **same** stream.
  - So every 1 000-episode arm flies the same 1 000 starting conditions, and the teacher and noise
    ablations are paired.
  - **DAgger** (user decision, 2026-10-27): iteration i ∈ 1…5 flies `demos` indices
    1000 + 200(i − 1) … 1000 + 200i − 1, the same for every DAgger seed. These are the starting
    conditions of `bc_matched`'s second 1 000 (`tests/test_data_episodes.py`).
- **`T_mix` assignment** (user decision, 2026-10-27): balanced per sea state.
  - Within each sea-state cell, a permutation seeded by (20261023, `"T_mix"`, the cell's ordered
    episode keys) gives exactly half the episodes to `T_ppo` and half to `gated`. In an odd cell
    the extra episode alternates, `ppo` first.
  - On `demos_0-999`: SS3 167/167, SS4 167/166, SS5 166/167, so 500/500 in total.
  - The function is `ila.data.mix.assign_rows`.
- **DART noise:**
  - The executed action is `float32(clip(label + σ z))`, z ~ N(0, I). It is rounded to float32
    **before** `env.step` (§8), so `action_exec` is exactly what ran. The label is the clean
    teacher action.
  - z comes from a per-episode generator seeded by (20261023, episode position, episode seed),
    **not** by σ. The three σ sets therefore share their standard-normal draws (common random
    numbers).
- **β attribute:** teacher demonstrations store β = 1.0.
- **Demo sets.**
  - Summary: `results/demos_summary.json`, SHA-256
    `c56fa83350fe1900704b1ef885a1b558c6114a0cae4ee03a1f42483bf58713b7`. It is committed with this
    entry.
  - Split files:
    - `splits/demos_0-999.parquet` `0012f80b3d67e5c0c239cce58186ad6655de72aeb17b4b0de21600f20b6a4307`;
    - `splits/demos_0-1999.parquet` `20940abcce53298c3ca26e4fb92103226116aac1a0f55b730b74381c0b5805ef`;
    - `splits/MANIFEST.csv` `2be435abb17148f9644a92fa303d0b2cb333b08a89288b9de26f7b58fd8902dc`;
    - `configs/data/demos.yaml` `1fd413e2823a1cefd4b38e6ddd00c7097ee182e45f2564e0f398ce19f5f63f11`.
  - Stores live under `${data_root}/demos/` (gitignored). Success rates carry Wilson 95 % intervals.

  | Set | Teacher | Episodes / frames | Store SHA-256 | ila commit | Success: all [Wilson]; SS3 / SS4 / SS5 % | Losses | Training filter |
  |---|---|---|---|---|---|---|---|
  | `ppo_2000` | `T_ppo` (SB3, P2-D1) | 2 000 / 122 925 | `35b70cbd…7cfd` | `fc5388a` | 99.95 [99.7, 100]; 100 / 100 / 99.8 | 1 bounce | success (kept 0.9995) |
  | `pid_1000` | `T_pid` | 1 000 / 152 061 | `429d3791…bb33` | `fc5388a` | 99.8 [99.3, 99.9]; 100 / 99.7 / 99.7 | 1 bounce, 1 hard landing | success (0.998) |
  | `mix_1000` | `T_mix`: 500 `ppo` + 500 `gated` | 1 000 / 102 176 | `8afdbee7…8717` | `c941246` | 97.8 [96.7, 98.5]; 100 / 98.8 / 94.6. `ppo` 500/500; `gated` 478/500 (100 / 97.6 / 89.2) | 20 timeouts, 2 bounces, all `gated` | success (0.978) |
  | `dart_0.05` | `T_ppo` + σ 0.05 | 1 000 / 61 746 | `5e37da24…f500` | `d198f08` | 99.5 [98.8, 99.8]; 99.7 / 99.4 / 99.4 | 4 bounces, 1 hard landing | none |
  | `dart_0.10` | `T_ppo` + σ 0.10 | 1 000 / 60 907 | `7c0a1210…6902` | `d198f08` | 75.1 [72.3, 77.7]; 72.8 / 75.4 / 77.2 | 242 bounces, 5 hard landings, 2 off pad | none |
  | `dart_0.20` | `T_ppo` + σ 0.20 | 1 000 / 58 536 | `0be2520c…8039` | `d198f08` | 15.4 [13.3, 17.8]; 13.2 / 13.8 / 19.2 | 437 bounces, 226 off pad, 140 hard landings, 43 crashes | none |

  - Mean episode length: `ppo` 61.5 steps (2.05 s model), `pid` 152.1, `mix` 102.2.
  - **Clear-colour frames** (P1-D1 far-plane residual), as max fraction / frames above 0:
    - `ppo_2000` and `mix_1000`: one pixel in 2 frames each;
    - `pid_1000` and `dart_0.05`: none;
    - `dart_0.10`: 0.148 / 24;
    - `dart_0.20`: 0.240 / 792, about 1.4 % of its frames.

    They occur almost only at the tumbling attitudes of failing DART episodes. Where that boundary
    falls depends on world altitude, so if σ* = 0.20 is chosen, this is stated with the result.
- **Training filter.** The configs carry it as `success_filter`. BC-family, `dp` and `modular`
  training read only the `success` episodes of the clean teacher sets, and report the kept
  fraction.
  - **DART sets are not filtered.** Their failures come from the injected noise, not from the
    teacher, and every label is the clean teacher action. That is the same reasoning as for DAgger
    rollouts.
    - Filtering would tie the kept data to σ: 995 / 751 / 154 kept.
    - The σ search would then be partly a search over data size.
  - **DAgger rollouts are never filtered** (`bc_dagger.yaml`: `success_filter: {demos: true,
    rollouts: false}`).
  - "First N" means the first N episodes by stream index, **before** filtering. A set's
    labelled-episode count is N attempted episodes.
- **Data audits** (`results/demos_summary.json`; seed 20261023; `ila.data.audit`), all six stores:
  - **Attribute violations: 0.** The check covers, from stored attributes alone: outside `train`,
    in `tune`, in any frozen list, SS6, 90°, a repeated (key, episode seed), and an episode not
    equal to its split row.
  - **Open-loop replay** of the stored `action_exec` (binding): 20 of 20 sampled frames and
    observations byte-identical, and no episode diverging.
  - **Closed-loop re-fly:** 20 of 20 frames identical.
  - **Re-labelling:** 1 000 of 1 000 sampled steps re-labelled identically (max-abs 0.0); stateful
    teachers re-flown in order.
  - **Bookkeeping:** runner rows and manifest SHA-256s match the stores.
  - The summary regenerates byte-identically.
  - `privileged-leak-auditor` re-checked 40 episodes per store independently:
    - `proprio` equals the allowlisted fields of `teacher_obs` by name;
    - no proprio column equals a privileged or clock column;
    - `last_action[t] = action_exec[t−1]`;
    - the `T_mix` teacher matches `assign_rows` on 1000 of 1000 episodes.

**3. Arms.** Configs are frozen by SHA-256. Changing a config file after the freeze is a deviation.

| Arm | Config (SHA-256) | Input | Data | Seeds | Evaluation lists |
|---|---|---|---|---:|---|
| `bc_state` | `bc_state.yaml` `36b65a7d…eb0f` | P8 25-d obs, `time_fraction` included (privileged by design; never a vision row) | `ppo_2000` first 1 000 | 5 | all 2 800 |
| `bc` | `bc.yaml` `3d3386bb…1299` | 2 frames + 13-d proprio | `ppo_2000` first 1 000 | 5 | all |
| `bc_matched` | `bc_matched.yaml` `f956e182…8de5` | same | `ppo_2000` all 2 000, on `bc_dagger`'s schedule | 5 | all |
| `bc_dart` | `bc_dart.yaml` `5dff64bf…0c02` | same | `dart_σ*`, all 1 000 | 5 | all |
| `bc_dagger` | `bc_dagger.yaml` `dbc7c0d2…9da7` | same | 1 000 demos + 5 × 200 on-policy = 2 000 | 5 | all |
| `dp` | `dp.yaml` `c8f90c35…6913` | same | `ppo_2000` first 1 000 | 5 | all, at `Ta = 8`, K = 10 |
| `modular` | `modular.yaml` `84153e6d…eadc` | same; `time_fraction` to the frozen teacher only | `ppo_2000` first 1 000 | 5 | all |
| `R1_lerobot_dp` (reference, not gating) | Phase 6 config, hashed at its creation | same | same 1 000 | 3 | `id` SS3–SS6, tag `reference_external`. Same selection rule |

- **Shared encoder** (every vision arm):
  - ResNet-18 with `weights=None`, GroupNorm with 16 groups, 32-keypoint spatial softmax, 64-d per
    frame;
  - `To = 2`;
  - proprio through a 2×64 MLP;
  - crop 96 → 88: random at training, centre at evaluation (`augment.crop`).
- **BC head:**
  - 2×256 MLP, tanh, MSE on normalised actions;
  - AdamW, lr 3e-4, wd 1e-4, cosine with 500 warm-up steps;
  - 30 k steps, batch 256, bf16, EMA 0.999, checkpoint every 5 k.
- **`bc_state`:** the same head on the 25-d obs, standardised by the per-dimension mean and std of
  its own training frames. It trains in fp32; the vision arms train in bf16. That precision
  confound is stated with H4.
- **`bc_dagger` and `bc_matched` share one schedule** (skeptic B2).
  - **Iteration 0.** It is the `bc` run of the same seed: 30 k steps on demos 0–999, its last-step
    EMA, *not* its `select`-selected checkpoint. So iteration 0 is identical across `bc`,
    `bc_matched` and `bc_dagger`.
  - **Iterations 1–5:** each trains 10 k warm-started steps on the aggregate, 80 k steps in total
    for both arms.
    - `bc_dagger` flies demos 1000 + 200(i − 1) … with β = (0.5, 0.25, 0.125, 0, 0) per step.
      Every visited state is labelled by the SB3 teacher.
    - `bc_matched` adds the teacher's own demos at the same indices.
  - **Loop settings, pinned** (`bc_dagger.yaml` `loop`; `bc_matched.yaml` repeats the keys that
    apply without rollouts):
    - rollouts of iteration i fly the EMA weights at the last step of iteration i − 1, and no list
      is read to choose them;
    - warm start copies the weights and the EMA, and resets the AdamW state;
    - each iteration runs cosine with 500 warm-up steps over 10 k steps;
    - frames are sampled uniformly from the aggregate;
    - the β coin comes from a per-episode generator seeded by (run_seed, iteration, episode seed);
    - the executed action goes through `ila.data.collect.executable` (float32 before `env.step`);
    - rollouts fly the CPU fp32 single-sample path.
  - **Selection candidates:** iteration 5's checkpoints only (5 k and 10 k), EMA weights.
  - Labelled episodes total 2 000 in both arms (asserted). Labelled **frames** are reported beside
    H1, because DAgger rollouts can run to 361 steps against about 61.5 for demos.
- **`dp`:**
  - Horizons: `To = 2`, `Tp = 16`, `Ta = 8`.
  - ConditionalUnet1D with FiLM, `down_dims` [128, 256, 512], kernel 5, step embedding 128.
  - DDPM training: 100 steps, `squaredcos_cap_v2`, ε-prediction, `clip_sample`.
  - DDIM inference, K = 10, **`timestep_spacing: trailing`** (§8).
  - Identity action normalisation.
  - AdamW, lr 1e-4, wd 1e-6, cosine with 500 warm-up steps. 100 k steps, batch 128, bf16.
  - EMA power 0.75, max 0.9999. Checkpoint every 10 k.
  - Per-episode noise generator under P2-D3 item 4.
- **`dp` inference sweeps** (the same checkpoints; nothing is re-selected):
  - `Ta ∈ {1, 2, 4, 8, 12}`, without and with ACT temporal ensembling (m = 0.01; i = 0 is the
    oldest prediction), on `id` SS4–SS6;
  - K ∈ {1, 2, 5, 10, 25, 100} on `id` SS5.
  - K = 1 and 2 are ill-conditioned for any network under ε-prediction at ᾱ₉₉ = 2.43e-7 (§8).
    Those cells are read as schedule-limited.
- **`modular`:**
  - The estimator regresses the 11-d `priv` block, normalised by `train`-set mean and std, with a
    Huber loss (δ = 1.0).
  - The frozen `T_ppo` MLP receives the proprio fields, the estimated privileged fields and the
    true `time_fraction`, in P8's field order.
  - **Clock path (leak RISK 2, binding on Phase 7):**
    - `time_fraction` reaches the teacher only through a `ModularInput(student_input,
      time_fraction: float32[To])` built in `ila/obs/split.py`;
    - a test sets the 11 privileged fields of the env observation to NaN and requires the
      `modular` action to be bit-identical;
    - a test asserts that the estimator's `forward` has no clock argument.
- **Ablations** (3 seeds each, the same selection rule, flown on `id` SS4–SS6; tags never mixed
  into core tables). They are frozen by `configs/train/ablations/*.yaml`, each a base config plus
  the stated overrides, with steps unchanged from the base:

  | Ablation | Config SHA-256 |
  |---|---|
  | `bc_mix`, `dp_mix` (`T_mix`) | `14f9efab…f42c`, `9a94d0ae…29f9` |
  | `bc_pid`, `dp_pid` (`T_pid`) | `7a0f9104…3804`, `3638e62c…feb8` |
  | `bc_n100`, `bc_n300` | `b6003291…37c4`, `ed345e7f…a84b` |
  | `dp_n100`, `dp_n300` | `b620251a…ccc4`, `bab82411…ac26` |
  | `bc_minus_last_action`, `dp_minus_last_action` | `a59757dd…c414`, `3f5609cd…9e40` |
  | `bc_plus_clock` | `505f3bef…9caa` |

  **Ablation inputs** (leak re-audit RISK). The `+clock` and `−last_action` datasets get their
  proprio only through `DemoStore.student_inputs(name, variant)`, which builds the variant's
  columns by field name inside `ila/data`. No vision-student dataset ever gets a
  `privileged_observations` allowlist entry. The poison test is parametrised by variant; only
  `plus_clock` leaves `time_fraction` unpoisoned.

  N = 100 at 30 k steps is about 1 250 epochs. That is kept on purpose, with one step rule for
  every N. The `select`-based rule over the 5 k checkpoints guards against overfitting.
- **Appearance shift** (evaluation only): `bc_dagger`, `dp` and `modular` on `id` SS5–SS6, every
  variant. No frame of any variant is written to a store: the writer refuses every variant and any
  camera SHA-256 other than P1-D1's (`tests/test_data_collect.py`).
- **Always printed** beside every student, from P8's committed rows:
  - the teacher (`ppo` s0, e07);
  - `pid_feedforward`;
  - `pid_track_descend`;
  - `oracle_gated` (a commit-timing oracle, not a bound).
- **Inference path** (skeptic S5; tightens P2-D3 item 3).
  - Selection, the σ* search and every DAgger rollout fly the CPU fp32 single-sample path.
  - Both sides of every scored contrast fly the same path.
  - **GPU acceptance.** The batched GPU path is accepted for an arm only if its outcome flips
    against the CPU path on the 150 `select` episodes are **≤ 3** (2 %). P8's single-draw median
    is 1.9 %; the 6.6 % figure is an any-of-20 envelope, not a single-draw rate.
  - The test runs on the arm's selected checkpoint for seed 0 only, and its verdict applies to
    all five seeds. The same holds for Phase 9 parity. This is stated with the results.
  - **If the GPU path is refused**, the arm flies on CPU. P2-D2 costs `dp` at 170 ms per call at
    K = 10, and the Phase 8 budget is re-projected.

**4. Selection, tuning and failed runs.**
- **Selection rule.** For each run, among its candidate checkpoints (every saved checkpoint, EMA
  weights, unless the arm restricts the set as in §3):
  - pick the highest closed-loop success on `select`;
  - break ties by the lower pooled p95 closing speed on `select` (touched-down episodes);
  - then by the lowest step.
  - `dp` is selected at `Ta = 8`, K = 10. `R1` uses the same rule.
  - Students may well sit near 150/150 on `select`. If so, selection is mostly decided by the
    closing-speed tie-break, and the README says so.
- **Frozen lists** (leak NOTE 3). No student is flown on a frozen list, and no student outcome on
  one is read, before every run of every arm is selected.
  - Reading frozen-list keys for disjointness guards is allowed.
  - Reading P8's committed teacher and baseline rows is allowed.
- **The only tuning: DART σ.**
  - One `bc_dart` seed (seed 0) per σ ∈ {0.05, 0.10, 0.20}: three trials, logged as tuning, not
    core seeds.
  - σ* = the highest `select` success of the selected checkpoint, with the same tie-breaks; if
    still tied, the smaller σ.
  - `bc_dart` seeds 0–4 are then trained at σ*, seed 0 included (re-trained).
- **Nothing else is tuned:** no learning rate, no architecture, no `Ta` or K for the core row.
  Sweep results are reported, never used for selection.
- **Failed runs** (skeptic B3). No seed is dropped, replaced or re-drawn.
  - **Interruption:** an infrastructure interruption resumes from the last checkpoint with
    identical RNG state, and the resume is logged.
  - **Divergence:** a run whose loss goes non-finite is selected by the rule among its finite
    checkpoints. With none, it scores 0 success in every cell.
  - **Bug:** a run found to have a bug is never excluded alone. The bug is fixed, and every seed of
    that arm, and of any arm sharing the code path, is retrained before any retrained `select`
    number is read.
  - The plan's "or the run is excluded" (Phases 4 and 7) is read under this rule.

**5. Training budget** (skeptic S6).
- **Timing run.** Before the first Phase 4 run, a timed `dp` seed-0 run of 2 k steps (wall clock
  only, no `select` evaluation) and a timed `bc` run of 2 k steps are made.
- **Projection.** It covers every planned run: core arms, DART trials, DAgger, ablations and R1.
  If it exceeds **120 h** of GPU time:
  - training steps are cut **once**, by one factor for every family, before any evaluation number
    (`select` included) is read;
  - the factor is the smallest that brings the projection to ≤ 120 h;
  - checkpoint cadence and warm-up scale with it.
- The cut, its factor and the projection go in a dated entry. No later cut is allowed.
- Phase 4 replaces P2-D2's student episode-length assumption with measured lengths.

**6. Hypotheses.**
- **Scoring phases.** H1–H4 are scored in Phase 8, H5(a) and H5(b) in Phase 9. Each is scored as
  **supported**, **not supported** or **inconclusive**, with the point estimate and its 95 % CI.
- **Two-sided reading.** A contrast whose CI lies wholly on the other side of 0 is reported as an
  effect in the opposite direction, and scored "not supported". This does not apply to H5(b): a CI
  wholly above 0 is non-inferior and reported as such.
- **SS6.** Every SS6 contrast carries the flip floor (§1).

*Ceilings, measured before any student exists.*
- **From P8's committed rows** (frozen lists, read-only):
  - `T_ppo` s0: 100 % at `id` SS3–SS5 and `static`; 98.5 % (197/200: 3 hard landings) at `id`
    SS6; 91.0 % at `unseen_heading` SS6; 97.5 % at `unseen_seastate`;
  - `gated`: 100 / 98.5 / 89.5 / 63.0 % at `id` SS3–SS6;
  - `pid_feedforward`: 100 / 100 / 99.0 / 90.5 %.
- **On `train`** (`results/demos_summary.json`):
  - `T_ppo` 100 / 100 / 99.8 % (SS3 / SS4 / SS5);
  - `T_pid` 100 / 99.7 / 99.7 %;
  - `T_mix` 100 / 98.8 / 94.6 %, of which `gated` alone is 100 / 97.6 / 89.2 %.
- **No magnitude is restated.** Every scored contrast is between two students, or a student and its
  teacher, at a cell whose ceiling is 98.5–100 %, so each predicted gap is arithmetically possible.
  Unlike P8's H1, no hypothesis compares against a baseline already at the ceiling.
- **The null the ceiling can still produce** (skeptic S1) still gets a verdict from the rule, but
  is printed as "not supported (ceiling-limited: the largest attainable gain is below +5 points)",
  with the SS6 secondary beside it. That applies when the reference arm's point estimate in the
  scored cell exceeds 100 % − magnitude:
  - `bc_matched` or `bc` above 95 % at `id` SS5, or at `id` SS6 for the SS6 secondaries;
  - `dp` at `Ta = 8` above 95 % at `id` SS6.

- **H1 — compounding error.** Cell `id` SS5. Statistic: success(`bc_dagger`) −
  success(`bc_matched`), P8 `paired_bootstrap`. That function treats seeds as unpaired, which is
  conservative here, because both arms share iteration 0.
  - *Supported:* point estimate ≥ **+5 points** and the CI excludes 0.
  - *Inconclusive:* the CI excludes 0 and the estimate is positive but below 5.
  - *Not supported:* every other case.
  - *Secondary, same rule:* the same contrast at `id` SS6.
  - *DART secondary* (user decision, 2026-10-27; corrects plan §1.8): success(`bc_dart`) −
    success(`bc`) at `id` SS5 and SS6, same rule. Both are at N = 1 000 labelled episodes on the
    same 1 000 starting conditions. `bc_dart` − `bc_matched` (1 000 vs 2 000) is reported
    descriptively, unscored.
  - *Matching.* `bc_dagger` and `bc_matched` share iteration 0, the loop, the step count, the
    candidate set and the 2 000 starting conditions. They differ only in who chose the actions on
    the second 1 000.
- **H2 — multimodality.** Cell `id` SS4, arms `bc_mix` and `dp_mix`. **Scored on 3 seeds per arm**,
  the ablation count.
  - *Statistic.* D = W₁(`bc_mix`, ref) − W₁(`dp_mix`, ref) over touched-down `time_to_touchdown_s`
    (§1). The fraction of episodes without a touchdown is reported beside it.
  - *Reference.* It is built from P8's committed rows.
    `ila.data.mix.assign_rows(20261023, id SS4 rows)` gives 100 episodes to `ppo` and 100 to
    `gated`; each takes P8's committed row (`ppo` s0 from e07, `gated` from e01). Nothing is flown
    on the frozen list for it.
  - *Supported:* D ≥ **0.033 s** (one control step) and the CI excludes 0.
  - *Inconclusive:* the CI excludes 0 and 0 < D < 0.033 s.
  - *Not supported:* every other case.
  - *Precondition, met before training.* The two modes differ clearly on `train`. `mix_1000`
    touchdown time, p10 / p50 / p90:
    - `ppo`-assigned: 1.44 / 1.54 / 1.63 s (500 touched down);
    - `gated`-assigned: 2.84 / 3.37 / 5.80 s (480 touched down, 20 timeouts);
    - W₁ between them **2.41 s** (model; 12.1 s full).
  - *Stated with the result:*
    - The training mixture is 500 : 478 episodes after filtering, but about 1 : 2 by frames
      (`gated` episodes are longer).
    - All 22 filtered episodes are `gated`, mostly its longest waits.
    - `last_action` in the proprio can reveal the mode after the first step.
    - W₁'s bias grows as touchdowns get fewer, which works against an arm that times out.
  - *Plan §1.8's secondary* (the success drop `T_ppo`-trained → `T_mix`-trained is larger for `bc`
    than for `dp`) is reported at `id` SS4, unscored (§8).
- **H3 — chunk length vs deck bandwidth.** The same `dp` checkpoints. Cell `id` SS6. Statistic:
  success(`Ta = 1`) − success(`Ta = 8`), same-checkpoint contrast (§1).
  - Supported / inconclusive / not supported as in H1, with the magnitude **+5 points**.
  - It is reported as a curve over `Ta ∈ {1, 2, 4, 8, 12}` at SS4–SS6, with temporal ensembling
    beside it, unscored.
  - *Biases toward a null, stated with the result:*
    - checkpoints were selected at `Ta = 8`;
    - `Ta = 1` without ensembling can switch modes between replans.

    So a null cannot separate "chunking is harmless" from "replanning causes mode switching".
- **H4 — where the shortfall comes from.** Cell `id` SS6.
  - *Gaps:* perception gap P = S(`bc_state`) − S(`bc`); imitation gap I = S(teacher) −
    S(`bc_state`); D = P − I.
  - *Bootstrap:* §1.
  - *Supported:* D > 0 and the CI excludes 0.
  - *Not supported:* every other case. A CI wholly below 0 is reported as "imitation gap larger".
  - *Stated with the result:*
    - P includes the effect of withholding `time_fraction`, which `bc_state` sees and `bc` does
      not. P′ = S(`bc_state`) − S(`bc_plus_clock`) (3 seeds, `id` SS6) is reported beside it,
      unscored.
    - P also includes `bc_state`'s fp32 against `bc`'s bf16.
    - H4 is conditional on teacher seed 0.
  - *`modular` vs the best end-to-end vision student* at `id` SS6: reported two-sided with its CI,
    unscored, with no predicted direction. "Best" = the highest mean `select` success of the
    selected checkpoints among `bc`, `bc_matched`, `bc_dart`, `bc_dagger` and `dp`. Ties go to the
    lower pooled `select` p95 closing speed, then to that order.
- **H5 — deployment: inference latency on this machine** (RTX A4000, i9-10980XE, WSL2), model scale.
  33.3 ms is one 30 Hz control step. No embedded target is measured.
  - **(a)** The exported `dp` (K = 10) runs one full replanning call (image stack + proprio →
    chunk) at batch 1 with **p99 < 33.3 ms** on at least one provider that passes parity.
    - The method is P8's latency harness: ≥ 200 warm-up calls, ≥ 2 000 timed, synchronised
      brackets.
    - **Parity, frozen here** (skeptic B5):
      - *exported model:* seed 0's selected checkpoint;
      - *reference:* torch fp32 CPU, B = 1, the same initial noise;
      - *open-loop:* 1 000 `select` inputs, per-step ε̂ max-abs ≤ 1e-4. The final-chunk max-abs is
        reported, not judged, because of the ×2 029 first-step gain;
      - *closed-loop:* the 150 `select` episodes (never a frozen list), outcome flips against the
        CPU single-sample path ≤ 3. This test is binding.
    - **Scored providers:** ORT CPU fp32, ORT CUDA fp32, torch-eager CUDA fp32 and ORT TensorRT
      fp16. fp16 passes only under the same tests. Any other variant (CUDA graphs,
      `torch.compile`) is reported, unscored.
    - **If no provider passes**, H5(a) is "not supported", and §7's d uses the p99 of the torch CPU
      fp32 single-sample path.
    - P2-D2 measured a dummy `dp` in eager PyTorch at 80.6 ms p50 on the GPU.
  - **(b)** The latency-arm `dp` (§7) at `id` SS5 minus synchronous `dp` at `id` SS5,
    same-checkpoint contrast (§1).
    - *Supported* (non-inferior): CI lower bound ≥ **−3 points**.
    - *Not supported:* otherwise.

**7. Latency-arm rule** (skeptic B4).
- **Delay.** d = ⌈p99 / 33.333 ms⌉ control steps. p99 is the batch-1 full-call latency of the
  fastest parity-passing provider (§6 H5(a)), with the CPU fallback stated there.
  - It is computed once per arm (`bc_dagger`, `dp`), and the same d is used for every seed.
  - **`bc_dagger` parity:** open-loop action max-abs ≤ 1e-4 on the same 1 000 `select` inputs,
    and closed-loop ≤ 3 flips on the 150 `select` episodes. The same providers and the same
    fallback apply.
- **One inference worker; requests never overlap.** A request is issued with observation t only
  when none is in flight. Its result is available at step t + d.
  - **BC:** the returned action is held from t + d until the next result arrives. The next request
    is issued at t + d.
  - **`dp`:** a request is issued when the executing chunk has d steps left (immediately if fewer).
    If the executing chunk runs out before the next result arrives, its last action is held.
    The returned chunk executes from index (To − 1) + d, aligned in time, for Ta steps. If
    (To − 1) + d + Ta > Tp, the last chunk action is held. At `To = 2`, `Tp = 16`, `Ta = 8`, that
    holding starts at d > 7.
  - **Start of an episode:** the first action is held at zero until the first result arrives.
  - **d = 0:** the arm equals the synchronous arm row for row. This is tested.
- **Flown** on `id` SS4–SS6, `dp` at `Ta = 8`, K = 10, tag `latency_arm`. H5(b) is scored at SS5
  only.

**8. Deviations, decisions and incidents before freeze.**
- **User decisions, 2026-10-27:**
  - `T_mix` balanced per sea state (`demos.yaml` had said `per_episode`). `mix_1000` was collected
    only after the change, at `c941246`; no coin-drawn store was ever made.
  - DAgger flies `demos` 1000–1999 instead of fresh streams.
  - The H1 DART secondary is `bc_dart` − `bc`, which corrects plan §1.8.
    - CLAUDE.md non-negotiable 5 names `bc_matched` as DART's comparator.
    - For DART at N = 1 000, the equal-count arm is `bc`, and that reading is the one adopted.
- **`dp.yaml` gains `timestep_spacing: trailing`** (the P2-D4 open item). Its cost was measured
  before training by `policy-trainer` (CPU and GPU, dummy `dp`, TF32 off):
  - ᾱ₉₉ = 2.43e-7, because β₉₉ is capped at 0.999.
  - The first step's gain from ε̂ to x̂₀ is √((1 − ᾱ)/ᾱ) = **2 029**, against 7.15 for `leading`'s
    first step (k = 90).
  - ε̂ differences of about 1e-6 become chunk gaps of up to 2.6e-4 (CPU B = 1 vs 64) and 7.7e-4
    (GPU vs CPU). Under `leading` they were 5e-6 and 1.5e-5.
  - A one-ulp change in the initial noise moves the chunk by up to 5.9e-4.
  - A trained network is predicted to show 1e-3 to 1e-2.
  - At a fixed batch size, a slot is bit-identical under permutation and under changes to the
    other slots, and the CPU single-sample path is bit-identical.
  - P2-D3 item 2's numbers and the dummy determinism study were measured under `leading`. This
    entry supersedes them for `trailing`.
  - `trailing` is kept because `leading` starts K = 10 at k = 90, which is not pure noise. K = 1 and
    2 are read as schedule-limited under either spacing (§3).
  - **Code:** `DPConfig.timestep_spacing` defaults to `trailing`, unknown values raise, and the
    slot-independence tests check:
    - bit-identity at a fixed batch size;
    - ε̂ within 1e-5 across batch sizes;
    - the chunk within gain × 1e-5.
- **P2-D3 item 3 is tightened**, from "≤ P8's 6.6 % floor" to ≤ 3 of 150 `select` flips (§3).
  P2-D3 also called `select` (SS3–SS5) against P8's SS6 figure "a conservative comparator". That
  was backwards: P8 saw no flips on SS3–SS5.
- **Plan Phase 9, item 2** judged closed-loop parity on 50 `id` SS5 episodes. That is a frozen-list
  read inside a choice that sets d for a scored cell. It is replaced by the 150 `select` episodes
  (§6 H5(a)).
- **Plan §1.8's H2 secondary** (directional) becomes unscored. The reasons: its arms are 3-seed
  ablations, and `T_ppo`-trained and `T_mix`-trained arms differ in data as well as in teacher.
- **`configs/eval/matrix.yaml`** now names the ablation arms by their config names, and gives the
  latency arm its CPU fallback (re-review).
- **Plan §1.8 "H5" scored in Phase 8** becomes Phase 9 for H5(b), because d exists only after
  Phase 9.
- **Budget timing:** a timed 2 k-step `dp` run comes before Phase 4, instead of the first full `dp`
  seed in Phase 6 (§5).
- **Conventions:** the il-conventions rule "the clock never enters a network" now names its three
  declared exceptions (`+clock`, `modular`'s teacher passthrough, `bc_state`). It is mirrored in
  `docs/CONVENTIONS.md` (leak NOTE 1).
- **Leak RISK 1 closed in code.**
  - `ObsContract.field` is renamed `audit_field`.
  - `tests/test_render_access.py` forbids `audit_field`, `privileged_observations`, `model_times`,
    `audit_arrays` and `estimator_targets` in `ila/train`, `ila/policies`, `ila/deploy` and
    `ila/eval`. It has an empty, reviewed per-file allowlist (a future `bc_state` dataset and the
    `modular` target code may each be added only after a leak review).
  - The poison test also scrambles the numeric episode attributes.
  - A test requires every class in `ila.train.dataset` (except `bc_state`'s) to be registered in
    `STUDENT_DATASET_CLASSES`. It is xfail until Phase 4 creates the module.
  - `tests/test_data_store.py`'s docstring fixes the windowed contract Phase 4 must meet.
- **DART stores re-collected (incident, before any training).**
  - The first DART sets (`fc5388a`) executed the float64 `clip(label + σz)` but stored it as
    float32 `action_exec`. Open-loop replay of the stored actions diverged from step 1–7, and one
    of 60 replayed episodes ended in a different class.
  - Collection now rounds to float32 before `env.step` (`d198f08`), and the three sets were
    re-collected. Open-loop replay is now binding for every store.
  - The old stores are kept outside the dataset in `${data_root}/demos/old_float64/`.
  - Teacher-only sets were unaffected: the teachers return float32. A re-fly gave 40 of 40
    episodes byte-identical per store.
- **Collection commits.**
  - `ppo_2000` and `pid_1000` were collected at `fc5388a`, `mix_1000` at `c941246`, and the DART
    sets at `d198f08`, all on clean trees.
  - The `demos.yaml` changes between them (`T_mix` draw, DAgger block) do not touch the earlier
    sets.
- **Frozen-list reads and teacher flights in tests**, for the record. Each is a key read, a P8 row
  read, or a scripted, zero-action or teacher-only flight; nothing is trained, selected or scored
  on them.
  - Phase 1 fixtures moved to `train` SS5 rows (P1-D2 obligation 1). Four stay on frozen lists,
    because `train` has no static deck and no other vessel:
    - `test_render_geometry` (`static`);
    - `test_render_scene` (`static`);
    - `test_p8_adapter::test_set_motion_matches_p8_run_chunk` (`static` and `unseen_vessel`);
    - the `test_phase2_jobs` render bench (`static`).
  - `tests/test_data_mix.py` reads the `id` SS4 keys.
  - `tests/test_data_episodes.py` reads frozen lists for its P8-equivalence and guard tests.
  - `tests/test_eval_runner.py::test_ppo_teacher_reproduces_committed_rows` flies two `id` SS5
    episodes with the teacher (Gate 2).
- **Not adopted, for the record:**
  - Re-flying `gated` on `id` SS4 to check the e01 rows used in H2's reference. P8 published them,
    and Phase 2 reproduced P8's runner bit for bit on `pid_feedforward` and `ppo`. If a reviewer
    asks before Phase 8, it is 200 teacher episodes.
  - Building H2's reference from all 397 touched-down rows at 50/50 weights. The user-approved
    `assign_rows` construction is kept.
- **Obligations for Phase 4** (P1-D2 items 3, 4 and 6):
  - the dataset reads only through `DemoStore.student_inputs` and `labels`, and uses `meta()` only
    for the success filter;
  - it registers in `STUDENT_DATASET_CLASSES`;
  - it meets the windowed contract in `tests/test_data_store.py`;
  - a timed 2 k-step `dp` and `bc` run comes first (§5).

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
| `make lint` green locally and in CI | Local: ruff and format clean, `mypy --strict` 46 files. CI run 40520506247 (job 122245991307, commit `28ff2cc`): same, 46 files | PASS |
| `make test` / `make smoke` green locally | 5 passed (includes the `checkpoints` provenance test) | PASS |
| Tests green in CI (CI skips `checkpoints`) | Run 40520506247: 4 passed, 1 deselected | PASS |
| `python -c "import rld, ila"` | Works locally and in the CI install step | PASS |
| `results/p8_provenance.json` matches P8's published hashes | Committed; `all_match: true` (P0-D1) | PASS |

Reviewers: none are assigned to Phase 0.

Incident (attempt 1 → 2): the first CI run was green, but its mypy saw 39 files against 46 locally.
An unanchored `data/` in `.gitignore` had also matched `src/ila/data/` and `configs/data/`, so
seven skeleton modules and the starter kit's `configs/data/demos.yaml` were not committed. Root
patterns are now anchored (`28ff2cc`) and the re-run checks all 46 files. No threshold changed.

### Gate 1 passed (2026-10-27)

| Criterion (plan Phase 1 exit gate) | Evidence | Verdict |
|---|---|---|
| All tests green | Local `make test`: 191 passed, including the `slow`, `pybullet` and `checkpoints` tests (render invariance: off / camera / full scene over 300 airborne steps, a descent to contact, re-attach after reset, and a negative control). `make smoke`: 6 passed. CI run 40533051917 (job 122290788981, commit `844b227`): 183 passed, 8 deselected (`slow`/`checkpoints`) | PASS |
| `make lint` | ruff and format clean; `mypy --strict` clean on 47 files, locally and in the same CI run | PASS |
| Leak tests | `tests/test_obs_split.py`: every P8 field classified once, offsets equal P8's `obs_fields()`, `StudentInput` refuses privileged or unnamed proprio, each ablation changes exactly one field, slice scanner (32 must-flag patterns). `tests/test_render_access.py`. Background and no-world-fixed-visual tests in `tests/test_render_scene.py` | PASS |
| Render-invariance tests | Listed in the first row; per-step state byte-identical across all three render modes | PASS |
| `privileged-leak-auditor` verdict | Round 1: FINDINGS (2 RISKs, 5 NOTEs). Round 2: FINDINGS (2 RISKs, 3 NOTEs). Round 3: **CLEAN**, no LEAK or RISK. Its three optional tightenings were applied afterwards; they touch tests and P1-D2 only | PASS |
| P1-D1 recorded | Plate visuals, the twin, P8's static marker hidden, ground recolour, camera numbers with `configs/render/camera.yaml` SHA-256 `53f80265…0380` (matches the committed file), far plane 20 m (deviation approved by the user) | PASS |
| `docs/CONVENTIONS.md` identical to the il-conventions skill | `tests/test_conventions_mirror.py` | PASS |
| P8 untouched | Submodule at `7237b706…`; no diff under `external/` | PASS |

Reviewers: `privileged-leak-auditor` CLEAN (third round).

Incident: the first CI run for Phase 1 (40531591466) failed twice in the install step with Errno 28
(no space left) while building `tensorrt_cu13_libs`, before any code ran. A "Free disk space"
step in `.github/workflows/ci.yml` (`844b227`) raised free disk from 13 GB to 35 GB, and the
re-run passed. No threshold changed.

### Gate 2 failed, attempt 1 (2026-10-27)

| Criterion (plan Phase 2 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` / `make test` | ruff and format clean; `mypy --strict` 53 files; 464 passed, including the `slow`, `pybullet`, `checkpoints` and `gpu` tests | PASS |
| Both reproduction comparisons pass with zero differing rows | `results/reproduce_p8/comparison.json` (`75b34e8b…31fa`): `pid_feedforward` 2 800/2 800 and `ppo` 400/400 identical, differ 0, missing 0; `run_info.json` (`4ea43413…e002`): render `full`, `ila_commit` `c88c106`, no dirty paths | PASS on row identity |
| …with render **on** | No committed artifact records that frames were rendered. `run_info.json` writes `render: "full"` as a constant, and no frame count is kept. The indirect evidence is consistent with rendering: every row carries `render_cfg_sha256`, and the wall time (254.8 s) matches P2-D2's rendered rates, not physics-only. But it is not an artifact of the run | UNVERIFIED |
| Label parity met (or SB3 path chosen) | `results/label_parity.json`: max-abs 3.58e-7 ≤ 1e-6 (P2-D1) | PASS |
| Throughput CSV committed | `results/throughput.csv`, `results/throughput_projection.csv` (P2-D2) | PASS |
| Budget and determinism decisions recorded | P2-D2, P2-D3 (P2-D4 for pulled-forward work) | PASS |
| `privileged-leak-auditor` (timing, render invariance) | **FINDINGS**: no LEAK, 1 RISK, 4 NOTEs. Pre-gate RISK 1 (noise-key clock) is closed. Render timing in `fly_chunk` is correct, and both backstops hold (`StaleSceneError`, attach after a step). New RISK: a later change could stop rendering for teachers, and the gate, every test and run_info would stay green | FAIL |

Reviewers: `privileged-leak-auditor`, pre-gate FINDINGS (RISK 1 since closed), then gate review
FINDINGS (new RISK below).

**Failing items and the smallest fix for each.**
1. **Render-on is not shown by an artifact; RISK, gate review.**
   - `fly_chunk` counts the frames rendered and the timed checks per episode, and asserts that
     frames == steps + 1 whenever render ≠ `off`.
   - `reproduce.run` adds up those counts in `run_info.json` and takes `render` and the SHA
     from the arms actually flown.
   - A spy test (workers = 1) asserts that render calls == steps + 1 in `full` and `camera`
     mode, 0 in `off`, and pins the timed-step schedule.
   - Then re-run `make reproduce-p8` and re-review.
2. **Guard scope (NOTE 2), done now because it is cheap.**
   - A test asserts that the `nn.Module` subclasses under `ila.policies` equal a reviewed list,
     so a new network fails until its `forward` joins the signature allowlist.
   - The fixed-noise invariance test is parametrised over the registry arms.

No threshold changes.

**Corrections recorded here** (entries are not edited):
- **P1-D2 item 5 is superseded.** Only `eval/runner.py` is exempt from
  `tests/test_render_access.py`. The `eval/batched.py` exemption was removed in Phase 2 (leak
  audit NOTE 2), and `batched.py` scans clean.
- **P2-D3 item 4's guard list** also includes the AST read-site allowlist
  (`tests/test_step_guard.py`).
- **Frozen-list use not named in P2-D1.**
  - `test_ppo_teacher_reproduces_committed_rows` (`tests/test_eval_runner.py`) flies two `id`
    SS5 episodes with the `ppo` teacher, and compares them with P8's published rows.
  - Like the gate, it trains, selects and tunes nothing.
  - It cannot move to the `train` pool, because P8's reference rows exist only on the frozen
    lists.

### Gate 2 passed (2026-10-27)

| Criterion (plan Phase 2 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` / `make test` | ruff and format clean; `mypy --strict` 53 files; 473 passed in 181.9 s, all markers (`slow`, `pybullet`, `checkpoints`, `gpu`), at `5d09486` | PASS |
| Both reproduction comparisons pass with zero differing rows | `results/reproduce_p8/comparison.json` (`89007066…09df1`): `passed: true`; `pid_feedforward` 2 800/2 800 vs P8 `e01`, `ppo` seed 0 400/400 vs P8 `e07`; differ 0, missing 0, headers identical | PASS |
| …with render **on** | `results/reproduce_p8/run_info.json` (`8764377e…cc65`), taken from the arms flown: render `full` and `render_cfg_sha256` `53f80265…0380` (= P1-D1) for both methods. Frames: `pid_feedforward` 428 781 = 425 981 steps + 2 800 episodes, `ppo` 25 197 = 24 797 + 400. Timed image/obs step-count checks: 18 343 and 1 469. `ila_commit` `e54fa2e`, no dirty paths, 254.4 s. P8-column row SHA-256 `d5ddc55d…ec31` / `a836275f…bf50`, the same as both earlier runs | PASS |
| Label parity met (or SB3 path chosen) | `results/label_parity.json`: max-abs 3.58e-7 ≤ 1e-6. Phase 3 still flies the teacher on SB3 (P2-D1) | PASS |
| Throughput CSV committed | `results/throughput.csv` (`11898e1e…1879`) and `results/throughput_projection.csv` (`59fb8fa3…9191`) | PASS |
| Budget and determinism decisions recorded | P2-D2: 96 px kept, about 12.2 h central / 70.6 h upper outside training. P2-D3: CPU single-sample path bit-identical; GPU path TF32 off, with a binding re-measurement on a trained `dp` before Phase 8. P2-D4: work pulled forward | PASS |
| `privileged-leak-auditor` (timing, render invariance) | Pre-gate: FINDINGS (noise-key RISK, closed). Attempt 1: FINDINGS (render evidence RISK). Attempt 2: **no LEAK, no RISK**, 2 NOTEs on the record, handled below | PASS |

Reviewers: `privileged-leak-auditor`, three rounds; final round no LEAK and no RISK.

**Supersessions** (entries are not edited):
- P2-D1's evidence named the `c88c106` re-run's files (`75b34e8b…`, `4ea43413…`, 254.8 s).
  Those files are superseded by the `e54fa2e` run in the table above, which adds the frame
  evidence. The rows are unchanged.
- The full-row `episodes_<method>.csv` under `artifacts/` now has two trailing columns,
  `frames_rendered` and `timed_checks`, after the P9 columns. Its SHA-256 has therefore changed.
  The P8-column CSVs that the comparison uses have not.

**Phase 9 obligations** (leak audit, gate review, NOTE B):
- Assert that the exported graph's input names are a subset of
  {images, proprio, noise, timestep}.
- Extend the step-guard read-site scan to any `BatchPolicy` in `ila/deploy`, including
  aliased reads of `key.step` (e.g. `k = r.key; k.step`, `r.key[2]`). Today the scan covers
  `eval/batched.py` and `DPStudent` only.

Incidents:
- Attempt 1 failed on the render-on evidence. The fix is `e54fa2e`.
- The first full reproduction ran on an uncommitted tree. It was re-run twice on committed
  trees.
- No threshold changed.

### Gate 3 passed (2026-10-27)

| Criterion (plan Phase 3 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` / `make test` | ruff and format clean; `mypy --strict` 55 files; 560 passed, 1 xfailed, all markers, at `2b9ae07`. The xfail is the intentional Phase 4 registration test | PASS |
| Splits and manifests committed | `splits/demos_0-999.parquet` `0012f80b…4307`, `splits/demos_0-1999.parquet` `20940abc…05ef`, `splits/select.parquet` `07ecf049…422b`, `splits/MANIFEST.csv` `2be435ab…02dc`; regenerated byte for byte by `scripts/make_splits.py` | PASS |
| Disjointness, no SS6/90°, train pool only | `tests/test_data_episodes.py`; `results/demos_summary.json`: 0 attribute violations on all six stores; independently re-checked by `privileged-leak-auditor` | PASS |
| Demos complete | `ppo_2000` 2 000, `pid_1000` 1 000, `mix_1000` 1 000, `dart_0.05/0.10/0.20` 3 × 1 000; collected on clean trees (`fc5388a`, `c941246`, `d198f08`); store and manifest SHA-256s in `results/demos_summary.json` (`c56fa833…13b7`, committed) | PASS |
| Audits clean | 20/20 re-flown frames byte-identical (open-loop replay, binding) and 1 000/1 000 labels identical per store; DART re-collected after the float32 incident (P3-D1 §8) | PASS |
| P3-D1 frozen with the user's dated approval | Approved by the user 2026-10-27 (revision 3), frozen at `2b9ae07`. Block SHA-256 **`00e223302d033ad16fd4882810a873c41d7108a9938cbf66eae32ebf48a73fc5`** (heading to the line before `### Gate 0 failed`, UTF-8, final newline included), recomputed independently by `results-skeptic` | PASS |
| `privileged-leak-auditor` | Round 1: FINDINGS (no LEAK, 3 RISKs, 6 NOTEs). Round 2: FINDINGS (1 new RISK, the ablation read path). Round 3: **CLEAN** | PASS |
| `results-skeptic` (on the P3-D1 draft) | Round 1: 5 BLOCKING, 10 SHOULD-FIX. Round 2: no BLOCKING, 3 SHOULD-FIX. Round 3: **nothing BLOCKING, no new findings**; every SHA-256 in §1–§3 matches the committed files | PASS |

Reviewers: `privileged-leak-auditor` CLEAN (third round); `results-skeptic` no BLOCKING (third round).

Notes:
- **Reproduction gate not re-run.** Phase 3 changed nothing under `ila/render/`, `ila/eval/runner.py`
  or the adapter. `ila/obs/split.py` gained one method (`audit_field`), and no runner path calls it.
- **Incidents:** the DART float32 re-collection and the `trailing` batch sensitivity, both before any
  training (P3-D1 §8). No threshold was relaxed. P2-D3 item 3's acceptance rule was tightened.

## Phase 4

*Before you start (from Phase 3).*
- **P3-D1 is frozen** (`00e22330…3fc5`). Configs are frozen by SHA-256, so any config edit is a dated
  deviation.
- **First task: the timed 2 k-step `dp` and `bc` runs** (P3-D1 §5), wall clock only and no `select`
  evaluation, then the 120 h projection over every planned run.
- **Dataset** (`ila/train/dataset.py`):
  - It reads only through `DemoStore.student_inputs(name, variant)` and `labels`, and uses `meta()`
    only for the success filter.
  - It registers every class in `STUDENT_DATASET_CLASSES`. The xfail turns into a hard failure as
    soon as a class exists.
  - It meets the windowed contract in `tests/test_data_store.py`.
  - `bc_state` may read `teacher_obs` only through a reviewed `AUDIT_ALLOWLIST` entry.
- **Iteration 0** of `bc_matched` and `bc_dagger` is the `bc` run of the same seed, using its
  last-step EMA. So `bc` must keep that checkpoint.
- **`bc_matched`** follows the DAgger schedule (80 k steps). It can only finish after the loop code
  of Phase 5 exists, or with an equivalent teacher-only loop.
- **DART σ search:** `bc_dart` seed 0 at each σ (3 trials, logged as tuning). DART sets are not
  success-filtered.
- **Selection, σ\* and DAgger rollouts** fly the CPU fp32 single-sample path. The GPU path needs ≤ 3
  of 150 `select` flips first (P3-D1 §3).
- **Stores:** `${data_root}/demos/{ppo_2000,pid_1000,mix_1000,dart_0.05,dart_0.10,dart_0.20}.h5`.
  `old_float64/` is not part of the dataset.

### P4-D1 — Training implementation choices P3-D1 leaves open (2026-10-28)
*Decision.* The settings below are not fixed by any frozen config or by P3-D1. They are
implementation choices within the frozen protocol, not deviations. They are recorded here before
any training run, and every run manifest repeats them.
1. **Proprio standardisation (vision arms).** The encoder's `proprio_mean`/`proprio_std` are
   fitted once per run, at train start, from the allowlisted proprio of that run's own training
   frames. They are stored in the checkpoint and restored on resume. `bc_matched` inherits the
   values of the `bc` run it starts from.
2. **Optimiser details.** AdamW betas (0.9, 0.999) and eps 1e-8 (torch defaults). No gradient
   clipping; the gradient norm is logged only.
3. **Held-out loss.**
   - It is computed on `ppo_2000` stream indices 1950–1999. These are 50 `train`-pool episodes,
     all `success`, read through the same student path.
   - Plan Phase 4 item 4's "held-out `tune` demos" is **not** followed. `tune` feeds `select`
     only (P3-D1 §2), so no `tune` episode may enter training code.
   - These episodes are disjoint from the training episodes 0–999 by episode, but not by
     realization: all 50 share wave realizations with training episodes. So held-out loss
     measures generalisation within a realization.
   - It is logged only. No selection, stopping or threshold reads it.
   - `bc_matched` trains on these episodes from iteration 5 on. The overlap is logged, and its
     held-out loss is contaminated from then on.
4. **`bc_matched` warm start** (P3-D1 §3, "copies the weights and the EMA"):
   - iteration 1 takes the same-seed `bc` run's step-30 000 raw weights as weights and its EMA as
     EMA;
   - each iteration resets AdamW and runs cosine with 500 warm-up steps over 10 k steps.
5. **`dp` EMA.** It follows the `diffusion_policy` reference `EMAModel`: the first two updates
   are copies, then decay 1 − (1 + n − 2)^−0.75, capped at 0.9999. The BC family uses a constant
   0.999 from init, without bias correction.
6. **Timing runs** (P3-D1 §5):
   - they use the full config's learning-rate schedule, so they time the first 2 k steps of a real
     run;
   - they skip held-out loss and checkpoints, are tagged `timing`, and are refused by selection.
7. **Run guards.** Training, tuning and timing runs refuse a dirty tree, and refuse a config whose
   SHA-256 differs from P3-D1's.
8. **GPU training is not bit-reproducible run to run** (cuDNN, bf16). Resume bit-identity is
   tested on the CPU path only. An infrastructure resume (P3-D1 §4) restores RNG state, but the
   continued GPU run is not claimed to be identical to an uninterrupted one.
9. **`bc_state` data path.** `ila/train/state_dataset.py` reads `teacher_obs` through one
   `AUDIT_ALLOWLIST` entry (`privileged_observations`). `privileged-leak-auditor` approved it on
   2026-10-28, on the condition that a name scan confines `StateFrames` and `state_dataset` to
   `train/{state_dataset,loop,smoke}.py`.

*Reason.* Each item is something a reviewer would otherwise have to reverse-engineer from code.
Item 3 also blocks a plan reading that would break P3-D1 §2.

*Evidence.* `privileged-leak-auditor`, Phase 4 pre-run review, 2026-10-28: no LEAK, 1 RISK (the
`StateFrames` name scan, fixed in code before the first run), 6 NOTEs.
- The six stores have 0 realization-key or episode-seed overlap with `select` and with the five
  frozen lists.
- Every store holds SS3–SS5 only, at 45°, 135° and 180°.

*Consequence.*
- `training-run-diagnostician` reads held-out loss as within-realization.
- Smoke runs use 8 000 (`bc`) and 10 000 (`bc_state`) steps. At 4 000 steps, the un-corrected EMA
  still weighted the random init by 0.999^4000 ≈ 1.8 %, and actions came out about 3 % short. At
  30 k steps that weight is about 1e-13. Only the smoke step counts changed; no frozen
  hyper-parameter did.

### P4-D2 — Training budget: projection 91.6 GPU-h, no cut (2026-10-28)
*Decision.* **No step cut** (P3-D1 §5). The projection over every planned training run is
**91.6 GPU-hours**, under the 120 h limit, so the cut factor is 1.0. Under P3-D1 §5, no later cut
is allowed.

*Evidence.* Timing runs at `661ed76` (clean tree), RTX A4000, 2 000 steps each, using the full
configs' schedules. Tagged `timing`; never selected; no `select` number existed when this entry
was written.

| Run | Loop rate | Wall clock (2 k steps) | Per-run overhead |
|---|---|---|---|
| `dp` s0 (batch 128, bf16) | 12.79 steps/s | 168 s | 11.4 s |
| `bc` s0 (batch 256, bf16) | 10.66 steps/s | 198 s | 10.9 s |
| `bc_state` s0 (batch 256, fp32) | 347.7 steps/s | 11 s | 4.8 s |

Projection: `results/phase4/budget_projection.csv` (SHA-256 `d5a4d1e0…4cec8980`).

| Group | GPU-h |
|---|---|
| Phase 4: `bc_state` 0.13, `bc` 3.92, `bc_matched` iterations 1–5 6.53, DART σ trials 2.35, `bc_dart` 3.92 | 16.9 |
| `bc_dagger` iterations 1–5 (rollouts on the CPU, not counted) | 6.5 |
| `dp` 5 seeds × 100 k steps | 10.9 |
| `modular` (costed as the `bc` encoder) | 3.9 |
| Ablations: 6 `bc` × 3 seeds (14.1), 5 `dp` × 3 seeds (32.6) | 46.8 |
| `R1_lerobot_dp` (its Phase 6 config isn't written yet; costed as `dp` at 100 k steps) | 6.5 |
| Timing runs already spent | 0.1 |
| **Total** | **91.6** |

*Assumptions.* Selection, DAgger rollouts and evaluation run on the CPU and are not training GPU
time. `R1` and `modular` are costed by proxy, as above. If either proxy is far off, the 28 h
margin absorbs it. A later overrun is reported, never fixed with a second cut.

*Consequence.*
- Every arm trains at its frozen step count.
- Measured GPU-hours per run go in each run's manifest and in the Gate 4 table.
- Student episode lengths (P3-D1 §5, last line) are measured from the `select` flights of the
  selected checkpoints, and re-project P2-D2's evaluation budget at Gate 4.

### P4-D3 — DART σ* = 0.10, the only tuning (2026-10-28)
*Decision.* σ* = **0.10**. `bc_dart` seeds 0–4 train on `dart_0.10`, seed 0 re-trained
(P3-D1 §4).

*Evidence.* Tuning log: three trials, `bc_dart` seed 0, 30 k steps each, at `32c8cf9`. The
selected checkpoint of each trial (EMA weights, CPU fp32 single-sample path) was flown on the 150
`select` episodes. Files: `results/phase4/dart_tune/selection_sigma_{0.05,0.10,0.20}.json` and
`sigma_star.json` (SHA-256 `516b21c1…31799826`).

| σ | Selected step | `select` success | Losses | p95 closing speed (m/s, model) |
|---|---|---|---|---|
| 0.05 | 25 000 | 149/150 | 1 bounce | 0.372 |
| **0.10** | **30 000** | **150/150** | none | **0.322** |
| 0.20 | 15 000 | 149/150 | 1 hard landing | 0.377 |

*Caveat, stated with every DART result.*
- σ* was decided by one `select` episode. All three trials sit at the `select` ceiling, and
  a 150/150 vs 149/150 difference is well inside the Wilson intervals.
- The choice is by rule, not evidence that 0.10 is better. It is final; no trial is added.
- σ* = 0.10 is not 0.20, so P3-D1 §2's clear-colour statement for 0.20 does not apply.
  `dart_0.10` has 24 frames with a far-plane residual (max fraction 0.148).

*Consequence.* No other tuning is permitted (P3-D1 §4). The trial runs are tuning runs, never
core seeds.

### P4-D4 — Run verdicts, two corrections, measured lengths (2026-10-29)
*Decision.* This entry records the `training-run-diagnostician` verdicts on all 23 Phase 4 runs,
corrects P4-D1 item 8 and P4-D3, and fixes the disclosures that every later use of these arms
carries. No run is excluded, no seed is dropped or replaced (P3-D1 §4), and no threshold changes.

*Verdicts.* Every run's provenance was re-verified: config and data SHA-256, clean tree, every
checkpoint SHA-256 recomputed against its manifest. Selection was recomputed from the rows.

| Arm | Runs | Verdict | Arm-level caveat |
|---|---|---|---|
| `bc_dart` σ trials | 3 | TRUSTWORTHY | σ* decided by one `select` episode (P4-D3) |
| `bc` | s0–s4 | TRUSTWORTHY | selection is noise-level (all candidates 140–149/150; winner's curse about +2 episodes); strong `last_action` reliance (∂a/∂last_action diagonal 0.75–0.91) |
| `bc_state` | s0–s4 | TRUSTWORTHY | at the `select` ceiling (148–150 for every candidate) |
| `bc_matched` | s0–s4 | CAVEAT (all five) | see below |
| `bc_dart` | s1–s4 TRUSTWORTHY; s0 CAVEAT | | s0 is the tuning trial (correction 2); selection at the ceiling |

**`bc_matched` caveats.** None is a defect in a run.
- (a) Iteration 5's held-out loss is contaminated (P4-D1 item 3). The last clean point is the
  end of iteration 4, where held-out loss is about 20 % below `bc`'s.
- (b) `bc` → `bc_matched` changes both data (2×) and steps (+50 k). `bc`'s own
  `last_action` reliance rises with steps alone (0.57 → 0.81 from 5 k to 30 k on s0), so this pair
  is not a clean data-scaling contrast and is not reported as one. `bc_matched` vs `bc_dagger`
  (H1) is matched in steps, data count and schedule, and is unaffected.
- (c) The two candidates differ by 1–4 `select` episodes, which is within binomial noise.
- (d) `last_action` reliance rises across iterations, from 0.78–0.82 to 0.86–0.91. On
  DART-perturbed states, open-loop MSE is 10–15 % worse than `bc`'s, even though it is about
  20 % better on clean states.
- (e) In s1, the selected checkpoint goes off pad on `select` SS4 #35 at 1.234 m/s normal
  closing speed (model scale), about 2× the fastest `bc` touchdown. It is scored as a failure.
  It is a dive mode to watch in Phase 8.
- Bookkeeping: checkpoint `global_step` counts this run's steps only (5 k–50 k). The candidates
  are cumulative steps 75 k and 80 k of P3-D1's 80 k schedule.

*Correction 1, P4-D1 item 8.* "GPU training is not bit-reproducible run to run" is wrong as
observed.
- Determinism is **not guaranteed**: non-deterministic cuDNN algorithms are allowed. But it was
  **observed** bit-identical at a fixed seed and code on this host: A4000, bf16, TF32 off,
  `cudnn.benchmark = False`.
- The evidence is `bc_dart` s0 against the σ = 0.10 trial, two processes about 13 h apart.
  Model, EMA and AdamW tensors are identical at all six checkpoints. The training logs are
  identical at every point, and the `select` rows are identical in every column except
  `ckpt_sha256`.
- Training randomness is keyed by step. The saved global CPU RNG state is never consumed.
- No Phase 4 run resumed, so the P3-D1 §4 resume rule was never exercised.

*Correction 2, P4-D3.* "`bc_dart` seed 0 re-trained" and "trial runs … never core seeds" did not
separate the two.
- The re-training reproduced the σ = 0.10 trial bit for bit, so **`bc_dart` s0 is the model whose
  `select` score chose σ\*** (best of 3 σ × 6 checkpoints).
- It is kept: P3-D1 §4 forbids replacing or re-drawing a seed. Its frozen-list rows remain an
  unbiased measurement of that model. But the 5-seed arm estimate includes one seed chosen partly
  for looking good on `select`.
- The evidence suggests the bias is small: seeds 1–4 score 149/150/150/150 on `select`, so s0's
  150 is typical.
- **Binding on Phase 8 and the README:**
  - a footnote on every `bc_dart` row stating the above;
  - a labelled sensitivity row giving `bc_dart`'s IQM and its scored contrast (H1 DART secondary,
    against `bc`) on seeds 1–4 alone, reported beside the 5-seed result and never substituted for
    it;
  - s0's `select` number is never quoted.
- 0.807 GPU-h went on the duplicate s0.

*Measured student episode lengths* (P3-D1 §5, last line). These are the mean steps on `select`
of the selected checkpoints, over 5 seeds:

| Arm | Mean steps |
|---|---|
| `bc` | 65.3 (max 237) |
| `bc_state` | 61.5 |
| `bc_matched` | 62.7 |
| `bc_dart` | 61.6 |

- P2-D2's central evaluation projection used the teacher's 61.5. The measured BC-family lengths
  are within 6 % of it, so the 12.2 h central estimate stands for these arms.
- `select` is SS3–SS5. SS6 and timeouts on frozen lists can lengthen episodes toward P2-D2's
  upper bound.

*Budget.* Phase 4 measured **17.50 GPU-h** against 16.9 projected (+3.5 %).

| Arm | Measured | Projected |
|---|---|---|
| σ trials | 2.37 | 2.35 |
| `bc` | 3.96 | 3.92 |
| `bc_state` | 0.21 | 0.13 |
| `bc_matched` | 6.95 | 6.53 |
| `bc_dart` | 4.02 | 3.92 |

There is no cut and none is possible (P4-D2). Per-run values are in
`results/phase4/runs_summary.csv` (SHA-256 `bc8dcdeb…8c26f123`).

*Evidence.*
- `results/phase4/runs/<arm>/s<n>/{manifest,selection}.json` and
  `results/phase4/runs_summary.csv`, committed at `9074803`.
- The diagnostician's scripts and logs are session scratch, not committed. The figures above are
  quoted from its reports.

### P4-D5 — Two user-approved readings, evidence committed, corrections to P4-D4 (2026-10-29)
*Decision.* Both readings below were approved by the user on 2026-10-29, after the
`results-skeptic` pre-gate review (no BLOCKING findings). Neither relaxes a threshold.
1. **CAVEAT verdicts and the Gate 4 criterion** (interpretation). The plan says "TRUSTWORTHY on
   each, or the run is excluded with a reason", and P3-D1 §4 says no run is excluded alone. Under
   those rules, a CAVEAT verdict meets the gate only if all of the following hold:
   - (i) it names no defect, so it is not a §4 bug;
   - (ii) every caveat becomes a binding disclosure or reporting restriction;
   - (iii) the reviewer's evidence is committed.

   A caveat that implicates a code path is a §4 bug instead. That would mean fix and retrain,
   including every arm that shares the path; for `bc_matched` this includes `bc_dagger`. The six
   CAVEATs are `bc_matched` s0–s4 and `bc_dart` s0. Each is classified **not a bug**:
   - `bc_matched` (a)–(e) are consequences of the design or observed behaviour (P4-D4);
   - `bc_dart` s0's caveat is about provenance.
2. **`bc_dart` s0 is the σ = 0.10 tuning trial** (deviation from P3-D1 §4).
   - §4 says trials are "logged as tuning, not core seeds", and also "seed 0 included
     (re-trained)" and "no seed … replaced or re-drawn". Because training turned out to be
     deterministic (P4-D4 correction 1), these cannot all hold.
   - **s0 is kept**, and seeds 0–4 stay as frozen.
   - **Binding on Phase 8 and the README:**
     - a footnote on every `bc_dart` row;
     - a sensitivity row on seeds 1–4, labelled "post hoc: added after `select`, before any
       frozen-list read". It gives the IQM and the H1 DART secondary contrast, printed beside the
       5-seed result and never substituted for it.
   - **H4's "best end-to-end" rule** (P3-D1 §6) uses all five seeds as frozen. The seeds 1–4 value
     is printed beside it. This supersedes P4-D4's "s0's `select` number is never quoted"; that
     restriction now applies only to quoting it as an estimate of performance. On the current
     numbers, `bc_dart` ranks first either way: 149.8 on five seeds, 149.75 on seeds 1–4, against
     146.6 for `bc` and for `bc_matched`.
3. **P4-D1 item 3 is relabelled a deviation from the plan.** Held-out loss uses `train`
   `ppo_2000` 1950–1999 instead of the plan's "held-out `tune` demos". P3-D1 §2 takes
   precedence, so this is a deviation, not an open choice.
4. **P4-D1 item 5, stated.** The constant 0.999 EMA leaves 0.999^5000 ≈ 0.7 % of the random-init
   weights in every 5 k-step candidate. No `bc`, `bc_state` or `bc_dart` 5 k candidate was
   selected. `bc_matched`'s 5 k candidates are iteration-5 checkpoints that started from
   initialised weights, so this does not apply to them.

*Evidence committed* (closes skeptic G1, S2, S3, S4).
- **Diagnostics:** `results/phase4/diagnostics/`. It contains one JSON per run (verdict,
  reasons, every quoted metric), `summary.json` (`dbc31be9…b7a84bed`, with arm-level figures and the SHA-256
  of every script and input), the 19 scripts, and the logs they read. The scripts are kept as run, for audit. They read
  session-scratch paths, so they are not a reproduction target.
- **Teacher reference for copycat (S4):** the `ppo` s0 teacher's ∂a/∂last_action diagonal on the
  same states (`inputs/teacher_jac.json` `b45c0b8a…64403447`). It was computed by central finite differences
  (ε = 0.05, raw units), because P8's SB3 path is not differentiable here.

  | Model | Diagonal (held-out `ppo_2000`) |
  |---|---|
  | Teacher | x 0.074, y 0.097, z 0.176 (dart_0.10 states: 0.066 / 0.090 / 0.161) |
  | `bc_state` | x, y 0.09–0.11; z 0.25–0.29 |
  | `bc_dart` | 0.13–0.16 |
  | `bc` | **0.70–0.91** (about 4–10× the teacher) |
  | `bc_matched`, end of iteration 5 | 0.78–0.96 |

- **Tunnelling and detector disagreement (S2).** `selection.json` and `runs_summary.csv`
  (`c1c0af76…0262c029`) now carry `n_tunnelled` and `n_detectors_disagree`.
  - They were rebuilt from the stored rows by `scripts/select.py --resummarise`, with no flight.
    Every selection reproduced exactly, `sigma_star.json` is byte-identical, and σ* is unchanged.
  - Over all 118 candidates: 41 tunnelled, 6 disagreements. Over the 23 selected checkpoints:
    10 tunnelled and 1 disagreement.
  - Three flagged episodes are scored `success`, because P8 counts tunnelling without changing the
    outcome class:
    - tunnelled: σ = 0.05 trial, SS5 #16, and `bc_matched` s4, SS5 #16;
    - detectors disagree: `bc_matched` s0.
  - **P4-D4 (e) is amended:** `bc_matched` s1's 1.234 m/s off-pad touchdown (SS4 #35) also
    tunnelled, with 15.8 mm penetration.
  - Phase 8 prints `n_tunnelled` beside success (P8 rule).
- **Timing manifests (S3):** `results/phase4/timing/{dp,bc,bc_state}_manifest.json`, the P4-D2
  inputs.

*Corrections to P4-D4.* Figures are reproduced from `results/phase4/diagnostics/summary.json` and
`runs_summary.csv`.

| P4-D4 said | Correct value |
|---|---|
| `bc` ∂a/∂last_action diagonal 0.75–0.91 | **0.70–0.91** |
| `bc_matched` 10–15 % worse on DART states | **10–16 %** (+10.3 to +15.5 %) |
| BC-family lengths "within 6 %" of 61.5 | `bc` is **+6.2 %** (65.29 against P2-D2's exact 61.47); `bc_state` +0.1 %, `bc_matched` +2.0 %, `bc_dart` +0.3 %. The central evaluation estimate stands for `bc_state`, `bc_matched` and `bc_dart`; `bc` runs about 6 % over it |
| Budget: `bc` 3.96, `bc_matched` 6.95 GPU-h | **3.95** and **6.94** |
| Phase 4 total +3.5 % | **+3.82 %** (17.498 against 16.855) |
| `runs_summary.csv` SHA-256 | superseded by `c1c0af76…0262c029` above, which adds the two columns; every earlier field is byte-identical |

- **Winner's curse** (selected minus the mean over each seed's candidates, averaged over seeds):

  | Arm | Episodes |
  |---|---|
  | `bc` | +2.33 |
  | `bc_state` | +0.7 |
  | `bc_matched` | +1.2 |
  | `bc_dart` | +1.07 |
  | σ trials | +1.28 |

- **Open-loop against closed-loop**, reported with H1's DART secondary. `bc_dart`'s open-loop MSE
  on clean teacher states is 31–62 % above `bc`'s, yet it is better closed-loop on `select`.

*Notes on the record.*
- The 4 k-step smoke behind P4-D1's "3 % short" has no committed artifact; its file was overwritten.
- `bc_matched` manifests record `init_from` and its SHA-256, but not the normaliser values. Those
  are inherited bit-identically from `bc` s<n>'s manifest (verified by the diagnostician).
- No teacher row on `select` is committed, so "at the `select` ceiling" is relative to 150/150,
  not to the teacher.
- `bc_matched`'s lowest-step tie-break uses run-local steps, which preserves the order.

### Gate 4 passed (2026-10-29)

| Criterion (plan Phase 4 exit gate; P3-D1 §4, §5; P4-D5) | Evidence | Verdict |
|---|---|---|
| `make lint` / `make test` | ruff and format clean; `mypy --strict` 60 files; **643 passed** in 264.5 s, all markers, at `b4ae711` | PASS |
| Smoke: overfit 8 episodes to MSE < 1e-4, land ≥ 7/8 | `results/phase4/smoke_bc.json`: MSE 8.9e-7, 8/8. `smoke_bc_state.json`: MSE 5.7e-7, 8/8. Replay 4/4 identical against a fresh teacher rollout. Code at `63750c1` | PASS |
| Every run present with a manifest (config SHA, data SHA, checkpoint SHA, wall clock) | 23 runs in `results/phase4/runs/**/manifest.json`. Config and data SHA-256 match P3-D1, every tree was clean, and there were no resumes. Every checkpoint SHA-256 was recomputed by the diagnostician and by `results-skeptic` | PASS |
| Diagnostician TRUSTWORTHY on each, or excluded with a reason | 17 TRUSTWORTHY and 6 CAVEAT (`bc_matched` s0–s4, `bc_dart` s0). Each CAVEAT is classified "not a bug" and carries binding disclosures. They meet the criterion under the user-approved reading in P4-D5 item 1 (2026-10-29). No run is excluded. Evidence: `results/phase4/diagnostics/` | PASS (under P4-D5 item 1) |
| Selection on `select` only | `select.parquet` `07ecf049…422b`, CPU fp32 single-sample path, EMA weights. `results-skeptic` recomputed all 118 candidates and found no student row on a frozen list | PASS |
| Only tuning = DART σ, 3 trials, logged | P4-D3, σ\* = 0.10. `bc_dart` s0 = the trial, a deviation approved by the user (P4-D5 item 2) | PASS |
| Budget rule before any `select` number; GPU-hours per run in the budget entry | P4-D2 (no cut) was committed at 11:03 on 2026-10-28. The first `select` row was written at 13:37. Per-run GPU-hours are in `results/phase4/runs_summary.csv`; Phase 4 totals 17.50 GPU-h (P4-D4, as corrected in P4-D5) | PASS |

Reviewers:
- `privileged-leak-auditor` (pre-run): no LEAK, 1 RISK, which was fixed before the first run (P4-D1).
- `training-run-diagnostician`: all 23 runs (P4-D4, P4-D5).
- `results-skeptic` round 1: no BLOCKING; 2 gate-procedural items, 4 SHOULD-FIX, 7 NOTEs.
- `results-skeptic` round 2: no BLOCKING. All four of its CAVEAT conditions met, and every round-1 item closed, with the two wording fixes below.

**Corrections recorded here** (entries are not edited):
- P4-D5 says "over the 23 selected checkpoints: 10 tunnelled". The committed figure is **9**:
  `bc` s0, s3 and s4 (1 each), the σ = 0.05 trial (1), and `bc_matched` s1 (2), s3 (2) and s4 (1).
  This is the sum of `n_tunnelled` in `runs_summary.csv`. The disagreement count of 1 is correct.
- P4-D5 says "`bc_dart`'s open-loop MSE on clean teacher states is 31–62 % above `bc`'s". The
  committed ratio is **+131–162 %** (2.3–2.6×; `summary.json`,
  `openloop_mse_comparisons_vs_bc_final_ema`). On DART states the same ratio is about −95 %.
- In P4-D5's teacher-reference table, `bc_dart` on held-out states is **0.125–0.153**. The 0.16
  is its upper value on DART states.
- `bc` relative to the teacher is **about 4–12× per axis**, not "4–10×".
- The 15.8 mm penetration of `bc_matched` s1's dive is read from the uncommitted select row
  (`artifacts/runs/bc_matched/s1/select/s5_step010000_rows.csv`, `max_penetration_m` −0.01576).
  No committed file holds it.

Incidents:
- `bc_dart` s0 reproduced the σ = 0.10 trial bit for bit (P4-D4 correction 1, P4-D5 item 2).
- No threshold was relaxed.
- The reproduction gate was not re-run. Phase 4 changed nothing under `ila/render/`,
  `ila/eval/runner.py` or the adapter.

## Phase 5

*Before you start (from Phase 4).*
- **Shared code path.** `bc_dagger` reuses `bc_matched`'s iteration driver in `ila/train/loop.py`:
  warm start, normaliser inheritance, stage data. A bug found there is a P3-D1 §4 bug for both
  arms (P4-D5 item 1).
- **Iteration 0.** It is `bc/s<n>/checkpoints/s0_step030000.pt`; the `bc_matched` manifests record
  its SHA-256. Rollouts fly the EMA at the last step of the previous iteration, on the CPU fp32
  single-sample path.
- **Determinism.** GPU training was observed bit-identical at a fixed seed and code (P4-D4
  correction 1). A DAgger re-run with unchanged code will reproduce, not resample.
- **Copycat.** `bc`'s ∂a/∂last_action is 0.70–0.91 against the teacher's 0.07–0.18 (P4-D5). Report
  it per DAgger iteration beside teacher agreement on the student's own states. DAgger rollouts
  break the `last_action[t] = label[t−1]` identity that demos have.
- **Report** `n_tunnelled` and `n_detectors_disagree` with every `select` result.
- **Budget.** `bc_dagger` is projected at 6.5 GPU-h plus CPU rollouts (P4-D2). `bc_matched` ran
  6.4 % over its projection.
- **Watch items for Phase 8.** `bc_matched` s1's tunnelled 1.234 m/s off-pad dive. `bc`'s
  slow-landing tail (max 237 steps on `select`).

### P5-D1 — DAgger implementation choices P3-D1 leaves open (2026-10-29)
*Decision.* The settings below are not fixed by `bc_dagger.yaml` (`dbc7c0d2…9da7`),
`demos.yaml#dagger` (`1fd413e2…3f11`) or P3-D1. They are implementation choices within the frozen
protocol, not deviations. They are recorded before any `bc_dagger` run, and every run manifest
repeats them. Neither frozen file was edited.
1. **Rollout student.** Iteration 1 flies the EMA of `bc/s<n>/checkpoints/s0_step030000.pt`, and
   iteration i > 1 flies the EMA of this run's `s<i−1>_step010000.pt`. The checkpoint's SHA-256 must
   equal the run manifest's entry and is written into every rollout episode
   (`student_ckpt_sha256`). Path: `ila.train.select.checkpoint_policy`, CPU fp32, one torch thread
   per worker, through `runner.StudentWindow` (frames + the 13-d allowlisted proprio only).
2. **Teacher shadowing.** `ppo` s0 on the SB3 label path is `reset` every episode and queried on
   every visited observation, including steps where the student's action is executed. The label is
   always the teacher's action at `teacher_obs[t]`.
3. **β coin.** Entropy `[run_seed, H("dagger_beta"), iteration, episode_seed]` (SHA-256 label
   entropy, as for DART). Exactly one `rng.random()` per step whatever β is; the teacher's action
   is executed when u < β. So the coin sequence does not depend on β, on the student or on the
   worker count. The executed action is `collect.executable(·)` of the chosen action (float32
   before `env.step`).
4. **Storage.** `<data_root>/dagger/bc_dagger/s<seed>/iter<i>.h5` uses the unchanged demo-store
   schema, with per-episode `beta`, `dagger_iteration` and `student_ckpt_sha256`. Beside it:
   - `iter<i>.sidecar.npz`: per step, the student's proposed action, the coin draw u and whether
     the teacher was executed. It is diagnostics only and never a training input.
   - `iter<i>.manifest.json` (written last), `iter<i>.audit.json`, `aggregate_iter<i>.json`,
     `iter<i>.episodes.csv`.
   Store and sidecar are written to `*.tmp` and renamed.
5. **Resume.** An existing iteration is reused only if its manifest's student SHA, purpose and
   limit match. A mismatch, or a store without a manifest, raises; nothing is overwritten.
   Rollouts are deterministic: the same checkpoint and code give byte-identical stores for 1 or 2
   workers (tested).
6. **Audit gate.** Before stage i trains, `ila.data.dagger.audit_iteration` must return clean:
   - re-label of 1 000 random visited states from `teacher_obs`, identical;
   - replay of 20 stored `action_exec` states, byte-identical frames and observations;
   - attributes: `train` pool only, indices = `dagger_indices(i)`, no `tune`/`select`/frozen-list
     row, no SS6, no 90°, β and iteration right, DART σ = 0;
   - `last_action[t] = action_exec[t−1]`;
   - the coin re-drawn bit-identically, and the executed-action choice recomputed against the store.
   If it is not clean, the stage is not trained.
7. **Aggregate and counts.** Stage i trains on `ppo_2000` 0–999 (success-filtered, as `bc`) plus
   rollout iterations 1..i (unfiltered), frames sampled uniformly. The labelled-episode count is
   asserted at 1 000 + 200 i and at 2 000 = `bc_matched` after iteration 5. The normaliser is
   inherited from `bc`, not refitted, as for `bc_matched`.
8. **Frozen-config check** now also covers `configs/data/demos.yaml`, because `bc_dagger.yaml`
   reads its `dagger` block.
9. **Held-out loss.** `ppo_2000` 1950–1999 is never in `bc_dagger`'s training data (overlap 0). But
   iteration 5 flies the same 50 starting conditions with the student, so their t = 0 frames match.
   Logged as `heldout_shared_starting_conditions`; held-out loss after iteration 5 is read with that
   caveat, as for `bc_matched`.
10. **Per-iteration report** (`make dagger-report` → `results/phase5/dagger_iterations.{csv,json}`),
    descriptive only. Selection stays iteration 5's 5 k and 10 k checkpoints, EMA (P3-D1 §3).
    - *Rollout success* on `train`. For iterations 1–3 (β > 0) it measures the β-mixture policy, not
      the student (column `policy_flown`).
    - *Teacher agreement* on the student's own proposals (sidecar): mean ‖a_student − label‖₂, and the
      fraction of steps with ‖Δ‖∞ ≤ 0.1 normalised units (0.15 m/s model scale, 0.75 m/s full
      scale). Both are over all steps and over steps where the student's action was executed.
    - *Copycat:* the mean autograd ∂a/∂last_action diagonal (EMA, eval, fp32 CPU) for three cases:
      the rollout student on its own states; the end-of-iteration EMA on P4-D5's held-out states;
      the same EMA on the iteration's own states. On `bc` s0's 30 k EMA, autograd gives
      0.909 / 0.785 / 0.746 against P4-D5's finite differences 0.896 / 0.782 / 0.755. The method is
      stated with every number.
    - *`select` success* of each iteration's end EMA (CPU fp32 path), with outcomes, p95 closing
      speed, `n_tunnelled` and `n_detectors_disagree`. The iteration-0 row is `bc`'s 30 k EMA.
11. **Wall clock.** The driver's `gpu_hours` for `bc_dagger` is wall clock and includes CPU rollouts
    and audits, which P4-D2 did not count. Per-stage `timing_s` is recorded so they can be
    separated.
12. **Layering.** `ila.data.dagger` imports `ila.train.select` (student loading, outcome order).
    `ila.train` imports `ila.data.dagger` lazily, and there is no cycle.

*Reason.* Each item is something a reviewer would otherwise reverse-engineer from code. Items 6–7
are the Phase 5 audits the plan requires. Item 9 extends P4-D1 item 3's held-out caveat to the
on-policy arm.

*Evidence.* `src/ila/data/dagger.py`, `src/ila/train/dagger_stage.py`,
`src/ila/train/diagnostics.py`, `src/ila/train/dagger_report.py`; `tests/test_data_dagger.py`,
`tests/test_train_dagger.py`.

*Consequence.* `bc_dagger` runs through `make dagger-bg SEED=<n>`. Its pre-flight is
`make train-smoke ARM=bc_dagger` → `results/phase5/smoke_bc_dagger.json`.
