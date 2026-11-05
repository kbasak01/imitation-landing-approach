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

### P5-D2 — `bc_dagger` runs, per-iteration findings, review corrections (2026-10-29)
*Decision.* All five `bc_dagger` runs (s0–s4) are kept, and none is excluded. All five are
TRUSTWORTHY (`training-run-diagnostician`). The selected checkpoints are final for Phase 8.
Everything below is descriptive and was measured in simulation (PyBullet, TinyRenderer frames).
`bc_dagger` is distilled from P8 `ppo` seed 0. Nothing here is an H1 result: H1 is scored only on
`id` SS5, in Phase 8.

*Runs* (`results/phase5/runs_summary.csv` `115ebc92…723875de`; `results/phase5/runs/bc_dagger/s<n>/`).
- Commit `576447c`, clean tree, P8 `7237b70`, one invocation each, no resumes. Config SHA-256s
  equal P3-D1's (`bc_dagger.yaml`, `bc.yaml`, `demos.yaml`).
- Iteration 1 flew `bc/s<n>` `s0_step030000.pt`, and iteration i flew `s<i−1>_step010000.pt`.
  The SHA-256s agree across the run manifest, `dagger/iter<i>.json`, the rollout manifest and
  every episode attribute (`results/phase5/diagnostics/bc_dagger/s<n>.json`).
- **Audits:** 25 of 25 seed × iteration audits are clean (`runs/bc_dagger/s<n>/rollouts/iter<i>.audit.json`).
  - Each re-labels 1 000 states, all identical, max-abs 0.0.
  - Each replays 20 states, byte-identical.
  - 0 attribute violations, and 0 coin, executed-action or `last_action` mismatches.
- **Counts.** Labelled episodes are 1 200 / 1 400 / 1 600 / 1 800 / 2 000 in every seed, so 2 000
  = `bc_matched`. Labelled frames at iteration 5 are 123 013–123 359 against `bc_matched`'s
  122 873. Rollouts add 61 495–61 841 frames against 61 355 for teacher demos 1000–1999. P3-D1's
  361-step concern did not materialise: the longest rollout is 133 steps and there are 0 timeouts
  (`diagnostics/summary.json` `arm.labelled_frames`, `arm.rollout_steps_max`, `arm.rollout_timeouts_total`).

*Selection* (`select` only, iteration 5's 5 k and 10 k checkpoints, EMA, CPU fp32 path;
`runs_summary.csv`, `diagnostics/summary.json` `arm.selection`, `arm.decided_by`).

| Seed | Selected | `select` | Failures | p95 closing speed (m/s, model scale) | `n_tunnelled` | `n_detectors_disagree` | Decided by |
|---|---|---|---|---|---|---|---|
| s0 | s5 5 k | 149/150 | 1 hard landing | 0.373 | 1 | 0 | success |
| s1 | s5 10 k | 149/150 | 1 bounce | 0.411 | 0 | 0 | p95 tie-break |
| s2 | s5 5 k | 148/150 | 1 hard landing, 1 bounce | 0.366 | 0 | 0 | success |
| s3 | s5 10 k | 149/150 | 1 bounce | 0.347 | 0 | 0 | p95 tie-break |
| s4 | s5 5 k | 150/150 | none | 0.381 | 0 | 0 | success |

- 0 crashes, 0 off-pad landings and 0 timeouts in the selected rows. s0's one failure (SS5 #16)
  is tunnelled. The same episode tunnelled for `bc_matched` s4 and for the σ = 0.05 trial.
- The two candidates of a seed differ by 0–2 episodes. Mean over all candidates is 148.6, so the
  winner's curse is +0.4.
- **Unpaired reading, on the selection list:**
  - selected mean 149.0 (`bc_dagger`) against 146.6 (`bc_matched`);
  - all-candidate mean 148.6 against 145.4 (winner's curse +0.4 against +1.2);
  - pooled failures over the selected checkpoints: 3 bounces and 2 hard landings, against 6
    bounces, 6 hard landings and 5 off-pad landings (`arm.selection`, `arm.bc_matched_selection`).
  This is not H1 evidence: `select` chose these checkpoints, sits near the ceiling, and is
  decided by a few episodes.

*Per-iteration results* (`results/phase5/dagger_iterations.csv` `da37f68a…52441693`, means over
seeds from `diagnostics/summary.json` `arm.per_iteration_*`). Row i's rollout columns are flown by
the EMA at the end of iteration i − 1, at β_i. Row i's `select` columns are the EMA at the end of
iteration i. The `select` numbers for iterations 1–4 were flown after selection had finished
(17:20–17:44 against 17:10–17:18). They are findings, not candidates.

| Iteration | β | `train` rollout success, mean of 200 | Rollout failures, 5 seeds (crash / off-pad / hard / bounce / timeout) | `select`, mean of 150 |
|---|---|---|---|---|
| 0 (`bc` 30 k) | – | – | – | 146.0 |
| 1 | 0.5 | 200.0 | 0 / 0 / 0 / 0 / 0 | 145.8 |
| 2 | 0.25 | 199.0 | 0 / 0 / 4 / 1 / 0 | 145.2 |
| 3 | 0.125 | 198.4 | 0 / 1 / 5 / 2 / 0 | 148.4 |
| 4 | 0 | 197.0 | 0 / 1 / 12 / 2 / 0 | 148.4 |
| 5 | 0 | 199.0 | 0 / 0 / 4 / 1 / 0 | 148.2 |

- **Rollout success.** For iterations 1–3 it measures the β-mixture, not the student. The
  iterations also fly different start slices. No interval was computed, so there is no
  "rises on `train`, flat on `select`" claim either way (plan Phase 5 item 3). Both stay near the
  ceiling. A transient dip at iteration 2: s1's end-of-iteration-2 EMA scores 141/150 on `select`.
- **Teacher agreement.** The fraction of steps with ‖a_student − label‖∞ ≤ 0.1 normalised
  (0.15 m/s model scale, 0.75 m/s full scale, λ = 1/25). Split by who executed the previous step
  (`results/phase5/agreement_by_previous_executor.csv` `84203043…432c7b3f`, script
  `results/phase5/scripts/agreement_split.py`):
  - after a teacher-executed step: 0.981–0.996 at every β > 0;
  - after a student-executed step: 0.944–0.953 (iteration 1), 0.897–0.906 (iteration 2),
    0.841–0.866 (iteration 3);
  - all steps: 0.968–0.973, 0.920–0.928, 0.859–0.881, 0.744–0.778 and 0.774–0.804 for
    iterations 1–5.
  - **Correction to the reading given in chat** (results-skeptic S1): the fall over iterations 1–4
    is not shown to be a β-driven state shift with an unchanged student. At β > 0 agreement is
    inflated by the student copying a teacher-chosen `last_action`. Iterations 1–3 are not
    comparable with 4–5.
  - Iterations 4 and 5 are not like-for-like either. They fly different start slices, and
    step-0 agreement moves the other way in all five seeds (0.89–0.915 → 0.84–0.885).
  - **Common-state check** (diagnostician; `arm.fixed_set_sibling_l2`): open-loop L2 error on one
    fixed set of β = 0 states from a sibling seed's iteration-5 rollouts. Neither run of a seed
    trained on that set. `bc_dagger` falls 5.6–7.5 % from iteration 0 to 4, and `bc_matched` falls
    0.8–1.2 % from iteration 0 to 5. Iteration 5 of `bc_dagger` is excluded because it has seen
    those starting conditions.
- **Copycat.** ∂a/∂last_action diagonal on held-out `ppo_2000` states, finite differences
  (`arm.copycat_J_diag_heldout_FD`; the CSV's autograd values agree to ≤ 0.011):
  - `bc_dagger`, x / y / z: 0.88–0.91 / 0.75–0.78 / 0.70–0.78 at iteration 0, falling to
    0.76–0.78 / 0.67–0.69 / 0.46–0.51 at iteration 5;
  - `bc_matched` at iteration 5 rises to 0.95–0.96 / 0.85–0.88 / 0.78–0.89;
  - the teacher is 0.074 / 0.097 / 0.176 (P4-D5), so at iteration 5 `bc_dagger` is 10.3–10.6×,
    6.9–7.1× and 2.6–2.9× the teacher per axis;
  - closed loop at β = 0, copycat steps are 0.02–0.10 % of moving steps, and MSE against label t
    is 0.009–0.012 against 0.076–0.079 against label t − 1, so there is no one-step lag
    (`arm.rollout_copycat_frac_range`, `arm.rollout_lag_mse_t_vs_tminus1_beta0`).
  - **For H1's mechanism (Phase 8):** on-policy rollouts break the `last_action[t] = label[t−1]`
    shortcut that demos carry. Part of any H1 effect may be shortcut removal rather than
    state-distribution correction.
- **Open-loop and closed-loop move in opposite directions.** On teacher states (`ppo_2000`
  1950–1999), `bc_dagger`'s fp32 open-loop MSE rises 46–60 % from iteration 0 to 5
  (`arm.heldout_mse_it0_to_it5_rel_change_bc_dagger`), starting at stage 2–3. Over the same
  iterations, error on student states falls and `select` success rises. This is the same pattern
  as `bc_dart` in Phase 4. Held-out loss must not be used to compare `bc_dagger` with `bc_matched`.

*Disclosures and corrections.*
1. **Held-out overlap, wider than P5-D1 item 9 says** (leak auditor NOTE 1).
   - Between `ppo_2000` 1950–1999 and iteration 5's rollouts, the first ~5 frames (mean 5.2, max 8)
     are pixel-identical, not only t = 0.
   - That is about 260 of the slice's 3 067 frames per seed.
   - The caveat applies to iteration 5's held-out loss and to its `copycat_heldout_*` columns.
   - It is not a leak: all of it is `train` pool, and nothing is decided on it.
2. **Report CSV columns** (diagnostician item 5). The first `dagger_iterations.csv` (`4d0f832e…aae6`)
   repeated `rollout_success` and `select_success` as outcome-class columns.
   - `ila.train.dagger_report` now omits `success` from the class breakdown.
   - The file was regenerated with `--no-fly`. Every value is unchanged; only the two duplicate
     columns (equal to the counts in every row) were removed, and `select_rows_source` now reads
     `reused:` (cached rows) instead of `flown`.
   - `diagnostics/summary.json` hashes the earlier file.
3. **Budget** (results-skeptic S5).
   - The figure for the budget tally is wall clock: **7.11 GPU-h** for the five runs,
     **+0.58 h (+8.9 %) over P4-D2's 6.53**.
   - Net of CPU rollouts, audits and aggregate loading (563–579 s per run) it is 6.32 h; net of
     rollouts and audits only, 6.45 h. `bc_matched` took 6.94 h.
   - Neither figure includes about 5 min of setup per run or the `select`/report flights.
   - Running total: 17.50 (Phase 4) + 7.11 = 24.61 GPU-h, against 23.48 projected through Phase 5 (`results/phase4/budget_projection.csv`: the Phase 4 lines plus `bc_dagger`). P4-D2's no-cut rule stands.
   - Sources: `arm.budget`; `runs_summary.csv` `cpu_in_wall_s`, `gpu_hours_net_of_cpu`.
4. **Smoke provenance.** `results/phase5/smoke_bc_dagger.json` was run on a dirty tree based on
   `dce79fe`. Its code files are identical to `576447c`; only `docs/protocol.md` changed before
   that commit.
5. **Different commits for the matched pair.** `bc_matched` ran at `6d8517f` and `bc_dagger` at
   `576447c`. The shared-path changes leave `bc_matched` unchanged: `first_n` still resolves to
   `first_added` = 1 000, and `_iteration_data` is `load_stage_data`. Iteration-0 SHA-256s, stages,
   candidate rule and the proprio normaliser are bit-identical across `bc`, `bc_matched` and
   `bc_dagger` (results-skeptic; `diagnostics/bc_dagger/s<n>.json`).
6. **Pre-flight split checks** (leak auditor NOTE 2). The check before flying tests SS6/90° only.
   The tune/select/frozen-list disjointness is checked by the per-iteration audit, after flying,
   and by `read_split`'s MANIFEST hash on `demos_0-1999.parquet`. Accepted as is.

*Reviewers.* `privileged-leak-auditor`: no LEAK, no RISK, 3 NOTEs (items 1 and 6 above; per-iteration
`select` wording). `training-run-diagnostician`: 5 TRUSTWORTHY (`results/phase5/diagnostics/`,
`summary.json` `f3a32c04…f00fbf49`). `results-skeptic`: no BLOCKING. Its SHOULD-FIX items S1–S6 are
all addressed in this entry; S6 (commit the evidence) is addressed in this commit.

*Consequence.* Phase 8 flies the selected `bc_dagger` checkpoints. H1 is scored as frozen in P3-D1,
and labelled frames (123 013–123 359 against 122 873) are printed beside it.

### Gate 5 passed (2026-10-29)

| Criterion (plan Phase 5 exit gate) | Evidence | Verdict |
|---|---|---|
| `make lint` / `make test` | `make lint`: ruff and format clean, `mypy --strict` on 63 files. Full suite with all markers: **680 passed** in 562.9 s, on the `576447c` tree with the evidence present. After the report fix: `tests/test_train_dagger.py` 21 passed, lint clean | PASS |
| 5 seeds × 5 iterations complete | `runs_summary.csv`: 5 runs `done`, 50 000 steps each; `runs/bc_dagger/s<n>/dagger/iter1–5.json` and `rollouts/iter1–5.manifest.json` for every seed | PASS |
| Audits clean (label audit per iteration; counts = `bc_matched`) | 25 of 25 `iter<i>.audit.json` `clean: true`, each with 1 000 of 1 000 re-labelled states identical. Labelled episodes 1 200…2 000 per seed = `bc_matched`'s 2 000. Leak auditor's own re-label: 6 180 steps, max-abs 0.0 | PASS |
| Final checkpoints selected on `select` only | Candidates: iteration 5 only, rule by config (`select.py`). 30 row files hold exactly the 150 `select.parquet` episodes. No `bc_dagger` row on any frozen list. Selection finished before any per-iteration flight | PASS |
| Per-iteration report: `train` rollout success, teacher agreement, `select` success | `results/phase5/dagger_iterations.csv`, `agreement_by_previous_executor.csv`; P5-D2 | PASS |

Reviewers: `privileged-leak-auditor` (no LEAK or RISK), `training-run-diagnostician` (5 TRUSTWORTHY),
`results-skeptic` (no BLOCKING; S1–S6 addressed in P5-D2). No threshold was relaxed. The
reproduction gate was not re-run, because Phase 5 changed nothing under `ila/render/`,
`ila/eval/runner.py` or the adapter.

## Phase 6

*Before you start (from Phase 5).*
- **Budget.** Training so far is 24.61 GPU-h of wall clock against 23.48 projected through Phase 5
  (P5-D2 item 3). `dp` is P4-D2's largest line; compare its first run with the projection
  before queuing the other seeds.
- **Held-out loss is not comparable across arms** after iteration 5 of either iterated arm
  (P5-D2), and open-loop loss on teacher states moved against closed-loop success for `bc_dagger`
  and `bc_dart`. Read `dp`'s held-out curves as diagnostics only.
- **`select` is decided by a few episodes** (SS5 #16 and #49 recur across arms). Expect `dp`'s
  selection to be decided by the p95 tie-break and say so.

### P6-D1 — Diffusion Policy implementation choices P3-D1 leaves open (2026-10-30)
*Status: frozen 2026-10-30. Written and frozen before the `dp` smoke result, any `dp` training run
or any `dp` number on `select` was read. Item 1 defines the ensembling that H3's curve uses, so it
is not revised after `select` sweep numbers exist (leak audit NOTE 1).*

*Decision.* The settings below are not fixed by `dp.yaml` (`c8f90c35…6913`, unchanged) or
P3-D1. They are implementation choices within the frozen protocol, not deviations.
1. **Temporal ensembling: which predictions are averaged.** P3-D1 §3 crosses `Ta` with
   ensembling and fixes the weights (`w_i = exp(−m i)`, m = 0.01, i = 0 the oldest), but not
   which past chunks count. Reading adopted: with ensembling the policy replans at **every**
   step; step t averages the chunks planned at s ∈ [max(0, t − Ta + 1), t], the chunk planned at
   s read at index To − 1 + (t − s) (`chunking.ensemble_sources`, oldest first). So `Ta` is the
   ensemble window, and each chunk contributes to exactly the steps it would have executed
   without ensembling.
   - At the start of an episode there are t + 1 < Ta predictions. The weights are renormalised
     over those.
   - `Ta` = 1 with ensembling is the same policy as `Ta` = 1 without: one source, weight 1.0.
     The actions are bit-identical (`tests/test_policies.py`).
   - Noise for the chunk planned at t is keyed (run_seed, episode_seed, t), as without
     ensembling. The weighted sum is taken in float64 and returned as float32.
   - Code: `DPStudent(temporal_ensembling=True)`, threaded through `registry.build_dummy` and
     `load_student`. The default stays off, and the default path is bit-identical to before
     (item 5).
2. **`dp` smoke** (`make train-smoke ARM=dp` → `results/phase6/smoke_dp.json`). It overfits the
   first 8 `ppo_2000` `train` episodes (success-filtered, as `dp.yaml` says) for **10 000
   steps**. Every other hyper-parameter is `dp.yaml`'s: batch 128, bf16, lr 1e-4, cosine with
   500 warm-up steps, EMA power 0.75. The checkpoint cadence equals the step count, and the
   purpose is `smoke`, so it can never be selected. It passes iff all three hold:
   - **0. Replay.** Stored steps 0–3 of the first episode are byte-identical to a fresh teacher
     rollout (as in Phase 4).
   - **1. Overfit.** Sampled executed-chunk MSE < **1e-3** (normalised action units squared;
     RMS 0.032 = 0.047 m/s model scale per axis).
     - At every frame t of the 8 episodes, one chunk is sampled from the EMA weights. Settings:
       CPU, fp32, eval mode (centre crop), DDIM K = 10 (trailing), the flight's noise key
       (0, episode_seed, t).
     - The executed indices at `Ta` = 8 (To − 1 … To + 6, steps t … t + 7) are compared with
       `label` at those steps, edge-repeated past the episode end.
     - The mean is taken over frames, steps and axes. The action at t alone is reported beside
       it.
   - **2. Closed loop.** ≥ 7/8 `success` flying the saved `ema_last.pt` at `Ta` = 8, K = 10,
     no ensembling, on the CPU fp32 single-sample path (`select.fly_checkpoint`, the selection
     path). Outcome classes are printed.
3. **Inference sweeps on `select`** (plan Phase 6 item 4; `make dp-sweeps` →
   `results/phase6/sweeps/<arm>/s<n>/<cell>_rows.csv` and `results/dp_sweeps.json`).
   - **Checkpoint.** For each run, the checkpoint P3-D1 §4 selected at `Ta` = 8, K = 10, read
     from `selection.json`. Its SHA-256 is re-checked. Nothing is re-selected.
   - **Path.** The selection path: EMA weights, CPU fp32 single-sample, render `full`, 150
     `select` episodes.
   - **Cells.** Taken from `dp.yaml`'s `inference_sweeps`:
     - `Ta` ∈ {1, 2, 4, 8, 12} × ensembling {off, on}, at K = 10;
     - K ∈ {1, 2, 5, 10, 25, 100}, at `Ta` = 8, off.
     That is 15 distinct settings. `Ta` = 1 on is recorded as an alias of `Ta` = 1 off (item 1),
     so 14 are flown.
   - **K = 1 and 2** are labelled `schedule_limited` (P3-D1 §3).
   - **Per cell:** success, outcome classes, p95 closing speed, mean steps, `n_tunnelled`,
     `n_detectors_disagree`. Rows carry the cell in their `Ta`/`K`/`ensemble` columns.
   - **Determinism check.** The `Ta` = 8 / K = 10 / off cell is flown first. It must equal the
     stored selection rows text for text in every column except those three; otherwise the job
     stops.
   - **Resumable per cell**, as P5-D1's iterations: a cell is reused only if its rows file
     holds the 150 `select` episodes in order, flown by this checkpoint at this setting.
   - **Recorded in the summary:** config SHA-256, checkpoint SHA-256, the training commit and
     the sweep's commit.
   - **Scope.** These are not P3-D1 §3's sweeps. Those fly the same cells on `id` SS4–SS6
     (`Ta`) and `id` SS5 (K) in Phase 8 (`configs/eval/matrix.yaml` `sweeps`), and H3 is scored
     there. The `select` sweep is reported only. It feeds no choice, and no frozen list is
     read.
4. **Diagnostics** (`make dp-diagnostics` → `results/phase6/diagnostics/<arm>/s<n>.json`;
   definitions in `ila.train.dp_diagnostics`). All are measured on the selected checkpoint's EMA
   weights, CPU fp32, eval mode, at `Ta` = 8, K = 10. Actions are normalised (× 1.5 m/s model
   scale), and one control step is 1/30 s model scale.
   - **Spread.** For each state, 16 chunks are sampled. Sample j of state i is keyed
     (run_seed, −1, 16 i + j). The episode slot is negative, so it never collides with a flight
     key. Two figures are reported:
     - the per-element standard deviation (ddof 1), averaged over the executed slice (8 steps ×
       3 axes);
     - the same at index To − 1 alone.
     The teacher is deterministic per state, so all spread is the student's own sampling.
   - **Clip fraction.** The fraction of sampled elements with |a| ≥ 1 − 1e-6. With
     `clip_sample` and `set_alpha_to_one`, the last DDIM step returns the clipped x̂₀, so these
     elements are exactly ±1. Reported beside it: the labels' fraction (held-out set) and the
     executed actions' fraction (closed loop).
   - **Chunk-boundary vs within-chunk jerk** of the executed actions, replanning at
     `chunking.is_replan_step(t, 8)`:
     - **First difference** |a_t − a_{t−1}| (× 45 m/s² model scale per unit). It is a boundary
       difference when t is a replanning step.
     - **Second difference** |a_t − 2a_{t−1} + a_{t−2}| (× 1 350 m/s³ model scale per unit). It
       is a boundary difference when t or t − 1 is a replanning step.
     - Reported for each class: mean, median, p95, n, and the ratio of the means.
   - **States.**
     - **P4-D5's held-out set** (`ppo_2000` 1950–1999, every second episode). Spread and clip
       are measured on 512 frames drawn by `default_rng(0)`. Jerk is measured **open loop**:
       each episode's recorded windows are fed in order, with noise keyed
       (run_seed, −2 − e, t). The labels' differences at the same steps are the reference.
     - **`select`, closed loop.** The selected checkpoint is re-flown through the runner with
       an actor that records executed actions and the replan-step inputs. Two checks must
       hold: its rows equal the stored selection rows, and its per-episode mean first
       difference equals P8's `action_jerk_mean` column. Jerk is measured on the executed
       actions; spread and clip on up to 512 recorded replan-step inputs (`default_rng(0)`).
     - The stored rows have no per-step actions, which is why `select` is re-flown.
   - The denoising loss by k-bucket is already logged by the training loop (held-out, raw and
     EMA).
5. **Sampler refactor, bit-identity kept.**
   - The DDIM loop is now the module function `ddim_sample(eps_fn, noise, scheduler)`, with the
     final clip to [−1, 1]. The schedulers are built by `ddim_scheduler(cfg, K)` and
     `ddpm_scheduler(cfg)`. `DiffusionPolicyNet.sample` calls these.
   - `ddpm_sample` is DDPM ancestral sampling with every step's noise an explicit input. It is
     tested equal to diffusers' `DDPMScheduler.step` loop given the same draws. It is used only
     by the toy test.
   - The refactored sampler was compared with the pre-refactor `policy.py` (git HEAD
     `b891e05`) on a dummy `dp`: chunks bit-identical at K ∈ {1, 2, 10, 100}, and a 17-step
     episode of actions bit-identical. All existing `tests/test_policies.py` tests pass
     unchanged.
6. **Toy test** (`tests/test_dp_toy.py`, CPU, about 11 s).
   - **Setup.** A 1-D mixture ½ N(−0.6, 0.08²) + ½ N(0.6, 0.08²) under a constant
     observation. A 3 × 128 Mish MLP ε-net is trained with the schedulers built from `dp.yaml`
     through `DPConfig`.
   - **(a)** DDIM at K = 100 matches DDPM ancestral sampling. Mode fractions agree within
     0.06, mode means within 0.05, and mode standard deviations within 0.03.
   - **(b)** Both modes are recovered at K = 10 and 100: each holds 30–70 % of samples, and
     fewer than 5 % fall between.
   - **(c)** MSE-BC on the same data lands at the data mean, in the gap.
   - **Sanity check.** Reversing the DDIM timesteps fails (a): mode mean −0.456 against
     −0.598. At K = 1 every sample saturates near ±1, which is the schedule-limited regime.

*Reason.*
- Item 1: the frozen sweep crosses `Ta` with ensembling, so `Ta` must mean something when
  ensembling is on. Two other readings were rejected:
  - averaging every prediction any earlier replan made for t (up to Tp − To + 1 = 15) makes
    the `Ta` axis empty under ensembling;
  - taking the window from the last replan only is not ensembling.
- Item 2: a diffusion student has no deterministic action, so Phase 4's action-MSE < 1e-4
  cannot apply as written. Two reasons for the threshold:
  - A K = 10 sample adds the sampler's error to the fit. The last DDIM step's x̂₀ carries
    √(1 − ᾱ₉) ≈ 0.15 times the ε error.
  - The comparison spans 8 steps of open-loop prediction, not one.
  1e-3 is ten times the BC threshold, and closed-loop landing remains the binding check.
- Item 2, step count:
  - 10 000 steps at batch 128 is about 2 600 passes over the ~490 frames.
  - At the timing run's 12.79 steps/s that is 13 min.
  - The power-law EMA has no init bias to wait out. Unlike BC's constant 0.999, after 10 000
    updates its decay is 0.999, a window of about 1 000 steps at the tail of the cosine
    schedule.
- Item 3: the approved Phase 6 plan (Step 1, item 6) specifies re-flying the selected
  checkpoint on `select`. The determinism check costs nothing extra, because the reference cell
  has to be flown anyway.

*Evidence.* Code:
- `src/ila/policies/dp/{chunking,policy}.py`, `src/ila/policies/registry.py`;
- `src/ila/train/{smoke,select,dp_sweeps,dp_diagnostics}.py`;
- `scripts/{train_smoke,dp_sweeps,dp_diagnostics}.py`, `Makefile`.

Tests:
- `tests/test_dp_chunking.py`: `ensemble_sources` tables for To ∈ {1, 2} × `Ta` ∈ {1, 8, 15},
  with start-of-episode truncation; the To = 1 executed and target tables;
- `tests/test_policies.py`: ensembling, `Ta` = 1 equivalence, `ddim_sample` identity;
- `tests/test_dp_toy.py`;
- `tests/test_dp_jobs.py`: smoke MSE keys; sweep cells; reproduction of the selection rows on a
  full simulated flight; a tampered row caught; resume; diagnostics end to end.

*Consequence.*
- The `dp` smoke runs first, in the background (about 16 min). `dp` s0 is queued only after
  it passes.
- Sweeps and diagnostics run after selection. Each takes the selected checkpoint and never
  chooses one.
- Wall clock per seed, from P2-D2's 170 ms per K = 10 call on one CPU thread, at 24 workers:
  - sweeps: 15–25 min at teacher-length episodes (61.5 steps), up to about 6 × that if
    episodes time out;
  - diagnostics: a few minutes.
- **Conventions amended** (item 1). `il-conventions` and its mirror `docs/CONVENTIONS.md` said
  "average all predictions made for t by earlier replans". Both now state the `Ta`-window reading
  above, and stay identical (`tests/test_conventions_mirror.py`).

### P6-D2 — Deviation: `.venv-lerobot` gains lerobot's dataset, diffusion and training extras (2026-10-30)
*Decision.* R1 could not run in `.venv-lerobot` as P0-D1 froze it. That env was a bare
`pip install lerobot==0.6.1` with no extras, so it was missing three things:
- `import lerobot.datasets` raised `ImportError` (no `datasets`);
- `DiffusionPolicy` needs `diffusers`;
- `lerobot_train` needs `accelerate`.

The extras `lerobot[dataset,diffusion,training]==0.6.1` were installed with torch 2.11.0,
torchvision 0.26.0, numpy 2.2.6 and gymnasium 1.4.0 held as constraints.

*Reason.* R1 (plan Phase 6 item 6) needs those three packages to build its LeRobotDataset and to
train. P8's `.venv` is untouched, and only ONNX files and the neutral intermediate cross between
the venvs.

*Evidence.*
- A `pip install --dry-run` first: it changed none of the held packages.
- The installed set: datasets 4.8.5, diffusers 0.39.0, accelerate 1.15.0, pandas, pyarrow, av,
  torchcodec, wandb, plus their dependencies. For R1's ONNX export, onnx 1.23.2 and onnxscript
  0.7.2 were added under the same constraints. ORT is not installed there: parity and every
  flight run ORT in `.venv`.
- `results/env/pip_freeze_lerobot.txt` was regenerated. Its SHA-256 goes from `c00401da…` (P0-D1)
  to `d9a49039d22091a8e1d7e8dd9db3419a093d54f82bdee637e9c19b853d854d69`, and the torch,
  torchvision, numpy, gymnasium and lerobot lines are unchanged.

*Consequence.* P0-D1's freeze hash is superseded for `.venv-lerobot` only. No `ila` code pins
it.

### P6-D3 — R1 configuration and pipeline (2026-10-30)
*Status: frozen 2026-11-02 at Gate 6 (written before any R1 training run; the 12-worker decision
confirmed by the user).*

*Decision.* `R1_lerobot_dp` (P3-D1 §3: reference row, tag `reference_external`, 3 seeds, `id`
SS3–SS6, the same selection rule at `Ta` = 8, K = 10) is configured, trained, exported and flown
as follows. Everything is simulation only. Times are model scale (one control step = 1/30 s model
scale, 1/6 s full scale).
1. **Config.** `configs/train/r1_lerobot_dp.yaml`, SHA-256
   `a55e2fa07763669b4d081df03bfa7a179eeb1c6d969b7ee16ddc23faadfc6720`, entered in
   `ila.train.manifest.FROZEN_CONFIG_SHA256` (P3-D1 §3: "hashed at its creation").
   `scripts/r1/train_r1.py` parses that table without importing `ila` and refuses a `train` run
   whose config differs from it, a dirty tree, or any step, episode, cadence or loader override.
   - **Set explicitly, matched to `dp.yaml`.** LeRobot `DiffusionConfig`: `n_obs_steps` 2,
     `horizon` 16, `n_action_steps` 8; `crop_shape` [88, 88], `crop_is_random` true (centre crop
     in eval mode); `resnet18`, `pretrained_backbone_weights` null, `use_group_norm` true;
     `spatial_softmax_num_keypoints` 32; `down_dims` [128, 256, 512], `kernel_size` 5,
     `diffusion_step_embed_dim` 128; DDPM 100 train steps, `squaredcos_cap_v2`, epsilon,
     `clip_sample` true with range 1.0. `TrainPipelineConfig`: 100 000 steps, batch 128,
     `save_freq` 10 000, wandb off; seeds 0, 1, 2. `push_to_hub` false (infrastructure).
   - **Set explicitly, infrastructure: `num_workers` 12** (LeRobot's default is 4; that default
     is still asserted, under `lerobot_defaults_overridden`). Item 7 gives the reason and the
     rates.
   - **Kept at LeRobot 0.6.1's defaults** (recorded in the config's `lerobot_defaults_kept`;
     `train_r1.py` asserts each against the installed source before any step):
     - normalisation VISUAL MEAN_STD, STATE MIN_MAX, ACTION MIN_MAX;
     - `drop_n_last_frames` 7 (the last 7 frames of each episode are never sampled as
       anchors);
     - `n_groups` 8 (U-Net GroupNorm); `use_film_scale_modulation` true;
     - `use_separate_rgb_encoder_per_camera` true; `resize_shape` null; `crop_ratio` 1.0;
     - `noise_scheduler_type` DDPM; `beta_start` / `beta_end` 1e-4 / 0.02 (unused by
       `squaredcos_cap_v2`); `num_inference_steps` null;
     - `do_mask_loss_for_padding` false; `compile_model` false; `gradient_checkpointing` false;
     - `use_amp` false, so R1 trains in **fp32** (`dp` trains in bf16);
     - optimiser preset: **Adam** (`torch.optim.Adam`, coupled L2 weight decay), lr 1e-4,
       betas (0.95, 0.999), eps 1e-8, weight decay 1e-6, gradient-norm clip 10; cosine with
       500 warm-up steps (diffusers `get_scheduler`). `dp` uses AdamW, betas (0.9, 0.999) and
       no clipping (P4-D1 item 2);
     - loader: prefetch 4, persistent, `spawn`; `cudnn_deterministic` false (cuDNN
       benchmark on, TF32 matmul allowed, as `lerobot_train` sets them); `log_freq` 200;
     - dataset: `use_imagenet_stats` true (VISUAL MEAN_STD uses ImageNet mean and std, not the
       data's); image transforms off; `eval_split` 0.
   - **EMA: none.** LeRobot 0.6.1 has no EMA for diffusion. There is no `EMAModel` in
     `policies/diffusion` or `scripts/lerobot_train.py`, and `train_r1.py` asserts this. R1
     saves and flies **raw** weights; `dp` flies its EMA. This difference is stated with every
     R1-vs-`dp` comparison.
2. **Training code.** `scripts/r1/train_r1.py` runs in `.venv-lerobot` only, with the standard
   library, yaml, torch and lerobot; it never imports `ila`. Its loop is `lerobot_train.train`
   step for step, built from LeRobot's own functions: `make_train_eval_datasets`, `make_policy`,
   `make_pre_post_processors`, `make_optimizer_and_scheduler`, `EpisodeAwareSampler`,
   `update_policy`, `save_checkpoint`, `load_training_state`, `compute_sampler_state`.
   `HF_HUB_OFFLINE=1`. Around it is `ila`'s bookkeeping:
   - Run dir `artifacts/runs/R1_lerobot_dp/s<seed>/`. It holds `status.json`, `train_log.jsonl`
     and `manifest.json`. The manifest records the config SHA-256 against the frozen table, the
     dataset path, its `.build.json` SHA-256 (every file it lists is re-hashed), the `ila` and
     P8 commits with the clean-tree flag, lerobot and torch versions, host, GPU and thread
     variables, wall clock and GPU-hours.
   - Checkpoints every 10 000 steps go to `checkpoints/<NNNNNN>/` in LeRobot's layout: weights,
     processors, optimizer, scheduler, step, and the python / numpy / torch / CUDA RNG states.
     Each is written to a temporary directory and renamed. The checkpoint SHA-256 is
     `model.safetensors`'s, and every file is hashed.
   - **Resume** (P3-D1 §4) restarts from the last complete checkpoint: weights, optimizer,
     scheduler and RNG state, plus the data order (`compute_sampler_state`). The resume is
     logged. It is not claimed bit-identical on the GPU (P4-D1 item 8). It was exercised on the
     CPU (step 4 to 6) and on the GPU (step 200 to 300).
   - **Leak audit R1, asserted after `make_policy`.** The policy's inputs are exactly
     `observation.image` [3, 96, 96] and `observation.state` [13], and its output is `action`.
     There is no `environment_state`, and the parquet's `timestamp`, `frame_index` and `index`
     never become inputs.
3. **Differences from `dp`.** These are LeRobot's recipe, reported, not tuned.
   - *Normalisation.* `dp` uses identity actions and standardised proprio through a 2 × 64
     MLP. R1 uses:
     - ACTION MIN_MAX, which maps each axis's training-label [min, max] to [−1, 1]. With
       `clip_sample` R1 can therefore only command vx ∈ [−0.558, 0.629], vy ∈ [−0.444, 0.564]
       and vz ∈ [−1.0, 0.265] (normalised; × 1.5 m/s model scale). These are the checkpoint
       postprocessor's statistics, equal to the extremes of the 61 518 training labels.
     - STATE MIN_MAX without clipping. The normalised state enters the conditioning raw, with
       no proprio MLP.
     - Images use ImageNet MEAN_STD.
   - *Encoder and conditioning.* GroupNorm with `num_features // 16` groups (`dp`: 16 groups).
     Spatial softmax is followed by Linear(64, 64) and ReLU per frame. `global_cond` = flatten
     over `To` of [state (13), image feature (64)], so C = **154** (`dp`: 256).
   - *Sampling of training data.* `drop_n_last_frames` 7; edge padding with no loss mask.
   - *Randomness in training.* The random crop (torchvision `RandomCrop`, one crop per batch)
     and the DDPM ε and k draw from the global torch RNG, seeded by `set_seed(run_seed)`. This
     is LeRobot's, not the data-loading generator of il-conventions. It affects training only;
     no flight noise comes from it.
   - *Inference.* LeRobot's native eval would sample DDPM-100 with the global RNG. R1 is flown
     by `ila` instead, with the same trailing DDIM, K = 10 and per-episode noise
     `(run_seed, episode_seed, t)` as `dp` (item 4).
4. **Interface contract** (fixed with `deploy-benchmarker`). Each checkpoint at step N is
   exported by `scripts/r1/export_onnx.py` (`.venv-lerobot`, CPU) to
   `<run>/onnx/step<N>/`:
   - `encoder.onnx`: `image` uint8[B, 2, 96, 96, 3] (HWC, oldest first) and `state`
     float32[B, 2, 13] (raw proprio) → `global_cond` float32[B, C]. LeRobot's input
     normalisation and the eval centre crop are inside the graph.
   - `denoiser.onnx`: `sample` float32[B, 16, 3] (normalised), `timestep` int64[B] and
     `global_cond` → `eps` float32[B, 16, 3]. Opset 17. No RNG op: `OnnxDPStudent` refuses a
     graph that holds one.
   - `meta.json` (`ila-r1-onnx/1`): horizons, C, both schedulers, the MIN_MAX statistics, and
     the source checkpoint's and graphs' SHA-256 and versions.
   - `reference.npz`: the torch reference for parity.

   `ila.policies.onnx_dp.OnnxDPStudent` (registry `R1_lerobot_dp`, kind `onnx_dp`, tag
   `reference_external`) runs ORT CPU with one thread, sequentially, through
   `ila.deploy.r1_parity`'s helpers. Its sampler is `ddim_sample` with `dp`'s trailing DDIM at
   K, initial noise `initial_noise((run_seed, episode_seed, t))`, then the final clip to
   [−1, 1] in normalised space, LeRobot's float32 MIN_MAX inverse, and a second clip to [−1, 1]
   (the action bound). Chunk execution, replanning and ensembling are `dp`'s own code: the
   executor moved, unchanged, into `ChunkedDiffusionStudent`, shared by `DPStudent` and
   `OnnxDPStudent`. `dp` stays bit-identical, checked against `efbf359`'s `policy.py` on a
   dummy `dp` at (`Ta` 8, K 10), (`Ta` 4, ensembling) and (`Ta` 1, K 2) over 17 steps.
5. **Selection.** The same rule (P3-D1 §4), at `Ta` = 8, K = 10, no ensembling, on the 150
   `select` episodes, through `select_run`. An R1 run's config sets `selection.candidates:
   onnx_exports`.
   - The candidates are its exports `onnx/step<N>/`, one per manifest checkpoint. Each export's
     `meta.json` must name that checkpoint's `model.safetensors` SHA-256.
   - A candidate's SHA-256 (the rows' `ckpt_sha256`) is `meta.json`'s. Both graphs are
     re-verified against it before flying.
   - **Parity gate.** Every export must carry a `parity.json` (`scripts/r1/parity.py`) written
     for its exact `meta.json` and graphs. A run with an unchecked export is refused. A failed
     parity bars that checkpoint: it is never flown and is listed under `excluded_by_parity`.
   - `selection.json` keeps the same format, with `weights: raw_no_ema` and
     `inference_path: ort_cpu_fp32_single_sample`.
6. **R1 smoke** (`make train-smoke ARM=R1_lerobot_dp` → `results/phase6/smoke_R1_lerobot_dp.json`).
   LeRobot trains 10 000 steps (the `dp` smoke's count, P6-D1 item 2) on LeRobotDataset
   episodes 0–7, with every other setting from the R1 config. The run is then exported, parity
   is run, and the episodes are flown through `OnnxDPStudent` at `Ta` = 8, K = 10. It passes iff:
   - (0) the replay check holds (P6-D1);
   - (1) LeRobot episodes 0–7 are the store's first 8 kept episodes, the `dp` smoke's, by group
     and demo index;
   - (2) the export's parity passed;
   - (3) at least 7 of 8 land `success`. Outcome classes are printed.

   `dp`'s sampled executed-chunk MSE is reported through the ONNX student but **not judged**.
   It is given over all frames and over the frames LeRobot samples as anchors (t ≤ T − 8):
   with `drop_n_last_frames` = 7, the last 7 frames of each episode are never trained as
   anchors, so the all-frame figure is not an overfit measure. `dp`'s 1e-3 is printed beside it
   as a reference.
7. **Loader workers, measured rate and projection** (P3-D1 §5 proxy check; dev runs tagged
   `test`, RTX A4000, full dataset, real schedule; `artifacts/runs/R1_lerobot_dp_test/`).
   - **At LeRobot's default 4 workers**, the steady rate is **3.7–3.9 steps/s** (3.88 and 3.73
     in the two segments of a 300-step run; 3.68 with `OMP_NUM_THREADS` unset). The rate over
     all steps, warm-up included, is 3.36. Training is bound by PNG decoding in the loader:
     data loading takes 0.175–0.19 s per step against 0.10–0.14 s for the fp32 update. GPU
     memory peaks at 1.39 GiB. Projection: ≈ 7.3 h per run, ≈ 21.9 GPU-h for 3.
   - **At 12 workers** (a 140-step `test` run), the steady rate is **9.82 steps/s**, with data
     loading at 0.05 s per step.
   - **Decision: 12 workers.** It was decided by the Phase 6 coordinator on 2026-10-30, before
     any R1 train run and before the config was hashed for training. It was recorded for the
     user's confirmation at the gate, and **the user confirmed it on 2026-11-02**.
     - It is infrastructure. The worker count changes neither the data order
       (`EpisodeAwareSampler` is a pure function of seed and epoch) nor any sample, because no
       randomness runs in the workers: image transforms are off, and the crop and the DDPM
       noise run in the main process.
     - It departs from "every other setting at LeRobot's default" in this one place.
   - **Projection at 12 workers:** 100 000 / 9.8 ≈ 10 200 s ≈ 2.8 h per run, so **≈ 8.5 GPU-h
     for 3 runs, against P4-D2's 6.5 GPU-h proxy (+2.0 GPU-h)**. P4-D2's 91.6 h becomes about
     93.6 h, under the 120 h limit. Under P3-D1 §5 any overrun is reported, not cut. The first
     train run's own rate is compared with this projection.

*Reason.* R1 tests whether `dp`'s numbers depend on `ila`'s implementation. It must be
LeRobot's recipe at the matched hyper-parameters, trained by LeRobot's code and flown by the same
runner, sampler, noise keys and selection as `dp`. Items 1–3 write down every place where it
nevertheless differs, before any number exists. The parity gate keeps an export that does not
match its torch weights out of selection.

*Evidence.*
- Code:
  - `configs/train/r1_lerobot_dp.yaml`;
  - `scripts/r1/train_r1.py`;
  - `src/ila/policies/{onnx_dp,registry}.py`, `src/ila/policies/dp/policy.py`
    (`ChunkedDiffusionStudent`);
  - `src/ila/train/{select,smoke,manifest}.py`;
  - `Makefile` (`r1-train`).
- Tests:
  - `tests/test_student_onnx_dp.py` (synthetic ONNX pair exported in-test): contract, torch
    reference, execution indices and noise keys, determinism, the MIN_MAX inverse, the
    `meta.json`, hash, RNG-op and graph guards, the registry, ONNX candidates, the parity gate,
    and a runner flight independent of the worker count;
  - `tests/test_step_guard.py`: the executor's clock sites; no clock read in `onnx_dp.py`; ORT
    session inputs invariant to the episode seed and step, with and without ensembling;
    ensembling invariance for `dp` (leak audit R2);
  - `tests/test_render_access.py` and `tests/test_data_export_lerobot.py`: the frozen-list scan
    covers R1's pathway (leak audit R3).
- Dev runs:
  - CPU, 8 episodes, 4 + 2 steps with a resume;
  - GPU, full data, 200 + 100 steps with a resume, plus 140-step loader measurements at 4
    workers (`OMP_NUM_THREADS` unset) and at 12 workers.

  None is selectable.

*Consequence.*
- R1 s0–s2 are queued only after the R1 smoke passes (rule 4).
- Each finished run is exported (≈ 46 s per checkpoint on the CPU), and parity is run on every
  export before `make select-bg SELECT="--arm R1_lerobot_dp --seeds 0 1 2"`.
- R1 is never mixed into core tables.

### P6-D4 — Deviation: R1 trains despite its smoke failing; what the smoke can and cannot show (2026-10-30)
*Decision* (approved by the user on 2026-10-30). R1 s0–s2 train as configured (P6-D3, config
`a55e2fa0…6720`), although `results/phase6/smoke_R1_lerobot_dp.json` failed the ≥ 7/8 closed-loop
rule (6/8, 2 `off_pad`). The overfit, parity, replay and episode-match checks passed. Nothing in
R1's recipe changes.

*Reason.* On one noise key, a ≥ 7/8 smoke cannot tell a diffusion student apart from a bug. The
investigation re-flew both smoke checkpoints on the same 8 `train` episodes (`ppo_2000` 0–7) at
run_seed 0–9 (`Ta` = 8, K = 10, CPU). Episodes and physics were unchanged; only the noise keys
moved. No `select` episode and no frozen list was flown. Each re-fly at run_seed 0 reproduced
its smoke JSON exactly.

| Student | Success per run_seed 0–9 | Pooled | run_seeds at ≥ 7/8 |
|---|---|---|---|
| R1 (ONNX, raw weights) | 6 5 3 4 6 6 6 4 4 5 | 49/80 | 0 |
| `dp` smoke, EMA | 7 4 5 6 4 5 7 6 6 7 | 57/80 | 3 |
| `dp` smoke, raw weights | 5 3 6 5 5 5 4 5 6 6 | 50/80 | 0 |

- **Contrasts** (bootstrap over seeds and episodes):
  - R1 − `dp` EMA = −0.10 [−0.31, +0.10];
  - `dp` raw − `dp` EMA = −0.09 [−0.30, +0.10].

  Neither excludes 0.
- **`dp`'s own smoke pass (P6-D1 item 2) was a favourable draw.** Its per-episode rates give a
  0.26 chance that one 8-episode flight passes. The `dp` runs are not affected: the smoke is a
  pre-flight check, and selection flies 150 `select` episodes.
- **No bug signature:**
  - The first chunk follows the demo to under 1 mm.
  - Labels shifted by ±1 step raise the open-loop MSE about 850×, so the alignment is right.
  - ONNX parity passes.
  - R1 sits at its MIN_MAX action bounds on 0.16 % of executed steps, so that is not a factor.
- **Where the failures come from.** All three students drift from the demo after the first
  replan, from 4–5 mm at t 8–15 to about 8 cm at touchdown (model scale). They land
  0.078–0.085 m (model scale) from the pad centre on average, against the teacher's 0.010 m.
- **`drop_n_last_frames` = 7** makes R1 about 12× worse than `dp` open loop on each episode's last
  7 frames. Those frames are 8–15 steps after first contact (the dwell), so this cannot cause
  `off_pad`, which is decided at first contact.
- **Training rate.** P6-D3's 9.82 steps/s was measured inside the 48-batch prefetch buffer of a
  140-step test run, so it overstates the rate.
  - The loader alone, built as `train_r1.py` builds it (12 workers, full dataset), gives 9.03
    batches/s steady. With 4 workers it gives 3.91.
  - R1 is data-bound at about 8.5–9 steps/s: 3.1–3.3 h per run, **9.3–9.8 GPU-h for 3 runs**,
    against P6-D3's 8.5. P4-D2's total becomes about 94.4–94.9 h of 120. This is reported, not
    cut.
  - The smoke's 2.93 steps/s is an artefact of 8 episodes (the pipeline restarts every 4
    batches).

*Evidence.*
- Summaries: `results/phase6/r1_investigation/`
  - `analysis_flights.json` (SHA-256 `211cdf03…f717`)
  - `divergence.json` (`731283c8…3087`)
  - `analysis_openloop.json` (`b3d7557b…458f`)
  - `analysis_ta1.json` (`12d77538…67e3`)
  - `loader_*.json`
- Scripts: `results/phase6/r1_investigation/scripts/`.
- Per-step flight pickles were not committed.

*Consequence and caveats.*
- **A diagnostic only, never used for a choice.** At `Ta` = 1 on the same 8 `train` episodes:
  - `dp` EMA lands 72/80, against 57/80 at `Ta` = 8;
  - R1 lands 59/80, against 49/80, with 5 hard landings.

  These are overfit-8 models. The core row's `Ta` stays 8 (P3-D1 §4), and H3 is scored only in
  Phase 8 on `id` SS6. The result is recorded because H3 is about exactly this mechanism.
- **R1 is a reference with a recorded recipe gap.** It has no EMA, MIN_MAX normalisation and raw
  state conditioning. Under per-step feedback it is weaker than `dp` (−0.16 [−0.41, +0.06] at
  `Ta` = 1, not significant). That is a caveat to any R1 vs `dp` comparison, not a bug.
- **Side finding.** `registry.load_student(weights="raw")` reads `ckpt["raw"]`, but
  `loop.py` saves raw weights under `"model"`. This is a diagnostics-only path; no selection or
  flight uses it. It is fixed after the Phase 6 queue, because the trainers refuse a dirty tree.
- R1 s0–s2 train after `dp` s4. Each is exported, passes parity, then is selected by the P3-D1 §4
  rule at `Ta` = 8, K = 10.

### P6-D5 — `dp` and R1 run verdicts, selection, the chunking finding, budget (2026-10-31; revised 2026-11-02 after the results-skeptic review)
*Decision.* All 8 Phase 6 runs are kept: no exclusions, no re-selection, nothing tuned. Final
verdicts from the `training-run-diagnostician`:

| Run | Verdict | Selected step | `select` success | Main reasons |
|---|---|---|---|---|
| `dp` s0 | TRUSTWORTHY | 100 k | 63/150 | |
| `dp` s1 | TRUSTWORTHY | 50 k | 66/150 | |
| `dp` s2 | TRUSTWORTHY | 60 k | 68/150 | |
| `dp` s3 | TRUSTWORTHY | 100 k | 67/150 | |
| `dp` s4 | TRUSTWORTHY | 100 k | 62/150 | |
| `R1_lerobot_dp` s0 | CAVEAT | 100 k | 54/150 | no EMA; over-trains past 40 k; LeRobot recipe gap; reference only |
| `R1_lerobot_dp` s1 | CAVEAT | 40 k | 57/150 | same |
| `R1_lerobot_dp` s2 | CAVEAT | 50 k | 53/150 | same |

Source: `results/phase6/runs_summary.csv` and `results/phase6/runs/<arm>/s<n>/`.

*Findings.* All are simulation results at model scale, on the 150 `select` episodes only. No
frozen list was flown.

1. **Provenance.**
   - Configs: `dp.yaml` `c8f90c35…6913`, `r1_lerobot_dp.yaml` `a55e2fa0…6720`. Data: `ppo_2000`
     `35b70cbd…7cfd`, first 1 000 episodes, 999 kept, 61 518 frames.
   - Every tree was clean. `dp` s0–s2 ran at `b912942`; `dp` s3–s4 and every R1 run at `f9659fa`.
     The two commits differ only in protocol and results files, with no code change.
   - No resumes. Every checkpoint SHA-256 was recomputed and matches.
   - R1's `selected_ckpt_sha256` is the SHA-256 of the export bundle's `meta.json`. That file
     records the `model.safetensors` SHA-256, which equals the manifest's.
2. **Selection followed P3-D1 §4 in all 8 runs**, and was decided by success alone. The p95
   tie-break was never used, so the Phase 6 note's expectation did not happen.
   - Winning margins were 1–5 episodes.
   - Differences between the 10 checkpoints of a run are mostly binomial noise: Cochran's Q is
     not significant in 6 of 8 runs (exceptions: `dp` s2 p = 0.002, R1 s1 p = 0.004).
   - The selected scores are what taking the maximum of 10 noisy draws predicts. Candidate means
     are `dp` 55.9 and R1 44.8; the expected maxima are 65.1 and 53.6; the selected means are 65.2
     and 54.7. **Phase 8 should expect lower numbers than the selected `select` scores.**
   - Source: `results/phase6/diagnostics/final_verdicts/{sel,pool}.out.txt`.
3. **Held-out denoising loss is not a selection proxy.**
   - Every `dp` seed reaches its held-out EMA minimum at 41–45 k, then rises 53–70 % by 100 k;
     89–90 % of the rise is in k 0–24 (`results/phase6/diagnostics/dp/heldout_curves.json`, EMA and
     raw by k-bucket at every evaluation). R1's offline reconstruction has its minimum at 40 k in
     3 of 3 seeds, then rises 55–72 % (`results/phase6/diagnostics/R1_lerobot_dp/heldout_reconstruction/`).
   - Yet `dp` success *rises* late. Mean success by step is 51.6 / 51.4 / 50.8 / 51.8 at 10–40 k
     and 56.4 / 59.2 / 61.0 / 61.6 at 70–100 k.
   - Two contrasts were tested (post hoc). 100 k − 40 k is positive in 5 of 5 seeds (+7, +2, +10,
     +16, +14): sign test p = 0.0625. 100 k − 50 k is +13, −15, +9, +4, +16: sign test p = 0.375.
     The episode-level McNemar values (p = 0.011 and 0.163) ignore clustering by seed and are not
     relied on. So success does not *follow* held-out loss; a late rise is suggestive only, and
     it sits within item 2's noise.
   - The diagnostician had predicted more `off_pad` and `hard_landing` late. `off_pad` did rise
     (44.0 % vs 40.4 %), but `hard_landing` fell (10.2 % vs 17.5 %), so the prediction was
     falsified.
   - Source: `results/phase6/diagnostics/final_verdicts/{pool,late}.out.txt`.
4. **`dp` at its frozen setting (`Ta` = 8, K = 10) lands far below the BC family.**
   - `dp` 62–68/150, mostly `off_pad`, against 145–150/150 for `bc`, `bc_matched`, `bc_dart`,
     `bc_dagger` and `bc_state` (`results/phase4/runs_summary.csv`, `results/phase5/runs_summary.csv`).
   - The median lateral touchdown offset is 0.092 m, against an `off_pad` boundary of 0.10 m
     (success maximum 0.100 m, `off_pad` minimum 0.100 m). At `Ta` = 8, outcomes sit on the
     threshold, which is why selection is noise-dominated.
   - Tunnelled episodes: 13–26 per run, against 0–2 for the BC family. They close faster: median
     0.559 m/s against 0.385 m/s overall. 17 tunnelled `dp` episodes are scored `success` under
     P8's rule.
   - Across the 5 runs there was 1 timeout (s1 SS5 #22) and 1 crash (s3 SS5 #48, tilt).
     Detectors disagreed 0 times.
   - Source: `results/phase6/diagnostics/final_verdicts/tun.out.txt`.
5. **Inference sweeps on `select`** (`results/dp_sweeps.json`; reported only; unscored; nothing
   re-selected). Success per seed s0–s4 and mean:

   | Cell | Success, s0–s4 | Mean | With ensembling (mean) |
   |---|---|---|---|
   | `Ta` 1 | 143 129 137 141 145 | 139.0 | (same policy) |
   | `Ta` 2 | 130 113 122 130 127 | 124.4 | 135.2 |
   | `Ta` 4 | 75 84 93 83 91 | 85.2 | 117.0 |
   | `Ta` 8 | 63 66 68 67 62 | 65.2 | 112.6 |
   | `Ta` 12 | 53 64 69 51 55 | 58.4 | 123.0 |

   - Every seed's reference cell reproduced its selection rows text for text.
   - The `Ta` = 8 / K = 10 cell is the selection cell and carries the full selection optimism of
     item 2. Every other cell flies the same checkpoint and carries part of it (`Ta` = 12 off
     executes the same first chunk).
   - Ensembling beats no ensembling in all 20 seed × `Ta` comparisons and recovers 59–80 % of the
     gap to `Ta` = 1. With ensembling, success is not monotone in `Ta`: `Ta` 12 beats `Ta` 8 in 5
     of 5 seeds.
   - **K.** K 5 / 25 / 100 give means of 59.8 / 56.8 / 54.2, against 65.2 for K = 10 in the
     selection cell (the candidate mean at K = 10 is 55.9). All carry some optimism, and they
     cannot be told apart. "K = 10 beats K = 25 or 100" must not be claimed.
     K = 1 and 2 (0/150 and 23–36/150) are labelled `schedule_limited`.
   - At `Ta` = 1 the leading failure is `hard_landing` (38 of 750), not `off_pad` (14 of 750).
   - Source: `results/phase6/diagnostics/final_verdicts/sweeps.out.txt`.
   - **Reading (revised after review, S1 and S2).** The loss at `Ta` = 8 is mostly a chunking
     effect: executing 8-step chunks (0.27 s model scale, 1.33 s full scale) open loop.
     - **It does not grow with deck motion.** Pooled over 5 seeds, `Ta` 1 → 8 goes 245 → 113 of
       250 at SS3, 235 → 116 at SS4 and 215 → 97 at SS5 (−52.8, −47.6, −47.2 points). In the
       calmest slice, SS3 at 0 kn, it goes 94 → 40 of 95, and the median lateral offset goes from
       0.020 to 0.096 m (model scale).
     - Together with the under-dispersed, biased chunk predictions of item 7, this points to
       multi-step prediction error that does not depend on sea state, not to deck bandwidth. A
       `Ta` contrast therefore cannot attribute the loss to deck motion. H3's verdict in Phase 8
       (`id` SS6, same checkpoints; wording and threshold as frozen) is reported with this
       caveat.
     - **`dp` at `Ta` = 1 is still below `bc`.** It lands mean 139.0/150 (129–145) against
       `bc`'s selected 145–149, below `bc` in every seed, with 38 hard landings in 750 against
       `bc`'s 7. This is an unpaired reading: different checkpoints and selection settings.
     - Source: `results/phase6/diagnostics/final_verdicts/review_fixes.out.txt`.
6. **No execution (index) bug.** The discriminating evidence is (a) and (c). (a) and (b) come
   from the overfit-8 smoke checkpoints on `train` episodes 0–7, not from the selected runs. (d)
   and (e) are the observations being explained, so they are consistency checks, not
   independent evidence.
   - (a) Shifting labels by ±1 step makes open-loop MSE 506× worse for `dp` and 850× worse for R1.
     *Correction to P6-D4:* its "about 850×" is R1's figure.
   - (b) In closed loop the first chunk stays within 0.007–0.009 (`dp`; R1 0.004–0.007) of the
     same-step label, with ≤ 1 mm xy drift (model scale) through t = 7.
   - (c) All 30 R1 parity files show LeRobot's own `select_action` matching the executed slice
     To − 1 … To + 6.
   - (d) `Ta` = 1 lands mean 139.0/150 over 5 seeds.
   - (e) Degradation grows with `Ta` and is mostly `off_pad`.
   - Sources: `results/phase6/r1_investigation/{analysis_openloop,divergence}.json` and the R1
     `parity.json` files.
7. **DP diagnostics** (`results/phase6/diagnostics/dp/s<n>.json`; normalised actions,
   × 1.5 m/s model scale).
   - **Spread.** Over the executed slice, the spread of 16 samples is 0.015–0.033. The
     sample-mean RMS error is 0.071–0.077, so samples are under-dispersed but not collapsed.
   - **Clipping.** In closed loop on `select`, executed actions at ±1 occur in 0.027–0.096 % of
     steps (`select.executed_at_bound`), against 0.195 % in the labels.
   - **Jerk.** First-difference ratio at chunk boundaries vs within chunks: 1.35–1.47 in closed
     loop, 1.15–1.25 open loop, 1.09 for the teacher's labels. The closed-loop re-fly reproduced
     the selection rows and P8's `action_jerk_mean` in all 5 seeds.
8. **R1 vs `dp` at `Ta` = 8** (descriptive; 3 vs 5 seeds).
   - Selected: −7.0 points, 95 % CI [−14.5, +0.7]. This is an ad hoc bootstrap (rng seed 0,
     20 000 replicates, `final_verdicts/scripts/r1dp.py`), not P8's `paired_bootstrap`, and it is
     descriptive only. Candidate means: −7.4 points (no CI computed).
   - Same failure profile: lateral offset median 0.099 m, `off_pad`-dominated.
   - The gap is in the direction the recorded recipe gaps predict: no EMA, MIN_MAX
     normalisation, fp32 Adam (P6-D3). It has the size of P6-D4's `dp` raw − EMA effect on the
     smoke models. Judged not large enough to need further investigation before Phase 8 (plan
     Phase 6 item 6), on weak support: the CI only just includes 0.
   - R1 reproduces the main result (`Ta` = 8 far below BC) with independent training code,
     dataset indexing and normalisation. It shares ila's executor, sampler and runner, which (c)
     covers.
   - Source: `results/phase6/diagnostics/final_verdicts/r1dp.out.txt`.
9. **Budget** (wall-clock GPU-h).
   - `dp`: 2.2095 / 2.2142 / 2.2160 / 2.1889 / 2.1715 = **11.000**, against 10.9 (+0.9 %).
   - R1: 2.9248 / 2.9548 / 2.9424 = **8.822**, against P4-D2's 6.5 proxy (+2.32 h). About 0.07 h
     of that came from diagnostics sharing the CPU with s1 and s2.
   - Cumulative training through Phase 6: 17.50 + 7.11 + 19.82 = **44.43 GPU-h**, against 40.9
     projected through Phase 6.
   - Projected total if `modular` and the ablations run at projection: about 95 of 120 h.
   - No cut (P4-D2).
   - Sweeps took 30–34 min per seed, against P6-D1's 15–25 min estimate.

*Evidence.*
- `results/phase6/runs_summary.csv`.
- `results/phase6/runs/<arm>/s<n>/{manifest.json, selection.json, select/*_rows.csv}`, plus R1's
  `onnx/step*/{parity,meta}.json`.
- `results/dp_sweeps.json` and `results/phase6/sweeps/`.
- `results/phase6/diagnostics/{dp, R1_lerobot_dp, final_verdicts}/`. The scripts read
  `artifacts/` paths; the same rows are committed under `results/phase6/runs/`.

*Consequence.*
- **The README must state:**
  - that `dp` at its frozen `Ta` = 8 lands far below BC on `select`, and that this is mostly a
    chunking effect that does not depend on sea state (item 5). At `Ta` = 1, `dp` (distilled from
    P8 `ppo` s0) lands mean 139.0/150 (129–145), still below `bc`'s 145–149 in every seed
    (unpaired);
  - that selection is noise-dominated and the selected scores are optimistic;
  - that held-out loss is not a selection proxy: success does not follow it (item 3).
- **Phase 8, GPU acceptance.** The binding GPU-acceptance test for `dp` (≤ 3 of 150 flips, P3-D1
  §3) has not been run. Outcomes at `Ta` = 8 sit on the 0.10 m threshold, so a failure is likely.
  Plan the CPU-path budget now:
  - 170 ms per K = 10 call;
  - the `Ta` = 1 and ensembling cells replan every step.

*Review response (results-skeptic, 2026-11-02; no BLOCKER).*
- **S1.** The deck-motion reading was withdrawn. The per-sea-state and 0 kn breakdown is in item 5.
- **S2.** "Not a weaker network" was replaced by the `dp` vs `bc` comparison at `Ta` = 1 (item 5
  and the README requirement).
- **M1.** The scope of each piece of item 6's evidence is now stated.
- **M2.** Item 5 now says the other cells share the selected checkpoint and part of its
  optimism.
- **M3.** Item 3 now reports both contrasts and the seed-level sign tests.
- **M4.** Held-out curves are committed in `results/phase6/diagnostics/dp/heldout_curves.json`,
  and the clip range comes from one source.
- **M6.** Item 8 now labels its bootstrap ad hoc and descriptive.
- **M7.** Wording fixed in P6-D1 ("full simulated flight") and model scale added in P6-D4; P6-D5
  is now dated to its commit.
- **M8.** `select.py` now writes `inference: {Ta, K, temporal_ensembling}` into
  `selection.json` (`ila.train.select.inference_settings`). The Phase 6 selections predate this.
  For them, `Ta` = 8, K = 10 is shown by the sweep determinism check: each reference cell flown
  at `Ta` = 8, K = 10 reproduced the selection rows text for text.
- **M5.** The user confirmed P6-D3's 12-worker decision on 2026-11-02, and P6-D3 is frozen.

### Gate 6 passed (2026-11-02)

| Criterion | Measured value / evidence | Result |
|---|---|---|
| CPU tests green | `make lint` clean (ruff, format, mypy --strict, 67 files). `make test`: 786 passed (11 min 48 s, GPU and slow included) at `a898704`. CI green on `phase-6` (run 41253888633, `c83019c`) | PASS |
| 5 seeds trained | `results/phase6/runs/dp/s0–s4/manifest.json`: `done`, 100 000 steps each, clean trees (`b912942` ×3, `f9659fa` ×2; no code difference between them), no resumes, `dp.yaml` `c8f90c35…6913` | PASS |
| 5 seeds selected | `results/phase6/runs/dp/s*/selection.json`: 10 EMA candidates each on `select` (`07ecf049…`), CPU fp32 path, at `Ta` = 8, K = 10 (shown by the sweep reference cells). Selected 63/66/68/67/62 of 150 (`runs_summary.csv`). The rule was recomputed independently from the committed rows by the diagnostician and the skeptic | PASS |
| Diagnostician TRUSTWORTHY | `dp` s0–s4 TRUSTWORTHY (P6-D5). R1 s0–s2 CAVEAT (a reference arm, not gating) | PASS |
| Sweep JSON committed | `results/dp_sweeps.json` with rows in `results/phase6/sweeps/dp/`; `reproduces_selection_rows: true` for all 5 seeds | PASS |

*Reviewers.*
- `privileged-leak-auditor` (Phase 6 code and the R1 data path): no LEAK. Three RISKs were fixed
  with guard tests (P6-D3).
- `training-run-diagnostician`, every run: `dp` TRUSTWORTHY ×5, R1 CAVEAT ×3 (P6-D5).
- `results-skeptic`: no BLOCKER. Two SERIOUS (the withdrawn deck-motion reading; `dp` vs `bc` at
  `Ta` = 1) and eight MINOR, all addressed in P6-D5's review response.

*Deviations this phase.*
- P6-D2: `.venv-lerobot` extras.
- P6-D4: R1 trained despite its smoke failing (user-approved).
- P6-D3's 12-worker decision was confirmed by the user.

No threshold was changed.

## Phase 7

*Before you start (from Phase 6).*
- **Budget.** Training so far is 44.43 GPU-h of wall clock against 40.9 projected through Phase 6
  (P6-D5 item 9). `modular` and the ablations at projection bring the total to about 95 of 120 h.
  - The five `dp` ablations (`dp_mix`, `dp_pid`, `dp_n100`, `dp_n300`, `dp_minus_last_action`)
    are 3 seeds each at about 2.2 GPU-h per run.
- **`dp` ablations inherit Phase 6's selection noise.** At `Ta` = 8, outcomes sit on the 0.10 m
  `off_pad` boundary, so per-checkpoint differences are mostly binomial noise and selected
  `select` scores run about 6 points above the candidate mean. Read `dp`-ablation selections the
  same way.
- **H2's touchdown-time plots** (plan Phase 7 item 3) use `dp`-mix at the frozen `Ta` = 8. `dp`'s
  failures there are lateral (`off_pad`) and do not depend on sea state, so do not read them as
  multimodality effects without checking.
- **Phase 8 risks, to plan now.**
  - The binding GPU-acceptance test for `dp` (≤ 3/150 flips) is likely to fail. Budget the CPU
    path: 170 ms per K = 10 call, and the `Ta` = 1 and ensembling cells replan every step.
  - Sweeps ran 30–34 min per seed on `select`, against an estimate of 15–25.
- **`load_student(weights="raw")`** now works (`c83019c`); it is used for diagnostics only.
  `selection.json` now records `inference`.

### P7-D1 — `modular` implementation choices P3-D1 leaves open; leak review (2026-11-02)
*Status: frozen 2026-11-02. This entry was written before any `modular` or ablation training run,
and before any `select` number from them was read.*

*Decision.* None of the following is fixed by `modular.yaml` (`84153e6d…eadc`, unchanged) or by
P3-D1 §3. They are implementation choices within the frozen protocol, not deviations.
1. **Estimator.** `ila.policies.modular.ModularEstimator` is built as follows:
   - the shared encoder (the same `EncoderConfig` as `bc`);
   - a 2×64 proprio MLP;
   - a 2×256 head that outputs the 11-d `priv` estimate, normalised by the mean and std of the
     training slice (`ppo_2000` first 1 000, success-filtered). Those statistics are stored as
     checkpoint buffers and in `manifest["normaliser"]`.

   `forward(images, proprio, crop_generator=None)` has **no clock argument**. Two tests assert
   this: `tests/test_modular.py` and `tests/test_step_guard.py`.
2. **Clock path** (P3-D1 §3, leak RISK 2).
   - `time_fraction` enters as `ila.obs.split.ModularInput(student_input, time_fraction[To])`.
     That object is built only by `ObsContract.modular_input` on the runner's `StudentWindow`,
     through `push_modular`.
   - `ModularStudent.act_modular` gives the estimator `student_input` only. It then builds the
     teacher's 25-d observation with `assemble_teacher_obs(contract, proprio[t], estimate,
     time_fraction[t])`, placing each block by P8 field name.
   - On a real `train`-pool episode, true inputs reproduce the env observation bit for bit
     (`tests/test_obs_split.py`).
   - The frozen `T_ppo` uses P8's VecNormalize with `training=False`.
   - `act_vision` and `act` raise.
   - When all 11 privileged env fields are set to NaN, the action is bit-identical. This holds
     through the runner and with the real teacher.
3. **Batched GPU path refuses `modular`.** `batched.PerSlotPolicy` raises `TypeError` for it,
   because `SlotRequest` has no clock field. Adding one would widen the clock's route to every
   batched student. So `modular` flies the in-worker CPU fp32 single-sample path. That covers
   selection, the smoke and the estimation study, and Phase 8 unless a reviewed change says
   otherwise.
4. **Loss and training.** The loss is Huber (δ = 1.0) on normalised targets, as `modular.yaml`
   specifies. Everything else is `modular.yaml`'s and shares `bc`'s loop: EMA, checkpoints every
   5 k, held-out loss on `loop.HELDOUT`, and the P3-D1 §4 selection rule over the 6 EMA
   checkpoints.
5. **`modular` smoke** (`make train-smoke ARM=modular` → `results/phase7/smoke_modular.json`).
   - It overfits the first 8 `ppo_2000` `train` episodes for **8 000 steps**, the `bc` smoke
     count, with `modular.yaml`'s other settings.
   - It passes only if all of these hold:
     - replay of the stored images, proprio, labels and `priv` for steps 0–3 is byte-identical
       to a fresh teacher rollout;
     - the EMA estimator's MSE is **< 1e-3** in normalised target units (CPU, fp32, centre
       crop), averaged over frames and the 11 dims;
     - the arm lands ≥ 7/8 `success` closed loop on the CPU single-sample path.
   - The 1e-3 threshold is the `dp` smoke's, applied to the estimator target.
6. **Ablations.**
   - The 11 ablation arms are registered (`tag="ablation"`), and each config's `obs_variant` is
     carried from training to evaluation:
     - proprio is 14-d for `bc_plus_clock`;
     - proprio is 10-d for both `*_minus_last_action`;
     - proprio is 13-d otherwise.
   - All 12 config SHA-256s match P3-D1 §3.
   - A selection row's `teacher` column now names the arm's training teacher (`T_pid`, `T_mix`).
   - **The held-out loss of `bc_pid`, `dp_pid`, `bc_mix` and `dp_mix` is still scored against
     `ppo_2000` labels**, because their stores have no episodes left to hold out. Those curves
     measure distance to `T_ppo`, not fit. They are diagnostic only; selection reads `select`.
7. **Leak review** (privileged-leak-auditor, 2026-11-02, before any training): **no LEAK, no
   RISK**, 4 NOTEs.
   - The `AUDIT_ALLOWLIST` entry `train/modular_targets.py: {estimator_targets, audit_field}` is
     accepted:
     - `estimator_targets` is the training target;
     - `audit_field` gives the closed-loop truth for the estimation study, used for logging only.
   - A name scan restricts `modular_targets` and `modular_eval` to `loop.py`, `smoke.py` and
     `modular_eval.py`.
   - A mutation test (`height` planted in proprio) is caught by all four poison tests.
   - The NOTEs and their dispositions:
     1. The ablation poison test now sets `priv` to NaN for the non-modular arms, and asserts a
        3-d target and no `EstimatorDataset`. Fixed.
     2. `modular_eval` was added to the name scan. Fixed.
     3. `ObsContract.split()` is not in the audit name list. This predates the change. No
        restricted file calls it, and the runtime poison tests cover it. It is deferred to the
        next `ila/obs` change.
     4. The first `modular` smoke ran on the pending code on a dirty tree. It used only
        `ppo_2000` episodes 0–7. It is re-run on the committed tree before training.

### P7-D2 — Deviation: data for `modular`'s estimation-error study (2026-11-02)
*Decision* (user, 2026-11-02). Plan Phase 7 item 1 asks for the per-field estimation error "on
held-out `tune` demos". No `tune` demo store exists, and P3-D1 §2 restricts `select` to named
uses. The study uses instead:
- **open loop:** `loop.HELDOUT`, which is `ppo_2000` stream indices 1950–1999 (`train` pool,
  outside `modular`'s first 1 000, never trained on by `modular`);
- **closed loop:** the selected checkpoint of each seed, re-flown on the 150 `select` episodes on
  the CPU fp32 single-sample path. The re-flown rows must reproduce the selection rows. True
  `priv` is logged per step through the reviewed audit path and never reaches the student.

*Reason.* Collecting a `tune` demo set would be a new collection and a new use of `select`. The
closed-loop part is a diagnostic that chooses nothing: no checkpoint, no hyper-parameter and no
threshold depends on it.

*Consequence.*
- Outputs: `results/phase7/modular_estimation.{csv,json}`. Errors are per field, per axis and as
  a norm, with RMSE and p95. They are in model scale (λ = 1/25), with full scale stated.
- `rel_position` and `height` are compared with P8's σ_p = 0.04 m. `rel_velocity` is compared
  with σ_v = 0.2 m/s.
- The open-loop numbers are on `train`-pool realizations. That is stated with them.

### P7-D3 — `modular` and ablation run verdicts, selection, estimation error, H2 plots, budget (2026-11-04; revised 2026-11-04 after the results-skeptic review)
*Decision.* All 38 Phase 7 runs are kept: no exclusions, no re-selection, nothing tuned.
- **Selection.** Every run was trained and selected under P3-D1 §4: the 150 `select` episodes
  (`07ecf049…`), EMA weights, the CPU fp32 single-sample path, and for `dp` arms `Ta` = 8, K = 10
  with no ensembling.
- **Scope of the numbers.** All are simulation results at model scale (λ = 1/25), in-sample on
  `select`. No frozen list was flown.
- **Verdicts** are from the `training-run-diagnostician`.
- **Tunnelled episodes** are reported per Gate 4 as the selected rows' `n_tunnelled` /
  `n_detectors_disagree`, summed over seeds.

| Arm (teacher) | Seeds | Verdict | Selected `select` success | Candidate mean | Tunnelled / disagree (selected) | Main reason |
|---|---|---|---|---|---|---|
| `modular` (`T_ppo`) | s0–s4 | TRUSTWORTHY ×5 | 143 / 142 / 144 / 145 / 145 | 139.9 | 2 / 2 | |
| `bc_mix` (`T_mix`) | s0–s2 | s0 CAVEAT; s1, s2 TRUSTWORTHY | 143 / 146 / 145 | 140.8 | 2 / 0 | s0: 5 k pick by 2 episodes; averaging policy |
| `bc_pid` (`T_pid`) | s0–s2 | CAVEAT ×3 | 146 / 143 / 143 | 122.9 | 3 / 0 | success falls with training; every pick at 5 k |
| `bc_n100` (`T_ppo`) | s0–s2 | CAVEAT ×3 | 117 / 122 / 125 | 106.7 | 1 / 0 | every pick at 5 k; closed-loop decline |
| `bc_n300` (`T_ppo`) | s0–s2 | TRUSTWORTHY ×3 | 142 / 138 / 143 | 137.4 | 4 / 0 | |
| `bc_minus_last_action` (`T_ppo`) | s0–s2 | TRUSTWORTHY ×3 | 147 / 148 / 146 | 141.8 | 1 / 0 | |
| `bc_plus_clock` (`T_ppo`) | s0–s2 | CAVEAT ×3 | 148 / 144 / 144 | 139.8 | 0 / 0 | clock reliance (item 5) |
| `dp_mix` (`T_mix`) | s0–s2 | TRUSTWORTHY ×3 | 112 / 121 / 109 | 104.9 | 16 / 1 | investigated (item 6) |
| `dp_pid` (`T_pid`) | s0–s2 | TRUSTWORTHY ×3 | 143 / 145 / 142 | 132.5 | 0 / 0 | investigated (item 6) |
| `dp_n100` (`T_ppo`) | s0–s2 | CAVEAT ×3 | 66 / 65 / 68 | 51.5 | 22 / 4 | overfits; picks +11 to +19 above candidate mean |
| `dp_n300` (`T_ppo`) | s0–s2 | CAVEAT ×3 | 63 / 62 / 63 | 52.2 | 30 / 0 | overfits; picks +9 to +13.5 above candidate mean |
| `dp_minus_last_action` (`T_ppo`) | s0–s2 | TRUSTWORTHY ×3 | 116 / 118 / 116 | 105.0 | 22 / 2 | investigated (item 7) |
| *core `bc`, Phase 4 (`T_ppo`)* | s0–s4 | — | 146 / 146 / 147 / 145 / 149 | 144.3 | — | reference |
| *core `dp`, Phase 6 (`T_ppo`)* | s0–s4 | — | 63 / 66 / 68 / 67 / 62 | 55.9 | — | reference |

*Sources.*
- **Runs:** `results/phase7/runs_summary.csv`, `results/phase7/runs_candidates.csv`, and
  `results/phase7/runs/<arm>/s<n>/{manifest.json, selection.json, select/*_rows.csv.gz}`.
- **Candidate means and every pooled rate, band and paired difference:**
  `results/phase7/diagnostics/ablation_summary.json` (`results/phase7/scripts/ablation_summary.py`,
  committed files only).
- **Held-out and training-loss curves:** `results/phase7/diagnostics/heldout_curves.json`
  (`results/phase7/scripts/heldout_curves.py`, extracted from the runs' `train_log.jsonl`).
- **Scratch-only figures** are marked *(not committed)*. They are from the diagnostician's
  working files: `results/phase7/diagnostics/verdicts/` holds their scripts, but those read
  `artifacts/` and do not re-run from a clone.

*Paired differences in this entry* are per-episode success differences against core `dp` on the
same 150 `select` episodes. The 95 % intervals are percentile bootstraps over episodes only (rng 0,
20 000 replicates). They are conditional on these seeds and checkpoints, leave out seed variance
(the dominant term), and are **not P3-D1's paired bootstrap**. They are descriptive.

*Findings.*
1. **Provenance.**
   - Every config SHA-256 equals P3-D1 §3 (`FROZEN_CONFIG_SHA256`), every tree was clean, and P8
     was at `7237b70`. There were no resumes, and every checkpoint SHA-256 was recomputed and
     matches.
   - `modular` s0 trained at `a10bc78` and the other 37 runs at `d39cd4b`. The code between the two
     commits adds only `ila.eval.h2_select` and `ila.eval.metrics.wasserstein_1`, which training
     and selection do not use. All 38 selections ran at `d39cd4b`.
   - **Schema gaps:**
     - `selection.json` does not record the `ila` commit; it is taken from the queue order and the
       manifests;
     - `dp_*` rows files leave `Ta`/`K` empty; `Ta` = 8, K = 10 is recorded in `selection.json`
       (`inference`).
   - **Demo counts** (manifests): `ppo_2000` first 1 000 (999 kept), `n100` 100, `n300` 300,
     `mix_1000` 978 kept, `pid_1000` 998 kept.
   - **Frame counts at equal steps:**
     - `ppo_2000` first 1 000: 61 518 frames;
     - `mix_1000`: 94 749 frames (1.54×);
     - `pid_1000`: 151 817 frames (2.47×).
   - **Obs variants** were checked on the checkpoint and the selection flights: `plus_clock` 14-d;
     `minus_last_action` 10-d (`proprio_mlp.0.weight` (64, 10)).
   - **File names.** Rows files are named `s0_step…`; the `s0` is the training *stage*, not the
     seed (`ila/train/loop.py`).
   - **Selection check.** The skeptic re-applied the P3-D1 §4 rule from `runs_candidates.csv` for
     38/38 runs: 0 mismatches, 5 picks decided by the p95 tie-break.
2. **Selection is noise-dominated wherever outcomes sit near a boundary.**
   - Picks sit above the run's candidate mean by:
     - +1.5 to +7.0 for `modular` and the TRUSTWORTHY `bc` ablations;
     - +9 to +19 for `dp_n100`, `dp_n300` and `dp_minus_last_action`, around what the maximum
       of 10 binomial draws gives (about +8.5 to +9). The top of that range also reflects a
       training-step trend (item 8);
     - +14.7 to +28.3 for `bc_pid`, a trend (item 4).
   - **Report both measures:** selected scores (what Phase 8 flies) and candidate means. The means
     mix overfit late checkpoints (`*_n100`, `*_n300`) with immature early ones (`dp_pid`). Expect
     Phase 8 to land below the selected `select` scores (as P6-D5 item 2).
3. **`modular` (estimator → frozen `T_ppo`) lands below its teacher, and below `bc` (unpaired).**
   - **Success.** Selected 142–145/150. The candidate mean is 139.9, against `bc`'s 144.3; that is
     an unpaired reading, since Phase 4 rows are not committed.
   - **Pooled over all candidates:** success by SS3/SS4/SS5 is 97.1 / 94.3 / 88.4 %, hard landings
     5.0 %, 0 crashes.
   - **Teacher.** `T_ppo` lands 149/150 on `select` (`results/phase7/diagnostics/clock_intervention.csv`,
     `T_ppo`, `none`).
   - **Estimation error** (`results/phase7/modular_estimation.{csv,json}`, the five selected
     checkpoints, P7-D2). Ranges over seeds, closed loop on `select` unless stated, model scale
     (×25 for lengths, ×5 for velocities at full scale):

     | Field | Open-loop RMSE (held-out `train`-pool demos) | Closed-loop RMSE | Closed-loop p95 \|error\| | Closed-loop SS5 p95 |
     |---|---|---|---|---|
     | `rel_position` x / y / z | — | 0.018–0.022 / 0.014–0.016 / 0.010–0.012 m | 0.034–0.046 / 0.029–0.036 / 0.021–0.024 m | — |
     | `rel_position` (norm) | 0.0130–0.0136 m | 0.0245–0.0287 m | 0.049–0.057 m | 0.056–0.072 m |
     | `height` | 0.0063–0.0076 m | 0.0096–0.0117 m | 0.021–0.025 m | 0.025–0.030 m |
     | `rel_velocity` z | 0.054–0.057 m/s | 0.119–0.132 m/s | 0.252–0.292 m/s | 0.313–0.367 m/s |
     | `rel_velocity` x / y | — | 0.008–0.010 / 0.006–0.007 m/s | 0.017–0.020 / 0.011–0.014 m/s | — |
     | `rel_tilt` | 0.56–0.63° | 0.78–0.91° | 1.52–1.62° | 2.12–2.32° |
     | `deck_normal` (norm) | 0.012–0.014 | 0.018–0.021 | 0.037–0.042 | 0.049–0.058 |

     - **Units.** The CSV's norm ratios are in units of √3·σ_p = 0.069 m (`sigma_name`).
       Per axis, the closed-loop position RMSE is 1.0–2.2 cm, and v_z RMSE is 0.12–0.13 m/s.
     - **Against P8's white-noise arm** (`external/rl-land-on-moving-deck/results/e07/noise/`,
       latency 0, `id` SS3–SS6):
       - at σ_p = 2 cm / σ_v = 0.10 m/s, `T_ppo` (P8 `ppo` seed 0, the teacher `modular` feeds)
         lands 99 / 99 / 99 / 97 %, and `pid_feedforward` 60 / 66 / 56 / 59.5 %;
       - at σ_p = 4 cm / σ_v = 0.20 m/s, `T_ppo` lands 70.5 / 75 / 71.5 / 66.5 %, and
         `pid_feedforward` 0 / 0 / 0 / 0.5 %.
     - **Where the estimator falls:** at or slightly above the 2 cm / 0.10 m/s condition (v_z is
       above 0.10), and below 4 cm / 0.20 m/s. The tails are below P8's white-noise per-axis p95
       of 1.96σ at 4 cm (0.078 m, 0.39 m/s) in every cell. The comparison is not like for like:
       the estimator's error is state-dependent, not white noise.
     - **Bias is small:** |bias| ≤ 1.5 mm in position and ≤ 6.6 mm/s in v_z. Time correlation was
       not measured.
     - **Open vs closed loop.** Closed-loop error exceeds open-loop error by 1.9–2.4× for
       `rel_position` and v_z, and by 1.4–1.5× for height, tilt and normal. The estimator sees its
       own states, a distribution shift. Error grows with sea state, and v_z is the worst field.
     - This is consistent with `modular`'s hard landings (0.50–0.63 m/s) and its success falling
       with sea state.
   - **Re-fly check.** Every closed-loop re-fly reproduced its selection rows
     (`rows_match_selection` true).
4. **BC ablations.**
   - **`bc_pid`: success falls with training.**
     - Pooled successes by step, 5 k → 30 k: 432 / 411 / 365 / 341 / 338 / 326 of 450.
     - Hard landings rise over the same steps: 12 → 98 (`runs_candidates.csv`).
     - The copycat proxy (d a / d `last_action`) rises with step *(not committed)*, while `T_pid`
       never reads `last_action`.
     - Every pick is at the 5 k grid boundary, so `bc_pid` is an early-stopped student. The
       `bc` vs `bc_pid` comparison confounds teacher with effective training length.
     - Its held-out loss is scored against `ppo_2000` labels (P7-D1 item 6), so it measures
       distance to `T_ppo`.
   - **`bc_n100`.**
     - It memorises: final train loss 1.5–1.8e-6, against held-out minima of 1.02–1.07e-3 that
       stay flat to 30 k.
     - Closed-loop success falls from 364 to 303 of 450, 5 k → 30 k, while `off_pad` rises from
       49 to 117.
     - Timeouts are most frequent at the selected 5 k checkpoints (19 of 450, against 5–15 later).
       That is a Phase 8 watch item.
     - Open-loop MSE did not anticipate the decline.
   - **`bc_minus_last_action`.** Held-out minima are 5.4–5.7e-4, about twice `bc`'s 2.6–2.8e-4,
     yet the selected scores (146–148) are at `bc`'s level (unpaired).
   - **`bc_mix`.**
     - s0 touchdown p50 is 2.62 s, between the teacher's modes.
     - Per-seed touchdown p50 is 1.95–2.62 s, with 0.36–0.54 of touchdowns in the reference's mode
       gap (`results/phase7/h2/h2_touchdown_select.json`, `per_seed`).
     - The diagnostician found no assignment-dependent split *(not committed)*.
5. **`bc_plus_clock` relies on `time_fraction` to a degree that varies by seed** (closed-loop clock
   intervention on `select`, `results/phase7/diagnostics/clock_intervention.{csv,json}`, P7-D4; a
   diagnostic that chooses nothing).

   | Subject | `select` success: true clock / + 1 s / held at its mean / clamped at 1.6 s | Touchdown shift, paired mean (const_mean / shift) |
   |---|---|---|
   | s0 | 148 / 133 / 145 / 148 | +0.77 s / +0.01 s |
   | s1 | 144 / 40 / 98 / 144 | — / −0.71 s |
   | s2 | 144 / 85 / 145 / 146 | +0.56 s / — |
   | `T_ppo` | 149 / 150 / 148 / 149 | — / touchdown p50 1.53 → 1.39 s |

   - **s1** depends strongly on the clock in both perturbations.
   - **s0 and s2** lose 15 and 59 successes under the 1 s shift, and change by −3 and +1 when the
     clock is held at its mean. Holding it delays their touchdown.
   - **The clock buys no fit:** held-out minima are 2.63–2.84e-4, against `bc`'s 2.61–2.82e-4
     (`heldout_curves.json`).
   - **`T_ppo` also moves under the shift**, though its success does not drop.
   - **Consequence for H4.** P′ = S(`bc_state`) − S(`bc_plus_clock`) compares against a student
     that relies on the clock to a seed-dependent degree. `bc_state`'s own clock reliance was not
     measured, so this caveat is one-sided. State both with P′ in Phase 8.
6. **`dp_mix` and `dp_pid` far exceed `dp`. Slow descent alone does not explain it.** CLAUDE.md
   required an investigation before these numbers were reported. Touchdown-time bands are fast
   < 1.8 s and slow > 2.8 s (model scale), pooled over all candidates.

   | Arm | Fast / between / slow share | Success fast / between / slow | Lateral p50 / p95 | `off_pad` | Paired vs `dp`, all candidates | Paired vs `dp`, selected |
   |---|---|---|---|---|---|---|
   | `dp` | 74.2 / 23.6 / 2.2 % | 38.4 / 34.4 / 33.5 % | 0.089 / 0.186 m | 41.5 % | — | — |
   | `dp_mix` | 36.7 / 21.0 / 42.3 % | 48.8 / 65.0 / 90.9 % | 0.043 / 0.148 m | 16.4 % | +0.327 [0.296, 0.356] | +0.325 [0.270, 0.380] |
   | `dp_pid` | 0 / 0.4 / 99.6 % | — / 60.0 / 88.5 % | 0.029 / 0.068 m | 0.4 % | +0.511 [0.486, 0.534] | +0.521 [0.480, 0.561] |

   - **Both arms beat `dp` in every band they occupy**, and `dp`'s own slow touchdowns succeed no
     more often than its fast ones (33.5 % against 38.4 %). Shifting touchdowns toward slower bands
     explains:
     - for `dp_mix`, about 50 % of its touched-down gain at its own band rates and −6 % at `dp`'s
       rates;
     - for `dp_pid`, −8 % at `dp`'s rates. At `dp_pid`'s own rates it exceeds the whole gain
       (142 %), because `dp_pid`'s rare between-band episodes succeed less often than its slow
       ones. So the slow-descent share is not identified.

     Source: `band_decomposition_vs_dp`. So the gain is **partly** associated with slow
     touchdowns, and is not explained by them.
   - **The gain is confounded with the teacher's data:**
     - `T_pid`'s and `T_mix`'s labels;
     - 1.54× and 2.47× the frames at equal steps (item 1).

     `dp_pid` matches `bc_pid`: selected rows land 95.6 % against 96.0 %. The 8 failing episodes
     they share are all at SS5 *(not committed)*. This points to the teacher's data, not to DP.
   - **No late decline in `dp_pid`**, unlike `bc_pid`. Pooled success rises from 350/450 at 10 k
     to 426/450 at 70 k (`runs_candidates.csv`).
   - **Mode choice in `dp_mix`.** On the same episode, slow vs fast agrees 0.84 within a seed
     across checkpoints and 0.54 across seeds (chance about 0.51; skeptic's recomputation from
     committed rows). That is *consistent with* the per-episode noise key setting the mode. The key
     contains `run_seed`, so this cannot be separated from the seed's network.
   - **Tunnelling does not explain the gains.** Tunnelled successes are 36 of 3 148 (`dp_mix`) and
     0 of 3 975 (`dp_pid`), against 131 of 2 796 for `dp`.
   - **Neither beats its teacher.**
     - `T_mix` lands 146/150 on `select` (`results/phase7/h2/tmix_select_rows.csv`). `bc_mix` s1
       ties it at 146.
     - `dp_pid` by sea state is 99.3 / 98.7 / 88.7 %, against `T_pid`'s 100 / 99.7 / 99.7 % on
       `train` (P3-D1 ceilings). That is an *unpaired, cross-pool reading*: clear at SS5, within
       about 1 point at SS3/SS4.
     - `T_pid` was not flown on `select` (item 10). Phase 8 repeats the check against P8's
       committed `pid_feedforward` rows at `id` SS4–SS6.
7. **`dp_minus_last_action` nearly doubles `dp`, through lateral precision, not slow descent.**
   CLAUDE.md required an investigation before this was reported.
   - **Same touchdown timing.** Band shares equal `dp`'s (74.1 / 23.8 / 2.1 %), and the band
     composition explains 0 % of the gain.
   - **Inside the fast band:**
     - success 72.2 % against 38.4 %;
     - lateral p50 0.050 against 0.090 m;
     - closing p50 0.302 against 0.394 m/s.
   - **All candidates:** `off_pad` 12.3 % against 41.5 %; hard landing 8.3 % against 13.9 %.
   - **Paired per-episode success vs `dp`:**
     - all candidates +0.327 [0.304, 0.350] (145 episodes better, 4 worse, 1 equal);
     - selected +0.343 [0.295, 0.389].
   - **Candidates and tunnelling.** The worst candidate (81) beats the best `dp` candidate (68),
     and success rises with training. Tunnelled successes are 75 of 3 150, against 131 of 2 796 for
     `dp`, so tunnelling does not explain the gain.
   - **Removing `last_action` costs `dp` no denoising fit.** Held-out EMA minima are 0.035–0.040 at
     44–48 k, against `dp`'s 0.035–0.039 at 41–45 k (P6-D5). `bc` pays about a twofold open-loop
     cost for the same removal and gains nothing in closed loop (item 4).
   - **Reading.** The evidence is *consistent with* a `last_action` copycat channel that carries
     stale chunk-end actions into the next chunk at `Ta` = 8.
     - In open-loop replay on held-out demos, feeding `dp` its own executed previous action raises
       lateral RMSE by 29–38 % across `dp` s0–s4
       (`results/phase7/diagnostics/verdicts/dp_minus_last_action/ol_out/ol.json`). The variant has
       no such channel.
     - This fits `dp` recovering at `Ta` = 1 (P6-D5 item 5).
     - The channel was not tested closed loop, and `Ta` = 1 was not flown for the variant (that
       would be a new use of `select`).
   - **`T_ppo` still beats it:** 149/150 against a best candidate of 118.
   - **Not a code change.** `dp` code is unchanged since `4ae5e3b`, and `dp_n100` and `dp_n300`,
     trained and selected at the same commit, still land 62–68.
8. **The `dp` data axis (100 → 300 → 1 000 demos) is hidden by the `Ta` = 8 ceiling and by
   selection.**
   - **Candidate means:** 51.5 → 52.2 → 55.9.
   - **Paired per-episode success vs `dp`:**
     - all candidates: `dp_n100` −0.029 [−0.052, −0.006], `dp_n300` −0.025 [−0.047, −0.003];
     - selected only: `dp_n100` +0.008 [−0.048, 0.064], `dp_n300` −0.017 [−0.078, 0.042].

     The selected-only intervals include 0. The all-candidate ones exclude it, but that is driven
     by overfit late checkpoints that Phase 8 will not fly.
   - **Overfitting comes earlier with fewer demos:**
     - held-out EMA minimum at 6 k → 13–14 k → 41–45 k;
     - rises by 100 k of 3.46–3.94× and 3.40–3.90× for the two small arms, against 1.5–1.7× for
       `dp`;
     - success falls with training step for both small arms.
   - **Fewer demos land with more sideways error but more gently:**

     | Demos | `off_pad` | Lateral p50 | Closing p50 | Hard landing |
     |---|---|---|---|---|
     | 100 | 54.3 % | 0.109 m | 0.279 m/s | 2.2 % |
     | 300 | 52.1 % | 0.104 m | 0.324 m/s | 4.4 % |
     | 1 000 | 41.5 % | 0.089 m | 0.400 m/s | 13.9 % |

   - **Wording.** Do not claim that `dp` is insensitive to data size, nor that it needs more data.
     Neither is shown at `Ta` = 8.
9. **H2 touchdown-time plots on `select`** (plan item 3; plotted, never scored;
   `results/phase7/h2/h2_touchdown_select.{png,csv,json}`).
   - **Reference.** `T_mix` lands 146/150. Its `ppo` and `gated` halves have touchdown p50 1.53 s
     and 3.48 s, and 9.6 % of its touchdowns fall in the gap between the modes.
   - **Students.** The share of touchdowns in that gap is 0.26–0.34 per seed for `dp_mix` and
     0.36–0.54 for `bc_mix`. Pooled W₁ to the reference is 0.31 s (`dp_mix`) and 0.49 s (`bc_mix`).
   - `dp_mix` is closer to committing to one mode than `bc_mix`, but neither reproduces the
     reference's separation.
   - **Different windows.** The diagnostician's "15–20 % between modes" used the fixed 1.8–2.8 s
     band *(not committed)*, a different window from the plot's per-cell gap. Both are
     descriptive and post hoc.
   - Seeds are shown separately because pooling mixes regimes.
   - **H2 pre-look.** The scored W₁ function has now been run on `select`, and its direction is
     written here. The README must not cite these `select` W₁ values as H2 evidence.
   - **Caveat.** `dp_mix`'s gain over `dp` is not shown to come from multimodality (item 6).
10. **Uses of `select` beyond selection** (none chose anything):
    - **Committed:**
      - the `T_mix` teacher flights for the H2 reference (`tmix_select_rows.csv`, 2026-11-03);
      - the `modular` closed-loop estimation re-fly (P7-D2, 2026-11-03);
      - the reviewed clock intervention (P7-D4, 2026-11-03).
    - **Scratch, not committed:**
      - the diagnostician's scratch clock wrapper on `bc_plus_clock` s0–s2 and `T_ppo` (2026-11-03).
        It read rows and preceded, and motivated, the reviewed path; its numbers are superseded by
        P7-D4;
      - the diagnostician's `T_pid` flight (2026-11-04). It hung and was stopped; no rows were
        read.
    - Earlier text cited a 1-ulp clock-shift sensitivity. It came from a scratch run and is
      withdrawn.
11. **Budget** (wall-clock GPU-h, `runs_summary.csv`).

    | Arm | GPU-h |
    |---|---|
    | `modular` | 3.87 |
    | `bc` ablations, 6 arms | 2.26–2.35 each, 13.70 in total |
    | `dp_mix` | 6.52 |
    | `dp_pid` | 6.72 |
    | `dp_n100` | 6.83 |
    | `dp_n300` | 6.70 |
    | `dp_minus_last_action` | 6.64 |
    | **Phase 7** | **50.98** |
    | **Cumulative** (44.43 through Phase 6) | **95.41 of 120** |

    The total is in line with P4-D2's projection of about 95. No cut was made.
12. **Tooling.** `dp_diagnostics` read held-out windows under the default contract. It now passes
    the checkpoint's `obs_variant` (`decc00a`, `tests/test_diagnostics_variants.py`). No committed
    number depended on the old path.

*Consequence for Phase 8.*
- **H3.** `dp_minus_last_action` (item 7) is a competing explanation for `dp` recovering at
  `Ta` = 1: an input-freshness effect, stale `last_action`, alongside chunk length. H3's statistic,
  S(`Ta` = 1) − S(`Ta` = 8), is stated with this confound, and any README sentence that says `dp`
  fails "because of chunking" must name it. The scoring rule does not change.
- **H4.** Report P′ with item 5's seed-dependent clock-reliance caveat, and with the fact that
  `bc_state`'s clock reliance was not measured.
- **H2.** Score only at `id` SS4 with seeds pooled (P3-D1). State that `dp_mix`'s gain over `dp` is
  not shown to come from multimodality (items 6 and 9).
- **Teacher checks.** `dp_pid`/`bc_pid` against `pid_feedforward` on P8's committed rows at
  `id` SS4–SS6.
- **Watch items:** `bc_n100`'s 5 k timeouts; detector disagreements in the `dp_*` arms (4 in
  `dp_n100`).
- **Expected numbers.** Expect evaluation numbers below the selected `select` scores, especially for
  every `dp` arm.
- **README.** It must state:
  - the `dp_minus_last_action` finding, worded "consistent with";
  - `modular`'s teacher gap;
  - where its estimation error falls relative to P8's noise arm (item 3).

*Review response (results-skeptic, 2026-11-04; no BLOCKER).*
- **S1.** Item 3 was rewritten:
  - norm ratios are labelled in √3·σ_p units, and per-axis errors are added;
  - the comparison is now against P8's 2 cm and 4 cm `T_ppo` and `pid_feedforward` rows, with
    tails against 1.96σ;
  - bias is quantified, and the time-correlation claim was dropped.
- **S2.** Item 6 now carries both decompositions, the frame counts, and "consistent with; not
  separated from seed" for mode choice. "About 70 %" was withdrawn.
- **S3.** Selected-only paired differences are printed beside all-candidate ones, with the
  conditioning stated.
- **S4.** The H3 confound and the one-sided H4 caveat were added to the consequences.
- **S5.** Item 10 now lists the scratch uses of `select`, and the 1-ulp figure was withdrawn.
- **S6.** Held-out and training-loss figures now trace to `heldout_curves.json`. The rest are
  marked *(not committed)*.
- **M1–M10.**
  - Number fixes (M1): +1.5 to +7.0; 29–38 %; per-field open/closed ratios; `bc_n100` timeouts;
    all 8 shared episodes at SS5.
  - Tunnelled and disagree counts were added to the table, and tunnelled successes to items 6
    and 7 (M2).
  - P7-D4 NOTE 1 is closed only by the rename commit (M3).
  - Unpaired labels were added (M4, M5).
  - Per-seed clock wording (M6).
  - Both selected scores and candidate means are reported (M7).
  - Schema gaps were added (M8).
  - The H2 pre-look was noted (M9).
  - The commit-difference wording was fixed (M10).

### P7-D4 — Clock-intervention diagnostic on `select`; leak review of the Phase 7 merges (2026-11-04)
*Decision.* The closed-loop clock intervention is recorded as a named diagnostic use of `select`,
following P7-D2's precedent. It **chooses nothing and is not scored**.
- **Subjects:** `bc_plus_clock` s0–s2 at their selected checkpoints, and `T_ppo`.
- **Scope:** the 150 `select` episodes, CPU fp32 single-sample path.
- **What changes:** only the `time_fraction` input value. Everything else is identical.
- **Four modes:**
  - `none`, the true clock;
  - `shift_s`, `clip(time_fraction + 1 s / 12 s)`;
  - `const_mean`, the training mean, 0.0843 (1.011 s);
  - `clamp`, `min(time_fraction, 1.6 s / 12 s)`.
- **Check:** `none` reproduces each student's selection rows and the bare `T_ppo` rows exactly.
- **Code and results:**
  - `ila.train.clock_diagnostics` and `scripts/clock_intervention.py`, added at `33b6e0c`;
  - `results/phase7/diagnostics/clock_intervention.{csv,json}` and
    `clock_intervention_episodes.csv.gz`, added at `d20669b`.
- **New `ila/obs/split.py` helpers** (`ObsContract.proprio_clock`, `with_proprio_clock`,
  `with_obs_clock`) replace the named `time_fraction` column in a *copy*. They refuse any variant
  other than `plus_clock` for students, and they refuse `modular`.

*Reason.* The diagnostician's scratch wrapper suggested that `bc_plus_clock` times its descent on
the clock. H4's P′ depends on that student, so the finding had to come from a reviewed, committed
path before anyone cites it (P7-D3 item 5).

*Timing.* 23 of the 38 Phase 7 selections predate `33b6e0c`. The 15 `dp_*` selections came later
but cannot use it: the diagnostic refuses their variants, and `select.py`, `loop.py` and
`configs/train/` are unchanged since `eaec162`.

*Leak review* (privileged-leak-auditor, 2026-11-04, HEAD `278d7d4`). This covers `33b6e0c`,
`d20669b` and `decc00a`, the diagnostics `obs_variant` pass-through.
- **Result: no LEAK, no RISK.** 219 focused tests pass on CPU.
- Only the clock column of a copy changes. The env observation and the images are bit-identical
  across the wrapper.
- Default, `minus_last_action` and `modular` students are refused.
- The literal-slice, audit-name and frozen-list scans cover the new files. `AUDIT_ALLOWLIST` is
  unchanged at two entries:
  - `train/state_dataset.py` (Phase 4);
  - `train/modular_targets.py` (P7-D1).
- `decc00a` routes variants through `DemoStore.variant_contract`, which refuses any
  `NEVER_STUDENT` field.

NOTEs and their dispositions:
1. P7-D1's deferred NOTE 3 is closed: `ObsContract.split()` was renamed to a distinct name and
   added to the audit name scan (see the commit that follows this entry).
2. This entry records the diagnostic.
3. `ClockInterventionTeacher` now refuses a student or `modular` inner policy (hardening).
