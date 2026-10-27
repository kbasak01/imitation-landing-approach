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
