# Third-party notices

This project is MIT-licensed (see [LICENSE](LICENSE)). It depends on Project 8 as a git submodule,
which in turn pins two submodules of its own, and on other packages at runtime. No third-party
source code is copied into this repository as of Phase 0. The per-dependency licence check is in
[`docs/LICENSES.md`](docs/LICENSES.md).

## Git submodule (`external/`, never modified)

### rl-land-on-moving-deck (P8)

- **Upstream:** <https://github.com/kbasak01/rl-land-on-moving-deck>, pinned at
  `7237b7065c8af8172b3848ac6e9d3ceb2cf82075` (merge of PR #9, "Gate 9 passed"). The upstream has
  no release tag (checked with `git ls-remote --tags` on 2026-10-27).
- **License:** MIT, Copyright (c) 2026 K. Basak. Its own `LICENSE` and `THIRD_PARTY_NOTICES.md`
  travel with the submodule.
- **What is used:** the landing environment, configs, frozen episode lists, success criteria,
  outcome classes, statistics functions, the `pid_feedforward`/`gated` controllers and the loader
  for the PPO teacher. Everything is imported at runtime through `src/ila/p8_adapter.py`. Its
  trained PPO checkpoint is a gitignored P8 artifact that is hashed (`results/p8_provenance.json`)
  but not redistributed.

### Nested in P8 (`external/rl-land-on-moving-deck/third_party/`)

- **deck-motion-forecast (`dmf`)**: <https://github.com/kbasak01/deck-motion-forecast> at
  `e9fa15cc35312a5c1ecf0c42668a690b5061c090`. MIT, Copyright (c) 2026 K. Basak. Deck motion,
  read-only, through P8.
- **gym-pybullet-drones**: <https://github.com/learnsyslab/gym-pybullet-drones> at
  `7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (`v1.0.0-324-g7ebad1e`). MIT, Copyright (c) 2020
  Jacopo Panerati. Used through P8: the aviary, `DSLPIDControl` and the Crazyflie 2.x model that
  the camera renders.

## Code that may be adapted later (notice slot)

### diffusion_policy (reference implementation)

- **Upstream:** <https://github.com/real-stanford/diffusion_policy>.
- **License:** MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
- **Status:** Phase 0 adapts nothing from it. The Phase 6 defaults are read from its
  `train_diffusion_unet_hybrid_workspace.yaml`. If `ConditionalUnet1D` code is adapted into
  `src/ila/policies/dp/unet1d.py`, that file will be marked in the source, and this entry will
  name the upstream commit and reproduce the MIT notice in full.

## Runtime dependencies (not vendored)

These are this project's additions, pinned in [`pyproject.toml`](pyproject.toml). P8's own
dependencies are listed in P8's `THIRD_PARTY_NOTICES.md`. Licences are as declared in the
installed packages' metadata on 2026-10-27.

| package | version | licence |
|---|---|---|
| torchvision | 0.28.0 | BSD-3-Clause |
| diffusers | 0.41.0 | Apache-2.0 |
| h5py | 3.16.0 | BSD-3-Clause |
| einops | 0.8.2 | MIT |

The reference arm R1 uses **LeRobot** 0.6.1 (Apache-2.0). It is installed only in the separate
`.venv-lerobot`, never imported from `src/ila`, and not redistributed.
