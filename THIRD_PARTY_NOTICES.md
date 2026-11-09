# Third-party notices

This project is MIT-licensed (see [LICENSE](LICENSE)). It depends on Project 8 as a git submodule,
which in turn pins two submodules of its own, and on other packages at runtime. No third-party
source code was copied into this repository in Phase 0. Phase 6 adapted three upstream files of
diffusion_policy into one file (below). No pretrained weights are used or redistributed anywhere
(see `docs/LICENSES.md` §3). The per-dependency licence check is in
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

## Adapted source code

### diffusion_policy (reference implementation)

- **Upstream:** <https://github.com/real-stanford/diffusion_policy>, `main` at
  `5ba07ac6661db573af695b419a7947ecb704690f` (2024-12-24). That was the head of `main` on the
  adaptation date, Phase 6, 2026-10-30, and it still is: the last push was 2024-12-24
  (`gh api repos/real-stanford/diffusion_policy`, checked 2026-11-09). The commit was not recorded at
  adaptation time. It is resolved here as the only upstream state available then.
- **License:** MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab (full
  text below).
- **What is adapted:** `src/ila/policies/dp/unet1d.py` is adapted from
  `diffusion_policy/model/diffusion/conditional_unet1d.py`, `conv1d_components.py` and
  `positional_embedding.py`. The file carries this notice in its header and lists the changes
  (type hints and docstrings; global conditioning only; FiLM scale-and-bias fixed on; einops
  replaced by `transpose`/`unsqueeze`; a horizon divisibility check). The Phase 6 defaults are
  read from its `train_diffusion_unet_hybrid_workspace.yaml`; nothing else is copied.

```
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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

PyBullet 3.2.7 (zlib) and PyTorch 2.13.0 (BSD-3-Clause style) come through P8's pins. See
[`docs/LICENSES.md`](docs/LICENSES.md) and P8's notices.

The reference arm R1 uses **LeRobot** 0.6.1 (Apache-2.0). It is installed only in the separate
`.venv-lerobot`, never imported from `src/ila`, and not redistributed. The release's three R1 ONNX
exports were traced from LeRobot's DiffusionPolicy architecture with this project's trained
weights; no LeRobot source is redistributed.

Figure and GIF text is set in DejaVu Sans from Matplotlib's bundled copy; no font file is
redistributed.
