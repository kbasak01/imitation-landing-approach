# LICENSES.md — dependency licences and the Phase 0 `(verify)` items

Checked 2026-10-27 on the reference workstation (WSL2, Python 3.12.3). Licences come from the
installed packages' metadata (`importlib.metadata`) or the upstream `LICENSE` file, as stated in
each row. This project is MIT. Notices and attributions are in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## 1. Licences

| Dependency | Version / pin | Licence | Source of the check | Compatible with MIT release |
|---|---|---|---|---|
| P8 `rl-land-on-moving-deck` | `7237b706…` | MIT | submodule `LICENSE` | yes |
| P4 `deck-motion-forecast` (via P8) | `e9fa15cc…` | MIT | `pyproject.toml` + `LICENSE` | yes |
| gym-pybullet-drones (via P8) | `7ebad1ec…` | MIT, (c) 2020 Jacopo Panerati | `LICENSE` | yes |
| PyBullet | 3.2.7 (P8 pin) | zlib | metadata | yes |
| PyTorch | 2.13.0+cu130 (P8 pin) | BSD-3-Clause-style. The metadata expression also lists bundled components: Apache-2.0, LLVM-exception, BSD-2, BSL-1.0, MIT | metadata | yes |
| torchvision | 0.28.0+cu130 | BSD-3-Clause | metadata ("BSD") | yes |
| diffusers | 0.41.0 | Apache-2.0 | metadata | yes (keep NOTICE if code is copied; none is) |
| h5py | 3.16.0 | BSD-3-Clause | metadata | yes |
| einops | 0.8.2 | MIT | metadata | yes |
| gymnasium / stable-baselines3 | 1.3.0 / 2.9.0 (P8 pins) | MIT / MIT | metadata | yes |
| onnx / onnxruntime-gpu | 1.22.0 / 1.29.0 (P8 pins) | Apache-2.0 / MIT | metadata | yes |
| tensorrt-cu13 | 10.16.1.11 (P8 pin) | NVIDIA proprietary | metadata | used at runtime for latency only; never redistributed |
| rliable | 1.2.0 (P8 pin) | Apache-2.0 | metadata | yes |
| huggingface-hub, safetensors (pulled in by diffusers) | 1.33.0 / 0.8.0 | Apache-2.0 | metadata / classifier | yes |
| certifi (pulled in by requests) | 2026.10.23 | MPL-2.0 | metadata | yes (unmodified, not vendored) |
| LeRobot (`.venv-lerobot` only) | 0.6.1 | Apache-2.0 | `pip show` | yes; not imported from `src/ila` and not redistributed |
| diffusion_policy (reference) | — | MIT | upstream `LICENSE` | yes. Three upstream files (`main` at `5ba07ac6`) are adapted into `src/ila/policies/dp/unet1d.py` (Phase 6); the MIT notice is in its header and in `THIRD_PARTY_NOTICES.md` |

The full resolved environments are committed as `results/env/pip_freeze_main.txt` and
`results/env/pip_freeze_lerobot.txt`.

## 2. The `(verify)` items from `SOURCES.md` and the plan

| Item | Verdict | Evidence |
|---|---|---|
| P8 release tag on `7237b70` | **Resolved: no tag exists.** The commit is pinned by SHA. | `git ls-remote --tags origin` on P8 returns nothing; `refs/heads/main` = `7237b7065c8af8172b3848ac6e9d3ceb2cf82075` |
| torchvision wheel matching torch 2.13.0 | **Resolved: 0.28.0+cu130** from the default PyPI index, the same index as P8's torch | `pip install --dry-run -c <P8 freeze>`; `torchvision.__version__` |
| diffusers 0.41.0 does not move torch or numpy | **Resolved: it does not.** With P8's environment frozen as constraints, the dry run adds only new packages (diffusers, huggingface-hub, safetensors, regex, requests stack, h5py, einops, torchvision). torch 2.13.0, numpy 2.5.2 and pybullet 3.2.7 stay put | dry-run report; `results/env/pip_freeze_main.txt` |
| h5py version | **Resolved: 3.16.0**, with lzf built in | pip resolution |
| LeRobot 0.6.1, Python ≥ 3.12 | **Resolved: installs on 3.12.3** | `.venv-lerobot` |
| LeRobot dataset format v3 | **Resolved: `CODEBASE_VERSION = "v3.0"`** in the installed package | `lerobot/datasets/dataset_metadata.py:60` |
| LeRobot coexistence with P8 pins | **Resolved: they cannot coexist.** LeRobot 0.6.1 resolves torch 2.11.0+cu130, torchvision 0.26.0, numpy 2.2.6 and gymnasium 1.4.0, all different from P8's pins. Coexistence was not attempted in `.venv`; the separate venv stands | `results/env/pip_freeze_lerobot.txt` |
| Xing et al., arXiv:2403.12203 venue | **Resolved: CoRL 2024** (PMLR vol. 270, published 2025) | <https://proceedings.mlr.press/v270/xing25a.html> |

## 3. Later phases (rows added 2026-11-09, Phase 10)

| Item | Verdict | Evidence |
|---|---|---|
| Adapted `ConditionalUnet1D` (Phase 6) | **MIT, notice kept.** `conditional_unet1d.py`, `conv1d_components.py` and `positional_embedding.py` from diffusion_policy `main` at `5ba07ac6` are adapted into one file | `src/ila/policies/dp/unet1d.py` header; `THIRD_PARTY_NOTICES.md` |
| Pretrained weights | **None used or redistributed.** The ResNet-18 trunk is randomly initialised (`weights=None`), and the encoder refuses `pretrained: true`. R1 sets `pretrained_backbone_weights: null`. No ImageNet statistics are used either: images are scaled to [0, 1] (`encoder.py` module docstring) | `src/ila/policies/encoder.py`; `configs/train/r1_lerobot_dp.yaml` |
| Release assets (student checkpoints, ONNX graphs) | **MIT**, as trained or exported by this project. They are derived from P8's MIT teacher and P8's MIT environment; no third-party weights are inside | `results/release/manifest.json` |
| Demo-store export (stretch) | **Not published.** Only a 256-frame sample could ship as a release asset; none does | — |

## 4. Still open

- None.
