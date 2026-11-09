# REPO_SETUP.md — imitation-landing-approach

How to stand the repository up on the workstation (RTX A4000, i9-10980XE, WSL2 — the machine P8 ran
on). Claude Code generates `pyproject.toml`, `Makefile` and CI in Phase 0 from this file; **you run
the `pip` commands yourself.**

## 1. Create the repo and the submodule

```bash
mkdir imitation-landing-approach && cd imitation-landing-approach
git init -b main
# copy the starter kit contents in (CLAUDE.md, docs/, configs/, .claude/, *.md)
git submodule add https://github.com/kbasak01/rl-land-on-moving-deck external/rl-land-on-moving-deck
cd external/rl-land-on-moving-deck
git checkout 7237b7065c8af8172b3848ac6e9d3ceb2cf82075   # P8 release merge (PR #9, "Gate 9 passed"); use a tag on it if one exists
git submodule update --init --recursive                  # dmf + gym-pybullet-drones, P8's pins
cd ../..
git add .gitmodules external/rl-land-on-moving-deck
git commit -m "Pin P8 (rl-land-on-moving-deck) as read-only submodule"
```

## 2. Main environment (`.venv`, Python 3.12)

P8's pins are reused verbatim so its results reproduce bit for bit. Nothing here may re-resolve
torch, numpy or pybullet.

```bash
sudo apt-get install -y build-essential python3.12-dev python3.12-venv   # pybullet builds from sdist on 3.12
python3.12 -m venv .venv && . .venv/bin/activate
pip install -U pip
P8=external/rl-land-on-moving-deck
pip install -e $P8/third_party/deck-motion-forecast      # must be editable (dmf resolves configs from its tree)
pip install -e $P8/third_party/gym-pybullet-drones
pip install -e "$P8[dev]"                                  # P8 itself, with its exact pins
pip install -e ".[dev]"                                    # this repo; its own pins only ADD packages
python -c "import rld, dmf, gym_pybullet_drones, torch; print(torch.__version__)"   # expect 2.13.0
```

`pyproject.toml` (Phase 0) — dependencies **in addition to** P8's resolved set, pinned to what
resolves on the reference machine (record in protocol P0-D1):

| package | why | note |
|---|---|---|
| `torchvision` | ResNet-18 trunk | the build matching torch 2.13.0 / the same CUDA wheel index as P8 |
| `diffusers` | DDPM/DDIM schedulers only | 0.41.0 is current (2026-10-06) (checked: it does not move torch or numpy; `docs/LICENSES.md` §2) |
| `h5py` | demo store | lzf compression is built in; no hdf5plugin needed |
| `einops` | DP tensor reshapes | optional |

Settings copied from P8: `requires-python = ">=3.12"`, ruff line length 100, same lint selection,
mypy strict, pytest markers. Add markers `gpu`, `slow`, `checkpoints` (needs P8's trained PPO
checkpoint, workstation only), `data` (needs the demo store).

## 3. Reference environment (`.venv-lerobot`, only for arm R1)

```bash
python3.12 -m venv .venv-lerobot && . .venv-lerobot/bin/activate
pip install "lerobot==0.6.1"     # Apache-2.0; requires Python >= 3.12 (PyPI, 2026-08-03)
```

Never `import lerobot` from `src/ila`. Only two things cross between the venvs: the exported
LeRobotDataset (written by `scripts/r1/export_dataset.py` + `scripts/r1/build_lerobot_dataset.py`, i.e. `make r1-export r1-dataset`, read in `.venv-lerobot`) and
the trained policy as ONNX (written in `.venv-lerobot`, flown in `.venv`).

## 4. Layout

```
imitation-landing-approach/
├── CLAUDE.md  README.md  REPO_SETUP.md  SOURCES.md  VALIDATION_CHECKLIST.md  KICKSTART.md
├── THIRD_PARTY_NOTICES.md  LICENSE (MIT)  Makefile  pyproject.toml
├── .claude/            agents/ skills/ hooks/ settings.json
├── configs/
│   ├── paths.local.example.yaml      (copy → paths.local.yaml, gitignored)
│   ├── obs/student.yaml              (the observation contract)
│   ├── render/camera.yaml            (camera, marking, background, appearance-shift variants)
│   ├── data/demos.yaml               (teachers, N, seeds, DART σ grid, DAgger schedule)
│   ├── train/{bc_state,bc,bc_matched,bc_dart,bc_dagger,dp,modular}.yaml
│   └── eval/matrix.yaml              (arms × lists × seeds; Ta/K sweeps; latency arm)
├── docs/  IMPLEMENTATION_PLAN.md  protocol.md  CONVENTIONS.md  LICENSES.md  LIMITATIONS.md
├── external/rl-land-on-moving-deck/  (submodule, read-only)
├── splits/            train/select episode manifests (+ SHA-256), committed
├── src/ila/
│   ├── p8_adapter.py                 the ONLY importer of rld
│   ├── obs/split.py
│   ├── render/{camera,scene,texture}.py
│   ├── data/{episodes,collect,store,dart,dagger,export_lerobot}.py
│   ├── policies/{encoder,bc,state_mlp,modular,registry}.py
│   ├── policies/dp/{unet1d,policy,chunking}.py
│   ├── train/{loop,dataset,select,manifest}.py
│   ├── eval/{runner,batched,arms,metrics,hypotheses,report}.py
│   ├── deploy/{export,parity,latency,latency_arm}.py
│   └── viz/{figures,gifs}.py
├── scripts/           argparse wrappers only
├── tests/
├── results/           committed CSV/JSON/figures
├── data/              demo store (gitignored)
└── artifacts/         runs, checkpoints, ONNX, logs (gitignored)
```

## 5. Makefile targets (Phase 0 creates stubs that print `not implemented: phase N`)

`lint test smoke verify-p8 reproduce-p8 throughput collect train-smoke train-bg dagger-bg eval
bench report figures gifs all`. Auto-detect `PY`/`RUFF`/`MYPY`/`PYTEST` as P8 does.

## 6. `.gitignore` additions

```
.venv*/  data/  artifacts/  configs/paths.local.yaml  *.h5  *.onnx  *.pt  *.zip  wandb/  runs/
```

## 7. CI (GitHub Actions, CPU)

`actions/checkout` with `submodules: recursive`; Python 3.12; apt `build-essential python3.12-dev`;
the three editable installs + this repo; `make lint`; `pytest -m "not gpu and not checkpoints and not
data and not slow"`. Cache pip (pybullet's sdist build takes minutes).

## 8. GitHub repository

Public repo `kbasak01/imitation-landing-approach`, MIT. Branch per phase (`phase-N`), PR per gate,
merged when the gate passes — P8's workflow. Release `phase-11-complete` with checkpoints, ONNX,
`SHA256SUMS.txt`.
