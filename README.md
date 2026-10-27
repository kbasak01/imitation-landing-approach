# imitation-landing-approach

Vision-based imitation learning for quadrotor landing on a heaving ship deck, **in simulation
only** (PyBullet physics, PyBullet's TinyRenderer). A privileged state-based teacher from
[`rl-land-on-moving-deck`](https://github.com/kbasak01/rl-land-on-moving-deck) (pinned, read-only
submodule in `external/`) is distilled into camera-based students and evaluated on that project's
frozen episode lists and statistics.

Work in progress (Phase 0: scaffold). No results yet. Setup: `REPO_SETUP.md`; plan:
`docs/IMPLEMENTATION_PLAN.md`; decision log: `docs/protocol.md`.
