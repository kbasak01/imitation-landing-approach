"""``OnnxDPStudent``: a diffusion student whose networks are an exported ONNX pair (R1).

``R1_lerobot_dp`` (P3-D1 §3, tag ``reference_external``; P6-D3) is LeRobot 0.6.1's
``DiffusionPolicy`` trained by LeRobot in ``.venv-lerobot`` and exported by
``scripts/r1/export_onnx.py`` to ``<run>/onnx/step<N>/``:

* ``encoder.onnx``: ``image`` ``uint8[B, To, 96, 96, 3]`` (HWC, oldest frame first) and ``state``
  ``float32[B, To, 13]`` (raw student proprio, P8 units) -> ``global_cond`` ``float32[B, C]``.
  LeRobot's input normalisation (VISUAL MEAN_STD with ImageNet statistics, STATE MIN_MAX) and
  the eval-mode centre crop (96 -> 88 px) are inside the graph.
* ``denoiser.onnx``: ``sample`` ``float32[B, Tp, 3]`` (normalised action space), ``timestep``
  ``int64[B]`` (the diffusion index k, the only integer a network sees), ``global_cond`` ->
  ``eps`` ``float32[B, Tp, 3]``.
* ``meta.json`` (format ``ila-r1-onnx/1``): horizons, ``C``, the schedulers, the action
  unnormalisation, the source checkpoint's and both graphs' SHA-256.

Sampling is ``ila``'s, identical to ``dp``'s: DDIM ``trailing`` with ``K`` steps
(:func:`ila.policies.dp.policy.ddim_scheduler` / :func:`~ila.policies.dp.policy.ddim_sample`,
eta = 0), initial noise :func:`~ila.policies.dp.policy.initial_noise` keyed
``noise_seed(run_seed, episode_seed, t)``, the final clip to ``[-1, 1]`` in normalised space;
then LeRobot's MIN_MAX unnormalisation (float32: ``d = max - min``, ``d = eps`` where ``d == 0``,
``a = (x + 1) / 2 * d + min``) and a second clip to ``[-1, 1]``, the action bound (normalised
world-frame velocity setpoint; the env scales by v_max = 1.5 m/s model scale with a norm cap).
Chunk execution, replanning and temporal ensembling are
:class:`ila.policies.dp.policy.ChunkedDiffusionStudent`'s: the executed chunk indices are
``To-1 ... To-2+Ta`` (steps ``t ... t+Ta-1``, 1/30 s model scale each), from ``chunking``.

ONNX Runtime runs on the CPU, one intra-op and one inter-op thread, sequential execution: the
selection path (P3-D1 §3, "CPU fp32 single-sample"). The ONNX plumbing (``meta.json`` reading,
hash checks, opset / RNG-op checks, ORT sessions, encode / denoise calls, the DDIM scheduler from
``meta.json``, the MIN_MAX inverse) is :mod:`ila.deploy.r1_parity`'s, the same code the parity
check runs; this module adds the contract checks, the clip to the action bound and the chunk
executor. Selection flies only exports whose ``parity.json`` passed (:func:`parity_passed`).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila.deploy import r1_parity as r1
from ila.policies.dp import chunking
from ila.policies.dp.policy import ChunkedDiffusionStudent, SeedTuple, ddim_sample, initial_noise

__all__ = [
    "DENOISER_FILE",
    "ENCODER_FILE",
    "META_FILE",
    "ONNX_META_FORMAT",
    "PARITY_FILE",
    "OnnxDPMeta",
    "OnnxDPStudent",
    "bundle_sha256",
    "bundle_step",
    "export_dirs",
    "load_onnx_student",
    "parity_passed",
    "read_meta",
]

#: ``meta.json``'s format tag (``scripts/r1/export_onnx.py``).
ONNX_META_FORMAT: Final[str] = r1.FORMAT
ENCODER_FILE: Final[str] = "encoder.onnx"
DENOISER_FILE: Final[str] = "denoiser.onnx"
META_FILE: Final[str] = "meta.json"
#: Written beside the graphs by ``scripts/r1/parity.py`` (:func:`ila.deploy.r1_parity.run_parity`).
PARITY_FILE: Final[str] = "parity.json"

#: The graph I/O contract (names; shapes are checked against ``meta.json``).
ENCODER_OUTPUTS: Final[tuple[str, ...]] = ("global_cond",)
DENOISER_OUTPUTS: Final[tuple[str, ...]] = ("eps",)

#: DDIM spacing of ``dp`` (P3-D1: ``trailing``); R1 samples with the same.
_SPACING: Final[str] = "trailing"

_STEP_DIR = re.compile(r"^step(\d+)$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class OnnxDPMeta:
    """What ``OnnxDPStudent`` reads from ``meta.json``.

    Attributes:
        step: Training step of the source checkpoint.
        n_obs_steps: ``To``.
        horizon: ``Tp``, control steps (1/30 s model scale each).
        n_action_steps: ``Ta`` the policy was configured with (``R1``: 8).
        action_dim: 3.
        state_dim: Proprio width (13).
        image_hwc: ``(96, 96, 3)``.
        cond_dim: ``C``, the ``global_cond`` width.
        num_train_timesteps: DDPM training steps (100).
        beta_schedule: ``squaredcos_cap_v2``.
        clip_sample: Clip ``x_0`` inside each DDIM step.
        clip_sample_range: Must be 1.0 (diffusers' default, which ``ddim_scheduler`` uses).
        action_min: Per-dim MIN_MAX minimum (normalised env action units).
        action_max: Per-dim MIN_MAX maximum.
        action_eps: LeRobot's ``eps`` for a zero range.
        source_sha256: SHA-256 of the source ``model.safetensors``.
        onnx_sha256: File name -> SHA-256 of both graphs.
    """

    step: int
    n_obs_steps: int
    horizon: int
    n_action_steps: int
    action_dim: int
    state_dim: int
    image_hwc: tuple[int, int, int]
    cond_dim: int
    num_train_timesteps: int
    beta_schedule: str
    clip_sample: bool
    clip_sample_range: float
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    action_eps: float
    source_sha256: str
    onnx_sha256: Mapping[str, str]

    @classmethod
    def from_mapping(cls, meta: Mapping[str, Any]) -> OnnxDPMeta:
        """Parse and validate a ``meta.json`` mapping.

        Raises:
            ValueError: On another format, a non-epsilon or non-DDIM-trailing contract, a
                clip range other than 1.0, or an unnormalisation other than MIN_MAX.
        """
        if meta.get("format") != r1.FORMAT:
            raise ValueError(f"meta.json format {meta.get('format')!r} != {r1.FORMAT!r}")
        if meta["prediction_type"] != "epsilon":
            raise ValueError(f"prediction_type {meta['prediction_type']!r} (epsilon only)")
        ila_inf = meta["scheduler"]["ila_inference"]
        native = meta["scheduler"]["lerobot_native"]
        if ila_inf["sampler"] != "ddim" or ila_inf["timestep_spacing"] != _SPACING:
            raise ValueError(
                f"inference contract {ila_inf['sampler']}/{ila_inf['timestep_spacing']}"
            )
        if not ila_inf.get("set_alpha_to_one", False) or int(ila_inf.get("steps_offset", 0)):
            raise ValueError("DDIM must use set_alpha_to_one=True, steps_offset=0 (dp's)")
        for key in ("num_train_timesteps", "beta_schedule", "clip_sample"):
            if ila_inf[key] != native[key]:
                raise ValueError(
                    f"scheduler {key}: inference {ila_inf[key]} != training {native[key]}"
                )
        if float(native["clip_sample_range"]) != 1.0:
            raise ValueError(f"clip_sample_range {native['clip_sample_range']} != 1.0")
        un = meta["action_unnormalisation"]
        if str(un["mode"]).upper() != "MIN_MAX":
            raise ValueError(f"action unnormalisation {un['mode']!r} (MIN_MAX only)")
        a_dim = int(meta["action_dim"])
        if len(un["min"]) != a_dim or len(un["max"]) != a_dim:
            raise ValueError("action min/max width differs from action_dim")
        hwc = tuple(int(v) for v in meta["image_shape_hwc"])
        if len(hwc) != 3:
            raise ValueError(f"image_shape_hwc {hwc}")
        return cls(
            step=int(meta["step"]),
            n_obs_steps=int(meta["n_obs_steps"]),
            horizon=int(meta["horizon"]),
            n_action_steps=int(meta["n_action_steps"]),
            action_dim=a_dim,
            state_dim=int(meta["state_dim"]),
            image_hwc=(hwc[0], hwc[1], hwc[2]),
            cond_dim=int(meta["C"]),
            num_train_timesteps=int(native["num_train_timesteps"]),
            beta_schedule=str(native["beta_schedule"]),
            clip_sample=bool(native["clip_sample"]),
            clip_sample_range=float(native["clip_sample_range"]),
            action_min=tuple(float(v) for v in un["min"]),
            action_max=tuple(float(v) for v in un["max"]),
            action_eps=float(un["eps"]),
            source_sha256=str(meta["source"]["checkpoint_sha256"]),
            onnx_sha256=dict(meta["onnx_sha256"]),
        )


def read_meta(bundle: Path) -> dict[str, Any]:
    """Read ``<bundle>/meta.json`` (:func:`ila.deploy.r1_parity.load_meta`: format checked)."""
    return r1.load_meta(Path(bundle))


def bundle_sha256(bundle: Path) -> str:
    """Verify both graphs against ``meta.json`` and return ``meta.json``'s SHA-256.

    ``meta.json`` records the graphs' and the source checkpoint's SHA-256, so its own hash
    identifies the whole export (the ``ckpt_sha256`` of an R1 row).

    Raises:
        ila.deploy.r1_parity.R1ExportError: If a graph is missing or its bytes differ from
            its recorded SHA-256.
    """
    bundle = Path(bundle)
    r1.verify_files(bundle, read_meta(bundle), reference=False)
    return _sha256(bundle / META_FILE)


def parity_passed(bundle: Path) -> tuple[bool, str]:
    """Whether this export's ``parity.json`` passed, for these exact files.

    Returns:
        ``(passed, reason)``: ``False`` when ``parity.json`` is missing, failed, or was
        written for another ``meta.json`` or other graphs.
    """
    import json

    bundle = Path(bundle)
    path = bundle / PARITY_FILE
    if not path.is_file():
        return False, "no parity.json (run scripts/r1/parity.py)"
    rec = json.loads(path.read_text(encoding="utf-8"))
    meta = read_meta(bundle)
    if rec.get("meta_sha256") != _sha256(bundle / META_FILE):
        return False, "parity.json was written for another meta.json"
    if rec.get("onnx_sha256") is not None and rec["onnx_sha256"] != meta["onnx_sha256"]:
        return False, "parity.json was written for other graphs"
    if rec.get("pass") is not True:
        return False, "parity failed"
    return True, "passed"


def bundle_step(bundle: Path) -> int:
    """Training step of an export dir ``.../onnx/step<N>`` (checked against ``meta.json``)."""
    m = _STEP_DIR.match(Path(bundle).name)
    if m is None:
        raise ValueError(f"{bundle}: not an onnx/step<N> directory")
    step = int(m.group(1))
    if int(read_meta(Path(bundle))["step"]) != step:
        raise ValueError(f"{bundle}: meta.json step differs from the directory name")
    return step


def export_dirs(run_dir: Path) -> dict[int, Path]:
    """Every ``<run>/onnx/step<N>/`` holding a ``meta.json``, keyed by step."""
    root = Path(run_dir) / "onnx"
    out: dict[int, Path] = {}
    for d in sorted(root.glob("step*")) if root.is_dir() else []:
        if d.is_dir() and _STEP_DIR.match(d.name) and (d / META_FILE).is_file():
            out[bundle_step(d)] = d
    return out


def _io(session: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ins = {i.name: i for i in session.get_inputs()}
    outs = {o.name: o for o in session.get_outputs()}
    return ins, outs


class OnnxDPStudent(ChunkedDiffusionStudent):
    """A diffusion student over an ONNX encoder/denoiser pair (module docstring).

    Attributes:
        bundle: The export directory.
        meta: Its parsed ``meta.json``.
        run_seed: The run's seed (first element of every noise key).
        chunk_spec: ``(To, Tp, Ta)`` in use.
        num_inference_steps: ``K`` in use.
        temporal_ensembling: Replan every step and average (P6-D1); off by default.
        device: Always the CPU.
    """

    def __init__(
        self,
        bundle: Path | str,
        run_seed: int,
        *,
        n_action_steps: int | None = None,
        num_inference_steps: int | None = None,
        temporal_ensembling: bool = False,
    ) -> None:
        """Load and check an export.

        Args:
            bundle: ``<run>/onnx/step<N>/``.
            run_seed: The training run's seed.
            n_action_steps: ``Ta``; the export's (8) when ``None``.
            num_inference_steps: ``K``; 10 (``dp``'s default) when ``None``.
            temporal_ensembling: Replan every step and average (P6-D1).

        Raises:
            ValueError: On a ``meta.json`` or graph that breaks the contract (names, dtypes,
                shapes, opset, an RNG op).
            ila.deploy.r1_parity.R1ExportError: On a missing file or a hash mismatch.
        """
        self.bundle = Path(bundle)
        bundle_sha256(self.bundle)
        self._meta_raw = read_meta(self.bundle)
        self.meta = OnnxDPMeta.from_mapping(self._meta_raw)
        graphs = r1.graph_checks(self.bundle)
        if not graphs["ok"]:
            raise ValueError(f"{self.bundle}: graph checks failed (opset, IR, RNG ops): {graphs}")
        self.device = torch.device("cpu")
        m = self.meta
        super().__init__(
            run_seed,
            chunking.ChunkSpec(m.n_obs_steps, m.horizon, m.n_action_steps),
            m.num_train_timesteps,
            n_action_steps=n_action_steps,
            num_inference_steps=10 if num_inference_steps is None else int(num_inference_steps),
            temporal_ensembling=temporal_ensembling,
        )
        self._encoder = r1.cpu_session(self.bundle / ENCODER_FILE, threads=1)
        self._denoiser = r1.cpu_session(self.bundle / DENOISER_FILE, threads=1)
        self._check_graphs()
        self._ddim: dict[int, Any] = {}

    def _check_graphs(self) -> None:
        m = self.meta
        h, w, c = m.image_hwc
        want_enc = {
            "image": ("tensor(uint8)", [m.n_obs_steps, h, w, c]),
            "state": ("tensor(float)", [m.n_obs_steps, m.state_dim]),
        }
        want_den = {
            "sample": ("tensor(float)", [m.horizon, m.action_dim]),
            "timestep": ("tensor(int64)", []),
            "global_cond": ("tensor(float)", [m.cond_dim]),
        }
        for session, want, outs in (
            (self._encoder, want_enc, ENCODER_OUTPUTS),
            (self._denoiser, want_den, DENOISER_OUTPUTS),
        ):
            ins, got_out = _io(session)
            if set(ins) != set(want) or tuple(got_out) != outs:
                raise ValueError(f"graph I/O {sorted(ins)} -> {list(got_out)}, want {sorted(want)}")
            for name, (dtype, tail) in want.items():
                shape = list(ins[name].shape)
                if ins[name].type != dtype or shape[1:] != tail:
                    raise ValueError(f"{name}: {ins[name].type} {shape}, want {dtype} [B, *{tail}]")
        if list(got_out["eps"].shape)[1:] != [m.horizon, m.action_dim]:
            raise ValueError(f"eps: {got_out['eps'].shape}")
        ins, outs_enc = _io(self._encoder)
        if list(outs_enc["global_cond"].shape)[1:] != [m.cond_dim]:
            raise ValueError(f"global_cond: {outs_enc['global_cond'].shape}, C = {m.cond_dim}")

    def _check_window(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Validate one window against the export's ``To``, frame shape and proprio width.

        Raises:
            ValueError: On a wrong dtype or shape (a 25-d proprio fails the width check).
        """
        m = self.meta
        img = np.asarray(image_stack)
        prop = np.asarray(proprio)
        if img.dtype != np.uint8 or img.shape != (m.n_obs_steps, *m.image_hwc):
            raise ValueError(
                f"image_stack must be uint8[{m.n_obs_steps}, {m.image_hwc}], got "
                f"{img.dtype} {img.shape}"
            )
        if prop.dtype != np.float32 or prop.shape != (m.n_obs_steps, m.state_dim):
            raise ValueError(
                f"proprio must be float32[{m.n_obs_steps}, {m.state_dim}], got "
                f"{prop.dtype} {prop.shape}"
            )
        return img, prop

    def ddim(self, num_inference_steps: int) -> Any:
        """``dp``'s DDIM scheduler for this export at ``K`` (:func:`r1.ddim_from_meta`; cached)."""
        k = int(num_inference_steps)
        if k not in self._ddim:
            self._ddim[k] = r1.ddim_from_meta(self._meta_raw, k)
        return self._ddim[k]

    def encode(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``encoder.onnx``: ``uint8[B, To, 96, 96, 3]``, ``float32[B, To, D]`` -> ``[B, C]``."""
        out = r1.encode(self._encoder, images, proprio)
        if out.shape != (images.shape[0], self.meta.cond_dim):
            raise ValueError(
                f"global_cond {out.shape}, want ({images.shape[0]}, {self.meta.cond_dim})"
            )
        return out

    def predict_noise(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: NDArray[np.float32]
    ) -> torch.Tensor:
        """``denoiser.onnx``: epsilon-hat for normalised ``[B, Tp, A]`` at diffusion step(s) k."""
        eps = r1.denoise(
            self._denoiser,
            sample.detach().cpu().numpy(),
            timestep.detach().cpu().numpy(),
            global_cond,
        )
        return torch.from_numpy(eps)

    def unnormalise(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """LeRobot's MIN_MAX inverse (:func:`r1.unnormalise`), then the action bound ``[-1, 1]``."""
        out: NDArray[np.float32] = np.clip(r1.unnormalise(self._meta_raw, x), -1.0, 1.0).astype(
            np.float32
        )
        return out

    def sample_batch(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        seeds: Sequence[SeedTuple],
        num_inference_steps: int | None = None,
    ) -> NDArray[np.float32]:
        """One full chunk per slot (module docstring); slot ``i``'s noise keyed ``seeds[i]``.

        Args:
            images: ``uint8[B, To, 96, 96, 3]``.
            proprio: ``float32[B, To, D]`` (raw, P8 units).
            seeds: ``B`` tuples ``(run_seed, episode_seed, step)``.
            num_inference_steps: ``K``; this student's ``K`` when ``None``.

        Returns:
            ``float32[B, Tp, A]`` in ``[-1, 1]`` (env action units; chunk index ``j`` = step
            ``t-To+1+j``).

        Raises:
            ValueError: If ``len(seeds) != B``.
        """
        img = np.asarray(images)
        if len(seeds) != img.shape[0]:
            raise ValueError(f"{len(seeds)} seed tuples for a batch of {img.shape[0]}")
        k = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        noise = initial_noise(seeds, self.meta.horizon, self.meta.action_dim)
        cond = self.encode(img, np.asarray(proprio, np.float32))

        def eps_fn(x: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
            return self.predict_noise(x, kk, cond)

        with torch.inference_mode():
            x = ddim_sample(eps_fn, noise, self.ddim(k))
        return self.unnormalise(x.numpy().astype(np.float32))


def load_onnx_student(
    bundle: Path | str,
    run_seed: int,
    *,
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
    temporal_ensembling: bool = False,
) -> OnnxDPStudent:
    """Build an :class:`OnnxDPStudent` from ``<run>/onnx/step<N>/`` (see the class)."""
    return OnnxDPStudent(
        bundle,
        run_seed,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        temporal_ensembling=temporal_ensembling,
    )
