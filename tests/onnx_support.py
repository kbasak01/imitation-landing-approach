"""A tiny synthetic R1-shaped ONNX export for tests (``OnnxDPStudent``, selection, guards).

``write_bundle`` exports two small random-init torch modules with the R1 graph contract
(``scripts/r1/export_onnx.py``; P6-D3) through ``torch.onnx.export`` (TorchScript exporter: the
test venv has no onnxscript) and writes a ``meta.json`` in the exporter's ``ila-r1-onnx/1``
format:

* ``encoder.onnx``: ``image`` ``uint8[B, 2, 96, 96, 3]``, ``state`` ``float32[B, 2, 13]`` ->
  ``global_cond`` ``float32[B, C]`` (per-frame channel means and the proprio, a linear map, tanh);
* ``denoiser.onnx``: ``sample`` ``float32[B, 16, 3]``, ``timestep`` ``int64[B]``,
  ``global_cond`` -> ``eps`` ``float32[B, 16, 3]`` (sinusoidal k embedding, 2-layer MLP).

Nothing here is LeRobot's network; it only has the contract's names, dtypes and shapes.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

TO, TP, A, D, C = 2, 16, 3, 13, 24
ACTION_MIN = [-0.55, -0.45, -1.0]
ACTION_MAX = [0.62, 0.56, 0.24]


class TinyEncoder(nn.Module):
    """``(image uint8, state) -> global_cond``; the image enters as float / 255."""

    def __init__(self) -> None:
        """Build the layer."""
        super().__init__()
        self.lin = nn.Linear(TO * (3 + D), C)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return ``global_cond``."""
        x = image.to(torch.float32) / 255.0
        feats = torch.cat([x.mean(dim=(2, 3)), state], dim=-1)
        out: torch.Tensor = torch.tanh(self.lin(feats.flatten(1)))
        return out


class TinyDenoiser(nn.Module):
    """``(sample, timestep, global_cond) -> eps``."""

    def __init__(self, rng_op: bool = False) -> None:
        """Build the layers (``rng_op``: plant a RandomNormalLike, a negative control)."""
        super().__init__()
        self.register_buffer("freqs", torch.linspace(0.01, 0.5, 8))
        self.l1 = nn.Linear(TP * A + 16 + C, 64)
        self.l2 = nn.Linear(64, TP * A)
        self.rng_op = rng_op

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """Return epsilon-hat."""
        arg = timestep.to(torch.float32)[:, None] * self.freqs[None, :]
        temb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        h = torch.cat([sample.flatten(1), temb, global_cond], dim=-1)
        out = self.l2(torch.tanh(self.l1(h))).reshape(-1, TP, A)
        if self.rng_op:  # negative control: a graph that reads an RNG
            out = out + 0.0 * torch.randn_like(out)
        return out


def _export(module: nn.Module, args: tuple[Any, ...], inputs: list[str], output: str) -> bytes:
    buf = io.BytesIO()
    dyn = {name: {0: "B"} for name in [*inputs, output]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            args,
            buf,
            input_names=inputs,
            output_names=[output],
            dynamic_axes=dyn,
            opset_version=17,
            dynamo=False,
        )
    return buf.getvalue()


def modules(seed: int = 0, rng_op: bool = False) -> tuple[TinyEncoder, TinyDenoiser]:
    """The two torch modules (seeded init, eval mode)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyEncoder().eval(), TinyDenoiser(rng_op).eval()


def write_bundle(
    out: Path,
    *,
    step: int = 10000,
    seed: int = 0,
    source_sha256: str = "0" * 64,
    action_min: list[float] | None = None,
    action_max: list[float] | None = None,
    rng_op: bool = False,
    meta_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write ``encoder.onnx``, ``denoiser.onnx`` and ``meta.json`` to ``out`` (created)."""
    out.mkdir(parents=True, exist_ok=True)
    enc, den = modules(seed, rng_op)
    files = {
        "encoder.onnx": _export(
            enc,
            (torch.zeros(1, TO, 96, 96, 3, dtype=torch.uint8), torch.zeros(1, TO, D)),
            ["image", "state"],
            "global_cond",
        ),
        "denoiser.onnx": _export(
            den,
            (torch.zeros(1, TP, A), torch.zeros(1, dtype=torch.long), torch.zeros(1, C)),
            ["sample", "timestep", "global_cond"],
            "eps",
        ),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
    meta: dict[str, Any] = {
        "format": "ila-r1-onnx/1",
        "step": step,
        "n_obs_steps": TO,
        "horizon": TP,
        "n_action_steps": 8,
        "action_dim": A,
        "state_dim": D,
        "image_shape_hwc": [96, 96, 3],
        "C": C,
        "prediction_type": "epsilon",
        "scheduler": {
            "lerobot_native": {
                "type": "DDPM",
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "clip_sample_range": 1.0,
            },
            "ila_inference": {
                "sampler": "ddim",
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon",
                "timestep_spacing": "trailing",
                "set_alpha_to_one": True,
                "steps_offset": 0,
                "eta": 0.0,
            },
        },
        "action_unnormalisation": {
            "mode": "MIN_MAX",
            "min": ACTION_MIN if action_min is None else action_min,
            "max": ACTION_MAX if action_max is None else action_max,
            "eps": 1e-8,
        },
        "source": {"checkpoint_sha256": source_sha256, "checkpoint_sha256_of": "model.safetensors"},
        "onnx_sha256": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
        "versions": {"lerobot": "synthetic", "torch": torch.__version__, "opset": 17},
    }
    for key, value in (meta_overrides or {}).items():
        meta[key] = value
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out


def write_parity(bundle: Path, *, passed: bool = True) -> Path:
    """A minimal ``parity.json`` for ``bundle``'s exact files (selection's parity gate)."""
    meta = json.loads((bundle / "meta.json").read_text())
    rec = {
        "pass": passed,
        "meta_sha256": hashlib.sha256((bundle / "meta.json").read_bytes()).hexdigest(),
        "onnx_sha256": meta["onnx_sha256"],
        "synthetic": True,
    }
    (bundle / "parity.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return bundle
