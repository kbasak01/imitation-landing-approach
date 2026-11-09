"""One numpy-in, numpy-out interface over every Phase 9 provider (P9-D1).

A :class:`Backend` runs the exported networks of ``bc_dagger`` and ``dp`` on one provider, at one
batch size, host to host (numpy in, numpy out, as P8's harness times them):

* :meth:`Backend.bc`: ``images uint8[B, To, 96, 96, 3]``, ``proprio float32[B, To, 13]`` ->
  ``action float32[B, 3]`` (normalised world-frame velocity setpoint; x v_max = 1.5 m/s model
  scale);
* :meth:`Backend.encode`: same inputs -> ``global_cond float32[B, 256]``;
* :meth:`Backend.denoise`: ``x_k float32[B, Tp, 3]``, ``k int64[B]``, ``global_cond`` ->
  ``eps float32[B, Tp, 3]``;
* :meth:`Backend.unrolled`: ``global_cond``, initial noise ``float32[B, Tp, 3]`` -> the K = 10
  chunk (``dp_K10.onnx``, or :class:`ila.deploy.export.UnrolledDDIM` in torch);
* :meth:`Backend.loop`: ``global_cond``, noise, ``K`` -> chunk, ``ila``'s sampler
  (:func:`ila.policies.dp.policy.ddim_sample`, DDIM trailing, eta = 0) around :meth:`denoise`.
  ORT providers run it on the host (sampling outside the graph, as R1 does); torch providers
  on their device;
* :meth:`Backend.dp_e2e`: images, proprio, noise -> the K = 10 chunk: ``dp_encoder.onnx`` then
  ``dp_K10.onnx`` on ORT; :meth:`ila.policies.dp.policy.DiffusionPolicyNet.sample` in torch
  (the ``DPStudent`` path).

Providers (:mod:`ila.deploy.settings`):

* ``ort_cpu_fp32``: ``CPUExecutionProvider``, ``threads`` intra-op and 1 inter-op thread,
  ``ORT_ENABLE_ALL``;
* ``ort_cuda_fp32`` / ``ort_cuda_fp16``: ``CUDAExecutionProvider`` with P8's options
  (``use_tf32 = 0``), CPU EP listed after it as P8 does;
* ``ort_trt_fp16`` / ``ort_trt_fp32``: ``TensorrtExecutionProvider`` (``trt_fp16_enable`` per
  precision, engine and timing caches under ``artifacts/deploy/trt_cache/<provider>/b<B>/``;
  the engine is built for the backend's one batch size), then the CUDA EP (``use_tf32 = 0``)
  for any node TensorRT does not take, then the CPU EP. ``NVIDIA_TF32_OVERRIDE = 0`` holds
  TensorRT's fp32 layers to fp32 (dmf's ``disable_tf32``; ORT 1.29's TRT EP has no TF32 option);
* ``torch_cpu_fp32`` / ``torch_cuda_fp32``: the EMA networks in torch eager, fp32, TF32 off,
  cuDNN autotuning off (:func:`ila.eval.batched.configure_torch_numerics`).

Every ORT session is checked to have realized its requested first provider (ORT falls back
silently otherwise; dmf's guard), and its realized provider list is recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ila import p8_adapter
from ila.deploy.export import (
    GRAPH_FILES,
    BCGraph,
    DenoiseStepGraph,
    EncoderGraph,
    LoadedArm,
    UnrolledDDIM,
    graph_path,
)
from ila.deploy.settings import GPU_PROVIDERS, precision_of
from ila.policies.bc import BCNet
from ila.policies.dp.policy import DiffusionPolicyNet, DPConfig, ddim_sample, ddim_scheduler

__all__ = [
    "ORT_PROVIDER_NAMES",
    "Backend",
    "BackendError",
    "OrtBackend",
    "TorchBackend",
    "make_backend",
    "ort_session",
    "synchronize",
]

#: ORT providers requested per ``ila`` provider, in priority order.
ORT_PROVIDER_NAMES: Final[Mapping[str, tuple[str, ...]]] = {
    "ort_cpu_fp32": ("CPUExecutionProvider",),
    "ort_cuda_fp32": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "ort_cuda_fp16": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "ort_trt_fp16": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
    "ort_trt_fp32": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
}


class BackendError(RuntimeError):
    """A provider did not realize, or a backend was asked for a network it does not hold."""


def synchronize(provider: str) -> None:
    """``torch.cuda.synchronize()`` for a GPU provider (P8's bracket); a no-op on the CPU."""
    if provider in GPU_PROVIDERS and torch.cuda.is_available():
        torch.cuda.synchronize()


class Backend(Protocol):
    """The interface (module docstring). Shapes: batch first; units as in the module docstring."""

    provider: str

    def bc(self, images: NDArray[np.uint8], proprio: NDArray[np.float32]) -> NDArray[np.float32]:
        """``bc_dagger``: window -> action."""
        ...

    def encode(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``dp`` encoder: window -> ``global_cond``."""
        ...

    def denoise(
        self, sample: NDArray[np.float32], timestep: NDArray[np.int64], global_cond: NDArray[Any]
    ) -> NDArray[np.float32]:
        """One U-Net call: ``x_k``, ``k``, ``global_cond`` -> epsilon-hat."""
        ...

    def unrolled(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """The unrolled K = 10 sampler."""
        ...

    def loop(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32], k: int
    ) -> NDArray[np.float32]:
        """``ila``'s DDIM loop around :meth:`denoise` with ``K = k``."""
        ...

    def dp_e2e(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        noise: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Image stack and proprio -> the K = 10 chunk (one full replanning call)."""
        ...

    def realized(self) -> dict[str, list[str]]:
        """Graph -> realized providers (ORT) or ``["torch:<device>"]``."""
        ...


def ort_session(
    path: Path,
    provider: str,
    *,
    threads: int = 1,
    batch: int = 1,
    trt_cache: Path | None = None,
) -> Any:
    """Build an ONNX Runtime session for one graph on one ``ila`` provider (module docstring).

    Args:
        path: The graph.
        provider: An ``ort_*`` provider name.
        threads: Intra-op threads (inter-op 1).
        batch: Batch size the session runs (TensorRT cache directory ``b<batch>``).
        trt_cache: TensorRT engine/timing cache directory.

    Returns:
        The session.

    Raises:
        BackendError: If the requested first provider is not realized.
    """
    import onnxruntime as ort

    h = p8_adapter.latency_harness()
    h.preload_gpu_libraries()
    h.disable_tf32()
    names = ORT_PROVIDER_NAMES[provider]
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    ort.set_default_logger_severity(3)
    plist: list[Any] = list(h.session_providers(names))
    if names[0] == "TensorrtExecutionProvider":
        if trt_cache is None:
            raise BackendError("a TensorRT session needs a cache directory")
        # No explicit profile: the graph is partitioned (TensorRT does not take the uint8 image
        # input and its first reshapes), and an explicit profile would have to name every
        # TensorRT subgraph's inputs. The engine is built for the first shape seen; a backend
        # runs one batch size, and each batch size has its own cache directory.
        cache = trt_cache / f"b{int(batch)}"
        cache.mkdir(parents=True, exist_ok=True)
        trt_opts = {
            "trt_fp16_enable": "True" if precision_of(provider) == "fp16" else "False",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(cache),
            "trt_timing_cache_enable": "True",
            "trt_timing_cache_path": str(cache),
        }
        plist[0] = ("TensorrtExecutionProvider", trt_opts)
    session = ort.InferenceSession(str(path), so, providers=plist)
    realized = [str(p) for p in session.get_providers()]
    if names[0] not in realized:
        raise BackendError(
            f"{path.name}: requested {names[0]} but the session realized {realized}; ORT falls "
            "back silently, so this provider is refused"
        )
    return session


class OrtBackend:
    """ONNX Runtime over the exported graphs (sessions built on first use).

    Attributes:
        provider: The ``ila`` provider.
        graph_dir: The graph directory.
        batch: Batch size (TensorRT profile).
        threads: Intra-op threads.
        dp_cfg: ``dp``'s config (sampler of :meth:`loop`).
    """

    def __init__(
        self,
        provider: str,
        graph_dir: Path,
        dp_cfg: DPConfig,
        *,
        batch: int = 1,
        threads: int = 1,
        trt_cache: Path | None = None,
    ) -> None:
        """Hold the settings; no session yet."""
        if provider not in ORT_PROVIDER_NAMES:
            raise BackendError(f"{provider!r} is not an ORT provider")
        self.provider = provider
        self.graph_dir = Path(graph_dir)
        self.batch = int(batch)
        self.threads = int(threads)
        self.dp_cfg = dp_cfg
        self.trt_cache = trt_cache
        self._sessions: dict[str, Any] = {}
        self._ddim: dict[int, Any] = {}

    def session(self, key: str) -> Any:
        """The session of graph ``key`` (built on first use)."""
        if key not in self._sessions:
            path = graph_path(self.graph_dir, key, precision_of(self.provider))
            self._sessions[key] = ort_session(
                path,
                self.provider,
                threads=self.threads,
                batch=self.batch,
                trt_cache=self.trt_cache,
            )
        return self._sessions[key]

    def _run(self, key: str, feeds: Mapping[str, NDArray[Any]]) -> NDArray[np.float32]:
        out = self.session(key).run(None, {k: np.ascontiguousarray(v) for k, v in feeds.items()})
        return np.asarray(out[0], dtype=np.float32)

    def bc(self, images: NDArray[np.uint8], proprio: NDArray[np.float32]) -> NDArray[np.float32]:
        """``bc_dagger.onnx``."""
        return self._run(
            "bc", {"images": images.astype(np.uint8), "proprio": proprio.astype(np.float32)}
        )

    def encode(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``dp_encoder.onnx``."""
        return self._run(
            "encoder", {"images": images.astype(np.uint8), "proprio": proprio.astype(np.float32)}
        )

    def denoise(
        self, sample: NDArray[np.float32], timestep: NDArray[np.int64], global_cond: NDArray[Any]
    ) -> NDArray[np.float32]:
        """``dp_denoise_step.onnx``."""
        return self._run(
            "step",
            {
                "sample": sample.astype(np.float32),
                "timestep": timestep.astype(np.int64),
                "global_cond": np.asarray(global_cond, np.float32),
            },
        )

    def unrolled(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``dp_K10.onnx``."""
        return self._run(
            "unrolled",
            {"global_cond": np.asarray(global_cond, np.float32), "noise": noise.astype(np.float32)},
        )

    def loop(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32], k: int
    ) -> NDArray[np.float32]:
        """:func:`ddim_sample` on the host around :meth:`denoise`."""
        if k not in self._ddim:
            self._ddim[k] = ddim_scheduler(self.dp_cfg, k)
        cond = np.asarray(global_cond, np.float32)

        def eps_fn(x: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
            return torch.from_numpy(self.denoise(x.numpy(), kk.numpy(), cond))

        out = ddim_sample(eps_fn, torch.from_numpy(np.asarray(noise, np.float32)), self._ddim[k])
        return np.asarray(out.numpy(), dtype=np.float32)

    def dp_e2e(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        noise: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """``dp_encoder.onnx`` then ``dp_K10.onnx``."""
        return self.unrolled(self.encode(images, proprio), noise)

    def realized(self) -> dict[str, list[str]]:
        """Graph file -> realized ORT providers, for every session built."""
        return {
            graph_path(self.graph_dir, k, precision_of(self.provider)).name: [
                str(p) for p in s.get_providers()
            ]
            for k, s in self._sessions.items()
        }


class TorchBackend:
    """Torch eager over the EMA networks (fp32), on the CPU or CUDA.

    Attributes:
        provider: ``torch_cpu_fp32`` or ``torch_cuda_fp32``.
        device: The torch device.
    """

    def __init__(
        self,
        provider: str,
        arms: Mapping[str, LoadedArm],
        *,
        threads: int = 1,
        k_unrolled: int = 10,
    ) -> None:
        """Move the networks to the device with TF32 off.

        Raises:
            BackendError: If CUDA is requested but unavailable, or the provider is unknown.
        """
        from ila.eval.batched import configure_torch_numerics

        if provider not in ("torch_cpu_fp32", "torch_cuda_fp32"):
            raise BackendError(f"{provider!r} is not a torch provider")
        self.provider = provider
        dev = "cuda" if provider == "torch_cuda_fp32" else "cpu"
        if dev == "cuda":
            if not torch.cuda.is_available():
                raise BackendError("torch_cuda_fp32 requested but CUDA is unavailable")
            p8_adapter.latency_harness().disable_tf32()
        else:
            torch.set_num_threads(int(threads))
        self.numerics = configure_torch_numerics(dev)
        self.device = torch.device(dev)
        self._bc: BCGraph | None = None
        self._dp: DiffusionPolicyNet | None = None
        self._modules: dict[str, torch.nn.Module] = {}
        bc = arms.get("bc_dagger")
        if bc is not None:
            assert isinstance(bc.net, BCNet)
            self._modules["bc"] = BCGraph(bc.net).to(self.device).eval()
        dp = arms.get("dp")
        if dp is not None:
            assert isinstance(dp.net, DiffusionPolicyNet)
            net = dp.net.to(self.device).eval()
            self._dp = net
            self._modules["encoder"] = EncoderGraph(net).eval()
            self._modules["step"] = DenoiseStepGraph(net).eval()
            self._modules["unrolled"] = UnrolledDDIM(net, k_unrolled).to(self.device).eval()

    def _module(self, key: str) -> torch.nn.Module:
        if key not in self._modules:
            raise BackendError(f"{self.provider}: no {GRAPH_FILES[key]} network loaded")
        return self._modules[key]

    def _t(self, x: NDArray[Any]) -> torch.Tensor:
        arr = np.ascontiguousarray(x)
        return torch.from_numpy(arr.copy() if not arr.flags.writeable else arr).to(self.device)

    @staticmethod
    def _np(x: torch.Tensor) -> NDArray[np.float32]:
        out: NDArray[np.float32] = x.detach().cpu().numpy().astype(np.float32)
        return out

    def _call(self, key: str, *args: NDArray[Any]) -> NDArray[np.float32]:
        with torch.inference_mode():
            return self._np(self._module(key)(*(self._t(a) for a in args)))

    def bc(self, images: NDArray[np.uint8], proprio: NDArray[np.float32]) -> NDArray[np.float32]:
        """:class:`BCGraph`."""
        return self._call("bc", images.astype(np.uint8), proprio.astype(np.float32))

    def encode(
        self, images: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """:class:`EncoderGraph`."""
        return self._call("encoder", images.astype(np.uint8), proprio.astype(np.float32))

    def denoise(
        self, sample: NDArray[np.float32], timestep: NDArray[np.int64], global_cond: NDArray[Any]
    ) -> NDArray[np.float32]:
        """:class:`DenoiseStepGraph`."""
        return self._call(
            "step",
            sample.astype(np.float32),
            timestep.astype(np.int64),
            np.asarray(global_cond, np.float32),
        )

    def unrolled(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """:class:`UnrolledDDIM` (K = 10)."""
        return self._call("unrolled", np.asarray(global_cond, np.float32), noise.astype(np.float32))

    def _net(self) -> DiffusionPolicyNet:
        if self._dp is None:
            raise BackendError(f"{self.provider}: no dp network loaded")
        return self._dp

    def loop(
        self, global_cond: NDArray[Any], noise: NDArray[np.float32], k: int
    ) -> NDArray[np.float32]:
        """:func:`ddim_sample` on the device with the U-Net."""
        net = self._net()
        with torch.inference_mode():
            cond = self._t(np.asarray(global_cond, np.float32))

            def eps_fn(x: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
                return net.predict_noise(x, kk, cond)

            out = ddim_sample(eps_fn, self._t(noise.astype(np.float32)), net.ddim(k))
            return self._np(out)

    def dp_e2e(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        noise: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """:meth:`DiffusionPolicyNet.sample` at K = 10 (the ``DPStudent`` path)."""
        net = self._net()
        with torch.inference_mode():
            out = net.sample(
                self._t(images.astype(np.uint8)),
                self._t(proprio.astype(np.float32)),
                torch.from_numpy(noise.astype(np.float32)),
                10,
            )
            return self._np(out)

    def realized(self) -> dict[str, list[str]]:
        """Every loaded network -> ``["torch:<device>"]``."""
        return {GRAPH_FILES[k]: [f"torch:{self.device.type}"] for k in self._modules}


def make_backend(
    provider: str,
    *,
    graph_dir: Path,
    dp_cfg: DPConfig,
    arms_loader: Callable[[], Mapping[str, LoadedArm]],
    batch: int = 1,
    threads: int = 1,
    trt_cache_root: Path | None = None,
) -> OrtBackend | TorchBackend:
    """Build the backend of one provider.

    Args:
        provider: An ``ila`` provider name.
        graph_dir: The exported graphs (ORT).
        dp_cfg: ``dp``'s config (ORT host sampler).
        arms_loader: Returns the pinned networks (torch providers only; called lazily).
        batch: Batch size (TensorRT profile).
        threads: Intra-op CPU threads.
        trt_cache_root: Root of the TensorRT caches (``<root>/<provider>``).

    Returns:
        The backend.
    """
    if provider.startswith("torch_"):
        return TorchBackend(provider, arms_loader(), threads=threads)
    cache = None if trt_cache_root is None else Path(trt_cache_root) / provider
    return OrtBackend(provider, graph_dir, dp_cfg, batch=batch, threads=threads, trt_cache=cache)
