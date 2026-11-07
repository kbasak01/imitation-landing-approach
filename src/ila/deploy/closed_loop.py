"""Closed-loop parity: the 150 ``select`` episodes per provider and arm (P3-D1 §3, §6 H5(a), §7).

Binding test (P3-D1 §6 H5(a), "closed-loop: the 150 ``select`` episodes (never a frozen list),
outcome flips against the CPU single-sample path <= 3"; §7 the same for ``bc_dagger``).

* **Reference flight**: the arm's pinned checkpoint on the CPU fp32 single-sample path, exactly
  as selection flew it (:func:`ila.train.select.fly_checkpoint`: EMA weights, one torch thread
  per worker, render ``full``, ``dp`` at the config defaults ``Ta = 8``, ``K = 10``). Its rows
  are compared with the committed selection rows of the same checkpoint
  (``results/phase{5,6}/runs/<arm>/s0/select/<rows_file>``); P2-D3 item 1 says they are
  byte-identical, and the record says whether they are.
* **Provider flights**: the same episodes, tags, render and contract through the same runner
  (:func:`ila.eval.runner.run_list_vision`; its semantics are unchanged), with a student whose
  networks run on the provider (:class:`ProviderBCStudent`, :class:`ProviderDPStudent`, built
  in each worker by :func:`provider_policy`). ``dp`` keeps ``ChunkedDiffusionStudent``'s chunk
  executor, its noise keys ``(run_seed, episode_seed, t)`` and ``Ta = 8``; its chunk is
  :meth:`ila.deploy.backends.Backend.dp_e2e` (``dp_encoder.onnx`` -> ``dp_K10.onnx`` on ORT,
  ``DiffusionPolicyNet.sample`` in torch). Batch 1 in every worker.
* **Flips**: episodes whose ``outcome`` differs from the reference flight's, in list order.
  ``pass`` iff ``flips <= 3``. Rows whose text differs are counted beside (expected for every
  GPU provider: P2-D3 item 2).

TensorRT engines are built once in the parent (batch-1 profile, cached under
``artifacts/deploy/trt_cache/<provider>/``) before the workers start, so no two workers build
the same engine. ``NVIDIA_TF32_OVERRIDE = 0`` is set in the parent and inherited.

Outputs under ``results/latency/closed_loop/<arm>/``: ``<provider>_rows.csv`` (the runner's
columns) and ``<provider>.json`` (flips, outcome counts, provenance). Simulation only; control
steps are 1/30 s model scale.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import episodes as data_episodes
from ila.deploy import backends as bk
from ila.deploy.export import LoadedArm, git_state, load_arm, sha256_file
from ila.deploy.settings import GPU_PROVIDERS, BenchSettings
from ila.eval import arms as ev_arms
from ila.eval import runner
from ila.policies.dp.policy import ChunkedDiffusionStudent, DPConfig, SeedTuple, initial_noise
from ila.policies.encoder import EncoderConfig, VisionStudentBase

__all__ = [
    "CLOSED_LOOP_COLUMNS",
    "ProviderBCStudent",
    "ProviderDPStudent",
    "closed_loop_csv",
    "count_flips",
    "fly_provider",
    "provider_policy",
    "rows_text",
    "run_closed_loop",
]

#: Columns of ``results/latency/closed_loop.csv``.
CLOSED_LOOP_COLUMNS: Final[tuple[str, ...]] = (
    "arm",
    "provider",
    "scored",
    "status",
    "n",
    "flips",
    "max_flips",
    "closed_loop_pass",
    "rows_text_differ",
    "success",
    "outcomes",
    "reference_equals_selection_rows",
    "workers",
    "wall_s",
    "ila_commit",
)


class ProviderBCStudent(VisionStudentBase):
    """``bc_dagger`` on a provider: one :meth:`Backend.bc` call per control step (batch 1)."""

    def __init__(self, backend: bk.Backend, cfg: EncoderConfig) -> None:
        """Hold the backend and the encoder config (window checks)."""
        self.backend = backend
        self.cfg = cfg
        self.device = torch.device("cpu")

    def _reset_episode(self, seed: int) -> None:
        del seed

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``uint8[To, 96, 96, 3]``, ``float32[To, 13]`` -> ``float32[3]`` in [-1, 1]."""
        img, prop = self._check_window(image_stack, proprio)
        out = self.backend.bc(img[None], prop[None])[0]
        self.inference_calls += 1
        return np.asarray(out, dtype=np.float32)


class ProviderDPStudent(ChunkedDiffusionStudent):
    """``dp`` on a provider: ``ChunkedDiffusionStudent``'s executor, the provider's chunk."""

    def __init__(
        self,
        backend: bk.Backend,
        cfg: EncoderConfig,
        dp_cfg: DPConfig,
        run_seed: int,
        *,
        n_action_steps: int | None = None,
        num_inference_steps: int = 10,
    ) -> None:
        """Set the executor (``Ta`` from the config when ``None``) and hold the backend."""
        super().__init__(
            run_seed,
            dp_cfg.chunk,
            dp_cfg.num_train_timesteps,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
        )
        self.backend = backend
        self.cfg = cfg
        self.dp_cfg = dp_cfg
        self.device = torch.device("cpu")

    def sample_batch(
        self,
        images: NDArray[np.uint8],
        proprio: NDArray[np.float32],
        seeds: Sequence[SeedTuple],
        num_inference_steps: int | None = None,
    ) -> NDArray[np.float32]:
        """One chunk per slot, noise keyed ``seeds[i]``; ``[B, Tp, 3]`` in [-1, 1]."""
        k = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        noise = initial_noise(seeds, self.dp_cfg.horizon, self.dp_cfg.action_dim).numpy()
        if k == 10:
            return self.backend.dp_e2e(images, proprio, noise)
        return self.backend.loop(self.backend.encode(images, proprio), noise, k)


def provider_policy(
    arm: str,
    provider: str,
    graph_dir: str,
    pins: str,
    trt_cache_root: str,
    cfgs: p8_adapter.EvalConfigs,
) -> Any:
    """Picklable ``PolicySpec.build``: the arm's student on a provider (in the worker).

    The pinned checkpoint is resolved and byte-verified again here (its config gives the
    encoder and sampler settings; torch providers also take its EMA weights).
    """
    del cfgs
    torch.set_num_threads(1)
    loaded: LoadedArm = load_arm(arm, 0, Path(pins))
    # bc never samples; its backend's host sampler config is unused.
    dp_cfg = loaded.net.cfg if arm == "dp" else DPConfig()
    assert isinstance(dp_cfg, DPConfig)
    backend = bk.make_backend(
        provider,
        graph_dir=Path(graph_dir),
        dp_cfg=dp_cfg,
        arms_loader=lambda: {arm: loaded},
        batch=1,
        threads=1,
        trt_cache_root=Path(trt_cache_root),
    )
    if arm == "dp":
        return ProviderDPStudent(backend, loaded.encoder_cfg, dp_cfg, loaded.seed)
    return ProviderBCStudent(backend, loaded.encoder_cfg)


def rows_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Runner rows as selection's CSV text (:func:`ila.train.select.write_rows_csv` format)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(runner.VISION_EPISODE_COLUMNS))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def count_flips(
    rows: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Outcome flips and text differences, episode by episode in list order.

    Raises:
        ValueError: If the two flights are not the same episodes in the same order.
    """
    if len(rows) != len(reference):
        raise ValueError(f"{len(rows)} rows vs {len(reference)} reference rows")
    keys = ("regime", "ss", "index", "episode_seed")
    flips: list[dict[str, str]] = []
    differ = 0
    for r, f in zip(rows, reference, strict=True):
        if any(str(r[k]) != str(f[k]) for k in keys):
            raise ValueError(
                f"episode order differs: {[r[k] for k in keys]} vs {[f[k] for k in keys]}"
            )
        if any(str(r[c]) != str(f[c]) for c in runner.VISION_EPISODE_COLUMNS):
            differ += 1
        if str(r["outcome"]) != str(f["outcome"]):
            flips.append(
                {
                    "ss": str(r["ss"]),
                    "index": str(r["index"]),
                    "reference": str(f["outcome"]),
                    "provider": str(r["outcome"]),
                }
            )
    return {"flips": len(flips), "flipped": flips, "rows_text_differ": differ}


def _selection_rows(ckpt: ev_arms.Checkpoint) -> tuple[Path, str]:
    sel = json.loads((p8_adapter.REPO_ROOT / ckpt.selection_json).read_text("utf-8"))
    path = (
        (p8_adapter.REPO_ROOT / ckpt.selection_json).parent
        / "select"
        / sel["selected"]["rows_file"]
    )
    return path, path.read_text("utf-8")


def fly_provider(
    arm: str,
    provider: str,
    settings: BenchSettings,
    episodes: Sequence[p8_adapter.ListedEpisode],
    *,
    graph_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    """Fly one arm on one provider (or the reference path) through the runner."""
    loaded = load_arm(arm, settings.arm_seeds[arm], settings.selection_pins)
    ck = loaded.ckpt
    if provider == settings.reference:
        from ila.train.select import fly_checkpoint

        return fly_checkpoint(
            ck.path,
            episodes,
            arm=arm,
            run_seed=loaded.seed,
            sha256=ck.sha256,
            workers=workers,
            chunk=settings.chunk,
            teacher=ck.teacher,
        )
    build = functools.partial(
        provider_policy,
        arm,
        provider,
        str(graph_dir),
        str(settings.selection_pins),
        str(settings.out_dir / "trt_cache"),
    )
    spec = p8_adapter.callable_spec(arm, build, run_seed=loaded.seed)
    tags = runner.RowTags(
        arm=arm, teacher=ck.teacher, student_seed=loaded.seed, ckpt_sha256=ck.sha256
    )
    return runner.run_list_vision(
        episodes,
        spec,
        p8_adapter.load_eval_configs(),
        tags,
        render="full",
        workers=workers,
        chunk=settings.chunk,
        contract_variant=loaded.obs_variant,
    )


def _prebuild_trt(provider: str, arm: str, settings: BenchSettings, graph_dir: Path) -> None:
    """Build (or load) the batch-1 TensorRT engines once, before any worker starts."""
    from ila.deploy.inputs import load_inputs

    dp = load_arm("dp", 0, settings.selection_pins).net
    backend = bk.OrtBackend(
        provider,
        graph_dir,
        dp.cfg,  # type: ignore[arg-type]
        batch=1,
        trt_cache=settings.out_dir / "trt_cache" / provider,
    )
    inp = load_inputs(settings.out_dir / "inputs" / "select_inputs.npz")
    img, prop = inp.images[:1], inp.proprio[:1]
    if arm == "bc_dagger":
        backend.bc(img, prop)
    else:
        backend.dp_e2e(img, prop, np.zeros((1, 16, 3), np.float32))


def run_closed_loop(
    settings: BenchSettings,
    *,
    arms: Sequence[str],
    providers: Sequence[str],
    graph_dir: Path,
    open_loop: Mapping[str, Mapping[str, Any]] | None,
    include_failed: bool = False,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Fly the reference and each provider for each arm; write rows and records.

    Args:
        settings: Bench settings.
        arms: Arms to fly.
        providers: Providers to fly (the reference is always flown first, or reused).
        graph_dir: Exported graphs.
        open_loop: ``parity.json``'s per-provider records; a provider whose open-loop parity
            failed is refused unless ``include_failed`` (then it is flown and reported, and it
            still fails parity).
        include_failed: Fly open-loop failures too.
        force: Re-fly even when a record exists.
        log: Progress sink.

    Returns:
        ``{arm: {provider: record}}``.
    """
    p8_adapter.latency_harness().disable_tf32()
    episodes = data_episodes.read_split("select")
    out_root = settings.results_dir / "closed_loop"
    git = git_state()
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        adir = out_root / arm
        adir.mkdir(parents=True, exist_ok=True)
        ref_path = adir / f"{settings.reference}_rows.csv"
        loaded = load_arm(arm, settings.arm_seeds[arm], settings.selection_pins)
        sel_path, sel_text = _selection_rows(loaded.ckpt)
        if force or not ref_path.is_file():
            t0 = time.perf_counter()
            ref_rows = fly_provider(
                arm,
                settings.reference,
                settings,
                episodes,
                graph_dir=graph_dir,
                workers=settings.workers_cpu,
            )
            ref_text = rows_text(ref_rows)
            ref_path.write_text(ref_text, "utf-8")
            _write_record(
                adir,
                settings.reference,
                arm,
                settings,
                {
                    "status": "reference",
                    "n": len(ref_rows),
                    "flips": 0,
                    "rows_text_differ": 0,
                    "reference_equals_selection_rows": ref_text == sel_text,
                    "selection_rows_file": ev_arms._rel(sel_path),
                    "selection_rows_sha256": sha256_file(sel_path),
                    "selection_rows_flips": count_flips(ref_rows, _read_rows(sel_text))["flips"],
                    "outcomes": dict(sorted(Counter(str(r["outcome"]) for r in ref_rows).items())),
                    "workers": settings.workers_cpu,
                    "wall_s": time.perf_counter() - t0,
                    "git": git,
                    "checkpoint_sha256": loaded.ckpt.sha256,
                },
            )
            same = ref_text == sel_text
            log(f"{arm} reference: {len(ref_rows)} rows; equals selection rows: {same}")
        reference = _read_rows(ref_path.read_text("utf-8"))
        results[arm] = {}
        for provider in providers:
            if provider == settings.reference:
                continue
            rec_path = adir / f"{provider}.json"
            if rec_path.is_file() and not force:
                kept = json.loads(rec_path.read_text("utf-8"))
                if str(kept.get("status", "")).startswith("flown"):
                    results[arm][provider] = kept
                    log(f"{arm} {provider}: flown before, kept")
                    continue
            ol = (open_loop or {}).get(provider, {}).get(arm)
            if (ol is None or not ol.get("open_loop_pass", False)) and not include_failed:
                rec = {"status": "refused (open-loop parity not passed)", "n": 0}
                results[arm][provider] = _write_record(adir, provider, arm, settings, rec)
                log(f"{arm} {provider}: refused (open-loop parity)")
                continue
            gpu = provider in GPU_PROVIDERS
            workers = settings.workers_gpu if gpu else settings.workers_cpu
            t0 = time.perf_counter()
            try:
                if provider.startswith("ort_trt"):
                    _prebuild_trt(provider, arm, settings, graph_dir)
                rows = fly_provider(
                    arm, provider, settings, episodes, graph_dir=graph_dir, workers=workers
                )
            except Exception as exc:  # recorded with no number (a refused or failed row)
                rec = {
                    "status": f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
                    "n": 0,
                }
                results[arm][provider] = _write_record(adir, provider, arm, settings, rec)
                log(f"{arm} {provider}: FAILED {rec['status']}")
                continue
            (adir / f"{provider}_rows.csv").write_text(rows_text(rows), "utf-8")
            fl = count_flips(rows, reference)
            rec = {
                "status": "flown"
                if (ol or {}).get("open_loop_pass")
                else "flown (open-loop parity failed)",
                "n": len(rows),
                **fl,
                "max_flips": settings.closed_loop_max_flips,
                "closed_loop_pass": bool(fl["flips"] <= settings.closed_loop_max_flips),
                "outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items())),
                "workers": workers,
                "wall_s": time.perf_counter() - t0,
                "git": git,
                "checkpoint_sha256": loaded.ckpt.sha256,
            }
            results[arm][provider] = _write_record(adir, provider, arm, settings, rec)
            log(f"{arm} {provider}: {fl['flips']} flips, {fl['rows_text_differ']} rows differ")
    return results


def _write_record(
    adir: Path, provider: str, arm: str, settings: BenchSettings, rec: Mapping[str, Any]
) -> dict[str, Any]:
    from ila.deploy.export import load_meta

    meta_path = settings.out_dir / "graphs" / "meta.json"
    full = {
        "arm": arm,
        "provider": provider,
        "scored": settings.is_scored(provider),
        "graphs_meta_sha256": sha256_file(meta_path) if meta_path.is_file() else None,
        "graphs": {
            k: v["sha256"]
            for k, v in (
                load_meta(meta_path.parent)["graphs"] if meta_path.is_file() else {}
            ).items()
        },
        **rec,
    }
    (adir / f"{provider}.json").write_text(
        json.dumps(full, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    return full


def closed_loop_csv(settings: BenchSettings, root: Path | None = None) -> str:
    """Render ``closed_loop.csv`` from the per-(arm, provider) records on disk."""
    root = settings.results_dir / "closed_loop" if root is None else root
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CLOSED_LOOP_COLUMNS)
    for arm in ("bc_dagger", "dp"):
        for provider in settings.providers:
            path = root / arm / f"{provider}.json"
            if not path.is_file():
                continue
            r = json.loads(path.read_text("utf-8"))
            outcomes = r.get("outcomes", {})
            w.writerow(
                [
                    arm,
                    provider,
                    r.get("scored", False),
                    r.get("status", ""),
                    r.get("n", 0),
                    "" if r.get("n", 0) == 0 else r.get("flips", ""),
                    settings.closed_loop_max_flips,
                    r.get(
                        "closed_loop_pass", "reference" if provider == settings.reference else False
                    ),
                    r.get("rows_text_differ", ""),
                    outcomes.get("success", ""),
                    ";".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
                    r.get("reference_equals_selection_rows", ""),
                    r.get("workers", ""),
                    "" if "wall_s" not in r else f"{float(r['wall_s']):.1f}",
                    (r.get("git") or {}).get("ila_commit", ""),
                ]
            )
    return buf.getvalue()
