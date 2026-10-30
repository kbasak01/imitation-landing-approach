"""Re-fly the smoke checkpoints on the 8 smoke train episodes over run_seed 0..9, recording traces.

Analysis only (scratchpad). Flight = ila.eval.runner.fly_chunk (the library runner loop) with the
library students (OnnxDPStudent via load_onnx_student, DPStudent via load_student) and the P8 ppo
teacher. A thin recording wrapper around the library StudentActor/TeacherActor stores, per step,
P8's full observation (for analysis: rel_position etc. -- never fed to the student; the student
still only gets act_vision(images, proprio) from StudentActor), the executed action, and every
replanned chunk (normalised-space pre-unnormalise sample for R1, final chunk for both).
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/kbasak/projects_02/imitation-landing-approach")
R1_BUNDLE = REPO / "artifacts/runs/smoke/R1_lerobot_dp/onnx/step10000"
DP_EMA = REPO / "artifacts/runs/smoke/dp/ema_last.pt"
DP_RAW = REPO / "artifacts/runs/smoke/dp/checkpoints/s0_step010000.pt"


def build_r1(run_seed: int, cfgs: Any, ta: int | None = None) -> Any:
    from ila.policies.onnx_dp import load_onnx_student

    del cfgs
    return load_onnx_student(R1_BUNDLE, run_seed, n_action_steps=ta)


def build_dp(path: str, weights: str, run_seed: int, cfgs: Any, ta: int | None = None) -> Any:
    from ila.policies.registry import load_student

    del cfgs
    if weights == "ema":
        s = load_student(Path(path), "cpu", weights="ema", n_action_steps=ta)
    else:
        assert ta in (None, 1)
        # registry.load_student(weights="raw") looks up ckpt["raw"], but loop.py saves the raw
        # weights under "model"; rebuild with the registry's own helpers from "model".
        from ila.policies.registry import _wrap, build_net_from_config, read_checkpoint

        ck = read_checkpoint(Path(path))
        net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
        net.load_state_dict(ck["model"])
        s = _wrap(net, int(ck["run_seed"]), "cpu", ta, None, str(ck["arm"]), False)
    s.run_seed = int(run_seed)  # noise key (run_seed, episode_seed, t); checkpoint's is 0
    return s


def build_teacher(cfgs: Any) -> Any:
    from ila import p8_adapter

    return p8_adapter.teacher("ppo", 0, cfgs)


class Recorder:
    """Wraps the library actor; records obs/action and the student's replans."""

    def __init__(self, inner: Any, policy: Any) -> None:
        self.inner = inner
        self.needs_frames = inner.needs_frames
        self.policy = policy
        self.episodes: list[dict[str, Any]] = []
        self.cur: dict[str, Any] | None = None
        if hasattr(policy, "sample_batch"):
            orig = policy.sample_batch

            def sb(images, proprio, seeds, num_inference_steps=None):  # type: ignore[no-untyped-def]
                out = orig(images, proprio, seeds, num_inference_steps)
                if self.cur is not None:
                    self.cur["chunks"].append((int(seeds[0][2]), np.array(out[0], np.float32)))
                return out

            policy.sample_batch = sb
        if hasattr(policy, "unnormalise"):
            orig_u = policy.unnormalise

            def un(x):  # type: ignore[no-untyped-def]
                if self.cur is not None:
                    self.cur["x_norm"].append(np.array(x[0], np.float32))
                return orig_u(x)

            policy.unnormalise = un

    def begin(self, listed: Any, context: Any) -> None:
        self.cur = {"episode_seed": listed.episode_seed, "obs": [], "act": [], "chunks": [],
                    "x_norm": []}
        self.episodes.append(self.cur)
        self.inner.begin(listed, context)

    def act(self, step: int, obs: Any, frame: Any) -> Any:
        a = self.inner.act(step, obs, frame)
        assert self.cur is not None
        self.cur["obs"].append(np.array(obs, np.float32))
        self.cur["act"].append(np.array(a, np.float64))
        return a


def run_task(job: tuple[str, int, list[Any]]) -> dict[str, Any]:
    from ila import p8_adapter
    from ila.eval import runner

    variant, run_seed, episodes = job
    if variant == "teacher":
        build = build_teacher
    elif variant == "r1":
        build = functools.partial(build_r1, run_seed)
    elif variant == "dp_ema":
        build = functools.partial(build_dp, str(DP_EMA), "ema", run_seed)
    elif variant == "dp_raw":
        build = functools.partial(build_dp, str(DP_RAW), "raw", run_seed)
    elif variant == "r1_ta1":
        build = functools.partial(build_r1, run_seed, ta=1)
    elif variant == "dp_raw_ta1":
        build = functools.partial(build_dp, str(DP_RAW), "raw", run_seed, ta=1)
    elif variant == "dp_ema_ta1":
        build = functools.partial(build_dp, str(DP_EMA), "ema", run_seed, ta=1)
    else:
        raise ValueError(variant)
    cfgs = p8_adapter.load_eval_configs()
    task = runner.VisionTask(
        episodes=episodes,
        spec=p8_adapter.callable_spec(f"inv:{variant}", build, run_seed=run_seed),
        cfgs=cfgs,
        tags=runner.RowTags(arm=f"inv:{variant}", student_seed=run_seed),
        render="full",
    )
    recs: list[Recorder] = []

    def make(cfgs_: Any, contract: Any) -> tuple[Any, object]:
        policy = build(cfgs_)
        if runner.is_student(policy):
            inner: Any = runner.StudentActor(policy, contract)
        else:
            inner = runner.TeacherActor(policy)
        rec = Recorder(inner, policy)
        recs.append(rec)
        return rec, policy

    t0 = time.perf_counter()
    rows = runner.fly_chunk(task, make)
    traces = []
    for ep in recs[0].episodes:
        traces.append({
            "episode_seed": ep["episode_seed"],
            "obs": np.stack(ep["obs"]),
            "act": np.stack(ep["act"]),
            "chunks": ep["chunks"],
            "x_norm": np.stack(ep["x_norm"]) if ep["x_norm"] else None,
        })
    return {"variant": variant, "run_seed": run_seed, "rows": rows, "traces": traces,
            "wall_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["teacher", "r1", "dp_ema", "dp_raw"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ila.data import episodes as data_episodes
    from ila.data.store import DemoStore
    from ila.eval import runner
    from ila.train import loop
    from ila.train.config import TrainConfig
    from ila.policies.registry import REGISTRY
    from ila.train.sampling import choose_episodes
    from ila import p8_adapter

    cfg = TrainConfig.load(REGISTRY["dp"].config)
    root = p8_adapter.load_paths()["data_root"]
    with DemoStore(loop.store_path(root, cfg.collection)) as store:
        choice = choose_episodes(store, 8, cfg.success_filter)
    assert list(choice.demo_indices) == list(range(8)), choice.demo_indices
    rows_all = data_episodes.read_split("demos_0-1999")
    listed = [rows_all[i] for i in choice.demo_indices]

    jobs: list[tuple[str, int, list[Any]]] = []
    for v in args.variants:
        for s in ([0] if v == "teacher" else args.seeds):
            jobs.append((v, s, listed))
    runner.set_single_thread_env()
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    out: list[dict[str, Any]] = []
    part = Path(args.out).with_suffix(".partial")
    part.mkdir(exist_ok=True)
    with ctx.Pool(processes=min(args.workers, len(jobs)),
                  initializer=runner.init_worker_threads) as pool:
        for i, res in enumerate(pool.imap_unordered(run_task, jobs)):
            out.append(res)
            with open(part / f"{res['variant']}_s{res['run_seed']}.pkl", "wb") as f:
                pickle.dump(res, f)
            succ = sum(r["outcome"] == "success" for r in res["rows"])
            print(f"[{i+1}/{len(jobs)}] {res['variant']} s{res['run_seed']}: {succ}/8 "
                  f"{[r['outcome'] for r in res['rows']]} ({res['wall_s']:.1f}s)", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump({"store_groups": list(choice.names), "demo_indices": list(choice.demo_indices),
                     "results": out, "wall_s": time.perf_counter() - t0}, f)
    print("wrote", args.out, f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
