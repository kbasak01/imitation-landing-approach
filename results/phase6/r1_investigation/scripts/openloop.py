"""Open-loop sampled executed-chunk error vs labels on the 8 smoke episodes (analysis only).

Same construction as ila.train.smoke.dp_sampled_mse (DPChunkDataset from the store, the student's
sample_batch with the flight noise key (run_seed, episode_seed, t), executed slice at Ta = 8), but
keeps the per-frame, per-executed-index, per-axis squared errors so they can be bucketed by steps
to the episode end, and repeats with the targets shifted by +-1 step (alignment probe).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

REPO = Path("/home/kbasak/projects_02/imitation-landing-approach")
R1_BUNDLE = REPO / "artifacts/runs/smoke/R1_lerobot_dp/onnx/step10000"
DP_EMA = REPO / "artifacts/runs/smoke/dp/ema_last.pt"
DP_RAW = REPO / "artifacts/runs/smoke/dp/checkpoints/s0_step010000.pt"


def _raw(path: Path):  # registry.load_student(weights="raw") looks up ckpt["raw"]; loop saves "model"
    from ila.policies.registry import _wrap, build_net_from_config, read_checkpoint

    ck = read_checkpoint(path)
    net = build_net_from_config(ck["kind"], ck["config"], int(ck["run_seed"]), str(ck["obs_variant"]))
    net.load_state_dict(ck["model"])
    return _wrap(net, int(ck["run_seed"]), "cpu", None, None, str(ck["arm"]), False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    from ila import p8_adapter
    from ila.data import episodes as data_episodes
    from ila.data.store import DemoStore
    from ila.obs.split import load_contract
    from ila.policies.dp import chunking
    from ila.policies.onnx_dp import load_onnx_student
    from ila.policies.registry import REGISTRY, load_student
    from ila.train import loop
    from ila.train.config import TrainConfig
    from ila.train.dataset import DPChunkDataset, FrameTable
    from ila.train.sampling import choose_episodes

    cfg = TrainConfig.load(REGISTRY["dp"].config)
    root = p8_adapter.load_paths()["data_root"]
    with DemoStore(loop.store_path(root, cfg.collection)) as store:
        choice = choose_episodes(store, 8, cfg.success_filter)
        table = FrameTable.from_store(store, list(choice.names), "default")
        labels = [store.labels(n) for n in choice.names]
    data = DPChunkDataset(table, load_contract().n_obs_steps, int(cfg.raw["horizons"]["horizon"]))
    rows_all = data_episodes.read_split("demos_0-1999")
    seeds_ep = [rows_all[i].episode_seed for i in choice.demo_indices]
    ex = chunking.executed_slice(data.n_obs_steps, data.horizon, 8)
    # Per-frame (episode, t, T) and shifted targets.
    ep_of, t_of, T_of = [], [], []
    for e, T in enumerate(table.episode_len):
        ep_of += [e] * int(T)
        t_of += list(range(int(T)))
        T_of += [int(T)] * int(T)
    ep_of, t_of, T_of = map(np.asarray, (ep_of, t_of, T_of))
    To = data.n_obs_steps

    def shifted_target(shift: int) -> np.ndarray:
        # executed index j (0..7) <-> step t + j; target label[clip(t + j + shift, 0, T-1)]
        out = np.zeros((len(t_of), 8, 3), np.float32)
        for i, (e, t, T) in enumerate(zip(ep_of, t_of, T_of)):
            idx = np.clip(t + np.arange(8) + shift, 0, T - 1)
            out[i] = labels[e][idx]
        return out

    students = {
        "r1": lambda: load_onnx_student(R1_BUNDLE, 0),
        "dp_ema": lambda: load_student(DP_EMA, "cpu", weights="ema"),
        "dp_raw": lambda: _raw(DP_RAW),
    }
    res: dict = {"ep": ep_of, "t": t_of, "T": T_of, "targets": {s: shifted_target(s) for s in (-2, -1, 0, 1, 2)}}
    # sanity: shift-0 target equals DPChunkDataset's executed target
    full = data.batch(np.arange(len(data), dtype=np.int64))
    assert np.array_equal(np.asarray(full["target"])[:, ex], res["targets"][0]), "target mismatch"
    for name, mk in students.items():
        st = mk()
        for rs in args.seeds:
            st.run_seed = rs
            chunks = []
            for start in range(0, len(data), 64):
                idx = np.arange(start, min(len(data), start + 64), dtype=np.int64)
                b = data.batch(idx)
                keys = [(rs, int(seeds_ep[ep_of[i]]), int(t_of[i])) for i in idx]
                chunks.append(st.sample_batch(b["images"], b["proprio"], keys, 10))
            ch = np.concatenate(chunks)
            res[(name, rs)] = ch
            err = (ch[:, ex].astype(np.float64) - res["targets"][0]) ** 2
            print(name, rs, "mse", err.mean(), flush=True)
    with open(args.out, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
