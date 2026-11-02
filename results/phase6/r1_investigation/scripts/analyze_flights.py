"""Analyse flights_s0-9.pkl: A (variance), B (failure anatomy), C (action range), D (raw dp)."""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
R1_MIN = np.array([-0.55811607837677, -0.4439578056335449, -1.0])
R1_MAX = np.array([0.6294874548912048, 0.563503086566925, 0.26473870873451233])
ORDER = ["crash", "off_pad", "hard_landing", "bounce", "success", "timeout"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    d = pickle.load(open(S / "flights_s0-9.pkl", "rb"))
    from ila import p8_adapter
    from ila.data.store import DemoStore
    from ila.obs.split import load_contract
    from ila.policies.registry import REGISTRY
    from ila.train import loop
    from ila.train.config import TrainConfig

    contract = load_contract()
    res = {(r["variant"], r["run_seed"]): r for r in d["results"]}
    variants = ["r1", "dp_ema", "dp_raw"]
    seeds = sorted({s for v, s in res if v != "teacher"})
    out: dict = {"A": {}, "B": {}, "C": {}, "D": {}}

    # ---------------- A / D: per-seed success and outcome classes
    succ = {}
    for v in variants:
        per_seed = []
        for s in seeds:
            rows = res[(v, s)]["rows"]
            c = Counter(r["outcome"] for r in rows)
            per_seed.append({"seed": s, "success": c["success"],
                             "outcomes": {o: c[o] for o in ORDER if c[o]},
                             "per_episode": [r["outcome"] for r in rows]})
        mat = np.array([[r["outcome"] == "success" for r in res[(v, s)]["rows"]] for s in seeds])
        succ[v] = mat
        k, n = int(mat.sum()), mat.size
        lo, hi = wilson(k, n)
        counts = mat.sum(1)
        pooled = Counter(r["outcome"] for s in seeds for r in res[(v, s)]["rows"])
        out["A"][v] = {
            "per_seed": per_seed,
            "success_counts": counts.tolist(),
            "mean_per_8": float(counts.mean()), "sd_per_8": float(counts.std(ddof=1)),
            "pooled": f"{k}/{n}", "pooled_rate": k / n, "wilson95": [lo, hi],
            "seeds_ge_7": int((counts >= 7).sum()),
            "per_episode_success_rate": mat.mean(0).tolist(),
            "pooled_outcomes": {o: pooled[o] for o in ORDER},
        }
    # paired-by-(seed, episode) bootstrap of the R1 - dp_ema success-rate difference, clustered
    # over episodes (8) and seeds (10)
    rng = np.random.default_rng(0)
    for a, b in (("r1", "dp_ema"), ("dp_raw", "dp_ema"), ("r1", "dp_raw")):
        diff = succ[a].astype(float) - succ[b].astype(float)
        boots = []
        for _ in range(20000):
            si = rng.integers(0, len(seeds), len(seeds))
            ei = rng.integers(0, 8, 8)
            boots.append(diff[np.ix_(si, ei)].mean())
        boots = np.array(boots)
        out["A"][f"diff_{a}_minus_{b}"] = {
            "point": float(diff.mean()),
            "ci95_two_way_cluster_bootstrap": [float(np.quantile(boots, 0.025)),
                                               float(np.quantile(boots, 0.975))],
        }
    # probability a single 8-episode smoke flight passes (>= 7) under each variant's per-episode
    # rates (independent Bernoulli per episode)
    for v in variants:
        p = succ[v].mean(0)
        # Poisson-binomial P(sum >= 7)
        dist = np.zeros(9); dist[0] = 1
        for pi in p:
            dist = np.concatenate([dist[:-1] * (1 - pi), [0]]) + np.concatenate([[0], dist[:-1] * pi])
        out["A"][v]["p_pass_ge7_model"] = float(dist[7:].sum())

    # ---------------- B: failure anatomy
    cfg = TrainConfig.load(REGISTRY["dp"].config)
    root = p8_adapter.load_paths()["data_root"]
    with DemoStore(loop.store_path(root, cfg.collection)) as store:
        demo_obs = [store.privileged_observations(n) for n in d["store_groups"]]
        demo_lab = [store.labels(n) for n in d["store_groups"]]
    teacher = p8_adapter.teacher("ppo", 0, p8_adapter.load_eval_configs())
    trow = res[("teacher", 0)]["rows"]
    ttr = res[("teacher", 0)]["traces"]
    # the teacher re-fly must equal the stored demo (obs and labels)
    out["B"]["teacher_refly_equals_demo"] = [
        bool(np.array_equal(ttr[e]["obs"], demo_obs[e][: len(ttr[e]["obs"])]))
        and len(ttr[e]["obs"]) == len(demo_obs[e]) for e in range(8)]
    out["B"]["teacher_rows"] = [
        {"demo": e, "outcome": r["outcome"], "steps": r["steps"],
         "td_step": float(r["td_t_episode_s"]) * 30, "lateral_offset_m": float(r["lateral_offset_m"]),
         "closing_speed_m_s": float(r["closing_speed_normal_m_s"])} for e, r in enumerate(trow)]

    def rp(o: np.ndarray) -> np.ndarray:
        return contract.audit_field(o, "rel_position")

    def anatomy(v: str, s: int, e: int) -> dict:
        r = res[(v, s)]["rows"][e]
        tr = res[(v, s)]["traces"][e]
        obs, act = tr["obs"], tr["act"]
        td = float(r["td_t_episode_s"]) if r["td_t_episode_s"] == r["td_t_episode_s"] else None
        td_step = None if td is None else int(round(td * 30))
        n = len(obs)
        T_demo = len(demo_obs[e])
        hi = min(td_step if td_step is not None else n, n, T_demo)
        lo = max(0, hi - 15)
        p_s, p_d = rp(obs[lo:hi]), rp(demo_obs[e][lo:hi])
        dxy = np.linalg.norm(p_s[:, :2] - p_d[:, :2], axis=1)
        dz = p_s[:, 2] - p_d[:, 2]
        # teacher's action on the student's own states (ppo is stateless given obs; analysis only)
        teacher.reset(int(tr["episode_seed"]), None)
        t_on_s = np.stack([np.asarray(teacher.act(o), np.float64) for o in obs[lo:hi]])
        a_err = act[lo:hi] - t_on_s
        a_vs_demo = act[lo:hi] - demo_lab[e][lo:hi]
        return {
            "variant": v, "seed": s, "demo": e, "outcome": r["outcome"], "steps": int(r["steps"]),
            "td_step": td_step, "teacher_td_step": round(float(trow[e]["td_t_episode_s"]) * 30),
            "lateral_offset_m": float(r["lateral_offset_m"]),
            "teacher_lateral_offset_m": float(trow[e]["lateral_offset_m"]),
            "closing_speed_m_s": float(r["closing_speed_normal_m_s"]),
            "rel_xy_at_td_student_m": p_s[-1, :2].round(4).tolist() if len(p_s) else None,
            "rel_xy_same_t_demo_m": p_d[-1, :2].round(4).tolist() if len(p_d) else None,
            "window_steps": [lo, hi],
            "dist_xy_to_demo_m": {"mean": float(dxy.mean()), "max": float(dxy.max()),
                                   "first": float(dxy[0]), "last": float(dxy[-1])},
            "dz_to_demo_m_last": float(dz[-1]),
            "rmse_action_vs_teacher_on_student_state": np.sqrt((a_err ** 2).mean(0)).round(4).tolist(),
            "rmse_action_vs_demo_label_same_t": np.sqrt((a_vs_demo ** 2).mean(0)).round(4).tolist(),
            "mean_action_err_vs_teacher_xy": a_err.mean(0).round(4).tolist(),
        }

    out["B"]["seed0_failures"] = [anatomy("r1", 0, 1), anatomy("r1", 0, 3), anatomy("dp_ema", 0, 0)]
    # all episodes, all seeds: lateral offset distributions, and the teacher-on-student error
    lat = {}
    for v in variants:
        offs = np.array([float(r["lateral_offset_m"]) for s in seeds for r in res[(v, s)]["rows"]])
        offs = offs[np.isfinite(offs)]
        lat[v] = {"n": int(offs.size), "mean": float(offs.mean()), "median": float(np.median(offs)),
                  "p90": float(np.quantile(offs, 0.9)), "frac_gt_0.10": float((offs > 0.10).mean())}
        an = [anatomy(v, s, e) for s in seeds for e in range(8)]
        lat[v]["last15_dist_xy_to_demo_mean_m"] = float(np.mean([a["dist_xy_to_demo_m"]["mean"] for a in an]))
        lat[v]["last15_dist_xy_to_demo_last_m"] = float(np.mean([a["dist_xy_to_demo_m"]["last"] for a in an]))
        lat[v]["last15_rmse_action_vs_teacher_on_student_state"] = np.sqrt(np.mean(
            [np.array(a["rmse_action_vs_teacher_on_student_state"]) ** 2 for a in an], axis=0)).round(4).tolist()
        fails = [a for a in an if a["outcome"] != "success"]
        oks = [a for a in an if a["outcome"] == "success"]
        lat[v]["fail_vs_success_last15_dist_xy_mean_m"] = [
            float(np.mean([a["dist_xy_to_demo_m"]["mean"] for a in fails])) if fails else None,
            float(np.mean([a["dist_xy_to_demo_m"]["mean"] for a in oks])) if oks else None]
        lat[v]["td_step_minus_teacher_td_step_mean"] = float(np.mean(
            [a["td_step"] - a["teacher_td_step"] for a in an if a["td_step"] is not None]))
    tl = np.array([float(r["lateral_offset_m"]) for r in trow])
    lat["teacher"] = {"n": 8, "mean": float(tl.mean()), "median": float(np.median(tl)),
                      "max": float(tl.max())}
    out["B"]["lateral_offset_all"] = lat

    # ---------------- C: action range
    for v in variants:
        ex_all = np.concatenate([res[(v, s)]["traces"][e]["act"] for s in seeds for e in range(8)])
        below = ex_all < R1_MIN - 1e-6
        above = ex_all > R1_MAX + 1e-6
        rec = {
            "n_executed_steps": int(len(ex_all)),
            "frac_outside_R1_range_per_axis": (below | above).mean(0).round(4).tolist(),
            "frac_steps_any_axis_outside": float((below | above).any(1).mean()),
            "frac_below_min_per_axis": below.mean(0).round(4).tolist(),
            "frac_above_max_per_axis": above.mean(0).round(4).tolist(),
            "max_excess_per_axis": np.maximum((R1_MIN - ex_all).max(0), (ex_all - R1_MAX).max(0)).round(4).tolist(),
            "executed_min": ex_all.min(0).round(4).tolist(),
            "executed_max": ex_all.max(0).round(4).tolist(),
        }
        if v == "r1":
            # normalised-space chunks (pre-unnormalise, post DDIM clip) at each replan; executed
            # slice 1..8 of each; the last replan may be cut by termination (count executed only)
            atb, atb_all, nx = [], [], 0
            for s in seeds:
                for e in range(8):
                    tr = res[(v, s)]["traces"][e]
                    n = len(tr["act"])
                    for (t, _), x in zip(tr["chunks"], tr["x_norm"]):
                        k = min(8, n - t)
                        xe = x[1:1 + k]
                        atb.append(np.abs(xe) >= 1 - 1e-6)
                        atb_all.append(np.abs(x) >= 1 - 1e-6)
            atb = np.concatenate(atb)
            atb_all = np.stack(atb_all)
            rec["executed_frac_at_minmax_bound_per_axis"] = atb.mean(0).round(4).tolist()
            rec["executed_frac_at_bound_any_axis"] = float(atb.any(1).mean())
            rec["executed_frac_at_lower_vs_upper"] = None
            rec["chunk_all16_frac_at_bound_per_axis"] = atb_all.reshape(-1, 3).mean(0).round(4).tolist()
            # equivalently, executed action within 1e-5 of R1_MIN/R1_MAX
            near = (np.abs(ex_all - R1_MIN) < 1e-5) | (np.abs(ex_all - R1_MAX) < 1e-5)
            rec["executed_action_equal_bound_per_axis"] = near.mean(0).round(4).tolist()
            rec["executed_action_at_lower_per_axis"] = (np.abs(ex_all - R1_MIN) < 1e-5).mean(0).round(4).tolist()
            rec["executed_action_at_upper_per_axis"] = (np.abs(ex_all - R1_MAX) < 1e-5).mean(0).round(4).tolist()
        out["C"][v] = rec
    # teacher labels/actions on its own flights and teacher-on-student-state actions: outside range?
    tex = np.concatenate([t["act"] for t in ttr])
    out["C"]["teacher_refly_frac_outside_R1_range_per_axis"] = (
        (tex < R1_MIN - 1e-6) | (tex > R1_MAX + 1e-6)).mean(0).round(4).tolist()
    tos = []
    for s in seeds:
        for e in range(8):
            teacher.reset(int(res[("dp_ema", s)]["traces"][e]["episode_seed"]), None)
            for o in res[("dp_ema", s)]["traces"][e]["obs"]:
                tos.append(np.asarray(teacher.act(o), np.float64))
    tos = np.stack(tos)
    out["C"]["teacher_on_dp_ema_states_frac_outside_R1_range_per_axis"] = (
        (tos < R1_MIN - 1e-6) | (tos > R1_MAX + 1e-6)).mean(0).round(4).tolist()
    tos = []
    for s in seeds:
        for e in range(8):
            teacher.reset(int(res[("r1", s)]["traces"][e]["episode_seed"]), None)
            for o in res[("r1", s)]["traces"][e]["obs"]:
                tos.append(np.asarray(teacher.act(o), np.float64))
    tos = np.stack(tos)
    out["C"]["teacher_on_r1_states_frac_outside_R1_range_per_axis"] = (
        (tos < R1_MIN - 1e-6) | (tos > R1_MAX + 1e-6)).mean(0).round(4).tolist()

    (S / "analysis_flights.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
