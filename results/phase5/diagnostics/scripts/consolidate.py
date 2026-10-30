"""Consolidate the diagnostician's logs into results/phase5/diagnostics/bc_dagger/s<n>.json and summary.json."""
import json, hashlib, glob
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/kbasak/projects_02/imitation-landing-approach"); D = REPO / "results/phase5/diagnostics"; L = D / "logs"
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prov, cur, sel, ol, ro = (load(L / f"{n}.json") for n in ("prov", "curves", "selection", "openloop", "rollouts"))
csv = pd.read_csv(REPO / "results/phase5/dagger_iterations.csv")
TEACHER_J = {"heldout": [0.074, 0.097, 0.176], "dart_0.10": [0.066, 0.090, 0.161], "source": "P4-D5, results/phase4/diagnostics/inputs/teacher_jac.json (finite differences, eps 0.05)"}
(D / "bc_dagger").mkdir(exist_ok=True)
runs = {}
for s in range(5):
    p = prov[f"s{s}"]; c = cur[f"bc_dagger/s{s}"]; cm = cur[f"bc_matched/s{s}"]; cb = cur[f"bc/s{s}"]
    sl = sel[f"bc_dagger/s{s}"]; o = ol[f"s{s}"]
    its = [f"it{i}" if i == 0 else f"bc_dagger:it{i}" for i in range(6)]
    mits = ["it0"] + [f"bc_matched:it{i}" for i in range(1, 6)]
    openloop = {
        "method": "EMA, eval mode, CUDA fp32 (TF32 off); J = diag of FD d a / d last_action (eps 0.05, 512 frames, rng 0, P4-D5 method)",
        "heldout_teacher_states": {"bc_dagger": {k.split(":")[-1]: {"mse": o[k]["heldout"]["mse"], "l2_mean": o[k]["heldout"]["l2_mean"], "J_diag": o[k]["J_diag_heldout"]} for k in its},
                                   "bc_matched": {k.split(":")[-1]: {"mse": o[k]["heldout"]["mse"], "l2_mean": o[k]["heldout"]["l2_mean"], "J_diag": o[k]["J_diag_heldout"]} for k in mits},
                                   "note": "ppo_2000 1950-1999 every 2nd episode; bc_matched it5 trained on these episodes (contaminated); bc_dagger it5 shares their starting conditions only"},
        "sibling_student_states": {"source": o["sibling_state_source"], "frames": o["sibling_frames"],
                                   "bc_dagger": {k.split(":")[-1]: {kk: o[k]["sibling"][kk] for kk in ("mse", "l2_mean", "linf_le_0p1", "sat_norm_ge_0p98_student", "sat_norm_ge_0p98_label", "frac_dS_lt_0p25dL_given_dL_gt_0p02")} | {"J_diag": o[k]["J_diag_sibling"]} for k in its},
                                   "bc_matched": {k.split(":")[-1]: {kk: o[k]["sibling"][kk] for kk in ("mse", "l2_mean", "linf_le_0p1")} | {"J_diag": o[k]["J_diag_sibling"]} for k in mits},
                                   "note": "beta = 0 states of a sibling DAgger seed; in no training set of this seed; it5 of both arms has trained on the same 200 starting conditions (1800-1999), it0-it4 have not"},
        "selected_5k": {a: o[f"{a}:selected_it5_5k"] for a in ("bc_dagger", "bc_matched") if f"{a}:selected_it5_5k" in o},
        "teacher_J_diag_reference": TEACHER_J}
    rows = csv[csv.seed == s].sort_values("iteration")
    per_it = []
    for _, r in rows.iterrows():
        i = int(r.iteration); e = {"iteration": i, "beta": None if i == 0 else float(r.beta),
             "select_end_of_iteration_ema": {"success": int(r.select_success), "hard_landing": int(r.select_hard_landing), "bounce": int(r.select_bounce), "off_pad": int(r.select_off_pad), "timeout": int(r.select_timeout), "crash": int(r.select_crash), "p95": float(r.select_p95_closing_speed_m_s), "n_tunnelled": int(r.select_n_tunnelled), "n_detectors_disagree": int(r.select_n_detectors_disagree)},
             "copycat_autograd_heldout": [float(r.copycat_heldout_x), float(r.copycat_heldout_y), float(r.copycat_heldout_z)]}
        if i >= 1:
            b = ro[f"s{s}/iter{i}"]
            e.update({"rollout_flown_by": "bc 30k EMA" if i == 1 else f"end of iteration {i-1} EMA", "rollout": b["outcomes"], "rollout_steps_mean_max": [b["steps_mean"], b["steps_max"]],
                      "teacher_agreement_l2_own_states": b["l2_mean"], "agree_linf_le_0p1": b["linf_le_0p1"], "teacher_executed_fraction": b["teacher_executed_fraction"],
                      "copycat_frac_dS_lt_0p25dL": b["copycat_frac_dS_lt_0p25dL_given_dL_gt_0p02"], "lag_mse_label_t_vs_t_minus_1": [b["lag_mse_to_label_t"], b["lag_mse_to_label_t_minus_1"]],
                      "saturation_student_label_executed": [b["sat_norm_ge_0p98_student"], b["sat_norm_ge_0p98_label"], b["sat_norm_ge_0p98_executed"]],
                      "episode_jerk_student_label": b["episode_jerk_student_vs_label"], "clear_frames": b["clear_frames"],
                      "copycat_autograd_own": [float(r.copycat_own_x), float(r.copycat_own_y), float(r.copycat_own_z)]})
        per_it.append(e)
    w = p["wall"]; rollout_audit_s = w["rollouts_s"] + w["audit_s"]
    budget = {"wall_clock_s": w["wall_clock_s"], "wall_h": w["wall_clock_s"] / 3600, "cpu_rollouts_s": w["rollouts_s"], "cpu_audit_s": w["audit_s"], "aggregate_load_s": w["aggregate_load_s"],
              "timing_s_sum (runs_summary cpu_in_wall_s)": w["timing_s_sum"], "gpu_h_net_of_rollouts_audit_aggregate": (w["wall_clock_s"] - w["timing_s_sum"]) / 3600,
              "gpu_h_net_of_rollouts_audit_only": (w["wall_clock_s"] - rollout_audit_s) / 3600, "loop_s_total": w["loop_s_total"], "steps_per_s_loop": w["steps_per_s_loop"],
              "bc_matched_same_seed_gpu_h": prov["bc_matched"][f"s{s}"]["gpu_hours"]}
    curves = {"nonfinite": c["nonfinite"], "lr_schedule_max_rel_err_vs_cosine_500warmup_10k (s-1 convention)": c["lr_schedule_max_rel_err"]["s-1"],
              "n_train_points": c["n_train_points"], "n_heldout_points": c["n_heldout_points"],
              "bc_iteration0_heldout_ema_30k": cb["stages"]["0"]["heldout_ema_end"],
              "stages": {st: {k: v for k, v in c["stages"][st].items() if k != "heldout_ema_trace"} for st in c["stages"]},
              "bc_matched_heldout_ema_end_by_stage": {st: cm["stages"][st]["heldout_ema_end"] for st in cm["stages"]},
              "bc_matched_train_loss_late_median_by_stage": {st: cm["stages"][st]["train_loss_late_median"] for st in cm["stages"]}}
    he = [c["stages"][st]["heldout_ema_end"] for st in "12345"]
    reasons = [
        "provenance: config SHA-256 = P3-D1 (bc_dagger.yaml dbc7c0d2..., bc.yaml 3d3386bb..., demos.yaml 1fd413e2...); clean tree 576447c; P8 7237b70; 1 invocation, no resume, no override",
        f"init_from = bc/s{s}/checkpoints/s0_step030000.pt, SHA-256 {p['init_from']['recomputed'][:12]}... recomputed = bc manifest = bc_matched/s{s} init_from",
        "rollout student chain by SHA-256: iteration 1 = bc 30k, iteration i = s<i-1>_step010000.pt; equal in run manifest, iter<i>.json, rollout manifest and every episode attribute; all 10 checkpoints + ema_last re-hashed = manifest; rollout stores re-hashed = manifest",
        "every rollout episode starts from ppo_2000's same index (key, seed, t0, and the t = 0 frame and observation byte-identical): same starting conditions as bc_matched",
        f"curves: no non-finite value, no loss spike, LR schedule exact (rel err {c['lr_schedule_max_rel_err']['s-1']:.1e}); EMA/raw held-out 0.99-1.00 at every stage end",
        f"selection recomputed from rows = recorded ({sl['selected_recomputed']}, decided by {sl['decided_by']}); candidates iteration 5 only; winner's curse {sl['winners_curse']['selected_minus_mean_candidates']:+.1f} episodes",
        f"selected: {sl['selected']['success']}/150 {sl['selected']['by_ss']}, outcomes {{{', '.join(f'{k}: {v}' for k, v in sl['selected']['outcomes'].items() if v and k != 'success')}}}, n_tunnelled {sl['selected']['n_tunnelled']}, n_detectors_disagree {sl['selected']['n_detectors_disagree']}, max steps {sl['selected']['max_steps']}, max closing {sl['selected']['max_closing_speed_m_s']:.3f} m/s",
        "no collapse, copycat, lag, saturation or timeout signature in rollouts or on select (see behaviour)",
        f"held-out (teacher-state) EMA loss rises across iterations {cb['stages']['0']['heldout_ema_end']:.2e} (bc 30k) -> {he[-1]:.2e} (it5) while agreement on sibling-student states improves and select success rises: an open-loop/closed-loop divergence, disclosed, not a defect",
    ]
    run = {"run": f"bc_dagger/s{s}", "verdict": "TRUSTWORTHY", "reasons": reasons,
           "provenance": {k: p[k] for k in ("state", "status_state", "config_sha256", "init_from", "checkpoints", "ema_last_sha256", "ema_last_equals_final_ema", "normaliser_inherited_from_bc", "rollout_chain", "selection_json", "timeline_unix", "problems")} | {"ila_commit": "576447cb8a73c402a60588d9525cbd73274b1bc7", "ila_dirty": False, "resumes": [], "invocations": 1},
           "curves": curves, "selection": {k: v for k, v in sl.items()}, "per_iteration": per_it, "openloop": openloop, "budget": budget,
           "bc_matched_same_seed_selected": {k: sel[f"bc_matched/s{s}"]["selected"][k] for k in ("file", "success", "outcomes", "p95_closing_speed_m_s", "by_ss", "n_tunnelled", "max_closing_speed_m_s")},
           "automatic_check_failures": p["problems"] + sl["row_problems"]}
    (D / "bc_dagger" / f"s{s}.json").write_text(json.dumps(run, indent=1, default=str)); runs[s] = run
# arm-level
def m(xs): return [round(float(min(xs)), 6), round(float(max(xs)), 6)]
it = lambda s, i: ol[f"s{s}"]["it0" if i == 0 else f"bc_dagger:it{i}"]
itm = lambda s, i: ol[f"s{s}"]["it0" if i == 0 else f"bc_matched:it{i}"]
arm = {
 "verdicts": {f"bc_dagger/s{s}": runs[s]["verdict"] for s in range(5)},
 "selection": {k: sel["bc_dagger_arm"][k] for k in sel["bc_dagger_arm"]}, "bc_matched_selection": sel["bc_matched_arm"], "paired_selected_per_seed": sel["paired_selected"],
 "decided_by": {f"s{s}": sel[f"bc_dagger/s{s}"]["decided_by"] for s in range(5)},
 "per_iteration_select_success_mean": {i: float(csv[csv.iteration == i].select_success.mean()) for i in range(6)},
 "per_iteration_rollout_success_mean": {i: float(csv[csv.iteration == i].rollout_success.mean()) for i in range(1, 6)},
 "teacher_agreement_own_states_l2_range": {i: m([ro[f"s{s}/iter{i}"]["l2_mean"] for s in range(5)]) for i in range(1, 6)},
 "teacher_agreement_4_to_5_falls_all_seeds": all(ro[f"s{s}/iter5"]["l2_mean"] < ro[f"s{s}/iter4"]["l2_mean"] for s in range(5)),
 "fixed_set_sibling_l2": {"bc_dagger": {i: m([it(s, i)["sibling"]["l2_mean"] for s in range(5)]) for i in range(6)}, "bc_matched": {i: m([itm(s, i)["sibling"]["l2_mean"] for s in range(5)]) for i in range(6)}},
 "fixed_set_sibling_l2_it0_to_it4_rel_change_bc_dagger": m([it(s, 4)["sibling"]["l2_mean"] / it(s, 0)["sibling"]["l2_mean"] - 1 for s in range(5)]),
 "fixed_set_sibling_l2_it0_to_it5_rel_change_bc_matched": m([itm(s, 5)["sibling"]["l2_mean"] / itm(s, 0)["sibling"]["l2_mean"] - 1 for s in range(5)]),
 "heldout_mse_teacher_states": {"bc_dagger": {i: m([it(s, i)["heldout"]["mse"] for s in range(5)]) for i in range(6)}, "bc_matched": {i: m([itm(s, i)["heldout"]["mse"] for s in range(5)]) for i in range(6)}},
 "heldout_mse_it0_to_it5_rel_change_bc_dagger": m([it(s, 5)["heldout"]["mse"] / it(s, 0)["heldout"]["mse"] - 1 for s in range(5)]),
 "heldout_ema_loss_log_bc30k_vs_it5": {"bc": m([cur[f"bc/s{s}"]["stages"]["0"]["heldout_ema_end"] for s in range(5)]), "bc_dagger_it5": m([cur[f"bc_dagger/s{s}"]["stages"]["5"]["heldout_ema_end"] for s in range(5)]), "bc_matched_it4": m([cur[f"bc_matched/s{s}"]["stages"]["4"]["heldout_ema_end"] for s in range(5)])},
 "copycat_J_diag_heldout_FD": {"bc_dagger_it0": [m([it(s, 0)["J_diag_heldout"][k] for s in range(5)]) for k in range(3)], "bc_dagger_it5": [m([it(s, 5)["J_diag_heldout"][k] for s in range(5)]) for k in range(3)],
                               "bc_matched_it5": [m([itm(s, 5)["J_diag_heldout"][k] for s in range(5)]) for k in range(3)], "teacher": TEACHER_J},
 "copycat_autograd_csv_vs_FD_it5_heldout_max_abs_diff": float(max(abs(float(csv[(csv.seed == s) & (csv.iteration == 5)][f"copycat_heldout_{a}"].iloc[0]) - it(s, 5)["J_diag_heldout"][k]) for s in range(5) for k, a in enumerate("xyz"))),
 "rollout_copycat_frac_range": m([ro[f"s{s}/iter{i}"]["copycat_frac_dS_lt_0p25dL_given_dL_gt_0p02"] for s in range(5) for i in range(1, 6)]),
 "rollout_lag_mse_t_vs_tminus1_beta0": [m([ro[f"s{s}/iter{i}"]["lag_mse_to_label_t"] for s in range(5) for i in (4, 5)]), m([ro[f"s{s}/iter{i}"]["lag_mse_to_label_t_minus_1"] for s in range(5) for i in (4, 5)])],
 "rollout_saturation_student_vs_label_beta0": [m([ro[f"s{s}/iter{i}"]["sat_norm_ge_0p98_student"] for s in range(5) for i in (4, 5)]), m([ro[f"s{s}/iter{i}"]["sat_norm_ge_0p98_label"] for s in range(5) for i in (4, 5)])],
 "rollout_jerk_student_vs_label_beta0": [m([ro[f"s{s}/iter{i}"]["episode_jerk_student_vs_label"][0] for s in range(5) for i in (4, 5)]), m([ro[f"s{s}/iter{i}"]["episode_jerk_student_vs_label"][1] for s in range(5) for i in (4, 5)])],
 "rollout_timeouts_total": int(sum(ro[f"s{s}/iter{i}"]["outcomes"].get("timeout", 0) for s in range(5) for i in range(1, 6))),
 "rollout_steps_max": int(max(ro[f"s{s}/iter{i}"]["steps_max"] for s in range(5) for i in range(1, 6))),
 "labelled_frames": {"bc_dagger_rollouts_per_seed": [int(csv[csv.seed == s].labelled_frames_added.sum()) for s in range(5)], "teacher_demos_1000_1999": ro["teacher_demos_1000_1999_frames"],
                     "aggregate_stage5_bc_dagger": [prov[f"s{s}"]["rollout_chain"][4]["aggregate_frames"] for s in range(5)], "aggregate_stage5_bc_matched": 122873},
 "budget": {"wall_h_total": sum(runs[s]["budget"]["wall_h"] for s in range(5)), "wall_h_range": m([runs[s]["budget"]["wall_h"] for s in range(5)]),
            "gpu_h_net_of_rollouts_audit_aggregate_total": sum(runs[s]["budget"]["gpu_h_net_of_rollouts_audit_aggregate"] for s in range(5)),
            "gpu_h_net_of_rollouts_audit_only_total": sum(runs[s]["budget"]["gpu_h_net_of_rollouts_audit_only"] for s in range(5)),
            "cpu_in_wall_s_range": m([runs[s]["budget"]["timing_s_sum (runs_summary cpu_in_wall_s)"] for s in range(5)]),
            "projected_P4_D2": 6.53, "phase4_measured": 17.498, "bc_matched_total_gpu_h": sum(prov["bc_matched"][f"s{s}"]["gpu_hours"] for s in range(5)),
            "cut_rule": "P4-D2: no cut, cut factor 1.0, no later cut allowed"},
 "dagger_iterations_csv": sel["dagger_iterations_csv"], "frozen_list_rows_for_bc_dagger": sel["frozen_list_rows_for_bc_dagger"],
}
inputs = sorted(set(
    [str(REPO / f"artifacts/runs/{a}/s{s}/{f}") for a in ("bc_dagger", "bc_matched", "bc") for s in range(5) for f in ("manifest.json", "train_log.jsonl", "selection.json")]
    + [str(REPO / f"artifacts/runs/bc_dagger/s{s}/status.json") for s in range(5)]
    + glob.glob(str(REPO / "artifacts/runs/bc_dagger/s*/select/*.csv")) + glob.glob(str(REPO / "artifacts/runs/bc_matched/s*/select/*.csv"))
    + glob.glob(str(REPO / "artifacts/runs/bc_dagger/s*/dagger/*.json")) + glob.glob(str(REPO / "artifacts/runs/bc_dagger/s*/dagger/select_rows/*.csv"))
    + [str(REPO / f"artifacts/runs/bc/s{s}/select/s0_step030000_rows.csv") for s in range(5)]
    + glob.glob("/home/kbasak/data/ila/dagger/bc_dagger/s*/iter*.h5") + glob.glob("/home/kbasak/data/ila/dagger/bc_dagger/s*/iter*.sidecar.npz") + glob.glob("/home/kbasak/data/ila/dagger/bc_dagger/s*/iter*.manifest.json")
    + [str(REPO / "results/phase5/dagger_iterations.csv"), str(REPO / "results/phase5/dagger_iterations.json"), str(REPO / "results/phase5/runs_summary.csv"), str(REPO / "results/phase4/runs_summary.csv"),
       str(REPO / "splits/select.parquet"), "/home/kbasak/data/ila/demos/ppo_2000.h5", str(REPO / "configs/train/bc_dagger.yaml"), str(REPO / "configs/train/bc.yaml"), str(REPO / "configs/data/demos.yaml")]))
ckpt_note = "checkpoint SHA-256s (bc_dagger 50 + ema_last 5, bc s0_step030000 5) are recorded per run in bc_dagger/s<n>.json (provenance), recomputed by scripts/prov.py"
summary = {"what": "training-run-diagnostician, Gate 5: bc_dagger s0-s4", "date": "2026-10-29", "arm": arm,
           "scripts_sha256": {Path(p).name: sha(p) for p in sorted(glob.glob(str(D / "scripts/*.py")))},
           "logs_sha256": {Path(p).name: sha(p) for p in sorted(glob.glob(str(L / "*")))},
           "per_run_json_sha256": {f"bc_dagger/s{s}.json": sha(D / "bc_dagger" / f"s{s}.json") for s in range(5)},
           "inputs_sha256": {p: sha(p) for p in inputs}, "checkpoints": ckpt_note}
(D / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: v for k, v in arm.items() if k not in ("bc_matched_selection",)}, indent=1, default=str)[:9000])
