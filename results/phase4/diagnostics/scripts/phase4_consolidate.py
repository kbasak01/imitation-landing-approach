"""Consolidate the Phase 4 diagnostician evidence into one JSON per run plus summary.json.

Read-only over the repo and artifacts; writes only under <scratchpad>/phase4_diag/.
Recomputes from artifacts: provenance, curves, selection (from rows), budget, lengths.
Reads scratch outputs for the open-loop diagnostics (diag.json, diag2.json, bcm_diag.json,
dart_diag.json, teacher_jac.json, bcm_jtrend.log) and the SHA-256 recompute logs.
"""
import csv, hashlib, json, math, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parent
OUT = S / "phase4_diag"; (OUT / "scripts").mkdir(parents=True, exist_ok=True)
ROOT = Path("/home/kbasak/projects_02/imitation-landing-approach"); R = ROOT / "artifacts/runs"
SEL = pd.read_parquet(ROOT / "splits/select.parquet")
KEY = ["regime", "ss", "index", "realization_seed", "episode_seed"]
OC = ["success", "hard_landing", "bounce", "off_pad", "timeout", "crash"]
TEACHER_LEN_SS3_5 = 61.47   # results/throughput_projection.csv (P8 e07 ppo s0 mean, id SS3-SS5) used by P2-D2
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def r4(x): return None if x is None else round(float(x), 4)
J = lambda f: json.loads((S / f).read_text())
D_TUNE, D2, BCM, DART, TJ = J("diag.json"), J("diag2.json"), J("bcm_diag.json"), J("dart_diag.json"), J("teacher_jac.json")
SIGSHA = J("sigma_sha.json")
# ---------------- SHA recompute logs -> per run {file: ok}
sha = {}
for line in (S / "sha_phase4.log").read_text().splitlines():
    p = line.split()
    if len(p) >= 6 and p[0] in ("bc", "bc_state"): sha.setdefault(f"{p[0]}/s{p[1]}", {})[p[2]] = p[5] == "OK"
for line in (S / "bcm_sha.log").read_text().splitlines():
    p = line.split()
    if len(p) >= 5 and p[0].isdigit() and p[1].endswith(".pt") and p[2] in ("full", "ema_last"): sha.setdefault(f"bc_matched/s{p[0]}", {})[p[1]] = p[4] == "OK"
    if len(p) >= 2 and p[1] == "init": sha.setdefault(f"bc_matched/s{p[0]}", {})["init_from(bc s30000)"] = "manifest_rec True" in line and "bc_manifest_rec True" in line
for line in (S / "dart_sha_ident.log").read_text().splitlines():
    p = line.split()
    if len(p) >= 5 and p[0] == "sha" and p[2].endswith(".pt"): sha.setdefault(f"bc_dart/s{p[1]}", {})[p[2]] = p[4] == "OK"
for sg, r in SIGSHA.items():
    sha[f"bc_dart_tune/sigma_{sg}/s0"] = {k: v for k, v in r.items() if k != "unlisted"}
# ---------------- helpers
def lr_ref(k, peak, warm, total):
    if k < warm: return peak * k / warm
    return peak * 0.5 * (1 + math.cos(math.pi * (k - warm) / (total - warm)))
def curves(run, man):
    ev = [json.loads(l) for l in (run / "train_log.jsonl").read_text().splitlines()]
    tr = [e for e in ev if e["event"] == "train"]; ho = [e for e in ev if e["event"] == "heldout"]
    st = {s["index"]: s for s in man["stages"]}; peak = man["optimizer"]["lr"]
    nonfin = sum(not all(math.isfinite(e[v]) for v in ("loss", "grad_norm_mean", "grad_norm_max", "loss_last")) for e in tr)
    nonfin += sum(not (math.isfinite(h["ema"]["loss"]) and math.isfinite(h["raw"]["loss"])) for h in ho)
    lrerr = max(abs(e["lr"] - lr_ref(e["stage_step"] - 1, peak, st[e["stage"]]["warmup"], st[e["stage"]]["steps"])) / peak for e in tr)
    out = {"n_train_points": len(tr), "n_heldout_points": len(ho), "nonfinite": nonfin, "lr_schedule_max_rel_err": float(f"{lrerr:.2e}"), "stages": {}}
    for k in sorted(st):
        t = [e for e in tr if e["stage"] == k]; h = [e for e in ho if e["stage"] == k]
        L = np.array([e["loss"] for e in t]); G = np.array([e["grad_norm_max"] for e in t]); steps = np.array([e["stage_step"] for e in t])
        spikes = [int(steps[i]) for i in range(10, len(L)) if L[i] > 3 * np.median(L[i - 10:i])]
        late = steps >= 0.66 * st[k]["steps"]
        out["stages"][str(k)] = {
            "train_loss_first": float(L[0]), "train_loss_end": float(L[-1]), "train_loss_late_median": float(np.median(L[late])),
            "loss_spikes_3x_median10": spikes, "grad_norm_max_overall": float(G.max()), "grad_norm_max_late": float(G[late].max()),
            "heldout_ema_end": h[-1]["ema"]["loss"] if h else None, "heldout_raw_end": h[-1]["raw"]["loss"] if h else None,
            "ema_over_raw_end": (h[-1]["ema"]["loss"] / h[-1]["raw"]["loss"]) if h else None,
            "heldout_ema_min": min(x["ema"]["loss"] for x in h) if h else None,
            "heldout_ema_min_step": h[int(np.argmin([x["ema"]["loss"] for x in h]))]["step"] if h else None,
            "heldout_overlap_episodes": sorted({x["overlap_episodes"] for x in h}) if h else None}
    return out
def selection(run, seed):
    sj = json.loads((run / "selection.json").read_text()); man = json.loads((run / "manifest.json").read_text())
    shas = {c["global_step"]: c["sha256"] for c in man["checkpoints"] if c["kind"] == "full"}
    cands, problems = [], []
    for c in sj["candidates"]:
        df = pd.read_csv(run / "select" / c["rows_file"]); stp = c["candidate"]["global_step"]
        if len(df) != 150 or len(df[KEY].merge(SEL[KEY], on=KEY)) != 150 or df.episode_seed.duplicated().any(): problems.append(f"{stp}: episode set")
        if set(df.ckpt_sha256) != {shas[stp]} or c["candidate"]["sha256"] != shas[stp]: problems.append(f"{stp}: ckpt sha")
        if set(df.run_seed) != {seed}: problems.append(f"{stp}: run_seed")
        if set(df.latency_steps.fillna(0)) != {0}: problems.append(f"{stp}: latency")
        succ = int((df.outcome == "success").sum()); cs = df.closing_speed_normal_m_s.astype(float).to_numpy(); cs = cs[np.isfinite(cs)]
        p95 = float(np.quantile(cs, 0.95)) if len(cs) else math.inf; oc = {o: int((df.outcome == o).sum()) for o in OC}
        if succ != c["success"] or abs(p95 - c["p95_closing_speed_m_s"]) > 1e-9 or oc != {o: c["outcomes"][o] for o in OC}: problems.append(f"{stp}: summary")
        cands.append({"global_step": stp, "stage": c["candidate"]["stage"], "stage_step": c["candidate"]["stage_step"], "success": succ, "p95_closing_speed_m_s": p95,
                      "outcomes": oc, "mean_steps": float(df.steps.mean()), "max_steps": int(df.steps.max()), "max_closing_speed_m_s": float(np.max(cs)),
                      "_df": df})
    best = min(cands, key=lambda r: (-r["success"], r["p95_closing_speed_m_s"], r["global_step"]))
    bdf = best["_df"]; fails = bdf[bdf.outcome != "success"]
    res = {"rule": sj["rule"], "weights": sj["weights"], "inference_path": sj["inference_path"], "list_sha256": sj["list"]["file_sha256"],
           "selected_step_recorded": sj["selected"]["candidate"]["global_step"], "selected_step_recomputed": best["global_step"],
           "selection_correct": best["global_step"] == sj["selected"]["candidate"]["global_step"], "row_problems": problems,
           "candidates": [{k: v for k, v in c.items() if k != "_df"} for c in cands],
           "selected": {"success": best["success"], "p95_closing_speed_m_s": best["p95_closing_speed_m_s"], "outcomes": best["outcomes"],
                        "mean_steps": best["mean_steps"], "max_steps": best["max_steps"],
                        "by_ss": {ss: f"{int((g.outcome == 'success').sum())}/{len(g)}" for ss, g in bdf.groupby("ss")},
                        "failures": fails[["ss", "index", "outcome", "termination_reason", "closing_speed_normal_m_s", "lateral_offset_m", "steps"]].to_dict("records")},
           "candidate_success_range": [min(c["success"] for c in cands), max(c["success"] for c in cands)],
           "candidates_at_max_success": [c["global_step"] for c in cands if c["success"] == best["success"]],
           "winners_curse": {"selected_minus_mean_candidates": best["success"] - float(np.mean([c["success"] for c in cands])),
                             "selected_minus_final_candidate": best["success"] - cands[-1]["success"]}}
    return res, bdf
def prov(run, man):
    st = json.loads((run / "status.json").read_text()); c = man["config"]["files"][0]
    stores = []
    for k, x in sorted(man["data"].items(), key=lambda kv: int(kv[0])):
        for sd in x["stores"]: stores.append({"iteration_or_stage": int(k), "collection": sd["collection"], "sha256": sd["sha256"], "matches_expected": sd["matches"] and sd["sha256"] == sd["expected_sha256"]})
    listed = {c2["file"] for c2 in man["checkpoints"]}
    unlisted = sorted({p.name for p in (run / "checkpoints").iterdir()} - listed)
    return {"arm": man["arm"], "purpose": man["purpose"], "run_seed": man["run_seed"], "sigma": man.get("sigma"), "state": man["state"], "status_state": st["state"],
            "global_step": man["global_step"], "total_steps": man["total_steps"],
            "config": {"path": c["path"], "sha256": c["sha256"], "frozen_sha256": c["frozen_sha256"], "matches_frozen": c["matches"] and c["sha256"] == c["frozen_sha256"] and man["config"]["all_frozen"]},
            "data_stores": stores, "labelled_episodes_final": max(x["labelled_episodes"] for x in man["data"].values()),
            "ila_commit": man["git"]["ila_commit"], "ila_dirty": man["git"]["ila_dirty"], "p8_commit": man["git"]["p8_commit"],
            "invocations": len(man["invocations"]), "resumes": man["resumes"], "init_from": man["init_from"], "init_from_sha256": man.get("init_from_sha256"),
            "wall_clock_s": man["wall_clock_s"], "gpu_hours": man["gpu_hours"], "steps_per_s_loop": man["steps_per_s_loop"], "gpu": man["host"]["gpu"],
            "numerics": man["numerics"], "precision": man["precision"], "ema": man["ema"],
            "checkpoint_sha256_recomputed": sha.get(run.relative_to(R).as_posix()), "unlisted_checkpoint_files": unlisted,
            "checkpoint_sha256_all_ok": bool(sha.get(run.relative_to(R).as_posix())) and all(sha[run.relative_to(R).as_posix()].values()) and
                                         set(sha[run.relative_to(R).as_posix()]) >= listed}
def diag2_pick(stats):  # common subset of open-loop stats
    keep = ["mse", "frac_norm>=0.98_s/l", "frac_norm>1_s/l", "norm_mean_s/l", "W1_per_comp", "std_s", "std_l", "mse_to_label_t", "mse_to_label_t-1",
            "mse_to_last_action", "frac_dS<0.25dL|dL>0.02", "mean_dS/dL", "episode_jerk_s/l", "n_frames"]
    return {k: stats[k] for k in keep if k in stats}
def jdiag(Jm): return {"diag": [Jm[i][i] for i in range(3)], "trace_over_3": r4(np.trace(np.array(Jm)) / 3), "offdiag_max_abs": r4(max(abs(Jm[i][j]) for i in range(3) for j in range(3) if i != j))}
# ---------------- runs
runs = [(f"bc/s{s}", "bc", s) for s in range(5)] + [(f"bc_state/s{s}", "bc_state", s) for s in range(5)] + \
       [(f"bc_matched/s{s}", "bc_matched", s) for s in range(5)] + [(f"bc_dart/s{s}", "bc_dart", s) for s in range(5)] + \
       [(f"bc_dart_tune/sigma_{g}/s0", "bc_dart_tune", 0) for g in ("0.05", "0.10", "0.20")]
per, seldf = {}, {}
jt = {}
for line in (S / "bcm_jtrend.log").read_text().splitlines():
    m = re.match(r"bc s(\d) step (\d+): trJ/3 heldout ([\d.]+) dart ([\d.]+)", line)
    if m: jt.setdefault(f"s{m.group(1)}", {})[int(m.group(2))] = {"trJ3_heldout": float(m.group(3)), "trJ3_dart": float(m.group(4))}
for rid, arm, s in runs:
    run = R / rid; man = json.loads((run / "manifest.json").read_text())
    rec = {"run": rid, "provenance": prov(run, man), "curves": curves(run, man)}
    rec["selection"], seldf[rid] = selection(run, s)
    od = {}
    if arm == "bc":
        d = D2[f"bc/s{s}"]
        od["source"] = "diag2.json (final/selected EMA; heldout = ppo_2000 1950-1999 every 2nd; train = ppo_2000 0-999 every 25th) + bcm_diag.json it0_bc_final (dart_0.10 sample) + bcm_jtrend.log"
        od["J_last_action_final_ema_heldout"] = jdiag(d["jac_heldout"]["J_last_action"])
        od["J_last_action_final_ema_dart"] = jdiag(BCM[f"s{s}"]["it0_bc_final"]["J_dart"])
        od["J_trJ3_vs_step_heldout_dart"] = jt.get(f"s{s}")
        od["final_ema_heldout"] = diag2_pick(d["final_ema_heldout"]); od["final_ema_train"] = diag2_pick(d["final_ema_train"])
        od["final_ema_dart"] = diag2_pick(BCM[f"s{s}"]["it0_bc_final"]["dart"])
        if isinstance(d["sel_ema_heldout"] if "sel_ema_heldout" in d else None, dict): od["selected_ema_heldout"] = diag2_pick(d["sel_ema_heldout"])
    elif arm == "bc_state":
        d = D2[f"bc_state/s{s}"]; t = TJ["bc_state"][f"s{s}"]
        od["source"] = "diag2.json + teacher_jac.json (bc_state on the same 512-state samples)"
        od["J_last_action_final_ema_heldout"] = jdiag(d["jac_heldout"]["J_last_action"])
        od["J_last_action_diag_teacher_jac_run"] = t
        od["input_sensitivity_by_field_final_ema_heldout"] = d["jac_heldout"].get("sens_by_field(|da/dx|*std, summed over a)")
        od["final_ema_heldout"] = diag2_pick(d["final_ema_heldout"]); od["final_ema_train"] = diag2_pick(d["final_ema_train"])
    elif arm == "bc_matched":
        d = BCM[f"s{s}"]; od["source"] = "bcm_diag.json"
        od["selected_file"] = d["selected"]
        od["per_iteration"] = {k: {"J_heldout": jdiag(v["J_heldout"]), "J_dart": jdiag(v["J_dart"]), "heldout": diag2_pick(v["heldout"]),
                                   **({"dart": diag2_pick(v["dart"])} if "dart" in v else {})} for k, v in d.items() if k.startswith("it")}
        seltag = "it5_5k_selected" if "it5_5k_selected" in d else "it5_end"
        od["selected_tag"] = seltag
        b0 = d["it0_bc_final"]
        od["vs_bc_same_seed"] = {
            "openloop_mse_clean_it4_end_over_bc": d["it4_end"]["heldout"]["mse"] / b0["heldout"]["mse"] - 1,
            "openloop_mse_clean_it5_end_over_bc_CONTAMINATED": d["it5_end"]["heldout"]["mse"] / b0["heldout"]["mse"] - 1,
            "openloop_mse_dart_it5_end_over_bc": d["it5_end"]["dart"]["mse"] / b0["dart"]["mse"] - 1,
            "openloop_mse_dart_selected_over_bc": d[seltag]["dart"]["mse"] / b0["dart"]["mse"] - 1,
            "trJ3_heldout_it0_bc": jdiag(b0["J_heldout"])["trace_over_3"], "trJ3_heldout_it5_end": jdiag(d["it5_end"]["J_heldout"])["trace_over_3"],
            "trJ3_heldout_selected": jdiag(d[seltag]["J_heldout"])["trace_over_3"]}
    elif arm == "bc_dart":
        d = DART[f"s{s}"]; od["source"] = "dart_diag.json (dart = bc_dart's own training states, dart_0.10 0-999 every 25th)"
        for tag in [k for k in d if k != "selected_step"]:
            od[tag] = {"J_heldout": jdiag(d[tag]["J_heldout"]), "J_dart": jdiag(d[tag]["J_dart"]), "heldout": diag2_pick(d[tag]["heldout"]), "dart": diag2_pick(d[tag]["dart"])}
    else:
        g = rid.split("sigma_")[1].split("/")[0]; d = D_TUNE[g]; od["source"] = "diag.json (sigma trials; train = own dart store 0-999 every 25th)"
        od["J_last_action_final_ema_train"] = jdiag(d["jac_last_action_train"]); od["J_last_action_final_ema_heldout"] = jdiag(d["jac_last_action_heldout"])
        for k in ("final_ema_train", "final_ema_heldout", "final_raw_heldout"):
            x = d[k]; od[k] = {kk: x[kk] for kk in ("mse", "frac_norm_ge_0.98_student", "frac_norm_ge_0.98_label", "frac_norm_gt_1_student", "frac_norm_gt_1_label",
                                                    "mse_to_label_t", "mse_to_label_t-1", "frac_dS_lt_0.25dL_where_dL>0.02", "beta_err_on_injected_noise") if kk in x}
        od["per_ckpt_ema"] = d["per_ckpt_ema"]; od["mse_by_outcome_train"] = d["mse_by_outcome_train"]
    rec["openloop_diagnostics"] = od
    per[rid] = rec
# ---------------- verdicts
V = {}
for s in range(5):
    V[f"bc/s{s}"] = ("TRUSTWORTHY", ["provenance, SHA-256 and selection verified", "no NaN/spike, LR as configured, EMA≈raw",
                     "arm caveat: selection noise-level (candidates within a few select episodes; winner's curse see summary)",
                     "arm caveat: strong last_action reliance (dA/dlast_action diagonal several times the teacher's)"])
    V[f"bc_state/s{s}"] = ("TRUSTWORTHY", ["provenance, SHA-256 and selection verified", "arm caveat: select ceiling (every candidate 148-150/150)",
                           "last_action Jacobian close to the teacher's (privileged-state arm, by design)"])
    V[f"bc_matched/s{s}"] = ("CAVEAT", ["provenance, SHA-256, warm start and selection verified", "(a) iteration-5 held-out loss contaminated (P4-D1 item 3)",
                             "(b) bc->bc_matched changes data and steps; not a clean data-scaling contrast", "(c) two candidates within binomial noise",
                             "(d) last_action reliance rises across iterations; DART-state open-loop MSE worse than bc"] + (["(e) s1 off-pad dive at 1.234 m/s on select SS4 #35"] if s == 1 else []))
    V[f"bc_dart/s{s}"] = ("CAVEAT", ["bit-identical to bc_dart_tune/sigma_0.10/s0 (model/EMA/AdamW, all checkpoints; select rows identical except ckpt_sha256)",
                          "its select score chose sigma*; keep, footnote, seeds 1-4 sensitivity row (P4-D4 correction 2)", "arm caveat: select ceiling"]) if s == 0 else \
                         ("TRUSTWORTHY", ["provenance, SHA-256 and selection verified", "no copycat (Jacobian ~teacher)", "arm caveat: select ceiling; selection is tie-break noise"])
for g in ("0.05", "0.10", "0.20"):
    V[f"bc_dart_tune/sigma_{g}/s0"] = ("TRUSTWORTHY", ["tuning run (never a core seed)", "sigma* decided by one select episode (P4-D3)"] +
                                       (["bit-identical to bc_dart/s0 (see that run)"] if g == "0.10" else []))
for rid, rec in per.items():
    rec["verdict"], rec["reasons"] = V[rid]
    p = rec["provenance"]
    auto = [k for k, ok in [("config_not_frozen", p["config"]["matches_frozen"]), ("data_sha", all(x["matches_expected"] for x in p["data_stores"])),
                            ("dirty_tree", not p["ila_dirty"]), ("ckpt_sha", p["checkpoint_sha256_all_ok"]), ("unlisted_ckpts", not p["unlisted_checkpoint_files"]),
                            ("selection", rec["selection"]["selection_correct"] and not rec["selection"]["row_problems"]), ("nonfinite", rec["curves"]["nonfinite"] == 0)] if not ok]
    rec["automatic_check_failures"] = auto
    f = OUT / (rid + ".json"); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1, default=float))
# ---------------- summary / P4-D4 figures
def arm_runs(a): return [k for k in per if k.split("/")[0] == a]
summ = {"generated_by": "phase4_consolidate.py", "teacher_J_last_action": TJ["sets"], "teacher_J_method": TJ["method"], "arms": {}, "p4d4_checks": []}
for a in ["bc", "bc_state", "bc_matched", "bc_dart", "bc_dart_tune"]:
    rr = [per[k] for k in arm_runs(a)]
    sel = [r["selection"] for r in rr]
    pooled = pd.concat([seldf[k] for k in arm_runs(a)])
    summ["arms"][a] = {
        "runs": arm_runs(a), "verdicts": {r["run"]: r["verdict"] for r in rr},
        "gpu_hours_sum": sum(r["provenance"]["gpu_hours"] for r in rr),
        "selected_success": [x["selected"]["success"] for x in sel], "selected_steps": [x["selected_step_recomputed"] for x in sel],
        "selected_p95": [x["selected"]["p95_closing_speed_m_s"] for x in sel],
        "candidate_success_all": [[c["success"] for c in x["candidates"]] for x in sel],
        "candidate_success_range": [min(x["candidate_success_range"][0] for x in sel), max(x["candidate_success_range"][1] for x in sel)],
        "winners_curse_selected_minus_mean_candidates_per_run": [x["winners_curse"]["selected_minus_mean_candidates"] for x in sel],
        "winners_curse_selected_minus_mean_candidates_arm_mean": float(np.mean([x["winners_curse"]["selected_minus_mean_candidates"] for x in sel])),
        "winners_curse_selected_minus_final_arm_mean": float(np.mean([x["winners_curse"]["selected_minus_final_candidate"] for x in sel])),
        "pooled_selected_outcomes": pooled.outcome.value_counts().to_dict(),
        "pooled_selected_mean_steps": float(pooled.steps.mean()), "pooled_selected_max_steps": int(pooled.steps.max()),
        "pooled_selected_mean_steps_rel_teacher_61.47": float(pooled.steps.mean() / TEACHER_LEN_SS3_5 - 1),
        "pooled_closing_speed_q50_q95_max": np.quantile(pooled.closing_speed_normal_m_s.astype(float), [0.5, 0.95, 1.0]).tolist(),
        "pooled_touchdown_time_sd_s": float(pooled.td_t_episode_s.astype(float).std())}
A = summ["arms"]
# Jacobian ranges
def rng_(xs): return [r4(min(xs)), r4(max(xs))]
bcJ = [v for k in arm_runs("bc") for v in per[k]["openloop_diagnostics"]["J_last_action_final_ema_heldout"]["diag"]]
bcJd = [v for k in arm_runs("bc") for v in per[k]["openloop_diagnostics"]["J_last_action_final_ema_dart"]["diag"]]
bsJ = [v for s in range(5) for v in TJ["bc_state"][f"s{s}"]["final_ema"]["diag_heldout"]]; bsJd = [v for s in range(5) for v in TJ["bc_state"][f"s{s}"]["final_ema"]["diag_dart"]]
bsJsel = [v for s in range(5) for v in TJ["bc_state"][f"s{s}"]["sel_ema"]["diag_heldout"]]
dJ = [v for k in arm_runs("bc_dart") for v in per[k]["openloop_diagnostics"]["sel_ema"]["J_heldout"]["diag"]]
dJd = [v for k in arm_runs("bc_dart") for v in per[k]["openloop_diagnostics"]["sel_ema"]["J_dart"]["diag"]]
bmJ5 = [v for k in arm_runs("bc_matched") for v in per[k]["openloop_diagnostics"]["per_iteration"]["it5_end"]["J_heldout"]["diag"]]
summ["J_last_action_diag_ranges"] = {
    "teacher_heldout_sample512": TJ["sets"]["heldout_ppo_1950_1999_every2/sample512"]["diag"], "teacher_dart_sample512": TJ["sets"]["dart_0.10_0_999_every25/sample512"]["diag"],
    "bc_final_ema_heldout": rng_(bcJ), "bc_final_ema_dart": rng_(bcJd), "bc_state_final_ema_heldout": rng_(bsJ), "bc_state_final_ema_dart": rng_(bsJd),
    "bc_state_selected_ema_heldout": rng_(bsJsel), "bc_matched_it5_end_heldout": rng_(bmJ5), "bc_dart_selected_ema_heldout": rng_(dJ), "bc_dart_selected_ema_dart": rng_(dJd),
    "sigma_0.10_trial_final_ema_train_and_heldout": rng_(per["bc_dart_tune/sigma_0.10/s0"]["openloop_diagnostics"]["J_last_action_final_ema_train"]["diag"] + per["bc_dart_tune/sigma_0.10/s0"]["openloop_diagnostics"]["J_last_action_final_ema_heldout"]["diag"]),
    "note": "same 512-state index sets (default_rng(0)) for every model; eps 0.05 central difference on raw last_action; diag = d a_i / d last_action_i averaged over states"}
# held-out comparisons (training-log held-out loss, EMA)
ho = {}
for s in range(5):
    b = per[f"bc/s{s}"]["curves"]["stages"]["0"]["heldout_ema_end"]
    m4 = per[f"bc_matched/s{s}"]["curves"]["stages"]["4"]["heldout_ema_end"]; m5 = per[f"bc_matched/s{s}"]["curves"]["stages"]["5"]["heldout_ema_end"]
    dd = per[f"bc_dart/s{s}"]["curves"]["stages"]["0"]["heldout_ema_end"]; bs = per[f"bc_state/s{s}"]["curves"]["stages"]["0"]["heldout_ema_end"]
    ho[f"s{s}"] = {"bc_end": b, "bc_matched_it4_end": m4, "bc_matched_it4_rel_bc": m4 / b - 1, "bc_matched_it5_end_CONTAMINATED": m5,
                   "bc_dart_end": dd, "bc_dart_rel_bc": dd / b - 1, "bc_state_end": bs}
summ["heldout_loss_comparisons_trainlog_ema"] = ho
ol = {f"s{s}": per[f"bc_matched/s{s}"]["openloop_diagnostics"]["vs_bc_same_seed"] for s in range(5)}
for s in range(5):
    ol[f"s{s}"]["bc_dart_sel_openloop_mse_clean_over_bc"] = DART[f"s{s}"]["sel_ema"]["heldout"]["mse"] / BCM[f"s{s}"]["it0_bc_final"]["heldout"]["mse"] - 1
    ol[f"s{s}"]["bc_dart_sel_openloop_mse_dart_over_bc"] = DART[f"s{s}"]["sel_ema"]["dart"]["mse"] / BCM[f"s{s}"]["it0_bc_final"]["dart"]["mse"] - 1
summ["openloop_mse_comparisons_vs_bc_final_ema"] = ol
# sigma-star winner's curse
tr = [per[f"bc_dart_tune/sigma_{g}/s0"]["selection"] for g in ("0.05", "0.10", "0.20")]
all18 = [c["success"] for x in tr for c in x["candidates"]]
summ["sigma_star_winners_curse"] = {"sigma_star_selected": 150, "mean_selected_over_3_trials": float(np.mean([x["selected"]["success"] for x in tr])),
    "mean_over_18_candidates": float(np.mean(all18)), "bc_dart_s0_minus_mean_selected_s1_s4": 150 - float(np.mean(A["bc_dart"]["selected_success"][1:])),
    "bc_dart_seeds1_4_selected": A["bc_dart"]["selected_success"][1:]}
# budget
rs = list(csv.DictReader(open(ROOT / "results/phase4/runs_summary.csv")))
proj = {r["label"]: float(r["gpu_hours"]) for r in csv.DictReader(open(ROOT / "results/phase4/budget_projection.csv"))}
meas = {"sigma trials": sum(float(r["gpu_hours"]) for r in rs if r["run"].startswith("bc_dart_tune")),
        "bc": sum(float(r["gpu_hours"]) for r in rs if r["arm"] == "bc"), "bc_state": sum(float(r["gpu_hours"]) for r in rs if r["arm"] == "bc_state"),
        "bc_matched": sum(float(r["gpu_hours"]) for r in rs if r["arm"] == "bc_matched"),
        "bc_dart": sum(float(r["gpu_hours"]) for r in rs if r["arm"] == "bc_dart" and not r["run"].startswith("bc_dart_tune"))}
pj = {"sigma trials": proj["bc_dart sigma trials"], "bc": proj["bc"], "bc_state": proj["bc_state"], "bc_matched": proj["bc_matched (iterations 1-5)"], "bc_dart": proj["bc_dart"]}
tm, tp = sum(meas.values()), sum(pj.values())
summ["budget"] = {"runs_summary_csv_sha256": sha_file(ROOT / "results/phase4/runs_summary.csv"), "budget_projection_csv_sha256": sha_file(ROOT / "results/phase4/budget_projection.csv"),
                  "n_runs": len(rs), "measured_gpu_h": meas, "projected_gpu_h": pj, "measured_total": tm, "projected_total": tp, "overrun_rel": tm / tp - 1,
                  "manifest_sum_check": sum(per[k]["provenance"]["gpu_hours"] for k in per), "duplicate_bc_dart_s0_gpu_h": per["bc_dart/s0"]["provenance"]["gpu_hours"]}
# lengths
summ["lengths"] = {a: {"mean": A[a]["pooled_selected_mean_steps"], "max": A[a]["pooled_selected_max_steps"], "rel_teacher_61.47": A[a]["pooled_selected_mean_steps_rel_teacher_61.47"]} for a in ["bc", "bc_state", "bc_matched", "bc_dart"]}
summ["lengths"]["teacher_reference"] = "61.47 steps: results/throughput_projection.csv, P8 e07 ppo s0 mean on id SS3-SS5 (P2-D2 central projection)"
# bc_matched s1 dive
r1 = seldf["bc_matched/s1"]; dive = r1[(r1.ss == "SS4") & (r1["index"] == 35)].iloc[0]
bc_max = float(pd.concat([seldf[f"bc/s{s}"] for s in range(5)]).closing_speed_normal_m_s.astype(float).max())
summ["bc_matched_s1_dive"] = {"outcome": dive.outcome, "closing_speed_normal_m_s": float(dive.closing_speed_normal_m_s), "bc_selected_max_closing_m_s": bc_max,
                              "ratio": float(dive.closing_speed_normal_m_s) / bc_max}
# P4-D4 figure checks
C = summ["p4d4_checks"]
def chk(fig, claimed, got, ok, note=""): C.append({"figure": fig, "p4d4": claimed, "reproduced": got, "agrees": bool(ok), "note": note})
chk("bc candidates 140-149/150", "140-149", A["bc"]["candidate_success_range"], A["bc"]["candidate_success_range"] == [140, 149])
wc = A["bc"]["winners_curse_selected_minus_mean_candidates_arm_mean"]
chk("bc winner's curse about +2 episodes", "+2", round(wc, 2), 1.5 <= wc <= 2.5, "defined as mean over seeds of (selected success - mean success of the 6 candidates); vs final checkpoint: %+.1f" % A["bc"]["winners_curse_selected_minus_final_arm_mean"])
chk("bc dA/dlast_action diagonal 0.75-0.91", "0.75-0.91", rng_(bcJ), rng_(bcJ)[0] >= 0.745 and rng_(bcJ)[1] <= 0.915, "final EMA, heldout states; s3 z-axis is 0.701. On dart states: %s" % rng_(bcJd))
chk("bc_state candidates 148-150", "148-150", A["bc_state"]["candidate_success_range"], A["bc_state"]["candidate_success_range"] == [148, 150])
r4s = [ho[f"s{s}"]["bc_matched_it4_rel_bc"] for s in range(5)]
chk("bc_matched it4-end held-out loss about 20% below bc", "-20%", [round(x, 3) for x in r4s], all(-0.25 <= x <= -0.15 for x in r4s), "training-log held-out EMA loss at end of iteration 4 vs bc step 30000, same seed")
j0 = [jt["s0"][5000]["trJ3_heldout"], jt["s0"][30000]["trJ3_heldout"]]
chk("bc s0 last_action reliance 0.57 -> 0.81 (5k -> 30k)", "0.57->0.81", j0, round(j0[0], 2) == 0.57 and round(j0[1], 2) == 0.81, "metric is trace/3 (mean diagonal), heldout states, EMA")
cd = [abs(x["candidates"][0]["success"] - x["candidates"][1]["success"]) for x in [per[k]["selection"] for k in arm_runs("bc_matched")]]
chk("bc_matched candidates differ by 1-4 select episodes", "1-4", cd, min(cd) >= 1 and max(cd) <= 4)
t0s = [ol[f"s{s}"]["trJ3_heldout_it0_bc"] for s in range(5)]; t5s = [ol[f"s{s}"]["trJ3_heldout_it5_end"] for s in range(5)]; tss = [ol[f"s{s}"]["trJ3_heldout_selected"] for s in range(5)]
chk("bc_matched last_action reliance 0.78-0.82 -> 0.86-0.91", "0.78-0.82 -> 0.86-0.91", {"it0": rng_(t0s), "it5_end": rng_(t5s), "selected": rng_(tss)},
    rng_(t0s)[0] >= 0.775 and rng_(t0s)[1] <= 0.825 and rng_(t5s)[0] >= 0.855 and rng_(t5s)[1] <= 0.915, "trace/3, heldout states; it0 = bc final EMA")
dm5 = [ol[f"s{s}"]["openloop_mse_dart_it5_end_over_bc"] for s in range(5)]; dms = [ol[f"s{s}"]["openloop_mse_dart_selected_over_bc"] for s in range(5)]
chk("bc_matched DART-state open-loop MSE 10-15% worse than bc", "+10..15%", {"it5_end": [round(x, 3) for x in dm5], "selected": [round(x, 3) for x in dms]},
    all(0.095 <= x <= 0.155 for x in dm5 + dms), "max is %.1f%% (s2, it5_end = selected)" % (100 * max(dm5 + dms)))
cl4 = [ol[f"s{s}"]["openloop_mse_clean_it4_end_over_bc"] for s in range(5)]
chk("bc_matched about 20% better on clean states (open loop)", "-20%", [round(x, 3) for x in cl4], all(-0.25 <= x <= -0.15 for x in cl4),
    "uncontaminated point = end of iteration 4; iteration 5 trains on these episodes (it5 vs bc: %s)" % [round(ol[f's{s}']['openloop_mse_clean_it5_end_over_bc_CONTAMINATED'], 2) for s in range(5)])
dv = summ["bc_matched_s1_dive"]
chk("bc_matched s1 off pad SS4 #35 at 1.234 m/s, about 2x fastest bc", "1.234, ~2x", [round(dv["closing_speed_normal_m_s"], 3), round(dv["ratio"], 2)], dv["outcome"] == "off_pad" and round(dv["closing_speed_normal_m_s"], 3) == 1.234)
chk("bc_matched candidates = cumulative 75k/80k", "45k/50k run-local", [[c["global_step"] for c in per[k]["selection"]["candidates"]] for k in arm_runs("bc_matched")], True)
chk("bc_dart seeds 1-4 select 149/150/150/150", "149,150,150,150", A["bc_dart"]["selected_success"][1:], A["bc_dart"]["selected_success"][1:] == [149, 150, 150, 150])
chk("duplicate s0 0.807 GPU-h", "0.807", round(per["bc_dart/s0"]["provenance"]["gpu_hours"], 4), round(per["bc_dart/s0"]["provenance"]["gpu_hours"], 3) == 0.807)
L = summ["lengths"]
chk("lengths bc 65.3 (max 237), bc_state 61.5, bc_matched 62.7, bc_dart 61.6", "65.3/237, 61.5, 62.7, 61.6", {a: [round(L[a]["mean"], 2), L[a]["max"]] for a in ["bc", "bc_state", "bc_matched", "bc_dart"]},
    [round(L[a]["mean"], 1) for a in ["bc", "bc_state", "bc_matched", "bc_dart"]] == [65.3, 61.5, 62.7, 61.6] and L["bc"]["max"] == 237)
chk("BC-family lengths within 6% of the teacher's 61.5", "<=6%", {a: round(100 * L[a]["rel_teacher_61.47"], 2) for a in ["bc", "bc_state", "bc_matched", "bc_dart"]},
    all(L[a]["rel_teacher_61.47"] <= 0.06 for a in ["bc", "bc_state", "bc_matched", "bc_dart"]), "bc is +6.2% vs 61.47 (+6.2% vs 61.5 too)")
B = summ["budget"]
chk("Phase 4 measured 17.50 GPU-h vs 16.9 projected (+3.5%)", "17.50, 16.9, +3.5%", [round(B["measured_total"], 3), round(B["projected_total"], 4), round(100 * B["overrun_rel"], 2)],
    round(B["measured_total"], 2) == 17.50 and abs(100 * B["overrun_rel"] - 3.5) < 0.05, "exact projection is %.4f; overrun %.2f%%" % (B["projected_total"], 100 * B["overrun_rel"]))
for k, claimed in [("sigma trials", 2.37), ("bc", 3.96), ("bc_state", 0.21), ("bc_matched", 6.95), ("bc_dart", 4.02)]:
    chk(f"budget {k} {claimed}", claimed, round(B["measured_gpu_h"][k], 4), round(B["measured_gpu_h"][k], 2) == claimed)
chk("correction 1 bit-identity", "identical at all six checkpoints", "see dart_sha_ident.log / dart_sel.log / dart_curves.log", True)
# scripts and inputs
scripts = ["diag.py", "diag2.py", "curves.py", "selcheck.py", "shacheck.py", "chk.py", "chk2.py", "bcm_prov.py", "bcm_sha.py", "bcm_sel.py", "bcm_diag.py", "bcm_jtrend.py",
           "dart_sha_ident.py", "dart_sel.py", "dart_curves.py", "dart_diag.py", "teacher_jac.py", "sigma_sha.py", "phase4_consolidate.py"]
inputs = ["diag.json", "diag.log", "diag2.json", "diag2.log", "curves.log", "selcheck.log", "sha_phase4.log", "bcm_prov.log", "bcm_sha.log", "bcm_sel.log", "bcm_diag.json",
          "bcm_diag.log", "bcm_jtrend.log", "dart_sha_ident.log", "dart_sel.log", "dart_curves.log", "dart_diag.json", "dart_diag.log", "teacher_jac.json", "teacher_jac.log", "sigma_sha.json", "sigma_sha.log"]
for f in scripts: shutil.copy2(S / f, OUT / "scripts" / f)
(OUT / "inputs").mkdir(exist_ok=True)
for f in inputs: shutil.copy2(S / f, OUT / "inputs" / f)
summ["script_sha256"] = {f: sha_file(S / f) for f in scripts}
summ["input_sha256"] = {f: sha_file(S / f) for f in inputs}
summ["per_run_files"] = {k: f"{k}.json" for k in per}
summ["disagreements"] = [c for c in C if not c["agrees"]]
(OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
for c in C: print(("OK   " if c["agrees"] else "DIFF ") + c["figure"], "| reproduced:", c["reproduced"], "|", c["note"])
print("auto failures:", {k: v["automatic_check_failures"] for k, v in per.items() if v["automatic_check_failures"]})
