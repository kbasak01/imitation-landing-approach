"""Read-only: recompute bc_dagger selection from the stored select rows (P3-D1 §4 rule; candidates =
iteration 5's 5 k and 10 k EMA), winner's curse, outcome breakdown, failures; recompute the
per-iteration select columns and rollout/agreement columns of results/phase5/dagger_iterations.csv;
bc_matched beside it on the same rule; frozen-list row scan."""
import json, math, sys, glob
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, h5py
REPO = Path("/home/kbasak/projects_02/imitation-landing-approach"); R = REPO / "artifacts/runs"
DAG = Path("/home/kbasak/data/ila/dagger/bc_dagger")
sel = pd.read_parquet(REPO / "splits/select.parquet")
KEY = ["regime", "ss", "index", "realization_seed", "episode_seed"]
OUT = ["crash", "off_pad", "hard_landing", "bounce", "success", "timeout"]
BETA = [0.5, 0.25, 0.125, 0.0, 0.0]
def flag(x): return int(str(x) == "True")
def summ(df):
    cs = df.closing_speed_normal_m_s.astype(float).to_numpy(); td = df.touchdown_contact.map(lambda x: str(x) == "True").to_numpy()
    csf = cs[np.isfinite(cs)]
    return {"n": len(df), "success": int((df.outcome == "success").sum()), "outcomes": {o: int((df.outcome == o).sum()) for o in OUT},
            "p95_closing_speed_m_s": float(np.quantile(csf, 0.95)) if len(csf) else math.inf, "n_finite_closing": int(len(csf)),
            "n_touchdown_contact": int(td.sum()), "mean_steps": float(df.steps.mean()), "max_steps": int(df.steps.max()),
            "n_tunnelled": int(df.tunnelled.map(flag).sum()), "n_detectors_disagree": int(df.detectors_disagree.map(flag).sum()),
            "max_closing_speed_m_s": float(csf.max()), "action_jerk_mean": float(df.action_jerk_mean.mean()),
            "by_ss": {ss: f"{int((g.outcome == 'success').sum())}/{len(g)}" for ss, g in df.groupby("ss")},
            "failures": df[df.outcome != "success"][["ss", "index", "outcome", "termination_reason", "closing_speed_normal_m_s", "lateral_offset_m", "steps", "tunnelled"]].to_dict("records")}
def check_rows(df, sha, seed, P, tag):
    if len(df) != 150: P.append(f"{tag}: n={len(df)}")
    m = df[KEY].merge(sel[KEY], on=KEY, how="inner")
    if len(m) != 150 or df.episode_seed.duplicated().any(): P.append(f"{tag}: episodes != select")
    if set(df.regime) != {"select"}: P.append(f"{tag}: regime {set(df.regime)}")
    if set(df.ckpt_sha256) != {sha}: P.append(f"{tag}: ckpt sha {set(df.ckpt_sha256)}")
    if set(df.run_seed) != {seed}: P.append(f"{tag}: run_seed")
    if set(df.latency_steps.fillna(0)) != {0}: P.append(f"{tag}: latency")
    if not (df.frames_rendered == df.steps + 1).all(): P.append(f"{tag}: frames_rendered")
res = {"problems": []}
allsel = {}
for arm in ["bc_dagger", "bc_matched"]:
    for s in range(5):
        d = R / arm / f"s{s}"; sj = json.loads((d / "selection.json").read_text()); man = json.loads((d / "manifest.json").read_text())
        shas = {c["file"]: c["sha256"] for c in man["checkpoints"] if c["kind"] == "full"}
        P = []; cands = []
        for c in sj["candidates"]:
            f = Path(c["candidate"]["file"]).name; df = pd.read_csv(d / "select" / c["rows_file"])
            check_rows(df, shas[f], s, P, f)
            if c["candidate"]["sha256"] != shas[f]: P.append(f"{f}: selection.json sha")
            sm = summ(df)
            for k in ("success", "outcomes", "n_tunnelled", "n_detectors_disagree"):
                if k in c and c[k] is not None and c[k] != sm[k]: P.append(f"{f}: {k} {c[k]} vs {sm[k]}")
            if abs(sm["p95_closing_speed_m_s"] - c["p95_closing_speed_m_s"]) > 1e-12: P.append(f"{f}: p95")
            if sm["n_finite_closing"] != c.get("n_touchdown", sm["n_finite_closing"]): P.append(f"{f}: n_touchdown {c.get('n_touchdown')} vs {sm['n_finite_closing']}")
            cands.append({"file": f, "global_step": c["candidate"]["global_step"], "stage": c["candidate"]["stage"], **sm, "_df": df})
        if sorted(x["file"] for x in cands) != ["s5_step005000.pt", "s5_step010000.pt"]: P.append("candidate set")
        best = min(cands, key=lambda x: (-x["success"], x["p95_closing_speed_m_s"], x["global_step"]))
        recorded = Path(sj["selected"]["candidate"]["file"]).name
        decided_by = "success" if len({x["success"] for x in cands}) > 1 else ("p95" if len({x["p95_closing_speed_m_s"] for x in cands}) > 1 else "step")
        r = {"selected_recorded": recorded, "selected_recomputed": best["file"], "selection_correct": recorded == best["file"], "decided_by": decided_by,
             "candidates": [{k: v for k, v in x.items() if k not in ("_df", "failures")} for x in cands],
             "selected": {k: v for k, v in best.items() if k != "_df"},
             "winners_curse": {"selected_minus_mean_candidates": best["success"] - float(np.mean([x["success"] for x in cands]))},
             "row_problems": P}
        if not r["selection_correct"]: P.append("selection wrong")
        allsel[(arm, s)] = (best, cands); res[f"{arm}/s{s}"] = r
        if P: res["problems"].append((arm, s, P))
        print(f"{arm} s{s}: sel {recorded} recomputed {best['file']} by {decided_by}; cands {[ (x['file'][3:13], x['success'], round(x['p95_closing_speed_m_s'],3), {k:v for k,v in x['outcomes'].items() if v and k!='success'}, x['n_tunnelled']) for x in cands]} problems {P}")
for arm in ["bc_dagger", "bc_matched"]:
    b = [allsel[(arm, s)][0] for s in range(5)]
    pooled = pd.concat([x["_df"] for x in b])
    fails = Counter(e for x in b for e in x["_df"].query("outcome!='success'").episode_seed)
    res[f"{arm}_arm"] = {"selected_success": [x["success"] for x in b], "mean_selected_success": float(np.mean([x["success"] for x in b])),
        "selected_steps": [x["global_step"] for x in b], "p95": [x["p95_closing_speed_m_s"] for x in b],
        "all_candidate_success": [[c["success"] for c in allsel[(arm, s)][1]] for s in range(5)],
        "mean_all_candidates": float(np.mean([[c["success"] for c in allsel[(arm, s)][1]] for s in range(5)])),
        "winners_curse_mean": float(np.mean([res[f"{arm}/s{s}"]["winners_curse"]["selected_minus_mean_candidates"] for s in range(5)])),
        "pooled_outcomes": pooled.outcome.value_counts().to_dict(), "pooled_by_ss": {ss: f"{int((g.outcome=='success').sum())}/{len(g)}" for ss, g in pooled.groupby("ss")},
        "pooled_closing_q50_q95_max": np.round(np.quantile(pooled.closing_speed_normal_m_s.astype(float).dropna(), [0.5, 0.95, 1.0]), 4).tolist(),
        "pooled_mean_steps": float(pooled.steps.mean()), "pooled_max_steps": int(pooled.steps.max()), "pooled_action_jerk_mean": float(pooled.action_jerk_mean.mean()),
        "pooled_lateral_offset_q95": float(pooled.lateral_offset_m.astype(float).quantile(0.95)),
        "n_tunnelled_selected": int(sum(x["n_tunnelled"] for x in b)), "n_detectors_disagree_selected": int(sum(x["n_detectors_disagree"] for x in b)),
        "failing_episodes_shared_across_seeds": {str(k): v for k, v in fails.items() if v > 1}, "distinct_failing_episodes": len(fails)}
    print(arm, json.dumps({k: v for k, v in res[f"{arm}_arm"].items()}, default=str))
# paired per seed dagger vs matched on selected checkpoints
pair = {}
for s in range(5):
    a = allsel[("bc_dagger", s)][0]["_df"]; m = allsel[("bc_matched", s)][0]["_df"]
    fa = set(a.query("outcome!='success'").episode_seed); fm = set(m.query("outcome!='success'").episode_seed)
    pair[f"s{s}"] = {"dagger": allsel[("bc_dagger", s)][0]["success"], "matched": allsel[("bc_matched", s)][0]["success"], "dagger_fail": len(fa), "matched_fail": len(fm), "shared_fail": len(fa & fm)}
res["paired_selected"] = pair; print("paired", pair)
# per-iteration select (end-of-iteration EMA) recompute and dagger_iterations.csv cross-check
csv = pd.read_csv(REPO / "results/phase5/dagger_iterations.csv")
res["dagger_iterations_csv"] = {"duplicate_column_names_in_header": [c for c, n in Counter(open(REPO / "results/phase5/dagger_iterations.csv").readline().strip().split(",")).items() if n > 1]}
mism = []
for s in range(5):
    man = json.loads((R / "bc_dagger" / f"s{s}" / "manifest.json").read_text()); shas = {c["file"]: c["sha256"] for c in man["checkpoints"] if c["kind"] == "full"}
    bcman = json.loads((R / "bc" / f"s{s}" / "manifest.json").read_text()); bcsha = [c["sha256"] for c in bcman["checkpoints"] if c["file"] == "s0_step030000.pt"][-1]
    for i in range(0, 6):
        row = csv[(csv.seed == s) & (csv.iteration == i)].iloc[0]
        if i == 0: f = R / "bc" / f"s{s}" / "select" / "s0_step030000_rows.csv"; sha = bcsha
        elif i < 5: f = R / "bc_dagger" / f"s{s}" / "dagger" / "select_rows" / f"s{i}_step010000_rows.csv"; sha = shas[f"s{i}_step010000.pt"]
        else: f = R / "bc_dagger" / f"s{s}" / "select" / "s5_step010000_rows.csv"; sha = shas["s5_step010000.pt"]
        df = pd.read_csv(f); P = []; check_rows(df, sha, s, P, f"s{s} it{i}"); sm = summ(df)
        if row.eval_ckpt_sha256 != sha: P.append("eval sha")
        for o in OUT:
            col = "select_success.1" if o == "success" else f"select_{o}"
            if int(row[col]) != sm["outcomes"][o]: P.append(f"select_{o}")
        if int(row.select_success) != sm["success"] or abs(row.select_p95_closing_speed_m_s - sm["p95_closing_speed_m_s"]) > 1e-9 or int(row.select_n_tunnelled) != sm["n_tunnelled"] or int(row.select_n_detectors_disagree) != sm["n_detectors_disagree"]: P.append("select summary")
        if i >= 1:
            sc = np.load(DAG / f"s{s}" / f"iter{i}.sidecar.npz")
            with h5py.File(DAG / f"s{s}" / f"iter{i}.h5", "r") as h:
                names = sorted(h.keys()); lab = np.concatenate([h[n]["label"][()] for n in names]); oc = Counter(str(h[n].attrs["outcome"]) for n in names)
                # sidecar order check
                if [int(h[n].attrs["demo_index"]) for n in names] != sc["demo_index"].tolist(): P.append("sidecar order")
            prop = sc["proposed"]; te = sc["teacher_exec"]
            l2 = np.linalg.norm(prop.astype(np.float64) - lab, axis=1); linf = np.abs(prop.astype(np.float64) - lab).max(1)
            st = ~te
            if abs(l2.mean() - row.agree_l2_mean) > 1e-6 or abs((linf <= 0.1).mean() - row.agree_linf_le_0p1_frac) > 1e-9: P.append(f"agree all {l2.mean()} {row.agree_l2_mean}")
            if abs(l2[st].mean() - row.agree_student_l2_mean) > 1e-6: P.append("agree student")
            if int(row.rollout_success) != oc["success"] or int(row.rollout_hard_landing) != oc.get("hard_landing", 0) or int(row.rollout_bounce) != oc.get("bounce", 0) or int(row.rollout_off_pad) != oc.get("off_pad", 0) or int(row.rollout_timeout) != oc.get("timeout", 0) or int(row.rollout_crash) != oc.get("crash", 0): P.append(f"rollout outcomes {dict(oc)}")
            if abs(float(row.beta) - BETA[i-1]) > 0: P.append("beta")
            if abs(te.mean() - BETA[i-1]) > 0.02: P.append(f"teacher exec frac {te.mean()}")
        if P: mism.append((s, i, P))
res["dagger_iterations_csv"]["mismatches"] = mism
print("dagger_iterations.csv mismatches:", mism, res["dagger_iterations_csv"])
# frozen-list scan: any CSV anywhere mentioning bc_dagger with a regime other than select/demos
bad = []
for p in glob.glob(str(R / "**/*.csv"), recursive=True) + glob.glob(str(REPO / "results/**/*.csv"), recursive=True):
    try: df = pd.read_csv(p, usecols=lambda c: c in ("regime", "arm", "method"))
    except Exception: continue
    if "regime" in df and (("arm" in df and (df.arm == "bc_dagger").any()) or ("method" in df and (df.method == "bc_dagger").any())):
        regs = set(df.regime.astype(str)) - {"select", "demos"}
        if regs: bad.append((p, sorted(regs)))
res["frozen_list_rows_for_bc_dagger"] = bad; print("frozen-list rows:", bad)
for k in list(res):
    pass
Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str)); print("wrote", sys.argv[1])
