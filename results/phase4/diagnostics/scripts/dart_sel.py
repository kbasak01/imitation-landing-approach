import json, math
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/home/kbasak/projects_02/imitation-landing-approach/artifacts/runs")
sel = pd.read_parquet("/home/kbasak/projects_02/imitation-landing-approach/splits/select.parquet")
key = ["regime","ss","index","realization_seed","episode_seed"]
OUT = ["success","hard_landing","bounce","off_pad","timeout","crash"]
allsel = {}
for a in ["bc", "bc_dart"]:
    for s in range(5):
        d = R / a / f"s{s}"
        if not (d / "selection.json").exists(): continue
        sj = json.loads((d / "selection.json").read_text()); man = json.loads((d / "manifest.json").read_text())
        shas = {c["global_step"]: c["sha256"] for c in man["checkpoints"] if c["kind"] == "full"}
        res = []; problems = []
        for c in sj["candidates"]:
            df = pd.read_csv(d / "select" / c["rows_file"])
            st = c["candidate"]["global_step"]
            if len(df) != 150: problems.append(f"{st}: n={len(df)}")
            m = df[key].merge(sel[key], on=key, how="inner")
            if len(m) != 150 or df.episode_seed.duplicated().any(): problems.append(f"{st}: episode mismatch {len(m)}")
            if set(df.ckpt_sha256) != {shas[st]} or c["candidate"]["sha256"] != shas[st]: problems.append(f"{st}: ckpt sha mismatch {set(df.ckpt_sha256)}")
            if set(df.run_seed) != {s} or set(df.student_seed.astype(str)) - {str(s), f"{s}.0"}: problems.append(f"{st}: seed col {set(df.run_seed)} {set(df.student_seed)}")
            if set(df.latency_steps.fillna(0)) != {0}: problems.append(f"{st}: latency {set(df.latency_steps)}")
            succ = int((df.outcome == "success").sum())
            cs = df.closing_speed_normal_m_s.astype(float).to_numpy(); cs = cs[np.isfinite(cs)]
            p95 = float(np.quantile(cs, 0.95)) if len(cs) else math.inf
            oc = {o: int((df.outcome == o).sum()) for o in OUT}
            if succ != c["success"] or abs(p95 - c["p95_closing_speed_m_s"]) > 1e-9 or oc != {o: c["outcomes"][o] for o in OUT}: problems.append(f"{st}: summary mismatch {succ} {p95} {oc} vs {c['success']} {c['p95_closing_speed_m_s']}")
            res.append((st, succ, p95, oc, df))
        best = min(res, key=lambda r: (-r[1], r[2], r[0]))
        ok = best[0] == sj["selected"]["candidate"]["global_step"]
        succs = [r[1] for r in res]
        # margin: how many candidates within 1 success of best
        print(f"== {a} s{s}: selected {sj['selected']['candidate']['global_step']} recomputed {best[0]} {'OK' if ok else 'WRONG'}; weights={sj['weights']} path={sj['inference_path']} list_sha={sj['list']['file_sha256'][:8]} render={sj['render']} problems={problems or 'none'}")
        for st, succ, p95, oc, df in res:
            nz = {k: v for k, v in oc.items() if v and k != 'success'}
            print(f"   {st:6d} {succ}/150 p95={p95:.3f} mean_steps={df.steps.mean():.1f} jerk={df.action_jerk_mean.mean():.4f} effort={df.effort_mean_sq.mean():.4f} fails={nz}{'  <-- selected' if st==best[0] else ''}")
        bdf = best[4]
        print("   selected by SS:", {ss: f"{int((g.outcome=='success').sum())}/{len(g)}" for ss, g in bdf.groupby('ss')}, "fails:", bdf[bdf.outcome!='success'][['ss','index','outcome','termination_reason','closing_speed_normal_m_s','lateral_offset_m','steps']].to_dict('records'))
        allsel[(a, s)] = (best, res)
# seed spread
for a in ["bc", "bc_dart"]:
    b = [allsel[(a, s)][0] for s in range(5)]
    print(f"\n{a}: selected success {[x[1] for x in b]} steps {[x[0] for x in b]} p95 {[round(x[2],3) for x in b]}")
    allc = [[r[1] for r in allsel[(a, s)][1]] for s in range(5)]
    print(f"   all-candidate success per seed: {allc}; mean over all ckpts {np.mean(allc):.2f}; final-ckpt {[c[-1] for c in allc]}")
    # failure overlap across seeds on selected checkpoints
    fails = [set(allsel[(a, s)][0][4].query("outcome!='success'").episode_seed) for s in range(5)]
    from collections import Counter
    cnt = Counter(e for f in fails for e in f)
    print("   failing episodes shared across seeds:", {k: v for k, v in cnt.items() if v > 1}, "total distinct", len(cnt))
    pooled = pd.concat([allsel[(a, s)][0][4] for s in range(5)])
    print("   pooled selected outcomes:", pooled.outcome.value_counts().to_dict(), "by SS:", {ss: f"{int((g.outcome=='success').sum())}/{len(g)}" for ss, g in pooled.groupby('ss')})
    print("   pooled steps mean", round(pooled.steps.mean(),1), "td_t mean", round(pooled.td_t_episode_s.astype(float).mean(),3), "closing mean", round(pooled.closing_speed_normal_m_s.astype(float).mean(),3), "lat offset mean", round(pooled.lateral_offset_m.astype(float).mean(),4), "jerk", round(pooled.action_jerk_mean.mean(),4), "effort", round(pooled.effort_mean_sq.mean(),4))
    allsel[a] = pooled
# episodes failed by bc but also by bc_state?
fb = set(allsel["bc"].query("outcome!='success'").episode_seed); fs = set(allsel["bc_dart"].query("outcome!='success'").episode_seed)
print("\nbc-fail episodes", len(fb), "bc_matched-fail episodes", len(fs), "overlap", len(fb & fs))

print()
for s in range(5):
    sj = json.loads((R / "bc_dart" / f"s{s}" / "selection.json").read_text())
    print(f"s{s} purpose={sj['purpose']} workers={sj['workers']} wall={sj['wall_clock_s']:.0f} rule={sj['rule'][:60]}")
    b, _ = allsel[("bc", s)]; m, _ = allsel[("bc_dart", s)]
    fb = set(b[4].query("outcome!='success'").episode_seed); fm = set(m[4].query("outcome!='success'").episode_seed)
    print(f"   paired s{s}: bc {b[1]} p95 {b[2]:.3f} vs bc_dart {m[1]} p95 {m[2]:.3f}; fails bc {len(fb)} dart {len(fm)} shared {len(fb & fm)}")
for nm in ["bc", "bc_dart"]:
    df = allsel[nm]; cs = df.closing_speed_normal_m_s.astype(float)
    print(nm, "closing q50/q95/max", np.round(np.quantile(cs, [0.5, 0.95, 1.0]), 3), "lat q95", round(float(df.lateral_offset_m.astype(float).quantile(0.95)), 4), "td_t sd", round(float(df.td_t_episode_s.astype(float).std()), 3), "steps q99/max", df.steps.quantile(0.99), df.steps.max(), "frames_rendered==steps+1", bool((df.frames_rendered == df.steps + 1).all()), "td_quiescent", round(float(df.td_in_quiescent_window.astype(float).mean()),3))
print("bc_dart fail episodes:", allsel["bc_dart"].query("outcome!='success'").groupby(['ss','index']).outcome.apply(list).to_dict())
# per-candidate p95 spread within seeds (tie-break margin)
for s in range(5):
    best, res = allsel[("bc_dart", s)]
    top = [r for r in res if r[1] == best[1]]
    print(f"bc_dart s{s} candidates at max success {best[1]}: {[(r[0], round(r[2],4)) for r in top]}")
# tuning trial identity, row by row, all six candidates
T = R / "bc_dart_tune/sigma_0.10/s0"
tsj = json.loads((T / "selection.json").read_text()); dsj = json.loads((R / "bc_dart/s0/selection.json").read_text())
print("\ntune selected", tsj["selected"]["candidate"]["global_step"], tsj["selected"]["p95_closing_speed_m_s"], "| bc_dart s0 selected", dsj["selected"]["candidate"]["global_step"], dsj["selected"]["p95_closing_speed_m_s"])
for c in dsj["candidates"]:
    f = c["rows_file"]; a = pd.read_csv(R / "bc_dart/s0/select" / f, dtype=str); b = pd.read_csv(T / "select" / f, dtype=str)
    diff = [col for col in a.columns if not a[col].fillna("NA").equals(b[col].fillna("NA"))]
    print(f"  {f}: rows {len(a)}/{len(b)} columns differing: {diff}; outcomes identical: {a.outcome.equals(b.outcome)}")
print("sj keys diff:", {k: (str(dsj[k])[:80], str(tsj.get(k))[:80]) for k in dsj if dsj[k] != tsj.get(k) and k not in ("candidates", "selected")})
