"""Read-only provenance audit of the five bc_dagger runs (training-run-diagnostician, Gate 5).

Writes one JSON (argv[1]) with, per seed: manifest/status fields, config SHA-256 vs P3-D1,
clean tree, init_from, every checkpoint SHA-256 recomputed, rollout-student chain by SHA,
rollout stores/sidecars re-hashed, starting conditions vs ppo_2000, normaliser inheritance,
ema_last vs final EMA, timeline (training -> selection -> per-iteration select flights).
"""
import hashlib, json, sys, glob, os, time
from pathlib import Path
import numpy as np, h5py, torch

REPO = Path("/home/kbasak/projects_02/imitation-landing-approach")
RUNS = REPO / "artifacts/runs"; DAG = Path("/home/kbasak/data/ila/dagger/bc_dagger")
PPO = Path("/home/kbasak/data/ila/demos/ppo_2000.h5")
FROZEN = {"configs/train/bc_dagger.yaml": "dbc7c0d22438abe96e5e80de95f8f46d64b81216147496dc15bef55adc989da7",
          "configs/train/bc.yaml": "3d3386bb38718f5f79b9a61b62f5044052bdb603cb8b33ffd3b7664de6a01299",
          "configs/data/demos.yaml": "1fd413e2823a1cefd4b38e6ddd00c7097ee182e45f2564e0f398ce19f5f63f11"}
PPO_SHA = "35b70cbdbd6034b8128eef91a1cb9a9e61af5785641d00a3e2c13215df127cfd"
SELECT_SHA = "07ecf049cbf91e646036582b4967607e26cf68773bd486c8d7375d043f29422b"
COMMIT = "576447cb8a73c402a60588d9525cbd73274b1bc7"; P8 = "7237b7065c8af8172b3848ac6e9d3ceb2cf82075"
BETA = [0.5, 0.25, 0.125, 0.0, 0.0]
_cache = {}
def sha(p):
    p = str(p)
    if p not in _cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
        _cache[p] = h.hexdigest()
    return _cache[p]
def load(p): return json.loads(Path(p).read_text())

out = {"repo_files_sha256": {k: sha(REPO / k) for k in FROZEN}, "ppo_2000_sha256": sha(PPO),
       "select_parquet_sha256": sha(REPO / "splits/select.parquet")}
out["repo_files_match_frozen"] = all(out["repo_files_sha256"][k] == v for k, v in FROZEN.items())
ppo = h5py.File(PPO, "r")
ppo_attrs = {int(ppo[n].attrs["demo_index"]): (ppo[n].attrs["realization_key"], int(ppo[n].attrs["episode_seed"]), float(ppo[n].attrs["t0_model_s"])) for n in ppo.keys()}
ppo_frame0 = {}
for s in range(5):
    R = RUNS / "bc_dagger" / f"s{s}"; m = load(R / "manifest.json"); st = load(R / "status.json")
    r = {"problems": []}; P = r["problems"]
    def chk(cond, msg):
        if not cond: P.append(msg)
    r["state"], r["status_state"] = m["state"], st["state"]
    chk(m["state"] == "done" and st["state"] == "done", "not done")
    chk(m["purpose"] == "train" and m["run_seed"] == s and m["selectable"] is True, "purpose/seed")
    chk(m["global_step"] == 50000 and m["total_steps"] == 50000 and m["steps_timed_total"] == 50000, "steps")
    chk(m["git"]["ila_commit"] == COMMIT and m["git"]["ila_dirty"] is False and not m["git"]["ila_dirty_paths"], "git")
    chk(m["git"]["p8_commit"] == P8, "p8")
    chk(len(m["invocations"]) == 1 and m["resumes"] == [], f"invocations {len(m['invocations'])} resumes {m['resumes']}")
    chk(all(inv["git"]["ila_commit"] == COMMIT and not inv["git"]["ila_dirty"] for inv in m["invocations"]), "invocation git")
    cf = {f["path"]: f["sha256"] for f in m["config"]["files"]}
    r["config_sha256"] = cf
    chk(cf == FROZEN and m["config"]["all_frozen"], "config sha")
    chk(m["steps_override"] is None and m["checkpoint_every_override"] is None and m["first_n_override"] is None and m["max_steps"] is None, "overrides")
    chk(m["dagger_run"]["rollout_limit"] is None and m["dagger_run"]["iterations_override"] is None and m["dagger_run"]["beta"] == BETA, "dagger_run")
    chk([ (x["index"], x["steps"], x["warmup"], x["checkpoint_every"]) for x in m["stages"]] == [(i, 10000, 500, 5000) for i in range(1, 6)], "stages")
    chk([(x["stage"], x["steps"], x["first_step"]) for x in m["segments"]] == [(i, 10000, 0) for i in range(1, 6)], "segments")
    # init_from
    init = Path(m["init_from"]); exp_init = RUNS / "bc" / f"s{s}" / "checkpoints" / "s0_step030000.pt"
    bcm = load(RUNS / "bc" / f"s{s}" / "manifest.json")
    bc_rec = [c["sha256"] for c in bcm["checkpoints"] if c["file"] == "s0_step030000.pt"]
    bcmm = load(RUNS / "bc_matched" / f"s{s}" / "manifest.json")
    r["init_from"] = {"path_ok": init == exp_init, "sha256": m["init_from_sha256"], "recomputed": sha(init),
                      "bc_manifest_record": bc_rec, "bc_matched_init_from_sha256": bcmm["init_from_sha256"]}
    chk(init == exp_init and sha(init) == m["init_from_sha256"] == bc_rec[-1] == bcmm["init_from_sha256"], "init_from")
    chk(m["iteration0"]["global_step"] == 30000 and Path(m["iteration0"]["file"]) == exp_init, "iteration0")
    # checkpoints
    rec = {c["file"]: c for c in m["checkpoints"] if c["kind"] == "full"}
    el_rec = [c for c in m["checkpoints"] if c["file"] == "ema_last.pt"]
    r["ema_last_sha256"] = {"recorded": [c["sha256"] for c in el_rec], "recomputed": sha(R / "ema_last.pt")}
    chk(len(el_rec) == 1 and el_rec[0]["sha256"] == sha(R / "ema_last.pt") and el_rec[0]["kind"] == "ema_last", "ema_last sha")
    files = sorted(os.path.basename(p) for p in glob.glob(str(R / "checkpoints" / "*.pt")))
    expected = [f"s{i}_step{k:06d}.pt" for i in range(1, 6) for k in (5000, 10000)]
    chk(files == expected and sorted(rec) == expected, f"checkpoint set {files}")
    ck = {}
    for f in expected:
        h = sha(R / "checkpoints" / f); c = rec[f]
        i = int(f[1]); k = int(f[7:13])
        ok = h == c["sha256"] and c["stage"] == i and c["stage_step"] == k and c["global_step"] == 30000 * 0 + (i - 1) * 10000 + k and c["kind"] == "full" and c["bytes"] == os.path.getsize(R / "checkpoints" / f)
        ck[f] = {"sha256": h, "matches_manifest": ok, "global_step": c["global_step"]}
        chk(ok, f"ckpt {f}")
    # train_log checkpoint events
    ev = [json.loads(l) for l in (R / "train_log.jsonl").read_text().splitlines()]
    cev = {e["file"]: e["sha256"] for e in ev if e["event"] == "checkpoint" and e["kind"] == "full"}
    chk(cev == {f: ck[f]["sha256"] for f in expected}, "train_log checkpoint events")
    r["checkpoints"] = ck
    # rollout-student chain
    chain = []
    for i in range(1, 6):
        d = m["data"][str(i)]["dagger"]; it = load(R / "dagger" / f"iter{i}.json")
        exp_path = exp_init if i == 1 else R / "checkpoints" / f"s{i-1}_step010000.pt"
        exp_sha = sha(exp_path)
        rm = load(DAG / f"s{s}" / f"iter{i}.manifest.json")
        h5p = DAG / f"s{s}" / f"iter{i}.h5"; scp = DAG / f"s{s}" / f"iter{i}.sidecar.npz"
        hs, ss = sha(h5p), sha(scp)
        with h5py.File(h5p, "r") as f:
            names = sorted(f.keys())
            ep_sha = {f[n].attrs["student_ckpt_sha256"] for n in names}
            ep_beta = {float(f[n].attrs["beta"]) for n in names}; ep_it = {int(f[n].attrs["dagger_iteration"]) for n in names}
            idx = [int(f[n].attrs["demo_index"]) for n in names]
            starts_ok = all((f[n].attrs["realization_key"], int(f[n].attrs["episode_seed"]), float(f[n].attrs["t0_model_s"])) == ppo_attrs[int(f[n].attrs["demo_index"])] for n in names)
            # first frame equals the teacher demo's first frame (same starting state, before any action)
            f0 = 0
            for n in names:
                j = int(f[n].attrs["demo_index"])
                if j not in ppo_frame0:
                    g = ppo[f"ep_{j:05d}"]; ppo_frame0[j] = (g["images"][0].tobytes(), g["teacher_obs"][0].tobytes())
                f0 += (f[n]["images"][0].tobytes(), f[n]["teacher_obs"][0].tobytes()) == ppo_frame0[j]
            store_ila = {f[n].attrs["ila_commit"] for n in names}
            frames = sum(f[n]["label"].shape[0] for n in names)
            outcomes = {}
            for n in names: o = str(f[n].attrs["outcome"]); outcomes[o] = outcomes.get(o, 0) + 1
        sc_ok = sha(scp) == rm["sidecar"]["sha256"] if isinstance(rm.get("sidecar"), dict) else None
        e = {"iteration": i, "beta": d["beta"], "expected_student": str(exp_path.relative_to(RUNS)), "expected_sha256": exp_sha,
             "manifest_rollout_student_sha256": d["rollout_student"]["sha256"], "iter_json_sha256": it["rollout_student"]["sha256"],
             "store_manifest_student": rm["student"] if isinstance(rm["student"], str) else rm["student"].get("sha256"),
             "episode_attr_student_sha256": sorted(ep_sha), "store_sha256": hs, "store_sha256_recorded": d["rollout"]["store_sha256"],
             "sidecar_sha256": ss, "demo_indices": [min(idx), max(idx)], "n_episodes": len(names), "frames": frames,
             "frames_recorded": d["rollout"]["labelled_frames"], "outcomes": outcomes, "starting_conditions_match_ppo_2000": starts_ok,
             "t0_frame_and_obs_identical_to_ppo_2000": f0, "audit_clean": d["audit"]["clean"], "relabel": d["audit"]["relabel"]["identical"],
             "replay_frames_identical": d["audit"]["replay"]["frames_identical"], "store_ila_commit": sorted(store_ila),
             "timing_s": {k: v for k, v in d["timing_s"].items() if k != "note"}, "labelled_episodes_cum": d["labelled_episodes"],
             "aggregate_frames": d["aggregate_frames"], "heldout_overlap": d["heldout_overlap_episodes"],
             "heldout_shared_starting_conditions": d["heldout_shared_starting_conditions"]}
        stud = e["store_manifest_student"]
        ok = (exp_sha == d["rollout_student"]["sha256"] == it["rollout_student"]["sha256"] and ep_sha == {exp_sha}
              and (stud == exp_sha or (isinstance(stud, str) and exp_sha in json.dumps(rm["student"]))) and hs == d["rollout"]["store_sha256"]
              and d["beta"] == BETA[i-1] and ep_beta == {BETA[i-1]} and ep_it == {i} and idx == list(range(1000 + 200*(i-1), 1000 + 200*i))
              and starts_ok and f0 == 200 and frames == d["rollout"]["labelled_frames"] and d["audit"]["clean"] is True
              and store_ila == {COMMIT} and d["labelled_episodes"] == 1000 + 200*i and d["heldout_overlap_episodes"] == 0
              and d["rollout_limit"] is None and d["strict_counts"] is True)
        e["ok"] = bool(ok); chk(ok, f"chain iteration {i}")
        e["store_manifest_student_raw"] = rm["student"]
        # recorded stores in later stages re-hash the same
        for j in range(i, 6):
            for stx in m["data"][str(j)]["stores"]:
                if stx["collection"] == f"bc_dagger_s{s}_iter{i}": chk(stx["sha256"] == hs, f"stage {j} store record iter{i}")
        chain.append(e)
    r["rollout_chain"] = chain
    # demos store in every stage
    for i in range(1, 6):
        for stx in m["data"][str(i)]["stores"]:
            if stx["collection"] == "ppo_2000": chk(stx["sha256"] == PPO_SHA == sha(PPO), "ppo_2000 sha")
        ch = m["data"][str(i)]["choices"]
        chk(ch[0]["attempted"] == 1000 and ch[0]["kept"] == 999 and ch[0]["success_filter"] is True and all(c["success_filter"] is False and c["kept"] == 200 for c in ch[1:]) and len(ch) == 1 + i, f"choices stage {i}")
        chk(m["data"][str(i)]["heldout_overlap_episodes"] == 0, "heldout overlap")
    # normaliser inheritance and ema_last
    def ck_load(p): return torch.load(p, map_location="cpu", weights_only=False)
    a = ck_load(exp_init); b = ck_load(R / "checkpoints" / "s5_step010000.pt"); e_last = ck_load(R / "ema_last.pt")
    r["checkpoint_keys"] = sorted(b.keys())
    norm = {}
    for k in ("proprio_mean", "proprio_std", "normaliser", "norm"):
        if k in b: norm[k] = bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k])))
    # also inside config/meta
    for k, v in b.items():
        if isinstance(v, dict):
            for kk in v:
                if "proprio_mean" in kk or "proprio_std" in kk:
                    norm[f"{k}.{kk}"] = bool(np.array_equal(np.asarray(a[k][kk]), np.asarray(v[kk])))
    # tensor-level search in model state dict
    for kk in b["ema"]:
        if "proprio" in kk and ("mean" in kk or "std" in kk):
            norm[f"ema.{kk}"] = bool(torch.equal(a["ema"][kk], b["ema"][kk]))
            for i in range(1, 6):
                c = ck_load(R / "checkpoints" / f"s{i}_step005000.pt")
                norm[f"s{i}_5k.ema.{kk}"] = bool(torch.equal(a["ema"][kk], c["ema"][kk]))
    r["normaliser_inherited_from_bc"] = norm
    chk(norm and all(norm.values()), "normaliser")
    el = e_last["ema"] if "ema" in e_last else e_last.get("model", e_last)
    r["ema_last_equals_final_ema"] = bool(all(torch.equal(el[k], b["ema"][k]) for k in b["ema"])) if isinstance(el, dict) else None
    chk(r["ema_last_equals_final_ema"], "ema_last")
    r["ckpt_meta_s5_10k"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else type(v).__name__) for k, v in b.items() if k not in ("model", "ema", "optimizer")}
    # timeline
    sj = load(R / "selection.json")
    r["timeline_unix"] = {"train_start": m["invocations"][0]["unix"], "train_finished": m["finished_unix"],
                          "select_rows_mtime": {os.path.basename(p): os.path.getmtime(p) for p in glob.glob(str(R / "select" / "*.csv"))},
                          "selection_json_mtime": os.path.getmtime(R / "selection.json"),
                          "per_iteration_select_rows_mtime": {os.path.basename(p): os.path.getmtime(p) for p in glob.glob(str(R / "dagger/select_rows/*.csv"))}}
    tl = r["timeline_unix"]
    chk(min(tl["per_iteration_select_rows_mtime"].values()) > tl["selection_json_mtime"], "per-iteration select flights before selection")
    chk(min(tl["select_rows_mtime"].values()) > tl["train_finished"], "select before training end")
    r["wall"] = {"wall_clock_s": m["wall_clock_s"], "gpu_hours_field": m["gpu_hours"], "loop_s_total": m["loop_s_total"],
                 "data_load_s": [m["data"][str(i)]["data_load_s"] for i in range(1, 6)],
                 "timing_s_sum": sum(sum(v for k, v in m["data"][str(i)]["dagger"]["timing_s"].items() if k != "note") for i in range(1, 6)),
                 "rollouts_s": sum(m["data"][str(i)]["dagger"]["timing_s"]["rollouts"] for i in range(1, 6)),
                 "audit_s": sum(m["data"][str(i)]["dagger"]["timing_s"]["audit"] for i in range(1, 6)),
                 "aggregate_load_s": sum(m["data"][str(i)]["dagger"]["timing_s"]["aggregate"] for i in range(1, 6)),
                 "steps_per_s_loop": m["steps_per_s_loop"], "gpu_mem_max_bytes": m["gpu_mem_max_bytes"]}
    r["selection_json"] = {"list_sha256": sj["list"]["file_sha256"], "weights": sj["weights"], "inference_path": sj["inference_path"],
                           "render": sj["render"], "purpose": sj["purpose"], "candidates": [c["candidate"]["file"].split("/")[-1] for c in sj["candidates"]],
                           "wall_clock_s": sj["wall_clock_s"]}
    chk(sj["list"]["file_sha256"] == SELECT_SHA and sj["weights"] == "ema" and sj["inference_path"] == "cpu_fp32_single_sample", "selection json")
    chk(r["selection_json"]["candidates"] == ["s5_step005000.pt", "s5_step010000.pt"], "candidate set")
    out[f"s{s}"] = r
    print(f"s{s}: problems={P}", flush=True)
# bc_matched timeline/commit, for the H1 comparison
out["bc_matched"] = {f"s{s}": {k: load(RUNS / "bc_matched" / f"s{s}" / "manifest.json")[k] for k in ("init_from_sha256", "wall_clock_s", "gpu_hours", "steps_per_s_loop", "loop_s_total")} | {"ila_commit": load(RUNS / "bc_matched" / f"s{s}" / "manifest.json")["git"]["ila_commit"]} for s in range(5)}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str)); print("wrote", sys.argv[1])
