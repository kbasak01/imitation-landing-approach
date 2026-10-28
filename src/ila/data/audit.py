"""Demo-store audits run before a dataset is called done (plan Phase 3 item 3).

* :func:`refly_audit`: pick random stored ``(episode, step)`` pairs, re-fly each episode from
  its listed reset through the same runner path and recorder (same teacher, same T_mix
  assignment, same DART noise), and require every stored array, the sampled frames included, to be
  byte-identical to the re-flown ones. ``ila/data`` may not touch PyBullet state (P1-D2
  item 5), so a "saved state" is re-created by re-simulation, not restored; P8's physics is
  deterministic on this path (Phase 2 gate), so any difference is a store or alignment bug.
* :func:`relabel_audit`: pick random stored steps; for each episode involved, ``reset`` its
  teacher with the episode seed and call it on ``teacher_obs[0 .. t]`` **in order** (a
  stateful teacher is never queried on an isolated state), then require
  ``label[t] == teacher(teacher_obs[t])`` exactly.
* :func:`counts`: episodes and frames by sea state, heading and speed; outcomes; kept fraction
  after success filtering; bytes on disk; clear-colour statistics.

Units: actions normalised; times s model scale (lambda = 1/25).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ila import p8_adapter
from ila.data import collect
from ila.data import episodes as data_episodes
from ila.data.store import STORE_KEYS, DemoStore

__all__ = ["counts", "refly_audit", "relabel_audit", "sample_steps", "split_audit"]


def sample_steps(store: DemoStore, n: int, seed: int) -> list[tuple[str, int]]:
    """Return ``n`` uniformly random stored ``(episode, step)`` pairs (over all frames).

    Args:
        store: The store.
        n: Pairs wanted (fewer if the store holds fewer frames).
        seed: Generator seed.

    Returns:
        Pairs sorted by episode then step.
    """
    names = store.episodes()
    lengths = np.asarray([store.meta(e).steps for e in names], dtype=np.int64)
    total = int(lengths.sum())
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), total]))
    flat = np.sort(rng.choice(total, size=min(n, total), replace=False))
    edges = np.concatenate([[0], np.cumsum(lengths)])
    out: list[tuple[str, int]] = []
    for f in flat:
        e = int(np.searchsorted(edges, f, side="right") - 1)
        out.append((names[e], int(f - edges[e])))
    return out


def _collection(store: DemoStore) -> collect.Collection:
    name = str(store.file_attrs["collection"])
    return collect.load_collections()[name]


def refly_audit(path: Path, n_states: int = 20, seed: int = 20261023) -> dict[str, Any]:
    """Re-fly the episodes of ``n_states`` random stored steps and compare byte for byte.

    Args:
        path: The store.
        n_states: Random ``(episode, step)`` pairs.
        seed: Sampling seed.

    Returns:
        ``{"n_states", "n_episodes", "frames_identical", "episodes_identical",
        "mismatches": [...], "passed"}``.
    """
    with DemoStore(path) as store:
        collection = _collection(store)
        rows = data_episodes.read_split(collection.split)
        picks = sample_steps(store, n_states, seed)
        names = sorted({e for e, _ in picks})
        metas = {e: store.meta(e) for e in names}
        stored = {e: store.audit_arrays(e) for e in names}
    cfgs = p8_adapter.load_eval_configs()
    prov = collect.provenance()
    policies = collect.collection_policies(collection, rows)
    episodes = tuple(
        (metas[e].demo_index, rows[metas[e].demo_index], policies[metas[e].demo_index])
        for e in names
    )
    records, _ = collect.collect_chunk(collect.CollectTask(collection, episodes, cfgs, prov))
    fresh = {f"ep_{r.meta.demo_index:05d}": r for r in records}
    mismatches: list[str] = []
    for e in names:
        rec = fresh[e]
        if rec.meta.episode_teacher != metas[e].episode_teacher:
            mismatches.append(f"{e}: teacher {rec.meta.episode_teacher} != stored")
            continue
        if rec.meta.outcome != metas[e].outcome or rec.meta.steps != metas[e].steps:
            mismatches.append(f"{e}: outcome/steps {rec.meta.outcome}/{rec.meta.steps}")
            continue
        for k in STORE_KEYS:
            a, b = stored[e][k], rec.arrays[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                mismatches.append(f"{e}: {k} differs")
    frames_ok = sum(
        fresh[e].arrays["images"].shape[0] > t
        and stored[e]["images"][t].tobytes() == fresh[e].arrays["images"][t].tobytes()
        for e, t in picks
    )
    return {
        "n_states": len(picks),
        "n_episodes": len(names),
        "frames_identical": int(frames_ok),
        "episodes_identical": len(names) - len({m.split(":")[0] for m in mismatches}),
        "mismatches": mismatches,
        "passed": not mismatches and frames_ok == len(picks),
    }


def relabel_audit(path: Path, n_steps: int = 1000, seed: int = 20261023) -> dict[str, Any]:
    """Re-label ``n_steps`` random stored steps with the episode's teacher, shadowed in order.

    Args:
        path: The store.
        n_steps: Random steps.
        seed: Sampling seed.

    Returns:
        ``{"n_steps", "n_episodes", "identical", "max_abs_diff", "by_teacher", "passed"}``.
    """
    cfgs = p8_adapter.load_eval_configs()
    teachers: dict[str, p8_adapter.EvalPolicy] = {}
    identical = 0
    max_abs = 0.0
    by_teacher: Counter[str] = Counter()
    with DemoStore(path) as store:
        picks = sample_steps(store, n_steps, seed + 1)
        wanted: dict[str, list[int]] = {}
        for e, t in picks:
            wanted.setdefault(e, []).append(t)
        for e, steps in wanted.items():
            meta = store.meta(e)
            name = meta.episode_teacher
            if name not in teachers:
                teachers[name] = p8_adapter.teacher(name, 0, cfgs)
            teacher = teachers[name]
            obs = store.privileged_observations(e)
            labels = store.labels(e)
            teacher.reset(meta.episode_seed, None)
            fresh = np.stack(
                [
                    np.asarray(teacher.act(obs[t]), dtype=np.float64).astype(np.float32)
                    for t in range(max(steps) + 1)
                ]
            )
            for t in steps:
                same = fresh[t].tobytes() == labels[t].tobytes()
                identical += int(same)
                by_teacher[name] += 1
                max_abs = max(max_abs, float(np.max(np.abs(fresh[t] - labels[t]))))
    return {
        "n_steps": len(picks),
        "n_episodes": len(wanted),
        "identical": identical,
        "max_abs_diff": max_abs,
        "by_teacher": dict(sorted(by_teacher.items())),
        "passed": identical == len(picks),
    }


def counts(path: Path) -> dict[str, Any]:
    """Return the store's counts, outcomes, kept fraction, bytes and clear-colour stats."""
    by: dict[str, Counter[str]] = {k: Counter() for k in ("ss", "heading_deg", "speed_kn")}
    outcomes: Counter[str] = Counter()
    teachers: Counter[str] = Counter()
    frames = kept_frames = clear_frames = 0
    clear_max = 0.0
    with DemoStore(path) as store:
        attrs = dict(store.file_attrs)
        for e in store.episodes():
            m = store.meta(e)
            by["ss"][m.ss] += 1
            by["heading_deg"][f"{m.heading_deg:g}"] += 1
            by["speed_kn"][f"{m.speed_kn:g}"] += 1
            outcomes[m.outcome] += 1
            teachers[m.episode_teacher] += 1
            frames += m.steps
            kept_frames += m.steps if m.success else 0
            clear_frames += m.clear_frames
            clear_max = max(clear_max, m.clear_frac_max)
        n = len(store)
    return {
        "collection": attrs["collection"],
        "episodes": n,
        "frames": frames,
        "by_ss": dict(sorted(by["ss"].items())),
        "by_heading_deg": dict(sorted(by["heading_deg"].items())),
        "by_speed_kn": dict(sorted(by["speed_kn"].items())),
        "outcomes": dict(sorted(outcomes.items())),
        "episode_teachers": dict(sorted(teachers.items())),
        "kept_fraction_success_filter": outcomes.get("success", 0) / n if n else float("nan"),
        "kept_frames": kept_frames,
        "clear_frac_max": clear_max,
        "clear_frames": clear_frames,
        "file_bytes": path.stat().st_size,
        "file_sha256": p8_adapter.sha256_file(path),
        "file_attrs": {
            k: (v if isinstance(v, str | int | float) else json.dumps(str(v)))
            for k, v in attrs.items()
        },
    }


def split_audit(path: Path) -> dict[str, Any]:
    """Check every stored episode against its split row and the pools.

    Every episode's identity attributes must equal the committed split row at its demo
    index, and those rows must pass :func:`ila.data.episodes.check_rows` for the ``train``
    pool: train realizations only, none in ``tune`` or in any of the five frozen lists (by
    realization key), no SS6, no 90 deg, no repeated ``(key, episode seed)``.

    Returns:
        ``{"n_episodes", "identity_mismatches", "passed"}``.

    Raises:
        ValueError: From ``check_rows`` if any row leaves the train pool.
    """
    with DemoStore(path) as store:
        collection = _collection(store)
        rows = data_episodes.read_split(collection.split)
        metas = [store.meta(e) for e in store.episodes()]
    mismatches = []
    flown = []
    for m in metas:
        row = rows[m.demo_index]
        key = p8_adapter.realization_key(
            row.ss, row.heading_deg, row.speed_kn, row.vessel, row.realization_seed
        )
        ident = (m.ss, m.index, m.episode_seed, m.t0_model_s, m.realization_key, m.stream)
        expect = (
            row.ss,
            row.index,
            row.episode_seed,
            row.t0_model_s,
            json.dumps(list(key)),
            row.regime,
        )
        if ident != expect:
            mismatches.append(f"ep_{m.demo_index:05d}")
        flown.append(row)
    cfg = data_episodes.load_splits_config()
    data_episodes.check_rows(data_episodes.p8_episode_api(), flown, cfg.train_pool)
    return {"n_episodes": len(metas), "identity_mismatches": mismatches, "passed": not mismatches}
